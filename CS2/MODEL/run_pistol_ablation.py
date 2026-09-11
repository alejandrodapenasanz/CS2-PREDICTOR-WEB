"""Test pistol history against the exact live architecture on a common temporal holdout.

Usage: python MODEL/run_pistol_ablation.py [--promote]. Builds the complete
chronological history first. Fits a control, pistols-only and pistol+conversion
challengers without changing production; --promote additionally requires the
existing promotion gate and deploys the EXACT evaluated artifact, not a refit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

import train
from cs2model.artifacts import ModelArtifact
from cs2model.config import load_config, set_runtime_config
from cs2model.features import _period_index, build_training_frame
from cs2model.metrics import metric_dict
from cs2model.pistols import MINIMAL_COLUMNS, PISTOL_COLUMNS, load_pistol_store, team_summary
from cs2model.pistol_opponents import (
    CHALLENGER_COLUMNS,
    PistolOpponentHistory,
    SymmetricPistolScorer,
    VERSION as OPPONENT_VERSION,
)
from cs2model.promotion import (
    assert_same_artifact_same_cohort_metrics,
    compare_candidate_to_incumbent,
    promote_candidate,
    reference_for_version,
)
from cs2model.reproducibility import set_global_determinism
from run_enhanced_info_ablation import fit_fixed_architecture

VARIANTS = {"control_refit": [], "pistols": MINIMAL_COLUMNS, "pistols_conversion": PISTOL_COLUMNS}
OPPONENT_VARIANTS = {"control_refit": [], "pistols": MINIMAL_COLUMNS, "pistols_opponent": CHALLENGER_COLUMNS}
# Already inspected before this hypothesis was specified. Never call these fresh.
OPPONENT_RESEARCH_CUTOFF = "2026-09-09"


def fresh_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude the previously inspected period from confirmatory promotion."""
    return [p for p in predictions if str(p["date"])[:10] > OPPONENT_RESEARCH_CUTOFF]


def pistol_target_report(history: PistolOpponentHistory, cutoff: str) -> dict[str, Any]:
    """Evaluate one row per pistol, never one per team or augmented orientation."""
    predictions = [p for p in history.predictions if p["date"] > cutoff]
    result: dict[str, Any] = {
        "eligible_pistols": len(history.events),
        "exclusions": dict(history.exclusions),
        "n": len(predictions),
        "matches": len({p["match_id"] for p in predictions}),
        "maps": len({p["map_id"] for p in predictions}),
        "policy": "prequential civil-day; no target map or starting side; CI clustered by day",
    }
    if predictions:
        y = np.asarray([p["actual"] for p in predictions])
        baseline = np.asarray([p["elo_only"] for p in predictions])
        dates = [p["date"] for p in predictions]
        result["date_min"], result["date_max"] = min(dates), max(dates)
        result["metrics"] = {
            k: metric_dict(y, np.asarray([p[k] for p in predictions])) for k in ("elo_only", "adjusted", "raw_rate")
        }
        result["metrics"]["coinflip"] = metric_dict(y, np.full(len(y), 0.5))
        result["ci95_adjusted_vs_elo"] = paired_intervals(
            y, baseline, np.asarray([p["adjusted"] for p in predictions]), dates
        )
        result["predictions"] = predictions
    return result


def calibration_fraction(
    dates: list[str], labels: np.ndarray, target_fraction: float, *, last_days: int | None = None
) -> tuple[float, dict[str, Any]]:
    """Choose a whole-day split and verify the shared fitter uses that exact boundary."""
    target = int(len(dates) * (1 - target_fraction))
    if last_days is not None:
        threshold = (datetime.fromisoformat(dates[-1]) - timedelta(days=last_days - 1)).date().isoformat()
        target = next((i for i, day in enumerate(dates) if day >= threshold), len(dates) - 1)
    cuts = [
        i
        for i in range(1, len(dates))
        if dates[i - 1] < dates[i] and len(np.unique(labels[:i])) == 2 and len(np.unique(labels[i:])) == 2
    ]
    if not cuts:
        raise ValueError("No whole-day calibration split with both classes")
    cut = min(cuts, key=lambda i: abs(i - target))
    fraction = 1 - cut / len(dates)
    fit, cal = train.chronological_holdout_indices(labels, fraction)
    if int(cal[0]) != cut:
        raise ValueError("Shared calibration fitter changed the whole-day split")
    return fraction, {
        "fit_n": len(fit),
        "fit_end": dates[cut - 1],
        "calibration_n": len(cal),
        "calibration_start": dates[cut],
        "calibration_end": dates[-1],
        "requested_last_days": last_days,
    }


def paired_intervals(
    y: np.ndarray, live: np.ndarray, challenger: np.ndarray, dates: list[str]
) -> dict[str, list[float]]:
    """Paired day-cluster bootstrap CI95; resampling is inference, NEVER a training split."""
    unique = sorted(set(dates))
    date_array = np.asarray(dates)
    rng = np.random.default_rng(42)
    clipped_live, clipped_challenger = (np.clip(p, 1e-9, 1 - 1e-9) for p in (live, challenger))
    differences = {
        "accuracy_delta": ((challenger >= 0.5) == y).astype(float) - ((live >= 0.5) == y).astype(float),
        "log_loss_delta": -(y * np.log(clipped_challenger) + (1 - y) * np.log(1 - clipped_challenger))
        + y * np.log(clipped_live)
        + (1 - y) * np.log(1 - clipped_live),
        "brier_delta": (challenger - y) ** 2 - (live - y) ** 2,
    }
    ns = np.asarray([np.sum(date_array == d) for d in unique])
    selections = rng.integers(0, len(unique), size=(2000, len(unique)))
    result = {}
    for name, values in differences.items():
        sums = np.asarray([np.sum(values[date_array == d]) for d in unique])
        boot = np.sum(sums[selections], axis=1) / np.sum(ns[selections], axis=1)
        result[name] = np.quantile(boot, [0.025, 0.975]).tolist()
    return result


def main() -> int:
    """Evaluate fixed, preregistered families; never promote an unscored refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "BBDD/cs2.db")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--opponent-adjusted",
        action="store_true",
        help="Test strength beyond opponent Elo, with a fresh-only promotion gate",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Build causal features and pistol-target metrics without fitting match challengers",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or ROOT / "MODEL/results" / (
        "pistol_opponents" if args.opponent_adjusted else "pistol_rounds"
    )
    variants = OPPONENT_VARIANTS if args.opponent_adjusted else VARIANTS
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_config()
    set_runtime_config(runtime)
    set_global_determinism(42)
    live, reference = train._load_validated_incumbent()
    if live is None or reference is None:
        raise ValueError("Pistol ablation requires a validated live model")
    if live.prediction_architecture != "model_a_no_odds":
        raise ValueError("Verify regime-specific whole-day calibration before testing a router champion")
    cutoff = str(live.metadata["date_max"])[:10]
    print(f"[pistols] live={reference.version}; cutoff={cutoff}; loading complete history", flush=True)
    rows = train.dataio.load_training_rows(None, ROOT / "PIPELINE/master/matches.json", cs2_only=True, db_path=args.db)
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        store = load_pistol_store(conn)
        names = {str(i): str(n) for i, n in conn.execute("SELECT hltv_id,name FROM teams WHERE hltv_id IS NOT NULL")}
    opponent_history = PistolOpponentHistory(rows, store) if args.opponent_adjusted else None
    features, labels_list, meta, _state = build_training_frame(
        rows, form_half_life=runtime.training.form_half_life_days, freeze_civil_day=args.opponent_adjusted
    )
    if opponent_history is not None:
        for row, feature in zip(rows, features, strict=True):
            feature.update(opponent_history.match_features(row))
    features = train.add_odds_features(features, rows)
    labels = np.asarray(labels_list, dtype=int)
    dates = [str(m["date"])[:10] for m in meta]
    mask = np.asarray([d <= cutoff for d in dates])
    coverage_column = "pistol_opponent_available" if args.opponent_adjusted else "pistol_available"
    available = np.asarray([f[coverage_column] >= 0.5 for f in features])
    holdout = np.flatnonzero(~mask)
    if len(holdout) < runtime.promotion.min_common_holdout_rows:
        raise ValueError("Insufficient common post-production-cutoff holdout")
    calibration_attempts = []
    # Coverage-only choice, before fitting any challenger or seeing its holdout
    # score. Never shrink below seven days or below 100 calibration rows.
    for last_days in (None, 14, 7):
        fraction, split = calibration_fraction(
            [d for d, fit in zip(dates, mask, strict=True) if fit],
            labels[mask],
            runtime.training.final_calibration_fraction,
            last_days=last_days,
        )
        base_available = int(available[mask][: int(split["fit_n"])].sum())
        calibration_attempts.append({**split, "base_fit_available": base_available})
        if base_available >= 100 and split["calibration_n"] >= 100:
            break
    periods = np.asarray([_period_index(r["date_obj"]) for r in rows])
    selected = [features[i] for i in holdout]
    test_y = labels[holdout]
    test_dates = [dates[i] for i in holdout]
    scored_live = SymmetricPistolScorer(live) if args.opponent_adjusted else live
    live_p = scored_live.predict_proba_team1(selected)
    cohort = [
        {"match_id": meta[i]["id"], "date": dates[i], "actual": int(labels[i]), "features": features[i]}
        for i in holdout
    ]
    split_at = int(split["fit_n"])
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "live_version": reference.version,
        "live_sha256": reference.sha256,
        "architecture": getattr(live, "prediction_architecture", "model_a_no_odds"),
        "causality": "played_day < D AND original_capture_day < D; complete warmup before masks",
        "fixed_design": {
            "window_days": 90,
            "prior_n": 20,
            "prior_mean": 0.5,
            "minimum_pistols_per_team": 10,
            "variants": variants,
            "freeze_civil_day": args.opponent_adjusted,
            "scoring": "production_symmetric_AB_BA" if args.opponent_adjusted else "artifact_directional",
            "seed": 42,
            "row_policy": "all",
            "calibration_method": str(live.metadata.get("production_calibration") or "sigmoid"),
            "components": [{"name": c.name, "weight": c.weight} for c in live.components],
        },
        "coverage": {
            "total_rows": len(rows),
            "available_rows": int(available.sum()),
            "prefix_rows": int(mask.sum()),
            "prefix_available": int(available[mask].sum()),
            "base_fit_available": int(available[mask][:split_at].sum()),
            "calibration_available": int(available[mask][split_at:].sum()),
            "holdout_available": int(available[~mask].sum()),
        },
        "split": split,
        "calibration_coverage_attempts": calibration_attempts,
        "holdout": {"n": len(holdout), "date_min": test_dates[0], "date_max": test_dates[-1]},
        "live_metrics": metric_dict(test_y, live_p),
        "comparison": {},
        "promotion": {"performed": False},
        "selection_warning": "Two preregistered families tested on one holdout: exploratory evidence; CI95 unadjusted for multiplicity.",
    }
    today = datetime.now(UTC).date()
    report["source_fingerprint"] = store["fingerprint"]
    report["history_maps"] = len(store["maps"])
    report["teams_today"] = [
        {"team_id": t, "team": names.get(t, t), **team_summary(store, t, today)} for t in store["teams"]
    ]
    report["teams_today"].sort(key=lambda r: r["n"], reverse=True)
    if opponent_history is not None:
        snapshot = opponent_history.snapshot(today)
        report["pistol_target"] = pistol_target_report(opponent_history, cutoff)
        report["opponent_history_fingerprint"] = opponent_history.fingerprint
        report["fresh_holdout_required_after"] = OPPONENT_RESEARCH_CUTOFF
        report["fresh_holdout_n"] = sum(day > OPPONENT_RESEARCH_CUTOFF for day in test_dates)
        report["opponent_teams_today"] = [
            {"team_id": t, "team": names.get(t, t), **v} for t, v in snapshot.teams.items()
        ]
        report["pistol_baseline_today"] = {
            "day": today.isoformat(),
            "fit_n": snapshot.fit_n,
            "fit_matches": snapshot.fit_matches,
            "beta_per_400_elo": snapshot.beta,
            "probability_by_elo_gap": {str(g): snapshot.neutral(g) for g in (-400, -200, 0, 200, 400)},
        }
        report["selection_warning"] = (
            "Previously inspected match holdout is exploratory only; promotion additionally requires a fresh post-2026-09-09 cohort."
        )
    if args.coverage_only:
        report["status"] = "coverage_only"
        (output / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("coverage", "split", "holdout")}, indent=2), flush=True)
        return 0
    if report["coverage"]["base_fit_available"] < 100 or split["calibration_n"] < 100:
        report["status"] = "insufficient_causal_training_coverage"
        (output / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "teams_today"}, indent=2), flush=True)
        return 0
    artifacts: dict[str, ModelArtifact] = {}
    control_probabilities: np.ndarray | None = None
    for variant, columns in variants.items():
        print(
            f"[pistols] fitting {variant}; train/calibration rows={int(mask.sum())}; holdout={len(holdout)}", flush=True
        )
        metadata = deepcopy(live.metadata)
        metadata.update(
            date_max=cutoff,
            date_min=dates[0],
            n_train_rows=int(mask.sum()),
            trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            pistol_variant=variant,
            pistol_source_fingerprint=store["fingerprint"],
            pistol_calibration_split=split,
            random_seed=42,
            production_model=f"pistol_ablation_{variant}",
            evaluation_policy=f"pistol_ablation_{variant}",
            generic_learner_feature_columns=list(
                dict.fromkeys(
                    [
                        *list(live.metadata.get("generic_learner_feature_columns") or live.feature_columns),
                        *columns,
                    ]
                )
            ),
        )
        metadata.setdefault("feature_policies", {})["pistol_history"] = {
            "enabled": bool(columns),
            "columns": list(columns),
            "activation": "champion_challenger_common_temporal_holdout",
            "availability": "played_day < D AND original_capture_day < D",
        }
        if args.opponent_adjusted:
            metadata["pistol_opponent_version"] = OPPONENT_VERSION
            metadata["freeze_civil_day"] = True
            metadata["opponent_history_fingerprint"] = opponent_history.fingerprint if opponent_history else None
        artifact = fit_fixed_architecture(
            live,
            features,
            labels,
            periods,
            mask,
            policy="all",
            half_life_days=runtime.training.recency_half_life_days,
            calibration_fraction=fraction,
            seed=42,
            metadata=metadata,
            additional_learner_columns=list(columns),
        )
        scored_artifact = SymmetricPistolScorer(artifact) if args.opponent_adjusted else artifact
        probabilities = scored_artifact.predict_proba_team1(selected)
        if variant == "control_refit":
            control_probabilities = probabilities
        assert control_probabilities is not None
        consistency = assert_same_artifact_same_cohort_metrics(
            scored_artifact, selected, test_y, probabilities.tolist()
        )
        prediction_rows = [
            {"match_id": meta[i]["id"], "date": dates[i], "actual": int(labels[i]), "prob_team1": float(p)}
            for i, p in zip(holdout, probabilities, strict=True)
        ]
        decision = compare_candidate_to_incumbent(
            scored_live,
            prediction_rows,
            cohort,
            min_samples=runtime.promotion.min_common_holdout_rows,
            log_loss_epsilon=runtime.promotion.log_loss_epsilon,
            brier_epsilon=runtime.promotion.brier_epsilon,
        )
        artifact.metadata["promotion_decision"] = decision.to_dict()
        report["comparison"][variant] = {
            "metrics": metric_dict(test_y, probabilities),
            "gate": decision.to_dict(),
            "consistency": consistency,
            "ci95_vs_live": paired_intervals(test_y, live_p, probabilities, test_dates),
            "ci95_vs_control_refit": paired_intervals(test_y, control_probabilities, probabilities, test_dates),
            "by_coverage": {},
        }
        if args.opponent_adjusted:
            report["comparison"][variant]["fresh_gate"] = compare_candidate_to_incumbent(
                scored_live,
                fresh_predictions(prediction_rows),
                cohort,
                min_samples=runtime.promotion.min_common_holdout_rows,
                log_loss_epsilon=runtime.promotion.log_loss_epsilon,
                brier_epsilon=runtime.promotion.brier_epsilon,
            ).to_dict()
        for label, subset in (("pistol_history", available[~mask]), ("no_pistol_history", ~available[~mask])):
            if np.any(subset):
                report["comparison"][variant]["by_coverage"][label] = {
                    "live": metric_dict(test_y[subset], live_p[subset]),
                    "challenger": metric_dict(test_y[subset], probabilities[subset]),
                    "ci95": paired_intervals(
                        test_y[subset],
                        live_p[subset],
                        probabilities[subset],
                        [d for d, yes in zip(test_dates, subset, strict=True) if yes],
                    ),
                }
        report["comparison"][variant]["changed_picks"] = [
            {
                "match_id": meta[i]["id"],
                "date": dates[i],
                "team1": meta[i]["team1"],
                "team2": meta[i]["team2"],
                "live_p": float(lp),
                "challenger_p": float(cp),
                "actual": int(labels[i]),
                "pistol_sample_min": features[i]["pistol_sample_min"],
            }
            for i, lp, cp in zip(holdout, live_p, probabilities, strict=True)
            if (lp >= 0.5) != (cp >= 0.5)
        ]
        report["comparison"][variant]["predictions"] = prediction_rows
        artifacts[variant] = artifact
        print(
            json.dumps(
                {"variant": variant, "metrics": report["comparison"][variant]["metrics"], "gate": decision.message}
            ),
            flush=True,
        )
        (output / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    winner = min(
        tuple(name for name in variants if name != "control_refit"),
        key=lambda name: (
            report["comparison"][name]["metrics"]["log_loss"],
            report["comparison"][name]["metrics"]["brier"],
        ),
    )
    report["winner"] = winner
    payload = report["comparison"][winner]
    report["winner_beats_live"] = bool(payload["gate"]["promote"])
    control = report["comparison"]["control_refit"]["metrics"]
    report["winner_vs_control"] = {m: payload["metrics"][m] - control[m] for m in ("accuracy", "log_loss", "brier")}
    hashes = {r["gate"]["holdout_sha256"] for r in report["comparison"].values()}
    if len(hashes) != 1:
        raise ValueError("Asymmetric cohorts across pistol experiments")
    report["holdout"]["sha256"] = next(iter(hashes))
    authorized = bool(payload["gate"]["promote"] and (not args.opponent_adjusted or payload["fresh_gate"]["promote"]))
    report["promotion"]["eligible"] = authorized
    if args.promote and authorized:
        artifact = artifacts[winner]
        candidate = output / "approved_candidate.pkl"
        artifact.save(candidate)
        reloaded = train.load_artifact(candidate)
        if reloaded is None:
            raise ValueError("Saved challenger cannot boot")
        assert_same_artifact_same_cohort_metrics(
            SymmetricPistolScorer(reloaded) if args.opponent_adjusted else reloaded,
            selected,
            test_y,
            [r["prob_team1"] for r in payload["predictions"]],
        )
        production_name = artifact.metadata["production_model"]
        glicko_p = np.asarray([f["glicko_prob_centered"] + 0.5 for f in selected])
        health_metrics = {production_name: payload["metrics"], "glicko": metric_dict(test_y, glicko_p)}
        (output / "metrics.json").write_text(json.dumps(health_metrics, indent=2), encoding="utf-8")
        with (output / "predictions_walkforward.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "match_id", "date", "actual", "prob_team1"])
            writer.writeheader()
            writer.writerows({"model": production_name, **row} for row in payload["predictions"])
        from health_gates import validate_candidate

        health = validate_candidate(args.db, candidate, output)
        (output / "candidate_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
        if health["status"] != "pass":
            raise ValueError("Pistol challenger failed health gates; production unchanged")
        registry = train.save_model_registry(artifact, health_metrics, [], [], {}, {}, {}, artifact_source=candidate)
        registered = reference_for_version(train.MODEL_REGISTRY_DIR, registry["version"])
        final_health = validate_candidate(args.db, Path(registry["registered_model"]), output)
        (output / "registered_health.json").write_text(json.dumps(final_health, indent=2), encoding="utf-8")
        if final_health["status"] != "pass":
            raise ValueError("Registered pistol challenger failed health gates; production unchanged")
        decision = compare_candidate_to_incumbent(
            scored_live,
            fresh_predictions(payload["predictions"]) if args.opponent_adjusted else payload["predictions"],
            cohort,
            min_samples=runtime.promotion.min_common_holdout_rows,
            log_loss_epsilon=runtime.promotion.log_loss_epsilon,
            brier_epsilon=runtime.promotion.brier_epsilon,
        )
        result = promote_candidate(
            train.MODEL_REGISTRY_DIR,
            registered.version,
            decision,
            production_path=train.ARTIFACT_PATH,
            expected_incumbent=reference,
            expected_candidate_sha256=registered.sha256,
        )
        report["promotion"] = {
            "performed": result.changed,
            "message": result.message,
            "evaluated_artifact_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
    report["status"] = "evaluated"
    (output / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("teams_today", "comparison", "opponent_teams_today", "pistol_target")
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate full and swiss-only segment interactions through the CS2 gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import train
from cs2model.config import load_config, set_runtime_config
from cs2model.features import (
    SEGMENT_INTERACTION_FEATURE_COLUMNS,
    _period_index,
    build_training_frame,
)
from cs2model.metrics import metric_dict
from cs2model.promotion import (
    METRIC_CONSISTENCY_ATOL,
    compare_candidate_to_incumbent,
    promote_candidate,
    reference_for_version,
)
from cs2model.reproducibility import set_global_determinism
from cs2model.segment_calibration import segment_calibration
from measure_segment_calibration import measure
from run_enhanced_info_ablation import fit_fixed_architecture

MINIMAL_SWISS_COLUMNS = ("elo_diff_x_stage_swiss",)
VARIANT_COLUMNS = {
    "control_refit": (),
    "full_interactions": tuple(SEGMENT_INTERACTION_FEATURE_COLUMNS),
    "minimal_swiss": MINIMAL_SWISS_COLUMNS,
}
PROMOTION_VARIANTS = ("full_interactions", "minimal_swiss")


def _rows_for_artifact(
    artifact: Any,
    feature_rows: list[dict[str, float]],
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, Any]]:
    probabilities = artifact.predict_proba_team1([feature_rows[index] for index in indices])
    return [
        {
            "match_id": metadata[index]["id"],
            "date": metadata[index]["date"],
            "actual": int(labels[index]),
            "prob_team1": float(probability),
            "prediction_regime": artifact.prediction_regime(feature_rows[index]),
            "stage": metadata[index].get("stage"),
            "environment": metadata[index].get("environment"),
            "event_tier": metadata[index].get("event_tier"),
            "elo_diff": metadata[index].get("elo_diff"),
            "format": metadata[index].get("format"),
        }
        for index, probability in zip(indices, probabilities, strict=True)
    ]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return metric_dict(
        np.asarray([row["actual"] for row in rows], dtype=int),
        np.asarray([row["prob_team1"] for row in rows], dtype=float),
    )


def _swiss(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return segment_calibration(rows, "stage", min_n=100).get(
        "swiss",
        {"n": 0, "status": "absent", "note": "no swiss rows in cohort"},
    )


def _evaluated_live_rows(db_path: Path) -> int:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return int(
            connection.execute("SELECT COUNT(*) FROM prediction_ledger WHERE ledger_status='evaluated'").fetchone()[0]
        )


def _write_predictions(path: Path, model: str, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "match_id",
        "date",
        "actual",
        "prob_team1",
        "stage",
        "environment",
        "event_tier",
        "elo_diff",
        "format",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": model, **{field: row.get(field) for field in fields[1:]}})


def _hypotheses(base_report: dict[str, Any]) -> dict[str, Any]:
    calibration = base_report["segment_calibration"]
    return {
        "elo_200_300": calibration["by_elo_gap"].get("200-300"),
        "stage_group": calibration["by_stage"].get("group"),
        "lan": calibration["by_environment"].get("lan"),
        "elo_300_plus": calibration["by_elo_gap"].get("300+"),
        "interpretation": (
            "Elo 200-300, group and LAN remain hypotheses when their mean-gap "
            "CI95 includes zero. Elo 300+ remains uninterpretable for small N."
        ),
    }


def validate_common_gate(
    variant_payloads: dict[str, Any],
    incumbent_metrics: dict[str, float],
) -> str:
    """Reject cohort or metric asymmetry before choosing a challenger."""

    variants = ("control_refit", *PROMOTION_VARIANTS)
    holdout_hashes = {str(variant_payloads[variant]["gate"]["holdout_sha256"]) for variant in variants}
    if len(holdout_hashes) != 1:
        raise RuntimeError("segment challengers were not evaluated on one cohort")
    for variant in variants:
        payload = variant_payloads[variant]
        gate = payload["gate"]
        for metric in ("log_loss", "brier"):
            if not math.isclose(
                float(gate["challenger"][metric]),
                float(payload["metrics"][metric]),
                rel_tol=0.0,
                abs_tol=METRIC_CONSISTENCY_ATOL,
            ):
                raise RuntimeError(f"{variant}: {metric} differs inside and outside the gate")
            if not math.isclose(
                float(gate["incumbent"][metric]),
                float(incumbent_metrics[metric]),
                rel_tol=0.0,
                abs_tol=METRIC_CONSISTENCY_ATOL,
            ):
                raise RuntimeError(f"{variant}: incumbent {metric} differs across comparisons")
    return next(iter(holdout_hashes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "BBDD" / "cs2.db"))
    parser.add_argument("--config", default=str(ROOT / "MODEL" / "config.yaml"))
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "MODEL" / "results" / "predictions_walkforward.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "MODEL" / "results" / "segment_interaction_ablation"),
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    runtime = load_config(args.config)
    set_runtime_config(runtime)
    set_global_determinism(runtime.random_seed)
    db_path = Path(args.db)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    incumbent, incumbent_reference = train._load_validated_incumbent()
    if incumbent is None or incumbent_reference is None:
        raise SystemExit("Segment-interaction ablation requires a validated live model")
    cutoff = str(incumbent.metadata.get("date_max") or "")[:10]
    if not cutoff:
        raise SystemExit("Live model has no date_max cutoff")

    base_report = measure(
        Path(args.predictions),
        db_path,
        Path(args.config),
        "nested_model_policy",
    )
    current_swiss = base_report["segment_calibration"]["by_stage"].get("swiss", {})

    rows = train.dataio.load_training_rows(
        None,
        ROOT / "PIPELINE" / "master" / "matches.json",
        cs2_only=True,
        db_path=db_path,
    )
    if not rows:
        raise SystemExit("No CS2 training rows")
    feature_rows, labels_list, metadata, _state = build_training_frame(
        rows, form_half_life=runtime.training.form_half_life_days
    )
    feature_rows = train.add_odds_features(feature_rows, rows)
    labels = np.asarray(labels_list, dtype=int)
    periods = np.asarray([_period_index(row.get("date_obj")) for row in rows])
    prefix_mask = np.asarray(
        [str(item.get("date") or "")[:10] <= cutoff for item in metadata],
        dtype=bool,
    )
    holdout_indices = [index for index, item in enumerate(metadata) if str(item.get("date") or "")[:10] > cutoff]
    if len(holdout_indices) < runtime.promotion.min_common_holdout_rows:
        raise SystemExit(
            f"Only {len(holdout_indices)} post-cutoff rows; gate requires {runtime.promotion.min_common_holdout_rows}"
        )
    incumbent_feature_rows = [
        {
            "match_id": item["id"],
            "date": item["date"],
            "actual": int(label),
            "features": features,
        }
        for features, label, item in zip(feature_rows, labels, metadata, strict=True)
    ]
    incumbent_predictions = _rows_for_artifact(incumbent, feature_rows, labels, metadata, holdout_indices)

    variant_payloads: dict[str, Any] = {}
    for variant, columns in VARIANT_COLUMNS.items():
        shadow = fit_fixed_architecture(
            incumbent,
            feature_rows,
            labels,
            periods,
            prefix_mask,
            policy="recency_weighted",
            half_life_days=runtime.training.recency_half_life_days,
            calibration_fraction=runtime.training.final_calibration_fraction,
            seed=runtime.random_seed,
            metadata={
                "date_max": cutoff,
                "segment_interaction_variant": variant,
            },
            additional_learner_columns=list(columns),
        )
        predictions = _rows_for_artifact(shadow, feature_rows, labels, metadata, holdout_indices)
        decision = compare_candidate_to_incumbent(
            incumbent,
            predictions,
            incumbent_feature_rows,
            min_samples=runtime.promotion.min_common_holdout_rows,
            log_loss_epsilon=runtime.promotion.log_loss_epsilon,
            brier_epsilon=runtime.promotion.brier_epsilon,
        )
        variant_payloads[variant] = {
            "columns": list(columns),
            "metrics": _metrics(predictions),
            "swiss": _swiss(predictions),
            "gate": decision.to_dict(),
            "predictions": predictions,
        }

    incumbent_metrics = _metrics(incumbent_predictions)
    incumbent_swiss = _swiss(incumbent_predictions)
    common_holdout_sha256 = validate_common_gate(variant_payloads, incumbent_metrics)

    winner = min(
        PROMOTION_VARIANTS,
        key=lambda variant: (
            variant_payloads[variant]["metrics"]["log_loss"],
            variant_payloads[variant]["metrics"]["brier"],
        ),
    )
    winner_gate = variant_payloads[winner]["gate"]
    evaluated_live = _evaluated_live_rows(db_path)
    live_threshold = runtime.segment_calibration.min_live_total_rows_for_decisions
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "live": {
            "version": incumbent_reference.version,
            "date_max": cutoff,
            "architecture": str(getattr(incumbent, "prediction_architecture", "model_a_no_odds") or "model_a_no_odds"),
        },
        "historical_signal_before_today_start": {
            "oos_n": 7914,
            "swiss_n": 188,
            "calibration_gap": 0.0708,
            "note": "user-provided prior cut; superseded by current measurement",
        },
        "current_long_oos": {
            "n": base_report["n"],
            "date_min": base_report["date_min"],
            "date_max": base_report["date_max"],
            "swiss": current_swiss,
        },
        "common_gate_holdout": {
            "n": len(holdout_indices),
            "date_min": metadata[holdout_indices[0]]["date"],
            "date_max": metadata[holdout_indices[-1]]["date"],
            "sha256": common_holdout_sha256,
        },
        "comparison": {
            "live": {"metrics": incumbent_metrics, "swiss": incumbent_swiss},
            **{
                variant: {key: value for key, value in variant_payloads[variant].items() if key != "predictions"}
                for variant in VARIANT_COLUMNS
            },
        },
        "winner": winner,
        "winner_beats_live_gate": bool(winner_gate["promote"]),
        "hypotheses": _hypotheses(base_report),
        "live_segment_policy": {
            "evaluated_rows": evaluated_live,
            "decision_threshold": live_threshold,
            "enabled_for_segment_decisions": evaluated_live >= live_threshold,
            "note": "live segment evidence cannot decide before 1,000 evaluated rows",
        },
        "promotion": {
            "requested": bool(args.promote),
            "performed": False,
            "message": (
                "winner rejected; segment interactions remain off"
                if not winner_gate["promote"]
                else "winner approved; pass --promote to publish"
            ),
        },
    }
    report_path = output / "segment_interaction_ablation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.promote or not winner_gate["promote"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    winning_columns = list(VARIANT_COLUMNS[winner])
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    production_name = f"segment_{winner}"
    artifact_metadata = deepcopy(incumbent.metadata)
    feature_policies = deepcopy(artifact_metadata.get("feature_policies") or {})
    feature_policies["segment_interactions"] = {
        "enabled": True,
        "columns": winning_columns,
        "activation": "champion_challenger_common_temporal_holdout",
        "variant": winner,
    }
    artifact_metadata.update(
        {
            "trained_at": trained_at,
            "date_min": rows[0]["date"],
            "date_max": rows[-1]["date"],
            "n_train_rows": len(rows),
            "production_model": production_name,
            "evaluation_policy": production_name,
            "segment_interaction_variant": winner,
            "segment_interaction_columns": winning_columns,
            "feature_policies": feature_policies,
            "generic_learner_feature_columns": list(
                dict.fromkeys(
                    [
                        *list(incumbent.metadata.get("generic_learner_feature_columns") or incumbent.feature_columns),
                        *winning_columns,
                    ]
                )
            ),
            "promotion_decision": winner_gate,
            "random_seed": runtime.random_seed,
        }
    )
    final_artifact = fit_fixed_architecture(
        incumbent,
        feature_rows,
        labels,
        periods,
        np.ones(len(rows), dtype=bool),
        policy="recency_weighted",
        half_life_days=runtime.training.recency_half_life_days,
        calibration_fraction=runtime.training.final_calibration_fraction,
        seed=runtime.random_seed,
        metadata=artifact_metadata,
        additional_learner_columns=winning_columns,
    )
    candidate_path = output / "candidate_model.pkl"
    final_artifact.save(candidate_path)
    winning_predictions = variant_payloads[winner]["predictions"]
    glicko_probabilities = np.asarray(
        [float(feature_rows[index].get("glicko_prob_centered", 0.0)) + 0.5 for index in holdout_indices]
    )
    metrics = {
        production_name: variant_payloads[winner]["metrics"],
        "glicko": metric_dict(labels[holdout_indices], glicko_probabilities),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(
        output / "predictions_walkforward.csv",
        production_name,
        winning_predictions,
    )

    from health_gates import store_report, validate_candidate

    health = validate_candidate(db_path, candidate_path, output)
    store_report(db_path, health)
    if health["status"] != "pass":
        raise SystemExit("Segment-interaction winner failed candidate health gates")
    registry = train.save_model_registry(
        final_artifact,
        metrics,
        [],
        [],
        {},
        {},
        {},
        artifact_source=candidate_path,
    )
    candidate_version = str(registry["version"])
    candidate_reference = reference_for_version(train.MODEL_REGISTRY_DIR, candidate_version)
    registered_model = Path(registry["registered_model"])
    final_health = validate_candidate(db_path, registered_model, output)
    store_report(db_path, final_health)
    if final_health["status"] != "pass":
        raise SystemExit("Registered segment-interaction winner failed final health gates")
    decision = compare_candidate_to_incumbent(
        incumbent,
        winning_predictions,
        incumbent_feature_rows,
        min_samples=runtime.promotion.min_common_holdout_rows,
        log_loss_epsilon=runtime.promotion.log_loss_epsilon,
        brier_epsilon=runtime.promotion.brier_epsilon,
    )
    deployment = promote_candidate(
        train.MODEL_REGISTRY_DIR,
        candidate_version,
        decision,
        production_path=train.ARTIFACT_PATH,
        expected_incumbent=incumbent_reference,
        expected_candidate_sha256=candidate_reference.sha256,
    )
    report["promotion"] = {
        "requested": True,
        "performed": bool(deployment.changed),
        "message": deployment.message,
        "latest": deployment.latest.version if deployment.latest else None,
        "last_good": deployment.last_good.version if deployment.last_good else None,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare enhanced-info training-row policies through the production gate.

The current live model supplies the frozen architecture and component weights.
All three challengers are fitted through the live model's ``date_max`` and are
then scored on the exact same later cohort.  Features are built once from the
complete chronological history before any training-row mask is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from cs2model.artifacts import ColumnSubsetEstimator, Component, ModelArtifact
from cs2model.config import load_config, set_runtime_config
from cs2model.enhanced_info import (
    ENHANCED_AVAILABLE_COLUMN,
    annotate_enhanced_info,
    enhanced_bias_report,
    recency_decay_weights,
    training_indices,
)
from cs2model.features import _period_index, build_training_frame
from cs2model.metrics import metric_dict
from cs2model.promotion import (
    compare_candidate_to_incumbent,
    promote_candidate,
    reference_for_version,
)
from cs2model.reproducibility import set_global_determinism

POLICIES = ("all", "only_enhanced", "recency_weighted")
KIND_BY_COMPONENT = {
    "logistic": "logistic",
    "lightgbm": "gbm",
    "gbm": "gbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "random_forest": "random_forest",
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _policy_weights(
    periods: np.ndarray,
    mask: np.ndarray,
    policy: str,
    half_life_days: float,
) -> np.ndarray | None:
    return recency_decay_weights(periods[mask], half_life_days) if policy == "recency_weighted" else None


def fit_fixed_architecture(
    incumbent: ModelArtifact,
    feature_rows: list[dict[str, float]],
    labels: np.ndarray,
    periods: np.ndarray,
    fit_mask: np.ndarray,
    *,
    policy: str,
    half_life_days: float,
    calibration_fraction: float,
    seed: int,
    metadata: dict[str, Any],
    additional_learner_columns: list[str] | None = None,
) -> ModelArtifact:
    """Refit the live architecture with a row policy and optional extra features."""

    if int(fit_mask.sum()) < 100:
        raise ValueError(f"{policy}: only {int(fit_mask.sum())} fit rows")
    extra_columns = list(
        additional_learner_columns if additional_learner_columns is not None else [ENHANCED_AVAILABLE_COLUMN]
    )
    feature_columns = _unique([*list(incumbent.feature_columns), *extra_columns])
    learner_columns = _unique(
        [
            *list(incumbent.metadata.get("generic_learner_feature_columns") or incumbent.feature_columns),
            *extra_columns,
        ]
    )
    learner_indices = [feature_columns.index(column) for column in learner_columns]
    matrix = train._matrix(feature_rows, feature_columns)
    learner_matrix = matrix[:, learner_indices]
    sample_weight = _policy_weights(periods, fit_mask, policy, half_life_days)
    calibration = str(incumbent.metadata.get("production_calibration") or "sigmoid")
    if calibration not in {"sigmoid", "isotonic", "beta"}:
        calibration = "sigmoid"

    fitted_kinds: dict[str, Any] = {}
    for live_component in incumbent.components:
        kind = KIND_BY_COMPONENT.get(str(live_component.name))
        if kind is None or kind in fitted_kinds:
            continue
        _base, calibrators = train.fit_calibrated_multi(
            kind,
            learner_matrix[fit_mask],
            labels[fit_mask],
            learner_columns,
            methods=(calibration,),
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=sample_weight,
        )
        fitted = calibrators.get(calibration) or calibrators.get("sigmoid")
        if fitted is None:
            raise ValueError(f"{policy}: failed to calibrate {kind}")
        fitted_kinds[kind] = fitted

    components: list[Component] = []
    for live_component in incumbent.components:
        name = str(live_component.name)
        rating_column = train.RATING_CANDIDATE_COLUMNS.get(name)
        if rating_column is not None:
            if rating_column not in feature_columns:
                raise ValueError(f"{policy}: missing rating column {rating_column}")
            estimator = train.fit_probability_column_estimator(
                matrix[fit_mask],
                labels[fit_mask],
                feature_columns.index(rating_column),
                sample_weight=sample_weight,
            )
            components.append(Component(name, estimator, None, float(live_component.weight)))
            continue
        kind = KIND_BY_COMPONENT.get(name)
        if kind is None:
            raise ValueError(f"unsupported live component for ablation: {name}")
        components.append(
            Component(
                name,
                ColumnSubsetEstimator(fitted_kinds[kind], learner_indices),
                None,
                float(live_component.weight),
            )
        )

    architecture = str(getattr(incumbent, "prediction_architecture", "model_a_no_odds") or "model_a_no_odds")
    live_odds_columns = list(getattr(incumbent, "odds_feature_columns", []) or [])
    if not live_odds_columns:
        live_odds_columns = [*feature_columns, *train.ODDS_FEATURE_COLUMNS]
    odds_columns = _unique([*live_odds_columns, *extra_columns])
    odds_components: list[Component] = []
    mixed_components: list[Component] = []
    odds_matrix = train._matrix(feature_rows, odds_columns)
    if "odds_available" in odds_columns:
        odds_mask = np.asarray(
            odds_matrix[:, odds_columns.index("odds_available")] >= 0.5,
            dtype=bool,
        )
    else:
        odds_mask = np.zeros(len(feature_rows), dtype=bool)

    if architecture == "router_two_models":
        primary_mask = fit_mask & odds_mask
        if int(primary_mask.sum()) < 120:
            raise ValueError(f"{policy}: router primary has only {int(primary_mask.sum())} causal odds rows")
        _base, calibrators = train.fit_calibrated_multi(
            "gbm",
            odds_matrix[primary_mask],
            labels[primary_mask],
            odds_columns,
            methods=("sigmoid",),
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=_policy_weights(periods, primary_mask, policy, half_life_days),
        )
        odds_components = [
            Component(
                "opening_odds_lightgbm_cal",
                calibrators["sigmoid"],
                None,
                1.0,
            )
        ]
    elif architecture == "single_mixed_lgbm":
        _base, calibrators = train.fit_calibrated_multi(
            "gbm",
            odds_matrix[fit_mask],
            labels[fit_mask],
            odds_columns,
            methods=("sigmoid",),
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=sample_weight,
        )
        mixed_components = [
            Component(
                "mixed_opening_odds_lightgbm_cal",
                calibrators["sigmoid"],
                None,
                1.0,
            )
        ]

    return ModelArtifact(
        feature_columns=feature_columns,
        components=components,
        metadata=metadata,
        prediction_architecture=architecture,
        odds_feature_columns=odds_columns,
        odds_components=odds_components,
        mixed_feature_columns=odds_columns,
        mixed_components=mixed_components,
    )


def _prediction_rows(
    artifact: ModelArtifact,
    feature_rows: list[dict[str, float]],
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, Any]]:
    selected = [feature_rows[index] for index in indices]
    probabilities = artifact.predict_proba_team1(selected)
    return [
        {
            "match_id": metadata[index]["id"],
            "date": metadata[index]["date"],
            "actual": int(labels[index]),
            "prob_team1": float(probability),
            "prediction_regime": artifact.prediction_regime(feature_rows[index]),
        }
        for index, probability in zip(indices, probabilities, strict=True)
    ]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return metric_dict(
        np.asarray([row["actual"] for row in rows], dtype=int),
        np.asarray([row["prob_team1"] for row in rows], dtype=float),
    )


def _write_predictions(path: Path, model: str, rows: list[dict[str, Any]]) -> None:
    fields = ["model", "match_id", "date", "actual", "prob_team1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": model, **{key: row[key] for key in fields[1:]}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "BBDD" / "cs2.db"))
    parser.add_argument("--config", default=str(ROOT / "MODEL" / "config.yaml"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "MODEL" / "results" / "enhanced_info_ablation"),
    )
    parser.add_argument("--half-life-days", type=float, default=None)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    runtime = load_config(args.config)
    set_runtime_config(runtime)
    set_global_determinism(runtime.random_seed)
    incumbent, incumbent_reference = train._load_validated_incumbent()
    if incumbent is None or incumbent_reference is None:
        raise SystemExit("Enhanced-info ablation requires a validated live model")
    cutoff = str(incumbent.metadata.get("date_max") or "")[:10]
    if not cutoff:
        raise SystemExit("Live model has no date_max cutoff")
    half_life_days = float(
        args.half_life_days if args.half_life_days is not None else runtime.training.recency_half_life_days
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = train.dataio.load_training_rows(
        None,
        ROOT / "PIPELINE" / "master" / "matches.json",
        cs2_only=True,
        db_path=args.db,
    )
    if not rows:
        raise SystemExit("No CS2 training rows")
    # Mandatory warm-up: exactly one chronological reconstruction over every
    # raw row.  The masks below never alter or delete this history.
    feature_rows, labels_list, metadata, _state = build_training_frame(
        rows, form_half_life=runtime.training.form_half_life_days
    )
    coverage = annotate_enhanced_info(feature_rows, rows, metadata)
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
    incumbent_rows = [
        {
            "match_id": item["id"],
            "date": item["date"],
            "actual": int(label),
            "features": features,
        }
        for features, label, item in zip(feature_rows, labels, metadata, strict=True)
    ]

    variants: dict[str, Any] = {}
    shadows: dict[str, ModelArtifact] = {}
    for policy in POLICIES:
        eligible = training_indices(feature_rows, policy)
        fit_mask = prefix_mask & eligible
        shadow = fit_fixed_architecture(
            incumbent,
            feature_rows,
            labels,
            periods,
            fit_mask,
            policy=policy,
            half_life_days=half_life_days,
            calibration_fraction=runtime.training.final_calibration_fraction,
            seed=runtime.random_seed,
            metadata={"date_max": cutoff, "training_row_policy": policy},
        )
        predictions = _prediction_rows(shadow, feature_rows, labels, metadata, holdout_indices)
        decision = compare_candidate_to_incumbent(
            incumbent,
            predictions,
            incumbent_rows,
            min_samples=runtime.promotion.min_common_holdout_rows,
            log_loss_epsilon=runtime.promotion.log_loss_epsilon,
            brier_epsilon=runtime.promotion.brier_epsilon,
        )
        variants[policy] = {
            "fit_rows": int(fit_mask.sum()),
            "holdout": _metrics(predictions),
            "gate": decision.to_dict(),
            "predictions": predictions,
        }
        shadows[policy] = shadow

    winner = min(
        POLICIES,
        key=lambda policy: (
            variants[policy]["holdout"]["log_loss"],
            variants[policy]["holdout"]["brier"],
        ),
    )
    winner_gate = variants[winner]["gate"]
    bias = enhanced_bias_report(feature_rows, metadata)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_reference_accuracy_long_walk_forward": {
            "accuracy": 0.6440,
            "correct": 5097,
            "n": 7914,
            "role": "product reference only; promotion is decided on common holdout",
        },
        "live": {
            "version": incumbent_reference.version,
            "date_max": cutoff,
            "architecture": str(getattr(incumbent, "prediction_architecture", "model_a_no_odds") or "model_a_no_odds"),
        },
        "warmup": {
            "raw_rows_preserved": len(rows),
            "features_built_before_filtering": True,
            "state_reconstructions": 1,
        },
        "enhanced_info": coverage,
        "recency_weighting": {
            "half_life_days": half_life_days,
            "formula": "max(1e-3, 0.5 ** (age_days / half_life_days))",
        },
        "common_holdout": {
            "n": len(holdout_indices),
            "date_min": metadata[holdout_indices[0]]["date"],
            "date_max": metadata[holdout_indices[-1]]["date"],
        },
        "variants": {
            policy: {key: value for key, value in payload.items() if key != "predictions"}
            for policy, payload in variants.items()
        },
        "winner": winner,
        "winner_beats_live_gate": bool(winner_gate["promote"]),
        "bias": bias,
        "promotion": {
            "requested": bool(args.promote),
            "performed": False,
            "message": (
                "winner rejected; live and last_good unchanged"
                if not winner_gate["promote"]
                else "winner approved; pass --promote to publish"
            ),
        },
    }

    report_path = output / "enhanced_info_ablation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.promote or not winner_gate["promote"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    eligible = training_indices(feature_rows, winner)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    production_name = f"enhanced_info_{winner}"
    artifact_metadata = deepcopy(incumbent.metadata)
    artifact_metadata.update(
        {
            "trained_at": trained_at,
            "date_min": rows[0]["date"],
            "date_max": rows[-1]["date"],
            "n_train_rows": int(eligible.sum()),
            "production_model": production_name,
            "evaluation_policy": production_name,
            "training_row_policy": winner,
            "enhanced_info": coverage,
            "enhanced_info_bias": bias,
            "recency_half_life_days": (half_life_days if winner == "recency_weighted" else 0.0),
            "promotion_decision": winner_gate,
            "random_seed": runtime.random_seed,
        }
    )
    final_artifact = fit_fixed_architecture(
        incumbent,
        feature_rows,
        labels,
        periods,
        eligible,
        policy=winner,
        half_life_days=half_life_days,
        calibration_fraction=runtime.training.final_calibration_fraction,
        seed=runtime.random_seed,
        metadata=artifact_metadata,
    )
    candidate_path = output / "candidate_model.pkl"
    final_artifact.save(candidate_path)

    winning_predictions = variants[winner]["predictions"]
    winning_metrics = variants[winner]["holdout"]
    glicko_probabilities = np.asarray(
        [float(feature_rows[index].get("glicko_prob_centered", 0.0)) + 0.5 for index in holdout_indices]
    )
    metrics = {
        production_name: winning_metrics,
        "glicko": metric_dict(labels[holdout_indices], glicko_probabilities),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(output / "predictions_walkforward.csv", production_name, winning_predictions)

    from health_gates import store_report, validate_candidate

    health = validate_candidate(Path(args.db), candidate_path, output)
    store_report(Path(args.db), health)
    if health["status"] != "pass":
        raise SystemExit("Enhanced-info winner failed candidate health gates")
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
    final_health = validate_candidate(Path(args.db), registered_model, output)
    store_report(Path(args.db), final_health)
    if final_health["status"] != "pass":
        raise SystemExit("Registered enhanced-info winner failed final health gates")
    decision = compare_candidate_to_incumbent(
        incumbent,
        winning_predictions,
        incumbent_rows,
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

"""Rebuild and audit one historical champion/challenger comparison.

This command is read-only with respect to the database, model registry and
production pointers. It recreates the frozen promotion shadow from the recipe
stored in a registry bundle, then checks its cohort and prediction fingerprints
against the original decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import train
from cs2model import dataio
from cs2model.artifacts import (
    ColumnSubsetEstimator,
    Component,
    ModelArtifact,
    ProbabilityColumnEstimator,
    load_artifact,
)
from cs2model.config import load_config, set_runtime_config
from cs2model.metrics import metric_dict
from cs2model.odds import ODDS_FEATURE_COLUMNS, training_opening_odds
from cs2model.promotion import (
    compare_candidate_to_incumbent,
    reference_for_version,
    sha256_file,
)
from cs2model.reproducibility import dataset_fingerprint, set_global_determinism


CS2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = CS2_ROOT / "MODEL" / "artifacts" / "registry"
DEFAULT_DB = CS2_ROOT / "BBDD" / "cs2.db"
DEFAULT_OUTPUT = CS2_ROOT / "MODEL" / "results" / "promotion_consistency_audit.json"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _metrics_by_regime(
    labels: np.ndarray,
    incumbent: np.ndarray,
    challenger: np.ndarray,
    regimes: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for regime in ("odds", "no_odds"):
        indices = np.asarray(
            [index for index, value in enumerate(regimes) if value == regime],
            dtype=int,
        )
        if len(indices) == 0:
            continue
        output[regime] = {
            "n": int(len(indices)),
            "incumbent": metric_dict(labels[indices], incumbent[indices]),
            "challenger": metric_dict(labels[indices], challenger[indices]),
        }
    return output


def _fit_shadow(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    envelope: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[ModelArtifact, list[dict[str, float]], list[int], list[dict[str, Any]], np.ndarray]:
    recipe = metadata["promotion_recipe"]
    arguments = manifest.get("arguments") or {}
    form_half_life = float(arguments.get("form_half_life", 120.0))
    X_dicts, labels, match_meta, _state = train.build_training_frame(rows, form_half_life=form_half_life)
    model_columns = [str(value) for value in envelope["feature_columns"]]
    learner_columns = [str(value) for value in recipe["feature_columns"]]
    missing = sorted(set(learner_columns) - set(model_columns))
    if missing:
        raise ValueError(f"Recipe columns absent from artifact feature contract: {missing}")
    X_all = train._matrix(X_dicts, model_columns)
    y_all = np.asarray(labels, dtype=int)
    periods = np.asarray([train._period_index(row.get("date_obj")) for row in rows], dtype=int)
    cutoff = str(recipe["cutoff"])
    recipe_mask = np.asarray(
        [str(row.get("date") or "")[:10] <= cutoff for row in match_meta],
        dtype=bool,
    )
    learner_indices = [model_columns.index(column) for column in learner_columns]
    calibration = str(recipe["calibration"])
    calibration_fraction = float(recipe["final_calibration_fraction"])
    recency_half_life = float(recipe["recency_half_life_days"])
    seed = int(recipe["seed"])
    logistic_c = float(recipe["logistic_c"])
    component_names = [str(value) for value in recipe["component_names"]]
    component_weights = [float(value) for value in recipe["component_weights"]]
    kind_by_candidate = {candidate: kind for kind, candidate in train.INDIVIDUAL_CANDIDATE.items()}
    kinds = sorted({kind_by_candidate[name] for name in component_names if name in kind_by_candidate})
    fitted: dict[str, Any] = {}
    fit_weights = train._recency_weights(periods[recipe_mask], recency_half_life)
    for kind in kinds:
        fitted[kind] = train.fit_calibrated_multi(
            kind,
            X_all[recipe_mask][:, learner_indices],
            y_all[recipe_mask],
            learner_columns,
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=fit_weights,
            estimator_params={"C": logistic_c} if kind == "logistic" else None,
        )

    display_names = {
        "logistic": "logistic",
        "gbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
        "random_forest": "random_forest",
    }
    components: list[Component] = []
    for name, weight in zip(component_names, component_weights, strict=True):
        if name in train.RATING_CANDIDATE_COLUMNS:
            column_index = model_columns.index(train.RATING_CANDIDATE_COLUMNS[name])
            estimator: ProbabilityColumnEstimator = train.fit_probability_column_estimator(
                X_all[recipe_mask],
                y_all[recipe_mask],
                column_index,
                sample_weight=fit_weights,
            )
            components.append(Component(name, estimator, None, weight))
            continue
        kind = kind_by_candidate[name]
        calibrated = fitted[kind][1].get(calibration) or fitted[kind][1].get("sigmoid")
        if calibrated is None:
            raise ValueError(f"Missing {calibration} calibrator for {kind}")
        components.append(
            Component(
                display_names[kind],
                ColumnSubsetEstimator(calibrated, learner_indices),
                None,
                weight,
            )
        )

    mixed_columns = learner_columns + list(ODDS_FEATURE_COLUMNS)
    mixed_dicts = train.add_odds_features(X_dicts, rows)
    X_mixed = train._matrix(mixed_dicts, mixed_columns)
    odds_index = mixed_columns.index("odds_available")
    validated_odds = np.asarray(X_mixed[:, odds_index] >= 0.5, dtype=bool)
    architecture = str(recipe["prediction_architecture"])
    odds_components: list[Component] = []
    mixed_components: list[Component] = []
    if architecture == "router_two_models":
        primary_mask = recipe_mask & validated_odds
        _base, calibrators = train.fit_calibrated_multi(
            "gbm",
            X_mixed[primary_mask],
            y_all[primary_mask],
            mixed_columns,
            methods=("sigmoid",),
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=train._recency_weights(periods[primary_mask], recency_half_life),
        )
        primary = calibrators.get("sigmoid")
        if primary is None:
            raise ValueError("Could not rebuild the opening-odds calibrator")
        odds_components.append(Component("opening_odds_lightgbm_cal", primary, None, 1.0))
    elif architecture == "single_mixed_lgbm":
        _base, calibrators = train.fit_calibrated_multi(
            "gbm",
            X_mixed[recipe_mask],
            y_all[recipe_mask],
            mixed_columns,
            methods=("sigmoid",),
            cal_frac=calibration_fraction,
            random_state=seed,
            sample_weight=fit_weights,
        )
        mixed = calibrators.get("sigmoid")
        if mixed is None:
            raise ValueError("Could not rebuild the mixed calibrator")
        mixed_components.append(Component("mixed_opening_odds_lightgbm_cal", mixed, None, 1.0))

    shadow = ModelArtifact(
        feature_columns=model_columns,
        components=components,
        metadata={"date_max": cutoff, "recipe_sha256": metadata["promotion_recipe_sha256"]},
        prediction_architecture=architecture,
        odds_feature_columns=mixed_columns,
        odds_components=odds_components,
        mixed_feature_columns=mixed_columns,
        mixed_components=mixed_components,
    )
    return shadow, mixed_dicts, labels, match_meta, periods


def audit(
    version: str,
    *,
    registry: Path = DEFAULT_REGISTRY,
    db_path: Path = DEFAULT_DB,
    require_dataset_match: bool = True,
) -> dict[str, Any]:
    bundle = registry / version
    envelope = _json(bundle / "metadata.json")
    metadata = envelope["metadata"]
    manifest = _json(bundle / "experiment_manifest.json")
    original_decision = _json(bundle / "promotion_decision.json")
    config = load_config(bundle / "config.yaml")
    set_runtime_config(config)
    seed = int((metadata.get("promotion_recipe") or {}).get("seed", 42))
    set_global_determinism(seed)

    current_rows = dataio.load_training_rows_from_db(db_path, cs2_only=True)
    candidate_max = str(metadata["date_max"])[:10]
    rows = [row for row in current_rows if str(row.get("date") or "")[:10] <= candidate_max]
    current_dataset_sha256 = dataset_fingerprint(rows)
    expected_dataset_sha256 = str(manifest["dataset_sha256"])
    if current_dataset_sha256 != expected_dataset_sha256 and require_dataset_match:
        raise ValueError(
            "Historical input rows no longer match the candidate manifest: "
            f"expected={expected_dataset_sha256} current={current_dataset_sha256}"
        )

    shadow, feature_dicts, labels, match_meta, periods = _fit_shadow(rows, metadata, envelope, manifest)
    cutoff = str(original_decision["cutoff"])
    holdout_indices = [index for index, row in enumerate(match_meta) if str(row.get("date") or "")[:10] > cutoff]
    holdout_features = [feature_dicts[index] for index in holdout_indices]
    holdout_labels = np.asarray([labels[index] for index in holdout_indices], dtype=int)
    challenger_probabilities = shadow.predict_proba_team1(holdout_features)
    regimes = [shadow.prediction_regime(row) for row in holdout_features]
    challenger_rows = [
        {
            "match_id": match_meta[index]["id"],
            "date": match_meta[index]["date"],
            "actual": int(labels[index]),
            "prob_team1": float(probability),
            "prediction_regime": regime,
        }
        for index, probability, regime in zip(holdout_indices, challenger_probabilities, regimes, strict=True)
    ]
    feature_rows = [
        {
            "match_id": item["id"],
            "date": item["date"],
            "actual": int(label),
            "features": features,
        }
        for item, label, features in zip(match_meta, labels, feature_dicts, strict=True)
    ]
    incumbent_version = str(original_decision["incumbent_version"])
    incumbent_reference = reference_for_version(registry, incumbent_version)
    incumbent = load_artifact(incumbent_reference.artifact)
    if incumbent is None:
        raise ValueError(f"Could not load incumbent {incumbent_version}")
    rebuilt_decision = compare_candidate_to_incumbent(
        incumbent,
        challenger_rows,
        feature_rows,
        min_samples=int(original_decision["min_samples"]),
        log_loss_epsilon=float(original_decision["log_loss_epsilon"]),
        brier_epsilon=float(original_decision["brier_epsilon"]),
    )
    if rebuilt_decision.challenger is None:
        raise ValueError("Rebuilt promotion cohort was unexpectedly insufficient")
    incumbent_probabilities = incumbent.predict_proba_team1(holdout_features)

    odds_evaluation = metadata["odds_architecture_evaluation"]
    fold_periods = {int(item["period"]) for item in odds_evaluation.get("folds") or []}
    oos_indices = [index for index, period in enumerate(periods.tolist()) if int(period) in fold_periods]
    oos_ids = {str(match_meta[index]["id"]) for index in oos_indices}
    gate_ids = {str(match_meta[index]["id"]) for index in holdout_indices}
    oos_dates = [str(match_meta[index]["date"]) for index in oos_indices]
    gate_dates = [str(match_meta[index]["date"]) for index in holdout_indices]
    oos_odds_n = sum(training_opening_odds(rows[index]) is not None for index in oos_indices)
    gate_odds_n = regimes.count("odds")
    recipe = metadata["promotion_recipe"]
    stored_gate = {
        "log_loss": float(original_decision["challenger"]["log_loss"]),
        "brier": float(original_decision["challenger"]["brier"]),
    }
    rebuilt_gate = {
        "log_loss": float(rebuilt_decision.challenger.log_loss),
        "brier": float(rebuilt_decision.challenger.brier),
    }
    return {
        "status": "pass"
        if (
            rebuilt_decision.holdout_sha256 == original_decision["holdout_sha256"]
            and rebuilt_decision.prediction_sha256 == original_decision["prediction_sha256"]
        )
        else "mismatch",
        "registry_version": version,
        "dataset": {
            "rows": len(rows),
            "date_min": str(rows[0]["date"]),
            "date_max": str(rows[-1]["date"]),
            "sha256": current_dataset_sha256,
            "matches_manifest": current_dataset_sha256 == expected_dataset_sha256,
        },
        "comparison": {
            "oos_prequential": {
                "n": len(oos_indices),
                "date_min": min(oos_dates),
                "date_max": max(oos_dates),
                "odds_n": oos_odds_n,
                "no_odds_n": len(oos_indices) - oos_odds_n,
                "no_odds_fraction": (len(oos_indices) - oos_odds_n) / len(oos_indices),
                "metrics": odds_evaluation["metrics"]["router_two_models"],
                "calibration": (
                    "legacy implementation default: sigmoid tail=0.20, opening-odds branch without recency weights"
                ),
                "object": "different fitted router per weekly fold",
            },
            "promotion_gate": {
                "n": len(holdout_indices),
                "date_min": min(gate_dates),
                "date_max": max(gate_dates),
                "odds_n": gate_odds_n,
                "no_odds_n": len(holdout_indices) - gate_odds_n,
                "no_odds_fraction": (len(holdout_indices) - gate_odds_n) / len(holdout_indices),
                "incumbent_metrics": original_decision["incumbent"],
                "challenger_metrics": stored_gate,
                "rebuilt_challenger_metrics": rebuilt_gate,
                "by_regime": _metrics_by_regime(
                    holdout_labels,
                    np.asarray(incumbent_probabilities, dtype=float),
                    np.asarray(challenger_probabilities, dtype=float),
                    regimes,
                ),
                "calibration": (
                    f"{recipe['calibration']} tail={recipe['final_calibration_fraction']}, "
                    f"recency half-life={recipe['recency_half_life_days']} days"
                ),
                "object": "one shadow frozen at the incumbent cutoff",
            },
            "overlap": {
                "n": len(oos_ids & gate_ids),
                "oos_share": len(oos_ids & gate_ids) / len(oos_ids),
                "gate_share": len(oos_ids & gate_ids) / len(gate_ids),
            },
        },
        "identity": {
            "candidate_artifact_sha256": sha256_file(bundle / "model.pkl"),
            "candidate_artifact_trained_through": candidate_max,
            "recipe_sha256": metadata["promotion_recipe_sha256"],
            "gate_recipe_sha256": original_decision["recipe_sha256"],
            "same_recipe": (metadata["promotion_recipe_sha256"] == original_decision["recipe_sha256"]),
            "same_fitted_object_oos_vs_gate": False,
            "reason": (
                "OOS is a prequential stream of fold-specific refits; the gate uses "
                "one shadow fitted only through the incumbent cutoff."
            ),
        },
        "fingerprints": {
            "stored_holdout_sha256": original_decision["holdout_sha256"],
            "rebuilt_holdout_sha256": rebuilt_decision.holdout_sha256,
            "stored_prediction_sha256": original_decision["prediction_sha256"],
            "rebuilt_prediction_sha256": rebuilt_decision.prediction_sha256,
        },
        "odds_contract": {
            "same_training_opening_odds_function": True,
            "rule": (
                "first stored opening with captured_at_utc < kickoff_utc; coherent two-way market; de-vig to sum 1"
            ),
        },
        "rebuilt_decision": rebuilt_decision.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Registry version YYYYMMDD_HHMMSSZ")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-dataset-drift",
        action="store_true",
        help=(
            "Continue diagnostically if the raw historical dataset fingerprint changed; "
            "the rebuilt prediction fingerprint still has to match for status=pass."
        ),
    )
    args = parser.parse_args()
    report = audit(
        args.version,
        registry=args.registry,
        db_path=args.db,
        require_dataset_match=not args.allow_dataset_drift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Richer BO3 scoreline target with temporal evaluation and auto-activation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

BO3_SCORE_CLASSES = ("0-2", "1-2", "2-1", "2-0")
BO3_SCORE_TO_CLASS = {(0, 2): 0, (1, 2): 1, (2, 1): 2, (2, 0): 3}


def bo3_score_class(row: dict[str, Any]) -> int | None:
    if str(row.get("format") or "").lower() != "bo3":
        return None
    try:
        score = (int(row.get("score1")), int(row.get("score2")))
    except (TypeError, ValueError):
        return None
    return BO3_SCORE_TO_CLASS.get(score)


def rich_target_policy(
    rows: list[dict[str, Any]],
    min_rows: int = 2000,
    min_class_rows: int = 300,
) -> dict[str, Any]:
    labels = [bo3_score_class(row) for row in rows]
    counts = Counter(label for label in labels if label is not None)
    class_counts = {BO3_SCORE_CLASSES[index]: int(counts.get(index, 0)) for index in range(4)}
    eligible = sum(class_counts.values())
    coverage_ready = eligible >= min_rows and min(class_counts.values(), default=0) >= min_class_rows
    return {
        "available_rows": eligible,
        "min_rows": min_rows,
        "min_class_rows": min_class_rows,
        "class_counts": class_counts,
        "coverage_ready": coverage_ready,
        "enabled": False,
        "activation": "automatic_after_walk_forward_multiclass_beats_empirical_baseline",
        "production_scope": "scoreline_props_auxiliary_not_winner_model_a",
    }


def _make_estimator(random_state: int = 42):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=0.5, random_state=random_state)),
    ])


def _augment_scorelines(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    directional_columns: Iterable[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    directional = set(directional_columns)
    reverse = np.asarray(X, dtype=float).copy()
    for index, column in enumerate(columns):
        if column in directional:
            reverse[:, index] *= -1.0
    weights = None if sample_weight is None else np.concatenate([sample_weight, sample_weight])
    return np.vstack([X, reverse]), np.concatenate([y, 3 - y]), weights


def _align_probabilities(estimator: Any, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = [int(value) for value in estimator.classes_]
    aligned = np.zeros((len(X), 4), dtype=float)
    for source_index, class_index in enumerate(classes):
        if 0 <= class_index < 4:
            aligned[:, class_index] = raw[:, source_index]
    totals = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, totals, out=np.full_like(aligned, 0.25), where=totals > 0)


def _multiclass_log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    selected = np.clip(probabilities[np.arange(len(y)), y], 1e-9, 1.0)
    return float(-np.mean(np.log(selected)))


def evaluate_rich_target_walk_forward(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    periods: np.ndarray,
    columns: list[str],
    directional_columns: Iterable[str],
    warmup_weeks: int,
    min_train: int = 1000,
    gap: int = 0,
    recency_half_life: float = 365.0,
    min_rows: int = 2000,
    min_class_rows: int = 300,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Temporal scoreline evaluation; no random split and no post-match features."""
    policy = rich_target_policy(rows, min_rows=min_rows, min_class_rows=min_class_rows)
    if not policy["coverage_ready"]:
        policy["note"] = "stored labels, waiting for balanced BO3 coverage"
        return policy

    labels = np.array([bo3_score_class(row) if bo3_score_class(row) is not None else -1 for row in rows], dtype=int)
    eligible = labels >= 0
    predictions: list[dict[str, Any]] = []
    baseline_losses: list[float] = []
    unique_periods = sorted(set(periods[eligible].tolist()))
    for fold_index, period in enumerate(unique_periods[warmup_weeks:]):
        train_mask = eligible & (periods < (period - gap))
        test_mask = eligible & (periods == period)
        if int(train_mask.sum()) < min_train or int(test_mask.sum()) == 0:
            continue
        y_train = labels[train_mask]
        counts = np.bincount(y_train, minlength=4)
        if int(counts.min()) < 20:
            continue
        sample_weight = None
        if recency_half_life and recency_half_life > 0:
            train_periods = periods[train_mask].astype(float)
            ages_days = (train_periods.max() - train_periods) * 7.0
            sample_weight = np.clip(0.5 ** (ages_days / recency_half_life), 1e-3, 1.0)
        X_train, y_aug, weight_aug = _augment_scorelines(
            X[train_mask], y_train, columns, directional_columns, sample_weight
        )
        estimator = _make_estimator(random_state=random_seed + fold_index)
        fit_kwargs = {"model__sample_weight": weight_aug} if weight_aug is not None else {}
        estimator.fit(X_train, y_aug, **fit_kwargs)
        test_indices = np.where(test_mask)[0]
        probabilities = _align_probabilities(estimator, X[test_indices])
        baseline = (counts + 1.0) / (counts.sum() + 4.0)
        for local_index, global_index in enumerate(test_indices):
            label = int(labels[global_index])
            probs = probabilities[local_index]
            baseline_losses.append(-math.log(float(np.clip(baseline[label], 1e-9, 1.0))))
            predictions.append({
                "match_id": rows[global_index].get("id"),
                "date": rows[global_index].get("date"),
                "actual_class": BO3_SCORE_CLASSES[label],
                "actual": int(label >= 2),
                "predicted_class": BO3_SCORE_CLASSES[int(np.argmax(probs))],
                "prob_team1": float(probs[2] + probs[3]),
                "score_probabilities": {name: float(probs[index]) for index, name in enumerate(BO3_SCORE_CLASSES)},
            })
    if not predictions:
        policy["note"] = "coverage ready but no evaluable temporal folds"
        return policy

    y_eval = np.array([BO3_SCORE_CLASSES.index(row["actual_class"]) for row in predictions], dtype=int)
    probs_eval = np.array([
        [row["score_probabilities"][name] for name in BO3_SCORE_CLASSES]
        for row in predictions
    ])
    multiclass_loss = _multiclass_log_loss(y_eval, probs_eval)
    baseline_loss = float(np.mean(baseline_losses))
    policy.update({
        "enabled": bool(multiclass_loss < baseline_loss),
        "n_walk_forward": len(predictions),
        "multiclass_log_loss": round(multiclass_loss, 6),
        "empirical_baseline_log_loss": round(baseline_loss, 6),
        "scoreline_accuracy": round(float(np.mean(np.argmax(probs_eval, axis=1) == y_eval)), 6),
        "winner_accuracy": round(float(np.mean((probs_eval[:, 2:].sum(axis=1) >= 0.5) == (y_eval >= 2))), 6),
        "predictions": predictions,
        "note": "auxiliary scoreline model enabled" if multiclass_loss < baseline_loss else "did not beat empirical scoreline baseline",
    })
    return policy


def fit_rich_target_model(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    columns: list[str],
    directional_columns: Iterable[str],
    sample_weight: np.ndarray | None = None,
    random_state: int = 42,
) -> Any:
    labels = np.array([bo3_score_class(row) if bo3_score_class(row) is not None else -1 for row in rows], dtype=int)
    mask = labels >= 0
    if int(mask.sum()) == 0:
        return None
    weights = None if sample_weight is None else np.asarray(sample_weight)[mask]
    X_aug, y_aug, weights_aug = _augment_scorelines(X[mask], labels[mask], columns, directional_columns, weights)
    estimator = _make_estimator(random_state=random_state)
    fit_kwargs = {"model__sample_weight": weights_aug} if weights_aug is not None else {}
    estimator.fit(X_aug, y_aug, **fit_kwargs)
    return estimator

"""Nested, purged Optuna tuning for the logistic component."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _augment(X: np.ndarray, y: np.ndarray, columns: list[str], directional: set[str]) -> tuple[np.ndarray, np.ndarray]:
    reverse = np.asarray(X, dtype=float).copy()
    for index, column in enumerate(columns):
        if column in directional:
            reverse[:, index] *= -1.0
    return np.vstack([X, reverse]), np.concatenate([y, 1 - y])


def _pipeline(c_value: float, random_state: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, C=float(c_value), random_state=random_state)),
    ])


def tune_logistic_c_purged(
    X: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    columns: list[str],
    directional_columns: Iterable[str],
    n_trials: int = 8,
    gap: int = 1,
    max_inner_folds: int = 4,
    min_train: int = 400,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimize log loss on inner expanding folds separated by an embargo."""
    base = {
        "available": False,
        "enabled": False,
        "best_c": 0.5,
        "n_trials": int(max(0, n_trials)),
        "gap": int(max(1, gap)),
        "objective": "purged_inner_cv_log_loss",
    }
    if n_trials <= 0 or len(X) < min_train + 20:
        base["reason"] = "disabled_or_insufficient_rows"
        return base
    try:
        import optuna
    except Exception:
        base["reason"] = "optuna_not_installed"
        return base

    unique = sorted(set(np.asarray(periods, dtype=int).tolist()))
    candidate_periods = [
        period for period in unique
        if int(np.sum(periods < period - max(1, gap))) >= min_train and int(np.sum(periods == period)) > 0
    ]
    inner_periods = candidate_periods[-max_inner_folds:]
    if len(inner_periods) < 2:
        base["reason"] = "insufficient_purged_inner_folds"
        return base
    directional = set(directional_columns)

    def objective(trial: Any) -> float:
        c_value = trial.suggest_float("logistic_c", 0.02, 5.0, log=True)
        losses: list[float] = []
        for period in inner_periods:
            train_mask = periods < period - max(1, gap)
            valid_mask = periods == period
            X_train, y_train = _augment(X[train_mask], y[train_mask], columns, directional)
            estimator = _pipeline(c_value, seed)
            estimator.fit(X_train, y_train)
            probabilities = np.clip(estimator.predict_proba(X[valid_mask])[:, 1], 1e-6, 1.0 - 1e-6)
            actual = y[valid_mask]
            losses.extend((-(actual * np.log(probabilities) + (1 - actual) * np.log(1 - probabilities))).tolist())
        return float(np.mean(losses)) if losses else math.inf

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    base.update({
        "available": True,
        "enabled": True,
        "best_c": float(study.best_params["logistic_c"]),
        "best_log_loss": float(study.best_value),
        "inner_folds": len(inner_periods),
        "inner_periods": [int(value) for value in inner_periods],
        "reason": "optimized_using_past_only",
    })
    return base

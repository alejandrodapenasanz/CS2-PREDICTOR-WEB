"""Feature-pruning diagnostics with a conservative anti-leakage policy."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _impute_from_train(X_train: np.ndarray, X_other: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(X_train), X_train, medians)
    other = None if X_other is None else np.where(np.isfinite(X_other), X_other, medians)
    return train, other


def prune_exact_redundancies(
    X: np.ndarray,
    columns: list[str],
    protected: Iterable[str] = (),
    atol: float = 1e-12,
) -> tuple[list[str], dict[str, Any]]:
    """Remove only constants and mathematically duplicate columns.

    This operation never uses the target. Supervised diagnostics are kept
    report-only, so feature selection cannot leak future labels into early
    walk-forward folds.
    """
    if X.ndim != 2 or X.shape[1] != len(columns):
        raise ValueError("X/columns shape mismatch")
    protected_set = set(protected)
    clean, _ = _impute_from_train(np.asarray(X, dtype=float))
    keep: list[int] = []
    constants: list[str] = []
    redundant: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        values = clean[:, index]
        if column not in protected_set and np.allclose(values, values[0], atol=atol, rtol=0.0):
            constants.append(column)
            continue
        duplicate_of: tuple[str, int] | None = None
        for kept_index in keep:
            kept_column = columns[kept_index]
            if column in protected_set and kept_column in protected_set:
                continue
            if np.allclose(values, clean[:, kept_index], atol=atol, rtol=0.0):
                duplicate_of = (kept_column, 1)
                break
            if np.allclose(values, -clean[:, kept_index], atol=atol, rtol=0.0):
                duplicate_of = (kept_column, -1)
                break
        if duplicate_of and column not in protected_set:
            redundant.append({"feature": column, "duplicate_of": duplicate_of[0], "sign": duplicate_of[1]})
            continue
        keep.append(index)
    kept_columns = [columns[index] for index in keep]
    return kept_columns, {
        "policy": "target_free_exact_only",
        "input_columns": len(columns),
        "output_columns": len(kept_columns),
        "removed_constants": constants,
        "removed_exact_redundancies": redundant,
    }


def feature_pruning_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    shap_rows: list[dict[str, Any]] | None = None,
    holdout_fraction: float = 0.20,
    permutation_repeats: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """A5: VIF + chronological permutation importance + RFE.

    The first chronological block is used for fitting and the last block only
    for permutation scoring. Results suggest candidates; they do not mutate the
    production columns because supervised pruning needs nested walk-forward.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != len(columns) or len(X) < 100 or len(columns) < 2:
        return {"available": False, "reason": "insufficient_rows_or_columns", "n_rows": len(X), "n_columns": len(columns)}
    split = max(50, min(len(X) - 20, int(len(X) * (1.0 - holdout_fraction))))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"available": False, "reason": "chronological_split_has_one_class", "n_rows": len(X), "n_columns": len(columns)}

    X_train, X_test = _impute_from_train(X_train, X_test)
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    estimator = LogisticRegression(max_iter=1000, C=0.5, random_state=random_state)
    estimator.fit(X_train_scaled, y_train)
    permutation = permutation_importance(
        estimator,
        X_test_scaled,
        y_test,
        scoring="neg_log_loss",
        n_repeats=permutation_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    n_select = max(5, len(columns) // 2)
    rfe = RFE(
        estimator=LogisticRegression(max_iter=1000, C=0.5, random_state=random_state),
        n_features_to_select=min(n_select, len(columns)),
        step=max(1, len(columns) // 10),
    )
    rfe.fit(X_train_scaled, y_train)

    corr = np.corrcoef(X_train_scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    inverse = np.linalg.pinv(corr + np.eye(len(columns)) * 1e-9)
    vif_values = np.clip(np.diag(inverse), 1.0, 1_000_000.0)
    max_pair_corr: list[float] = []
    for index in range(len(columns)):
        values = np.abs(corr[index]).copy()
        values[index] = 0.0
        max_pair_corr.append(float(values.max()) if len(values) else 0.0)

    shap_by_feature = {
        str(row.get("feature")): float(row.get("mean_abs_shap") or 0.0)
        for row in (shap_rows or [])
    }
    shap_order = sorted(columns, key=lambda feature: shap_by_feature.get(feature, 0.0), reverse=True)
    shap_rank = {feature: rank for rank, feature in enumerate(shap_order, start=1)}
    perm_order = np.argsort(-permutation.importances_mean)
    perm_rank = {columns[int(index)]: rank for rank, index in enumerate(perm_order, start=1)}
    shap_low_cutoff = max(1, int(len(columns) * 0.75))

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(columns):
        review_drop = bool(
            vif_values[index] >= 10.0
            and permutation.importances_mean[index] <= 0.0
            and not bool(rfe.support_[index])
            and shap_rank[feature] >= shap_low_cutoff
        )
        rows.append({
            "feature": feature,
            "vif": round(float(vif_values[index]), 4),
            "max_abs_pair_correlation": round(max_pair_corr[index], 6),
            "permutation_logloss_mean": round(float(permutation.importances_mean[index]), 8),
            "permutation_logloss_std": round(float(permutation.importances_std[index]), 8),
            "permutation_rank": perm_rank[feature],
            "mean_abs_shap": round(shap_by_feature.get(feature, 0.0), 8),
            "shap_rank": shap_rank[feature],
            "rfe_selected": bool(rfe.support_[index]),
            "rfe_rank": int(rfe.ranking_[index]),
            "review_drop": review_drop,
        })
    rows.sort(key=lambda row: (not row["review_drop"], row["permutation_rank"]))
    return {
        "available": True,
        "policy": "diagnostic_only_supervised_pruning_requires_nested_walk_forward",
        "n_rows": len(X),
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
        "n_columns": len(columns),
        "review_drop_candidates": [row["feature"] for row in rows if row["review_drop"]],
        "features": rows,
    }

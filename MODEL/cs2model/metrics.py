"""Métricas de evaluación probabilística (PROJECT.md §8)."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(y_true)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        last = high >= 1.0
        mask = (y_prob >= low) & (y_prob <= high if last else y_prob < high)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / n) * abs(np.mean(y_prob[mask]) - np.mean(y_true[mask]))
    return float(ece)


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)
    out: dict[str, float] = {"n": int(len(y_true))}
    out["accuracy"] = float(np.mean((y_prob >= 0.5).astype(int) == y_true))
    out["log_loss"] = float(np.mean(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))))
    out["brier"] = float(np.mean((y_prob - y_true) ** 2))
    out["ece_10"] = expected_calibration_error(y_true, y_prob, 10)
    # ROC-AUC sin sklearn (rank-based).
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) and len(neg):
        order = np.argsort(y_prob)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_prob) + 1)
        # promediar rangos en empates
        _, inv, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg_rank = (cum - (counts - 1) / 2.0)
        ranks = avg_rank[inv]
        sum_pos = np.sum(ranks[y_true == 1])
        n_pos, n_neg = len(pos), len(neg)
        out["roc_auc"] = float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        out["roc_auc"] = float("nan")
    return out


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        last = high >= 1.0
        mask = (y_prob >= low) & (y_prob <= high if last else y_prob < high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i,
                "low": float(low),
                "high": float(high),
                "n": int(np.sum(mask)),
                "avg_pred": float(np.mean(y_prob[mask])),
                "observed": float(np.mean(y_true[mask])),
            }
        )
    return rows

"""Calibration diagnostics for causal backtests and frozen live predictions.

The functions in this module are descriptive.  They never fit or mutate a
model and therefore cannot turn the current live sample into a training input.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


ELO_GAP_BINS: tuple[tuple[str, float, float], ...] = (
    ("0-50", 0.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100-200", 100.0, 200.0),
    ("200-300", 200.0, 300.0),
    ("300+", 300.0, math.inf),
)
UNKNOWN_VALUES = {"", "unknown", "none", "null", "nan"}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def elo_gap_bin(value: Any) -> str | None:
    """Return the absolute pre-match Elo-gap bin used by every report."""
    parsed = _finite_float(value)
    if parsed is None:
        return None
    gap = abs(parsed)
    for label, lower, upper in ELO_GAP_BINS:
        if lower <= gap < upper:
            return label
    return ELO_GAP_BINS[-1][0]


def _segment_value(row: dict[str, Any], key: str) -> str:
    if key == "elo_gap_bin":
        return elo_gap_bin(row.get("elo_diff")) or "unknown"
    value = str(row.get(key) or "unknown").strip().lower()
    return value if value not in UNKNOWN_VALUES else "unknown"


def _reliability_bins(
    outcomes: list[int], probabilities: list[float], n_bins: int
) -> tuple[list[dict[str, Any]], float, int]:
    cells: list[dict[str, Any]] = []
    weighted_gap = 0.0
    material_cells = 0
    total = len(outcomes)
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        selected = [
            row_index
            for row_index, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == n_bins - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        cell_outcomes = [outcomes[row_index] for row_index in selected]
        cell_probabilities = [probabilities[row_index] for row_index in selected]
        residuals = [outcome - probability for outcome, probability in zip(cell_outcomes, cell_probabilities)]
        mean_predicted = sum(cell_probabilities) / len(selected)
        observed_rate = sum(cell_outcomes) / len(selected)
        gap = observed_rate - mean_predicted
        if len(residuals) > 1:
            residual_mean = sum(residuals) / len(residuals)
            variance = sum((value - residual_mean) ** 2 for value in residuals) / (len(residuals) - 1)
            standard_error = math.sqrt(variance / len(residuals))
        else:
            standard_error = None
        ci_low = gap - 1.96 * standard_error if standard_error is not None else None
        ci_high = gap + 1.96 * standard_error if standard_error is not None else None
        material = bool(
            len(selected) >= 20
            and ci_low is not None
            and ci_high is not None
            and (ci_low > 0.0 or ci_high < 0.0)
            and abs(gap) >= 0.03
        )
        material_cells += int(material)
        weighted_gap += len(selected) * abs(gap)
        cells.append(
            {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "n": len(selected),
                "mean_predicted": round(mean_predicted, 6),
                "observed_rate": round(observed_rate, 6),
                "calibration_gap": round(gap, 6),
                "ci95_low": round(ci_low, 6) if ci_low is not None else None,
                "ci95_high": round(ci_high, 6) if ci_high is not None else None,
                "material_deviation": material,
            }
        )
    return cells, weighted_gap / total if total else 0.0, material_cells


def _segment_summary(
    rows: list[dict[str, Any]],
    *,
    min_n: int,
    n_bins: int,
    min_absolute_gap: float,
    min_ece: float,
) -> dict[str, Any]:
    outcomes = [int(row["actual"]) for row in rows]
    probabilities = [min(1.0 - 1e-9, max(1e-9, float(row["prob_team1"]))) for row in rows]
    n = len(rows)
    mean_predicted = sum(probabilities) / n
    observed_rate = sum(outcomes) / n
    gap = observed_rate - mean_predicted
    residuals = [outcome - probability for outcome, probability in zip(outcomes, probabilities)]
    if n > 1:
        variance = sum((value - gap) ** 2 for value in residuals) / (n - 1)
        standard_error = math.sqrt(variance / n)
    else:
        standard_error = None
    ci_low = gap - 1.96 * standard_error if standard_error is not None else None
    ci_high = gap + 1.96 * standard_error if standard_error is not None else None
    reliability, ece, material_cells = _reliability_bins(outcomes, probabilities, n_bins)
    mean_gap_material = bool(
        ci_low is not None and ci_high is not None and (ci_low > 0.0 or ci_high < 0.0) and abs(gap) >= min_absolute_gap
    )
    deviation_detected = bool(mean_gap_material or (ece >= min_ece and material_cells > 0))
    status = (
        "insufficient_sample"
        if n < min_n
        else "deviation_detected"
        if deviation_detected
        else "compatible_with_sampling_noise"
    )
    output: dict[str, Any] = {
        "n": n,
        "accuracy": round(
            sum(int((probability >= 0.5) == bool(outcome)) for outcome, probability in zip(outcomes, probabilities))
            / n,
            6,
        ),
        "log_loss": round(
            sum(
                -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability))
                for outcome, probability in zip(outcomes, probabilities)
            )
            / n,
            6,
        ),
        "brier": round(
            sum((probability - outcome) ** 2 for outcome, probability in zip(outcomes, probabilities)) / n, 6
        ),
        "mean_predicted": round(mean_predicted, 6),
        "observed_rate": round(observed_rate, 6),
        "calibration_gap": round(gap, 6),
        "ci95_low": round(ci_low, 6) if ci_low is not None else None,
        "ci95_high": round(ci_high, 6) if ci_high is not None else None,
        "ece_10": round(ece, 6),
        "reliability": reliability,
        "material_reliability_bins": material_cells,
        "status": status,
        "deviation_detected": deviation_detected if n >= min_n else False,
    }
    if n < min_n:
        output["note"] = f"sin muestra suficiente: n={n} < {min_n}; no se interpreta"
    elif deviation_detected:
        output["note"] = "desviacion descriptiva material; debe validarse por la puerta antes de cambiar el modelo"
    else:
        output["note"] = "sin desviacion material distinguible del ruido muestral con este criterio"
    return output


def segment_calibration(
    rows: Iterable[dict[str, Any]],
    key: str,
    *,
    min_n: int = 100,
    n_bins: int = 10,
    min_absolute_gap: float = 0.03,
    min_ece: float = 0.04,
) -> dict[str, Any]:
    """Measure reliability and deviation for one pre-match segmentation."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        probability = _finite_float(row.get("prob_team1"))
        outcome = row.get("actual")
        if probability is None or not 0.0 <= probability <= 1.0 or outcome not in {0, 1, False, True}:
            continue
        groups[_segment_value(row, key)].append(row)
    output: dict[str, Any] = {}
    for name, group_rows in groups.items():
        if name == "unknown":
            output[name] = {
                "n": len(group_rows),
                "status": "unknown_segment",
                "note": "segmento desconocido; excluido de conclusiones",
            }
        else:
            output[name] = _segment_summary(
                group_rows,
                min_n=min_n,
                n_bins=n_bins,
                min_absolute_gap=min_absolute_gap,
                min_ece=min_ece,
            )
    return output


def segment_calibration_suite(
    rows: Iterable[dict[str, Any]],
    *,
    min_n: int = 100,
    n_bins: int = 10,
    min_absolute_gap: float = 0.03,
    min_ece: float = 0.04,
) -> dict[str, Any]:
    """Audit Elo gap, stage and environment, retaining legacy dimensions."""
    materialized = list(rows)
    dimensions = {
        "by_elo_gap": "elo_gap_bin",
        "by_stage": "stage",
        "by_environment": "environment",
        "by_format": "format",
        "by_event_tier": "event_tier",
    }
    suite: dict[str, Any] = {
        "policy": {
            "elo_gap_bins": [label for label, _lower, _upper in ELO_GAP_BINS],
            "elo_gap_uses_absolute_difference": True,
            "min_segment_rows": min_n,
            "reliability_bins": n_bins,
            "min_absolute_gap": min_absolute_gap,
            "min_ece": min_ece,
            "criterion": "95% residual-mean CI plus material gap, or material reliability cell plus ECE",
            "multiple_testing_warning": "descriptive screen only; the temporal promotion gate decides model changes",
        }
    }
    coverage: dict[str, Any] = {}
    for label, key in dimensions.items():
        known = [row for row in materialized if _segment_value(row, key) != "unknown"]
        suite[label] = segment_calibration(
            materialized,
            key,
            min_n=min_n,
            n_bins=n_bins,
            min_absolute_gap=min_absolute_gap,
            min_ece=min_ece,
        )
        coverage[key] = {
            "known_rows": len(known),
            "total_rows": len(materialized),
            "coverage": round(len(known) / len(materialized), 6) if materialized else 0.0,
            "min_group_rows": min_n,
        }
    suite["coverage"] = coverage
    return suite

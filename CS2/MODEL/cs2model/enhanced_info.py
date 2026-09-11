"""Causal availability contract for the enhanced-information ablation.

The flag in this module is deliberately stricter than a generic ``has_data``
flag.  A source counts only when its feature family is usable *and* the stored
availability timestamp is demonstrably earlier than the match kick-off.  Box
scores, labels and any post-match payload are intentionally absent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from .dataio import _iso_date
from .odds import training_opening_odds

ENHANCED_AVAILABLE_COLUMN = "enhanced_available"
ENHANCED_SOURCES = (
    "opening_odds",
    "player_snapshots",
    "rankings",
    "analytics",
    "context",
    "announced_lineup",
)
SOURCE_FEATURE_COLUMNS = {source: f"enhanced_source_{source}" for source in ENHANCED_SOURCES}
ENHANCED_AUDIT_COLUMNS = (
    ENHANCED_AVAILABLE_COLUMN,
    *SOURCE_FEATURE_COLUMNS.values(),
)


@dataclass(frozen=True)
class EnhancedAvailability:
    available: bool
    sources: tuple[str, ...]
    evidence: dict[str, str]


def _strictly_before(captured_at: Any, kickoff_utc: Any) -> bool:
    captured = _iso_date(captured_at)
    kickoff = _iso_date(kickoff_utc)
    return captured is not None and kickoff is not None and captured < kickoff


def _source_timestamp(row: Mapping[str, Any], source: str) -> Any:
    if source == "opening_odds":
        return row.get("opening_odds_captured_at") or row.get("opening_odds_captured_at_utc")
    if source == "player_snapshots":
        return (row.get("player_snapshot_features") or {}).get("captured_at_max")
    if source == "rankings":
        return (row.get("ranking_snapshot_evidence") or {}).get("captured_at_max")
    if source == "analytics":
        return (row.get("analytics") or {}).get("captured_at")
    if source == "context":
        return row.get("context_captured_at_utc") or row.get("captured_at")
    if source == "announced_lineup":
        lineups = row.get("prematch_lineups") or row.get("announced_lineups") or {}
        return lineups.get("captured_at") if isinstance(lineups, Mapping) else None
    raise KeyError(source)


def enhanced_availability(row: Mapping[str, Any], features: Mapping[str, Any]) -> EnhancedAvailability:
    """Return all causally verified enhanced sources for one match row."""

    kickoff = row.get("kickoff_utc") or row.get("datetime_utc")
    eligible = {
        "opening_odds": training_opening_odds(row) is not None
        or float(features.get("odds_available", 0.0) or 0.0) >= 0.5,
        "player_snapshots": float(features.get("player_snapshot_available", 0.0) or 0.0) >= 0.5,
        "rankings": float(features.get("ranking_available", 0.0) or 0.0) >= 0.5,
        "analytics": (
            float(features.get("analytics_available", 0.0) or 0.0) >= 0.5
            or float(features.get("analytics_extended_available", 0.0) or 0.0) >= 0.5
        ),
        "context": float(features.get("context_available", 0.0) or 0.0) >= 0.5,
        "announced_lineup": float(features.get("announced_lineup_available", 0.0) or 0.0) >= 0.5,
    }
    sources: list[str] = []
    evidence: dict[str, str] = {}
    for source in ENHANCED_SOURCES:
        captured_at = _source_timestamp(row, source)
        if eligible[source] and _strictly_before(captured_at, kickoff):
            sources.append(source)
            evidence[source] = str(captured_at)
    return EnhancedAvailability(bool(sources), tuple(sources), evidence)


def enhanced_feature_values(row: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, float]:
    """Numeric learner/audit columns for one already assembled feature row."""

    availability = enhanced_availability(row, features)
    values = {ENHANCED_AVAILABLE_COLUMN: float(availability.available)}
    values.update({column: float(source in availability.sources) for source, column in SOURCE_FEATURE_COLUMNS.items()})
    return values


def annotate_enhanced_info(
    feature_rows: Sequence[MutableMapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    metadata: Sequence[MutableMapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Annotate an already warmed-up frame without rebuilding causal state."""

    if len(feature_rows) != len(rows) or (metadata is not None and len(metadata) != len(rows)):
        raise ValueError("feature rows, raw rows and metadata must align")
    source_counts: Counter[str] = Counter()
    enhanced_rows = 0
    for index, (features, row) in enumerate(zip(feature_rows, rows, strict=True)):
        availability = enhanced_availability(row, features)
        features.update(enhanced_feature_values(row, features))
        source_counts.update(availability.sources)
        enhanced_rows += int(availability.available)
        if metadata is not None:
            metadata[index][ENHANCED_AVAILABLE_COLUMN] = availability.available
            metadata[index]["enhanced_sources"] = list(availability.sources)
            metadata[index]["enhanced_source_evidence"] = availability.evidence
    return {
        "total_rows": len(rows),
        "enhanced_rows": enhanced_rows,
        "enhanced_fraction": enhanced_rows / len(rows) if rows else 0.0,
        "source_counts": {source: int(source_counts[source]) for source in ENHANCED_SOURCES},
    }


def training_indices(feature_rows: Sequence[Mapping[str, Any]], policy: str) -> np.ndarray:
    """Select learner rows only; callers must warm up on the full history first."""

    if policy not in {"all", "only_enhanced", "recency_weighted"}:
        raise ValueError(f"unknown enhanced-info training policy: {policy}")
    if policy == "only_enhanced":
        return np.asarray(
            [float(row.get(ENHANCED_AVAILABLE_COLUMN, 0.0) or 0.0) >= 0.5 for row in feature_rows],
            dtype=bool,
        )
    return np.ones(len(feature_rows), dtype=bool)


def recency_decay_weights(periods_subset: Sequence[float] | np.ndarray, half_life_days: float) -> np.ndarray | None:
    """Dixon-Coles weights: ``0.5 ** (age_days / half_life_days)``.

    CS2 periods are weekly, so ``age_days = (max_period - period) * 7``.  The
    floor avoids numerically zero rows while retaining the chronological decay.
    """

    if not half_life_days or half_life_days <= 0:
        return None
    periods = np.asarray(periods_subset, dtype=float)
    if periods.size == 0:
        return None
    ages = (periods.max() - periods) * 7.0
    return np.clip(0.5 ** (ages / float(half_life_days)), 1e-3, 1.0)


def enhanced_bias_report(
    feature_rows: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Coverage by tier, environment and recency, without using outcomes."""

    if len(feature_rows) != len(metadata):
        raise ValueError("feature rows and metadata must align")
    dates = [_iso_date(row.get("date")) for row in metadata]
    reference = max((value for value in dates if value is not None), default=None)

    def recency_bucket(value: datetime | None) -> str:
        if reference is None or value is None:
            return "unknown"
        age = (reference - value).days
        if age <= 90:
            return "0_90d"
        if age <= 365:
            return "91_365d"
        return "older_365d"

    dimensions: dict[str, Counter[tuple[str, bool]]] = {
        "tier": Counter(),
        "environment": Counter(),
        "recency": Counter(),
    }
    for features, meta, date_value in zip(feature_rows, metadata, dates, strict=True):
        enhanced = float(features.get(ENHANCED_AVAILABLE_COLUMN, 0.0) or 0.0) >= 0.5
        labels = {
            "tier": str(meta.get("event_tier") or "unknown").lower(),
            "environment": str(meta.get("environment") or "unknown").lower(),
            "recency": recency_bucket(date_value),
        }
        for dimension, label in labels.items():
            dimensions[dimension][(label, enhanced)] += 1

    output: dict[str, Any] = {}
    warnings: list[str] = []
    for dimension, counts in dimensions.items():
        groups: dict[str, Any] = {}
        group_labels = sorted({label for label, _enhanced in counts})
        fractions: list[float] = []
        for label in group_labels:
            covered = counts[(label, True)]
            total = covered + counts[(label, False)]
            fraction = covered / total if total else 0.0
            fractions.append(fraction)
            groups[label] = {
                "n": int(total),
                "enhanced_n": int(covered),
                "enhanced_fraction": fraction,
            }
        output[dimension] = groups
        if fractions and max(fractions) - min(fractions) >= 0.20:
            warnings.append(
                f"enhanced_available is uneven by {dimension} (max-min={max(fractions) - min(fractions):.1%})"
            )
    output["warnings"] = warnings
    output["interpretation"] = (
        "only_enhanced may specialize toward covered tiers, LAN/online regimes "
        "or recent seasons; its hold-out score must therefore be measured on all "
        "matches, not only covered matches."
    )
    return output


__all__ = [
    "ENHANCED_AVAILABLE_COLUMN",
    "ENHANCED_AUDIT_COLUMNS",
    "ENHANCED_SOURCES",
    "SOURCE_FEATURE_COLUMNS",
    "EnhancedAvailability",
    "annotate_enhanced_info",
    "enhanced_availability",
    "enhanced_feature_values",
    "enhanced_bias_report",
    "recency_decay_weights",
    "training_indices",
]

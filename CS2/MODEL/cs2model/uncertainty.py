"""Heuristic uncertainty around the model's estimated win probability.

This module deliberately does not model the Bernoulli outcome variance
``p * (1 - p)``.  That aleatoric uncertainty is already represented by the
reported win probability.  The band below describes uncertainty about the
estimate of ``p`` itself, using ensemble disagreement and point-in-time data
coverage.  It is an operational diagnostic, not a statistical confidence or
prediction interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

UNCERTAINTY_METHOD = "ensemble_dispersion_history_v1"
FULL_HISTORY_MATCHES = 30
MAX_HISTORY_PENALTY = 0.10
SINGLE_MEMBER_PENALTY = 0.05
MAX_BAND_HALF_WIDTH = 0.25

ConfidenceLevel = Literal["low", "medium", "high"]


def _unit_interval(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return min(1.0, max(0.0, value))


def count_history_strictly_before_day(
    match_dates: Sequence[datetime],
    as_of: datetime | None,
) -> int:
    """Count observations from days strictly before ``as_of``.

    Daily source data cannot establish a safe intraday order, so observations
    from the same calendar day are deliberately excluded.  Future appends are
    therefore unable to change historical coverage.
    """
    if as_of is None:
        return 0
    cutoff = as_of.date()
    return sum(1 for observed_at in match_dates if observed_at.date() < cutoff)


@dataclass(frozen=True)
class EstimateUncertainty:
    """Versioned, cheap diagnostic for uncertainty about an estimated ``p``."""

    ensemble_disagreement: float
    band_half_width: float
    lower: float
    upper: float
    confidence_level: ConfidenceLevel
    history_coverage: float
    effective_members: int
    method: str = UNCERTAINTY_METHOD

    def prediction_fields(self) -> dict[str, float | int | str]:
        """Fields embedded in the prediction snapshot and consumed downstream."""
        return {
            "ensemble_disagreement": round(self.ensemble_disagreement, 6),
            "estimate_band_half_width": round(self.band_half_width, 6),
            "estimate_band_lower_team1": round(self.lower, 6),
            "estimate_band_upper_team1": round(self.upper, 6),
            "estimate_confidence_level": self.confidence_level,
            "estimate_history_coverage": round(self.history_coverage, 6),
            "estimate_effective_members": self.effective_members,
            "estimate_uncertainty_method": self.method,
        }


def estimate_probability_uncertainty(
    probability: float,
    ensemble_disagreement: float,
    *,
    team1_history_matches: int,
    team2_history_matches: int,
    component_weights: Sequence[float],
) -> EstimateUncertainty:
    """Build a symmetric heuristic band around a calibrated probability.

    ``ensemble_disagreement`` is the weighted standard deviation of calibrated
    ensemble member probabilities.  Coverage reaches one when both teams have
    at least ``FULL_HISTORY_MATCHES`` prior matches.  Lower coverage widens the
    band in quadrature; a single effective model gets a separate penalty so a
    zero dispersion cannot be mistaken for certainty.
    """
    probability = _unit_interval(float(probability))
    disagreement = float(ensemble_disagreement)
    if not math.isfinite(disagreement) or disagreement < 0.0:
        raise ValueError("ensemble_disagreement must be finite and non-negative")

    histories = (max(0, int(team1_history_matches)), max(0, int(team2_history_matches)))
    coverage = min(histories) / float(FULL_HISTORY_MATCHES)
    coverage = _unit_interval(coverage)
    effective_members = sum(1 for weight in component_weights if math.isfinite(float(weight)) and float(weight) > 1e-12)

    history_penalty = MAX_HISTORY_PENALTY * (1.0 - coverage)
    member_penalty = SINGLE_MEMBER_PENALTY if effective_members < 2 else 0.0
    half_width = math.sqrt(
        disagreement * disagreement + history_penalty * history_penalty + member_penalty * member_penalty
    )
    half_width = min(MAX_BAND_HALF_WIDTH, half_width)

    if coverage >= 0.80 and effective_members >= 2 and half_width <= 0.05:
        level: ConfidenceLevel = "high"
    elif coverage >= 0.40 and effective_members >= 2 and half_width <= 0.10:
        level = "medium"
    else:
        level = "low"

    return EstimateUncertainty(
        ensemble_disagreement=disagreement,
        band_half_width=half_width,
        lower=max(0.0, probability - half_width),
        upper=min(1.0, probability + half_width),
        confidence_level=level,
        history_coverage=coverage,
        effective_members=effective_members,
    )

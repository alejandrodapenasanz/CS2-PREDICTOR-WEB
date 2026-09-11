"""Immutable domain models used by the CS2 Telegram publisher."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class PredictionPick:
    """Represent one validated CS2 prediction selected for publication."""

    match_id: str
    match_date: date
    hltv_url: str
    hour: str
    team1: str
    team2: str
    winner: str
    win_probability: float
    is_best_opportunity: bool
    confidence_level: str | None = None

    @property
    def confidence_label(self) -> str:
        """Display the exported estimate level, never infer it from probability."""

        return self.confidence_level.upper() if self.confidence_level else "NOT AVAILABLE"


@dataclass(frozen=True, slots=True)
class DailyPredictions:
    """Contain all qualified opportunities and remaining picks for one date."""

    target_date: date
    source_run_id: str
    opportunities: tuple[PredictionPick, ...]
    others: tuple[PredictionPick, ...]

    @property
    def best(self) -> PredictionPick | None:
        """Return the top opportunity for backwards-compatible callers."""

        return self.opportunities[0] if self.opportunities else None

    @property
    def total(self) -> int:
        """Return the total number of predictions selected for the date."""

        return len(self.others) + len(self.opportunities)

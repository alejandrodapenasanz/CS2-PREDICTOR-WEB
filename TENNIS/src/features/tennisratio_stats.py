"""Candidate TennisRatio service/return features with strict as-of cutoffs.

These values are intentionally kept outside ``MODEL_FEATURE_COLUMNS`` until
forward-collected coverage is sufficient and a temporal ablation passes the
normal challenger promotion gate.  Historical source dates alone do not make
a value historically available: both ``effective_date`` and ``available_date``
must be strictly earlier than the predicted match date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Iterable

from ..tennisratio.types import Gender, MappedMatchStats


class TennisRatioStatsFeatureError(ValueError):
    """Raised when candidate stat features would violate their causal contract."""


@dataclass(frozen=True, slots=True)
class TennisRatioStatsSnapshot:
    """Recent per-player service/return aggregates available before one date."""

    gender: Gender
    player_id: int
    as_of_date: date
    surface: str | None
    matches: int
    coverage: float
    first_serve_accuracy_pct: float | None
    first_serve_points_won_pct: float | None
    second_serve_points_won_pct: float | None
    aces_per_service_game: float | None
    double_faults_per_service_game: float | None
    breakpoints_saved_pct: float | None
    service_games_won_pct: float | None
    return_first_serve_points_won_pct: float | None
    return_second_serve_points_won_pct: float | None
    breakpoints_converted_pct: float | None
    return_games_won_pct: float | None
    serve_pressure_points_won_pct: float | None
    return_pressure_points_won_pct: float | None

    def candidate_model_values(self, *, prefix: str = "tr") -> dict[str, float | int | None]:
        """Return a stable candidate namespace for a future temporal ablation."""

        return {
            f"{prefix}_stats_matches": self.matches,
            f"{prefix}_stats_coverage": self.coverage,
            f"{prefix}_first_serve_accuracy_pct": self.first_serve_accuracy_pct,
            f"{prefix}_first_serve_points_won_pct": self.first_serve_points_won_pct,
            f"{prefix}_second_serve_points_won_pct": self.second_serve_points_won_pct,
            f"{prefix}_aces_per_service_game": self.aces_per_service_game,
            f"{prefix}_double_faults_per_service_game": self.double_faults_per_service_game,
            f"{prefix}_breakpoints_saved_pct": self.breakpoints_saved_pct,
            f"{prefix}_service_games_won_pct": self.service_games_won_pct,
            f"{prefix}_return_first_serve_points_won_pct": (self.return_first_serve_points_won_pct),
            f"{prefix}_return_second_serve_points_won_pct": (
                self.return_second_serve_points_won_pct
            ),
            f"{prefix}_breakpoints_converted_pct": self.breakpoints_converted_pct,
            f"{prefix}_return_games_won_pct": self.return_games_won_pct,
            f"{prefix}_serve_pressure_points_won_pct": self.serve_pressure_points_won_pct,
            f"{prefix}_return_pressure_points_won_pct": self.return_pressure_points_won_pct,
        }


def build_tennisratio_stats_snapshot(
    observations: Iterable[MappedMatchStats],
    *,
    gender: Gender,
    player_id: int,
    as_of_date: date,
    surface: str | None = None,
    recent_matches: int = 20,
) -> TennisRatioStatsSnapshot:
    """Aggregate the newest eligible observations without same-day information."""

    if gender not in {"M", "F"}:
        raise TennisRatioStatsFeatureError("gender must be M or F.")
    if isinstance(player_id, bool) or not isinstance(player_id, int) or player_id <= 0:
        raise TennisRatioStatsFeatureError("player_id must be a positive integer.")
    if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
        raise TennisRatioStatsFeatureError("as_of_date must be a strict date.")
    if isinstance(recent_matches, bool) or recent_matches <= 0:
        raise TennisRatioStatsFeatureError("recent_matches must be positive.")

    eligible: list[MappedMatchStats] = []
    for item in observations:
        if item.gender != gender or item.sackmann_player_id != player_id:
            continue
        if item.effective_date >= as_of_date or item.available_date >= as_of_date:
            continue
        if surface is not None and item.surface != surface:
            continue
        eligible.append(item)
    selected = sorted(
        eligible,
        key=lambda item: (item.effective_date, item.first_seen_at_utc, item.source_sha256),
        reverse=True,
    )[:recent_matches]
    stats = [item.stats for item in selected]
    total_fields = len(stats) * 19
    coverage = (
        0.0 if total_fields == 0 else sum(item.available_fields for item in stats) / total_fields
    )

    return TennisRatioStatsSnapshot(
        gender=gender,
        player_id=player_id,
        as_of_date=as_of_date,
        surface=surface,
        matches=len(stats),
        coverage=coverage,
        first_serve_accuracy_pct=_mean(item.first_serve_accuracy_pct for item in stats),
        first_serve_points_won_pct=_mean(item.first_serve_points_won_pct for item in stats),
        second_serve_points_won_pct=_mean(item.second_serve_points_won_pct for item in stats),
        aces_per_service_game=_paired_rate(
            ((item.aces, item.service_games_played) for item in stats)
        ),
        double_faults_per_service_game=_paired_rate(
            ((item.double_faults, item.service_games_played) for item in stats)
        ),
        breakpoints_saved_pct=_paired_rate(
            ((item.breakpoints_saved, item.breakpoints_faced) for item in stats),
            percentage=True,
        ),
        service_games_won_pct=_paired_rate(
            ((item.service_games_won, item.service_games_played) for item in stats),
            percentage=True,
        ),
        return_first_serve_points_won_pct=_mean(
            item.return_first_serve_points_won_pct for item in stats
        ),
        return_second_serve_points_won_pct=_mean(
            item.return_second_serve_points_won_pct for item in stats
        ),
        breakpoints_converted_pct=_paired_rate(
            ((item.breakpoints_converted, item.breakpoint_opportunities) for item in stats),
            percentage=True,
        ),
        return_games_won_pct=_paired_rate(
            ((item.return_games_won, item.return_games_played) for item in stats),
            percentage=True,
        ),
        serve_pressure_points_won_pct=_paired_rate(
            ((item.serve_pressure_points_won, item.serve_pressure_points) for item in stats),
            percentage=True,
        ),
        return_pressure_points_won_pct=_paired_rate(
            ((item.return_pressure_points_won, item.return_pressure_points) for item in stats),
            percentage=True,
        ),
    )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else math.fsum(present) / len(present)


def _paired_rate(
    values: Iterable[tuple[int | None, int | None]],
    *,
    percentage: bool = False,
) -> float | None:
    numerator_total = 0
    denominator_total = 0
    for numerator, denominator in values:
        if numerator is None or denominator is None:
            continue
        numerator_total += numerator
        denominator_total += denominator
    if denominator_total <= 0:
        return None
    result = numerator_total / denominator_total
    return result * 100.0 if percentage else result

"""Anti-leak tests for candidate TennisRatio service/return aggregates."""

from __future__ import annotations

from datetime import UTC, date, datetime

from src.features.tennisratio_stats import build_tennisratio_stats_snapshot
from src.tennisratio.types import MappedMatchStats, ProfileMatchStats


def _observation(
    *,
    effective: date,
    available: date,
    first_serve: float,
    sha: str,
) -> MappedMatchStats:
    stats = ProfileMatchStats(
        first_serve_accuracy_pct=first_serve,
        first_serve_points_won_pct=70.0,
        second_serve_points_won_pct=50.0,
        aces=5,
        double_faults=2,
        breakpoints_saved=4,
        breakpoints_faced=5,
        service_games_won=9,
        service_games_played=10,
        return_first_serve_points_won_pct=30.0,
        return_second_serve_points_won_pct=55.0,
        breakpoints_converted=3,
        breakpoint_opportunities=6,
        return_games_won=3,
        return_games_played=10,
        serve_pressure_points=12,
        serve_pressure_points_won=8,
        return_pressure_points=10,
        return_pressure_points_won=4,
    )
    return MappedMatchStats(
        gender="M",
        sackmann_player_id=7,
        source_player_key="tennisratio:PlayerSeven",
        effective_date=effective,
        available_date=available,
        first_seen_at_utc=datetime.combine(available, datetime.min.time(), tzinfo=UTC),
        surface="Hard",
        tour_level="ATP",
        stats=stats,
        source_url="https://www.tennisratio.com/players/PlayerSeven.html",
        source_sha256=sha,
    )


def test_stats_snapshot_is_strictly_as_of_and_append_future_invariant() -> None:
    """Same-day/unavailable rows and appended future rows cannot alter D."""

    base = [
        _observation(
            effective=date(2026, 8, 20),
            available=date(2026, 8, 21),
            first_serve=60.0,
            sha="a" * 64,
        ),
        _observation(
            effective=date(2026, 8, 22),
            available=date(2026, 8, 23),
            first_serve=80.0,
            sha="b" * 64,
        ),
    ]
    expected = build_tennisratio_stats_snapshot(
        base,
        gender="M",
        player_id=7,
        as_of_date=date(2026, 8, 23),
        surface="Hard",
    )
    assert expected.matches == 1
    assert expected.first_serve_accuracy_pct == 60.0
    assert expected.service_games_won_pct == 90.0

    appended = [
        *base,
        _observation(
            effective=date(2026, 8, 24),
            available=date(2026, 8, 25),
            first_serve=10.0,
            sha="c" * 64,
        ),
    ]
    assert (
        build_tennisratio_stats_snapshot(
            appended,
            gender="M",
            player_id=7,
            as_of_date=date(2026, 8, 23),
            surface="Hard",
        )
        == expected
    )

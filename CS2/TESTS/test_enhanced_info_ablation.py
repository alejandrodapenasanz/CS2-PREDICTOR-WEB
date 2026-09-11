from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.enhanced_info import (  # noqa: E402
    ENHANCED_AVAILABLE_COLUMN,
    annotate_enhanced_info,
    enhanced_availability,
    recency_decay_weights,
    training_indices,
)
from cs2model.features import build_training_frame  # noqa: E402


@pytest.mark.parametrize(
    ("source", "row_patch", "feature_patch"),
    [
        (
            "opening_odds",
            {
                "opening_odds_decimal_t1": 1.8,
                "opening_odds_decimal_t2": 2.1,
                "opening_bookmaker_count": 1,
                "opening_odds_captured_at": "2026-08-25T10:00:00Z",
            },
            {},
        ),
        (
            "player_snapshots",
            {"player_snapshot_features": {"captured_at_max": "2026-08-25T10:00:00Z"}},
            {"player_snapshot_available": 1.0},
        ),
        (
            "rankings",
            {"ranking_snapshot_evidence": {"captured_at_max": "2026-08-25T10:00:00Z"}},
            {"ranking_available": 1.0},
        ),
        (
            "analytics",
            {
                "analytics": {
                    "available": True,
                    "captured_at": "2026-08-25T10:00:00Z",
                }
            },
            {"analytics_available": 1.0},
        ),
        (
            "context",
            {"context_captured_at_utc": "2026-08-25T10:00:00Z"},
            {"context_available": 1.0},
        ),
        (
            "announced_lineup",
            {"prematch_lineups": {"captured_at": "2026-08-25T10:00:00Z"}},
            {"announced_lineup_available": 1.0},
        ),
    ],
)
def test_each_source_requires_strict_prematch_evidence(
    source: str,
    row_patch: dict[str, object],
    feature_patch: dict[str, float],
) -> None:
    row = {"kickoff_utc": "2026-08-26T12:00:00Z", **row_patch}

    availability = enhanced_availability(row, feature_patch)

    assert availability.available
    assert availability.sources == (source,)
    assert source in availability.evidence


def test_equal_or_postmatch_timestamp_and_box_score_do_not_count() -> None:
    row = {
        "kickoff_utc": "2026-08-26T12:00:00Z",
        "context_captured_at_utc": "2026-08-26T12:00:00Z",
        "asset": {"mapstats": [{"winner": "alpha"}]},
    }

    availability = enhanced_availability(row, {"context_available": 1.0})

    assert not availability.available
    assert availability.sources == ()


def _match(
    index: int,
    team1: str,
    team2: str,
    team1_win: bool,
    *,
    enhanced: bool = False,
) -> dict[str, object]:
    played_at = datetime(2026, 1, 1) + timedelta(days=index)
    return {
        "id": str(index),
        "date": played_at.date().isoformat(),
        "date_obj": played_at,
        "kickoff_utc": played_at.isoformat() + "Z",
        "datetime_precision": "exact",
        "event": "test",
        "event_tier": "a",
        "format": "bo3",
        "team1": team1,
        "team2": team2,
        "team1_key": team1,
        "team2_key": team2,
        "score1": 2 if team1_win else 0,
        "score2": 0 if team1_win else 2,
        "team1_win": int(team1_win),
        "match_context": {"environment": "lan"} if enhanced else {},
        "context_captured_at_utc": ((played_at - timedelta(hours=1)).isoformat() + "Z" if enhanced else None),
    }


def test_filtering_training_rows_preserves_full_history_warmup_features() -> None:
    rows = [
        _match(0, "alpha", "beta", True),
        _match(1, "alpha", "gamma", True),
        _match(2, "alpha", "delta", True, enhanced=True),
    ]
    full_features, _labels, metadata, _state = build_training_frame(rows)
    before = dict(full_features[-1])

    annotate_enhanced_info(full_features, rows, metadata)
    mask = training_indices(full_features, "only_enhanced")
    selected = [row for row, keep in zip(full_features, mask, strict=True) if keep]

    assert mask.tolist() == [False, False, True]
    assert {key: selected[0][key] for key in before} == before
    filtered_features, *_rest = build_training_frame([rows[-1]])
    assert selected[0]["elo_diff"] != filtered_features[0]["elo_diff"]
    assert selected[0][ENHANCED_AVAILABLE_COLUMN] == 1.0


def test_recency_weight_formula_is_documented_and_deterministic() -> None:
    weights = recency_decay_weights([10, 11, 12], 14.0)

    assert weights is not None
    assert weights.tolist() == pytest.approx([0.5, 2**-0.5, 1.0])

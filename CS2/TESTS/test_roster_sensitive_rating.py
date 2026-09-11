from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

import train
from cs2model.features import ChronologicalState, _period_index, build_training_frame
from cs2model.promotion import compare_candidate_to_incumbent


def lineup(*player_ids: str) -> dict[str, Any]:
    return {"players": [{"hltv_player_id": player_id, "nickname": player_id} for player_id in player_ids]}


ALPHA_OLD = lineup("a1", "a2", "a3", "a4", "a5")
ALPHA_NEW = lineup("a1", "a2", "a3", "a6", "a7")
BETA = lineup("b1", "b2", "b3", "b4", "b5")


def match(
    match_id: str,
    date_obj: datetime,
    *,
    alpha_lineup: dict[str, Any] = ALPHA_OLD,
    alpha_wins: bool = True,
) -> dict[str, Any]:
    return {
        "id": match_id,
        "date": date_obj.strftime("%Y-%m-%d"),
        "date_obj": date_obj,
        "datetime_precision": "exact",
        "event": "Roster Cup",
        "format": "bo3",
        "team1": "Alpha",
        "team2": "Beta",
        "team1_key": "alpha",
        "team2_key": "beta",
        "score1": 2 if alpha_wins else 0,
        "score2": 0 if alpha_wins else 2,
        "team1_win": int(alpha_wins),
        "prematch_lineups": {"team1": alpha_lineup, "team2": BETA},
        "actual_lineups": {"team1": alpha_lineup, "team2": BETA},
    }


def matured_state() -> tuple[ChronologicalState, datetime]:
    state = ChronologicalState()
    start = datetime(2025, 1, 1, 12)
    for index in range(12):
        state.observe(match(f"history-{index}", start + timedelta(days=8 * index)))
    as_of = start + timedelta(days=8 * 12)
    return state, as_of


def test_core_change_discount_is_proportional_and_inflates_rd() -> None:
    state, as_of = matured_state()
    period = _period_index(as_of)
    state._advance_to(period)
    before = state.roster_rating_asof("alpha", period)
    pending = state.roster_period_buffer.get("alpha") or []
    pre_discount = state.glicko.update(before, list(pending), period) if pending else before

    features = state.emit_features("alpha", "beta", as_of, "Roster Cup", "bo3", ALPHA_NEW, BETA)

    expected_fraction = 0.20 + 0.80 * (3 / 5)
    expected_rating = 1500.0 + expected_fraction * (pre_discount.rating - 1500.0)
    expected_rd = math.sqrt(expected_fraction * pre_discount.rd**2 + (1.0 - expected_fraction) * 350.0**2)
    assert features["roster_core_change_team1"] == 1.0
    assert features["roster_replacements_team1"] == 2.0
    assert features["roster_retained_players_team1"] == 3.0
    assert features["roster_credit_fraction_team1"] == expected_fraction
    assert features["roster_glicko_rating_team1"] == expected_rating
    assert features["roster_glicko_rd_team1"] == expected_rd
    assert abs(expected_rating - 1500.0) < abs(pre_discount.rating - 1500.0)
    assert expected_rd > pre_discount.rd


def test_adjusted_rating_recalibrates_with_new_results_without_repeated_decay() -> None:
    state, as_of = matured_state()
    changed = state.emit_features("alpha", "beta", as_of, "Roster Cup", "bo3", ALPHA_NEW, BETA)
    adjusted_rating = changed["roster_glicko_rating_team1"]
    adjusted_rd = changed["roster_glicko_rd_team1"]
    state.observe(match("rebuild", as_of, alpha_lineup=ALPHA_NEW))

    for index in range(1, 9):
        state.observe(
            match(
                f"new-core-{index}",
                as_of + timedelta(days=8 * index),
                alpha_lineup=ALPHA_NEW,
            )
        )
    future_at = as_of + timedelta(days=8 * 10)
    future = state.emit_features("alpha", "beta", future_at, "Roster Cup", "bo3", ALPHA_NEW, BETA)

    assert future["roster_core_change_team1"] == 0.0
    assert future["roster_credit_fraction_team1"] == 1.0
    assert future["roster_glicko_rating_team1"] > adjusted_rating
    assert future["roster_glicko_rd_team1"] < adjusted_rd


def test_roster_rating_asof_is_invariant_when_future_matches_are_appended() -> None:
    start = datetime(2025, 1, 1, 12)
    rows = [match(f"past-{index}", start + timedelta(days=8 * index)) for index in range(6)]
    target = match("target", start + timedelta(days=8 * 6), alpha_lineup=ALPHA_NEW)
    future = [
        match(
            f"future-{index}",
            start + timedelta(days=8 * (7 + index)),
            alpha_lineup=ALPHA_NEW,
            alpha_wins=bool(index % 2),
        )
        for index in range(5)
    ]

    prefix_features, *_ = build_training_frame([*rows, target])
    full_features, *_ = build_training_frame([*rows, target, *future])
    for column in train.ROSTER_RATING_FEATURE_COLUMNS:
        assert full_features[len(rows)][column] == prefix_features[len(rows)][column]


class FeatureProbabilityModel:
    metadata = {"date_max": "2026-01-01"}

    def predict_proba_team1(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray([row["incumbent_probability"] for row in rows], dtype=float)


def gate_decision(incumbent_probability: float, challenger_probability: float):
    labels = [1, 0, 1, 0]
    challengers = []
    features = []
    for index, actual in enumerate(labels, start=2):
        probability = challenger_probability if actual else 1.0 - challenger_probability
        incumbent = incumbent_probability if actual else 1.0 - incumbent_probability
        identity = {
            "match_id": f"roster-{index}",
            "date": f"2026-01-{index:02d}",
            "actual": actual,
        }
        challengers.append({**identity, "prob_team1": probability})
        features.append({**identity, "features": {"incumbent_probability": incumbent}})
    return compare_candidate_to_incumbent(FeatureProbabilityModel(), challengers, features, min_samples=4)


def test_roster_candidate_is_wired_through_gate_and_only_improvement_promotes() -> None:
    assert train.RATING_CANDIDATE_COLUMNS["roster_glicko_cal"] == "roster_glicko_prob_centered"
    assert gate_decision(0.70, 0.80).promote is True
    assert gate_decision(0.80, 0.60).promote is False

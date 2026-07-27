"""Leakage-safe compositional BO3 probability from pre-match map analytics."""

from __future__ import annotations

import math
from typing import Any


MAP_POOL_MIN_ROWS = 500
MAP_POOL_MIN_MAP_SAMPLE = 10


def compositional_series_probability(p1: float, p2: float, p3: float) -> float:
    """P(A wins a BO3) for independent map probabilities in play order."""
    values = [min(max(float(value), 1e-6), 1.0 - 1e-6) for value in (p1, p2, p3)]
    p1, p2, p3 = values
    return p1 * p2 + p1 * (1.0 - p2) * p3 + (1.0 - p1) * p2 * p3


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _shrunk_winrate(win_pct: Any, played: Any, prior_maps: float = 10.0) -> tuple[float, int]:
    sample = _safe_int(played)
    rate = _safe_float(win_pct)
    if rate is None or sample <= 0:
        return 0.5, 0
    observed = min(max(rate / 100.0, 0.0), 1.0)
    return (observed * sample + 0.5 * prior_maps) / (sample + prior_maps), sample


def _map_probability(team1_row: dict[str, Any], team2_row: dict[str, Any]) -> tuple[float, int]:
    wr1, n1 = _shrunk_winrate(team1_row.get("win_pct"), team1_row.get("played"))
    wr2, n2 = _shrunk_winrate(team2_row.get("win_pct"), team2_row.get("played"))
    logit1 = math.log(min(max(wr1, 1e-6), 1.0 - 1e-6) / min(max(1.0 - wr1, 1e-6), 1.0))
    logit2 = math.log(min(max(wr2, 1e-6), 1.0 - 1e-6) / min(max(1.0 - wr2, 1e-6), 1.0))
    probability = 1.0 / (1.0 + math.exp(-(logit1 - logit2)))
    return probability, min(n1, n2)


def compositional_bo3_features(
    analytics: dict[str, Any] | None,
    team1: str,
    team2: str,
    fmt: str,
) -> dict[str, float]:
    """Estimate pick1, pick2 and decider without using a post-match veto."""
    unavailable = {
        "bo3_compositional_available": 0.0,
        "bo3_compositional_prob_centered": 0.0,
        "bo3_map_probability_spread": 0.0,
        "bo3_map_sample_min": 0.0,
        "bo3_veto_confidence": 0.0,
    }
    if str(fmt or "").lower() != "bo3" or not analytics:
        return unavailable
    t1, t2 = _clean(team1), _clean(team2)
    by_map: dict[str, dict[str, dict[str, Any]]] = {}
    for row in analytics.get("map_stats") or []:
        name = _clean(row.get("map"))
        team = _clean(row.get("team"))
        if not name or team not in {t1, t2}:
            continue
        by_map.setdefault(name, {})[team] = row
    complete = {name: sides for name, sides in by_map.items() if t1 in sides and t2 in sides}
    if len(complete) < 3:
        return unavailable

    def pick_score(name: str, team: str) -> float:
        row = complete[name][team]
        pick = _safe_float(row.get("first_pick_pct")) or 0.0
        ban = _safe_float(row.get("first_ban_pct")) or 0.0
        sample = _safe_int(row.get("played"))
        return pick - 0.75 * ban + 0.05 * math.log1p(sample)

    first = max(complete, key=lambda name: (pick_score(name, t1), name))
    remaining = [name for name in complete if name != first]
    second = max(remaining, key=lambda name: (pick_score(name, t2), name))
    remaining = [name for name in remaining if name != second]

    def decider_score(name: str) -> float:
        rows = complete[name]
        sample = _safe_int(rows[t1].get("played")) + _safe_int(rows[t2].get("played"))
        bans = (_safe_float(rows[t1].get("first_ban_pct")) or 0.0) + (_safe_float(rows[t2].get("first_ban_pct")) or 0.0)
        return math.log1p(sample) - 0.02 * bans

    third = max(remaining, key=lambda name: (decider_score(name), name))
    map_results = [_map_probability(complete[name][t1], complete[name][t2]) for name in (first, second, third)]
    probabilities = [item[0] for item in map_results]
    sample_min = min(item[1] for item in map_results)
    confidence = min(1.0, sample_min / 30.0)
    if sample_min < MAP_POOL_MIN_MAP_SAMPLE:
        return {**unavailable, "bo3_map_sample_min": float(sample_min), "bo3_veto_confidence": confidence}
    series_probability = compositional_series_probability(*probabilities)
    return {
        "bo3_compositional_available": 1.0,
        "bo3_compositional_prob_centered": series_probability - 0.5,
        "bo3_map_probability_spread": max(probabilities) - min(probabilities),
        "bo3_map_sample_min": float(sample_min),
        "bo3_veto_confidence": confidence,
    }


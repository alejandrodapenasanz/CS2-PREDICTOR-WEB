"""Causal pistol/conversion features from independently timestamped map captures.

Only facts played AND captured before civil day D qualify. A 90-day window,
20-observation neutral prior and sample flags are fixed before evaluation.
No current-match map/veto/starting side or future roster information is used.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import sqlite3
from typing import Any

WINDOW_DAYS = 90
PRIOR_N = 20.0
MIN_PISTOLS = 10
RATE_NAMES = ("win", "ct_win", "t_win", "convert2", "convert3", "sweep3", "break2", "recover3")
PISTOL_DIFF_COLUMNS = [f"pistol_{name}_90_diff" for name in RATE_NAMES] + ["pistol_sample_log_diff"]
PISTOL_COLUMNS = PISTOL_DIFF_COLUMNS + ["pistol_available", "pistol_sample_min", "pistol_sample_total"]
MINIMAL_COLUMNS = ["pistol_win_90_diff", "pistol_available", "pistol_sample_min"]
# Keep the column contract usable by ingestion without importing ML runtimes.
OPPONENT_DIFF_COLUMNS = [
    "pistol_opponent_edge_diff",
    "pistol_opponent_probability_centered",
    "pistol_opponent_baseline_centered",
    "pistol_opponent_residual_diff",
]


def civil_date(value: str | datetime | date) -> date:
    """Use the same civil match day as the core feature builder, never intraday ordering."""
    return (
        value.date()
        if isinstance(value, datetime)
        else (value if isinstance(value, date) else date.fromisoformat(value[:10]))
    )


def load_pistol_store(connection: sqlite3.Connection) -> dict[str, Any]:
    """Read first observed valid map capture; duplicated source captures do not add samples."""
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='map_round_sources' AND type='table'").fetchone():
        return {"teams": {}, "maps": [], "fingerprint": "absent"}
    sources = connection.execute(
        "SELECT s.round_source_id,s.map_id,s.captured_at_utc,s.played_at_utc,s.round_format,"
        "s.map_name,s.team_left_hltv_id,s.team_right_hltv_id,s.sha256,s.parser_version,m.hltv_match_id "
        "FROM map_round_sources s JOIN maps p ON p.map_id=s.map_id "
        "JOIN matches m ON m.match_id=p.match_id ORDER BY s.captured_at_utc,s.round_source_id"
    ).fetchall()
    by_source: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for row in connection.execute(
        "SELECT round_source_id,round_number,half,round_in_half,overtime,is_pistol,winner_hltv_id,team_left_side "
        "FROM map_rounds ORDER BY round_source_id,round_number"
    ):
        by_source[int(row[0])].append(tuple(row))
    seen: set[int] = set()
    teams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    map_records: list[dict[str, Any]] = []
    fingerprints: list[tuple[Any, ...]] = []
    for source in sources:
        sid, mid, captured, played, fmt, map_name, left, right, sha, version, match_id = tuple(source)
        if int(mid) in seen or fmt != "mr12" or civil_date(played) < date(2023, 9, 27):
            continue
        seen.add(int(mid))
        rounds = by_source[int(sid)]
        fingerprints.append((mid, captured, played, sha, version))
        map_records.append({"map_id": mid, "captured_at": captured, "played_at": played, "map_name": map_name})
        for row in rounds:
            if not row[5]:
                continue
            followers = {int(r[3]): r for r in rounds if r[2] == row[2] and not r[4]}
            for team, rival, side in (
                (str(left), str(right), row[7]),
                (str(right), str(left), "t" if row[7] == "ct" else "ct"),
            ):
                won = str(row[6]) == team
                second = float(str(followers[2][6]) == team) if 2 in followers else None
                third = float(str(followers[3][6]) == team) if 3 in followers else None
                teams[team].append(
                    {
                        "map_id": mid,
                        "match_id": str(match_id),
                        "half": int(row[2]),
                        "map_name": map_name,
                        "played_at": played,
                        "captured_at": captured,
                        "side": side,
                        "opponent": rival,
                        "win": float(won),
                        "ct_win": float(won) if side == "ct" else None,
                        "t_win": float(won) if side == "t" else None,
                        "convert2": second if won else None,
                        "convert3": third if won else None,
                        "sweep3": second * third if won and second is not None and third is not None else None,
                        "break2": second if not won else None,
                        "recover3": third if not won else None,
                        "rounds_first3": float(won) + second + third
                        if second is not None and third is not None
                        else None,
                    }
                )
    return {
        "teams": dict(teams),
        "maps": map_records,
        "fingerprint": hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
    }


def team_summary(
    store: dict[str, Any], team: str, as_of: str | date | datetime, *, map_name: str | None = None
) -> dict[str, Any]:
    """Return empirical rates, denominators and shrunk estimates from eligible past captures."""
    day = civil_date(as_of)
    floor = day - timedelta(days=WINDOW_DAYS)
    eligible = [
        r
        for r in store.get("teams", {}).get(str(team), [])
        if floor <= civil_date(r["played_at"]) < day
        and civil_date(r["captured_at"]) < day
        and (map_name is None or r["map_name"] == map_name)
    ]
    summary: dict[str, Any] = {"n": len(eligible), "rates": {}, "smoothed": {}, "denominators": {}}
    for name in RATE_NAMES:
        values = [float(r[name]) for r in eligible if r[name] is not None]
        n = len(values)
        summary["denominators"][name] = n
        summary["rates"][name] = sum(values) / n if n else None
        summary["smoothed"][name] = (sum(values) + PRIOR_N * 0.5) / (n + PRIOR_N)
    opening_sequences = [r["rounds_first3"] for r in eligible if r["win"] and r["rounds_first3"] is not None]
    summary["mean_first3_rounds_when_pistol_won"] = (
        sum(opening_sequences) / len(opening_sequences) if opening_sequences else None
    )
    summary["first3_complete_n"] = len(opening_sequences)
    return summary


def pistol_features_asof(store: dict[str, Any], team1: Any, team2: Any, as_of: Any) -> dict[str, float]:
    """Emit antisymmetric differences and explicit sample coverage; never an outcome of D."""
    if not as_of:
        return {column: 0.0 for column in PISTOL_COLUMNS}
    a, b = (team_summary(store, str(team), as_of) for team in (team1, team2))
    available = min(a["n"], b["n"]) >= MIN_PISTOLS
    result = {
        f"pistol_{name}_90_diff": a["smoothed"][name] - b["smoothed"][name] if available else 0.0 for name in RATE_NAMES
    }
    result.update(
        pistol_sample_log_diff=math.log1p(a["n"]) - math.log1p(b["n"]),
        pistol_available=float(available),
        pistol_sample_min=float(min(a["n"], b["n"])),
        pistol_sample_total=float(a["n"] + b["n"]),
    )
    return result

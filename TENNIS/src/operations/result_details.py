"""Structured, monotone result details outside the sacred operational triplet.

These tables are settlement/persistence data, never a feature source. Historical
observations remain unchanged; an aggregate score cannot erase detailed sets.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import re
import sqlite3
from typing import Any, Mapping

from .identifiers import optional_scalar
from .validation import optional_integer
from .types import OperationsValidationError


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS result_details (
    source_match_id TEXT PRIMARY KEY REFERENCES matches(source_match_id),
    status TEXT NOT NULL,
    player_1_slug TEXT NOT NULL, player_2_slug TEXT NOT NULL,
    winner_slug TEXT,
    player_1_sets_won INTEGER, player_2_sets_won INTEGER,
    detail_status TEXT NOT NULL,
    complete INTEGER NOT NULL,
    event_date TEXT,
    obtained_at_utc TEXT NOT NULL,
    source_system TEXT NOT NULL,
    source_url TEXT,
    observation_id TEXT NOT NULL REFERENCES observations(observation_id)
);
CREATE TABLE IF NOT EXISTS result_sets (
    source_match_id TEXT NOT NULL REFERENCES matches(source_match_id),
    set_number INTEGER NOT NULL,
    player_1_games INTEGER NOT NULL, player_2_games INTEGER NOT NULL,
    tiebreak_player_1 INTEGER, tiebreak_player_2 INTEGER,
    score_kind TEXT NOT NULL,
    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    PRIMARY KEY(source_match_id,set_number)
);
CREATE TABLE IF NOT EXISTS result_recovery (
    source_match_id TEXT PRIMARY KEY REFERENCES matches(source_match_id),
    last_attempt_at_utc TEXT NOT NULL,
    next_attempt_at_utc TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS result_detail_conflicts (
    observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id),
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS result_detail_processed (
    observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id)
);
CREATE VIEW IF NOT EXISTS result_details_v1 AS
SELECT d.*, CASE WHEN winner_slug=player_1_slug THEN player_2_slug
                WHEN winner_slug=player_2_slug THEN player_1_slug END AS loser_slug
FROM result_details d;
CREATE VIEW IF NOT EXISTS result_settlement_v1 AS
SELECT s.source_match_id, s.official_prediction_id,
       CASE WHEN p.model_probability_a >= .5 THEN p.player_a_slug ELSE p.player_b_slug END AS predicted_winner,
       s.winner_slug AS actual_winner,
       CASE WHEN (p.model_probability_a >= .5) = s.actual_outcome_a THEN 1 ELSE 0 END AS prediction_correct,
       p.model_probability_a AS predicted_probability_a,
       p.model_fingerprint AS model_version,
       p.prediction_as_of_utc AS prediction_timestamp,
       o.sets_score AS final_result,
       s.settled_at_utc AS settlement_timestamp,
       p.market_probability_a, p.market_probability_b,
       CASE WHEN p.market_probability_a > p.market_probability_b THEN p.player_a_slug
            WHEN p.market_probability_b > p.market_probability_a THEN p.player_b_slug END AS market_favorite,
       CASE WHEN p.market_probability_a > p.market_probability_b THEN p.player_b_slug
            WHEN p.market_probability_b > p.market_probability_a THEN p.player_a_slug END AS market_underdog,
       p.payload_json AS frozen_prediction_json,
       CASE WHEN p.market_probability_a > p.market_probability_b THEN 1-s.actual_outcome_a
            WHEN p.market_probability_b > p.market_probability_a THEN s.actual_outcome_a END AS market_upset
FROM settlements s JOIN predictions p ON p.prediction_id=s.official_prediction_id
JOIN observations o ON o.observation_id=s.observation_id;
"""

TOKEN = re.compile(r"(?P<bracket>\[)?(?P<a>\d{1,2})[-–:](?P<b>\d{1,2})(?:\((?P<tb>\d{1,2})\))?\]?")


def parse_sets(score: str, *, reverse: bool = False) -> list[dict[str, Any]]:
    """Parse explicit set scores; never turn a 2-1 aggregate into three sets.

    Single parenthesized tie-break notation supplies only the loser's points. The
    winner's points stay unknown. Bracketed super tie-breaks are points, not games.
    """
    tokens = list(TOKEN.finditer(score))
    if len(tokens) == 1 and max(int(tokens[0]["a"]), int(tokens[0]["b"])) < 4:
        return []
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(tokens, 1):
        a, b = int(token["a"]), int(token["b"])
        if reverse:
            a, b = b, a
        tb = int(token["tb"]) if token["tb"] is not None else None
        rows.append(
            {
                "set_number": index,
                "player_1_games": a,
                "player_2_games": b,
                "tiebreak_player_1": tb if a < b else None,
                "tiebreak_player_2": tb if b < a else None,
                "score_kind": "match_tiebreak_points" if token["bracket"] else "games",
            }
        )
    return rows


def result_status(row: Mapping[str, Any]) -> str:
    """Preserve exceptional outcomes rather than calling every winner finished."""
    status = str(row.get("status") or "unknown").lower()
    score = str(optional_scalar(row.get("sets_score")) or "").upper()
    if re.search(r"\b(W/O|W\.O\.?|WALKOVER)\b", score) or status in {"wo", "walkover"}:
        return "walkover"
    if re.search(r"\b(RET|RETIRED|RETIREMENT)\b", score) or status in {"retired", "retirement"}:
        return "retirement"
    return {"completed": "finished", "canceled": "cancelled", "in_progress": "live"}.get(
        status, status
    )


def merge_observation(
    conn: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    observation_id: str,
    source: str,
    obtained_at: str,
) -> dict[str, int]:
    """Merge one registered observation in the caller's transaction.

    The source observation is never modified, even on contradiction. Callers use a
    savepoint and record conflicts while continuing the remaining matches.
    """
    key = str(row["source_match_id"])
    match = conn.execute("SELECT * FROM matches WHERE source_match_id=?", (key,)).fetchone()
    if match is None:
        return {"new_results": 0, "enriched_results": 0, "sets_added": 0}
    old_row = conn.execute(
        "SELECT * FROM result_details WHERE source_match_id=?", (key,)
    ).fetchone()
    old = dict(old_row) if old_row else {}
    status = result_status(row)
    winner = optional_scalar(row.get("winner_slug"))
    p1 = optional_scalar(row.get("player_1_slug"))
    p2 = optional_scalar(row.get("player_2_slug"))
    # Ratio adapters use the official A/B orientation; native observations carry
    # their slugs. Infer only the complement of an explicit participant identity.
    pair = {match["player_1_slug"], match["player_2_slug"]}
    if not p1 or not p2:
        if winner in pair and row.get("winner_side") in {"player_1", "player_2"}:
            other = next(iter(pair - {winner})) if len(pair) == 2 else None
            p1, p2 = (winner, other) if row["winner_side"] == "player_1" else (other, winner)
        else:
            p1, p2 = match["player_1_slug"], match["player_2_slug"]
    if not p1 or not p2 or p1 == p2 or {p1, p2} != pair or (winner and winner not in pair):
        raise ValueError("Result participant identity mismatch")
    swap = bool(old and old["player_1_slug"] != p1)
    score_orientation = row.get("score_orientation")
    if score_orientation is None and str(row.get("result_evidence", "")).startswith("tennisratio_"):
        score_orientation = "winner"
    reverse = (
        (score_orientation == "winner" and winner == p2)
        or (score_orientation == "subject" and row.get("score_subject_slug") == p2)
    ) ^ swap
    sets = parse_sets(str(optional_scalar(row.get("sets_score")) or ""), reverse=reverse)
    a = optional_integer(row.get("player_1_sets_won"), "player_1_sets_won")
    b = optional_integer(row.get("player_2_sets_won"), "player_2_sets_won")
    if swap:
        p1, p2, a, b = p2, p1, b, a
    terminal = {"finished", "cancelled", "walkover", "retirement"}
    if old.get("status") in terminal and status in terminal:
        for name, value in (
            ("winner_slug", winner),
            ("player_1_sets_won", a),
            ("player_2_sets_won", b),
        ):
            if old.get(name) is not None and value is not None and old[name] != value:
                raise ValueError(f"Conflicting terminal {name}")
        if old["status"] != status:
            raise ValueError("Conflicting terminal state")
    if old.get("status") in terminal and status not in terminal:
        return {"new_results": 0, "enriched_results": 0, "sets_added": 0}
    added = 0
    for item in sets:
        previous = conn.execute(
            "SELECT * FROM result_sets WHERE source_match_id=? AND set_number=?",
            (key, item["set_number"]),
        ).fetchone()
        if previous:
            growing = old.get("status") not in terminal and all(
                item[field] >= previous[field] for field in ("player_1_games", "player_2_games")
            )
            for field in (
                "player_1_games",
                "player_2_games",
                "tiebreak_player_1",
                "tiebreak_player_2",
                "score_kind",
            ):
                if (
                    previous[field] is not None
                    and item[field] is not None
                    and previous[field] != item[field]
                ):
                    if not (growing and field in {"player_1_games", "player_2_games"}):
                        raise ValueError(f"Conflicting set {item['set_number']} {field}")
                if item[field] is None:
                    item[field] = previous[field]
        conn.execute(
            """INSERT INTO result_sets VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(source_match_id,set_number) DO UPDATE SET
            player_1_games=excluded.player_1_games,player_2_games=excluded.player_2_games,
            tiebreak_player_1=COALESCE(result_sets.tiebreak_player_1,excluded.tiebreak_player_1),
            tiebreak_player_2=COALESCE(result_sets.tiebreak_player_2,excluded.tiebreak_player_2),
            observation_id=CASE WHEN result_sets.player_1_games<>excluded.player_1_games
              OR result_sets.player_2_games<>excluded.player_2_games
              OR (result_sets.tiebreak_player_1 IS NULL AND excluded.tiebreak_player_1 IS NOT NULL)
              OR (result_sets.tiebreak_player_2 IS NULL AND excluded.tiebreak_player_2 IS NOT NULL)
              THEN excluded.observation_id ELSE result_sets.observation_id END
        """,
            (
                key,
                item["set_number"],
                item["player_1_games"],
                item["player_2_games"],
                item["tiebreak_player_1"],
                item["tiebreak_player_2"],
                item["score_kind"],
                observation_id,
            ),
        )
        added += int(previous is None)
    all_sets = conn.execute(
        "SELECT * FROM result_sets WHERE source_match_id=? ORDER BY set_number", (key,)
    ).fetchall()
    a = old.get("player_1_sets_won") if a is None else int(a)
    b = old.get("player_2_sets_won") if b is None else int(b)
    winner = winner or old.get("winner_slug")
    complete = status in {"cancelled", "walkover"}
    detail = "not_applicable" if complete else "unknown"
    if status == "finished" and a is not None and b is not None:
        if min(a, b) < 0 or a == b or max(a, b) > 3:
            raise ValueError("Invalid aggregate sets")
        complete = len(all_sets) == a + b and len(all_sets) > 0
        if complete:
            won = sum(s["player_1_games"] > s["player_2_games"] for s in all_sets)
            if won != a or any(s["player_1_games"] == s["player_2_games"] for s in all_sets):
                raise ValueError("Set scores contradict aggregate")
            # Explicit terminal evidence is still needed for each normal set.
            # Short-set formats remain partial rather than inventing their rules.
            complete = all(
                (
                    max(s["player_1_games"], s["player_2_games"]) >= 10
                    and abs(s["player_1_games"] - s["player_2_games"]) >= 2
                )
                if s["score_kind"] == "match_tiebreak_points"
                else (
                    (
                        max(s["player_1_games"], s["player_2_games"]) >= 6
                        and abs(s["player_1_games"] - s["player_2_games"]) >= 2
                    )
                    or sorted((s["player_1_games"], s["player_2_games"])) == [6, 7]
                )
                for s in all_sets
            )
        detail = "detailed" if complete else "partial" if all_sets else "global"
    elif status == "retirement" and all_sets:
        # The last set can be unfinished. Without an accredited expected set
        # count, keep retrying instead of assuming the detail is exhaustive.
        detail = "partial"
    elif all_sets:
        detail = "partial"
    if old.get("complete"):
        complete, detail = True, old["detail_status"]
        if not sets:
            return {"new_results": 0, "enriched_results": 0, "sets_added": 0}
    conn.execute(
        """INSERT INTO result_details VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_match_id) DO UPDATE SET status=excluded.status,
        winner_slug=excluded.winner_slug,player_1_sets_won=excluded.player_1_sets_won,
        player_2_sets_won=excluded.player_2_sets_won,detail_status=excluded.detail_status,
        complete=excluded.complete,obtained_at_utc=excluded.obtained_at_utc,
        source_system=excluded.source_system,source_url=excluded.source_url,observation_id=excluded.observation_id
    """,
        (
            key,
            status,
            p1,
            p2,
            winner,
            a,
            b,
            detail,
            int(complete),
            match["match_date"],
            obtained_at,
            source,
            optional_scalar(row.get("source_url")),
            observation_id,
        ),
    )
    return {
        "new_results": int(not old and winner is not None),
        "enriched_results": int(added > 0),
        "sets_added": added,
    }


def register_detail(
    conn: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    observation_id: str,
    source: str,
    obtained_at: str,
) -> dict[str, int]:
    """Quarantine one conflicting detail without losing its raw observation."""
    if conn.execute(
        "SELECT 1 FROM result_detail_processed WHERE observation_id=?", (observation_id,)
    ).fetchone():
        return {"new_results": 0, "enriched_results": 0, "sets_added": 0}
    conn.execute("SAVEPOINT result_detail")
    try:
        report = merge_observation(
            conn, row, observation_id=observation_id, source=source, obtained_at=obtained_at
        )
        conn.execute("INSERT OR IGNORE INTO result_detail_processed VALUES(?)", (observation_id,))
        conn.execute("RELEASE result_detail")
        return report
    except (ValueError, TypeError, KeyError, OperationsValidationError) as exc:
        conn.execute("ROLLBACK TO result_detail")
        conn.execute("RELEASE result_detail")
        conn.execute(
            "INSERT OR IGNORE INTO result_detail_conflicts VALUES(?,?)", (observation_id, str(exc))
        )
        return {"new_results": 0, "enriched_results": 0, "sets_added": 0, "errors": 1}


def replay_details(conn: sqlite3.Connection) -> dict[str, int]:
    """Recover existing detail locally; never redownload or rewrite observations."""
    report = {"new_results": 0, "enriched_results": 0, "sets_added": 0, "errors": 0}
    for (
        record
    ) in conn.execute("""SELECT o.* FROM observations o JOIN matches m USING(source_match_id)
        WHERE NOT EXISTS(SELECT 1 FROM result_detail_conflicts c WHERE c.observation_id=o.observation_id)
        AND NOT EXISTS(SELECT 1 FROM result_detail_processed p WHERE p.observation_id=o.observation_id)
        ORDER BY o.observed_at_utc,o.observation_id""").fetchall():
        row = json.loads(record["payload_json"])
        row["source_match_id"] = record["source_match_id"]
        outcome = register_detail(
            conn,
            row,
            observation_id=record["observation_id"],
            source=record["source_system"],
            obtained_at=record["observed_at_utc"],
        )
        for key, value in outcome.items():
            report[key] += value
    return report


def mark_attempt(
    conn: sqlite3.Connection, keys: list[str], now: datetime, error: str | None
) -> None:
    """Back off each attempted match; settled-but-incomplete matches stay eligible."""
    for key in keys:
        row = conn.execute(
            "SELECT attempts FROM result_recovery WHERE source_match_id=?", (key,)
        ).fetchone()
        attempts = int(row[0]) + 1 if row else 1
        due = now + timedelta(hours=min(168, 6 * 2 ** min(attempts - 1, 5)))
        conn.execute(
            """INSERT INTO result_recovery VALUES(?,?,?,?,?) ON CONFLICT(source_match_id)
            DO UPDATE SET last_attempt_at_utc=excluded.last_attempt_at_utc,next_attempt_at_utc=excluded.next_attempt_at_utc,
            attempts=excluded.attempts,error=excluded.error""",
            (key, now.astimezone(UTC).isoformat(), due.isoformat(), attempts, error),
        )


def pending(conn: sqlite3.Connection, now: datetime) -> list[dict[str, Any]]:
    """Include all registered past matches, not just those with an official pick."""
    return [
        dict(row)
        for row in conn.execute(
            """SELECT m.*,r.next_attempt_at_utc,r.last_attempt_at_utc
        FROM matches m LEFT JOIN result_details d USING(source_match_id)
        LEFT JOIN result_recovery r USING(source_match_id)
        WHERE m.match_date < ? AND COALESCE(d.complete,0)=0
        ORDER BY COALESCE(r.last_attempt_at_utc,''),m.match_date DESC,m.source_match_id
    """,
            (now.date().isoformat(),),
        )
    ]


def coverage(conn: sqlite3.Connection, now: datetime) -> dict[str, int]:
    """Count current best coverage once per match, independently of settlements."""
    row = conn.execute(
        """SELECT COUNT(*) AS matches,
        SUM(d.winner_slug IS NOT NULL) AS with_winner,
        SUM(d.status='finished' AND d.player_1_sets_won IS NOT NULL
            AND d.player_2_sets_won IS NOT NULL AND d.player_1_sets_won<>d.player_2_sets_won) AS with_global_score,
        SUM(d.detail_status='detailed') AS with_detailed_sets,
        SUM(d.detail_status IN ('global','partial')) AS partial_results,
        SUM(COALESCE(d.complete,0)=0) AS pending
        FROM matches m LEFT JOIN result_details d USING(source_match_id) WHERE m.match_date < ?
    """,
        (now.date().isoformat(),),
    ).fetchone()
    return {key: int(value or 0) for key, value in dict(row).items()}

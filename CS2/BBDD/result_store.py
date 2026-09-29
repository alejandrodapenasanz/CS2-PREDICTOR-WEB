"""Additive result recovery, deliberately isolated from model input tables.

The legacy matches/maps and frozen predictions are never rewritten here. A
recovered fact has both an event time and an acquisition time. Only the sanctioned
ledger finalizer consumes terminal results; feature loaders do not read this store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from typing import Any
from urllib.parse import urlsplit


SCHEMA = """
CREATE TABLE IF NOT EXISTS result_evidence (
    evidence_id TEXT PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    event_time TEXT NOT NULL,
    obtained_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS result_status (
    match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
    status TEXT NOT NULL,
    score_t1 INTEGER, score_t2 INTEGER,
    winner_team_id INTEGER REFERENCES teams(team_id),
    best_of INTEGER,
    detail_status TEXT NOT NULL,
    complete INTEGER NOT NULL DEFAULT 0,
    evidence_id TEXT NOT NULL REFERENCES result_evidence(evidence_id)
);
CREATE TABLE IF NOT EXISTS result_maps (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    map_number INTEGER NOT NULL,
    map_name TEXT,
    rounds_t1 INTEGER, rounds_t2 INTEGER,
    winner_team_id INTEGER REFERENCES teams(team_id),
    overtime INTEGER,
    evidence_id TEXT NOT NULL REFERENCES result_evidence(evidence_id),
    PRIMARY KEY(match_id,map_number)
);
CREATE TABLE IF NOT EXISTS result_fetch_state (
    match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
    attempted_at_utc TEXT NOT NULL,
    next_attempt_at_utc TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT
);
CREATE VIEW IF NOT EXISTS result_settlement_v1 AS
WITH market AS (
 SELECT ledger_id, CASE
   WHEN julianday(opening_odds_captured_at_utc) < julianday(kickoff_utc)
    AND json_valid(prediction_json)
    AND json_extract(prediction_json,'$.odds_prob_team1') > 0
    AND json_extract(prediction_json,'$.odds_prob_team1') < 1
   THEN json_extract(prediction_json,'$.odds_prob_team1') END AS probability
 FROM prediction_ledger
)
SELECT p.ledger_id, p.match_id, p.model_version, p.predicted_at_utc,
       p.prob_team1 AS predicted_probability_team1,
       CASE WHEN p.prob_team1 >= .5 THEN p.team1_id ELSE p.team2_id END AS predicted_winner,
       CASE WHEN p.actual_team1_win=1 THEN p.team1_id
            WHEN p.actual_team1_win=0 THEN p.team2_id END AS actual_winner,
       p.prediction_correct, p.updated_at_utc AS settlement_timestamp,
       COALESCE(r.score_t1,m.score_t1) AS score_t1,
       COALESCE(r.score_t2,m.score_t2) AS score_t2,
       market.probability AS market_probability_team1,
       1-market.probability AS market_probability_team2,
       CASE WHEN market.probability > .5 THEN p.team1_id
            WHEN market.probability < .5 THEN p.team2_id END AS market_favorite,
       CASE WHEN market.probability > .5 THEN p.team2_id
            WHEN market.probability < .5 THEN p.team1_id END AS market_underdog,
       CASE WHEN market.probability > .5 THEN 1-p.actual_team1_win
            WHEN market.probability < .5 THEN p.actual_team1_win END AS market_upset,
       p.prediction_json AS frozen_prediction_json
FROM prediction_ledger p JOIN matches m USING(match_id)
JOIN market USING(ledger_id)
LEFT JOIN result_status r USING(match_id)
WHERE p.ledger_status='evaluated';
CREATE VIEW IF NOT EXISTS result_details_v1 AS
SELECT r.*,m.team1_id,m.team2_id,e.event_time,e.obtained_at_utc,e.source_url,
       CASE WHEN r.winner_team_id=m.team1_id THEN m.team2_id
            WHEN r.winner_team_id=m.team2_id THEN m.team1_id END AS loser_team_id
FROM result_status r JOIN matches m USING(match_id)
JOIN result_evidence e USING(evidence_id);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create only additive sidecars, without implicit transaction commits."""
    for statement in SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)


def utc(value: str) -> datetime:
    """Require an aware timestamp; never manufacture an intraday ordering."""
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Result acquisition timestamp requires a timezone")
    return stamp.astimezone(timezone.utc)


def _rows(conn: sqlite3.Connection, query: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Read named rows regardless of the caller's row factory."""
    cur = conn.execute(query, args)
    columns = [column[0] for column in cur.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cur]


def inventory(conn: sqlite3.Connection, now: str) -> list[dict[str, Any]]:
    """Classify legacy and recovered coverage without migrating historical rows."""
    entries = _rows(
        conn,
        """
        SELECT m.*, t1.hltv_id AS team1_hltv_id, t2.hltv_id AS team2_hltv_id,
               r.status AS recovered_status, r.score_t1 AS recovered_t1,
               r.score_t2 AS recovered_t2, r.complete AS recovered_complete,
               r.detail_status, f.next_attempt_at_utc, f.attempted_at_utc
        FROM matches m JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        LEFT JOIN result_status r USING(match_id)
        LEFT JOIN result_fetch_state f USING(match_id)
        WHERE substr(m.datetime_utc,1,10) <= substr(?,1,10)
    """,
        (now,),
    )
    maps: dict[int, dict[int, dict[str, Any]]] = {}
    for table in ("maps", "result_maps"):
        for row in _rows(conn, f"SELECT match_id,map_number,map_name,rounds_t1,rounds_t2 FROM {table}"):
            target = maps.setdefault(row["match_id"], {}).setdefault(row["map_number"], {})
            target.update({k: v for k, v in row.items() if v is not None})
    for row in entries:
        s1 = row["recovered_t1"] if row["recovered_t1"] is not None else row["score_t1"]
        s2 = row["recovered_t2"] if row["recovered_t2"] is not None else row["score_t2"]
        row["has_global"] = s1 is not None and s2 is not None and s1 != s2
        expected = (1 if row["best_of"] == 1 else s1 + s2) if row["has_global"] else None
        valid_maps = [
            v
            for v in maps.get(row["match_id"], {}).values()
            if v.get("map_name") not in (None, "Unknown", "TBA", "Default")
            and v.get("rounds_t1") is not None
            and v.get("rounds_t2") is not None
            and v["rounds_t1"] != v["rounds_t2"]
        ]
        row["maps_known"] = len(valid_maps)
        row["complete"] = bool(row["recovered_complete"]) or (
            row["status"] == "completed" and expected is not None and len(valid_maps) == expected
        )
        row["status"] = row["recovered_status"] or row["status"]
    return entries


def coverage(conn: sqlite3.Connection, now: str) -> dict[str, int]:
    """Report matches, not observations, including already settled partials."""
    rows = inventory(conn, now)
    return {
        "matches": len(rows),
        "with_winner": sum(r["winner_team_id"] is not None or r["has_global"] for r in rows),
        "with_global_score": sum(r["has_global"] for r in rows),
        "with_all_maps": sum(r["complete"] and r["maps_known"] > 0 for r in rows),
        "incomplete_maps": sum(r["has_global"] and not r["complete"] for r in rows),
        "pending": sum(not r["complete"] for r in rows),
    }


def plan(conn: sqlite3.Connection, master: dict[str, Any], *, now: str, limit: int = 20) -> dict[str, Any]:
    """Reuse acquisition work items; only observed, exact HLTV URLs are eligible."""
    references = {str(k): v.get("link") for k, v in master.items() if isinstance(v, dict)}
    for row in _rows(conn, "SELECT hltv_match_id,payload_json FROM raw_results"):
        if references.get(str(row["hltv_match_id"])):
            continue
        try:
            payload = json.loads(row["payload_json"])
            references[str(row["hltv_match_id"])] = payload.get("link")
        except (TypeError, ValueError, AttributeError):
            continue
    candidates = []
    unresolved = 0
    for row in inventory(conn, now):
        if row["complete"] or (row["next_attempt_at_utc"] and utc(row["next_attempt_at_utc"]) > utc(now)):
            continue
        if "T" in row["datetime_utc"] and utc(row["datetime_utc"]) >= utc(now):
            continue
        key = str(row["hltv_match_id"] or "")
        link = references.get(key)
        parts = urlsplit(link or "")
        if not link or parts.netloc not in ("", "www.hltv.org") or not parts.path.startswith(f"/matches/{key}/"):
            unresolved += 1
            continue
        candidates.append({"entity_type": "match_result", "entity_key": key, "url": link, "record": row})
    # Old attempts get another turn; never let one unavailable match monopolize a run.
    candidates.sort(key=lambda v: (v["record"]["attempted_at_utc"] or "", -int(v["record"]["match_id"])))
    return {"selected": candidates[: max(0, limit)], "eligible": len(candidates), "without_url": unresolved}


def record_attempt(conn: sqlite3.Connection, match_id: int, now: str, error: str | None = None) -> None:
    """Persist bounded exponential retries; errors never mean permanently done."""
    previous = conn.execute("SELECT attempts FROM result_fetch_state WHERE match_id=?", (match_id,)).fetchone()
    attempts = int(previous[0]) + 1 if previous else 1
    delay_hours = min(168, 6 * 2 ** min(attempts - 1, 5))
    due = (utc(now) + timedelta(hours=delay_hours)).isoformat()
    conn.execute(
        """INSERT INTO result_fetch_state VALUES(?,?,?,?,?)
        ON CONFLICT(match_id) DO UPDATE SET attempted_at_utc=excluded.attempted_at_utc,
        next_attempt_at_utc=excluded.next_attempt_at_utc,attempts=excluded.attempts,error=excluded.error
    """,
        (match_id, now, due, attempts, error),
    )


def persist(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, int]:
    """Merge compatible fields only; contradictions fail before canonical writes."""
    rows = _rows(
        conn,
        """SELECT m.*,t1.hltv_id AS h1,t2.hltv_id AS h2 FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.hltv_match_id=?""",
        (str(payload["hltv_match_id"]),),
    )
    if len(rows) != 1:
        raise ValueError("Result has no unique registered match")
    match = rows[0]
    key = int(match["match_id"])
    source_ids = [str(payload.get("team1_hltv_id")), str(payload.get("team2_hltv_id"))]
    if source_ids == [str(match["h2"]), str(match["h1"])]:
        # Only an exact reversed stable-ID pair authorizes reorientation.
        payload = {
            **payload,
            "source_payload": payload,
            "team1_hltv_id": match["h1"],
            "team2_hltv_id": match["h2"],
            "score_t1": payload.get("score_t2"),
            "score_t2": payload.get("score_t1"),
            "maps": [
                {**m, "rounds_t1": m.get("rounds_t2"), "rounds_t2": m.get("rounds_t1")} for m in payload.get("maps", [])
            ],
        }
    if [str(payload.get("team1_hltv_id")), str(payload.get("team2_hltv_id"))] != [str(match["h1"]), str(match["h2"])]:
        raise ValueError("Result participant orientation does not match registered IDs")
    utc(str(payload["obtained_at_utc"]))
    status = str(payload["status"])
    if status not in {"unknown", "scheduled", "live", "finished", "cancelled", "postponed", "walkover", "retirement"}:
        raise ValueError("Unsupported result status")
    previous = _rows(conn, "SELECT * FROM result_status WHERE match_id=?", (key,))
    old = previous[0] if previous else {}
    s1, s2 = payload.get("score_t1"), payload.get("score_t2")
    terminal = {"finished", "cancelled", "walkover", "retirement"}
    if old.get("status") in terminal and status not in terminal:
        return {"new_results": 0, "enriched_results": 0, "maps_added": 0}
    for value in (s1, s2):
        if value is not None and (type(value) is not int or value < 0 or value > 3):
            raise ValueError("Invalid series score")
    legacy_scores: tuple[Any, Any] = (match.get("score_t1"), match.get("score_t2"))
    if (
        payload.get("best_of") == 1
        and all(isinstance(v, int) for v in legacy_scores)
        and legacy_scores[0] != legacy_scores[1]
    ):
        legacy_scores = (1, 0) if legacy_scores[0] > legacy_scores[1] else (0, 1)
    for name, incoming in (("score_t1", s1), ("score_t2", s2)):
        known = (
            old.get(name)
            if old.get("status") == "finished"
            else (legacy_scores[0 if name == "score_t1" else 1] if match["status"] == "completed" else None)
        )
        # Only terminal evidence establishes immutable score facts.
        if status == "finished" and known is not None and incoming is not None and incoming != known:
            raise ValueError(f"Conflicting final {name}; preserved original")
    s1 = old.get("score_t1") if s1 is None else s1
    s2 = old.get("score_t2") if s2 is None else s2
    if status == "finished" and (s1 is None or s2 is None or s1 == s2):
        raise ValueError("Finished result requires a non-tied series score")
    winner = None
    if status == "finished":
        assert s1 is not None and s2 is not None
        winner = match["team1_id"] if s1 > s2 else match["team2_id"]
    if old.get("status") in {"finished", "cancelled", "walkover", "retirement"}:
        if status in {"finished", "cancelled", "walkover", "retirement"} and status != old["status"]:
            raise ValueError("Conflicting terminal status")
        status = old["status"]
        winner = old["winner_team_id"]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    semantic = {k: v for k, v in payload.items() if k not in {"obtained_at_utc"}}
    evidence = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO result_evidence VALUES(?,?,?,?,?,?)",
        (evidence, key, match["datetime_utc"], payload["obtained_at_utc"], payload["source_url"], encoded),
    )
    added = 0
    for item in payload.get("maps", []):
        number = int(item["map_number"])
        if number < 1 or number > 5:
            raise ValueError("Invalid map ordinal")
        old_maps = _rows(conn, "SELECT * FROM result_maps WHERE match_id=? AND map_number=?", (key, number))
        prior = old_maps[0] if old_maps else {}
        fields = ("map_name", "rounds_t1", "rounds_t2", "overtime")
        values = {name: item.get(name) if item.get(name) is not None else prior.get(name) for name in fields}
        for name in fields:
            if prior.get(name) is not None and item.get(name) is not None and prior[name] != item[name]:
                raise ValueError(f"Conflicting map {number} {name}")
        a, b = values["rounds_t1"], values["rounds_t2"]
        if any(v is not None and (type(v) is not int or v < 0) for v in (a, b)):
            raise ValueError("Invalid rounds")
        map_winner = None if a is None or b is None or a == b else match["team1_id"] if a > b else match["team2_id"]
        conn.execute(
            """INSERT INTO result_maps VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(match_id,map_number)
            DO UPDATE SET map_name=excluded.map_name,rounds_t1=excluded.rounds_t1,
            rounds_t2=excluded.rounds_t2,winner_team_id=excluded.winner_team_id,
            overtime=excluded.overtime,evidence_id=excluded.evidence_id""",
            (key, number, values["map_name"], a, b, map_winner, values["overtime"], evidence),
        )
        added += int(not prior)
    maps = _rows(conn, "SELECT * FROM result_maps WHERE match_id=? ORDER BY map_number", (key,))
    full = [
        m for m in maps if m["map_name"] not in (None, "Unknown", "TBA", "Default") and m["winner_team_id"] is not None
    ]
    complete = status in {"cancelled", "walkover"}
    detail_status = "not_applicable" if complete else "unknown"
    if status == "finished":
        assert s1 is not None and s2 is not None
        expected = s1 + s2
        complete = len(full) == expected and [m["map_number"] for m in full] == list(range(1, expected + 1))
        if complete and (sum(m["winner_team_id"] == match["team1_id"] for m in full) != s1):
            raise ValueError("Map wins contradict series score")
        detail_status = "detailed" if complete else "partial" if maps else "global"
    elif maps:
        detail_status = "partial"
    conn.execute(
        """INSERT INTO result_status VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(match_id)
        DO UPDATE SET status=excluded.status,score_t1=excluded.score_t1,score_t2=excluded.score_t2,
        winner_team_id=excluded.winner_team_id,best_of=excluded.best_of,detail_status=excluded.detail_status,
        complete=excluded.complete,evidence_id=excluded.evidence_id""",
        (
            key,
            status,
            s1,
            s2,
            winner,
            payload.get("best_of") or old.get("best_of"),
            detail_status,
            int(complete),
            evidence,
        ),
    )
    return {
        "new_results": int(status == "finished" and not old and match["winner_team_id"] is None),
        "enriched_results": int(added > 0),
        "maps_added": added,
    }

"""Idempotent integrity repair for the live CS2 database.

The repair only promotes evidence already captured in the master/raw database:
it never guesses participant identities, event values, timestamps or closing
prices. Derived ratings are rebuilt after the core facts are consistent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from BBDD import build_db, deduplicate_matches
from cs2model import dataio
from cs2model.identity import (
    choose_match_team,
    is_provisional_team_name,
    resolved_team,
)


DEFAULT_REPORT = ROOT / "BBDD" / "integrity_repair_report.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_master(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _score_from_record(record: dict[str, Any]) -> tuple[int | None, int | None]:
    score = record.get("score") or {}
    left = build_db.parse_int(score.get("team1"))
    right = build_db.parse_int(score.get("team2"))
    detail_match = ((record.get("detail") or {}).get("match") or {})
    if left is None:
        left = build_db.parse_int((detail_match.get("team1") or {}).get("score"))
    if right is None:
        right = build_db.parse_int((detail_match.get("team2") or {}).get("score"))
    return left, right


def _match_identity_key(
    date_value: Any,
    event: Any,
    best_of: Any,
    team1: Any,
    score1: Any,
    team2: Any,
    score2: Any,
) -> tuple[Any, ...] | None:
    left_score = build_db.parse_int(score1)
    right_score = build_db.parse_int(score2)
    if left_score is None or right_score is None:
        return None
    teams = tuple(
        sorted(
            (
                (dataio.clean_team(team1), left_score),
                (dataio.clean_team(team2), right_score),
            )
        )
    )
    if not all(name for name, _score in teams):
        return None
    return (
        str(date_value or "")[:10],
        dataio.clean_team(event),
        int(best_of or 0),
        teams,
    )


def _backfill_match_ids_from_rows(
    conn: sqlite3.Connection,
    source_rows: list[dict[str, Any]],
) -> dict[str, int]:
    source_groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in source_rows:
        fmt = str(row.get("format") or "")
        best_of = int(fmt[-1]) if fmt in {"bo1", "bo2", "bo3", "bo5"} else 0
        key = _match_identity_key(
            row.get("date"),
            row.get("event"),
            best_of,
            row.get("team1"),
            row.get("score1"),
            row.get("team2"),
            row.get("score2"),
        )
        hltv_id = str(row.get("id") or "").strip()
        if key and hltv_id.isdigit():
            source_groups[key].append(hltv_id)

    target_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    targets = conn.execute(
        """
        SELECT m.match_id, m.datetime_utc, m.best_of, m.score_t1, m.score_t2,
               e.name AS event_name, t1.name AS team1_name, t2.name AS team2_name
        FROM matches m
        JOIN events e ON e.event_id=m.event_id
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.data_tier='historical_seed' AND m.hltv_match_id IS NULL
        """
    ).fetchall()
    for row in targets:
        key = _match_identity_key(
            row["datetime_utc"],
            row["event_name"],
            row["best_of"],
            row["team1_name"],
            row["score_t1"],
            row["team2_name"],
            row["score_t2"],
        )
        if key:
            target_groups[key].append(int(row["match_id"]))

    used = {
        str(row[0])
        for row in conn.execute(
            "SELECT hltv_match_id FROM matches WHERE hltv_match_id IS NOT NULL"
        )
    }
    updated = conflicts = 0
    for key, match_ids in target_groups.items():
        source_ids = sorted(
            {value for value in source_groups.get(key, []) if value not in used},
            key=int,
        )
        match_ids = sorted(match_ids)
        if not source_ids:
            continue
        if len(source_ids) != len(match_ids):
            conflicts += len(match_ids)
            continue
        for match_id, hltv_id in zip(match_ids, source_ids):
            conn.execute(
                "UPDATE matches SET hltv_match_id=? WHERE match_id=?",
                (hltv_id, match_id),
            )
            used.add(hltv_id)
            updated += 1
    return {"updated": updated, "conflicts": conflicts}


def backfill_historical_match_ids(
    conn: sqlite3.Connection,
    raw_path: Path,
    master_path: Path,
) -> dict[str, int]:
    """Recover stable HLTV IDs only from exact, cardinality-preserving evidence."""
    raw_rows = dataio.load_results(raw_path, cs2_only=True) if raw_path.exists() else []
    raw_result = _backfill_match_ids_from_rows(conn, raw_rows)
    master_rows = (
        dataio.load_daily_completed(master_path) if master_path.exists() else []
    )
    master_result = _backfill_match_ids_from_rows(conn, master_rows)
    remaining = int(
        conn.execute(
            "SELECT COUNT(*) FROM matches "
            "WHERE data_tier='historical_seed' AND hltv_match_id IS NULL"
        ).fetchone()[0]
    )
    return {
        "historical_hltv_ids_from_raw": raw_result["updated"],
        "historical_hltv_ids_from_master": master_result["updated"],
        "historical_hltv_id_conflicts": (
            raw_result["conflicts"] + master_result["conflicts"]
        ),
        "historical_hltv_ids_missing": remaining,
    }


def repair_completed_participants(
    conn: sqlite3.Connection, master: dict[str, dict[str, Any]]
) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT m.match_id, m.hltv_match_id, t1.name AS team1, t2.name AS team2
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.status='completed'
        """
    ).fetchall()
    repaired = unresolved = 0
    team_cache, team_hltv_cache = build_db._load_team_caches(conn)
    for row in rows:
        if not (
            is_provisional_team_name(row["team1"])
            or is_provisional_team_name(row["team2"])
        ):
            continue
        record = master.get(str(row["hltv_match_id"])) or {}
        team1 = choose_match_team(record, "team1")
        team2 = choose_match_team(record, "team2")
        if not resolved_team(team1) or not resolved_team(team2):
            unresolved += 1
            continue
        team1_id = build_db._team_id_for(
            conn, team_cache, team_hltv_cache,
            name=team1["name"], hltv_id=team1.get("id"),
        )
        team2_id = build_db._team_id_for(
            conn, team_cache, team_hltv_cache,
            name=team2["name"], hltv_id=team2.get("id"),
        )
        score1, score2 = _score_from_record(record)
        winner_id = (
            team1_id if score1 is not None and score2 is not None and score1 > score2
            else team2_id if score1 is not None and score2 is not None and score2 > score1
            else None
        )
        conn.execute(
            """
            UPDATE matches
            SET team1_id=?, team2_id=?,
                winner_team_id=COALESCE(?, winner_team_id),
                score_t1=COALESCE(?, score_t1),
                score_t2=COALESCE(?, score_t2)
            WHERE match_id=?
            """,
            (team1_id, team2_id, winner_id, score1, score2, row["match_id"]),
        )
        repaired += 1
    return {"participants_repaired": repaired, "participants_unresolved": unresolved}


def repair_event_metadata(
    conn: sqlite3.Connection, master: dict[str, dict[str, Any]]
) -> int:
    event_states: dict[int, dict[str, tuple[Any, ...]]] = {}
    for hltv_id, record in master.items():
        metadata: dict[str, Any] = {}
        analytics = record.get("analytics") or {}
        if isinstance(analytics.get("event_metadata"), dict):
            metadata.update(analytics["event_metadata"])
        if isinstance(record.get("event_metadata"), dict):
            metadata.update(record["event_metadata"])
        if not metadata:
            continue
        prize_raw = metadata.get("prize_pool_raw")
        prize = build_db.parse_money_amount(
            metadata.get("prize_pool") if metadata.get("prize_pool") is not None else prize_raw
        )
        current = conn.execute(
            """
            SELECT e.event_id, e.hltv_event_id, e.prize_pool, e.prize_pool_raw,
                   e.teams_competing, e.tier, e.tier_source,
                   e.source_captured_at_utc, e.source_file
            FROM events e
            JOIN matches m ON m.event_id=e.event_id
            WHERE m.hltv_match_id=?
            """,
            (str(hltv_id),),
        ).fetchone()
        if current is None:
            continue
        event_id = int(current[0])
        original = tuple(current[index] for index in range(1, 9))
        state = event_states.setdefault(
            event_id, {"original": original, "resolved": original}
        )
        incoming = (
            str(metadata.get("hltv_event_id") or "") or None,
            prize,
            prize_raw,
            build_db.parse_int(metadata.get("teams_competing")),
            metadata.get("tier"),
            metadata.get("tier_source"),
            analytics.get("captured_at") or metadata.get("captured_at"),
            analytics.get("source_file") or metadata.get("source_file"),
        )
        existing = state["resolved"]
        resolved = (
            existing[0] if existing[0] is not None else incoming[0],
            incoming[1] if incoming[1] is not None else existing[1],
            incoming[2] if incoming[2] is not None else existing[2],
            incoming[3] if incoming[3] is not None else existing[3],
            incoming[4] if incoming[4] is not None else existing[4],
            incoming[5] if incoming[5] is not None else existing[5],
            incoming[6] if incoming[6] is not None else existing[6],
            incoming[7] if incoming[7] is not None else existing[7],
        )
        state["resolved"] = resolved

    updated = 0
    for event_id, state in event_states.items():
        if state["resolved"] == state["original"]:
            continue
        conn.execute(
            """
            UPDATE events
            SET hltv_event_id=?, prize_pool=?, prize_pool_raw=?,
                teams_competing=?, tier=?, tier_source=?,
                source_captured_at_utc=?, source_file=?
            WHERE event_id=?
            """,
            (*state["resolved"], event_id),
        )
        updated += 1
    return updated


def repair_time_precision_and_prematch(conn: sqlite3.Connection) -> dict[str, int]:
    precision = conn.execute(
        """
        UPDATE matches
        SET datetime_precision=CASE
            WHEN instr(datetime_utc, 'T')=0 THEN 'date_only'
            ELSE 'exact'
        END
        WHERE datetime_precision <> CASE
            WHEN instr(datetime_utc, 'T')=0 THEN 'date_only'
            ELSE 'exact'
        END
        """
    ).rowcount
    conn.execute(
        """
        WITH candidates AS (
            SELECT m.match_id, MIN(captured_at_utc) AS captured_at
            FROM matches m
            JOIN (
                SELECT match_id, captured_at_utc FROM odds
                UNION ALL
                SELECT m2.match_id, rs.captured_at_utc
                FROM raw_snapshots rs
                JOIN matches m2 ON m2.hltv_match_id=rs.hltv_match_id
                WHERE rs.kind IN (
                    'match_snapshot','predictions_enriched','match_assets',
                    'match_analytics','upcoming_matches'
                )
            ) evidence ON evidence.match_id=m.match_id
            WHERE m.datetime_precision='exact'
              AND evidence.captured_at_utc IS NOT NULL
              AND evidence.captured_at_utc <= m.datetime_utc
            GROUP BY m.match_id
        )
        UPDATE matches
        SET prematch_captured_at_utc=(
            SELECT captured_at FROM candidates WHERE candidates.match_id=matches.match_id
        )
        WHERE match_id IN (SELECT match_id FROM candidates)
          AND (
              prematch_captured_at_utc IS NULL
              OR prematch_captured_at_utc > datetime_utc
              OR prematch_captured_at_utc > (
                  SELECT captured_at FROM candidates WHERE candidates.match_id=matches.match_id
              )
          )
        """
    )
    prematch = int(conn.execute("SELECT changes()").fetchone()[0])
    return {"datetime_precision_repaired": precision, "prematch_times_repaired": prematch}


def repair_closing_odds(conn: sqlite3.Connection, max_age_hours: float = 6.0) -> dict[str, int]:
    downgraded = conn.execute(
        """
        UPDATE odds
        SET quality='legacy_proxy', is_observed_closing=0, seconds_to_start=NULL
        WHERE market_type='closing'
          AND NOT EXISTS (
              SELECT 1 FROM matches m
              WHERE m.match_id=odds.match_id
                AND m.datetime_precision='exact'
                AND odds.captured_at_utc <= m.datetime_utc
                AND (julianday(m.datetime_utc)-julianday(odds.captured_at_utc))*24
                    BETWEEN 0 AND ?
          )
          AND (
              quality <> 'legacy_proxy'
              OR is_observed_closing <> 0
              OR seconds_to_start IS NOT NULL
          )
        """
        ,
        (float(max_age_hours),),
    ).rowcount
    promoted = conn.execute(
        """
        UPDATE odds
        SET quality='closing_observed',
            is_observed_closing=1,
            seconds_to_start=CAST(
                ROUND((julianday(m.datetime_utc)-julianday(odds.captured_at_utc))*86400)
                AS INTEGER
            )
        FROM matches m
        WHERE odds.match_id=m.match_id
          AND odds.market_type='closing'
          AND m.datetime_precision='exact'
          AND odds.captured_at_utc <= m.datetime_utc
          AND (julianday(m.datetime_utc)-julianday(odds.captured_at_utc))*24
              BETWEEN 0 AND ?
          AND (
              odds.quality <> 'closing_observed'
              OR odds.is_observed_closing <> 1
              OR odds.seconds_to_start IS NULL
          )
        """,
        (float(max_age_hours),),
    ).rowcount
    return {"closing_downgraded": downgraded, "closing_observed": promoted}


def repair_impossible_series_formats(conn: sqlite3.Connection) -> int:
    """A series score above five is a map/BO1 score, never BO3/BO5."""
    conn.execute(
        """
        UPDATE matches
        SET best_of=1
        WHERE status='completed'
          AND best_of<>1
          AND (score_t1>5 OR score_t2>5)
        """
    )
    return int(conn.execute("SELECT changes()").fetchone()[0])


def _team_reference_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    tables = [
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for table in tables:
        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            if fk[2] == "teams":
                references.append((table, fk[3]))
    return references


def merge_unambiguous_team_identities(conn: sqlite3.Connection) -> dict[str, int]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in conn.execute("SELECT team_id, name, hltv_id FROM teams ORDER BY team_id"):
        key = dataio.clean_team(row["name"])
        if key:
            grouped[key].append(row)
    references = _team_reference_columns(conn)
    merged = skipped = 0
    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        known_ids = {str(row["hltv_id"]) for row in rows if row["hltv_id"] is not None}
        if len(known_ids) > 1:
            skipped += len(rows) - 1
            continue
        survivor = next((row for row in rows if row["hltv_id"] is not None), rows[0])
        survivor_id = int(survivor["team_id"])
        conn.execute(
            """
            INSERT INTO team_aliases(alias_key, alias_name, team_id, source, created_at_utc)
            VALUES (?,?,?,?,?)
            ON CONFLICT(alias_key) DO UPDATE SET team_id=excluded.team_id
            """,
            (key, survivor["name"], survivor_id, "integrity_repair", utcnow()),
        )
        for old in rows:
            old_id = int(old["team_id"])
            if old_id == survivor_id:
                continue
            conn.execute("SAVEPOINT merge_team")
            try:
                # Match participants are core facts: never delete a match to
                # resolve a uniqueness conflict.
                for table, column in references:
                    conn.execute(
                        f'UPDATE OR IGNORE "{table}" SET "{column}"=? WHERE "{column}"=?',
                        (survivor_id, old_id),
                    )
                    if table != "matches":
                        conn.execute(
                            f'DELETE FROM "{table}" WHERE "{column}"=?',
                            (old_id,),
                        )
                conn.execute(
                    "UPDATE OR IGNORE ratings_history SET entity_id=? "
                    "WHERE entity_type='team' AND entity_id=?",
                    (survivor_id, old_id),
                )
                conn.execute(
                    "DELETE FROM ratings_history WHERE entity_type='team' AND entity_id=?",
                    (old_id,),
                )
                remaining_match_refs = conn.execute(
                    """
                    SELECT COUNT(*) FROM matches
                    WHERE team1_id=? OR team2_id=? OR winner_team_id=?
                    """,
                    (old_id, old_id, old_id),
                ).fetchone()[0]
                if remaining_match_refs:
                    raise sqlite3.IntegrityError("team remains referenced by a match")
                conn.execute("DELETE FROM teams WHERE team_id=?", (old_id,))
                conn.execute("RELEASE merge_team")
                merged += 1
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK TO merge_team")
                conn.execute("RELEASE merge_team")
                skipped += 1
    return {"team_identities_merged": merged, "team_identity_conflicts": skipped}


def delete_unreferenced_provisional_teams(conn: sqlite3.Connection) -> int:
    deleted = 0
    for row in conn.execute("SELECT team_id, name FROM teams").fetchall():
        if not is_provisional_team_name(row["name"]):
            continue
        conn.execute("DELETE FROM team_aliases WHERE team_id=?", (row["team_id"],))
        referenced = 0
        for table, column in _team_reference_columns(conn):
            if table == "team_aliases":
                continue
            referenced += conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{column}"=?',
                (row["team_id"],),
            ).fetchone()[0]
        if not referenced:
            conn.execute("DELETE FROM teams WHERE team_id=?", (row["team_id"],))
            deleted += 1
    return deleted


def repair_prediction_references(conn: sqlite3.Connection) -> dict[str, int]:
    conn.execute(
        """
        UPDATE predictions
        SET match_id=(
            SELECT m.match_id FROM matches m
            WHERE m.hltv_match_id=predictions.hltv_match_id
        )
        WHERE match_id IS NULL
          AND EXISTS (
            SELECT 1 FROM matches m
            WHERE m.hltv_match_id=predictions.hltv_match_id
          )
        """
    )
    linked = int(conn.execute("SELECT changes()").fetchone()[0])
    conn.execute(
        """
        UPDATE predictions
        SET hltv_match_id=(
                SELECT m.hltv_match_id FROM matches m
                WHERE m.match_id=predictions.match_id
            ),
            favorite_team_id=CASE decision_favorite_side
                WHEN 'team1' THEN (
                    SELECT m.team1_id FROM matches m
                    WHERE m.match_id=predictions.match_id
                )
                WHEN 'team2' THEN (
                    SELECT m.team2_id FROM matches m
                    WHERE m.match_id=predictions.match_id
                )
                ELSE favorite_team_id
            END,
            favorite_name=CASE decision_favorite_side
                WHEN 'team1' THEN (
                    SELECT t.name FROM matches m
                    JOIN teams t ON t.team_id=m.team1_id
                    WHERE m.match_id=predictions.match_id
                )
                WHEN 'team2' THEN (
                    SELECT t.name FROM matches m
                    JOIN teams t ON t.team_id=m.team2_id
                    WHERE m.match_id=predictions.match_id
                )
                ELSE favorite_name
            END
        WHERE match_id IS NOT NULL
          AND (
            hltv_match_id <> (
                SELECT m.hltv_match_id FROM matches m
                WHERE m.match_id=predictions.match_id
            )
            OR (
                decision_favorite_side='team1'
                AND favorite_team_id <> (
                    SELECT m.team1_id FROM matches m
                    WHERE m.match_id=predictions.match_id
                )
            )
            OR (
                decision_favorite_side='team2'
                AND favorite_team_id <> (
                    SELECT m.team2_id FROM matches m
                    WHERE m.match_id=predictions.match_id
                )
            )
          )
        """
    )
    repaired = int(conn.execute("SELECT changes()").fetchone()[0])
    conn.execute(
        """
        DELETE FROM predictions
        WHERE match_id IS NULL
          AND NOT EXISTS (
            SELECT 1 FROM matches m
            WHERE m.hltv_match_id=predictions.hltv_match_id
          )
        """
    )
    deleted = int(conn.execute("SELECT changes()").fetchone()[0])
    return {
        "prediction_match_links_repaired": linked,
        "prediction_references_repaired": repaired,
        "orphan_predictions_deleted": deleted,
    }


def unconfirmed_live_match_ids(conn: sqlite3.Connection) -> list[int]:
    """Return normalized non-seed matches lacking two stable HLTV teams."""
    rows = conn.execute(
        """
        SELECT m.match_id, t1.name, t1.hltv_id, t2.name, t2.hltv_id
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.data_tier <> 'historical_seed'
        """
    ).fetchall()
    return [
        int(row[0])
        for row in rows
        if row[2] is None
        or row[4] is None
        or is_provisional_team_name(row[1])
        or is_provisional_team_name(row[3])
    ]


def provisional_match_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT m.match_id, t1.name, t2.name
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        """
    ).fetchall()
    return [
        int(row[0])
        for row in rows
        if is_provisional_team_name(row[1])
        or is_provisional_team_name(row[2])
    ]


def _purge_match_ids(conn: sqlite3.Connection, match_ids: list[int]) -> int:
    for match_id in match_ids:
        map_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT map_id FROM maps WHERE match_id=?", (match_id,)
            )
        ]
        for map_id in map_ids:
            conn.execute("DELETE FROM map_player_side_stats WHERE map_id=?", (map_id,))
            conn.execute("DELETE FROM map_player_stats WHERE map_id=?", (map_id,))
        analytics_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT analytics_snapshot_id FROM match_analytics_snapshots "
                "WHERE match_id=?",
                (match_id,),
            )
        ]
        for snapshot_id in analytics_ids:
            conn.execute(
                "DELETE FROM match_analytics_map_handicap "
                "WHERE analytics_snapshot_id=?",
                (snapshot_id,),
            )
            conn.execute(
                "DELETE FROM match_analytics_map_stats "
                "WHERE analytics_snapshot_id=?",
                (snapshot_id,),
            )
        for table, column in (
            ("prediction_ledger", "match_id"),
            ("predictions", "match_id"),
            ("odds", "match_id"),
            ("prematch_lineup_snapshots", "match_id"),
            ("match_analytics_snapshots", "match_id"),
            ("match_lineups", "match_id"),
            ("veto", "match_id"),
            ("match_features", "match_id"),
            ("ratings_history", "before_match_id"),
            ("maps", "match_id"),
        ):
            conn.execute(
                f'DELETE FROM "{table}" WHERE "{column}"=?',
                (match_id,),
            )
        conn.execute("DELETE FROM matches WHERE match_id=?", (match_id,))
    return len(match_ids)


def purge_unconfirmed_live_matches(conn: sqlite3.Connection) -> int:
    """Remove unresolved normalized rows while preserving raw recovery evidence."""
    return _purge_match_ids(conn, unconfirmed_live_match_ids(conn))


def purge_provisional_matches(conn: sqlite3.Connection) -> int:
    """Remove any legacy match that still contains a bracket placeholder."""
    return _purge_match_ids(conn, provisional_match_ids(conn))


def run_repair(
    db_path: Path,
    master_path: Path,
    *,
    raw_path: Path = build_db.DEFAULT_RAW,
    rebuild_mart: bool = True,
    backup: bool = True,
) -> dict[str, Any]:
    if backup and db_path.exists():
        build_db.backup_database(db_path, build_db.DEFAULT_BACKUP_DIR)
    conn = build_db.connect_live_db(db_path)
    conn.row_factory = sqlite3.Row
    conn.commit()
    master = _read_master(master_path)
    report: dict[str, Any] = {
        "started_at_utc": utcnow(),
        "db": str(db_path),
        "master": str(master_path),
    }
    try:
        conn.execute("BEGIN IMMEDIATE")
        report.update(repair_completed_participants(conn, master))
        report["event_metadata_repaired"] = repair_event_metadata(conn, master)
        report.update(repair_time_precision_and_prematch(conn))
        report.update(repair_closing_odds(conn))
        report["impossible_series_formats_repaired"] = (
            repair_impossible_series_formats(conn)
        )
        report.update(merge_unambiguous_team_identities(conn))
        report.update(backfill_historical_match_ids(conn, raw_path, master_path))
        report.update(repair_prediction_references(conn))
        report["provisional_matches_purged"] = purge_provisional_matches(conn)
        report["unconfirmed_live_matches_purged"] = purge_unconfirmed_live_matches(conn)
        report["provisional_teams_deleted"] = delete_unreferenced_provisional_teams(conn)
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"foreign_key_check failed: {foreign_key_errors[:5]}")
        conn.commit()

        duplicate_pairs = deduplicate_matches.duplicate_pairs(conn)
        report["physical_duplicate_pairs_found"] = len(duplicate_pairs)
        if duplicate_pairs:
            for key, value in deduplicate_matches.consolidate(
                conn, duplicate_pairs
            ).items():
                report[f"deduplicate_{key}"] = value

        if rebuild_mart:
            rows = dataio.load_training_rows(db_path=db_path)
            conn.execute("BEGIN IMMEDIATE")
            report.update(build_db._recompute_mart(conn, rows))
            conn.commit()

        completed_names = conn.execute(
            """
            SELECT t1.name, t2.name
            FROM matches m
            JOIN teams t1 ON t1.team_id=m.team1_id
            JOIN teams t2 ON t2.team_id=m.team2_id
            WHERE m.status='completed'
            """
        ).fetchall()
        report["completed_provisional_remaining"] = sum(
            1 for row in completed_names
            if is_provisional_team_name(row[0]) or is_provisional_team_name(row[1])
        )
        report["unconfirmed_live_matches_remaining"] = len(
            unconfirmed_live_match_ids(conn)
        )
        report["provisional_matches_remaining"] = len(provisional_match_ids(conn))
        report["provisional_teams_remaining"] = sum(
            1
            for row in conn.execute("SELECT name FROM teams")
            if is_provisional_team_name(row[0])
        )
        report["orphan_predictions_remaining"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE match_id IS NULL"
            ).fetchone()[0]
        )
        report["matches_without_hltv_id_remaining"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM matches WHERE hltv_match_id IS NULL"
            ).fetchone()[0]
        )
        report["physical_duplicate_pairs_remaining"] = len(
            deduplicate_matches.duplicate_pairs(conn)
        )
        report["historical_hltv_ids_missing"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM matches "
                "WHERE data_tier='historical_seed' AND hltv_match_id IS NULL"
            ).fetchone()[0]
        )
        report["foreign_key_errors"] = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        report["finished_at_utc"] = utcnow()
        report["status"] = (
            "ok"
            if report["completed_provisional_remaining"] == 0
            and report["unconfirmed_live_matches_remaining"] == 0
            and report["provisional_matches_remaining"] == 0
            and report["provisional_teams_remaining"] == 0
            and report["orphan_predictions_remaining"] == 0
            and report["matches_without_hltv_id_remaining"] == 0
            and report["physical_duplicate_pairs_remaining"] == 0
            and report["foreign_key_errors"] == 0
            else "partial"
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(build_db.DEFAULT_DB))
    parser.add_argument("--master", default=str(build_db.DEFAULT_MASTER))
    parser.add_argument("--raw", default=str(build_db.DEFAULT_RAW))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--no-rebuild-mart", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    report = run_repair(
        Path(args.db),
        Path(args.master),
        raw_path=Path(args.raw),
        rebuild_mart=not args.no_rebuild_mart,
        backup=not args.no_backup,
    )
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

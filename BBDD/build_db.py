"""Inicializa/migra la base SQLite viva del predictor.

Materializa la arquitectura de tres capas de PROJECT.md (§4.4):

  1. STAGING (raw_results): cada serie scrapeada tal cual, con provenance.
  2. CORE (teams, events, matches): hechos inmutables normalizados.
  3. FEATURE MART (ratings_history, match_features, predictions): vistas
     calculadas POINT-IN-TIME mediante el motor cronológico Glicko-2 de
     cs2model. ratings_history guarda el estado del rating JUSTO ANTES de cada
     partido (before_match_id) → reconstrucción sin fuga temporal.

SQLite viene incluido en Python: no requiere instalar nada.

Uso:
    python BBDD/build_db.py
    python BBDD/build_db.py --raw <results_all.json> --db <salida.db> --master <matches.json>

Si `matches` ya contiene filas, no borra ni reconstruye: solo asegura el
esquema vivo (`fetch_state`, `ingest_runs`, flags de cobertura). La ingesta
diaria incremental vive en `BBDD/ingest.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
sys.path.insert(0, str(MODEL_DIR))
DAILY_ROOT = ROOT / "DAILY_SNAPSHOTS"

from cs2model import dataio
from cs2model.features import ChronologicalState, _period_index
from DAILY_SNAPSHOTS.match_context import parse_match_context_meta, schema_stage

SCHEMA = ROOT / "BBDD" / "cs2_prediction_schema.sql"
DEFAULT_RAW = (
    ROOT / "SCRAPPER" / "hltv-scraper-api" / "hltv_scraper" / "data" / "raw"
    / "history_10000_2026-06-28" / "results_all.json"
)
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_MASTER = ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"
DEFAULT_ROSTER_HISTORY = ROOT / "DAILY_SNAPSHOTS" / "master" / "roster_history.json"
DEFAULT_BACKUP_DIR = ROOT / "BBDD" / "backups"
DEFAULT_MIRROR_BACKUP_DIR = Path(os.environ.get("CS2_BACKUP_MIRROR_DIR", ROOT.parent / "CS2-Predictor-Backups"))


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))


MATCH_LIVE_COLUMNS: dict[str, str] = {
    "hltv_match_id": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('scheduled','pending_result','completed'))",
    "data_tier": "TEXT NOT NULL DEFAULT 'historical_seed' CHECK (data_tier IN ('historical_seed','prematch_captured','completed'))",
    "prematch_captured_at_utc": "TEXT",
    "result_filled_at_utc": "TEXT",
    "has_prematch_odds": "INTEGER NOT NULL DEFAULT 0 CHECK (has_prematch_odds IN (0,1))",
    "has_player_snapshot": "INTEGER NOT NULL DEFAULT 0 CHECK (has_player_snapshot IN (0,1))",
    "has_ranking_snapshot": "INTEGER NOT NULL DEFAULT 0 CHECK (has_ranking_snapshot IN (0,1))",
    "has_analytics": "INTEGER NOT NULL DEFAULT 0 CHECK (has_analytics IN (0,1))",
    "has_context": "INTEGER NOT NULL DEFAULT 0 CHECK (has_context IN (0,1))",
    "has_box_score": "INTEGER NOT NULL DEFAULT 0 CHECK (has_box_score IN (0,1))",
    "has_veto": "INTEGER NOT NULL DEFAULT 0 CHECK (has_veto IN (0,1))",
}

EVENT_LIVE_COLUMNS: dict[str, str] = {
    "hltv_event_id": "TEXT",
    "teams_competing": "INTEGER",
    "source_captured_at_utc": "TEXT",
    "source_file": "TEXT",
}


def configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def ensure_live_schema(conn: sqlite3.Connection) -> None:
    configure_connection(conn)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not tables:
        create_schema(conn)
        return

    existing = table_columns(conn, "matches")
    for column, ddl in MATCH_LIVE_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {column} {ddl}")

    event_columns = table_columns(conn, "events")
    for column, ddl in EVENT_LIVE_COLUMNS.items():
        if column not in event_columns:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} {ddl}")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fetch_state (
            entity_type   TEXT NOT NULL CHECK (entity_type IN
                            ('team_profile','player_stats','ranking_hltv','ranking_valve',
                             'match_detail','match_assets','match_analytics')),
            entity_key    TEXT NOT NULL,
            last_fetched_at_utc  TEXT,
            last_status   TEXT CHECK (last_status IN ('ok','partial','blocked','not_found','error')),
            fetch_count   INTEGER NOT NULL DEFAULT 0,
            next_eligible_at_utc TEXT,
            note          TEXT,
            PRIMARY KEY (entity_type, entity_key)
        );
        CREATE TABLE IF NOT EXISTS ingest_runs (
            ingest_id     INTEGER PRIMARY KEY,
            run_id        TEXT NOT NULL,
            started_at_utc  TEXT NOT NULL,
            finished_at_utc TEXT,
            status        TEXT CHECK (status IN ('ok','partial','failed')),
            rows_upserted_json TEXT,
            requests_made INTEGER,
            requests_skipped_by_freshness INTEGER,
            note          TEXT
        );
        CREATE TABLE IF NOT EXISTS prematch_lineup_snapshots (
            prematch_lineup_snapshot_id INTEGER PRIMARY KEY,
            match_id    INTEGER NOT NULL REFERENCES matches(match_id),
            team_id     INTEGER NOT NULL REFERENCES teams(team_id),
            player_id   INTEGER NOT NULL REFERENCES players(player_id),
            captured_at_utc TEXT NOT NULL,
            run_id      TEXT NOT NULL,
            is_standin  INTEGER NOT NULL DEFAULT 0 CHECK (is_standin IN (0,1)),
            source_file TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE (match_id, team_id, player_id, captured_at_utc, source_file)
        );
        CREATE TABLE IF NOT EXISTS match_analytics_snapshots (
            analytics_snapshot_id INTEGER PRIMARY KEY,
            match_id        INTEGER NOT NULL REFERENCES matches(match_id),
            captured_at_utc TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            source_file     TEXT NOT NULL,
            event_name      TEXT,
            hltv_event_id   TEXT,
            prize_pool      INTEGER,
            teams_competing INTEGER,
            team1_core_matches_lt INTEGER,
            team2_core_matches_lt INTEGER,
            team1_matches_sample INTEGER,
            team1_maps_sample INTEGER,
            team2_matches_sample INTEGER,
            team2_maps_sample INTEGER,
            team1_overtime_pct REAL,
            team2_overtime_pct REAL,
            payload_json    TEXT NOT NULL,
            UNIQUE (match_id, captured_at_utc, source_file)
        );
        CREATE TABLE IF NOT EXISTS match_analytics_map_stats (
            analytics_snapshot_id INTEGER NOT NULL REFERENCES match_analytics_snapshots(analytics_snapshot_id),
            team_id         INTEGER REFERENCES teams(team_id),
            team_name       TEXT,
            map_name        TEXT NOT NULL,
            first_pick_pct  REAL,
            first_ban_pct   REAL,
            win_pct         REAL,
            played          INTEGER,
            comment         TEXT,
            PRIMARY KEY (analytics_snapshot_id, team_name, map_name)
        );
        CREATE TABLE IF NOT EXISTS match_analytics_map_handicap (
            analytics_snapshot_id INTEGER NOT NULL REFERENCES match_analytics_snapshots(analytics_snapshot_id),
            team_id         INTEGER REFERENCES teams(team_id),
            team_name       TEXT,
            map_name        TEXT NOT NULL,
            avg_rounds_lost_in_wins REAL,
            avg_rounds_won_in_losses REAL,
            PRIMARY KEY (analytics_snapshot_id, team_name, map_name)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_hltv ON matches(hltv_match_id);
        CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
        CREATE INDEX IF NOT EXISTS idx_matches_tier ON matches(data_tier);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_events_hltv ON events(hltv_event_id) WHERE hltv_event_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_fetch_state_eligible ON fetch_state(entity_type, next_eligible_at_utc);
        CREATE INDEX IF NOT EXISTS idx_prematch_lineups_match ON prematch_lineup_snapshots(match_id, captured_at_utc);
        CREATE INDEX IF NOT EXISTS idx_analytics_snapshot_match ON match_analytics_snapshots(match_id, captured_at_utc);
        CREATE INDEX IF NOT EXISTS idx_analytics_map_stats_snapshot ON match_analytics_map_stats(analytics_snapshot_id, map_name);
        """
    )


def connect_live_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_live_schema(conn)
    return conn


def table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0] or 0)
    except sqlite3.DatabaseError:
        return 0


def safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stat_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().replace("%", "").replace(",", "."))
    except ValueError:
        return None


def first_stat_value(stats: dict, *keys: str):
    for key in keys:
        value = stats.get(key)
        if value is not None and str(value).strip().lower() not in {"", "-", "--", "n/a", "na"}:
            return value
    return None


def parse_int(value) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", str(value).replace(",", ""))
    return int(match.group(0)) if match else None


def load_master(master_path: Path | None) -> dict:
    if not master_path or not master_path.exists():
        return {}
    return json.loads(master_path.read_text(encoding="utf-8"))


def odds_point_key(point: dict) -> tuple[str, str]:
    return (str(point.get("captured_at") or ""), str(point.get("run_id") or ""))


def iter_odds_rows(point: dict, market_type: str):
    """Expande un punto de odds a filas por bookmaker.

    Los snapshots antiguos solo guardaban la media. Los nuevos pueden guardar
    tambien `providers`, asi que este adaptador soporta ambos formatos.
    """
    if not point:
        return
    captured_at = point.get("captured_at")
    if not captured_at:
        return
    providers = point.get("providers") or []
    if providers:
        for provider in providers:
            yield {
                "bookmaker": provider.get("bookmaker") or "unknown",
                "captured_at_utc": captured_at,
                "market_type": market_type,
                "odds_t1": safe_float(provider.get("team1_decimal")),
                "odds_t2": safe_float(provider.get("team2_decimal")),
                "prob_t1": safe_float(provider.get("team1_implied_prob_norm")),
                "prob_t2": safe_float(provider.get("team2_implied_prob_norm")),
                "overround": safe_float(provider.get("overround")),
                "run_id": point.get("run_id"),
                "source_file": point.get("source_file"),
            }
        return
    yield {
        "bookmaker": "average",
        "captured_at_utc": captured_at,
        "market_type": market_type,
        "odds_t1": safe_float(point.get("team1_decimal")),
        "odds_t2": safe_float(point.get("team2_decimal")),
        "prob_t1": safe_float(point.get("team1_implied_prob_norm")),
        "prob_t2": safe_float(point.get("team2_implied_prob_norm")),
        "overround": safe_float(point.get("overround")),
        "run_id": point.get("run_id"),
        "source_file": point.get("source_file"),
    }


def insert_odds(cur: sqlite3.Cursor, master: dict, match_id_map: dict[str, int]) -> int:
    inserted = 0
    for hltv_id, record in master.items():
        mid = match_id_map.get(str(hltv_id))
        if mid is None:
            continue
        points: list[tuple[str, dict]] = []
        opening = record.get("opening_odds") or {}
        closing = record.get("closing_odds") or {}
        if opening:
            points.append(("opening", opening))
        if closing:
            points.append(("closing", closing))
        seen_live: set[tuple[str, str]] = set()
        for point in record.get("odds_history") or []:
            key = odds_point_key(point)
            if key in seen_live:
                continue
            seen_live.add(key)
            points.append(("live", point))

        for market_type, point in points:
            for row in iter_odds_rows(point, market_type):
                cur.execute(
                    "INSERT OR IGNORE INTO odds(match_id, bookmaker, captured_at_utc, market_type, "
                    "odds_t1, odds_t2, prob_t1, prob_t2, overround, run_id, source_file) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        mid,
                        row["bookmaker"],
                        row["captured_at_utc"],
                        row["market_type"],
                        row["odds_t1"],
                        row["odds_t2"],
                        row["prob_t1"],
                        row["prob_t2"],
                        row["overround"],
                        row["run_id"],
                        row["source_file"],
                    ),
                )
                inserted += cur.rowcount
    return inserted


def resolve_team_id(
    team: dict | None,
    team_ids: dict[str, int],
    team_ids_by_hltv: dict[str, int],
) -> int | None:
    if not team:
        return None
    hltv_id = str(team.get("hltv_id") or team.get("id") or "").strip()
    if hltv_id and hltv_id in team_ids_by_hltv:
        return team_ids_by_hltv[hltv_id]
    name = team.get("name") or team.get("team_name")
    if name:
        return team_ids.get(dataio.clean_team(name))
    return None


def load_asset_payload(record: dict) -> tuple[dict | None, str | None]:
    meta = record.get("hltv_assets") or {}
    source_file = meta.get("source_file") or record.get("hltv_assets_file")
    if not source_file:
        return None, None
    path = DAILY_ROOT / source_file
    if not path.exists():
        return None, source_file
    try:
        return json.loads(path.read_text(encoding="utf-8")), source_file
    except json.JSONDecodeError:
        return None, source_file


def asset_match_context(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    veto = payload.get("veto") or {}
    if not isinstance(veto, dict):
        return {}
    meta = veto.get("meta")
    if meta:
        return parse_match_context_meta(meta)
    context = veto.get("context")
    if isinstance(context, dict) and context:
        return context
    return {}


def build_match_context_index(master: dict) -> dict[str, dict]:
    contexts: dict[str, dict] = {}
    for hltv_match_id, record in master.items():
        context = record.get("match_context")
        if not isinstance(context, dict) or not context:
            payload, _source_file = load_asset_payload(record)
            context = asset_match_context(payload)
        if isinstance(context, dict) and context:
            contexts[str(hltv_match_id)] = context
    return contexts


def get_player_id(
    cur: sqlite3.Cursor,
    player: dict,
    cache: dict[str, int],
) -> int:
    hltv_id = str(player.get("player_hltv_id") or player.get("hltv_id") or "").strip()
    cache_key = hltv_id or f"name:{player.get('player_name') or player.get('nick') or 'unknown'}"
    if cache_key in cache:
        return cache[cache_key]
    if hltv_id:
        row = cur.execute("SELECT player_id FROM players WHERE hltv_id = ?", (int(hltv_id),)).fetchone()
        if row:
            cache[cache_key] = int(row[0])
            return int(row[0])
    nick = player.get("player_name") or player.get("nick") or "unknown"
    try:
        cur.execute(
            "INSERT INTO players(nick, hltv_id) VALUES (?,?)",
            (nick, int(hltv_id) if hltv_id.isdigit() else None),
        )
    except sqlite3.IntegrityError:
        row = cur.execute("SELECT player_id FROM players WHERE hltv_id = ?", (int(hltv_id),)).fetchone()
        if row:
            cache[cache_key] = int(row[0])
            return int(row[0])
        raise
    cache[cache_key] = cur.lastrowid
    return cur.lastrowid


def insert_roster_history(cur: sqlite3.Cursor, team_ids_by_hltv: dict[str, int]) -> int:
    if not DEFAULT_ROSTER_HISTORY.exists():
        return 0
    roster_history = json.loads(DEFAULT_ROSTER_HISTORY.read_text(encoding="utf-8"))
    player_cache: dict[str, int] = {}
    inserted = 0
    source_file = str(DEFAULT_ROSTER_HISTORY.relative_to(ROOT))

    for team_hltv_id, entry in roster_history.items():
        team_hltv_id = str(entry.get("team_id") or team_hltv_id or "").strip()
        if not team_hltv_id:
            continue
        team_id = team_ids_by_hltv.get(team_hltv_id)
        if team_id is None:
            cur.execute(
                "INSERT OR IGNORE INTO teams(name, hltv_id) VALUES (?,?)",
                (entry.get("team_name") or f"HLTV team {team_hltv_id}", int(team_hltv_id) if team_hltv_id.isdigit() else None),
            )
            row = cur.execute("SELECT team_id FROM teams WHERE hltv_id = ?", (int(team_hltv_id),)).fetchone() if team_hltv_id.isdigit() else None
            if not row:
                row = cur.execute("SELECT team_id FROM teams WHERE name = ?", (entry.get("team_name"),)).fetchone()
            if not row:
                continue
            team_id = int(row[0])
            team_ids_by_hltv[team_hltv_id] = team_id

        snapshots = sorted(entry.get("snapshots") or [], key=lambda s: s.get("captured_at") or "")
        active: dict[str, dict] = {}
        intervals: list[dict] = []
        for snap in snapshots:
            captured_at = snap.get("captured_at")
            if not captured_at:
                continue
            player_ids = [str(pid).strip() for pid in (snap.get("player_ids") or []) if str(pid).strip()]
            player_names = snap.get("player_names") or []
            current = set(player_ids)
            for player_hltv_id in list(active):
                if player_hltv_id not in current:
                    intervals.append({**active.pop(player_hltv_id), "valid_to": captured_at})
            for idx, player_hltv_id in enumerate(player_ids):
                if player_hltv_id in active:
                    continue
                active[player_hltv_id] = {
                    "player_hltv_id": player_hltv_id,
                    "player_name": player_names[idx] if idx < len(player_names) else None,
                    "valid_from": captured_at,
                    "source_run_id": snap.get("run_id"),
                    "source_signature": snap.get("signature"),
                }
        intervals.extend({**item, "valid_to": None} for item in active.values())

        for interval in intervals:
            player_id = get_player_id(
                cur,
                {
                    "hltv_id": interval["player_hltv_id"],
                    "nick": interval.get("player_name") or interval["player_hltv_id"],
                },
                player_cache,
            )
            cur.execute(
                "INSERT OR IGNORE INTO team_rosters(team_id, player_id, valid_from, valid_to, "
                "source_run_id, source_signature, source_file) VALUES (?,?,?,?,?,?,?)",
                (
                    team_id,
                    player_id,
                    interval["valid_from"],
                    interval.get("valid_to"),
                    interval.get("source_run_id"),
                    interval.get("source_signature"),
                    source_file,
                ),
            )
            inserted += cur.rowcount
    return inserted


def orient_pair(
    left_value,
    right_value,
    *,
    left_team_id: int | None,
    right_team_id: int | None,
    team1_id: int,
    team2_id: int,
) -> tuple:
    if left_team_id == team1_id and right_team_id == team2_id:
        return left_value, right_value
    if left_team_id == team2_id and right_team_id == team1_id:
        return right_value, left_value
    return left_value, right_value


def map_key(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def insert_hltv_assets(
    cur: sqlite3.Cursor,
    master: dict,
    match_id_map: dict[str, int],
    match_teams: dict[int, tuple[int, int]],
    team_ids: dict[str, int],
    team_ids_by_hltv: dict[str, int],
) -> dict[str, int]:
    counts = {
        "match_assets_loaded": 0,
        "veto_rows": 0,
        "maps_rows": 0,
        "lineup_rows": 0,
        "map_player_stats_rows": 0,
        "map_player_side_stats_rows": 0,
    }
    player_cache: dict[str, int] = {}
    lineup_seen: set[tuple[int, int, int]] = set()
    for hltv_match_id, record in master.items():
        mid = match_id_map.get(str(hltv_match_id))
        if mid is None:
            continue
        payload, source_file = load_asset_payload(record)
        if not payload:
            continue
        counts["match_assets_loaded"] += 1
        team1_id, team2_id = match_teams[mid]

        pick_by_map: dict[str, int | None] = {}
        for step in (payload.get("veto") or {}).get("steps") or []:
            team_id = resolve_team_id({"name": step.get("team_name")}, team_ids, team_ids_by_hltv)
            cur.execute(
                "INSERT OR IGNORE INTO veto(match_id, step_order, team_id, action, map_name) VALUES (?,?,?,?,?)",
                (mid, step.get("step_order"), team_id, step.get("action"), step.get("map_name")),
            )
            counts["veto_rows"] += cur.rowcount
            if step.get("action") == "pick":
                pick_by_map[map_key(step.get("map_name"))] = team_id
            elif step.get("action") == "decider":
                pick_by_map.setdefault(map_key(step.get("map_name")), None)

        for map_number, map_payload in enumerate(payload.get("mapstats") or [], start=1):
            info = map_payload.get("info") or {}
            left_team = info.get("team_left") or {}
            right_team = info.get("team_right") or {}
            left_team_id = resolve_team_id(left_team, team_ids, team_ids_by_hltv)
            right_team_id = resolve_team_id(right_team, team_ids, team_ids_by_hltv)
            breakdown = info.get("breakdown") or {}
            left_score = left_team.get("score") if left_team.get("score") is not None else breakdown.get("score_left")
            right_score = right_team.get("score") if right_team.get("score") is not None else breakdown.get("score_right")
            rounds_t1, rounds_t2 = orient_pair(
                left_score,
                right_score,
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                team1_id=team1_id,
                team2_id=team2_id,
            )
            winner_team_id = None
            if left_score is not None and right_score is not None and left_score != right_score:
                winner_team_id = left_team_id if left_score > right_score else right_team_id
            side_rounds = breakdown.get("side_rounds") or {}
            left_side = side_rounds.get("left") or {}
            right_side = side_rounds.get("right") or {}
            if left_team_id == team2_id and right_team_id == team1_id:
                team1_ct, team1_t = right_side.get("ct"), right_side.get("t")
                team2_ct, team2_t = left_side.get("ct"), left_side.get("t")
            else:
                team1_ct, team1_t = left_side.get("ct"), left_side.get("t")
                team2_ct, team2_t = right_side.get("ct"), right_side.get("t")
            overtime_t1, overtime_t2 = orient_pair(
                side_rounds.get("overtime_left"),
                side_rounds.get("overtime_right"),
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                team1_id=team1_id,
                team2_id=team2_id,
            )
            ratings = info.get("team_rating_3_0") or {}
            team1_rating, team2_rating = orient_pair(
                ratings.get("left"),
                ratings.get("right"),
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                team1_id=team1_id,
                team2_id=team2_id,
            )
            first_kills = info.get("first_kills") or {}
            team1_first_kills, team2_first_kills = orient_pair(
                first_kills.get("left"),
                first_kills.get("right"),
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                team1_id=team1_id,
                team2_id=team2_id,
            )
            clutches = info.get("clutches_won") or {}
            team1_clutches, team2_clutches = orient_pair(
                clutches.get("left"),
                clutches.get("right"),
                left_team_id=left_team_id,
                right_team_id=right_team_id,
                team1_id=team1_id,
                team2_id=team2_id,
            )
            map_name = info.get("map_name") or "Unknown"
            cur.execute(
                "INSERT OR IGNORE INTO maps("
                "match_id, map_number, map_name, picked_by_team_id, winner_team_id, rounds_t1, rounds_t2, "
                "team1_ct_rounds, team1_t_rounds, team2_ct_rounds, team2_t_rounds, overtime_t1, overtime_t2, "
                "team1_rating, team2_rating, team1_first_kills, team2_first_kills, team1_clutches, team2_clutches, "
                "hltv_mapstats_id, source_file) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    map_number,
                    map_name,
                    pick_by_map.get(map_key(map_name)),
                    winner_team_id,
                    rounds_t1,
                    rounds_t2,
                    team1_ct,
                    team1_t,
                    team2_ct,
                    team2_t,
                    overtime_t1,
                    overtime_t2,
                    team1_rating,
                    team2_rating,
                    team1_first_kills,
                    team2_first_kills,
                    team1_clutches,
                    team2_clutches,
                    map_payload.get("mapstats_id"),
                    source_file,
                ),
            )
            counts["maps_rows"] += cur.rowcount
            map_row = cur.execute(
                "SELECT map_id FROM maps WHERE match_id = ? AND map_number = ?",
                (mid, map_number),
            ).fetchone()
            if not map_row:
                continue
            map_id = int(map_row[0])
            total_rounds = None
            if rounds_t1 is not None and rounds_t2 is not None:
                total_rounds = int(rounds_t1) + int(rounds_t2)

            for stat in map_payload.get("player_stats") or []:
                team_id = resolve_team_id(
                    {"hltv_id": stat.get("team_hltv_id"), "name": stat.get("team_name")},
                    team_ids,
                    team_ids_by_hltv,
                )
                if team_id is None:
                    continue
                player_id = get_player_id(cur, stat, player_cache)
                lineup_key = (mid, team_id, player_id)
                if lineup_key not in lineup_seen:
                    cur.execute(
                        "INSERT OR IGNORE INTO match_lineups(match_id, team_id, player_id, is_standin) VALUES (?,?,?,0)",
                        lineup_key,
                    )
                    counts["lineup_rows"] += cur.rowcount
                    lineup_seen.add(lineup_key)
                cur.execute(
                    "INSERT OR IGNORE INTO map_player_side_stats("
                    "map_id, player_id, team_id, side, kills, deaths, assists, adr, kast, rating, round_swing, "
                    "opening_kills, opening_deaths, headshots, flash_assists, multi_kill_rounds, clutches_won, "
                    "traded_deaths, eco_kills, eco_deaths, eco_adr, eco_kast, source_file, payload_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        map_id,
                        player_id,
                        team_id,
                        stat.get("side"),
                        stat.get("kills"),
                        stat.get("deaths"),
                        stat.get("assists"),
                        stat.get("adr"),
                        stat.get("kast"),
                        stat.get("rating"),
                        stat.get("round_swing"),
                        stat.get("opening_kills"),
                        stat.get("opening_deaths"),
                        stat.get("headshots"),
                        stat.get("flash_assists"),
                        stat.get("multi_kill_rounds"),
                        stat.get("clutches_won"),
                        stat.get("traded_deaths"),
                        stat.get("eco_kills"),
                        stat.get("eco_deaths"),
                        stat.get("eco_adr"),
                        stat.get("eco_kast"),
                        source_file,
                        json.dumps(stat, ensure_ascii=False),
                    ),
                )
                counts["map_player_side_stats_rows"] += cur.rowcount
                if stat.get("side") != "total":
                    continue
                kills = stat.get("kills")
                deaths = stat.get("deaths")
                opening_kills = stat.get("opening_kills")
                opening_deaths = stat.get("opening_deaths")
                kpr = kills / total_rounds if kills is not None and total_rounds else None
                dpr = deaths / total_rounds if deaths is not None and total_rounds else None
                fk_diff = None
                if opening_kills is not None and opening_deaths is not None:
                    fk_diff = opening_kills - opening_deaths
                cur.execute(
                    "INSERT OR IGNORE INTO map_player_stats("
                    "map_id, player_id, team_id, kills, deaths, assists, adr, kast, kpr, dpr, rating, "
                    "round_swing, opening_kills, opening_deaths, headshots, flash_assists, multi_kill_rounds, "
                    "clutches_won, traded_deaths, fk_diff, source_file) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        map_id,
                        player_id,
                        team_id,
                        kills,
                        deaths,
                        stat.get("assists"),
                        stat.get("adr"),
                        stat.get("kast"),
                        kpr,
                        dpr,
                        stat.get("rating"),
                        stat.get("round_swing"),
                        opening_kills,
                        opening_deaths,
                        stat.get("headshots"),
                        stat.get("flash_assists"),
                        stat.get("multi_kill_rounds"),
                        stat.get("clutches_won"),
                        stat.get("traded_deaths"),
                        fk_diff,
                        source_file,
                    ),
                )
                counts["map_player_stats_rows"] += cur.rowcount
    return counts


def _analytics_team_value(payload: dict, team_name: str | None, field: str) -> int | None:
    if not team_name:
        return None
    wanted = dataio.clean_team(team_name)
    for team, value in (payload or {}).items():
        if dataio.clean_team(str(team)) != wanted or not isinstance(value, dict):
            continue
        return parse_int(value.get(field))
    return None


def insert_prematch_lineup_snapshots(
    cur: sqlite3.Cursor,
    run_dir: Path,
    match_id_map: dict[str, int],
    team_ids: dict[str, int],
    team_ids_by_hltv: dict[str, int],
) -> dict[str, int]:
    counts = {"prematch_lineup_matches": 0, "prematch_lineup_rows": 0}
    player_cache: dict[str, int] = {}
    for path in sorted((run_dir / "match_snapshots").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hltv_match_id = str(payload.get("id") or path.stem)
        match_id = match_id_map.get(hltv_match_id)
        lineups = payload.get("prematch_lineups") or {}
        if match_id is None or not isinstance(lineups, dict):
            continue
        captured_at = payload.get("captured_at")
        if not captured_at:
            continue
        inserted_for_match = False
        for lineup in lineups.values():
            if not isinstance(lineup, dict):
                continue
            team_id = resolve_team_id(
                {"hltv_id": lineup.get("hltv_team_id"), "name": lineup.get("team_name")},
                team_ids,
                team_ids_by_hltv,
            )
            if team_id is None:
                continue
            for player in lineup.get("players") or []:
                player_hltv_id = str(player.get("hltv_player_id") or "").strip()
                if not player_hltv_id:
                    continue
                player_id = get_player_id(
                    cur,
                    {"hltv_id": player_hltv_id, "nick": player.get("nickname") or player_hltv_id},
                    player_cache,
                )
                cur.execute(
                    "INSERT OR IGNORE INTO prematch_lineup_snapshots("
                    "match_id, team_id, player_id, captured_at_utc, run_id, is_standin, source_file, payload_json"
                    ") VALUES (?,?,?,?,?,?,?,?)",
                    (
                        match_id,
                        team_id,
                        player_id,
                        captured_at,
                        run_dir.name,
                        1 if player.get("is_standin") else 0,
                        payload.get("source_file") or source_name(path),
                        json.dumps(player, ensure_ascii=False),
                    ),
                )
                counts["prematch_lineup_rows"] += cur.rowcount
                inserted_for_match = inserted_for_match or bool(cur.rowcount)
        counts["prematch_lineup_matches"] += int(inserted_for_match)
    return counts


def insert_match_analytics_snapshots(
    cur: sqlite3.Cursor,
    run_dir: Path,
    match_id_map: dict[str, int],
    team_ids: dict[str, int],
    team_ids_by_hltv: dict[str, int],
) -> dict[str, int]:
    counts = {"analytics_snapshot_rows": 0, "analytics_map_stats_rows": 0, "analytics_map_handicap_rows": 0}
    for path in sorted((run_dir / "analytics").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hltv_match_id = str(payload.get("match_id") or path.stem)
        match_id = match_id_map.get(hltv_match_id)
        captured_at = payload.get("captured_at")
        if match_id is None or not captured_at or not payload.get("available"):
            continue
        event = payload.get("event_metadata") or {}
        series = payload.get("series_stats") or {}
        core = payload.get("core_lineup") or {}
        team1 = (series.get("team1") or {}).get("team")
        team2 = (series.get("team2") or {}).get("team")
        source_file = payload.get("source_file") or source_name(path)
        event_row = cur.execute("SELECT event_id FROM matches WHERE match_id=?", (match_id,)).fetchone()
        if event_row:
            cur.execute(
                """
                UPDATE events
                SET prize_pool=COALESCE(?, prize_pool),
                    teams_competing=COALESCE(?, teams_competing),
                    source_captured_at_utc=?, source_file=?
                WHERE event_id=?
                """,
                (
                    parse_int(event.get("prize_pool")),
                    parse_int(event.get("teams_competing")),
                    captured_at,
                    source_file,
                    int(event_row[0]),
                ),
            )
        cur.execute(
            "INSERT OR IGNORE INTO match_analytics_snapshots("
            "match_id, captured_at_utc, run_id, source_file, event_name, hltv_event_id, prize_pool, teams_competing, "
            "team1_core_matches_lt, team2_core_matches_lt, team1_matches_sample, team1_maps_sample, "
            "team2_matches_sample, team2_maps_sample, team1_overtime_pct, team2_overtime_pct, payload_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                match_id,
                captured_at,
                run_dir.name,
                source_file,
                event.get("name"),
                event.get("hltv_event_id"),
                parse_int(event.get("prize_pool")),
                parse_int(event.get("teams_competing")),
                _analytics_team_value(core, team1, "matches_lt"),
                _analytics_team_value(core, team2, "matches_lt"),
                parse_int((series.get("team1") or {}).get("matches")),
                parse_int((series.get("team1") or {}).get("maps")),
                parse_int((series.get("team2") or {}).get("matches")),
                parse_int((series.get("team2") or {}).get("maps")),
                safe_float((series.get("team1") or {}).get("overtime_pct")),
                safe_float((series.get("team2") or {}).get("overtime_pct")),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        counts["analytics_snapshot_rows"] += cur.rowcount
        snapshot_row = cur.execute(
            "SELECT analytics_snapshot_id FROM match_analytics_snapshots "
            "WHERE match_id=? AND captured_at_utc=? AND source_file=?",
            (match_id, captured_at, source_file),
        ).fetchone()
        if not snapshot_row:
            continue
        analytics_snapshot_id = int(snapshot_row[0])
        for row in payload.get("map_stats") or []:
            if not isinstance(row, dict) or not row.get("map"):
                continue
            team_id = resolve_team_id({"name": row.get("team")}, team_ids, team_ids_by_hltv)
            cur.execute(
                "INSERT INTO match_analytics_map_stats("
                "analytics_snapshot_id, team_id, team_name, map_name, first_pick_pct, first_ban_pct, win_pct, played, comment"
                ") VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(analytics_snapshot_id, team_name, map_name) DO UPDATE SET "
                "team_id=COALESCE(match_analytics_map_stats.team_id, excluded.team_id)",
                (
                    analytics_snapshot_id,
                    team_id,
                    row.get("team"),
                    row.get("map"),
                    safe_float(row.get("first_pick_pct")),
                    safe_float(row.get("first_ban_pct")),
                    safe_float(row.get("win_pct")),
                    parse_int(row.get("played")),
                    row.get("comment"),
                ),
            )
            counts["analytics_map_stats_rows"] += cur.rowcount
        for row in payload.get("map_handicap") or []:
            if not isinstance(row, dict) or not row.get("map"):
                continue
            team_id = resolve_team_id({"name": row.get("team")}, team_ids, team_ids_by_hltv)
            cur.execute(
                "INSERT INTO match_analytics_map_handicap("
                "analytics_snapshot_id, team_id, team_name, map_name, avg_rounds_lost_in_wins, avg_rounds_won_in_losses"
                ") VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(analytics_snapshot_id, team_name, map_name) DO UPDATE SET "
                "team_id=COALESCE(match_analytics_map_handicap.team_id, excluded.team_id)",
                (
                    analytics_snapshot_id,
                    team_id,
                    row.get("team"),
                    row.get("map"),
                    safe_float(row.get("avg_rounds_lost_in_wins")),
                    safe_float(row.get("avg_rounds_won_in_losses")),
                ),
            )
            counts["analytics_map_handicap_rows"] += cur.rowcount
    return counts


def insert_team_rankings(cur: sqlite3.Cursor, team_ids_by_hltv: dict[str, int]) -> int:
    runs_dir = DAILY_ROOT / "runs"
    if not runs_dir.exists():
        return 0
    inserted = 0
    for path in sorted(runs_dir.glob("*/rankings/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.parents[1].name
        ranking_type = payload.get("ranking_type") or path.stem
        for item in payload.get("ranking") or []:
            hltv_id = str(item.get("id") or "")
            cur.execute(
                "INSERT OR IGNORE INTO team_ranking_snapshots("
                "team_id, hltv_team_id, ranking_type, ranking_date_text, position, points, run_id, "
                "captured_at_utc, source_file, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    team_ids_by_hltv.get(hltv_id),
                    hltv_id,
                    ranking_type,
                    payload.get("date_text") or payload.get("date"),
                    parse_int(item.get("position")),
                    parse_int(item.get("points")),
                    run_id,
                    payload.get("captured_at"),
                    source_name(path),
                    json.dumps(item, ensure_ascii=False),
                ),
            )
            inserted += cur.rowcount
    return inserted


def source_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def insert_raw_snapshot(
    cur: sqlite3.Cursor,
    *,
    kind: str,
    run_id: str,
    path: Path,
    payload: dict | list,
    captured_at: str | None,
    hltv_match_id: str | None = None,
    hltv_team_id: str | None = None,
    hltv_player_id: str | None = None,
    source_file: str | None = None,
) -> int:
    cur.execute(
        "INSERT OR IGNORE INTO raw_snapshots(kind, hltv_match_id, hltv_team_id, hltv_player_id, "
        "run_id, captured_at_utc, source_file, payload_json) VALUES (?,?,?,?,?,?,?,?)",
        (
            kind,
            hltv_match_id,
            hltv_team_id,
            hltv_player_id,
            run_id,
            captured_at,
            source_file or source_name(path),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    return cur.rowcount


def insert_daily_archives(cur: sqlite3.Cursor) -> dict[str, int]:
    runs_dir = ROOT / "DAILY_SNAPSHOTS" / "runs"
    if not runs_dir.exists():
        return {"raw_snapshots_rows": 0, "player_stat_snapshots_rows": 0}

    raw_count = 0
    player_count = 0
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run_id = run_dir.name
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        run_captured_at = manifest.get("finished_at") or manifest.get("started_at")
        if manifest_path.exists():
            raw_count += insert_raw_snapshot(
                cur,
                kind="run_manifest",
                run_id=run_id,
                path=manifest_path,
                payload=manifest,
                captured_at=run_captured_at,
            )

        upcoming_path = run_dir / "raw" / "upcoming_matches.json"
        if upcoming_path.exists():
            payload = json.loads(upcoming_path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="upcoming_matches",
                run_id=run_id,
                path=upcoming_path,
                payload=payload,
                captured_at=run_captured_at,
            )

        predictions_path = run_dir / "predictions_enriched.json"
        if predictions_path.exists():
            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="predictions_enriched",
                run_id=run_id,
                path=predictions_path,
                payload=payload,
                captured_at=run_captured_at,
            )

        quality_path = run_dir / "data_quality_report.json"
        if quality_path.exists():
            payload = json.loads(quality_path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="data_quality_report",
                run_id=run_id,
                path=quality_path,
                payload=payload,
                captured_at=payload.get("generated_at") or run_captured_at,
            )

        for path in sorted((run_dir / "raw_html").glob("**/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="raw_html",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=payload.get("captured_at") or run_captured_at,
                hltv_match_id=str(payload.get("identifier") or "") if payload.get("kind") in {"match_page", "match_snapshot", "analytics_page"} else None,
            )

        for path in sorted((run_dir / "match_snapshots").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="match_snapshot",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=payload.get("captured_at") or run_captured_at,
                hltv_match_id=str(payload.get("id") or ""),
            )

        for path in sorted((run_dir / "analytics").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="match_analytics",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=payload.get("captured_at") or run_captured_at,
                hltv_match_id=str(payload.get("match_id") or path.stem),
            )

        for path in sorted((run_dir / "match_assets").glob("*/assets.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="match_assets",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=payload.get("captured_at") or run_captured_at,
                hltv_match_id=str(payload.get("match_id") or ""),
            )

        for path in sorted((run_dir / "rankings").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="team_ranking",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=payload.get("captured_at") or run_captured_at,
                source_file=source_name(path),
            )

        team_profiles_path = run_dir / "team_profiles.json"
        team_profiles = json.loads(team_profiles_path.read_text(encoding="utf-8")) if team_profiles_path.exists() else []
        for item in team_profiles:
            team_id = str(item.get("id") or "")
            raw_count += insert_raw_snapshot(
                cur,
                kind="team_profile",
                run_id=run_id,
                path=team_profiles_path,
                payload=item,
                captured_at=run_captured_at,
                hltv_team_id=team_id,
                source_file=f"{source_name(team_profiles_path)}#{team_id}",
            )

        for path in sorted(run_dir.glob("player_compare_stats_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_count += insert_raw_snapshot(
                cur,
                kind="player_compare_stats",
                run_id=run_id,
                path=path,
                payload=payload,
                captured_at=run_captured_at,
            )
            year = parse_int(payload.get("year"))
            for comparison in payload.get("results") or []:
                time_filter = str(comparison.get("time_filter") or payload.get("time_filter") or year or "")
                match_filter = None
                map_filter = None
                for player in comparison.get("players") or []:
                    player_id = str(player.get("id") or "")
                    if not player_id:
                        continue
                    time_filter = str(player.get("time_filter") or time_filter or "")
                    stats = player.get("stats") or {}
                    cur.execute(
                        "INSERT OR IGNORE INTO player_stat_snapshots("
                        "hltv_player_id, player_name, player_slug, player_link, run_id, captured_at_utc, "
                        "season_year, time_filter, match_filter, map_filter, maps, rating, kpr, dpr, apr, kast, "
                        "impact, adr, round_swing, multi_kill_rating, awp_kpr, hs_pct, opening_kpr, opening_dpr, "
                        "flash_assists, source_file, payload_json) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            player_id,
                            player.get("name"),
                            player.get("slug"),
                            player.get("link"),
                            run_id,
                            run_captured_at,
                            year,
                            time_filter,
                            match_filter,
                            map_filter,
                            parse_int(player.get("maps")),
                            stat_float(first_stat_value(stats, "Rating 3.0", "Rating 2.1", "Rating 2.0", "Rating 1.0")),
                            stat_float(stats.get("KPR")),
                            stat_float(stats.get("DPR")),
                            stat_float(stats.get("APR")),
                            stat_float(stats.get("KAST")),
                            stat_float(stats.get("Impact")),
                            stat_float(first_stat_value(stats, "ADR", "Average Damage per Round", "Damage / Round")),
                            stat_float(stats.get("Round Swing")),
                            stat_float(stats.get("Multi-kill rating")),
                            stat_float(stats.get("AWP KPR")),
                            stat_float(stats.get("HS %")),
                            stat_float(stats.get("Opening KPR")),
                            stat_float(stats.get("Opening DPR")),
                            stat_float(stats.get("Flash assists")),
                            source_name(path),
                            json.dumps(player, ensure_ascii=False),
                        ),
                    )
                    player_count += cur.rowcount

    return {"raw_snapshots_rows": raw_count, "player_stat_snapshots_rows": player_count}


def backup_database(db_path: Path, backup_dir: Path | None, mirror_dir: Path | None = None) -> dict[str, str | None]:
    result: dict[str, str | None] = {"backup": None, "mirror_backup": None}
    if backup_dir is None:
        return result
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    target = backup_dir / f"{db_path.stem}_{stamp}{db_path.suffix}"
    shutil.copy2(db_path, target)
    result["backup"] = str(target)
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_target = mirror_dir / target.name
        shutil.copy2(db_path, mirror_target)
        for source in [DEFAULT_MASTER, ROOT / "DAILY_SNAPSHOTS" / "master" / "manifest.json", ROOT / "DAILY_SNAPSHOTS" / "master" / "roster_history.json"]:
            if source.exists():
                shutil.copy2(source, mirror_dir / source.name)
        result["mirror_backup"] = str(mirror_target)
    return result


def _event_id_for(
    cur: sqlite3.Cursor,
    event_cache: dict[str, int],
    name: str | None,
    metadata: dict | None = None,
) -> int:
    """Resuelve el evento y enriquece solo campos observados por HLTV."""
    metadata = metadata if isinstance(metadata, dict) else {}
    event_name = metadata.get("name") or name or "Unknown event"
    hltv_event_id = str(metadata.get("hltv_event_id") or "").strip() or None
    if event_name in event_cache:
        event_id = event_cache[event_name]
    else:
        row = None
        if hltv_event_id:
            row = cur.execute("SELECT event_id FROM events WHERE hltv_event_id = ?", (hltv_event_id,)).fetchone()
        if row is None:
            row = cur.execute("SELECT event_id FROM events WHERE name = ?", (event_name,)).fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO events(name, hltv_event_id, is_lan, prize_pool, teams_competing, source_captured_at_utc, source_file) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    event_name,
                    hltv_event_id,
                    0,
                    parse_int(metadata.get("prize_pool")),
                    parse_int(metadata.get("teams_competing")),
                    metadata.get("captured_at"),
                    metadata.get("source_file"),
                ),
            )
            event_id = int(cur.lastrowid)
        else:
            event_id = int(row[0])
        event_cache[event_name] = event_id

    cur.execute(
        """
        UPDATE events
        SET hltv_event_id=COALESCE(hltv_event_id, ?),
            prize_pool=COALESCE(?, prize_pool),
            teams_competing=COALESCE(?, teams_competing),
            source_captured_at_utc=CASE WHEN ? IS NOT NULL THEN ? ELSE source_captured_at_utc END,
            source_file=CASE WHEN ? IS NOT NULL THEN ? ELSE source_file END
        WHERE event_id=?
        """,
        (
            hltv_event_id,
            parse_int(metadata.get("prize_pool")),
            parse_int(metadata.get("teams_competing")),
            metadata.get("captured_at"),
            metadata.get("captured_at"),
            metadata.get("source_file"),
            metadata.get("source_file"),
            event_id,
        ),
    )
    row = cur.execute("SELECT event_id FROM events WHERE event_id = ?", (event_id,)).fetchone()
    if not row:
        raise RuntimeError(f"No se pudo crear/leer event: {event_name}")
    event_cache[event_name] = int(row[0])
    return int(row[0])


def _team_id_for(
    cur: sqlite3.Cursor,
    team_cache: dict[str, int],
    team_hltv_cache: dict[str, int],
    *,
    name: str,
    hltv_id: str | None = None,
) -> int:
    hltv_key = str(hltv_id or "").strip()
    if hltv_key and hltv_key in team_hltv_cache:
        return team_hltv_cache[hltv_key]
    key = dataio.clean_team(name)
    if key in team_cache:
        return team_cache[key]
    hltv_int = int(hltv_key) if hltv_key.isdigit() else None
    cur.execute(
        "INSERT OR IGNORE INTO teams(name, hltv_id) VALUES (?,?)",
        (name, hltv_int),
    )
    if hltv_int is not None:
        row = cur.execute("SELECT team_id FROM teams WHERE hltv_id = ?", (hltv_int,)).fetchone()
    else:
        row = cur.execute("SELECT team_id FROM teams WHERE lower(name) = lower(?) ORDER BY team_id LIMIT 1", (name,)).fetchone()
    if not row:
        raise RuntimeError(f"No se pudo crear/leer team: {name}")
    team_id = int(row[0])
    team_cache[key] = team_id
    if hltv_key:
        team_hltv_cache[hltv_key] = team_id
    return team_id


def _load_team_caches(conn: sqlite3.Connection) -> tuple[dict[str, int], dict[str, int]]:
    by_name: dict[str, int] = {}
    by_hltv: dict[str, int] = {}
    for team_id, name, hltv_id in conn.execute("SELECT team_id, name, hltv_id FROM teams"):
        by_name[dataio.clean_team(name)] = int(team_id)
        if hltv_id is not None:
            by_hltv[str(hltv_id)] = int(team_id)
    return by_name, by_hltv


def _load_event_cache(conn: sqlite3.Connection) -> dict[str, int]:
    return {str(name): int(event_id) for event_id, name in conn.execute("SELECT event_id, name FROM events")}


def _match_id_for_hltv(conn: sqlite3.Connection, hltv_match_id: str) -> int | None:
    row = conn.execute("SELECT match_id FROM matches WHERE hltv_match_id = ?", (hltv_match_id,)).fetchone()
    return int(row[0]) if row else None


def _recompute_mart(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Recalcula ratings_history/match_features point-in-time para filas con hltv id.

    Se borra solo el mart derivado: los hechos core y snapshots no se tocan. Cada
    feature se emite antes de observar el resultado del propio partido.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM ratings_history")
    cur.execute("DELETE FROM match_features")
    state = ChronologicalState()
    n_ratings = 0
    n_features = 0
    for row in rows:
        match_id = int(row["db_match_id"]) if row.get("db_match_id") else None
        if match_id is None:
            hltv_id = str(row.get("id") or "")
            match_id = _match_id_for_hltv(conn, hltv_id)
        if match_id is None:
            continue
        date_obj = row.get("date_obj")
        period = _period_index(date_obj)
        state._advance_to(period)
        for key_name, id_name in (("team1_key", "team1_id"), ("team2_key", "team2_id")):
            team_key = row[key_name]
            db_team_id_row = conn.execute(
                "SELECT team1_id, team2_id FROM matches WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            if not db_team_id_row:
                continue
            db_team_id = int(db_team_id_row[0] if key_name == "team1_key" else db_team_id_row[1])
            card = state.rating_card(team_key, date_obj)
            cur.execute(
                "INSERT OR REPLACE INTO ratings_history(entity_type, entity_id, before_match_id, "
                "as_of_date, rating, rd, sigma) VALUES ('team',?,?,?,?,?,?)",
                (db_team_id, match_id, row["date"], card["glicko_rating"], card["glicko_rd"], card["glicko_sigma"]),
            )
            n_ratings += 1
            fr = state.team_feature_row(team_key, date_obj)
            cur.execute(
                "INSERT OR REPLACE INTO match_features(match_id, team_id, data_up_to_utc, glicko_rating, "
                "glicko_rd, glicko_sigma, elo, form_winrate_10, form_winrate_20, winrate_overall, "
                "avg_score_diff, recent_opp_elo, streak, matches_played, days_since_last) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    match_id, db_team_id, row["date"], fr["glicko_rating"], fr["glicko_rd"], fr["glicko_sigma"],
                    fr["elo"], fr["form_winrate_10"], fr["form_winrate_20"], fr["winrate_overall"],
                    fr["avg_score_diff"], fr["recent_opp_elo"], fr["streak"], fr["matches_played"],
                    fr["days_since_last"],
                ),
            )
            n_features += 1
        state.observe(row)
    return {"ratings_history_rows": n_ratings, "match_features_rows": n_features}


def seed_database_once(
    raw_path: Path,
    db_path: Path,
    master_path: Path | None = DEFAULT_MASTER,
    backup_dir: Path | None = DEFAULT_BACKUP_DIR,
    mirror_backup_dir: Path | None = DEFAULT_MIRROR_BACKUP_DIR,
) -> dict[str, Any]:
    conn = connect_live_db(db_path)
    before_count = table_count(conn, "matches")
    if before_count > 0:
        conn.close()
        stats = {
            "db": str(db_path),
            "mode": "already_seeded",
            "matches": before_count,
            "note": "BBDD viva ya contiene matches; no se ejecuta semilla ni reconstruccion.",
        }
        return stats

    started = utcnow()
    rows = dataio.load_results(raw_path, cs2_only=True)
    master = load_master(master_path)
    match_contexts = build_match_context_index(master)
    raw_payload = dataio.read_json(raw_path, [])
    cur = conn.cursor()
    team_cache, team_hltv_cache = _load_team_caches(conn)
    event_cache = _load_event_cache(conn)
    now = utcnow()
    counts: dict[str, int] = defaultdict(int)

    for item in raw_payload:
        cur.execute(
            "INSERT OR IGNORE INTO raw_results(hltv_match_id, source_file, ingested_at_utc, payload_json) "
            "VALUES (?,?,?,?)",
            (str(item.get("id")), str(raw_path.name), now, json.dumps(item, ensure_ascii=False)),
        )
        counts["raw_results"] += cur.rowcount

    for row in rows:
        event_id = _event_id_for(cur, event_cache, row.get("event"))
        team1_id = _team_id_for(cur, team_cache, team_hltv_cache, name=row["team1"], hltv_id=row.get("team1_id"))
        team2_id = _team_id_for(cur, team_cache, team_hltv_cache, name=row["team2"], hltv_id=row.get("team2_id"))
        context = match_contexts.get(str(row.get("id"))) or asset_match_context(row.get("asset"))
        environment = context.get("environment") if context and context.get("environment") in {"lan", "online"} else "unknown"
        stage = schema_stage(context.get("stage")) if context else None
        bracket = context.get("bracket") if context and context.get("bracket") in {"upper", "lower"} else None
        winner_team_id = team1_id if row["team1_win"] else team2_id
        cur.execute(
            "INSERT INTO matches(hltv_match_id, event_id, datetime_utc, team1_id, team2_id, best_of, "
            "stage, environment, stage_detail, incentive_label, high_stakes, opening_match, "
            "winner_advances, loser_eliminated, bracket, context_json, status, data_tier, "
            "prematch_captured_at_utc, result_filled_at_utc, has_context, winner_team_id, score_t1, score_t2) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(hltv_match_id) DO NOTHING",
            (
                str(row["id"]),
                event_id,
                row["date"],
                team1_id,
                team2_id,
                {"bo1": 1, "bo3": 3, "bo5": 5}.get(row["format"], 3),
                stage,
                environment,
                context.get("stage_detail") if context else None,
                context.get("incentive_label") if context else None,
                1 if context and context.get("high_stakes") else 0,
                1 if context and context.get("opening_match") else 0,
                1 if context and context.get("winner_advances") else 0,
                1 if context and context.get("loser_eliminated") else 0,
                bracket,
                json.dumps(context, ensure_ascii=False) if context else None,
                "completed",
                "historical_seed",
                None,
                now,
                0,
                winner_team_id,
                row["score1"],
                row["score2"],
            ),
        )
        counts["matches"] += cur.rowcount

    conn.commit()
    team_cache, team_hltv_cache = _load_team_caches(conn)
    n_rosters = insert_roster_history(cur, team_hltv_cache)
    match_id_map = {
        str(hltv_id): int(match_id)
        for match_id, hltv_id in conn.execute("SELECT match_id, hltv_match_id FROM matches WHERE hltv_match_id IS NOT NULL")
    }
    match_teams = {
        int(match_id): (int(team1_id), int(team2_id))
        for match_id, team1_id, team2_id in conn.execute("SELECT match_id, team1_id, team2_id FROM matches")
    }
    n_odds = insert_odds(cur, master, match_id_map)
    asset_stats = insert_hltv_assets(cur, master, match_id_map, match_teams, team_cache, team_hltv_cache)
    n_rankings = insert_team_rankings(cur, team_hltv_cache)
    archive_stats = insert_daily_archives(cur)
    mart_stats = _recompute_mart(conn, rows)
    conn.commit()

    finished = utcnow()
    rows_upserted = {
        "raw_results": counts["raw_results"],
        "matches": counts["matches"],
        "team_rosters_rows": n_rosters,
        "odds_rows": n_odds,
        "team_ranking_snapshots_rows": n_rankings,
        **asset_stats,
        **archive_stats,
        **mart_stats,
    }
    cur.execute(
        "INSERT INTO ingest_runs(run_id, started_at_utc, finished_at_utc, status, rows_upserted_json, "
        "requests_made, requests_skipped_by_freshness, note) VALUES (?,?,?,?,?,?,?,?)",
        (
            "historical_seed",
            started,
            finished,
            "ok",
            json.dumps(rows_upserted, ensure_ascii=False),
            0,
            0,
            "Semilla inicial idempotente desde results_all.json y master JSON disponible.",
        ),
    )
    conn.commit()
    stats = {
        "db": str(db_path),
        "mode": "seeded",
        "raw_rows": len(raw_payload),
        "teams": table_count(conn, "teams"),
        "events": table_count(conn, "events"),
        "matches": table_count(conn, "matches"),
        **rows_upserted,
    }
    conn.close()
    stats.update(backup_database(db_path, backup_dir, mirror_backup_dir))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Inicializa/siembra la BBDD viva SQLite del predictor CS2.")
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--mirror-backup-dir", default=str(DEFAULT_MIRROR_BACKUP_DIR))
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-mirror-backup", action="store_true")
    args = parser.parse_args()
    backup_dir = None if args.no_backup else Path(args.backup_dir)
    mirror_dir = None if args.no_backup or args.no_mirror_backup else Path(args.mirror_backup_dir)
    stats = seed_database_once(
        Path(args.raw),
        Path(args.db),
        Path(args.master) if args.master else None,
        backup_dir,
        mirror_dir,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

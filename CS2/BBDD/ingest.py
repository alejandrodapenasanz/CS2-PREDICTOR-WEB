"""Ingest incremental de runs diarios hacia la BBDD viva SQLite.

Este modulo no scrapea. Lee la evidencia cruda ya guardada en
PIPELINE/runs/<run_id>/ y hace upserts idempotentes en BBDD/cs2.db.
La BBDD conserva la foto PRE-MATCH y, cuando aparece el resultado, solo rellena
el bloque RESULT.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from BBDD import build_db
from cs2model import dataio
from PIPELINE.match_context import parse_match_context_meta, schema_stage
from PIPELINE.opportunity import (
    MIN_DECISION_CONFIDENCE,
    MIN_RELIABILITY,
    POLICY_VERSION as OPPORTUNITY_POLICY_VERSION,
    is_opportunity_eligible,
    opportunity_score,
)


DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_MASTER = ROOT / "PIPELINE" / "master" / "matches.json"
DEFAULT_RUNS = ROOT / "PIPELINE" / "runs"
DEFAULT_BACKUP_DIR = ROOT / "BBDD" / "backups"
DEFAULT_MIRROR_BACKUP_DIR = build_db.DEFAULT_MIRROR_BACKUP_DIR

TTL_DAYS = {
    "team_profile": int(os.environ.get("BBDD_TEAM_PROFILE_TTL_DAYS", "7")),
    "player_stats": int(os.environ.get("BBDD_PLAYER_STATS_TTL_DAYS", "3")),
    "ranking_hltv": int(os.environ.get("BBDD_RANKING_TTL_DAYS", "7")),
    "ranking_valve": int(os.environ.get("BBDD_RANKING_TTL_DAYS", "7")),
}
PLAYER_CORE_STATS = ("Rating 3.0", "KPR", "DPR", "APR", "KAST", "Impact", "ADR")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def player_stat_value_present(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "-", "--", "n/a", "na"}


def player_snapshot_is_complete(player: dict[str, Any]) -> bool:
    stats = player.get("stats") or {}
    return all(player_stat_value_present(stats.get(key)) for key in PLAYER_CORE_STATS)


def player_snapshot_fetch_status(player: dict[str, Any]) -> str:
    if player_snapshot_is_complete(player):
        return "ok"
    try:
        if int(player.get("maps")) <= 0:
            # Schema-compatible: HLTV returned the player page but no usable
            # sample exists for the selected window.
            return "not_found"
    except (TypeError, ValueError):
        pass
    return "partial"


def parse_datetime(date_text: str | None, hour_text: str | None = None) -> str | None:
    if not date_text:
        return None
    if "T" in str(date_text):
        return str(date_text)
    hour = "12:00"
    if hour_text:
        import re

        match = re.search(r"(\d{1,2}):(\d{2})", str(hour_text))
        if match:
            hour = f"{int(match.group(1)):02d}:{match.group(2)}"
    return f"{date_text}T{hour}:00Z"


def next_eligible(entity_type: str, status: str, fetched_at: str) -> str | None:
    if status in {"blocked", "error", "partial"}:
        delta = timedelta(hours=1)
    elif entity_type in {"match_assets", "match_analytics"}:
        delta = timedelta(days=3650)
    else:
        delta = timedelta(days=TTL_DAYS.get(entity_type, 1))
    base = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def upsert_fetch_state(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_key: str,
    status: str,
    *,
    fetched_at: str | None = None,
    note: str | None = None,
) -> int:
    if not entity_key:
        return 0
    fetched_at = fetched_at or utcnow()
    eligible = next_eligible(entity_type, status, fetched_at)
    cur = conn.execute(
        """
        INSERT INTO fetch_state(entity_type, entity_key, last_fetched_at_utc, last_status,
                                fetch_count, next_eligible_at_utc, note)
        VALUES (?,?,?,?,1,?,?)
        ON CONFLICT(entity_type, entity_key) DO UPDATE SET
            last_fetched_at_utc=excluded.last_fetched_at_utc,
            last_status=excluded.last_status,
            fetch_count=CASE
                WHEN fetch_state.last_fetched_at_utc=excluded.last_fetched_at_utc
                 AND COALESCE(fetch_state.last_status,'')=COALESCE(excluded.last_status,'')
                 AND COALESCE(fetch_state.note,'')=COALESCE(excluded.note,'')
                THEN fetch_state.fetch_count
                ELSE fetch_state.fetch_count + 1
            END,
            next_eligible_at_utc=excluded.next_eligible_at_utc,
            note=excluded.note
        """,
        (entity_type, str(entity_key), fetched_at, status, eligible, note),
    )
    return cur.rowcount


def entity_is_fresh(conn: sqlite3.Connection, entity_type: str, entity_key: str, now: str | None = None) -> bool:
    now = now or utcnow()
    row = conn.execute(
        "SELECT next_eligible_at_utc, last_status FROM fetch_state WHERE entity_type=? AND entity_key=?",
        (entity_type, str(entity_key)),
    ).fetchone()
    if not row or not row[0]:
        return False
    return str(row[1]) in {"ok", "partial", "blocked", "not_found", "error"} and str(row[0]) > now


def team_from_record(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    direct = record.get(side)
    if isinstance(direct, dict) and direct.get("name"):
        return direct
    detail = record.get("detail") or {}
    match = detail.get("match") or {}
    direct = match.get(side)
    if isinstance(direct, dict) and direct.get("name"):
        return direct
    return None


def context_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    context = record.get("match_context")
    if isinstance(context, dict) and context:
        return context
    detail = record.get("detail") or {}
    meta = ((detail.get("match") or {}).get("maps_box_meta") or detail.get("maps_box_meta"))
    if meta:
        try:
            return parse_match_context_meta(meta)
        except Exception:
            return None
    return None


def upsert_match(conn: sqlite3.Connection, record: dict[str, Any], captured_at: str | None) -> int:
    hltv_id = str(record.get("id") or "")
    if not hltv_id:
        return 0
    team1 = team_from_record(record, "team1")
    team2 = team_from_record(record, "team2")
    if not team1 or not team2:
        return 0
    cur = conn.cursor()
    team_cache, team_hltv_cache = build_db._load_team_caches(conn)
    event_cache = build_db._load_event_cache(conn)
    team1_id = build_db._team_id_for(cur, team_cache, team_hltv_cache, name=team1["name"], hltv_id=team1.get("id"))
    team2_id = build_db._team_id_for(cur, team_cache, team_hltv_cache, name=team2["name"], hltv_id=team2.get("id"))
    analytics = record.get("analytics") or {}
    event_metadata = dict(record.get("event_metadata") or {})
    if isinstance(analytics, dict) and isinstance(analytics.get("event_metadata"), dict):
        event_metadata.update({key: value for key, value in analytics["event_metadata"].items() if value is not None})
        event_metadata.setdefault("captured_at", analytics.get("captured_at"))
        event_metadata.setdefault("source_file", analytics.get("source_file"))
    event_id = build_db._event_id_for(cur, event_cache, record.get("event"), event_metadata)
    context = context_from_record(record) or {}
    environment = context.get("environment") if context.get("environment") in {"lan", "online"} else "unknown"
    stage = schema_stage(context.get("stage")) if context else None
    bracket = context.get("bracket") if context.get("bracket") in {"upper", "lower"} else None
    score = record.get("score") or {}
    s1 = build_db.parse_int(score.get("team1"))
    s2 = build_db.parse_int(score.get("team2"))
    completed = record.get("status") == "completed" and s1 is not None and s2 is not None and s1 != s2
    winner_team_id = team1_id if completed and s1 > s2 else team2_id if completed else None
    status = "completed" if completed else "scheduled"
    data_tier = "completed" if completed else "prematch_captured"
    best_of = {"bo1": 1, "bo2": 2, "bo3": 3, "bo5": 5}.get(str(record.get("format") or "").lower(), 3)
    has_odds = 1 if record.get("opening_odds") or ((record.get("odds") or {}).get("available")) else 0
    has_analytics = 1 if record.get("analytics") or record.get("latest_analytics_file") else 0
    has_context = 1 if context else 0
    dt = parse_datetime(record.get("date"), record.get("hour")) or record.get("date")
    cur.execute(
        """
        INSERT INTO matches(
            hltv_match_id, event_id, datetime_utc, team1_id, team2_id, best_of,
            stage, environment, stage_detail, incentive_label, high_stakes, opening_match,
            winner_advances, loser_eliminated, bracket, context_json, status, data_tier,
            prematch_captured_at_utc, result_filled_at_utc, has_prematch_odds, has_analytics,
            has_context, winner_team_id, score_t1, score_t2
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(hltv_match_id) DO UPDATE SET
            -- Antes de empezar HLTV puede corregir/renombrar un participante,
            -- hora o fase. El hecho actual se actualiza; las fotos raw y la
            -- primera cuota permanecen append-only en sus propias tablas.
            event_id=CASE WHEN matches.status <> 'completed' THEN excluded.event_id ELSE matches.event_id END,
            datetime_utc=CASE WHEN matches.status <> 'completed' THEN excluded.datetime_utc ELSE matches.datetime_utc END,
            team1_id=CASE WHEN matches.status <> 'completed' THEN excluded.team1_id ELSE matches.team1_id END,
            team2_id=CASE WHEN matches.status <> 'completed' THEN excluded.team2_id ELSE matches.team2_id END,
            best_of=CASE WHEN matches.status <> 'completed' THEN excluded.best_of ELSE matches.best_of END,
            stage=CASE WHEN matches.status <> 'completed' THEN excluded.stage ELSE matches.stage END,
            environment=CASE WHEN matches.status <> 'completed' THEN excluded.environment ELSE matches.environment END,
            stage_detail=CASE WHEN matches.status <> 'completed' THEN excluded.stage_detail ELSE matches.stage_detail END,
            incentive_label=CASE WHEN matches.status <> 'completed' THEN excluded.incentive_label ELSE matches.incentive_label END,
            high_stakes=CASE WHEN matches.status <> 'completed' THEN excluded.high_stakes ELSE matches.high_stakes END,
            opening_match=CASE WHEN matches.status <> 'completed' THEN excluded.opening_match ELSE matches.opening_match END,
            winner_advances=CASE WHEN matches.status <> 'completed' THEN excluded.winner_advances ELSE matches.winner_advances END,
            loser_eliminated=CASE WHEN matches.status <> 'completed' THEN excluded.loser_eliminated ELSE matches.loser_eliminated END,
            bracket=CASE WHEN matches.status <> 'completed' THEN excluded.bracket ELSE matches.bracket END,
            context_json=CASE WHEN matches.status <> 'completed' THEN excluded.context_json ELSE matches.context_json END,
            status=CASE WHEN excluded.status='completed' THEN 'completed' ELSE matches.status END,
            data_tier=CASE WHEN excluded.status='completed' THEN 'completed' ELSE matches.data_tier END,
            result_filled_at_utc=COALESCE(matches.result_filled_at_utc, excluded.result_filled_at_utc),
            has_prematch_odds=CASE
                WHEN excluded.status='completed' THEN matches.has_prematch_odds
                ELSE MAX(matches.has_prematch_odds, excluded.has_prematch_odds)
            END,
            has_analytics=CASE
                WHEN excluded.status='completed' THEN matches.has_analytics
                ELSE MAX(matches.has_analytics, excluded.has_analytics)
            END,
            has_context=CASE
                WHEN excluded.status='completed' THEN matches.has_context
                ELSE MAX(matches.has_context, excluded.has_context)
            END,
            winner_team_id=COALESCE(matches.winner_team_id, excluded.winner_team_id),
            score_t1=COALESCE(matches.score_t1, excluded.score_t1),
            score_t2=COALESCE(matches.score_t2, excluded.score_t2)
        """,
        (
            hltv_id,
            event_id,
            dt,
            team1_id,
            team2_id,
            best_of,
            stage,
            environment,
            context.get("stage_detail"),
            context.get("incentive_label"),
            1 if context.get("high_stakes") else 0,
            1 if context.get("opening_match") else 0,
            1 if context.get("winner_advances") else 0,
            1 if context.get("loser_eliminated") else 0,
            bracket,
            json.dumps(context, ensure_ascii=False) if context else None,
            status,
            data_tier,
            captured_at if not completed else None,
            utcnow() if completed else None,
            has_odds,
            has_analytics,
            has_context,
            winner_team_id,
            s1,
            s2,
        ),
    )
    return cur.rowcount


def insert_predictions(conn: sqlite3.Connection, run_dir: Path) -> int:
    path = run_dir / "predictions_enriched.json"
    payload = read_json(path, [])
    if not isinstance(payload, list):
        return 0
    version = f"glicko2+model@{run_dir.name}"
    inserted = 0
    for row in payload:
        pred = row.get("prediction") or {}
        prob = pred.get("model_prob_team1")
        if prob is None:
            continue
        hltv_id = str(row.get("id") or "")
        match_id = build_db._match_id_for_hltv(conn, hltv_id)
        match_row = conn.execute(
            "SELECT team1_id, team2_id FROM matches WHERE match_id=?",
            (match_id,),
        ).fetchone()
        decision_prob = pred.get("decision_prob_team1")
        if decision_prob is None:
            decision_prob = pred.get("risk_adjusted_prob_team1")
        if decision_prob is None:
            decision_prob = prob
        decision_prob = float(decision_prob)
        decision_confidence = float(
            pred.get("decision_confidence")
            if pred.get("decision_confidence") is not None
            else max(decision_prob, 1.0 - decision_prob)
        )
        reliability = float(pred.get("reliability_score") or 0.0)
        favorite_side = pred.get("decision_favorite_side") or (
            "team1" if decision_prob >= 0.5 else "team2"
        )
        favorite_team_id = None
        if match_row is not None:
            favorite_team_id = match_row[0] if favorite_side == "team1" else match_row[1]
        context = ((row.get("controls") or {}).get("tournament_context") or {})
        policy = ((row.get("controls") or {}).get("decision_policy") or {})
        context_environment = str(context.get("environment") or "unknown").lower()
        if context_environment not in {"lan", "online", "unknown"}:
            context_environment = "unknown"
        context_bracket = context.get("bracket")
        if context_bracket not in {"upper", "lower"}:
            context_bracket = None
        opp_score = pred.get("opportunity_score")
        if opp_score is None:
            opp_score = opportunity_score(decision_confidence, reliability)
        opp_eligible = pred.get("opportunity_eligible")
        if opp_eligible is None:
            opp_eligible = is_opportunity_eligible(decision_confidence, reliability)
        team1_min_odds = 1.0 / decision_prob if decision_prob > 0.0 else None
        team2_min_odds = 1.0 / (1.0 - decision_prob) if decision_prob < 1.0 else None
        decision_min_odds = team1_min_odds if favorite_side == "team1" else team2_min_odds
        values = {
            "match_id": match_id,
            "hltv_match_id": hltv_id,
            "model_version": version,
            "predicted_at_utc": row.get("captured_at") or utcnow(),
            "prob_team1": prob,
            "odds_prob_team1": pred.get("odds_prob_team1"),
            "blended_prob_team1": pred.get("blended_prob_team1"),
            "risk_adjusted_prob_team1": pred.get("risk_adjusted_prob_team1"),
            "decision_prob_team1": decision_prob,
            "decision_confidence": decision_confidence,
            "decision_probability_source": pred.get("decision_probability_source"),
            "decision_market_weight": pred.get("decision_market_weight"),
            "decision_market_weight_reasons": json.dumps(
                pred.get("decision_market_weight_reasons"), ensure_ascii=False
            ),
            "decision_policy_json": json.dumps(policy, ensure_ascii=False),
            "reliability_score": reliability,
            "opportunity_score": opp_score,
            "opportunity_eligible": int(bool(opp_eligible)),
            "opportunity_rank": pred.get("opportunity_rank"),
            "opportunity_rank_day": pred.get("opportunity_rank_day"),
            "is_best_opportunity": int(bool(pred.get("is_best_opportunity"))),
            "opportunity_policy_version": (
                pred.get("opportunity_policy_version") or OPPORTUNITY_POLICY_VERSION
            ),
            "opportunity_min_confidence": (
                pred.get("opportunity_min_confidence") or MIN_DECISION_CONFIDENCE
            ),
            "opportunity_min_reliability": (
                pred.get("opportunity_min_reliability") or MIN_RELIABILITY
            ),
            "decision_edge_team1_vs_market": pred.get("decision_edge_team1_vs_market"),
            "favorite_team_id": favorite_team_id,
            "favorite_name": pred.get("decision_favorite"),
            "decision_favorite_side": favorite_side,
            "decision_min_value_odds": decision_min_odds,
            "team1_min_value_odds": team1_min_odds,
            "team2_min_value_odds": team2_min_odds,
            "confidence": pred.get("confidence"),
            "context_environment": context_environment,
            "context_stage": context.get("stage"),
            "context_stage_detail": context.get("stage_detail"),
            "context_incentive_label": context.get("incentive_label"),
            "context_high_stakes": int(bool(context.get("high_stakes"))),
            "context_winner_advances": int(bool(context.get("winner_advances"))),
            "context_loser_eliminated": int(bool(context.get("loser_eliminated"))),
            "context_opening_match": int(bool(context.get("opening_match"))),
            "context_bracket": context_bracket,
            "context_json": json.dumps(context, ensure_ascii=False),
            "prediction_json": json.dumps(pred, ensure_ascii=False),
            "features_json": json.dumps(row.get("features"), ensure_ascii=False),
            "odds_json": json.dumps(row.get("odds"), ensure_ascii=False),
            "staking_json": json.dumps(row.get("staking"), ensure_ascii=False),
            "controls_json": json.dumps(row.get("controls"), ensure_ascii=False),
            "flags_json": json.dumps(row.get("flags"), ensure_ascii=False),
            "data_quality_json": json.dumps(row.get("data_quality"), ensure_ascii=False),
            "rosters_json": json.dumps(row.get("rosters"), ensure_ascii=False),
        }
        columns = list(values)
        update_columns = [
            column
            for column in columns
            if column not in {"match_id", "hltv_match_id", "model_version"}
        ]
        conn.execute(
            f"""
            INSERT INTO predictions({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            ON CONFLICT(hltv_match_id, model_version) DO UPDATE SET
                {", ".join(f"{column}=excluded.{column}" for column in update_columns)}
            """,
            tuple(values[column] for column in columns),
        )
        inserted += 1
    build_db.backfill_prediction_opportunities(conn)
    return inserted


def _prediction_has_player_snapshot(row: dict[str, Any]) -> bool:
    features = row.get("features") or {}
    try:
        coverage1 = float(features.get("player_coverage_team1") or 0.0)
        coverage2 = float(features.get("player_coverage_team2") or 0.0)
    except (TypeError, ValueError):
        coverage1 = coverage2 = 0.0
    if min(coverage1, coverage2) > 0:
        return True

    rosters = row.get("rosters") or {}
    for side in ("team1", "team2"):
        players = ((rosters.get(side) or {}).get("players") or [])
        if not any((player.get("stats") or {}).get("stats") for player in players):
            return False
    return True


def update_player_snapshot_flags(conn: sqlite3.Connection, run_dir: Path) -> int:
    payload = read_json(run_dir / "predictions_enriched.json", [])
    if not isinstance(payload, list):
        return 0
    touched = 0
    for row in payload:
        hltv_id = str(row.get("id") or "")
        if not hltv_id or not _prediction_has_player_snapshot(row):
            continue
        cur = conn.execute(
            """
            UPDATE matches
            SET has_player_snapshot = 1
            WHERE hltv_match_id = ?
              AND data_tier <> 'historical_seed'
              AND has_player_snapshot = 0
            """,
            (hltv_id,),
        )
        touched += cur.rowcount
    return touched


def update_fetch_state_from_run(conn: sqlite3.Connection, run_dir: Path) -> int:
    touched = 0
    manifest = read_json(run_dir / "manifest.json", {})
    captured_at = manifest.get("finished_at") or manifest.get("started_at") or utcnow()
    for item in read_json(run_dir / "team_profiles.json", []):
        team_id = str(item.get("id") or "")
        status = "ok" if item else "partial"
        touched += upsert_fetch_state(conn, "team_profile", team_id, status, fetched_at=captured_at)
    for path in sorted(run_dir.glob("player_compare_stats_*.json")):
        payload = read_json(path, {})
        player_states: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for comparison in payload.get("results") or []:
            for player in comparison.get("players") or []:
                player_id = str(player.get("id") or "")
                if not player_id or player.get("fetch_origin") == "cache":
                    continue
                status = player_snapshot_fetch_status(player)
                player_states[player_id].append((status, player.get("captured_at")))
        for player_id, states in player_states.items():
            statuses = {status for status, _ in states}
            status = "ok" if "ok" in statuses else "partial" if "partial" in statuses else "not_found"
            player_captured_at = max(
                (value for _, value in states if value),
                default=captured_at,
            )
            touched += upsert_fetch_state(
                conn,
                "player_stats",
                player_id,
                status,
                fetched_at=player_captured_at,
            )
    rankings = read_json(run_dir / "rankings_index.json", {})
    for key, entity_type in {"hltv": "ranking_hltv", "valve": "ranking_valve"}.items():
        state = rankings.get(key) or {}
        status = "ok" if state.get("ok") else "error" if state else "partial"
        touched += upsert_fetch_state(conn, entity_type, "global", status, fetched_at=state.get("captured_at") or captured_at)
    assets = read_json(run_dir / "match_assets_index.json", {})
    for item in assets.get("index") or []:
        status = "ok" if item.get("status") == "ok" else "partial" if item.get("status") == "partial" else "error"
        touched += upsert_fetch_state(conn, "match_assets", str(item.get("id") or ""), status, fetched_at=item.get("captured_at") or captured_at)
    for path in sorted((run_dir / "analytics").glob("*.json")):
        payload = read_json(path, {})
        status = "ok" if payload.get("available") else "partial"
        touched += upsert_fetch_state(conn, "match_analytics", str(payload.get("match_id") or path.stem), status, fetched_at=payload.get("captured_at") or captured_at)
    for path in sorted((run_dir / "match_snapshots").glob("*.json")):
        payload = read_json(path, {})
        touched += upsert_fetch_state(conn, "match_detail", str(payload.get("id") or path.stem), "ok", fetched_at=payload.get("captured_at") or captured_at)
    return touched


def reconcile_known_missing_player_stats(conn: sqlite3.Connection) -> int:
    """Reconcile one player state from all windows in its latest capture."""
    rows = conn.execute(
        """
        WITH latest_capture AS (
            SELECT hltv_player_id, MAX(captured_at_utc) AS captured_at_utc
            FROM player_stat_snapshots
            GROUP BY hltv_player_id
        ),
        latest AS (
            SELECT p.*
            FROM player_stat_snapshots
            p JOIN latest_capture l
              ON l.hltv_player_id=p.hltv_player_id
             AND l.captured_at_utc=p.captured_at_utc
        )
        SELECT hltv_player_id, MAX(captured_at_utc) AS captured_at_utc,
               MAX(CASE WHEN maps > 0
                         AND rating IS NOT NULL AND kpr IS NOT NULL AND dpr IS NOT NULL
                         AND apr IS NOT NULL AND kast IS NOT NULL
                         AND impact IS NOT NULL AND adr IS NOT NULL
                        THEN 1 ELSE 0 END) AS has_complete,
               MAX(COALESCE(maps, 0)) AS max_maps
        FROM latest
        GROUP BY hltv_player_id
        """
    ).fetchall()
    touched = 0
    for player_id, captured_at, has_complete, max_maps in rows:
        state = conn.execute(
            "SELECT last_status FROM fetch_state WHERE entity_type='player_stats' AND entity_key=?",
            (str(player_id),),
        ).fetchone()
        if not state:
            continue
        current = str(state[0]) if state else None
        derived = "ok" if has_complete else "not_found" if max_maps <= 0 else "partial"
        # Promote false legacy partial/not_found states when another window from
        # the same capture is complete. Never downgrade historical OK rows here.
        if current == derived or (current == "ok" and derived != "ok"):
            continue
        touched += upsert_fetch_state(
            conn,
            "player_stats",
            str(player_id),
            derived,
            fetched_at=captured_at or utcnow(),
            note=(
                "Reconciled from all player windows in the latest capture"
                if derived == "ok"
                else "HLTV returned no usable stats sample for the selected player window"
                if derived == "not_found"
                else "Latest player capture is incomplete"
            ),
        )
    return touched


def ingest_run(
    run_dir: Path,
    db_path: Path = DEFAULT_DB,
    master_path: Path = DEFAULT_MASTER,
    *,
    requests_made: int | None = None,
    requests_skipped_by_freshness: int | None = None,
    backup_dir: Path | None = DEFAULT_BACKUP_DIR,
    mirror_backup_dir: Path | None = DEFAULT_MIRROR_BACKUP_DIR,
) -> dict[str, Any]:
    started = utcnow()
    conn = build_db.connect_live_db(db_path)
    manifest = read_json(run_dir / "manifest.json", {})
    run_id = run_dir.name
    master = read_json(master_path, {})
    counts: dict[str, int] = defaultdict(int)
    status = "ok"
    note = ""
    try:
        for record in master.values():
            counts["matches"] += upsert_match(conn, record, manifest.get("started_at"))
            conn.commit()

        match_id_map = {
            str(hltv_id): int(match_id)
            for match_id, hltv_id in conn.execute("SELECT match_id, hltv_match_id FROM matches WHERE hltv_match_id IS NOT NULL")
        }
        team_ids = {
            dataio.clean_team(name): int(team_id)
            for team_id, name in conn.execute("SELECT team_id, name FROM teams")
        }
        team_ids_by_hltv = {
            str(hltv_id): int(team_id)
            for team_id, hltv_id in conn.execute("SELECT team_id, hltv_id FROM teams WHERE hltv_id IS NOT NULL")
        }
        match_teams = {
            int(match_id): (int(team1_id), int(team2_id))
            for match_id, team1_id, team2_id in conn.execute("SELECT match_id, team1_id, team2_id FROM matches")
        }
        counts["odds_rows"] += build_db.insert_odds(conn.cursor(), master, match_id_map)
        counts["roster_history_rows"] += build_db.insert_roster_history(conn.cursor(), team_ids_by_hltv)
        prematch_lineup_counts = build_db.insert_prematch_lineup_snapshots(
            conn.cursor(),
            run_dir,
            match_id_map,
            team_ids,
            team_ids_by_hltv,
        )
        for key, value in prematch_lineup_counts.items():
            counts[key] += value
        analytics_counts = build_db.insert_match_analytics_snapshots(
            conn.cursor(),
            run_dir,
            match_id_map,
            team_ids,
            team_ids_by_hltv,
        )
        for key, value in analytics_counts.items():
            counts[key] += value
        asset_counts = build_db.insert_hltv_assets(
            conn.cursor(),
            master,
            match_id_map,
            match_teams,
            team_ids,
            team_ids_by_hltv,
        )
        for key, value in asset_counts.items():
            counts[key] += value
        counts["team_ranking_snapshots_rows"] += build_db.insert_team_rankings(conn.cursor(), team_ids_by_hltv)
        conn.execute(
            """
            UPDATE matches
            SET has_box_score = CASE WHEN EXISTS (
                    SELECT 1 FROM maps WHERE maps.match_id = matches.match_id
                ) AND matches.data_tier <> 'historical_seed' THEN 1 ELSE has_box_score END,
                has_veto = CASE WHEN EXISTS (
                    SELECT 1 FROM veto WHERE veto.match_id = matches.match_id
                ) AND matches.data_tier <> 'historical_seed' THEN 1 ELSE has_veto END,
                has_ranking_snapshot = CASE WHEN EXISTS (
                    SELECT 1 FROM team_ranking_snapshots trs
                    WHERE trs.team_id IN (matches.team1_id, matches.team2_id)
                      AND trs.captured_at_utc <= matches.datetime_utc
                ) AND matches.data_tier <> 'historical_seed' THEN 1 ELSE has_ranking_snapshot END
            """
        )
        conn.commit()

        archive_stats = build_db.insert_daily_archives(conn.cursor())
        for key, value in archive_stats.items():
            counts[key] += value
        conn.commit()

        counts["predictions_rows"] += insert_predictions(conn, run_dir)
        counts["player_snapshot_flags"] += update_player_snapshot_flags(conn, run_dir)
        counts["fetch_state_rows"] += update_fetch_state_from_run(conn, run_dir)
        counts["reconciled_player_fetch_state_rows"] += reconcile_known_missing_player_stats(conn)
        conn.commit()

        rows = dataio.load_training_rows_from_db(db_path)
        mart_stats = build_db._recompute_mart(conn, rows)
        for key, value in mart_stats.items():
            counts[key] += value
        conn.commit()
    except Exception as exc:
        conn.rollback()
        status = "failed"
        note = str(exc)
        raise
    finally:
        finished = utcnow()
        try:
            diag = read_json(run_dir / "fetch_diagnostics.json", {})
            req_made = requests_made if requests_made is not None else int(diag.get("http_attempts") or 0)
            req_skip = requests_skipped_by_freshness if requests_skipped_by_freshness is not None else int(diag.get("freshness_skipped") or 0)
            conn.execute(
                "INSERT INTO ingest_runs(run_id, started_at_utc, finished_at_utc, status, rows_upserted_json, "
                "requests_made, requests_skipped_by_freshness, note) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    started,
                    finished,
                    status,
                    json.dumps(dict(counts), ensure_ascii=False),
                    req_made,
                    req_skip,
                    note,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    result = {
        "db": str(db_path),
        "run_id": run_id,
        "status": status,
        "rows_upserted": dict(counts),
    }
    if status == "ok":
        result.update(build_db.backup_database(db_path, backup_dir, mirror_backup_dir))
    return result


def latest_run_dir() -> Path:
    manifest = read_json(ROOT / "PIPELINE" / "master" / "manifest.json", {})
    run_id = manifest.get("last_run_id")
    if run_id:
        path = DEFAULT_RUNS / run_id
        if path.exists():
            return path
    runs = sorted(path for path in DEFAULT_RUNS.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError("No hay runs en PIPELINE/runs")
    return runs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest incremental run -> BBDD/cs2.db")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--requests-made", type=int, default=None)
    parser.add_argument("--requests-skipped-by-freshness", type=int, default=None)
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--mirror-backup-dir", default=str(DEFAULT_MIRROR_BACKUP_DIR))
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-mirror-backup", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    backup_dir = None if args.no_backup else Path(args.backup_dir)
    mirror = None if args.no_backup or args.no_mirror_backup else Path(args.mirror_backup_dir)
    result = ingest_run(
        run_dir,
        Path(args.db),
        Path(args.master),
        requests_made=args.requests_made,
        requests_skipped_by_freshness=args.requests_skipped_by_freshness,
        backup_dir=backup_dir,
        mirror_backup_dir=mirror,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

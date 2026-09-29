"""Read-only recovery planning and sanctioned additive acquisition persistence.

Never train, rewrite predictions, restore the database or clear a failure without
evidence. New observations retain their actual availability, including old matches.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from BBDD.recovery_references import resolve_references

ISSUES = {"partial", "blocked", "error"}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def readonly(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def plan_recovery(conn: sqlite3.Connection, master: dict[str, Any], *, now: str, limit: int) -> dict[str, Any]:
    from BBDD.ingest import match_detail_fetch_status, next_eligible

    health = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
    foreign_keys = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
    if health != ["ok"] or foreign_keys:
        raise RuntimeError(
            f"Integridad SQLite no válida: {health}; FK={len(foreign_keys)}. No se repara ni restaura a ciegas."
        )
    states = {(str(r["entity_type"]), str(r["entity_key"])): dict(r) for r in conn.execute("SELECT * FROM fetch_state")}
    snapshots: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT hltv_match_id,payload_json FROM raw_snapshots WHERE kind='match_snapshot' "
        "ORDER BY captured_at_utc,raw_snapshot_id"
    ):
        try:
            payload = json.loads(row["payload_json"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            snapshots[str(row["hltv_match_id"])] = payload
    corrections = []
    for key, snapshot in snapshots.items():
        state = states.get(("match_detail", key))
        status = match_detail_fetch_status(snapshot)
        captured = str(snapshot.get("captured_at") or "")
        if (
            state
            and state["last_status"] == "ok"
            and status != "ok"
            and captured >= str(state["last_fetched_at_utc"] or "")
        ):
            correction = {
                **state,
                "status": status,
                "captured_at": captured,
                "note": "Detalle fallback/error: no es una descarga OK.",
            }
            corrections.append(correction)
            state.update(last_status=status, next_eligible_at_utc=next_eligible("match_detail", status, captured))
    selected, deferred = [], []
    references = resolve_references(
        conn,
        {
            ("player" if kind == "player_stats" else "team", key)
            for (kind, key), state in states.items()
            if kind in {"player_stats", "team_profile"} and state["last_status"] in ISSUES
        },
    )
    for (kind, key), state in states.items():
        if state["last_status"] not in ISSUES:
            continue
        record = snapshots.get(key) or master.get(key) or {}
        item: dict[str, Any] = {**state, "record": record}
        if kind.startswith("match_"):
            item["url"] = record.get("link") or (master.get(key) or {}).get("link")
        elif kind == "player_stats":
            player = references.get(("player", key))
            if player:
                item["player"] = player
                item["url"] = player["link"]
                item["reference_provenance"] = player["provenance"]
        elif kind == "team_profile":
            team = references.get(("team", key))
            if team:
                item["url"] = f"/team/{key}/{team['slug']}"
                item["reference_provenance"] = team["provenance"]
        elif kind in {"ranking_hltv", "ranking_valve"}:
            item["url"] = "/ranking/teams" if kind == "ranking_hltv" else "/valve-ranking/teams"
        reason = None
        if state.get("next_eligible_at_utc") and str(state["next_eligible_at_utc"]) > now:
            reason = "cooldown"
        elif not item.get("url"):
            reason = "sin_url_verificable"
        if reason:
            deferred.append({**item, "deferred_reason": reason})
        else:
            selected.append(item)
    selected.sort(
        key=lambda x: (
            0 if str(x["record"].get("date") or "") >= now[:10] else 1,
            0 if x["entity_type"] == "match_detail" else 1,
            str(x.get("last_fetched_at_utc") or ""),
            x["entity_type"],
            x["entity_key"],
        )
    )
    if limit > 0:
        deferred.extend({**item, "deferred_reason": "limite"} for item in selected[limit:])
        selected = selected[:limit]
    return {
        "schema_version": "cs2-fetch-recovery-v1",
        "planned_at": now,
        "sqlite": "ok",
        "issue_count": sum(s["last_status"] in ISSUES for s in states.values()),
        "no_sample_count": sum(s["last_status"] == "not_found" for s in states.values()),
        "corrections": corrections,
        "selected": selected,
        "deferred": deferred,
        "by_entity": dict(Counter(s["entity_type"] for s in states.values() if s["last_status"] in ISSUES)),
    }


def protect_history(
    action: int, table: str | None, column: str | None, database: str | None, trigger: str | None
) -> int:
    """Defense in depth: no destructive SQL; the ledger is never a write target."""
    if action == sqlite3.SQLITE_DELETE:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_UPDATE and table != "fetch_state":
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_INSERT and table not in {
        "fetch_state",
        "raw_snapshots",
        "player_stat_snapshots",
        "players",
        "prematch_lineup_snapshots",
        "match_analytics_snapshots",
        "match_analytics_map_stats",
        "match_analytics_map_handicap",
        "team_ranking_snapshots",
        "veto",
        "maps",
        "match_lineups",
        "map_player_stats",
        "map_player_side_stats",
    }:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def persist_recovery(conn: sqlite3.Connection, plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from BBDD import build_db
    from BBDD.ingest import upsert_fetch_state

    report_path = run_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"results": []}
    conn.set_authorizer(protect_history)
    try:
        with conn:
            for correction in plan["corrections"]:
                current = conn.execute(
                    "SELECT last_status,last_fetched_at_utc FROM fetch_state WHERE entity_type=? AND entity_key=?",
                    (correction["entity_type"], correction["entity_key"]),
                ).fetchone()
                if current and tuple(current) == ("ok", correction["last_fetched_at_utc"]):
                    upsert_fetch_state(
                        conn,
                        correction["entity_type"],
                        correction["entity_key"],
                        correction["status"],
                        fetched_at=correction["captured_at"],
                        note=correction["note"],
                    )
            counts = build_db.insert_daily_archives(conn.cursor(), run_dirs=[run_dir])
            matches = {str(r[1]): int(r[0]) for r in conn.execute("SELECT match_id,hltv_match_id FROM matches")}
            teams = {
                build_db.dataio.clean_team(r[1]): int(r[0]) for r in conn.execute("SELECT team_id,name FROM teams")
            }
            hltv_teams = {str(r[1]): int(r[0]) for r in conn.execute("SELECT team_id,hltv_id FROM teams")}
            match_teams = {
                int(r[0]): (int(r[1]), int(r[2]))
                for r in conn.execute("SELECT match_id,team1_id,team2_id FROM matches")
            }
            counts.update(build_db.insert_prematch_lineup_snapshots(conn.cursor(), run_dir, matches, teams, hltv_teams))
            counts.update(
                build_db.insert_match_analytics_snapshots(
                    conn.cursor(), run_dir, matches, teams, hltv_teams, update_events=False
                )
            )
            counts["rankings"] = build_db.insert_team_rankings(conn.cursor(), hltv_teams, run_dirs=[run_dir])
            assets = {
                p.parent.name: {"hltv_assets_file": str(p.resolve())}
                for p in (run_dir / "match_assets").glob("*/assets.json")
            }
            counts.update(build_db.insert_hltv_assets(conn.cursor(), assets, matches, match_teams, teams, hltv_teams))
            for result in report["results"]:
                if not result.get("attempted"):
                    continue
                # A resumed/old report cannot make the cache newer than a later observation.
                current = conn.execute(
                    "SELECT last_fetched_at_utc FROM fetch_state WHERE entity_type=? AND entity_key=?",
                    (result["entity_type"], result["entity_key"]),
                ).fetchone()
                if current and str(current[0] or "") > result["captured_at"]:
                    continue
                upsert_fetch_state(
                    conn,
                    result["entity_type"],
                    result["entity_key"],
                    result["status"],
                    fetched_at=result["captured_at"],
                    note=result.get("note"),
                )
            return counts
    finally:
        conn.set_authorizer(None)

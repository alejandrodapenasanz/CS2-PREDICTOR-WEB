from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "CS2"
DAILY_ROOT = ROOT / "PIPELINE"
MODEL_ROOT = ROOT / "MODEL"
WEB_ROOT = REPO_ROOT / "WEB"
BBDD_ROOT = ROOT / "BBDD"
DB_PATH = BBDD_ROOT / "cs2.db"


def configure_sport_root(sport_root: Path) -> None:
    """Select the sport domain while keeping the web output repository-wide."""
    global ROOT, DAILY_ROOT, MODEL_ROOT, BBDD_ROOT, DB_PATH
    ROOT = sport_root.resolve()
    DAILY_ROOT = ROOT / "PIPELINE"
    MODEL_ROOT = ROOT / "MODEL"
    BBDD_ROOT = ROOT / "BBDD"
    DB_PATH = BBDD_ROOT / "cs2.db"


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run() -> Path:
    manifest = read_json(DAILY_ROOT / "master" / "manifest.json", {})
    if manifest.get("last_run_id"):
        return DAILY_ROOT / "runs" / manifest["last_run_id"]
    runs = sorted([path for path in (DAILY_ROOT / "runs").iterdir() if path.is_dir()])
    if not runs:
        raise FileNotFoundError("No PIPELINE runs found.")
    return runs[-1]


def model_payload() -> dict:
    """Métricas del modelo (walk-forward), SHAP y metadatos para el panel del modelo."""
    metrics = read_json(MODEL_ROOT / "results" / "metrics.json", {})
    shap = read_json(MODEL_ROOT / "results" / "shap_importance.json", [])
    seg = read_json(MODEL_ROOT / "results" / "segmented_eval.json", {})
    favorite_accuracy = read_json(
        MODEL_ROOT / "results" / "favorite_accuracy_bands.json",
        seg.get("favorite_accuracy", {}),
    )
    favorite_accuracy_timeline = build_favorite_accuracy_timeline(
        MODEL_ROOT / "results" / "predictions_walkforward.csv",
        str(favorite_accuracy.get("model") or "nested_model_policy"),
    )
    return {
        "metrics": metrics,
        "shap_top": shap[:15] if isinstance(shap, list) else [],
        "segments": seg.get("segments", []),
        "favorite_accuracy": favorite_accuracy,
        "favorite_accuracy_timeline": favorite_accuracy_timeline,
        "market": seg.get("market", {}),
        "production_model": _production_model(),
    }


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _next_month(value: date) -> date:
    return (
        value.replace(year=value.year + 1, month=1, day=1)
        if value.month == 12
        else value.replace(month=value.month + 1, day=1)
    )


def _bucket_start(value: date, bucket: str) -> date:
    if bucket == "week":
        return value - timedelta(days=value.weekday())
    if bucket == "month":
        return _month_start(value)
    return value


def _next_bucket(value: date, bucket: str) -> date:
    if bucket == "week":
        return value + timedelta(days=7)
    if bucket == "month":
        return _next_month(value)
    return value + timedelta(days=1)


def build_favorite_accuracy_timeline(path: Path, model: str) -> dict:
    """Build deduplicated OOS favorite-accuracy series for the model dashboard."""
    base = {
        "source": "MODEL/results/predictions_walkforward.csv",
        "method": "deduplicated_walk_forward_favorite_accuracy",
        "model": model,
        "available_start": None,
        "available_end": None,
        "windows": {},
    }
    if not path.exists():
        return base

    by_match: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("model") or "") != model:
                continue
            match_id = str(row.get("match_id") or "")
            try:
                played_on = date.fromisoformat(str(row.get("date") or "")[:10])
                probability = float(row["prob_team1"])
                actual = int(row["actual"])
            except (KeyError, TypeError, ValueError):
                continue
            if not match_id or actual not in {0, 1} or not 0.0 <= probability <= 1.0:
                continue
            by_match[match_id] = {
                "match_id": match_id,
                "date": played_on,
                "correct": int((probability >= 0.5) == bool(actual)),
            }
    rows = sorted(by_match.values(), key=lambda row: (row["date"], row["match_id"]))
    if not rows:
        return base

    available_start = rows[0]["date"]
    available_end = rows[-1]["date"]
    base["available_start"] = available_start.isoformat()
    base["available_end"] = available_end.isoformat()
    definitions = {
        "7d": {"label": "7 dias", "days": 7, "bucket": "day"},
        "1m": {"label": "1 mes", "days": 30, "bucket": "day"},
        "3m": {"label": "3 meses", "days": 90, "bucket": "week"},
        "6m": {"label": "6 meses", "days": 180, "bucket": "week"},
        "1y": {"label": "1 ano", "days": 365, "bucket": "month"},
    }
    for key, definition in definitions.items():
        requested_start = available_end - timedelta(days=int(definition["days"]) - 1)
        period_start = max(requested_start, available_start)
        selected = [row for row in rows if period_start <= row["date"] <= available_end]
        grouped: dict[date, list[dict]] = defaultdict(list)
        bucket = str(definition["bucket"])
        for row in selected:
            grouped[_bucket_start(row["date"], bucket)].append(row)

        points = []
        cursor = _bucket_start(period_start, bucket)
        while cursor <= available_end:
            next_cursor = _next_bucket(cursor, bucket)
            point_rows = grouped.get(cursor, [])
            point_start = max(cursor, period_start)
            point_end = min(next_cursor - timedelta(days=1), available_end)
            correct = sum(int(row["correct"]) for row in point_rows)
            n = len(point_rows)
            points.append(
                {
                    "date": point_start.isoformat(),
                    "end_date": point_end.isoformat(),
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n if n else None,
                }
            )
            cursor = next_cursor
        total_correct = sum(int(row["correct"]) for row in selected)
        total_n = len(selected)
        base["windows"][key] = {
            **definition,
            "start": period_start.isoformat(),
            "end": available_end.isoformat(),
            "n": total_n,
            "correct": total_correct,
            "accuracy": total_correct / total_n if total_n else None,
            "points": points,
        }
    return base


def _production_model() -> str:
    """Lee el modelo de producción del artefacto si está disponible (import perezoso)."""
    try:
        import sys

        sys.path.insert(0, str(MODEL_ROOT))
        from cs2model.artifacts import load_artifact

        art = load_artifact()
        if art:
            return str(art.metadata.get("production_model", "")) + " · " + str(art.metadata.get("model", ""))
    except Exception:
        pass
    return ""


def _table_count(conn: sqlite3.Connection, table: str) -> int | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not exists:
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _fetch_dicts(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "relative_path": str(path.relative_to(ROOT)), "exists": False, "size_bytes": 0}
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _rows_summary(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, (int, float)) and int(v) != 0}


def database_payload() -> dict:
    """Operational SQLite status for the local web debug tab."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict = {
        "generated_at_utc": now,
        "db": _file_info(DB_PATH),
        "wal": _file_info(Path(str(DB_PATH) + "-wal")),
        "shm": _file_info(Path(str(DB_PATH) + "-shm")),
        "dump": _file_info(BBDD_ROOT / "cs2_dump.sql"),
        "exists": DB_PATH.exists(),
        "health": "missing",
        "notes": [],
    }
    if not DB_PATH.exists():
        payload["notes"].append("BBDD/cs2.db no existe todavia.")
        return payload

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        journal_mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        fk_issues = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        tables = [
            "matches",
            "teams",
            "players",
            "events",
            "odds",
            "maps",
            "map_player_stats",
            "map_player_side_stats",
            "veto",
            "raw_snapshots",
            "player_stat_snapshots",
            "team_ranking_snapshots",
            "predictions",
            "ratings_history",
            "match_features",
            "fetch_state",
            "ingest_runs",
        ]
        table_counts = [
            {"name": table, "rows": count}
            for table in tables
            if (count := _table_count(conn, table)) is not None
        ]
        flag_columns = [
            "has_prematch_odds",
            "has_player_snapshot",
            "has_ranking_snapshot",
            "has_analytics",
            "has_context",
            "has_box_score",
            "has_veto",
        ]
        coverage = []
        non_seed_total = int(
            conn.execute("SELECT COUNT(*) FROM matches WHERE data_tier <> 'historical_seed'").fetchone()[0]
        )
        for flag in flag_columns:
            count = int(
                conn.execute(
                    f"SELECT COALESCE(SUM({flag}),0) FROM matches WHERE data_tier <> 'historical_seed'"
                ).fetchone()[0]
            )
            coverage.append(
                {
                    "flag": flag,
                    "count": count,
                    "total": non_seed_total,
                    "ratio": (count / non_seed_total) if non_seed_total else None,
                }
            )
        recent_ingests = []
        for row in _fetch_dicts(
            conn,
            """
            SELECT ingest_id, run_id, started_at_utc, finished_at_utc, status,
                   requests_made, requests_skipped_by_freshness, rows_upserted_json, note
            FROM ingest_runs
            ORDER BY ingest_id DESC
            LIMIT 8
            """,
        ):
            row["rows_upserted"] = _rows_summary(row.pop("rows_upserted_json", None))
            recent_ingests.append(row)

        blocked = int(
            conn.execute("SELECT COUNT(*) FROM fetch_state WHERE last_status='blocked'").fetchone()[0]
        )
        errors = int(
            conn.execute("SELECT COUNT(*) FROM fetch_state WHERE last_status='error'").fetchone()[0]
        )
        partial = int(
            conn.execute("SELECT COUNT(*) FROM fetch_state WHERE last_status='partial'").fetchone()[0]
        )
        fresh = int(
            conn.execute(
                "SELECT COUNT(*) FROM fetch_state WHERE next_eligible_at_utc IS NOT NULL AND next_eligible_at_utc > ?",
                (now,),
            ).fetchone()[0]
        )
        leakage_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM match_features mf
                JOIN matches m ON m.match_id = mf.match_id
                WHERE mf.data_up_to_utc > m.datetime_utc
                """
            ).fetchone()[0]
        )
        payload.update(
            {
                "health": "ok" if quick_check == "ok" and fk_issues == 0 and leakage_rows == 0 else "warning",
                "pragmas": {
                    "foreign_keys": foreign_keys,
                    "journal_mode": journal_mode,
                    "quick_check": quick_check,
                    "foreign_key_issues": fk_issues,
                },
                "table_counts": table_counts,
                "matches_by_status": _fetch_dicts(
                    conn,
                    "SELECT status AS label, COUNT(*) AS count FROM matches GROUP BY status ORDER BY count DESC",
                ),
                "matches_by_tier": _fetch_dicts(
                    conn,
                    "SELECT data_tier AS label, COUNT(*) AS count FROM matches GROUP BY data_tier ORDER BY count DESC",
                ),
                "matches_by_environment": _fetch_dicts(
                    conn,
                    "SELECT environment AS label, COUNT(*) AS count FROM matches GROUP BY environment ORDER BY count DESC",
                ),
                "coverage": coverage,
                "fetch_summary": _fetch_dicts(
                    conn,
                    """
                    SELECT entity_type,
                           COUNT(*) AS total,
                           SUM(CASE WHEN last_status='ok' THEN 1 ELSE 0 END) AS ok,
                           SUM(CASE WHEN last_status='partial' THEN 1 ELSE 0 END) AS partial,
                           SUM(CASE WHEN last_status='blocked' THEN 1 ELSE 0 END) AS blocked,
                           SUM(CASE WHEN last_status='error' THEN 1 ELSE 0 END) AS error,
                           SUM(CASE WHEN last_status='not_found' THEN 1 ELSE 0 END) AS not_found,
                           SUM(CASE WHEN next_eligible_at_utc IS NOT NULL AND next_eligible_at_utc > ? THEN 1 ELSE 0 END) AS fresh
                    FROM fetch_state
                    GROUP BY entity_type
                    ORDER BY entity_type
                    """,
                    (now,),
                ),
                "fetch_problem_rows": _fetch_dicts(
                    conn,
                    """
                    SELECT entity_type, entity_key, last_status, last_fetched_at_utc,
                           next_eligible_at_utc, note
                    FROM fetch_state
                    WHERE last_status IN ('blocked','error','partial')
                    ORDER BY last_fetched_at_utc DESC
                    LIMIT 12
                    """,
                ),
                "recent_ingests": recent_ingests,
                "integrity": {
                    "leakage_rows": leakage_rows,
                    "blocked_fetches": blocked,
                    "error_fetches": errors,
                    "partial_fetches": partial,
                    "fresh_entities": fresh,
                    "non_seed_matches": non_seed_total,
                },
                "latest_matches": _fetch_dicts(
                    conn,
                    """
                    SELECT m.hltv_match_id, m.datetime_utc, m.status, m.data_tier,
                           m.environment, m.stage, t1.name AS team1, t2.name AS team2
                    FROM matches m
                    JOIN teams t1 ON t1.team_id = m.team1_id
                    JOIN teams t2 ON t2.team_id = m.team2_id
                    ORDER BY m.datetime_utc DESC
                    LIMIT 10
                    """,
                ),
            }
        )
    except Exception as exc:
        payload["health"] = "error"
        payload["notes"].append(str(exc))
    finally:
        conn.close()

    backup_dir = BBDD_ROOT / "backups"
    backups = sorted(backup_dir.glob("*.db"), key=lambda path: path.stat().st_mtime, reverse=True)[:8]
    payload["backups"] = [_file_info(path) for path in backups]
    return payload


def parse_match_datetime(date_text: str | None, hour_text: str | None = None) -> datetime | None:
    if not date_text:
        return None
    try:
        date_obj = datetime.strptime(str(date_text), "%Y-%m-%d")
    except ValueError:
        return None
    if hour_text:
        match = re.search(r"(\d{1,2}):(\d{2})", str(hour_text))
        if match:
            return date_obj.replace(hour=int(match.group(1)), minute=int(match.group(2)))
    return date_obj.replace(hour=12, minute=0)


def publishable_matches(matches: list[dict], grace_minutes: int = 15) -> tuple[list[dict], dict]:
    now_local = datetime.now(ZoneInfo("Europe/Madrid")).replace(tzinfo=None)
    cutoff = now_local - timedelta(minutes=grace_minutes)
    kept = []
    skipped = {"past_or_started": 0, "completed": 0}
    for match in matches:
        if match.get("status") == "completed":
            skipped["completed"] += 1
            continue
        match_dt = parse_match_datetime(match.get("date"), match.get("hour"))
        if match_dt and match_dt < cutoff:
            skipped["past_or_started"] += 1
            continue
        kept.append(match)
    return kept, {key: value for key, value in skipped.items() if value}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local static web dashboard.")
    parser.add_argument("--sport-root", type=Path, default=REPO_ROOT / "CS2")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    configure_sport_root(args.sport_root)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    raw_data = read_json(run_dir / "predictions_enriched.json", [])
    data, skipped_web = publishable_matches(raw_data)
    manifest = read_json(run_dir / "manifest.json", {})
    master_manifest = read_json(DAILY_ROOT / "master" / "manifest.json", {})
    calibration = read_json(run_dir / "calibration.json", {})

    payload = {
        "sport": ROOT.name,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runDir": str(run_dir),
        "manifest": manifest,
        "masterManifest": master_manifest,
        "calibration": calibration,
        "model": model_payload(),
        "database": database_payload(),
        "webFilter": {
            "sourceMatches": len(raw_data),
            "publishedMatches": len(data),
            "skipped": skipped_web,
            "timezone": "Europe/Madrid",
        },
        "matches": data,
    }
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "data.js").write_text(
        "window.__CS2_PREDICTOR_DATA__ = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    model_label = str(payload["model"].get("production_model") or "").encode("ascii", "replace").decode("ascii")
    print(f"Built {WEB_ROOT / 'data.js'} with {len(data)} matches - model={model_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the static multi-sport dashboard payload.

The script reads the latest CS2 pipeline run and, when available, the latest
tennis prediction run from ``TENNIS/BBDD/tennis.sqlite3``.  Tennis SQLite is
opened in read-only mode and an absent or partially migrated schema is exposed
as an empty, diagnostic payload instead of preventing publication.

Run from any directory with ``python WEB/build_web.py``.  ``--sport-root`` and
``--run-dir`` retain their existing CS2 meanings.
"""

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
TENNIS_DB_PATH = REPO_ROOT / "TENNIS" / "BBDD" / "tennis.sqlite3"


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


def _quote_sqlite_identifier(identifier: str) -> str:
    """Quote a SQLite identifier obtained from SQLite's own schema."""
    return '"' + identifier.replace('"', '""') + '"'


def _sqlite_schema(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    """Return user-table names and columns without assuming a migration level."""
    table_rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    schema: dict[str, tuple[str, ...]] = {}
    for table_row in table_rows:
        table = str(table_row[0])
        columns = conn.execute(
            f"PRAGMA table_info({_quote_sqlite_identifier(table)})"
        ).fetchall()
        schema[table] = tuple(str(column[1]) for column in columns)
    return schema


def _schema_name(names: list[str] | tuple[str, ...], aliases: tuple[str, ...]) -> str | None:
    """Resolve the first documented alias against case-insensitive schema names."""
    by_lower = {str(name).lower(): str(name) for name in names}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _table_name(
    schema: dict[str, tuple[str, ...]],
    aliases: tuple[str, ...],
) -> str | None:
    """Resolve a table alias from an introspected SQLite schema."""
    return _schema_name(list(schema), aliases)


def _row_value(row: dict | None, aliases: tuple[str, ...]):
    """Read the first non-null semantic alias from an introspected row."""
    if not row:
        return None
    by_lower = {str(key).lower(): value for key, value in row.items()}
    for alias in aliases:
        value = by_lower.get(alias.lower())
        if value is not None:
            return value
    return None


def _first_value(rows: tuple[dict | None, ...], aliases: tuple[str, ...]):
    """Read a semantic value from several rows in priority order."""
    for row in rows:
        value = _row_value(row, aliases)
        if value is not None:
            return value
    return None


def _as_text(value) -> str | None:
    """Convert a scalar SQLite value to non-empty display text."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_probability(value) -> float | None:
    """Return a finite probability in the closed interval [0, 1]."""
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if probability != probability or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _as_number(value) -> float | None:
    """Return a finite numeric value or ``None`` for malformed SQLite data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _as_flags(value) -> list[str]:
    """Normalize JSON or delimited confidence flags without inventing flags."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded if str(item).strip()]
    return [item.strip() for item in re.split(r"[|;,]", text) if item.strip()]


def _as_json_object(value) -> dict:
    """Decode an immutable JSON payload when it contains an object."""
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _sqlite_select_expression(
    source_alias: str,
    columns: tuple[str, ...],
    output_name: str,
    aliases: tuple[str, ...] = (),
) -> str:
    """Build a nullable SELECT expression for one introspected source column."""
    column = _schema_name(columns, (output_name, *aliases))
    output_sql = _quote_sqlite_identifier(output_name)
    if column is None:
        return f"NULL AS {output_sql}"
    return (
        f"{source_alias}.{_quote_sqlite_identifier(column)} "
        f"AS {output_sql}"
    )


def _official_tennis_rows(
    conn: sqlite3.Connection,
    schema: dict[str, tuple[str, ...]],
) -> tuple[list[dict], dict | None, list[str]]:
    """Read official immutable predictions from the latest contributing run."""
    notes: list[str] = []
    table_aliases = {
        "runs": ("runs", "prediction_runs"),
        "matches": ("matches", "tennis_matches"),
        "predictions": ("predictions", "daily_predictions"),
        "official_predictions": ("official_predictions",),
        "settlements": ("settlements",),
    }
    tables = {
        semantic: _table_name(schema, aliases)
        for semantic, aliases in table_aliases.items()
    }
    required_tables = ("runs", "matches", "predictions", "official_predictions")
    missing_tables = [
        semantic for semantic in required_tables if tables[semantic] is None
    ]
    if missing_tables:
        notes.append(
            "Faltan tablas operativas requeridas: " + ", ".join(missing_tables) + "."
        )
        return [], None, notes

    runs_table = str(tables["runs"])
    matches_table = str(tables["matches"])
    predictions_table = str(tables["predictions"])
    official_table = str(tables["official_predictions"])
    settlements_table = tables["settlements"]
    run_columns = schema[runs_table]
    match_columns = schema[matches_table]
    prediction_columns = schema[predictions_table]
    official_columns = schema[official_table]
    settlement_columns = schema[str(settlements_table)] if settlements_table else ()

    join_columns = {
        "runs.run_id": _schema_name(run_columns, ("run_id", "id")),
        "predictions.run_id": _schema_name(
            prediction_columns, ("run_id", "prediction_run_id")
        ),
        "predictions.prediction_id": _schema_name(
            prediction_columns, ("prediction_id", "id")
        ),
        "predictions.source_match_id": _schema_name(
            prediction_columns, ("source_match_id", "match_id")
        ),
        "official_predictions.prediction_id": _schema_name(
            official_columns, ("prediction_id",)
        ),
        "official_predictions.source_match_id": _schema_name(
            official_columns, ("source_match_id", "match_id")
        ),
        "matches.source_match_id": _schema_name(
            match_columns, ("source_match_id", "match_id", "id")
        ),
    }
    missing_columns = [
        semantic for semantic, column in join_columns.items() if column is None
    ]
    if missing_columns:
        notes.append(
            "Faltan columnas de relación requeridas: "
            + ", ".join(missing_columns)
            + "."
        )
        return [], None, notes

    run_type = _schema_name(run_columns, ("run_type",))
    run_status = _schema_name(run_columns, ("status", "run_status"))
    prediction_run = str(join_columns["predictions.run_id"])
    run_id = str(join_columns["runs.run_id"])
    prediction_id = str(join_columns["predictions.prediction_id"])
    prediction_source_match = str(
        join_columns["predictions.source_match_id"]
    )
    official_prediction_id = str(
        join_columns["official_predictions.prediction_id"]
    )
    official_source_match = str(
        join_columns["official_predictions.source_match_id"]
    )
    latest_filters = [
        (
            "EXISTS ("
            f"SELECT 1 FROM {_quote_sqlite_identifier(predictions_table)} AS px "
            f"JOIN {_quote_sqlite_identifier(official_table)} AS ox "
            f"ON ox.{_quote_sqlite_identifier(official_prediction_id)} "
            f"= px.{_quote_sqlite_identifier(prediction_id)} "
            f"AND ox.{_quote_sqlite_identifier(official_source_match)} "
            f"= px.{_quote_sqlite_identifier(prediction_source_match)} "
            f"WHERE px.{_quote_sqlite_identifier(prediction_run)} "
            f"= r.{_quote_sqlite_identifier(run_id)}"
            ")"
        )
    ]
    latest_params: list[str] = []
    if run_type:
        latest_filters.append(
            f"r.{_quote_sqlite_identifier(run_type)} = ?"
        )
        latest_params.append("prediction")
    if run_status:
        latest_filters.append(
            f"r.{_quote_sqlite_identifier(run_status)} = ?"
        )
        latest_params.append("complete")
    order_columns = []
    for alias in (
        "target_date",
        "completed_at_utc",
        "created_at_utc",
        "started_at_utc",
        "run_id",
        "id",
    ):
        column = _schema_name(run_columns, (alias,))
        if column and column not in order_columns:
            order_columns.append(column)
    if not order_columns:
        notes.append("La tabla runs no tiene columnas que permitan ordenar ejecuciones.")
        return [], None, notes
    latest_query = (
        f"SELECT r.* FROM {_quote_sqlite_identifier(runs_table)} AS r "
        f"WHERE {' AND '.join(latest_filters)} "
        "ORDER BY "
        + ", ".join(
            f"r.{_quote_sqlite_identifier(column)} DESC"
            for column in order_columns
        )
        + " LIMIT 1"
    )
    latest_sqlite_row = conn.execute(
        latest_query, tuple(latest_params)
    ).fetchone()
    if latest_sqlite_row is None:
        notes.append(
            "No hay una ejecución de predicción completa con predicciones oficiales."
        )
        return [], None, notes
    latest_run = dict(latest_sqlite_row)
    latest_run_id = _row_value(latest_run, (run_id,))

    select_spec = [
        ("r", run_columns, "run_id", ("id",)),
        ("r", run_columns, "target_date", ("prediction_date", "run_date")),
        ("r", run_columns, "run_status", ("status",)),
        ("r", run_columns, "started_at_utc", ()),
        ("r", run_columns, "completed_at_utc", ("finished_at_utc",)),
        (
            "m",
            match_columns,
            "source_match_id",
            ("match_id", "tennis_match_id", "id"),
        ),
        ("m", match_columns, "match_date", ("prediction_date", "date")),
        ("m", match_columns, "tournament", ("tourney_name", "event_name")),
        ("m", match_columns, "tour_level", ("level", "category")),
        ("m", match_columns, "gender", ()),
        ("m", match_columns, "player_1_slug", ("player_a_slug",)),
        ("m", match_columns, "player_2_slug", ("player_b_slug",)),
        ("p", prediction_columns, "prediction_id", ("id",)),
        ("p", prediction_columns, "prediction_as_of_utc", ()),
        ("p", prediction_columns, "source_status", ("match_status",)),
        ("p", prediction_columns, "player_a_name", ("player1_name",)),
        ("p", prediction_columns, "player_b_name", ("player2_name",)),
        ("p", prediction_columns, "player_a_slug", ("player1_slug",)),
        ("p", prediction_columns, "player_b_slug", ("player2_slug",)),
        ("p", prediction_columns, "model_probability_a", ("probability_a",)),
        ("p", prediction_columns, "model_probability_b", ("probability_b",)),
        ("p", prediction_columns, "confidence", ("reliability_label",)),
        ("p", prediction_columns, "confidence_flags", ("reliability_flags",)),
        ("p", prediction_columns, "prediction_status", ("status",)),
        ("p", prediction_columns, "is_valid", ()),
        ("p", prediction_columns, "invalid_reason", ()),
        ("p", prediction_columns, "payload_json", ()),
        ("p", prediction_columns, "created_at_utc", ()),
    ]
    select_parts = [
        _sqlite_select_expression(source, columns, output, aliases)
        for source, columns, output, aliases in select_spec
    ]
    if settlements_table:
        select_parts.extend(
            [
                _sqlite_select_expression(
                    "s",
                    settlement_columns,
                    "actual_winner_slug",
                    ("winner_slug",),
                ),
                _sqlite_select_expression(
                    "s", settlement_columns, "actual_outcome_a"
                ),
                _sqlite_select_expression(
                    "s", settlement_columns, "settled_at_utc"
                ),
            ]
        )
    else:
        select_parts.extend(
            (
                f"NULL AS {_quote_sqlite_identifier(output)}"
                for output in (
                    "actual_winner_slug",
                    "actual_outcome_a",
                    "settled_at_utc",
                )
            )
        )
        notes.append(
            "No existe tabla settlements; los ganadores reales quedan vacíos."
        )

    match_source_match = str(join_columns["matches.source_match_id"])
    from_sql = (
        f"{_quote_sqlite_identifier(official_table)} AS o "
        f"JOIN {_quote_sqlite_identifier(predictions_table)} AS p "
        f"ON o.{_quote_sqlite_identifier(official_prediction_id)} "
        f"= p.{_quote_sqlite_identifier(prediction_id)} "
        f"JOIN {_quote_sqlite_identifier(matches_table)} AS m "
        f"ON p.{_quote_sqlite_identifier(prediction_source_match)} "
        f"= m.{_quote_sqlite_identifier(match_source_match)} "
        f"AND o.{_quote_sqlite_identifier(official_source_match)} "
        f"= p.{_quote_sqlite_identifier(prediction_source_match)} "
        f"JOIN {_quote_sqlite_identifier(runs_table)} AS r "
        f"ON p.{_quote_sqlite_identifier(prediction_run)} "
        f"= r.{_quote_sqlite_identifier(run_id)}"
    )
    if settlements_table:
        settlement_source_match = _schema_name(
            settlement_columns, ("source_match_id", "match_id")
        )
        settlement_prediction_id = _schema_name(
            settlement_columns,
            ("official_prediction_id", "prediction_id"),
        )
        if settlement_source_match and settlement_prediction_id:
            from_sql += (
                f" LEFT JOIN {_quote_sqlite_identifier(str(settlements_table))} AS s "
                f"ON s.{_quote_sqlite_identifier(settlement_source_match)} "
                f"= o.{_quote_sqlite_identifier(official_source_match)} "
                f"AND s.{_quote_sqlite_identifier(settlement_prediction_id)} "
                f"= p.{_quote_sqlite_identifier(prediction_id)}"
            )
        else:
            notes.append(
                "settlements no tiene sus claves de relación; no se publican resultados."
            )
            select_parts[-3:] = [
                f"NULL AS {_quote_sqlite_identifier(output)}"
                for output in (
                    "actual_winner_slug",
                    "actual_outcome_a",
                    "settled_at_utc",
                )
            ]
    match_date = _schema_name(match_columns, ("match_date", "prediction_date"))
    created_at = _schema_name(prediction_columns, ("created_at_utc",))
    order_parts = []
    if match_date:
        order_parts.append(f"m.{_quote_sqlite_identifier(match_date)}")
    if created_at:
        order_parts.append(f"p.{_quote_sqlite_identifier(created_at)}")
    order_sql = " ORDER BY " + ", ".join(order_parts) if order_parts else ""
    rows_query = (
        f"SELECT {', '.join(select_parts)} FROM {from_sql} "
        f"WHERE p.{_quote_sqlite_identifier(prediction_run)} = ?"
        f"{order_sql}"
    )
    rows = [
        dict(row)
        for row in conn.execute(rows_query, (latest_run_id,)).fetchall()
    ]
    return rows, latest_run, notes


def _side_from_value(value) -> str | None:
    """Normalize an explicit player-side value to ``A`` or ``B``."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"a", "1", "player_a", "player1", "player_1"}:
        return "A"
    if normalized in {"b", "2", "player_b", "player2", "player_2"}:
        return "B"
    return None


def _tennis_match_payload(
    prediction: dict,
    match: dict,
    settlement: dict | None,
) -> dict:
    """Normalize one operational tennis record for presentation only."""
    immutable_payload = _as_json_object(
        _row_value(prediction, ("payload_json",))
    )
    sources = (match, prediction, immutable_payload)
    player_a = _as_text(
        _first_value(
            sources,
            ("player_a_name", "player1_name", "player_1_name", "player_a"),
        )
    )
    player_b = _as_text(
        _first_value(
            sources,
            ("player_b_name", "player2_name", "player_2_name", "player_b"),
        )
    )
    player_a_slug = _as_text(
        _first_value(sources, ("player_a_slug", "player1_slug", "player_1_slug"))
    )
    player_b_slug = _as_text(
        _first_value(sources, ("player_b_slug", "player2_slug", "player_2_slug"))
    )
    probability_a = _as_probability(
        _row_value(
            prediction,
            (
                "model_probability_a",
                "calibrated_probability_a",
                "probability_a",
                "prob_a",
            ),
        )
    )
    probability_b = _as_probability(
        _row_value(
            prediction,
            (
                "model_probability_b",
                "calibrated_probability_b",
                "probability_b",
                "prob_b",
            ),
        )
    )
    prediction_status = _as_text(
        _row_value(prediction, ("prediction_status", "status"))
    )
    is_valid = _row_value(prediction, ("is_valid",))
    predicted_side = _side_from_value(
        _row_value(
            prediction,
            ("predicted_winner_side", "winner_predicted_side", "decision_side"),
        )
    )
    predicted_winner = _as_text(
        _row_value(
            prediction,
            ("predicted_winner_name", "winner_predicted", "predicted_winner"),
        )
    )
    unavailable = bool(
        (is_valid is not None and str(is_valid) != "1")
        or (
            prediction_status
            and any(
                token in prediction_status.lower()
                for token in ("unavailable", "not_predicted", "skipped", "error")
            )
        )
    )
    if predicted_side is None and not unavailable:
        if probability_a is not None and probability_b is not None:
            if probability_a > probability_b:
                predicted_side = "A"
            elif probability_b > probability_a:
                predicted_side = "B"
    if predicted_winner is None:
        predicted_winner = (
            player_a if predicted_side == "A" else player_b if predicted_side == "B" else None
        )
    predicted_probability = _as_probability(
        _row_value(
            prediction,
            ("predicted_winner_probability", "winner_probability", "probability"),
        )
    )
    if predicted_probability is None:
        predicted_probability = (
            probability_a
            if predicted_side == "A"
            else probability_b if predicted_side == "B" else None
        )
    confidence_flags = _as_flags(
        _row_value(
            prediction,
            ("confidence_flags", "reliability_flags", "flags"),
        )
    )
    invalid_reason = _as_text(
        _row_value(prediction, ("invalid_reason",))
    )
    if invalid_reason and invalid_reason not in confidence_flags:
        confidence_flags.append(invalid_reason)

    actual_side = _side_from_value(
        _row_value(
            settlement,
            ("winner_side", "actual_winner_side", "outcome_winner_side"),
        )
    )
    actual_outcome_a = _row_value(settlement, ("actual_outcome_a",))
    if actual_side is None and actual_outcome_a in {0, 1, "0", "1"}:
        actual_side = "A" if int(actual_outcome_a) == 1 else "B"
    actual_slug = _as_text(
        _row_value(settlement, ("winner_slug", "actual_winner_slug"))
    )
    actual_winner = _as_text(
        _row_value(
            settlement,
            ("winner_name", "actual_winner_name", "actual_winner"),
        )
    )
    if actual_side is None and actual_slug is not None:
        if actual_slug == player_a_slug:
            actual_side = "A"
        elif actual_slug == player_b_slug:
            actual_side = "B"
    if actual_winner is None:
        actual_winner = (
            player_a if actual_side == "A" else player_b if actual_side == "B" else None
        )
    outcome_status = _as_text(
        _row_value(
            settlement,
            ("outcome_status", "settlement_status", "status"),
        )
    )
    settled_at_utc = _as_text(
        _row_value(settlement, ("settled_at_utc",))
    )
    actual = None
    if settlement and (
        actual_winner
        or actual_side
        or actual_slug
        or outcome_status
        or settled_at_utc
    ):
        actual = {
            "winner": actual_winner,
            "winner_side": actual_side,
            "winner_slug": actual_slug,
            "status": outcome_status,
            "settled_at_utc": settled_at_utc,
        }

    return {
        "match_id": _as_text(
            _first_value(
                sources,
                ("source_match_id", "match_id", "tennis_match_id", "id"),
            )
        ),
        "date": _as_text(
            _first_value(
                sources,
                ("prediction_date", "match_date", "date", "jornada"),
            )
        ),
        "scheduled_time": _as_text(
            _first_value(sources, ("scheduled_time", "match_time", "time"))
        ),
        "tournament": _as_text(
            _first_value(
                sources,
                ("tournament", "tourney_name", "event_name", "event"),
            )
        ),
        "level": _as_text(
            _first_value(
                sources,
                ("canonical_tour_level", "tour_level", "category", "level"),
            )
        ),
        "gender": _as_text(_first_value(sources, ("gender",))),
        "surface": _as_text(_first_value(sources, ("surface",))),
        "status": _as_text(
            _first_value(sources, ("source_status", "match_status", "status"))
        ),
        "player_a": {"name": player_a, "slug": player_a_slug},
        "player_b": {"name": player_b, "slug": player_b_slug},
        "predicted": {
            "winner": None if unavailable else predicted_winner,
            "winner_side": None if unavailable else predicted_side,
            "probability": None if unavailable else predicted_probability,
            "probability_a": None if unavailable else probability_a,
            "probability_b": None if unavailable else probability_b,
            "status": prediction_status,
        },
        "reliability": {
            "label": _as_text(
                _row_value(
                    prediction,
                    ("confidence", "confidence_label", "reliability_label"),
                )
            ),
            "score": _as_number(
                _row_value(
                    prediction,
                    ("confidence_score", "reliability_score", "reliability"),
                )
            ),
            "flags": confidence_flags,
        },
        "actual": actual,
    }


def tennis_payload() -> dict:
    """Read the latest tennis execution from operational SQLite in read-only mode."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict = {
        "generated_at_utc": generated_at,
        "database": {
            "relative_path": "TENNIS/BBDD/tennis.sqlite3",
            "exists": TENNIS_DB_PATH.exists(),
            "size_bytes": TENNIS_DB_PATH.stat().st_size if TENNIS_DB_PATH.exists() else 0,
        },
        "health": "missing",
        "notes": [],
        "latest_run": None,
        "matches": [],
    }
    if not TENNIS_DB_PATH.exists():
        payload["notes"].append(
            "TENNIS/BBDD/tennis.sqlite3 no existe; la sección se publica vacía."
        )
        return payload

    database_uri = TENNIS_DB_PATH.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(database_uri, uri=True)
    except sqlite3.Error as exc:
        payload["health"] = "error"
        payload["notes"].append(f"No se pudo abrir SQLite en modo read-only: {exc}")
        return payload
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        schema = _sqlite_schema(conn)
        payload["tables"] = sorted(schema)
        official_rows, latest_run, notes = _official_tennis_rows(conn, schema)
        payload["notes"].extend(notes)
        payload["latest_run"] = (
            {
                "run_id": _as_text(
                    _row_value(
                        latest_run,
                        ("run_id", "id", "prediction_run_id", "execution_id"),
                    )
                ),
                "date": _as_text(
                    _row_value(
                        latest_run,
                        (
                            "target_date",
                            "prediction_date",
                            "run_date",
                            "match_date",
                            "date",
                            "jornada",
                        ),
                    )
                ),
                "status": _as_text(_row_value(latest_run, ("status", "run_status"))),
                "started_at_utc": _as_text(
                    _row_value(
                        latest_run,
                        ("started_at_utc", "created_at_utc", "generated_at_utc"),
                    )
                ),
                "finished_at_utc": _as_text(
                    _row_value(
                        latest_run,
                        ("finished_at_utc", "completed_at_utc", "published_at_utc"),
                    )
                ),
            }
            if latest_run
            else None
        )
        normalized_matches = [
            _tennis_match_payload(row, row, row) for row in official_rows
        ]
        payload["matches"] = normalized_matches
        payload["health"] = (
            "ok"
            if normalized_matches
            else "unavailable" if notes and not latest_run else "empty"
        )
        if not normalized_matches:
            payload["notes"].append(
                "No hay predicciones oficiales de la última ejecución para mostrar."
            )
    except sqlite3.Error as exc:
        payload["health"] = "error"
        payload["notes"].append(f"SQLite de tenis no pudo consultarse: {exc}")
    finally:
        conn.close()
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
        "tennis": tennis_payload(),
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

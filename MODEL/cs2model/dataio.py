"""Carga y normalización de resultados scrapeados de HLTV.

Entrada: `results_all.json` (lista de series tal como las devuelve el scraper).
Salida: lista de dicts normalizados, ordenados cronológicamente, con la verdad
de cada serie (quién ganó, marcador, evento, fecha). Es la capa "hechos
inmutables" sobre la que se calcula todo lo demás point-in-time.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"

try:
    from DAILY_SNAPSHOTS.match_context import parse_match_context_meta
except Exception:  # pragma: no cover - train can still run from MODEL cwd
    parse_match_context_meta = None

# Formatos de serie válidos. El scraper a veces mete el nombre de un mapa
# (bo1 jugado a mapa único) o marcadores de no-jugado.
SERIES_FORMATS = {"bo1", "bo3", "bo5"}
SINGLE_MAP_TOKENS = {
    "anc", "anubis", "anb", "d2", "dust2", "inf", "inferno", "mrg", "mirage",
    "nuke", "ovp", "overpass", "trn", "train", "vtg", "vertigo", "cbl", "cache",
}
NON_PLAYED = {"def", "cch", "-", ""}

# Inicio de la era CS2 (MR12). PROJECT.md §4.2: entrenar solo era CS2.
CS2_ERA_START = "2023-10-01"


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_team(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _odds_average(point: dict[str, Any] | None) -> dict[str, Any]:
    if not point:
        return {}
    return point.get("average") or point


def _asset_match_info(master_path: str | Path, rec: dict[str, Any]) -> dict[str, Any] | None:
    payload = _asset_payload(master_path, rec)
    if not payload:
        return None
    mapstats = payload.get("mapstats") or []
    if not mapstats:
        return None
    info = (mapstats[0].get("info") or {}) if isinstance(mapstats[0], dict) else {}
    left = info.get("team_left") or {}
    right = info.get("team_right") or {}
    if not left.get("name") or not right.get("name"):
        return None
    return {
        "team1": {"name": left.get("name"), "id": left.get("hltv_id")},
        "team2": {"name": right.get("name"), "id": right.get("hltv_id")},
    }


def _asset_payload(master_path: str | Path, rec: dict[str, Any]) -> dict[str, Any] | None:
    meta = rec.get("hltv_assets") or {}
    source_file = meta.get("source_file")
    if not source_file:
        return None
    daily_root = Path(master_path).resolve().parents[1]
    path = daily_root / source_file
    if not path.exists():
        return None
    try:
        return read_json(path, {})
    except Exception:
        return None


def _analytics_payload(master_path: str | Path, rec: dict[str, Any]) -> dict[str, Any] | None:
    payload = rec.get("analytics")
    if isinstance(payload, dict) and payload.get("available"):
        return payload
    source_file = rec.get("latest_analytics_file")
    if not source_file:
        return None
    daily_root = Path(master_path).resolve().parents[1]
    path = daily_root / source_file
    if not path.exists():
        return None
    try:
        payload = read_json(path, {})
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload.get("available") else None


def _analytics_is_point_in_time(payload: dict[str, Any] | None, rec: dict[str, Any]) -> bool:
    if not payload:
        return False
    captured_at = payload.get("captured_at")
    match_date = rec.get("date")
    if not captured_at or not match_date:
        return False
    # Conservative date-level guard. The scraper captures analytics for upcoming
    # matches; if a payload appears after the match date, do not train on it.
    return str(captured_at)[:10] <= str(match_date)[:10]


def _match_context_payload(rec: dict[str, Any], asset_payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    context = rec.get("match_context")
    if isinstance(context, dict) and context:
        raw_meta = context.get("raw_meta")
        if raw_meta and parse_match_context_meta:
            try:
                return parse_match_context_meta(raw_meta)
            except Exception:
                return context
        return context
    veto = (asset_payload or {}).get("veto") or {}
    if isinstance(veto, dict):
        context = veto.get("context")
        if isinstance(context, dict) and context:
            return context
        meta = veto.get("meta")
        if meta and parse_match_context_meta:
            try:
                return parse_match_context_meta(meta)
            except Exception:
                return None
    return None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[: len(fmt) + 2], fmt) if "T" in value else datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", ""))
    except ValueError:
        return None


def normalize_format(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in SERIES_FORMATS:
        return value
    if value in SINGLE_MAP_TOKENS:
        return "bo1"
    return "other"


def load_results(
    path: str | Path,
    cs2_only: bool = True,
    min_date: str | None = None,
) -> list[dict[str, Any]]:
    """Normaliza `results_all.json` a una lista de series cronológica.

    Cada fila: id, date, date_obj, event, format, team1/team2 (+ keys),
    score1/score2, team1_win (1/0). Se descartan filas sin marcador válido,
    empates imposibles y formatos no reconocidos.
    """
    raw = read_json(path, [])
    floor = min_date or (CS2_ERA_START if cs2_only else None)
    rows: list[dict[str, Any]] = []
    for item in raw:
        t1 = item.get("team1") or {}
        t2 = item.get("team2") or {}
        s1 = _parse_int(t1.get("score"))
        s2 = _parse_int(t2.get("score"))
        if s1 is None or s2 is None or s1 == s2:
            continue
        raw_map = (item.get("map") or "").strip().lower()
        if raw_map in NON_PLAYED:
            continue
        fmt = normalize_format(item.get("map"))
        if fmt == "other":
            continue
        date_str = item.get("date")
        if floor and date_str and date_str < floor:
            continue
        name1, name2 = t1.get("name"), t2.get("name")
        if not name1 or not name2:
            continue
        rows.append(
            {
                "id": str(item.get("id")),
                "date": date_str,
                "date_obj": parse_date(date_str),
                "event": item.get("event") or "",
                "link": item.get("link") or "",
                "format": fmt,
                "team1": name1,
                "team2": name2,
                "team1_key": clean_team(name1),
                "team2_key": clean_team(name2),
                "team1_id": str(t1.get("id")) if t1.get("id") else None,
                "team2_id": str(t2.get("id")) if t2.get("id") else None,
                "score1": s1,
                "score2": s2,
                "team1_win": int(s1 > s2),
            }
        )
    rows.sort(key=lambda r: (r["date_obj"] or datetime.min, r["id"]))
    return rows


def load_daily_completed(master_path: str | Path) -> list[dict[str, Any]]:
    """Series COMPLETADAS del master diario (con resultado y, a veces, odds).

    Extiende el histórico hasta la fecha más reciente capturada por el pipeline
    diario y aporta cuotas de apertura cuando están disponibles. Mismo esquema
    de fila que `load_results`, con campos extra `opening_odds_*` si hay.
    """
    master = read_json(master_path, {})
    rows: list[dict[str, Any]] = []
    for mid, rec in (master or {}).items():
        if rec.get("status") != "completed" or not rec.get("score"):
            continue
        detail = rec.get("detail") or {}
        match = detail.get("match") or {}
        t1 = (match.get("team1") or {}).get("name")
        t2 = (match.get("team2") or {}).get("name")
        asset_info = None
        asset_payload = _asset_payload(master_path, rec)
        analytics_payload = _analytics_payload(master_path, rec)
        context_payload = _match_context_payload(rec, asset_payload)
        if not _analytics_is_point_in_time(analytics_payload, rec):
            analytics_payload = None
        if not t1 or not t2:
            asset_info = _asset_match_info(master_path, rec)
            if asset_info:
                match = asset_info
                t1 = (match.get("team1") or {}).get("name")
                t2 = (match.get("team2") or {}).get("name")
        if not t1 or not t2:
            continue
        score = rec["score"]
        s1, s2 = _parse_int(score.get("team1")), _parse_int(score.get("team2"))
        if s1 is None or s2 is None or s1 == s2:
            continue
        fmt = normalize_format(rec.get("format"))
        if fmt == "other":
            fmt = "bo3"
        # Apertura: preferimos opening_odds; si falta, el primer punto de
        # odds_history. Nunca usamos closing_odds como feature de entrenamiento.
        opening_avg = _odds_average(rec.get("opening_odds"))
        if opening_avg.get("team1_implied_prob_norm") is None:
            oh = rec.get("odds_history") or []
            if oh:
                first_avg = _odds_average(oh[0])
                if first_avg.get("team1_implied_prob_norm") is not None:
                    opening_avg = first_avg
        row = {
            "id": str(mid),
            "date": rec.get("date"),
            "date_obj": parse_date(rec.get("date")),
            "event": rec.get("event") or "",
            "link": rec.get("link") or "",
            "format": fmt,
            "team1": t1,
            "team2": t2,
            "team1_key": clean_team(t1),
            "team2_key": clean_team(t2),
            "team1_id": str((match.get("team1") or {}).get("id")) if (match.get("team1") or {}).get("id") else None,
            "team2_id": str((match.get("team2") or {}).get("id")) if (match.get("team2") or {}).get("id") else None,
            "score1": s1,
            "score2": s2,
            "team1_win": int(s1 > s2),
            "opening_odds_t1": opening_avg.get("team1_implied_prob_norm"),
            "opening_odds_t2": opening_avg.get("team2_implied_prob_norm"),
            "opening_odds_decimal_t1": opening_avg.get("team1_decimal"),
            "opening_odds_decimal_t2": opening_avg.get("team2_decimal"),
            "opening_odds_captured_at": opening_avg.get("captured_at"),
            "opening_bookmaker_count": opening_avg.get("bookmaker_count"),
            "asset": asset_payload,
            "analytics": analytics_payload,
            "match_context": context_payload,
        }
        rows.append(row)
    return rows


def _json_or_none(value: str | None) -> Any | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _latest_snapshot_by_match(conn: sqlite3.Connection, kind: str) -> dict[str, Any]:
    try:
        cur = conn.execute(
            """
            SELECT hltv_match_id, payload_json
            FROM raw_snapshots rs
            WHERE kind = ?
              AND hltv_match_id IS NOT NULL
              AND captured_at_utc = (
                  SELECT MAX(captured_at_utc)
                  FROM raw_snapshots newer
                  WHERE newer.kind = rs.kind
                    AND newer.hltv_match_id = rs.hltv_match_id
              )
            """,
            (kind,),
        )
    except sqlite3.Error:
        return {}
    out: dict[str, Any] = {}
    for hltv_match_id, payload_json in cur.fetchall():
        payload = _json_or_none(payload_json)
        if payload is not None:
            out[str(hltv_match_id)] = payload
    return out


def _opening_odds_by_match(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT match_id, bookmaker, captured_at_utc, odds_t1, odds_t2, prob_t1, prob_t2
            FROM odds
            WHERE market_type = 'opening'
              AND (prob_t1 IS NOT NULL OR odds_t1 IS NOT NULL)
            ORDER BY match_id, captured_at_utc, bookmaker
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["match_id"]), []).append(row)
    out: dict[int, dict[str, Any]] = {}
    for match_id, items in grouped.items():
        probs1: list[float] = []
        probs2: list[float] = []
        odds1: list[float] = []
        odds2: list[float] = []
        bookmakers: set[str] = set()
        captured_at = items[0]["captured_at_utc"]
        for item in items:
            if item["captured_at_utc"] != captured_at:
                break
            if item["bookmaker"]:
                bookmakers.add(str(item["bookmaker"]))
            if item["prob_t1"] is not None:
                probs1.append(float(item["prob_t1"]))
            if item["prob_t2"] is not None:
                probs2.append(float(item["prob_t2"]))
            if item["odds_t1"] is not None:
                odds1.append(float(item["odds_t1"]))
            if item["odds_t2"] is not None:
                odds2.append(float(item["odds_t2"]))
        out[match_id] = {
            "team1_implied_prob_norm": sum(probs1) / len(probs1) if probs1 else None,
            "team2_implied_prob_norm": sum(probs2) / len(probs2) if probs2 else None,
            "team1_decimal": sum(odds1) / len(odds1) if odds1 else None,
            "team2_decimal": sum(odds2) / len(odds2) if odds2 else None,
            "captured_at": captured_at,
            "bookmaker_count": len(bookmakers) or len(items),
        }
    return out


PLAYER_TIME_FILTER_PRIORITY = ("past3months", "past6months", "past12months")
PLAYER_METRICS = ("rating", "kpr", "kast", "adr", "impact", "round_swing", "opening_kpr")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return parse_date(str(value))


def _pick_player_snapshot(rows: list[sqlite3.Row], match_dt: datetime | None) -> sqlite3.Row | None:
    if not rows:
        return None
    year_key = str(match_dt.year) if match_dt else ""
    priority = (*PLAYER_TIME_FILTER_PRIORITY, year_key)
    by_filter: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_filter.setdefault(str(row["time_filter"] or row["season_year"] or ""), []).append(row)
    for time_filter in priority:
        candidates = by_filter.get(time_filter) or []
        enough = [row for row in candidates if (row["maps"] or 0) >= 5]
        if enough:
            return enough[0]
        if candidates:
            return candidates[0]
    return rows[0]


def _team_player_snapshot_summary(
    conn: sqlite3.Connection,
    team_id: int,
    match_dt_text: str,
) -> dict[str, Any]:
    match_dt = _iso_date(match_dt_text)
    rows = conn.execute(
        """
        SELECT ps.*, p.hltv_id AS player_hltv_id
        FROM team_rosters tr
        JOIN players p ON p.player_id = tr.player_id
        JOIN player_stat_snapshots ps ON ps.hltv_player_id = CAST(p.hltv_id AS TEXT)
        WHERE tr.team_id = ?
          AND tr.valid_from <= ?
          AND (tr.valid_to IS NULL OR tr.valid_to > ?)
          AND ps.captured_at_utc IS NOT NULL
          AND ps.captured_at_utc <= ?
        ORDER BY ps.hltv_player_id, ps.captured_at_utc DESC, ps.player_stat_snapshot_id DESC
        """,
        (team_id, match_dt_text, match_dt_text, match_dt_text),
    ).fetchall()
    by_player: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_player.setdefault(str(row["hltv_player_id"]), []).append(row)
    selected = [_pick_player_snapshot(player_rows, match_dt) for player_rows in by_player.values()]
    selected = [row for row in selected if row is not None]
    roster_size = max(len(by_player), 5)
    out: dict[str, Any] = {
        "coverage": len(selected) / roster_size if roster_size else 0.0,
        "players": len(selected),
        "maps_total": sum(int(row["maps"] or 0) for row in selected),
        "age_days_max": 0.0,
        "metrics": {},
    }
    if match_dt:
        ages = []
        for row in selected:
            captured = _iso_date(row["captured_at_utc"])
            if captured:
                ages.append(max(0.0, (match_dt - captured).total_seconds() / 86400.0))
        out["age_days_max"] = max(ages) if ages else 0.0
    for metric in PLAYER_METRICS:
        values = [_safe_float(row[metric]) for row in selected]
        values = [value for value in values if value is not None]
        if not values:
            continue
        avg = sum(values) / len(values)
        out["metrics"][metric] = {
            "avg": avg,
            "max": max(values),
            "min": min(values),
            "std": math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)) if len(values) > 1 else 0.0,
            "spread": max(values) - min(values),
        }
    return out


def _metric(summary: dict[str, Any], metric: str, field: str = "avg", default: float = 0.0) -> float:
    return float(((summary.get("metrics") or {}).get(metric) or {}).get(field, default) or default)


def _player_snapshot_features(
    conn: sqlite3.Connection,
    match_id: int,
    team1_id: int,
    team2_id: int,
    match_dt_text: str,
) -> dict[str, float]:
    t1 = _team_player_snapshot_summary(conn, team1_id, match_dt_text)
    t2 = _team_player_snapshot_summary(conn, team2_id, match_dt_text)
    coverage_min = min(float(t1.get("coverage") or 0.0), float(t2.get("coverage") or 0.0))
    rating1 = _metric(t1, "rating", "avg", 1.0)
    rating2 = _metric(t2, "rating", "avg", 1.0)
    rating1_max = _metric(t1, "rating", "max", rating1)
    rating2_max = _metric(t2, "rating", "max", rating2)
    rating1_min = _metric(t1, "rating", "min", rating1)
    rating2_min = _metric(t2, "rating", "min", rating2)
    return {
        "player_snapshot_available": 1.0 if coverage_min >= 0.8 else 0.0,
        "player_coverage_min": coverage_min,
        "player_maps_min": float(min(t1.get("maps_total") or 0, t2.get("maps_total") or 0)),
        "player_rating_diff": rating1 - rating2,
        "player_rating_max_diff": rating1_max - rating2_max,
        "player_rating_min_diff": rating1_min - rating2_min,
        "player_rating_std_diff": _metric(t1, "rating", "std", 0.0) - _metric(t2, "rating", "std", 0.0),
        "player_rating_spread_diff": _metric(t1, "rating", "spread", 0.0) - _metric(t2, "rating", "spread", 0.0),
        "player_star_gap_diff": (rating1_max - rating1) - (rating2_max - rating2),
        "player_weak_link_gap_diff": (rating1 - rating1_min) - (rating2 - rating2_min),
        "player_kpr_diff": _metric(t1, "kpr", "avg", 0.0) - _metric(t2, "kpr", "avg", 0.0),
        "player_kast_diff": _metric(t1, "kast", "avg", 70.0) - _metric(t2, "kast", "avg", 70.0),
        "player_adr_diff": _metric(t1, "adr", "avg", 75.0) - _metric(t2, "adr", "avg", 75.0),
        "player_impact_diff": _metric(t1, "impact", "avg", 1.0) - _metric(t2, "impact", "avg", 1.0),
        "player_round_swing_diff": _metric(t1, "round_swing", "avg", 0.0) - _metric(t2, "round_swing", "avg", 0.0),
        "player_opening_kpr_diff": _metric(t1, "opening_kpr", "avg", 0.0) - _metric(t2, "opening_kpr", "avg", 0.0),
    }


def load_external_feature_store(conn_or_path: sqlite3.Connection | str | Path) -> dict[str, Any]:
    """Carga rankings y rosters una vez para joins point-in-time eficientes."""
    owns_connection = not isinstance(conn_or_path, sqlite3.Connection)
    conn = sqlite3.connect(conn_or_path) if owns_connection else conn_or_path
    conn.row_factory = sqlite3.Row
    try:
        ranking_index: dict[tuple[str, str], dict[str, list[Any]]] = {}
        for row in conn.execute(
            """
            SELECT hltv_team_id, ranking_type, position, points, captured_at_utc
            FROM team_ranking_snapshots
            WHERE hltv_team_id IS NOT NULL AND captured_at_utc IS NOT NULL
            ORDER BY hltv_team_id, ranking_type, captured_at_utc, team_ranking_snapshot_id
            """
        ):
            captured = _iso_date(row["captured_at_utc"])
            if captured is None:
                continue
            key = (str(row["hltv_team_id"]), str(row["ranking_type"]))
            bucket = ranking_index.setdefault(key, {"dates": [], "rows": []})
            bucket["dates"].append(captured)
            bucket["rows"].append(dict(row))

        roster_index: dict[str, list[dict[str, Any]]] = {}
        for row in conn.execute(
            """
            SELECT CAST(t.hltv_id AS TEXT) AS hltv_team_id, tr.player_id,
                   tr.valid_from, tr.valid_to
            FROM team_rosters tr
            JOIN teams t ON t.team_id = tr.team_id
            WHERE t.hltv_id IS NOT NULL
            ORDER BY t.hltv_id, tr.valid_from, tr.player_id
            """
        ):
            roster_index.setdefault(str(row["hltv_team_id"]), []).append(dict(row))
        return {"rankings": ranking_index, "rosters": roster_index}
    except sqlite3.Error:
        return {"rankings": {}, "rosters": {}}
    finally:
        if owns_connection:
            conn.close()


def _ranking_asof(
    store: dict[str, Any],
    hltv_team_id: str | int | None,
    ranking_type: str,
    match_dt: datetime | None,
) -> dict[str, Any] | None:
    if hltv_team_id is None or match_dt is None:
        return None
    bucket = (store.get("rankings") or {}).get((str(hltv_team_id), ranking_type))
    if not bucket:
        return None
    index = bisect_right(bucket["dates"], match_dt) - 1
    return bucket["rows"][index] if index >= 0 else None


def _active_roster_asof(
    store: dict[str, Any],
    hltv_team_id: str | int | None,
    match_dt: datetime | None,
) -> dict[str, float]:
    if hltv_team_id is None or match_dt is None:
        return {"size": 0.0, "days": 0.0, "standin_risk": 1.0}
    active: list[tuple[dict[str, Any], datetime]] = []
    for row in (store.get("rosters") or {}).get(str(hltv_team_id), []):
        valid_from = _iso_date(row.get("valid_from"))
        valid_to = _iso_date(row.get("valid_to"))
        if valid_from and valid_from <= match_dt and (valid_to is None or valid_to > match_dt):
            active.append((row, valid_from))
    if not active:
        return {"size": 0.0, "days": 0.0, "standin_risk": 1.0}
    size = float(len({int(row["player_id"]) for row, _valid_from in active}))
    newest_member = max(valid_from for _row, valid_from in active)
    days = max(0.0, (match_dt - newest_member).total_seconds() / 86400.0)
    return {"size": size, "days": days, "standin_risk": float(size != 5.0)}


def external_snapshot_features_asof(
    store: dict[str, Any],
    team1_hltv_id: str | int | None,
    team2_hltv_id: str | int | None,
    match_dt_value: str | datetime | None,
) -> dict[str, dict[str, float]]:
    """Construye ranking/roster usando exclusivamente evidencia anterior."""
    match_dt = match_dt_value if isinstance(match_dt_value, datetime) else _iso_date(match_dt_value)
    ranking: dict[str, float] = {
        "ranking_available": 0.0,
        "ranking_sources": 0.0,
        "ranking_age_days_max": 0.0,
        "ranking_hltv_position_advantage": 0.0,
        "ranking_hltv_points_diff": 0.0,
        "ranking_valve_position_advantage": 0.0,
        "ranking_valve_points_diff": 0.0,
    }
    ages: list[float] = []
    for ranking_type in ("hltv", "valve"):
        row1 = _ranking_asof(store, team1_hltv_id, ranking_type, match_dt)
        row2 = _ranking_asof(store, team2_hltv_id, ranking_type, match_dt)
        if not row1 or not row2:
            continue
        pos1 = _safe_float(row1.get("position"))
        pos2 = _safe_float(row2.get("position"))
        points1 = _safe_float(row1.get("points"))
        points2 = _safe_float(row2.get("points"))
        if pos1 is not None and pos2 is not None:
            ranking[f"ranking_{ranking_type}_position_advantage"] = pos2 - pos1
        if points1 is not None and points2 is not None:
            ranking[f"ranking_{ranking_type}_points_diff"] = points1 - points2
        ranking["ranking_sources"] += 1.0
        if match_dt:
            for row in (row1, row2):
                captured = _iso_date(row.get("captured_at_utc"))
                if captured:
                    ages.append(max(0.0, (match_dt - captured).total_seconds() / 86400.0))
    ranking["ranking_available"] = float(ranking["ranking_sources"] > 0)
    ranking["ranking_age_days_max"] = max(ages) if ages else 0.0

    roster1 = _active_roster_asof(store, team1_hltv_id, match_dt)
    roster2 = _active_roster_asof(store, team2_hltv_id, match_dt)
    roster_available = roster1["size"] >= 5.0 and roster2["size"] >= 5.0
    roster = {
        "roster_available": float(roster_available),
        "roster_days_log_diff": (
            math.log1p(roster1["days"]) - math.log1p(roster2["days"])
            if roster_available else 0.0
        ),
        "roster_size_diff": roster1["size"] - roster2["size"] if roster_available else 0.0,
        "roster_standin_risk_advantage": (
            roster2["standin_risk"] - roster1["standin_risk"]
            if roster_available else 0.0
        ),
        "roster_days_min": min(roster1["days"], roster2["days"]) if roster_available else 0.0,
        "roster_size_min": min(roster1["size"], roster2["size"]) if roster_available else 0.0,
    }
    return {"ranking_snapshot_features": ranking, "roster_snapshot_features": roster}


def _looks_like_hltv_match_id(value: Any) -> bool:
    try:
        return int(str(value)) >= 1_000_000
    except (TypeError, ValueError):
        return False


def deduplicate_cross_source_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Elimina duplicados semilla+HLTV sin borrar evidencia de SQLite.

    Solo colapsa un grupo cuando conviven IDs internos de la semilla e IDs HLTV.
    Si hubiera dos IDs HLTV reales para los mismos equipos el mismo dia, ambos
    se conservan para no confundir un double-header con un duplicado.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        team_score = sorted(
            (
                (clean_team(row.get("team1")), int(row.get("score1") or 0)),
                (clean_team(row.get("team2")), int(row.get("score2") or 0)),
            ),
            key=lambda item: item[0],
        )
        key = (
            str(row.get("date") or "")[:10],
            clean_team(row.get("event")),
            str(row.get("format") or ""),
            tuple(team_score),
        )
        groups.setdefault(key, []).append(row)

    deduplicated: list[dict[str, Any]] = []
    for group in groups.values():
        hltv_rows = [row for row in group if _looks_like_hltv_match_id(row.get("id"))]
        legacy_rows = [row for row in group if not _looks_like_hltv_match_id(row.get("id"))]
        deduplicated.extend(hltv_rows if hltv_rows and legacy_rows else group)
    deduplicated.sort(key=lambda row: (row.get("date_obj") or datetime.min, str(row.get("id") or "")))
    return deduplicated


def load_training_rows_from_db(
    db_path: str | Path = DEFAULT_DB,
    cs2_only: bool = True,
    min_date: str | None = None,
) -> list[dict[str, Any]]:
    """Carga entrenamiento desde cs2.db usando solo snapshots pre-partido."""
    path = Path(db_path)
    if not path.exists():
        return []
    floor = min_date or (CS2_ERA_START if cs2_only else None)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    try:
        odds_by_match = _opening_odds_by_match(conn)
        assets_by_hltv = _latest_snapshot_by_match(conn, "match_assets")
        analytics_by_hltv = _latest_snapshot_by_match(conn, "match_analytics")
        rows = conn.execute(
            """
            SELECT
                m.match_id, m.hltv_match_id, m.datetime_utc, m.team1_id, m.team2_id, m.best_of, m.stage,
                m.environment, m.stage_detail, m.incentive_label, m.high_stakes,
                m.opening_match, m.winner_advances, m.loser_eliminated, m.bracket,
                m.context_json, m.score_t1, m.score_t2, m.winner_team_id,
                e.name AS event_name, t1.name AS team1_name, t2.name AS team2_name,
                t1.hltv_id AS team1_hltv_id, t2.hltv_id AS team2_hltv_id
            FROM matches m
            JOIN teams t1 ON t1.team_id = m.team1_id
            JOIN teams t2 ON t2.team_id = m.team2_id
            JOIN events e ON e.event_id = m.event_id
            WHERE m.status = 'completed'
              AND m.score_t1 IS NOT NULL
              AND m.score_t2 IS NOT NULL
              AND m.score_t1 <> m.score_t2
            ORDER BY m.datetime_utc, COALESCE(m.hltv_match_id, m.match_id)
            """
        ).fetchall()
        player_features_by_match = {
            int(row["match_id"]): _player_snapshot_features(
                conn,
                int(row["match_id"]),
                int(row["team1_id"]),
                int(row["team2_id"]),
                str(row["datetime_utc"] or ""),
            )
            for row in rows
        }
        external_store = load_external_feature_store(conn)
        external_features_by_match = {
            int(row["match_id"]): external_snapshot_features_asof(
                external_store,
                row["team1_hltv_id"],
                row["team2_hltv_id"],
                str(row["datetime_utc"] or ""),
            )
            for row in rows
        }
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        date_text = str(row["datetime_utc"] or "")[:10]
        if floor and date_text and date_text < floor:
            continue
        hltv_match_id = str(row["hltv_match_id"] or row["match_id"])
        bo = int(row["best_of"])
        fmt = f"bo{bo}" if bo in (1, 3, 5) else "other"
        if fmt == "other":
            continue
        context_payload = _json_or_none(row["context_json"]) or {
            "environment": row["environment"],
            "stage": row["stage"],
            "stage_detail": row["stage_detail"],
            "incentive_label": row["incentive_label"],
            "high_stakes": row["high_stakes"],
            "opening_match": row["opening_match"],
            "winner_advances": row["winner_advances"],
            "loser_eliminated": row["loser_eliminated"],
            "bracket": row["bracket"],
        }
        opening_avg = odds_by_match.get(int(row["match_id"]), {})
        analytics_payload = analytics_by_hltv.get(hltv_match_id)
        if not _analytics_is_point_in_time(analytics_payload, {"date": date_text}):
            analytics_payload = None
        out.append(
            {
                "id": hltv_match_id,
                "db_match_id": int(row["match_id"]),
                "date": date_text,
                "date_obj": parse_date(str(row["datetime_utc"])),
                "event": row["event_name"] or "",
                "link": f"https://www.hltv.org/matches/{hltv_match_id}/",
                "format": fmt,
                "team1": row["team1_name"],
                "team2": row["team2_name"],
                "team1_key": clean_team(row["team1_name"]),
                "team2_key": clean_team(row["team2_name"]),
                "team1_id": str(row["team1_hltv_id"]) if row["team1_hltv_id"] is not None else None,
                "team2_id": str(row["team2_hltv_id"]) if row["team2_hltv_id"] is not None else None,
                "score1": int(row["score_t1"]),
                "score2": int(row["score_t2"]),
                "team1_win": int(row["score_t1"] > row["score_t2"]),
                "opening_odds_t1": opening_avg.get("team1_implied_prob_norm"),
                "opening_odds_t2": opening_avg.get("team2_implied_prob_norm"),
                "opening_odds_decimal_t1": opening_avg.get("team1_decimal"),
                "opening_odds_decimal_t2": opening_avg.get("team2_decimal"),
                "opening_odds_captured_at": opening_avg.get("captured_at"),
                "opening_bookmaker_count": opening_avg.get("bookmaker_count"),
                "asset": assets_by_hltv.get(hltv_match_id),
                "analytics": analytics_payload,
                "match_context": context_payload,
                "player_snapshot_features": player_features_by_match.get(int(row["match_id"]), {}),
                **external_features_by_match.get(int(row["match_id"]), {}),
            }
        )
    out.sort(key=lambda r: (r["date_obj"] or datetime.min, r["id"]))
    return deduplicate_cross_source_matches(out)


def load_training_rows(
    raw_path: str | Path | None = None,
    master_path: str | Path | None = None,
    cs2_only: bool = True,
    db_path: str | Path | None = DEFAULT_DB,
) -> list[dict[str, Any]]:
    """Histórico backfill + series completadas del master (dedup por id, cronológico).

    Así el dataset de entrenamiento se extiende automáticamente hasta la última
    fecha capturada por el pipeline diario, sin re-scrapear el histórico.
    """
    if not raw_path and db_path:
        db_rows = load_training_rows_from_db(db_path, cs2_only=cs2_only)
        if db_rows:
            return db_rows
    if not raw_path:
        return []
    rows = load_results(raw_path, cs2_only=cs2_only)
    seen = {r["id"] for r in rows}
    if master_path:
        for r in load_daily_completed(master_path):
            if r["id"] not in seen:
                rows.append(r)
                seen.add(r["id"])
    rows.sort(key=lambda r: (r["date_obj"] or datetime.min, r["id"]))
    return rows

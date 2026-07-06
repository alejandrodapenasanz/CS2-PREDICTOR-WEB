"""Carga y normalización de resultados scrapeados de HLTV.

Entrada: `results_all.json` (lista de series tal como las devuelve el scraper).
Salida: lista de dicts normalizados, ordenados cronológicamente, con la verdad
de cada serie (quién ganó, marcador, evento, fecha). Es la capa "hechos
inmutables" sobre la que se calcula todo lo demás point-in-time.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def load_training_rows(
    raw_path: str | Path,
    master_path: str | Path | None = None,
    cs2_only: bool = True,
) -> list[dict[str, Any]]:
    """Histórico backfill + series completadas del master (dedup por id, cronológico).

    Así el dataset de entrenamiento se extiende automáticamente hasta la última
    fecha capturada por el pipeline diario, sin re-scrapear el histórico.
    """
    rows = load_results(raw_path, cs2_only=cs2_only)
    seen = {r["id"] for r in rows}
    if master_path:
        for r in load_daily_completed(master_path):
            if r["id"] not in seen:
                rows.append(r)
                seen.add(r["id"])
    rows.sort(key=lambda r: (r["date_obj"] or datetime.min, r["id"]))
    return rows

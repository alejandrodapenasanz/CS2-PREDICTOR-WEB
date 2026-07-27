from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from parsel import Selector


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Motor de modelo entrenado (MODEL/cs2model). Carga perezosa y tolerante a
# fallos: si el artefacto o la librería no están disponibles, el pipeline cae
# de forma segura al scorer logístico ligero (logistic_probability).
# ---------------------------------------------------------------------------
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
try:
    from cs2model import dataio as cs2_dataio
    from cs2model.features import (
        build_training_frame as cs2_build_state,
        analytics_match_features as cs2_analytics_match_features,
        announced_lineup_features as cs2_announced_lineup_features,
        context_match_features as cs2_context_match_features,
        event_metadata_features as cs2_event_metadata_features,
        PLAYER_DIFF_COLUMNS as CS2_PLAYER_DIFF_COLUMNS,
        PLAYER_SYM_COLUMNS as CS2_PLAYER_SYM_COLUMNS,
        RANKING_DIFF_COLUMNS as CS2_RANKING_DIFF_COLUMNS,
        RANKING_SYM_COLUMNS as CS2_RANKING_SYM_COLUMNS,
        ROSTER_DIFF_COLUMNS as CS2_ROSTER_DIFF_COLUMNS,
        ROSTER_SYM_COLUMNS as CS2_ROSTER_SYM_COLUMNS,
    )
    from cs2model.artifacts import load_artifact as cs2_load_artifact

    _CS2_OK = True
except Exception:  # pragma: no cover - entorno sin librería
    _CS2_OK = False

    def cs2_analytics_match_features(_match: dict[str, Any]) -> dict[str, float]:
        return {}

    def cs2_announced_lineup_features(_match: dict[str, Any]) -> dict[str, float]:
        return {}

    def cs2_context_match_features(_match: dict[str, Any]) -> dict[str, float]:
        return {}

    def cs2_event_metadata_features(_match: dict[str, Any]) -> dict[str, float]:
        return {}
    CS2_PLAYER_DIFF_COLUMNS = []
    CS2_PLAYER_SYM_COLUMNS = []
    CS2_RANKING_DIFF_COLUMNS = []
    CS2_RANKING_SYM_COLUMNS = []
    CS2_ROSTER_DIFF_COLUMNS = []
    CS2_ROSTER_SYM_COLUMNS = []
DAILY_ROOT = ROOT / "PIPELINE"
RUNS_DIR = DAILY_ROOT / "runs"
MASTER_MANIFEST = DAILY_ROOT / "master" / "manifest.json"
MASTER_MATCHES = DAILY_ROOT / "master" / "matches.json"
MASTER_ROSTERS = DAILY_ROOT / "master" / "roster_history.json"
MASTER_CALIBRATION = DAILY_ROOT / "master" / "calibration.json"
LIVE_DB = ROOT / "BBDD" / "cs2.db"
ROSTER_CHANGE_WINDOW_DAYS = 90
COMPLETE_LINEUP_SIZE = 5

try:
    from PIPELINE.match_context import parse_match_context_meta
    from PIPELINE.opportunity import annotate_opportunities
except Exception:  # pragma: no cover - direct script execution
    from match_context import parse_match_context_meta
    from opportunity import annotate_opportunities
DEFAULT_HISTORY = (
    ROOT
    / "SCRAPER"
    / "hltv-scraper-api"
    / "hltv_scraper"
    / "data"
    / "raw"
    / "history_10000_2026-06-28"
    / "results_all.json"
)


NON_PLAYED = {"def", "cch"}
MAP_POOL = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"]
MARKET_PRIOR_MIN_WEIGHT = 0.05
MARKET_PRIOR_MAX_WEIGHT = 0.30
MARKET_BLEND_LEARN_MIN_SAMPLES = 120
MARKET_BLEND_WALK_FORWARD_MIN_TRAIN = 80
MARKET_BLEND_CANDIDATE_WEIGHTS = [i / 100 for i in range(0, 51, 5)]
MAP_ALIASES = {
    "anc": "Ancient",
    "ancient": "Ancient",
    "anb": "Anubis",
    "anubis": "Anubis",
    "d2": "Dust2",
    "dust2": "Dust2",
    "dust 2": "Dust2",
    "inf": "Inferno",
    "inferno": "Inferno",
    "mrg": "Mirage",
    "mirage": "Mirage",
    "nuke": "Nuke",
    "ovp": "Overpass",
    "overpass": "Overpass",
    "trn": "Train",
    "train": "Train",
    "vtg": "Vertigo",
    "vertigo": "Vertigo",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_team(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_date(date_text: str | None) -> datetime | None:
    if not date_text:
        return None
    try:
        return datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return None


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        return None


def parse_iso_as_local(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    try:
        return parsed.astimezone(ZoneInfo("Europe/Madrid")).replace(tzinfo=None)
    except Exception:
        return parsed.astimezone().replace(tzinfo=None)


def parse_match_datetime(date_text: str | None, hour_text: str | None = None) -> datetime | None:
    date_obj = parse_date(date_text)
    if not date_obj:
        return None
    if hour_text:
        match = re.search(r"(\d{1,2}):(\d{2})", hour_text)
        if match:
            return date_obj.replace(hour=int(match.group(1)), minute=int(match.group(2)))
    return date_obj.replace(hour=12, minute=0)


def detail_has_completed_score(detail: dict[str, Any]) -> bool:
    match = detail.get("match") or {}
    team1 = match.get("team1") or {}
    team2 = match.get("team2") or {}
    return str(team1.get("score") or "").isdigit() and str(team2.get("score") or "").isdigit()


def is_snapshot_publishable(snapshot: dict[str, Any], run_started_local: datetime, grace_minutes: int = 15) -> tuple[bool, str]:
    if snapshot.get("status") == "completed":
        return False, "completed"
    detail = snapshot.get("detail") or {}
    if isinstance(detail, dict) and detail_has_completed_score(detail):
        return False, "completed_detail"
    match_dt = parse_match_datetime(snapshot.get("date"), snapshot.get("hour"))
    if match_dt and match_dt < run_started_local - timedelta(minutes=grace_minutes):
        return False, "already_started"
    return True, ""


def normalize_map_name(value: str | None) -> str | None:
    if not value:
        return None
    key = " ".join(value.strip().lower().replace("-", " ").split())
    return MAP_ALIASES.get(key)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().replace("%", "").replace(",", "."))
    except ValueError:
        return None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def latest_run_dir() -> Path:
    manifest = read_json(MASTER_MANIFEST, {})
    if manifest.get("last_run_id"):
        return RUNS_DIR / manifest["last_run_id"]
    runs = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()])
    if not runs:
        raise FileNotFoundError("No daily runs found. Execute PIPELINE/start.py first.")
    return runs[-1]


def normalize_format(raw: str | None) -> str:
    value = (raw or "").lower().strip()
    if value in {"bo1", "bo3", "bo5"}:
        return value
    return "other"


def load_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_json(path, []):
        score1 = parse_int((item.get("team1") or {}).get("score"))
        score2 = parse_int((item.get("team2") or {}).get("score"))
        if score1 is None or score2 is None or score1 == score2:
            continue
        fmt = normalize_format(item.get("map"))
        raw_map = (item.get("map") or "").lower()
        if raw_map in NON_PLAYED or fmt == "other" and raw_map not in {"anc", "d2", "inf", "mrg", "nuke", "ovp"}:
            continue
        rows.append(
            {
                "id": str(item.get("id")),
                "date": item.get("date"),
                "date_obj": parse_date(item.get("date")),
                "team1": (item.get("team1") or {}).get("name"),
                "team2": (item.get("team2") or {}).get("name"),
                "team1_key": clean_team((item.get("team1") or {}).get("name")),
                "team2_key": clean_team((item.get("team2") or {}).get("name")),
                "score1": score1,
                "score2": score2,
                "winner_key": clean_team((item.get("team1") if score1 > score2 else item.get("team2") or {}).get("name")),
                "format": fmt if fmt != "other" else "bo1",
                "event": item.get("event") or "",
            }
        )
    rows.sort(key=lambda row: (row["date_obj"] or datetime.min, row["id"]))
    return rows


def elo_prob(elo1: float, elo2: float) -> float:
    return 1 / (1 + 10 ** (-(elo1 - elo2) / 400))


def update_elo(elo1: float, elo2: float, team1_win: bool, k: float = 32.0) -> tuple[float, float]:
    p1 = elo_prob(elo1, elo2)
    actual = 1.0 if team1_win else 0.0
    return elo1 + k * (actual - p1), elo2 + k * ((1 - actual) - (1 - p1))


def winrate(wins: list[int], last: int | None = None) -> float:
    hist = wins if last is None else wins[-last:]
    return (sum(hist) + 1) / (len(hist) + 2)


def avg(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def build_history_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elos = defaultdict(lambda: 1500.0)
    wins = defaultdict(list)
    diffs = defaultdict(list)
    matches = defaultdict(int)
    match_dates = defaultdict(list)
    opponent_elos = defaultdict(list)
    event_stats = defaultdict(lambda: {"matches": 0, "wins": 0})
    event_outcomes = defaultdict(list)

    for row in rows:
        t1, t2 = row["team1_key"], row["team2_key"]
        pre1, pre2 = elos[t1], elos[t2]
        team1_win = row["score1"] > row["score2"]

        matches[t1] += 1
        matches[t2] += 1
        if row.get("date_obj"):
            match_dates[t1].append(row["date_obj"])
            match_dates[t2].append(row["date_obj"])
        wins[t1].append(int(team1_win))
        wins[t2].append(int(not team1_win))
        diffs[t1].append(row["score1"] - row["score2"])
        diffs[t2].append(row["score2"] - row["score1"])
        opponent_elos[t1].append(pre2)
        opponent_elos[t2].append(pre1)

        event = row["event"]
        event_stats[(t1, event)]["matches"] += 1
        event_stats[(t1, event)]["wins"] += int(team1_win)
        event_stats[(t2, event)]["matches"] += 1
        event_stats[(t2, event)]["wins"] += int(not team1_win)
        event_outcomes[event].append(int(team1_win))

        elos[t1], elos[t2] = update_elo(pre1, pre2, team1_win)

    return {
        "elos": elos,
        "wins": wins,
        "diffs": diffs,
        "matches": matches,
        "match_dates": match_dates,
        "opponent_elos": opponent_elos,
        "event_stats": event_stats,
        "event_outcomes": event_outcomes,
    }


PLAYER_TIME_FILTER_PRIORITY = ["past3months", "past6months", "past12months"]
PLAYER_STAT_ALIASES = {
    "rating": ("Rating 3.0", "Rating 2.1", "Rating 2.0"),
    "kpr": ("KPR",),
    "dpr": ("DPR",),
    "apr": ("APR",),
    "kast": ("KAST",),
    "impact": ("Impact",),
    "adr": ("ADR", "Average Damage per Round"),
    "round_swing": ("Round Swing",),
    "multi_kill_rating": ("Multi-kill rating",),
    "awp_kpr": ("AWP KPR",),
    "hs_pct": ("HS %",),
    "opening_kpr": ("Opening KPR",),
    "opening_dpr": ("Opening DPR",),
    "flash_assists": ("Flash assists",),
}
PLAYER_LEGACY_NAMES = {
    "rating": "Rating 3.0",
    "kpr": "KPR",
    "dpr": "DPR",
    "apr": "APR",
    "kast": "KAST",
    "impact": "Impact",
    "adr": "ADR",
    "round_swing": "Round Swing",
    "multi_kill_rating": "Multi-kill rating",
    "awp_kpr": "AWP KPR",
    "hs_pct": "HS %",
    "opening_kpr": "Opening KPR",
    "opening_dpr": "Opening DPR",
    "flash_assists": "Flash assists",
}


def load_player_stats(run_dir: Path) -> dict[str, dict[str, Any]]:
    files = sorted(run_dir.glob("player_compare_stats_*.json"))
    if not files:
        return {}
    data = read_json(files[-1], {})
    players = {}
    for comp in data.get("results", []):
        comp_time_filter = str(comp.get("time_filter") or data.get("time_filter") or data.get("year") or "unknown")
        for player in comp.get("players", []):
            player_id = str(player.get("id"))
            if not player_id:
                continue
            time_filter = str(
                player.get("time_filter")
                or player.get("selected_from_time_filter")
                or comp_time_filter
            )
            bucket = players.setdefault(
                player_id,
                {
                    "id": player_id,
                    "name": player.get("name"),
                    "slug": player.get("slug"),
                    "link": player.get("link"),
                    "windows": {},
                },
            )
            bucket["windows"][time_filter] = player
            if not bucket.get("name"):
                bucket["name"] = player.get("name")
    for bucket in players.values():
        preferred = choose_player_snapshot(bucket)
        if preferred:
            bucket["preferred"] = preferred
            bucket["preferred_time_filter"] = preferred.get("time_filter")
            bucket["stats"] = preferred.get("stats") or {}
            bucket["maps"] = preferred.get("maps")
    return players


def team_profile_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for item in read_json(run_dir / "team_profiles.json", []):
        profile = item.get("profile") or {}
        team_id = str(item.get("id") or "")
        if team_id:
            index[team_id] = profile
    return index


def numeric_stat(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("%", ""))
    except ValueError:
        return None


def player_stat_value(stats: dict[str, Any], metric: str) -> float | None:
    for key in PLAYER_STAT_ALIASES.get(metric, (metric,)):
        value = numeric_stat(stats.get(key))
        if value is not None:
            return value
    return None


def choose_player_snapshot(bundle: dict[str, Any], min_maps: int = 5) -> dict[str, Any] | None:
    windows = bundle.get("windows") or {}
    years = sorted([key for key in windows if key.isdigit()], reverse=True)
    for key in [*PLAYER_TIME_FILTER_PRIORITY, *years, "allTime"]:
        player = windows.get(key)
        if not player:
            continue
        maps = parse_int(player.get("maps"))
        if maps is None or maps >= min_maps:
            return player
    return next(iter(windows.values()), None)


def aggregate_player_stats(players: list[dict[str, Any]], time_filter: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "covered_players": 0,
        "maps_total": 0,
        "maps_min": 0,
        "metrics": {},
        "top": {},
        "bottom": {},
    }
    per_metric: dict[str, list[tuple[float, str]]] = defaultdict(list)
    player_maps: list[int] = []
    for player in players:
        snapshot = None
        if time_filter:
            snapshot = (player.get("stats_by_window") or {}).get(time_filter)
        else:
            snapshot = player.get("stats")
        if not snapshot:
            continue
        if (parse_int(snapshot.get("maps")) or 0) <= 0:
            continue
        stats = snapshot.get("stats") or {}
        if stats:
            out["covered_players"] += 1
        maps = parse_int(snapshot.get("maps")) or 0
        out["maps_total"] += maps
        player_maps.append(maps)
        for metric in PLAYER_STAT_ALIASES:
            value = player_stat_value(stats, metric)
            if value is not None:
                per_metric[metric].append((value, player.get("name") or snapshot.get("name") or "?"))

    for metric, pairs in per_metric.items():
        values = [value for value, _ in pairs]
        ordered = sorted(values)
        top_two = ordered[-min(2, len(ordered)) :]
        bottom_two = ordered[: min(2, len(ordered))]
        top_value, top_name = max(pairs, key=lambda item: item[0])
        low_value, low_name = min(pairs, key=lambda item: item[0])
        avg_value = avg(values, default=None)
        std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
        out["metrics"][metric] = {
            "avg": avg_value,
            "max": top_value,
            "min": low_value,
            "top2_avg": avg(top_two, default=None),
            "bottom2_avg": avg(bottom_two, default=None),
            "median": statistics.median(ordered),
            "std": std_value,
            "spread": top_value - low_value,
            "n": len(values),
        }
        out["top"][metric] = {"name": top_name, "value": top_value}
        out["bottom"][metric] = {"name": low_name, "value": low_value}
    out["maps_min"] = min(player_maps) if player_maps else 0
    return out


def roster_metric(roster: dict[str, Any], metric: str, field: str, default: float | None = None) -> float | None:
    return (((roster.get("distribution") or {}).get("metrics") or {}).get(metric) or {}).get(field, default)


def roster_summary(team_id: str | None, profiles: dict[str, Any], player_stats: dict[str, Any]) -> dict[str, Any]:
    if not team_id or team_id not in profiles:
        return {"coverage": 0.0, "players": [], "avg": {}, "windows": {}, "distribution": {}}
    roster = profiles[team_id].get("squad") or []
    players = []
    for player in roster:
        bundle = player_stats.get(str(player.get("id"))) or {}
        preferred = bundle.get("preferred") if isinstance(bundle, dict) else bundle
        windows = bundle.get("windows", {}) if isinstance(bundle, dict) else {}
        players.append(
            {
                "id": player.get("id"),
                "name": player.get("name"),
                "stats": preferred,
                "stats_by_window": windows,
                "preferred_time_filter": bundle.get("preferred_time_filter") if isinstance(bundle, dict) else None,
            }
        )
    summary = {
        "coverage": 0.0,
        "players": players,
        "avg": {},
        "windows": {},
        "distribution": {},
        "preferred_time_filter": None,
    }
    available_windows = sorted({window for p in players for window in (p.get("stats_by_window") or {})})
    for window in [*PLAYER_TIME_FILTER_PRIORITY, *[w for w in available_windows if w not in PLAYER_TIME_FILTER_PRIORITY]]:
        window_agg = aggregate_player_stats(players, window)
        if window_agg.get("covered_players"):
            summary["windows"][window] = window_agg
    # Use one window for the whole roster: mixing a three-month sample from one
    # player with a year-long sample from another makes star and depth signals
    # incomparable. Widen only when the current window lacks a real five-man sample.
    for window in PLAYER_TIME_FILTER_PRIORITY:
        candidate = summary["windows"].get(window)
        if candidate and candidate["covered_players"] >= 4 and candidate["maps_min"] >= 5:
            summary["preferred_time_filter"] = window
            break
    if not summary["preferred_time_filter"] and summary["windows"]:
        summary["preferred_time_filter"] = max(
            summary["windows"],
            key=lambda window: (
                summary["windows"][window]["covered_players"],
                summary["windows"][window]["maps_min"],
                summary["windows"][window]["maps_total"],
            ),
        )
    if summary["preferred_time_filter"]:
        summary["distribution"] = summary["windows"][summary["preferred_time_filter"]]
    else:
        summary["distribution"] = aggregate_player_stats(players)
    summary["coverage"] = summary["distribution"]["covered_players"] / max(len(players), 1)
    for metric, values in summary["distribution"].get("metrics", {}).items():
        summary["avg"][metric] = values.get("avg")
        legacy = PLAYER_LEGACY_NAMES.get(metric)
        if legacy:
            summary["avg"][legacy] = values.get("avg")
    return summary


def load_model_engine(history_path: Path) -> dict[str, Any] | None:
    """Carga el artefacto entrenado y reconstruye el estado cronológico Glicko-2.

    El estado se reconstruye desde el MISMO histórico usado en entrenamiento, con
    la misma normalización (cs2model.dataio), de modo que las features en vivo son
    idénticas a las de entrenamiento (cero fuga, consistencia train/serve).
    """
    if not _CS2_OK:
        return None
    try:
        artifact = cs2_load_artifact()
        if artifact is None:
            return None
        rows = cs2_dataio.load_training_rows(
            history_path,
            MASTER_MATCHES if MASTER_MATCHES.exists() else None,
            cs2_only=True,
        )
        if not rows:
            return None
        _, _, _, state = cs2_build_state(rows)
        return {"artifact": artifact, "state": state}
    except Exception:
        return None


def model_external_features_for_order(features: dict[str, Any], reverse: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    diff_columns = CS2_PLAYER_DIFF_COLUMNS + CS2_RANKING_DIFF_COLUMNS + CS2_ROSTER_DIFF_COLUMNS
    sym_columns = CS2_PLAYER_SYM_COLUMNS + CS2_RANKING_SYM_COLUMNS + CS2_ROSTER_SYM_COLUMNS
    for col in diff_columns:
        try:
            value = float(features.get(col) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        out[col] = -value if reverse else value
    for col in sym_columns:
        try:
            out[col] = float(features.get(col) or 0.0)
        except (TypeError, ValueError):
            out[col] = 0.0
    return out


def model_probability_team1(
    engine: dict[str, Any],
    team1_key: str,
    team2_key: str,
    match_dt: datetime | None,
    event: str,
    fmt: str,
    analytics_match: dict[str, Any] | None = None,
    match_context: dict[str, Any] | None = None,
    extra_features: dict[str, float] | None = None,
    announced_lineups: dict[str, Any] | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> float:
    """Probabilidad calibrada de que gane team1 según el modelo entrenado.

    Se promedia la predicción en ambos órdenes (A vs B y B vs A) para garantizar
    invariancia al orden de los equipos (predicción simétrica).
    """
    artifact = engine["artifact"]
    f1, f2 = _match_feature_pair(
        engine, team1_key, team2_key, match_dt, event, fmt,
        analytics_match, match_context, extra_features, announced_lineups, event_metadata,
    )
    p1 = float(artifact.predict_proba_team1([f1])[0])
    p2 = float(artifact.predict_proba_team1([f2])[0])
    return clamp(0.5 * (p1 + (1.0 - p2)), 1e-4, 1 - 1e-4)


def _match_feature_pair(
    engine: dict[str, Any],
    team1_key: str,
    team2_key: str,
    match_dt: datetime | None,
    event: str,
    fmt: str,
    analytics_match: dict[str, Any] | None = None,
    match_context: dict[str, Any] | None = None,
    extra_features: dict[str, float] | None = None,
    announced_lineups: dict[str, Any] | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Par de features simetrico (A vs B, B vs A). Compartido por el scoring y la
    estimacion de incertidumbre (A2) para no duplicar logica."""
    state = engine["state"]
    fmt = fmt if fmt in {"bo1", "bo3", "bo5"} else "bo3"
    f1 = state.emit_features(team1_key, team2_key, match_dt, event or "", fmt)
    f2 = state.emit_features(team2_key, team1_key, match_dt, event or "", fmt)
    if hasattr(state, "regime_features"):
        regime_features = state.regime_features(
            {"date_obj": match_dt, "match_context": match_context or {}}
        )
        f1.update(regime_features)
        f2.update(regime_features)
    if match_context:
        context_features = cs2_context_match_features({"match_context": match_context})
        f1.update(context_features)
        f2.update(context_features)
    if analytics_match:
        f1.update(cs2_analytics_match_features({**analytics_match, "format": fmt}))
        f2.update(
            cs2_analytics_match_features(
                {
                    **analytics_match,
                    "format": fmt,
                    "team1": analytics_match.get("team2"),
                    "team2": analytics_match.get("team1"),
                    "team1_key": team2_key,
                    "team2_key": team1_key,
                }
            )
        )
    if isinstance(announced_lineups, dict) and announced_lineups:
        f1.update(cs2_announced_lineup_features({"prematch_lineups": announced_lineups}))
        f2.update(
            cs2_announced_lineup_features(
                {
                    "prematch_lineups": {
                        "team1": announced_lineups.get("team2"),
                        "team2": announced_lineups.get("team1"),
                    }
                }
            )
        )
    if isinstance(event_metadata, dict) and event_metadata:
        event_features = cs2_event_metadata_features({"event_metadata": event_metadata})
        f1.update(event_features)
        f2.update(event_features)
    if extra_features:
        f1.update(model_external_features_for_order(extra_features, reverse=False))
        f2.update(model_external_features_for_order(extra_features, reverse=True))
    return f1, f2


def model_probability_and_uncertainty(
    engine: dict[str, Any],
    team1_key: str,
    team2_key: str,
    match_dt: datetime | None,
    event: str,
    fmt: str,
    analytics_match: dict[str, Any] | None = None,
    match_context: dict[str, Any] | None = None,
    extra_features: dict[str, float] | None = None,
    announced_lineups: dict[str, Any] | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """A2: (prob_team1 calibrada simetrica, std_epistemica del ensemble).

    std = cuanto discrepan los miembros del ensemble => incertidumbre del modelo,
    usada para reducir el stake cuando no lo tiene claro.
    """
    artifact = engine["artifact"]
    f1, f2 = _match_feature_pair(
        engine, team1_key, team2_key, match_dt, event, fmt,
        analytics_match, match_context, extra_features, announced_lineups, event_metadata,
    )
    m1, s1 = artifact.predict_proba_team1_with_uncertainty([f1])
    m2, s2 = artifact.predict_proba_team1_with_uncertainty([f2])
    prob = clamp(0.5 * (float(m1[0]) + (1.0 - float(m2[0]))), 1e-4, 1 - 1e-4)
    std = 0.5 * (float(s1[0]) + float(s2[0]))
    return prob, std


def model_series_score_distribution(
    engine: dict[str, Any],
    team1_key: str,
    team2_key: str,
    match_dt: datetime | None,
    event: str,
    fmt: str,
    analytics_match: dict[str, Any] | None = None,
    match_context: dict[str, Any] | None = None,
    extra_features: dict[str, float] | None = None,
    announced_lineups: dict[str, Any] | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """A6: BO3 scoreline distribution from the auto-gated rich target."""
    if str(fmt or "").lower() != "bo3":
        return None
    artifact = engine["artifact"]
    f1, _f2 = _match_feature_pair(
        engine, team1_key, team2_key, match_dt, event, fmt,
        analytics_match, match_context, extra_features, announced_lineups, event_metadata,
    )
    distributions = artifact.predict_series_score_distribution([f1])
    return distributions[0] if distributions else None


def logistic_probability(features: dict[str, float]) -> float:
    # Fallback ligero (si no hay artefacto entrenado): scorer logístico con
    # coeficientes derivados de la importancia del backtest 10k.
    z = 0.0
    z += 1.35 * features["elo_prob_centered"]
    z += 0.65 * features["winrate_diff"]
    z += 0.45 * features["last10_winrate_diff"]
    z += 0.25 * features["h2h_winrate_centered"]
    z += 0.12 * features["avg_score_diff"]
    z += 0.0015 * features["recent_opponent_elo_diff"]
    z += 0.02 * features["matches_log_diff"]
    z += 0.10 * features.get("player_rating_diff", 0.0)
    z += 0.20 * features.get("player_real_rating_diff", 0.0)
    z += 0.28 * features.get("map_pool_advantage_team1", 0.0)
    z -= 0.06 * features.get("fatigue_last24_diff_team1", 0.0)
    z += 0.015 * features.get("roster_days_log_diff", 0.0)
    return 1 / (1 + math.exp(-z))


def event_volatility(event_outcomes: dict[str, list[int]], event: str) -> dict[str, Any]:
    outcomes = event_outcomes.get(event, [])
    n = len(outcomes)
    if n < 20:
        return {"n": n, "volatility": None, "label": "unknown_event_history"}
    p = sum(outcomes) / n
    volatility = 1 - abs(p - 0.5) * 2
    label = "high" if volatility >= 0.85 else "medium" if volatility >= 0.7 else "low"
    return {"n": n, "volatility": volatility, "label": label}


def similar_probability_bucket(history_predictions: list[dict[str, Any]], p: float, width: float = 0.05) -> dict[str, Any]:
    bucket = [row for row in history_predictions if abs(row["p"] - p) <= width]
    if len(bucket) < 30:
        return {"n": len(bucket), "favorite_loss_rate": None}
    fav_losses = [int((row["p"] >= 0.5 and not row["team1_win"]) or (row["p"] < 0.5 and row["team1_win"])) for row in bucket]
    return {"n": len(bucket), "favorite_loss_rate": sum(fav_losses) / len(fav_losses)}


def historical_probability_rows(rows: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    # Uses final state as an approximation for bucketing. It is only a control metric, not training.
    out = []
    elos = state["elos"]
    for row in rows[-3000:]:
        p = elo_prob(elos[row["team1_key"]], elos[row["team2_key"]])
        out.append({"p": p, "team1_win": row["score1"] > row["score2"]})
    return out


def load_master() -> dict[str, Any]:
    return read_json(MASTER_MATCHES, {})


def team_names_from_detail(detail: dict[str, Any] | None) -> tuple[str | None, str | None]:
    match = (detail or {}).get("match") or {}
    return (
        (match.get("team1") or {}).get("name"),
        (match.get("team2") or {}).get("name"),
    )


def odds_average_from(odds: dict[str, Any] | None) -> dict[str, Any] | None:
    if not odds or not odds.get("available"):
        return None
    average = odds.get("average") or {}
    if average.get("team1_implied_prob_norm") is None:
        return None
    return average


def odds_history_points(match_id: str, master: dict[str, Any], current_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    record = master.get(match_id, {})
    points = list(record.get("odds_history") or [])
    seen = {(point.get("captured_at"), point.get("run_id")) for point in points}
    current_avg = odds_average_from(current_snapshot.get("odds"))
    if current_avg:
        point = {
            "captured_at": current_snapshot.get("captured_at"),
            "run_id": current_snapshot.get("run_id") or current_snapshot.get("data_quality", {}).get("run_id"),
            "bookmaker_count": (current_snapshot.get("odds") or {}).get("bookmaker_count", 0),
            "team1_decimal": current_avg.get("team1_decimal"),
            "team2_decimal": current_avg.get("team2_decimal"),
            "team1_implied_prob_norm": current_avg.get("team1_implied_prob_norm"),
            "team2_implied_prob_norm": current_avg.get("team2_implied_prob_norm"),
            "overround": current_avg.get("overround"),
        }
        key = (point.get("captured_at"), point.get("run_id"))
        if key not in seen:
            points.append(point)
    points.sort(key=lambda point: point.get("captured_at") or "")
    return points


def market_metrics(odds: dict[str, Any], history_points: list[dict[str, Any]], model_prob_team1: float) -> dict[str, Any]:
    average = odds_average_from(odds)
    providers = odds.get("providers") or []
    provider_probs = [
        provider.get("team1_implied_prob_norm")
        for provider in providers
        if provider.get("team1_implied_prob_norm") is not None
    ]
    consensus_std = statistics.pstdev(provider_probs) if len(provider_probs) > 1 else None
    consensus_range = (max(provider_probs) - min(provider_probs)) if len(provider_probs) > 1 else None
    if consensus_std is None:
        consensus_label = "single_or_missing"
    elif consensus_std <= 0.025:
        consensus_label = "high"
    elif consensus_std <= 0.06:
        consensus_label = "medium"
    else:
        consensus_label = "low"

    opening = history_points[0] if history_points else None
    latest = history_points[-1] if history_points else None
    if not latest and average:
        latest = {
            "team1_implied_prob_norm": average.get("team1_implied_prob_norm"),
            "team2_implied_prob_norm": average.get("team2_implied_prob_norm"),
            "team1_decimal": average.get("team1_decimal"),
            "team2_decimal": average.get("team2_decimal"),
            "overround": average.get("overround"),
            "bookmaker_count": odds.get("bookmaker_count", 0),
        }
    if not opening:
        opening = latest

    latest_p = latest.get("team1_implied_prob_norm") if latest else None
    opening_p = opening.get("team1_implied_prob_norm") if opening else None
    drift = latest_p - opening_p if latest_p is not None and opening_p is not None else None
    if drift is None:
        drift_label = "unknown"
    elif abs(drift) < 0.025:
        drift_label = "stable"
    elif drift >= 0.06:
        drift_label = "strong_to_team1"
    elif drift <= -0.06:
        drift_label = "strong_to_team2"
    elif drift > 0:
        drift_label = "moderate_to_team1"
    else:
        drift_label = "moderate_to_team2"

    edge_team1 = model_prob_team1 - latest_p if latest_p is not None else None
    if edge_team1 is None:
        value_label = "no_market"
    elif abs(edge_team1) < 0.04:
        value_label = "aligned"
    elif edge_team1 > 0:
        value_label = "model_team1"
    else:
        value_label = "model_team2"

    return {
        "available": average is not None,
        "bookmaker_count": odds.get("bookmaker_count", 0),
        "opening": opening,
        "latest": latest,
        "drift_team1_prob": drift,
        "drift_abs": abs(drift) if drift is not None else None,
        "drift_label": drift_label,
        "consensus_std": consensus_std,
        "consensus_range": consensus_range,
        "consensus_label": consensus_label,
        "market_favorite_side": "team1" if latest_p is not None and latest_p >= 0.5 else "team2" if latest_p is not None else None,
        "model_edge_team1": edge_team1,
        "model_edge_abs": abs(edge_team1) if edge_team1 is not None else None,
        "value_label": value_label,
        "overround": (average or {}).get("overround"),
        "history_points": len(history_points),
    }


def reliability_score(entry: dict[str, Any], features: dict[str, Any]) -> float:
    """Mismo score conceptual que la web: cuanto respaldo tienen los datos."""
    score = 1.0
    min_matches = min(features.get("matches_team1") or 0, features.get("matches_team2") or 0)
    if min_matches < 5:
        score *= 0.35
    elif min_matches < 12:
        score *= 0.60
    elif min_matches < 25:
        score *= 0.85

    rd1 = features.get("glicko_rd_team1")
    rd2 = features.get("glicko_rd_team2")
    rd = ((rd1 if rd1 is not None else 160) + (rd2 if rd2 is not None else 160)) / 2
    if rd > 140:
        score *= 0.55
    elif rd > 110:
        score *= 0.78
    elif rd > 90:
        score *= 0.92

    if entry.get("format") == "bo1":
        score *= 0.82
    for side in ("team1", "team2"):
        name = ((entry.get(side) or {}).get("name") or "").strip()
        if not name or re.search(r"tbd|winner|/", name, re.I):
            score *= 0.25
            break
    if not (entry.get("data_quality") or {}).get("real_pre_match_snapshot", True):
        score *= 0.85
    coverage = min(features.get("player_coverage_team1") or 1, features.get("player_coverage_team2") or 1)
    if coverage < 0.6:
        score *= 0.92
    return clamp(score, 0.05, 1.0)


def reliability_adjusted_probability(model_prob_team1: float, reliability: float) -> float:
    """Shrink raw model probability toward 50/50 when pre-match data is thin."""
    return clamp(0.5 + clamp(reliability, 0.05, 1.0) * (model_prob_team1 - 0.5), 1e-4, 1 - 1e-4)


def probability_log_loss(y: int, p: float) -> float:
    eps = 1e-9
    p = clamp(p, eps, 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def dynamic_market_prior_weight(reliability: float, market: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    """Conservative odds prior while the project lacks enough odds history.

    More market weight is allowed when internal data reliability is low; high
    model reliability keeps the market as a small benchmark/prior instead of an
    oracle. Market-quality penalties avoid overreacting to thin/discordant odds.
    """
    rel = clamp(reliability, 0.05, 1.0)
    weight = MARKET_PRIOR_MIN_WEIGHT + (MARKET_PRIOR_MAX_WEIGHT - MARKET_PRIOR_MIN_WEIGHT) * (1.0 - rel)
    reasons = ["dynamic_reliability_prior"]
    if market:
        bookmaker_count = parse_int(market.get("bookmaker_count")) or 0
        consensus = market.get("consensus_label")
        if bookmaker_count and bookmaker_count < 3:
            weight *= 0.75
            reasons.append("thin_bookmaker_count")
        if consensus in {"low", "single_or_missing"}:
            weight *= 0.70
            reasons.append("weak_market_consensus")
        if market.get("drift_abs") is not None and market["drift_abs"] >= 0.08:
            weight *= 0.85
            reasons.append("large_market_drift")
    return clamp(weight, 0.0, MARKET_PRIOR_MAX_WEIGHT), reasons


def _prediction_actual_from_master(master: dict[str, Any], match_id: str) -> int | None:
    record = master.get(str(match_id), {})
    score = record.get("score") or {}
    try:
        s1 = int(score.get("team1"))
        s2 = int(score.get("team2"))
    except (TypeError, ValueError):
        return None
    if s1 == s2:
        return None
    return 1 if s1 > s2 else 0


def _prediction_market_prob(entry: dict[str, Any]) -> float | None:
    pred = entry.get("prediction") or {}
    if pred.get("odds_prob_team1") is not None:
        return safe_float(pred.get("odds_prob_team1"))
    market = ((entry.get("controls") or {}).get("market") or {})
    for source in (market.get("latest") or {}, (entry.get("odds") or {}).get("average") or {}, market.get("opening") or {}):
        p = safe_float(source.get("team1_implied_prob_norm"))
        if p is not None and 0.0 < p < 1.0:
            return p
    return None


def _market_blend_metrics(samples: list[dict[str, Any]], weight: float) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    probs = [clamp((1 - weight) * row["model_p"] + weight * row["market_p"], 1e-9, 1 - 1e-9) for row in samples]
    y = [row["y"] for row in samples]
    return {
        "n": len(samples),
        "accuracy": sum(int((p >= 0.5) == bool(yy)) for p, yy in zip(probs, y)) / len(samples),
        "log_loss": sum(probability_log_loss(yy, p) for p, yy in zip(probs, y)) / len(samples),
        "brier": sum((yy - p) ** 2 for p, yy in zip(probs, y)) / len(samples),
    }


def build_market_blend_policy(master: dict[str, Any]) -> dict[str, Any]:
    """Learn model/market blend weight only after enough point-in-time odds data.

    Until the sample is large enough, production uses the conservative dynamic
    prior. When enough closed odds predictions exist, this performs expanding
    walk-forward selection of the blend weight and reports the out-of-sample
    result before exposing a production weight.
    """
    latest_prediction: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS_DIR.glob("*/predictions_enriched.json")):
        for entry in read_json(path, []):
            if not (entry.get("data_quality") or {}).get("real_pre_match_snapshot", True):
                continue
            pred = entry.get("prediction") or {}
            model_p = safe_float(pred.get("model_prob_team1"))
            market_p = _prediction_market_prob(entry)
            if model_p is None or market_p is None:
                continue
            captured_at = entry.get("captured_at") or ""
            current = latest_prediction.get(str(entry.get("id")))
            if current is None or captured_at > (current.get("captured_at") or ""):
                latest_prediction[str(entry.get("id"))] = entry

    samples: list[dict[str, Any]] = []
    for match_id, entry in latest_prediction.items():
        y = _prediction_actual_from_master(master, match_id)
        if y is None:
            continue
        pred = entry.get("prediction") or {}
        model_p = safe_float(pred.get("model_prob_team1"))
        market_p = _prediction_market_prob(entry)
        if model_p is None or market_p is None:
            continue
        samples.append({
            "match_id": match_id,
            "completed_at": (master.get(str(match_id), {}) or {}).get("completed_at") or entry.get("captured_at") or "",
            "model_p": model_p,
            "market_p": market_p,
            "y": y,
        })
    samples.sort(key=lambda row: row["completed_at"])
    n = len(samples)
    base = {
        "kind": "market_blend_policy",
        "n_closed_odds_matches": n,
        "min_samples_for_learning": MARKET_BLEND_LEARN_MIN_SAMPLES,
        "walk_forward_min_train": MARKET_BLEND_WALK_FORWARD_MIN_TRAIN,
        "candidate_weights": MARKET_BLEND_CANDIDATE_WEIGHTS,
    }
    if n < MARKET_BLEND_LEARN_MIN_SAMPLES:
        return {
            **base,
            "available": False,
            "production_market_weight": None,
            "source": "dynamic_reliability_prior_until_enough_odds",
            "note": "Insufficient closed point-in-time odds matches; use conservative dynamic prior.",
            "model_metrics": _market_blend_metrics(samples, 0.0),
            "market_metrics": _market_blend_metrics(samples, 1.0),
        }

    wf_rows = []
    for index in range(MARKET_BLEND_WALK_FORWARD_MIN_TRAIN, n):
        train = samples[:index]
        test = samples[index]
        train_scores = [
            (_market_blend_metrics(train, weight).get("log_loss", float("inf")), weight)
            for weight in MARKET_BLEND_CANDIDATE_WEIGHTS
        ]
        chosen_weight = min(train_scores, key=lambda item: item[0])[1]
        p = clamp((1 - chosen_weight) * test["model_p"] + chosen_weight * test["market_p"], 1e-9, 1 - 1e-9)
        wf_rows.append({"y": test["y"], "p": p, "weight": chosen_weight})

    wf_metrics = {
        "n": len(wf_rows),
        "accuracy": sum(int((row["p"] >= 0.5) == bool(row["y"])) for row in wf_rows) / len(wf_rows),
        "log_loss": sum(probability_log_loss(row["y"], row["p"]) for row in wf_rows) / len(wf_rows),
        "brier": sum((row["y"] - row["p"]) ** 2 for row in wf_rows) / len(wf_rows),
        "avg_selected_weight": sum(row["weight"] for row in wf_rows) / len(wf_rows),
    }
    final_scores = [
        (_market_blend_metrics(samples, weight).get("log_loss", float("inf")), weight)
        for weight in MARKET_BLEND_CANDIDATE_WEIGHTS
    ]
    production_weight = min(final_scores, key=lambda item: item[0])[1]
    return {
        **base,
        "available": True,
        "source": "learned_expanding_walk_forward",
        "production_market_weight": production_weight,
        "walk_forward_metrics": wf_metrics,
        "model_metrics": _market_blend_metrics(samples, 0.0),
        "market_metrics": _market_blend_metrics(samples, 1.0),
        "production_metrics_in_sample": _market_blend_metrics(samples, production_weight),
    }


def decision_probability_team1(
    model_prob_team1: float,
    odds_prob_team1: float | None,
    reliability: float,
    market: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Operational probability used for ranking/staking, kept separate from Model A.

    If market odds exist, blend the calibrated stats model with normalized market
    probability. If no market exists, shrink low-reliability model confidence toward
    50/50. This avoids treating low-data 80% predictions like fully-backed 80%.
    """
    risk_adjusted = reliability_adjusted_probability(model_prob_team1, reliability)
    if odds_prob_team1 is None:
        return {
            "prob_team1": risk_adjusted,
            "risk_adjusted_prob_team1": risk_adjusted,
            "source": "model_reliability_shrink",
            "market_weight": 0.0,
            "market_weight_reasons": ["no_market_odds"],
        }
    policy = policy or {}
    if policy.get("available") and policy.get("production_market_weight") is not None:
        weight = clamp(float(policy["production_market_weight"]), 0.0, 0.60)
        source = "model_market_learned_walk_forward"
        reasons = [policy.get("source") or "learned_policy"]
    else:
        weight, reasons = dynamic_market_prior_weight(reliability, market)
        source = "model_market_dynamic_prior"
    prob = clamp(
        (1.0 - weight) * model_prob_team1 + weight * odds_prob_team1,
        1e-4,
        1 - 1e-4,
    )
    return {
        "prob_team1": prob,
        "risk_adjusted_prob_team1": risk_adjusted,
        "source": source,
        "market_weight": weight,
        "market_weight_reasons": reasons,
    }


def model_calibration_factor(model_metadata: dict[str, Any] | None) -> tuple[float, float | None, int | None]:
    """Factor de prudencia segun ECE walk-forward del modelo de produccion."""
    if not model_metadata:
        return 0.60, None, None
    prod = model_metadata.get("production_model")
    metrics = (model_metadata.get("walk_forward_metrics") or {}).get(prod or "") or {}
    ece = safe_float(metrics.get("ece_10"))
    n = parse_int(metrics.get("n"))
    if ece is None:
        return 0.60, None, n
    return clamp(1.0 - 5.0 * ece, 0.25, 1.0), ece, n


def market_consensus_factor(market: dict[str, Any]) -> float:
    label = market.get("consensus_label")
    if label == "high":
        return 1.0
    if label == "medium":
        return 0.80
    if label == "low":
        return 0.55
    return 0.65


def staking_recommendation(
    entry: dict[str, Any],
    features: dict[str, Any],
    prediction: dict[str, Any],
    market: dict[str, Any],
    model_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recomendacion de stake como Kelly fraccional ajustado por calidad de datos.

    Full Kelly para cuota decimal d y probabilidad p: f = (p*d - 1)/(d - 1).
    Usamos 25% Kelly antes de penalizaciones por incertidumbre porque nuestras p
    son estimadas, no probabilidades verdaderas conocidas.
    """
    latest = market.get("latest") or {}
    avg = (entry.get("odds") or {}).get("average") or {}
    if not market.get("available") or not latest and not avg:
        return {
            "action": "no_bet",
            "reason_code": "missing_odds",
            "recommended_fraction": 0.0,
            "recommended_pct_bankroll": 0.0,
            "reason": "Sin odds suficientes para calcular EV/Kelly.",
        }

    model_p_team1 = prediction.get("model_prob_team1")
    decision_p_team1 = prediction.get("decision_prob_team1")
    if decision_p_team1 is None:
        decision_p_team1 = prediction.get("risk_adjusted_prob_team1")
    if decision_p_team1 is None:
        decision_p_team1 = model_p_team1
    if model_p_team1 is None or decision_p_team1 is None:
        return {
            "action": "no_bet",
            "reason_code": "missing_model_probability",
            "recommended_fraction": 0.0,
            "recommended_pct_bankroll": 0.0,
            "reason": "Sin probabilidad calibrada del modelo.",
        }

    reliability = reliability_score(entry, features)
    cal_factor, ece, eval_n = model_calibration_factor(model_metadata)
    consensus_factor = market_consensus_factor(market)
    # A2: penaliza el stake por incertidumbre epistemica (discrepancia del ensemble).
    epistemic_std = safe_float(prediction.get("model_epistemic_std")) or 0.0
    uncertainty_factor = 1.0 / (1.0 + 8.0 * epistemic_std) if epistemic_std > 0 else 1.0
    fractional_kelly = 0.25
    max_single_match = 0.025

    candidates = []
    for side in ("team1", "team2"):
        p_model = decision_p_team1 if side == "team1" else 1.0 - decision_p_team1
        p_raw_model = model_p_team1 if side == "team1" else 1.0 - model_p_team1
        decimal = safe_float(latest.get(f"{side}_decimal")) or safe_float(avg.get(f"{side}_decimal"))
        market_prob = (
            safe_float(latest.get(f"{side}_implied_prob_norm"))
            or safe_float(avg.get(f"{side}_implied_prob_norm"))
        )
        if decimal is None or decimal <= 1.0:
            continue
        ev = p_model * decimal - 1.0
        raw_kelly = max(0.0, ev / (decimal - 1.0))
        adjusted = raw_kelly * fractional_kelly * reliability * consensus_factor * cal_factor * uncertainty_factor
        capped = min(adjusted, max_single_match)
        candidates.append(
            {
                "side": side,
                "team": (entry.get(side) or {}).get("name"),
                "model_prob": p_model,
                "raw_model_prob": p_raw_model,
                "probability_source": prediction.get("decision_probability_source") or "decision_prob_team1",
                "market_prob": market_prob,
                "decimal_odds": decimal,
                "expected_roi": ev,
                "raw_kelly_fraction": raw_kelly,
                "adjusted_fraction": adjusted,
                "capped_fraction": capped,
            }
        )

    candidates.sort(key=lambda row: row["capped_fraction"], reverse=True)
    best = candidates[0] if candidates else None
    if not best or best["raw_kelly_fraction"] <= 0 or best["capped_fraction"] <= 0:
        return {
            "action": "no_bet",
            "reason_code": "no_positive_ev",
            "recommended_fraction": 0.0,
            "recommended_pct_bankroll": 0.0,
            "reason": "No hay EV positivo tras comparar modelo y cuota disponible.",
            "candidates": candidates,
            "method": "quarter_kelly_reliability_adjusted",
        }

    risk_notes = []
    if best["adjusted_fraction"] > max_single_match:
        risk_notes.append("cap_single_match_applied")
    if reliability < 0.55:
        risk_notes.append("low_data_reliability")
    if consensus_factor < 0.8:
        risk_notes.append("weak_or_single_bookmaker_consensus")
    if ece is not None and ece > 0.03:
        risk_notes.append("model_calibration_error_penalty")
    if entry.get("format") == "bo1":
        risk_notes.append("bo1_high_variance")

    return {
        "action": "bet",
        "side": best["side"],
        "team": best["team"],
        "recommended_fraction": best["capped_fraction"],
        "recommended_pct_bankroll": round(best["capped_fraction"] * 100, 3),
        "expected_roi": best["expected_roi"],
        "raw_kelly_fraction": best["raw_kelly_fraction"],
        "fractional_kelly": fractional_kelly,
        "reliability_factor": reliability,
        "consensus_factor": consensus_factor,
        "calibration_factor": cal_factor,
        "uncertainty_factor": round(uncertainty_factor, 4),
        "model_epistemic_std": round(epistemic_std, 5) if epistemic_std else None,
        "walk_forward_ece": ece,
        "walk_forward_n": eval_n,
        "max_single_match_fraction": max_single_match,
        "probability_source": prediction.get("decision_probability_source") or "decision_prob_team1",
        "risk_notes": risk_notes,
        "method": "quarter_kelly_reliability_adjusted",
        "candidates": candidates,
    }


def build_map_state(master: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    teams: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"played": 0, "wins": 0}))
    for record in master.values():
        detail = record.get("detail") or (record.get("result") or {}).get("detail")
        if not isinstance(detail, dict):
            continue
        team1, team2 = team_names_from_detail(detail)
        if not team1 or not team2:
            continue
        for item in detail.get("maps") or []:
            map_name = normalize_map_name(item.get("map_name"))
            score = item.get("score") or {}
            if not map_name:
                continue
            s1 = safe_float(score.get(team1))
            s2 = safe_float(score.get(team2))
            if s1 is None or s2 is None or s1 == s2:
                continue
            for team, won in [(team1, s1 > s2), (team2, s2 > s1)]:
                bucket = teams[clean_team(team)][map_name]
                bucket["played"] += 1
                bucket["wins"] += int(won)

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for team, maps in teams.items():
        output[team] = {}
        for map_name, stats in maps.items():
            played = stats["played"]
            output[team][map_name] = {
                "played": played,
                "wins": stats["wins"],
                "winrate": stats["wins"] / played if played else None,
            }
    return output


def map_pool_estimate(team1_name: str, team2_name: str, map_state: dict[str, Any]) -> dict[str, Any]:
    maps1 = map_state.get(clean_team(team1_name), {})
    maps2 = map_state.get(clean_team(team2_name), {})
    rows = []
    for map_name in MAP_POOL:
        s1 = maps1.get(map_name, {})
        s2 = maps2.get(map_name, {})
        wr1 = s1.get("winrate")
        wr2 = s2.get("winrate")
        played1 = int(s1.get("played") or 0)
        played2 = int(s2.get("played") or 0)
        diff = (wr1 - wr2) if wr1 is not None and wr2 is not None else None
        rows.append(
            {
                "map": map_name,
                "team1_played": played1,
                "team2_played": played2,
                "team1_winrate": wr1,
                "team2_winrate": wr2,
                "diff_team1": diff,
                "usable": wr1 is not None and wr2 is not None and min(played1, played2) >= 2,
            }
        )
    usable = [row for row in rows if row["usable"]]
    coverage = len(usable) / max(len(MAP_POOL), 1)
    if usable:
        weighted_diffs = [
            row["diff_team1"] * min(row["team1_played"], row["team2_played"])
            for row in usable
            if row["diff_team1"] is not None
        ]
        weights = [min(row["team1_played"], row["team2_played"]) for row in usable]
        advantage = sum(weighted_diffs) / sum(weights) if sum(weights) else None
        team1_best = max(usable, key=lambda row: row["diff_team1"] or -99)
        team2_best = min(usable, key=lambda row: row["diff_team1"] or 99)
        decider = min(usable, key=lambda row: abs(row["diff_team1"] or 0))
        permaban_team1 = min(rows, key=lambda row: (row["team1_winrate"] if row["team1_winrate"] is not None else 1.0))
        permaban_team2 = min(rows, key=lambda row: (row["team2_winrate"] if row["team2_winrate"] is not None else 1.0))
    else:
        advantage = None
        team1_best = team2_best = decider = permaban_team1 = permaban_team2 = None

    if coverage >= 0.65:
        confidence = "medium"
    elif coverage >= 0.45:
        confidence = "low"
    else:
        confidence = "very_low"

    return {
        "kind": "pre_match_map_pool_estimate",
        "is_estimated": True,
        "actual_veto_known": False,
        "confidence": confidence,
        "note": "Pre-partido no se conoce el veto real; esto estima ventaja/riesgo por historial de mapas.",
        "coverage": coverage,
        "status": "ok" if coverage >= 0.45 else "low_data",
        "map_pool_advantage_team1": advantage,
        "team1_best_map": team1_best,
        "team2_best_map": team2_best,
        "team1_weak_map_estimate": permaban_team1,
        "team2_weak_map_estimate": permaban_team2,
        "decider_estimate": decider,
        "maps": rows,
    }


def roster_signature_from_profile(profile: dict[str, Any]) -> str:
    ids = sorted(str(player.get("id") or player.get("name") or "").strip() for player in profile.get("squad") or [])
    return "|".join([player_id for player_id in ids if player_id])


def build_roster_history(run_dir: Path, profiles: dict[str, Any]) -> dict[str, Any]:
    history = read_json(MASTER_ROSTERS, {})
    run_started = read_json(run_dir / "manifest.json", {}).get("started_at")
    for team_id, profile in profiles.items():
        signature = roster_signature_from_profile(profile)
        if not signature:
            continue
        entry = history.setdefault(
            team_id,
            {"team_id": team_id, "team_name": profile.get("name"), "snapshots": []},
        )
        entry["team_name"] = profile.get("name") or entry.get("team_name")
        point = {
            "captured_at": run_started,
            "run_id": run_dir.name,
            "signature": signature,
            "player_ids": [str(player.get("id")) for player in profile.get("squad") or [] if player.get("id")],
            "player_names": [player.get("name") for player in profile.get("squad") or [] if player.get("name")],
            "roster_size": len(profile.get("squad") or []),
        }
        snapshots = entry.setdefault("snapshots", [])
        if not any(row.get("run_id") == point["run_id"] for row in snapshots):
            snapshots.append(point)
        snapshots.sort(key=lambda row: row.get("captured_at") or "")
    write_json(MASTER_ROSTERS, history)
    return history


def roster_stability(team_id: str | None, profile: dict[str, Any] | None, roster_history: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    profile = profile or {}
    team_id = str(team_id or "")
    squad = profile.get("squad") or []
    signature = roster_signature_from_profile(profile)
    entry = roster_history.get(team_id, {}) if team_id else {}
    snapshots = sorted(entry.get("snapshots") or [], key=lambda row: row.get("captured_at") or "")
    current_time = parse_iso(read_json(run_dir / "manifest.json", {}).get("started_at")) or datetime.utcnow()
    last_different = None
    first_current = None
    changes_30d = 0
    previous_sig = None
    for snap in snapshots:
        snap_time = parse_iso(snap.get("captured_at")) or current_time
        if previous_sig and snap.get("signature") != previous_sig and current_time - snap_time <= timedelta(days=30):
            changes_30d += 1
        previous_sig = snap.get("signature")
        if snap.get("signature") == signature:
            if last_different is None or snap_time > last_different:
                first_current = first_current or snap_time
        else:
            last_different = snap_time
            first_current = None
    if first_current is None:
        first_current = current_time
    days_current = max(0.0, (current_time - first_current).total_seconds() / 86400)
    roster_size = len(squad)
    return {
        "team_id": team_id or None,
        "roster_size": roster_size,
        "signature": signature,
        "snapshots": len(snapshots),
        "first_seen_current_roster_at": first_current.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days_with_current_roster": days_current,
        "changes_30d": changes_30d,
        "recent_change": days_current < 7 and len(snapshots) > 1,
        "standin_risk": roster_size != 5 or (days_current < 3 and len(snapshots) > 1),
        "coverage": "tracked" if snapshots else "current_only",
    }


def load_actual_lineup_store(
    conn_or_path: sqlite3.Connection | str | Path = LIVE_DB,
) -> dict[str, Any]:
    """Load complete actual lineups once for point-in-time roster comparisons."""
    owns_connection = not isinstance(conn_or_path, sqlite3.Connection)
    conn = sqlite3.connect(conn_or_path) if owns_connection else conn_or_path
    try:
        match_datetimes = {
            str(hltv_match_id): parsed
            for hltv_match_id, datetime_utc in conn.execute(
                """
                SELECT hltv_match_id, datetime_utc
                FROM matches
                WHERE hltv_match_id IS NOT NULL
                  AND datetime_utc IS NOT NULL
                """
            )
            if hltv_match_id and (parsed := parse_iso(datetime_utc)) is not None
        }
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in conn.execute(
            """
            SELECT CAST(t.hltv_id AS TEXT) AS hltv_team_id,
                   COALESCE(m.hltv_match_id, CAST(m.match_id AS TEXT)) AS hltv_match_id,
                   m.datetime_utc,
                   CAST(p.hltv_id AS TEXT) AS hltv_player_id,
                   p.nick,
                   ml.is_standin
            FROM match_lineups ml
            JOIN matches m ON m.match_id = ml.match_id
            JOIN teams t ON t.team_id = ml.team_id
            JOIN players p ON p.player_id = ml.player_id
            WHERE m.status = 'completed'
              AND m.datetime_utc IS NOT NULL
              AND t.hltv_id IS NOT NULL
              AND p.hltv_id IS NOT NULL
            ORDER BY m.datetime_utc, m.match_id, t.team_id, p.player_id
            """
        ):
            team_id, match_id, datetime_utc, player_id, nick, is_standin = row
            key = (str(team_id), str(match_id), str(datetime_utc))
            grouped.setdefault(
                key,
                {
                    "team_id": str(team_id),
                    "match_id": str(match_id),
                    "datetime_utc": str(datetime_utc),
                    "datetime": parse_iso(str(datetime_utc)),
                    "players": {},
                },
            )["players"][str(player_id)] = {
                "id": str(player_id),
                "name": str(nick or player_id),
                "is_standin": bool(is_standin),
            }

        teams: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in grouped.values():
            players = record.pop("players")
            if record["datetime"] is None or len(players) != COMPLETE_LINEUP_SIZE:
                continue
            record["player_ids"] = frozenset(players)
            record["players"] = list(players.values())
            teams[record["team_id"]].append(record)
        for records in teams.values():
            records.sort(key=lambda item: (item["datetime"], item["match_id"]))
        return {"teams": dict(teams), "match_datetimes": match_datetimes}
    except sqlite3.Error:
        return {"teams": {}, "match_datetimes": {}}
    finally:
        if owns_connection:
            conn.close()


def _match_datetime_utc(snapshot: dict[str, Any], lineup_store: dict[str, Any]) -> datetime | None:
    stored = (lineup_store.get("match_datetimes") or {}).get(str(snapshot.get("id") or ""))
    if stored is not None:
        return stored
    local_dt = parse_match_datetime(snapshot.get("date"), snapshot.get("hour"))
    if local_dt is None:
        return None
    return (
        local_dt.replace(tzinfo=ZoneInfo("Europe/Madrid"))
        .astimezone(timezone.utc)
        .replace(tzinfo=None)
    )


def _announced_lineup(
    snapshot: dict[str, Any],
    side: str,
    expected_team_id: str | int | None,
) -> dict[str, Any]:
    lineups = snapshot.get("prematch_lineups") or {}
    expected = str(expected_team_id or "")
    candidate = lineups.get(side) or {}
    if expected and str(candidate.get("hltv_team_id") or "") != expected:
        candidate = next(
            (
                lineup
                for lineup in lineups.values()
                if str((lineup or {}).get("hltv_team_id") or "") == expected
            ),
            {},
        )

    players: dict[str, dict[str, Any]] = {}
    for player in candidate.get("players") or []:
        player_id = str(player.get("hltv_player_id") or "")
        if not player_id:
            continue
        players[player_id] = {
            "id": player_id,
            "name": str(player.get("nickname") or player_id),
            "is_standin": bool(player.get("is_standin")),
        }
    team_matches = bool(expected) and str(candidate.get("hltv_team_id") or "") == expected
    return {
        "available": bool(candidate),
        "expected_team_id_available": bool(expected),
        "team_matches": team_matches,
        "team_id": expected or None,
        "players": list(players.values()),
        "player_ids": frozenset(players),
        "complete": team_matches and len(players) == COMPLETE_LINEUP_SIZE,
        "standins": [player for player in players.values() if player["is_standin"]],
    }


def _lineup_players(
    player_ids: set[str] | frozenset[str],
    *sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    names = {
        str(player.get("id")): str(player.get("name") or player.get("id"))
        for source in sources
        for player in source
        if player.get("id")
    }
    return [
        {"id": player_id, "name": names.get(player_id, player_id)}
        for player_id in sorted(player_ids, key=lambda value: names.get(value, value).lower())
    ]


def roster_change_90d(
    snapshot: dict[str, Any],
    side: str,
    team: dict[str, Any],
    lineup_store: dict[str, Any],
    window_days: int = ROSTER_CHANGE_WINDOW_DAYS,
) -> dict[str, Any]:
    """Compare a pre-match announced lineup with prior actual lineups only."""
    current = _announced_lineup(snapshot, side, team.get("id"))
    result: dict[str, Any] = {
        "team_id": str(team.get("id") or "") or None,
        "team_name": team.get("name"),
        "window_days": int(window_days),
        "source": "announced_match_lineup_vs_actual_match_lineups",
        "status": "unknown",
        "red_flag": False,
        "current_lineup_size": len(current["player_ids"]),
        "current_players": current["players"],
        "announced_standins": current["standins"],
        "previous_matches_compared": 0,
        "distinct_lineups_90d": 0,
        "changed_from_latest": None,
        "changed_from_modal": None,
        "players_in": [],
        "players_out": [],
    }
    if not current["available"]:
        result["status"] = "current_lineup_missing"
        return result
    if not current["expected_team_id_available"]:
        result["status"] = "team_id_missing"
        return result
    if not current["team_matches"]:
        result["status"] = "team_lineup_mismatch"
        return result
    if not current["complete"]:
        result["status"] = "current_lineup_incomplete"
        return result

    match_dt = _match_datetime_utc(snapshot, lineup_store)
    if match_dt is None:
        result["status"] = "match_datetime_unknown"
        return result
    captured_at = parse_iso(snapshot.get("captured_at"))
    if captured_at is not None and captured_at > match_dt:
        result["status"] = "post_start_snapshot_rejected"
        return result

    window_start = match_dt - timedelta(days=window_days)
    current_match_id = str(snapshot.get("id") or "")
    history = [
        record
        for record in (lineup_store.get("teams") or {}).get(str(team.get("id") or ""), [])
        if window_start <= record["datetime"] < match_dt
        and record["match_id"] != current_match_id
    ]
    history.sort(key=lambda item: (item["datetime"], item["match_id"]), reverse=True)
    result["window_start_utc"] = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["match_datetime_utc"] = match_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["previous_matches_compared"] = len(history)
    if not history:
        result["status"] = "no_recent_complete_history"
        return result

    latest = history[0]
    signatures = Counter(record["player_ids"] for record in history)
    max_count = max(signatures.values())
    modal_signature = next(
        record["player_ids"]
        for record in history
        if signatures[record["player_ids"]] == max_count
    )
    current_ids = current["player_ids"]
    changed_latest = current_ids != latest["player_ids"]
    historical_player_ids = set().union(*(record["player_ids"] for record in history))
    result.update(
        {
            "status": "changed" if changed_latest else "unchanged",
            "red_flag": changed_latest,
            "changed_from_latest": changed_latest,
            "changed_from_modal": current_ids != modal_signature,
            "previous_matches_compared": len(history),
            "distinct_lineups_90d": len(signatures),
            "modal_lineup_matches": max_count,
            "latest_previous_match_id": latest["match_id"],
            "latest_previous_at_utc": latest["datetime"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latest_previous_players": latest["players"],
            "players_in": _lineup_players(
                current_ids - latest["player_ids"],
                current["players"],
                latest["players"],
            ),
            "players_out": _lineup_players(
                latest["player_ids"] - current_ids,
                current["players"],
                latest["players"],
            ),
            "new_players_in_90d": _lineup_players(
                current_ids - historical_player_ids,
                current["players"],
                *(record["players"] for record in history),
            ),
        }
    )
    return result


def build_player_match_form(master: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    forms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in master.values():
        if not (record.get("data_quality") or {}).get("real_pre_match_snapshot"):
            continue
        if record.get("status") != "completed":
            continue
        detail = record.get("detail") or (record.get("result") or {}).get("detail")
        if not isinstance(detail, dict):
            continue
        completed_at = parse_iso(record.get("completed_at")) or parse_match_datetime(record.get("date"), record.get("hour")) or datetime.min
        for table in detail.get("stats") or []:
            team_key = clean_team(table.get("team"))
            for player in table.get("stats") or []:
                player_key = clean_team(player.get("name"))
                if not player_key:
                    continue
                sample = {
                    "completed_at": completed_at,
                    "team": team_key,
                    "rating": safe_float(player.get("rating 3.0")),
                    "adr": safe_float(player.get("adr")),
                    "kd": safe_float(player.get("kd")),
                    "plus_minus": safe_float(player.get("+/-")),
                    "source": "completed_match_detail",
                }
                forms[player_key].append(sample)
    for rows in forms.values():
        rows.sort(key=lambda row: row["completed_at"])
    return forms


def average_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    return avg(values, default=None) if values else None


def player_form_summary(roster: dict[str, Any], player_forms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    players = []
    window_team_values: dict[int, dict[str, list[float]]] = {
        5: defaultdict(list),
        10: defaultdict(list),
        20: defaultdict(list),
    }
    season_values: dict[str, list[float]] = defaultdict(list)
    real_covered = 0
    season_covered = 0
    for player in roster.get("players") or []:
        player_key = clean_team(player.get("name"))
        real_rows = player_forms.get(player_key, [])
        if real_rows:
            real_covered += 1
        season_stats = (player.get("stats") or {}).get("stats") or {}
        if season_stats:
            season_covered += 1
        season = {
            "rating": numeric_stat(season_stats.get("Rating 3.0")),
            "kpr": numeric_stat(season_stats.get("KPR")),
            "apr": numeric_stat(season_stats.get("APR")),
            "kast": numeric_stat(season_stats.get("KAST")),
            "opening_kpr": numeric_stat(season_stats.get("Opening KPR")),
            "maps": (player.get("stats") or {}).get("maps"),
        }
        for metric, value in season.items():
            if isinstance(value, (int, float)):
                season_values[metric].append(float(value))
        windows = {}
        for window in (5, 10, 20):
            rows = real_rows[-window:]
            metrics = {
                "rating": average_metric(rows, "rating"),
                "adr": average_metric(rows, "adr"),
                "kd": average_metric(rows, "kd"),
                "plus_minus": average_metric(rows, "plus_minus"),
                "samples": len(rows),
            }
            windows[f"last{window}"] = metrics
            for metric, value in metrics.items():
                if metric != "samples" and value is not None:
                    window_team_values[window][metric].append(value)
        players.append(
            {
                "id": player.get("id"),
                "name": player.get("name"),
                "season": season,
                "windows": windows,
                "real_samples": len(real_rows),
            }
        )

    team_windows = {}
    for window, metrics in window_team_values.items():
        team_windows[f"last{window}"] = {
            metric: avg(values, default=None) if values else None
            for metric, values in metrics.items()
        }
        team_windows[f"last{window}"]["covered_players"] = sum(
            1 for player in players if player["windows"][f"last{window}"]["samples"] > 0
        )
    return {
        "players": players,
        "real_form_coverage": real_covered / max(len(players), 1),
        "season_form_coverage": season_covered / max(len(players), 1),
        "team_windows": team_windows,
        "season_avg": {
            metric: avg(values, default=None) if values else None
            for metric, values in season_values.items()
        },
        "source_note": "last5/10/20 require completed real match details; season stats come from HLTV compare.",
    }


def match_context_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    context = snapshot.get("match_context")
    if isinstance(context, dict) and context:
        raw_meta = context.get("raw_meta")
        return parse_match_context_meta(raw_meta) if raw_meta else context

    for container in (
        snapshot.get("veto"),
        (snapshot.get("detail") or {}).get("veto") if isinstance(snapshot.get("detail"), dict) else None,
    ):
        if not isinstance(container, dict):
            continue
        meta = container.get("meta")
        if meta:
            return parse_match_context_meta(meta)

    raw_html = snapshot.get("raw_html") or {}
    source_file = raw_html.get("source_file") if isinstance(raw_html, dict) else None
    if source_file:
        path = DAILY_ROOT / Path(source_file)
        if path.exists():
            try:
                if path.suffix == ".gz":
                    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                        html = handle.read()
                else:
                    html = path.read_text(encoding="utf-8", errors="replace")
                selector = Selector(text=html)
                boxes = selector.css(".veto-box")
                if boxes:
                    meta = " ".join(text.strip() for text in boxes[0].css("::text").getall() if text.strip())
                    if meta:
                        return parse_match_context_meta(meta)
            except Exception:
                return {}
    return {}


def tournament_context(
    event: str,
    link: str | None,
    fmt: str | None,
    event_volatility_label: str,
    match_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    match_context = match_context or {}
    text = f"{event or ''} {link or ''}".lower()
    stage = "unknown"
    parsed_stage = match_context.get("stage")
    if parsed_stage and parsed_stage != "unknown":
        stage = parsed_stage
    elif "grand-final" in text or "grand final" in text:
        stage = "grand_final"
    elif "semi" in text:
        stage = "semi_final"
    elif "quarter" in text:
        stage = "quarter_final"
    elif "playoff" in text:
        stage = "playoff"
    elif "lower" in text:
        stage = "lower_bracket"
    elif "upper" in text:
        stage = "upper_bracket"
    elif "group" in text:
        stage = "group"
    elif "qualifier" in text:
        stage = "qualifier"
    elif "league" in text:
        stage = "league"

    if match_context.get("incentive_uncertainty"):
        incentive_uncertainty = match_context["incentive_uncertainty"]
    elif any(token in text for token in ["showmatch", "3rd-place", "third-place", "placement", "consolation"]):
        incentive_uncertainty = "high"
    elif stage in {"group", "league", "unknown"}:
        incentive_uncertainty = "medium"
    else:
        incentive_uncertainty = "low"

    bracket = match_context.get("bracket") or ("lower" if "lower" in text else "upper" if "upper" in text else None)
    environment = match_context.get("environment") or "unknown"
    elimination_likely = bool(match_context.get("loser_eliminated")) or stage in {
        "lower_bracket",
        "playoff",
        "quarter",
        "quarter_final",
        "semi",
        "semi_final",
        "final",
        "grand_final",
    }
    high_stakes = bool(match_context.get("high_stakes")) or stage in {
        "quarter",
        "quarter_final",
        "semi",
        "semi_final",
        "final",
        "grand_final",
        "playoff",
        "lower_bracket",
    }
    source = "hltv_maps_box" if match_context.get("raw_meta") or match_context.get("stage_detail") else "heuristic_event_text"
    return {
        "stage": stage,
        "stage_detail": match_context.get("stage_detail"),
        "format": fmt,
        "environment": environment,
        "is_lan": True if environment == "lan" else False if environment == "online" else None,
        "bracket": bracket,
        "elimination_likely": elimination_likely,
        "high_stakes": high_stakes,
        "winner_advances": bool(match_context.get("winner_advances")),
        "loser_eliminated": bool(match_context.get("loser_eliminated")),
        "swiss_round": match_context.get("swiss_round"),
        "swiss_record": match_context.get("swiss_record"),
        "incentive_label": match_context.get("incentive_label"),
        "incentive_uncertainty": incentive_uncertainty,
        "has_substitution_note": bool(match_context.get("has_substitution_note")),
        "substitution_notes": match_context.get("substitution_notes") or [],
        "prize_context_known": False,
        "event_volatility_label": event_volatility_label,
        "source": source,
    }


def add_schedule_item(index: dict[str, list[dict[str, Any]]], team_name: str | None, when: datetime | None, source: str, match_id: str | None) -> None:
    if not team_name or not when:
        return
    index[clean_team(team_name)].append({"datetime": when, "source": source, "match_id": match_id})


def build_schedule_index(history: list[dict[str, Any]], master: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        when = row.get("date_obj")
        if when:
            when = when.replace(hour=12, minute=0)
        add_schedule_item(index, row.get("team1"), when, "legacy_history_date", row.get("id"))
        add_schedule_item(index, row.get("team2"), when, "legacy_history_date", row.get("id"))
    for record in master.values():
        detail = record.get("detail") or {}
        team1, team2 = team_names_from_detail(detail if isinstance(detail, dict) else None)
        when = parse_match_datetime(record.get("date"), record.get("hour")) or parse_iso(record.get("completed_at"))
        add_schedule_item(index, team1, when, "real_master", record.get("id"))
        add_schedule_item(index, team2, when, "real_master", record.get("id"))
    for snapshot in snapshots:
        detail = snapshot.get("detail") or {}
        match = detail.get("match") or {}
        team1 = (match.get("team1") or (snapshot.get("upcoming_row") or {}).get("team1") or {}).get("name")
        team2 = (match.get("team2") or (snapshot.get("upcoming_row") or {}).get("team2") or {}).get("name")
        when = parse_match_datetime(snapshot.get("date"), snapshot.get("hour"))
        add_schedule_item(index, team1, when, "current_run", snapshot.get("id"))
        add_schedule_item(index, team2, when, "current_run", snapshot.get("id"))
    for rows in index.values():
        rows.sort(key=lambda row: row["datetime"])
    return index


def fatigue_metrics(team_name: str, match_dt: datetime | None, schedule_index: dict[str, list[dict[str, Any]]], match_id: str) -> dict[str, Any]:
    if not match_dt:
        return {"status": "unknown_time", "last24_count": 0, "last48_count": 0, "same_day_total": 0}
    rows = [row for row in schedule_index.get(clean_team(team_name), []) if row.get("match_id") != match_id]
    previous = [row for row in rows if row["datetime"] < match_dt]
    next_rows = [row for row in rows if row["datetime"] > match_dt]
    last24 = [row for row in previous if match_dt - row["datetime"] <= timedelta(hours=24)]
    last48 = [row for row in previous if match_dt - row["datetime"] <= timedelta(hours=48)]
    same_day = [row for row in rows if row["datetime"].date() == match_dt.date()]
    next24 = [row for row in next_rows if row["datetime"] - match_dt <= timedelta(hours=24)]
    return {
        "status": "ok",
        "last24_count": len(last24),
        "last48_count": len(last48),
        "same_day_total": len(same_day) + 1,
        "next24_count": len(next24),
        "back_to_back": len(last24) > 0 or len(same_day) > 0,
        "schedule_density_48h": len(last48) + len([row for row in next24 if row["datetime"] - match_dt <= timedelta(hours=48)]),
        "online_lan": "unknown",
        "travel_risk": "unknown",
        "sources": sorted({row["source"] for row in last48 + same_day + next24}),
    }


def calibration_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(samples)
    if not n:
        return {
            "n": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
            "note": "Sin partidos completados con snapshot real pre-partido.",
        }
    eps = 1e-6
    brier = avg([(row["p"] - row["y"]) ** 2 for row in samples])
    log_loss = avg([
        -(row["y"] * math.log(clamp(row["p"], eps, 1 - eps)) + (1 - row["y"]) * math.log(clamp(1 - row["p"], eps, 1 - eps)))
        for row in samples
    ])
    accuracy = avg([int((row["p"] >= 0.5) == bool(row["y"])) for row in samples])
    bins = [[] for _ in range(10)]
    for row in samples:
        bins[min(9, int(clamp(row["p"]) * 10))].append(row)
    ece = 0.0
    bin_rows = []
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        conf = avg([row["p"] for row in bucket])
        acc = avg([row["y"] for row in bucket])
        ece += (len(bucket) / n) * abs(conf - acc)
        bin_rows.append({"bin": index, "n": len(bucket), "avg_prob": conf, "empirical_winrate": acc})
    return {
        "n": n,
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "bins": bin_rows,
    }


def compute_daily_calibration(master: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    latest_prediction: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS_DIR.glob("*/predictions_enriched.json")):
        for entry in read_json(path, []):
            if not (entry.get("data_quality") or {}).get("real_pre_match_snapshot"):
                continue
            captured_at = entry.get("captured_at") or ""
            current = latest_prediction.get(entry.get("id"))
            if current is None or captured_at > (current.get("captured_at") or ""):
                latest_prediction[entry.get("id")] = entry

    samples = []
    for match_id, entry in latest_prediction.items():
        record = master.get(str(match_id), {})
        if record.get("status") != "completed" or not record.get("score"):
            continue
        score = record["score"]
        y = 1 if int(score.get("team1", 0)) > int(score.get("team2", 0)) else 0
        pred = entry.get("prediction") or {}
        # Validamos el modelo de stats calibrado (Model A); si falta, el blend.
        p = pred.get("model_prob_team1")
        if p is None:
            p = pred.get("blended_prob_team1")
        if p is None:
            continue
        samples.append(
            {
                "id": match_id,
                "p": p,
                "y": y,
                "captured_at": entry.get("captured_at"),
                "completed_at": record.get("completed_at"),
            }
        )

    samples.sort(key=lambda row: row.get("completed_at") or row.get("captured_at") or "")
    calibration = {
        "all": calibration_metrics(samples),
        "last_50": calibration_metrics(samples[-50:]),
        "last_200": calibration_metrics(samples[-200:]),
        "samples": samples[-500:],
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(MASTER_CALIBRATION, calibration)
    write_json(run_dir / "calibration.json", calibration)
    return calibration


def _market_probability_team1(entry: dict[str, Any], record: dict[str, Any]) -> float | None:
    """Normalized market probability for team1 from a saved pre-match snapshot.

    Favorite/upset history should be an external market fact, not a model echo.
    Prefer the odds captured in the prediction snapshot; fall back to the
    master opening odds if the snapshot did not embed odds.
    """
    sources = [
        ((entry.get("odds") or {}).get("average") or {}),
        record.get("opening_odds") or {},
        record.get("latest_odds_point") or {},
    ]
    for source in sources:
        try:
            p = float(source.get("team1_implied_prob_norm"))
        except (TypeError, ValueError):
            continue
        if 0.0 < p < 1.0 and abs(p - 0.5) >= 0.005:
            return p
    return None


def build_favorite_upset_records(master: dict[str, Any]) -> list[dict[str, Any]]:
    """Completed pre-match predictions used to audit market-favorite upset history."""
    latest_prediction: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS_DIR.glob("*/predictions_enriched.json")):
        for entry in read_json(path, []):
            if not (entry.get("data_quality") or {}).get("real_pre_match_snapshot"):
                continue
            match_id = str(entry.get("id") or "")
            if not match_id:
                continue
            captured_at = entry.get("captured_at") or ""
            current = latest_prediction.get(match_id)
            if current is None or captured_at > (current.get("captured_at") or ""):
                latest_prediction[match_id] = entry

    records: list[dict[str, Any]] = []
    for match_id, entry in latest_prediction.items():
        record = master.get(match_id, {})
        if record.get("status") != "completed" or not record.get("score"):
            continue
        captured_at = parse_iso(entry.get("captured_at"))
        completed_at = parse_iso(record.get("completed_at"))
        if captured_at and completed_at and captured_at > completed_at:
            continue

        score = record.get("score") or {}
        score1 = parse_int(score.get("team1"))
        score2 = parse_int(score.get("team2"))
        if score1 is None or score2 is None or score1 == score2:
            continue
        match_detail = ((record.get("detail") or {}).get("match") or {})
        team1 = ((entry.get("team1") or {}).get("name") or (match_detail.get("team1") or {}).get("name"))
        team2 = ((entry.get("team2") or {}).get("name") or (match_detail.get("team2") or {}).get("name"))
        t1, t2 = clean_team(team1), clean_team(team2)
        if not t1 or not t2:
            continue
        market_p1 = _market_probability_team1(entry, record)
        favorite_side = "team1" if market_p1 is not None and market_p1 >= 0.5 else "team2" if market_p1 is not None else None
        favorite_key = t1 if favorite_side == "team1" else t2 if favorite_side == "team2" else None
        winner_key = t1 if score1 > score2 else t2
        records.append(
            {
                "match_id": match_id,
                "date": record.get("date") or entry.get("date"),
                "date_obj": parse_date(record.get("date") or entry.get("date")),
                "team1_key": t1,
                "team2_key": t2,
                "favorite_key": favorite_key,
                "favorite_side": favorite_side,
                "favorite_probability": (
                    max(market_p1, 1.0 - market_p1) if market_p1 is not None else None
                ),
                "favorite_source": "market_opening_or_snapshot_odds" if market_p1 is not None else "missing_market_odds",
                "winner_key": winner_key,
                "favorite_lost": bool(favorite_key and winner_key != favorite_key),
            }
        )
    records.sort(key=lambda row: (row.get("date_obj") or datetime.min, row.get("match_id") or ""))
    return records


def favorite_upset_summary(
    records: list[dict[str, Any]],
    favorite_key: str,
    match_dt: datetime | None,
    window_days: int = 90,
) -> dict[str, Any]:
    if not favorite_key:
        return {
            "window_days": window_days,
            "favorite_losses": 0,
            "favorite_matches": 0,
            "played_matches": 0,
            "market_odds_matches": 0,
            "favorite_loss_rate": None,
            "risk_label": "sin datos",
            "source": "market_opening_or_snapshot_odds",
        }
    asof = match_dt or datetime.utcnow()
    start = asof - timedelta(days=window_days)
    team_rows = [
        row for row in records
        if row.get("date_obj") and start <= row["date_obj"] < asof
        and favorite_key in {row.get("team1_key"), row.get("team2_key")}
    ]
    market_rows = [row for row in team_rows if row.get("favorite_source") == "market_opening_or_snapshot_odds"]
    favorite_rows = [row for row in market_rows if row.get("favorite_key") == favorite_key]
    favorite_losses = sum(1 for row in favorite_rows if row.get("favorite_lost"))
    favorite_matches = len(favorite_rows)
    played_matches = len(team_rows)
    market_odds_matches = len(market_rows)
    loss_rate = favorite_losses / favorite_matches if favorite_matches else None
    if market_odds_matches == 0:
        risk_label = "sin odds hist."
    elif favorite_matches < 3:
        risk_label = "muestra baja"
    elif loss_rate is not None and loss_rate >= 0.35:
        risk_label = "upset alto"
    elif loss_rate is not None and loss_rate >= 0.20:
        risk_label = "upset medio"
    else:
        risk_label = "upset bajo"
    return {
        "window_days": window_days,
        "favorite_losses": favorite_losses,
        "favorite_matches": favorite_matches,
        "played_matches": played_matches,
        "market_odds_matches": market_odds_matches,
        "favorite_loss_rate": loss_rate,
        "risk_label": risk_label,
        "source": "market_opening_or_snapshot_odds",
        "window_start": start.strftime("%Y-%m-%d"),
        "as_of": asof.strftime("%Y-%m-%d"),
    }


def flag_match(entry: dict[str, Any], features: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, str]]:
    flags = []
    odds = entry.get("odds") or {}
    controls = entry.get("controls") or {}
    market = controls.get("market") or {}
    map_pool = controls.get("map_pool") or controls.get("map_veto") or {}
    roster = controls.get("roster_stability") or {}
    roster_changes = controls.get("roster_change_90d") or {}
    player_form = controls.get("player_form") or {}
    context = controls.get("tournament_context") or {}
    fatigue = controls.get("fatigue") or {}
    calibration = controls.get("calibration") or {}
    favorite_upset = controls.get("favorite_upset") or {}
    if not odds.get("available"):
        flags.append({"level": "warning", "code": "NO_ODDS", "message": "No hay cuotas disponibles."})
    elif odds.get("bookmaker_count", 0) < 3:
        flags.append({"level": "info", "code": "LOW_ODDS_SOURCES", "message": "Pocas casas de apuestas disponibles."})

    if market.get("drift_abs") is not None and market["drift_abs"] >= 0.06:
        flags.append({"level": "warning", "code": "ODDS_STRONG_DRIFT", "message": "Movimiento fuerte de odds desde apertura."})
    if market.get("consensus_label") == "low":
        flags.append({"level": "warning", "code": "LOW_MARKET_CONSENSUS", "message": "Las casas no estan alineadas."})

    if entry.get("format") == "bo1":
        flags.append({"level": "warning", "code": "BO1_HIGH_VARIANCE", "message": "BO1 suele tener mas varianza."})

    if min(features["matches_team1"], features["matches_team2"]) < 10:
        flags.append({"level": "warning", "code": "LOW_TEAM_HISTORY", "message": "Al menos un equipo tiene poca historia reciente."})

    if abs(features["elo_diff"]) < 40:
        flags.append({"level": "info", "code": "CLOSE_ELO", "message": "Elo muy parejo; partido menos estable."})

    if features["event_volatility_label"] == "high":
        flags.append({"level": "warning", "code": "VOLATILE_EVENT", "message": "Torneo historicamente impredecible."})
    elif features["event_volatility_label"] == "unknown_event_history":
        flags.append({"level": "info", "code": "UNKNOWN_EVENT", "message": "Poco historico para este torneo."})

    if min(features["player_coverage_team1"], features["player_coverage_team2"]) < 0.6:
        flags.append({"level": "warning", "code": "LOW_PLAYER_STATS", "message": "Baja cobertura de stats de jugadores."})

    if map_pool.get("status") == "low_data":
        flags.append({"level": "info", "code": "LOW_MAP_POOL_DATA", "message": "Poco dato real para estimar pool de mapas pre-partido."})
    elif abs(map_pool.get("map_pool_advantage_team1") or 0) >= 0.12:
        flags.append({"level": "info", "code": "MAP_POOL_EDGE", "message": "Hay ventaja relevante en pool de mapas."})

    changed_rosters = [
        side_roster
        for side_roster in (
            roster_changes.get("team1") or {},
            roster_changes.get("team2") or {},
        )
        if side_roster.get("red_flag")
    ]
    if changed_rosters:
        details = []
        for side_roster in changed_rosters:
            players_in = (
                ", ".join(player["name"] for player in side_roster.get("players_in") or [])
                or "sin alta identificada"
            )
            players_out = (
                ", ".join(player["name"] for player in side_roster.get("players_out") or [])
                or "sin baja identificada"
            )
            details.append(
                f"{side_roster.get('team_name') or 'Equipo'}: entra {players_in}; "
                f"sale {players_out}; comparado con "
                f"{side_roster.get('latest_previous_match_id') or 'ultimo partido'}."
            )
        flags.insert(
            0,
            {
                "level": "danger",
                "code": "ROSTER_CHANGE_90D",
                "message": " ".join(details),
            },
        )

    if any(
        side_roster.get("announced_standins")
        for side_roster in roster_changes.values()
    ):
        flags.append(
            {
                "level": "warning",
                "code": "STANDIN_RISK",
                "message": "La alineacion anunciada identifica al menos un stand-in.",
            }
        )
    elif any(
        (roster.get(side) or {}).get("standin_risk")
        for side in ("team1", "team2")
    ):
        flags.append(
            {
                "level": "warning",
                "code": "ROSTER_PROFILE_INCOMPLETE",
                "message": "El perfil de al menos un equipo no contiene exactamente cinco jugadores.",
            }
        )

    if min((player_form.get("team1") or {}).get("real_form_coverage", 0), (player_form.get("team2") or {}).get("real_form_coverage", 0)) < 0.6:
        flags.append({"level": "info", "code": "PLAYER_FORM_BACKFILL_ONLY", "message": "Forma 5/10/20 aun depende poco de matches reales."})

    if context.get("incentive_uncertainty") == "high":
        flags.append({"level": "warning", "code": "INCENTIVE_RISK", "message": "Contexto con incentivo competitivo potencialmente raro."})
    elif context.get("incentive_uncertainty") == "medium" and not context.get("prize_context_known"):
        flags.append({"level": "info", "code": "INCENTIVE_UNKNOWN", "message": "No se conoce aun el incentivo exacto del partido."})
    if context.get("high_stakes"):
        flags.append({"level": "info", "code": "HIGH_STAKES_PLAYOFF", "message": "Partido de alto contexto competitivo."})
    if context.get("environment") == "lan":
        flags.append({"level": "info", "code": "LAN_MATCH", "message": "Partido marcado por HLTV como LAN."})
    elif context.get("environment") == "online":
        flags.append({"level": "info", "code": "ONLINE_MATCH", "message": "Partido marcado por HLTV como online."})
    if context.get("winner_advances"):
        flags.append({"level": "info", "code": "WINNER_ADVANCES", "message": "HLTV indica que el ganador avanza."})
    if context.get("loser_eliminated"):
        flags.append({"level": "warning", "code": "ELIMINATION_MATCH", "message": "HLTV indica contexto de eliminacion."})
    if context.get("has_substitution_note"):
        flags.append({"level": "warning", "code": "SUBSTITUTION_NOTE", "message": "HLTV indica sustitucion o stand-in en el partido."})

    fatigue_values = [fatigue.get("team1") or {}, fatigue.get("team2") or {}]
    if any(row.get("last24_count", 0) > 0 for row in fatigue_values):
        flags.append({"level": "warning", "code": "FATIGUE_BACK_TO_BACK", "message": "Equipo con partido en las ultimas 24h."})
    elif any(row.get("same_day_total", 0) >= 2 for row in fatigue_values):
        flags.append({"level": "warning", "code": "MULTI_MATCH_DAY", "message": "Equipo con varios partidos el mismo dia."})
    if any(row.get("last48_count", 0) >= 2 for row in fatigue_values):
        flags.append({"level": "warning", "code": "HIGH_SCHEDULE_DENSITY", "message": "Densidad alta de partidos en 48h."})

    if odds.get("available") and odds.get("average"):
        odds_p = odds["average"]["team1_implied_prob_norm"]
        if abs(odds_p - prediction["model_prob_team1"]) > 0.18:
            flags.append({"level": "warning", "code": "MODEL_ODDS_DISAGREE", "message": "Modelo y mercado discrepan fuerte."})

    if market.get("model_edge_abs") is not None and market["model_edge_abs"] >= 0.08:
        flags.append({"level": "info", "code": "VALUE_EDGE", "message": "El modelo ve edge relevante contra el mercado."})

    if prediction["confidence"] < 0.58:
        flags.append({"level": "info", "code": "LOW_CONFIDENCE", "message": "Prediccion cerca de 50/50."})

    fav_matches = favorite_upset.get("favorite_matches") or 0
    fav_loss_rate = favorite_upset.get("favorite_loss_rate")
    if fav_matches >= 3 and fav_loss_rate is not None and fav_loss_rate >= 0.35:
        flags.append({
            "level": "warning",
            "code": "FAV_UPSET_HISTORY",
            "message": "El favorito actual ha perdido a menudo cuando era favorito en la ventana reciente.",
        })
    elif fav_matches >= 3 and fav_loss_rate is not None and fav_loss_rate >= 0.20:
        flags.append({
            "level": "info",
            "code": "FAV_UPSET_HISTORY",
            "message": "El favorito actual tiene algunos upsets recientes como favorito.",
        })

    rel = prediction.get("reliability_score")
    decision_conf = prediction.get("decision_confidence")
    if rel is not None and rel < 0.35 and prediction.get("confidence", 0.5) >= 0.65:
        flags.append({
            "level": "warning",
            "code": "LOW_RELIABILITY_OVERCONFIDENCE",
            "message": "Modelo seguro, pero con respaldo de datos bajo.",
        })
    if decision_conf is not None and decision_conf < 0.58 and prediction.get("confidence", 0.5) >= 0.58:
        flags.append({
            "level": "info",
            "code": "DECISION_SHRUNK_TO_COINFLIP",
            "message": "La capa operativa reduce la confianza por fiabilidad/mercado.",
        })

    if (calibration.get("all") or {}).get("n", 0) < 30:
        flags.append({"level": "info", "code": "CALIBRATION_LOW_SAMPLE", "message": "Aun hay pocos resultados reales para calibracion diaria."})

    return flags


def enrich(run_dir: Path, history_path: Path) -> list[dict[str, Any]]:
    history = load_history(history_path)
    engine = load_model_engine(history_path)
    print(
        "[model] artefacto entrenado cargado: "
        + (engine["artifact"].metadata.get("production_model", "?") if engine else "NO (fallback logístico)")
    )
    model_metadata = engine["artifact"].metadata if engine else {}
    external_feature_store = (
        cs2_dataio.load_external_feature_store(LIVE_DB)
        if _CS2_OK and LIVE_DB.exists() else {"rankings": {}, "rosters": {}}
    )
    actual_lineup_store = (
        load_actual_lineup_store(LIVE_DB)
        if LIVE_DB.exists()
        else {"teams": {}, "match_datetimes": {}}
    )
    master = load_master()
    state = build_history_state(history)
    historical_p = historical_probability_rows(history, state)
    profiles = team_profile_index(run_dir)
    roster_history = build_roster_history(run_dir, profiles)
    map_state = build_map_state(master)
    player_forms = build_player_match_form(master)
    player_stats = load_player_stats(run_dir)
    manifest = read_json(run_dir / "manifest.json", {})
    now_local = datetime.now(ZoneInfo("Europe/Madrid")).replace(tzinfo=None)
    run_started_local = parse_iso_as_local(manifest.get("started_at")) or now_local
    publish_reference_local = max(run_started_local, now_local)
    snapshots = [read_json(path, {}) for path in sorted((run_dir / "match_snapshots").glob("*.json"))]
    for snapshot in snapshots:
        snapshot["run_id"] = run_dir.name
    schedule_index = build_schedule_index(history, master, snapshots)
    calibration = compute_daily_calibration(master, run_dir)
    market_blend_policy = build_market_blend_policy(master)
    favorite_upset_records = build_favorite_upset_records(master)
    enriched = []
    skipped_unpublishable: dict[str, int] = defaultdict(int)

    for snapshot in snapshots:
        publishable, skip_reason = is_snapshot_publishable(snapshot, publish_reference_local)
        if not publishable:
            skipped_unpublishable[skip_reason] += 1
            continue
        detail = snapshot.get("detail") or {}
        match = detail.get("match") or {}
        team1 = match.get("team1") or (snapshot.get("upcoming_row") or {}).get("team1") or {}
        team2 = match.get("team2") or (snapshot.get("upcoming_row") or {}).get("team2") or {}
        t1, t2 = clean_team(team1.get("name")), clean_team(team2.get("name"))
        if not t1 or not t2:
            continue

        elo1 = state["elos"][t1]
        elo2 = state["elos"][t2]
        wins1 = state["wins"][t1]
        wins2 = state["wins"][t2]
        diffs1 = state["diffs"][t1]
        diffs2 = state["diffs"][t2]
        event = snapshot.get("event") or ""
        event1 = state["event_stats"][(t1, event)]
        event2 = state["event_stats"][(t2, event)]
        roster1 = roster_summary(team1.get("id"), profiles, player_stats)
        roster2 = roster_summary(team2.get("id"), profiles, player_stats)
        player_form1 = player_form_summary(roster1, player_forms)
        player_form2 = player_form_summary(roster2, player_forms)
        map_pool = map_pool_estimate(team1.get("name") or t1, team2.get("name") or t2, map_state)
        roster_control1 = roster_stability(team1.get("id"), profiles.get(str(team1.get("id") or "")), roster_history, run_dir)
        roster_control2 = roster_stability(team2.get("id"), profiles.get(str(team2.get("id") or "")), roster_history, run_dir)
        match_context = match_context_from_snapshot(snapshot)
        match_dt = parse_match_datetime(snapshot.get("date"), snapshot.get("hour"))
        roster_change1 = roster_change_90d(
            snapshot,
            "team1",
            team1,
            actual_lineup_store,
        )
        roster_change2 = roster_change_90d(
            snapshot,
            "team2",
            team2,
            actual_lineup_store,
        )
        fatigue1 = fatigue_metrics(team1.get("name") or t1, match_dt, schedule_index, snapshot["id"])
        fatigue2 = fatigue_metrics(team2.get("name") or t2, match_dt, schedule_index, snapshot["id"])

        rating1 = roster1["avg"].get("Rating 3.0") or 1.0
        rating2 = roster2["avg"].get("Rating 3.0") or 1.0
        rating1_max = roster_metric(roster1, "rating", "max", rating1) or rating1
        rating2_max = roster_metric(roster2, "rating", "max", rating2) or rating2
        rating1_min = roster_metric(roster1, "rating", "min", rating1) or rating1
        rating2_min = roster_metric(roster2, "rating", "min", rating2) or rating2
        rating1_std = roster_metric(roster1, "rating", "std", 0.0) or 0.0
        rating2_std = roster_metric(roster2, "rating", "std", 0.0) or 0.0
        rating1_top2 = roster_metric(roster1, "rating", "top2_avg", rating1_max)
        rating2_top2 = roster_metric(roster2, "rating", "top2_avg", rating2_max)
        rating1_bottom2 = roster_metric(roster1, "rating", "bottom2_avg", rating1_min)
        rating2_bottom2 = roster_metric(roster2, "rating", "bottom2_avg", rating2_min)
        rating1_median = roster_metric(roster1, "rating", "median", rating1)
        rating2_median = roster_metric(roster2, "rating", "median", rating2)
        rating1_top2 = rating1_max if rating1_top2 is None else rating1_top2
        rating2_top2 = rating2_max if rating2_top2 is None else rating2_top2
        rating1_bottom2 = rating1_min if rating1_bottom2 is None else rating1_bottom2
        rating2_bottom2 = rating2_min if rating2_bottom2 is None else rating2_bottom2
        rating1_median = rating1 if rating1_median is None else rating1_median
        rating2_median = rating2 if rating2_median is None else rating2_median
        real_rating1 = ((player_form1.get("team_windows") or {}).get("last5") or {}).get("rating")
        real_rating2 = ((player_form2.get("team_windows") or {}).get("last5") or {}).get("rating")
        map_advantage = map_pool.get("map_pool_advantage_team1") or 0.0
        roster_days1 = roster_control1.get("days_with_current_roster") or 0.0
        roster_days2 = roster_control2.get("days_with_current_roster") or 0.0
        roster_snapshot_available = (
            min(roster_control1.get("roster_size", 0), roster_control2.get("roster_size", 0)) >= 5
        )
        player_coverage_min = min(roster1["coverage"], roster2["coverage"])
        player_maps_min = min(
            (roster1.get("distribution") or {}).get("maps_total") or 0,
            (roster2.get("distribution") or {}).get("maps_total") or 0,
        )
        player_maps_per_player_min = min(
            (roster1.get("distribution") or {}).get("maps_min") or 0,
            (roster2.get("distribution") or {}).get("maps_min") or 0,
        )
        features = {
            "elo_team1": elo1,
            "elo_team2": elo2,
            "elo_diff": elo1 - elo2,
            "elo_prob_team1": elo_prob(elo1, elo2),
            "elo_prob_centered": elo_prob(elo1, elo2) - 0.5,
            "matches_team1": state["matches"][t1],
            "matches_team2": state["matches"][t2],
            "matches_log_diff": math.log1p(state["matches"][t1]) - math.log1p(state["matches"][t2]),
            "winrate_team1": winrate(wins1),
            "winrate_team2": winrate(wins2),
            "winrate_diff": winrate(wins1) - winrate(wins2),
            "last10_winrate_diff": winrate(wins1, 10) - winrate(wins2, 10),
            "avg_score_diff": avg(diffs1) - avg(diffs2),
            "recent_opponent_elo_diff": avg(state["opponent_elos"][t1], 1500.0) - avg(state["opponent_elos"][t2], 1500.0),
            "h2h_winrate_centered": 0.0,
            "event_team1_matches": event1["matches"],
            "event_team2_matches": event2["matches"],
            "event_team1_winrate": (event1["wins"] + 1) / (event1["matches"] + 2),
            "event_team2_winrate": (event2["wins"] + 1) / (event2["matches"] + 2),
            "player_coverage_team1": roster1["coverage"],
            "player_coverage_team2": roster2["coverage"],
            "player_coverage_min": player_coverage_min,
            "player_snapshot_available": 1.0 if player_coverage_min >= 0.8 and player_maps_per_player_min >= 5 else 0.0,
            "player_maps_min": player_maps_min,
            "player_maps_per_player_min": player_maps_per_player_min,
            "player_rating_team1": rating1,
            "player_rating_team2": rating2,
            "player_rating_diff": rating1 - rating2,
            "player_rating_max_team1": rating1_max,
            "player_rating_max_team2": rating2_max,
            "player_rating_max_diff": rating1_max - rating2_max,
            "player_rating_min_team1": rating1_min,
            "player_rating_min_team2": rating2_min,
            "player_rating_min_diff": rating1_min - rating2_min,
            "player_rating_std_team1": rating1_std,
            "player_rating_std_team2": rating2_std,
            "player_rating_std_diff": rating1_std - rating2_std,
            "player_rating_top2_avg_team1": rating1_top2,
            "player_rating_top2_avg_team2": rating2_top2,
            "player_rating_top2_avg_diff": rating1_top2 - rating2_top2,
            "player_rating_bottom2_avg_team1": rating1_bottom2,
            "player_rating_bottom2_avg_team2": rating2_bottom2,
            "player_rating_bottom2_avg_diff": rating1_bottom2 - rating2_bottom2,
            "player_rating_median_team1": rating1_median,
            "player_rating_median_team2": rating2_median,
            "player_rating_median_diff": rating1_median - rating2_median,
            "player_rating_spread_team1": rating1_max - rating1_min,
            "player_rating_spread_team2": rating2_max - rating2_min,
            "player_rating_spread_diff": (rating1_max - rating1_min) - (rating2_max - rating2_min),
            "player_star_gap_team1": rating1_max - rating1,
            "player_star_gap_team2": rating2_max - rating2,
            "player_star_gap_diff": (rating1_max - rating1) - (rating2_max - rating2),
            "player_weak_link_gap_team1": rating1 - rating1_min,
            "player_weak_link_gap_team2": rating2 - rating2_min,
            "player_weak_link_gap_diff": (rating1 - rating1_min) - (rating2 - rating2_min),
            "player_kpr_diff": (roster_metric(roster1, "kpr", "avg", 0.0) or 0.0) - (roster_metric(roster2, "kpr", "avg", 0.0) or 0.0),
            "player_kast_diff": (roster_metric(roster1, "kast", "avg", 70.0) or 70.0) - (roster_metric(roster2, "kast", "avg", 70.0) or 70.0),
            "player_adr_diff": (roster_metric(roster1, "adr", "avg", 75.0) or 75.0) - (roster_metric(roster2, "adr", "avg", 75.0) or 75.0),
            "player_impact_diff": (roster_metric(roster1, "impact", "avg", 1.0) or 1.0) - (roster_metric(roster2, "impact", "avg", 1.0) or 1.0),
            "player_round_swing_diff": (roster_metric(roster1, "round_swing", "avg", 0.0) or 0.0) - (roster_metric(roster2, "round_swing", "avg", 0.0) or 0.0),
            "player_round_swing_top2_avg_diff": (roster_metric(roster1, "round_swing", "top2_avg", 0.0) or 0.0)
            - (roster_metric(roster2, "round_swing", "top2_avg", 0.0) or 0.0),
            "player_opening_kpr_diff": (roster_metric(roster1, "opening_kpr", "avg", 0.0) or 0.0) - (roster_metric(roster2, "opening_kpr", "avg", 0.0) or 0.0),
            "player_opening_kpr_top2_avg_diff": (roster_metric(roster1, "opening_kpr", "top2_avg", 0.0) or 0.0)
            - (roster_metric(roster2, "opening_kpr", "top2_avg", 0.0) or 0.0),
            "player_stats_window_team1": roster1.get("preferred_time_filter"),
            "player_stats_window_team2": roster2.get("preferred_time_filter"),
            "player_real_form_coverage_team1": player_form1["real_form_coverage"],
            "player_real_form_coverage_team2": player_form2["real_form_coverage"],
            "player_real_rating_team1": real_rating1,
            "player_real_rating_team2": real_rating2,
            "player_real_rating_diff": (real_rating1 - real_rating2) if real_rating1 is not None and real_rating2 is not None else 0.0,
            "map_pool_advantage_team1": map_advantage,
            "map_pool_coverage": map_pool.get("coverage"),
            "roster_days_team1": roster_days1,
            "roster_days_team2": roster_days2,
            "roster_days_log_diff": math.log1p(roster_days1) - math.log1p(roster_days2) if roster_snapshot_available else 0.0,
            "roster_available": float(roster_snapshot_available),
            "roster_days_min": min(roster_days1, roster_days2) if roster_snapshot_available else 0.0,
            "roster_size_min": min(roster_control1.get("roster_size", 0), roster_control2.get("roster_size", 0)) if roster_snapshot_available else 0.0,
            "roster_size_diff": roster_control1.get("roster_size", 0) - roster_control2.get("roster_size", 0) if roster_snapshot_available else 0.0,
            "roster_standin_risk_advantage": (
                float(bool(roster_control2.get("standin_risk"))) - float(bool(roster_control1.get("standin_risk")))
                if roster_snapshot_available else 0.0
            ),
            "fatigue_last24_team1": fatigue1.get("last24_count", 0),
            "fatigue_last24_team2": fatigue2.get("last24_count", 0),
            "fatigue_last24_diff_team1": fatigue1.get("last24_count", 0) - fatigue2.get("last24_count", 0),
            "fatigue_last48_team1": fatigue1.get("last48_count", 0),
            "fatigue_last48_team2": fatigue2.get("last48_count", 0),
        }
        analytics_match = {
            "team1": team1.get("name") or t1,
            "team2": team2.get("name") or t2,
            "team1_key": t1,
            "team2_key": t2,
            "format": snapshot.get("format") or "bo3",
            "analytics": snapshot.get("analytics"),
        }
        announced_lineups = snapshot.get("prematch_lineups") or {}
        event_metadata = snapshot.get("event_metadata") or ((snapshot.get("analytics") or {}).get("event_metadata")) or {}
        features.update(cs2_analytics_match_features(analytics_match))
        features.update(cs2_announced_lineup_features({"prematch_lineups": announced_lineups}))
        features.update(cs2_event_metadata_features({"event_metadata": event_metadata}))
        volatility = event_volatility(state["event_outcomes"], event)
        features["event_volatility"] = volatility["volatility"]
        features["event_volatility_label"] = volatility["label"]
        tournament = tournament_context(event, snapshot.get("link"), snapshot.get("format"), volatility["label"], match_context)
        fatigue1["online_lan"] = tournament.get("environment") or "unknown"
        fatigue2["online_lan"] = tournament.get("environment") or "unknown"
        features.update(
            {
                "context_is_lan": 1.0 if tournament.get("is_lan") is True else 0.0,
                "context_is_online": 1.0 if tournament.get("environment") == "online" else 0.0,
                "context_environment_known": 1.0 if tournament.get("environment") in {"lan", "online"} else 0.0,
                "context_high_stakes": 1.0 if tournament.get("high_stakes") else 0.0,
                "context_winner_advances": 1.0 if tournament.get("winner_advances") else 0.0,
                "context_loser_eliminated": 1.0 if tournament.get("loser_eliminated") else 0.0,
                "context_substitution_note": 1.0 if tournament.get("has_substitution_note") else 0.0,
            }
        )
        features.update(cs2_context_match_features({"match_context": tournament}))
        if _CS2_OK:
            external = cs2_dataio.external_snapshot_features_asof(
                external_feature_store,
                team1.get("id"),
                team2.get("id"),
                match_dt,
            )
            ranking_features = external.get("ranking_snapshot_features") or {}
            roster_features = external.get("roster_snapshot_features") or {}
            features.update(ranking_features)
            if roster_features.get("roster_available"):
                features.update(roster_features)

        model_epistemic_std = None
        series_score_distribution = None
        if engine is not None:
            try:
                model_p, model_epistemic_std = model_probability_and_uncertainty(
                    engine,
                    t1,
                    t2,
                    match_dt,
                    event,
                    snapshot.get("format") or "bo3",
                    analytics_match,
                    tournament,
                    features,
                    announced_lineups=announced_lineups,
                    event_metadata=event_metadata,
                )
                series_score_distribution = model_series_score_distribution(
                    engine,
                    t1,
                    t2,
                    match_dt,
                    event,
                    snapshot.get("format") or "bo3",
                    analytics_match,
                    tournament,
                    features,
                    announced_lineups=announced_lineups,
                    event_metadata=event_metadata,
                )
                card1 = engine["state"].rating_card(t1, match_dt)
                card2 = engine["state"].rating_card(t2, match_dt)
                features["glicko_rating_team1"] = card1["glicko_rating"]
                features["glicko_rating_team2"] = card2["glicko_rating"]
                features["glicko_rd_team1"] = card1["glicko_rd"]
                features["glicko_rd_team2"] = card2["glicko_rd"]
                features["glicko_rating_diff"] = card1["glicko_rating"] - card2["glicko_rating"]
                features["model_source"] = engine["artifact"].metadata.get("production_model", "model")
            except Exception:
                model_p = logistic_probability(features)
                features["model_source"] = "logistic_fallback"
        else:
            model_p = logistic_probability(features)
            features["model_source"] = "logistic_fallback"
        odds = snapshot.get("odds") or {}
        market = market_metrics(odds, odds_history_points(snapshot["id"], master, snapshot), model_p)
        odds_p = (odds.get("average") or {}).get("team1_implied_prob_norm")
        reliability_entry = {
            "format": snapshot.get("format"),
            "data_quality": snapshot.get("data_quality"),
            "team1": team1,
            "team2": team2,
        }
        reliability = reliability_score(reliability_entry, features)
        decision = decision_probability_team1(model_p, odds_p, reliability, market=market, policy=market_blend_policy)
        decision_p = decision["prob_team1"]
        decision_side = "team1" if decision_p >= 0.5 else "team2"
        decision_favorite_key = t1 if decision_side == "team1" else t2
        favorite_upset = favorite_upset_summary(favorite_upset_records, decision_favorite_key, match_dt)
        market_weight = decision["market_weight"]
        blended = clamp((1.0 - market_weight) * model_p + market_weight * odds_p, 1e-4, 1 - 1e-4) if odds_p is not None else model_p

        # Primary fields keep the calibrated stats model (Model A). Decision
        # fields are an operational layer for ranking/staking: shrink weak-data
        # confidence and blend normalized market odds when they exist.
        similar = similar_probability_bucket(historical_p, model_p)
        favorite_side = "team1" if model_p >= 0.5 else "team2"
        prediction = {
            "model_prob_team1": model_p,
            "model_epistemic_std": round(model_epistemic_std, 5) if model_epistemic_std is not None else None,
            "series_score_distribution": series_score_distribution,
            "predicted_series_score": (
                max(series_score_distribution, key=series_score_distribution.get)
                if series_score_distribution else None
            ),
            "bo3_compositional_prob_team1": (
                round(0.5 + float(features.get("bo3_compositional_prob_centered", 0.0)), 5)
                if (
                    features.get("bo3_compositional_available", 0.0) >= 0.5
                    and ((model_metadata.get("feature_policies") or {}).get("bo3_map_compositional") or {}).get("enabled")
                ) else None
            ),
            "odds_prob_team1": odds_p,
            "blended_prob_team1": blended,
            "risk_adjusted_prob_team1": decision["risk_adjusted_prob_team1"],
            "decision_prob_team1": decision_p,
            "decision_probability_source": decision["source"],
            "decision_market_weight": market_weight,
            "decision_market_weight_reasons": decision.get("market_weight_reasons"),
            "decision_team1_win_percent": round(decision_p * 100, 2),
            "decision_team2_win_percent": round((1 - decision_p) * 100, 2),
            "decision_favorite": team1.get("name") if decision_p >= 0.5 else team2.get("name"),
            "decision_favorite_side": decision_side,
            "decision_confidence": max(decision_p, 1 - decision_p),
            "favorite_upset_90d": favorite_upset,
            "reliability_score": reliability,
            "team1_win_percent": round(model_p * 100, 2),
            "team2_win_percent": round((1 - model_p) * 100, 2),
            "favorite": team1.get("name") if model_p >= 0.5 else team2.get("name"),
            "favorite_side": favorite_side,
            "confidence": max(model_p, 1 - model_p),
            "blended_favorite": team1.get("name") if blended >= 0.5 else team2.get("name"),
            "blended_confidence": max(blended, 1 - blended),
            "similar_probability_bucket": similar,
            "value_label": market.get("value_label"),
            "model_edge_team1_vs_market": market.get("model_edge_team1"),
            "decision_edge_team1_vs_market": (decision_p - odds_p) if odds_p is not None else None,
        }
        controls = {
            "market": market,
            "map_pool": map_pool,
            "map_veto": map_pool,
            "roster_stability": {"team1": roster_control1, "team2": roster_control2},
            "roster_change_90d": {
                "team1": roster_change1,
                "team2": roster_change2,
            },
            "player_form": {"team1": player_form1, "team2": player_form2},
            "tournament_context": tournament,
            "fatigue": {"team1": fatigue1, "team2": fatigue2},
            "favorite_upset": favorite_upset,
            "decision_policy": market_blend_policy,
            "calibration": {
                "all": calibration.get("all"),
                "last_50": calibration.get("last_50"),
                "last_200": calibration.get("last_200"),
            },
        }

        entry = {
            "id": snapshot["id"],
            "date": snapshot.get("date"),
            "hour": snapshot.get("hour"),
            "captured_at": snapshot.get("captured_at"),
            "data_quality": snapshot.get("data_quality") or {
                "real_pre_match_snapshot": True,
                "legacy_backfill": False,
                "training_weight_hint": "high_after_result",
            },
            "event": event,
            "format": snapshot.get("format"),
            "link": snapshot.get("link"),
            "team1": team1,
            "team2": team2,
            "odds": odds,
            "features": features,
            "rosters": {"team1": roster1, "team2": roster2},
            "prediction": prediction,
            "controls": controls,
        }
        staking = staking_recommendation(entry, features, prediction, market, model_metadata)
        entry["staking"] = staking
        entry["controls"]["staking"] = staking
        prediction["recommended_stake_pct_bankroll"] = staking.get("recommended_pct_bankroll")
        prediction["recommended_stake_team"] = staking.get("team")
        entry["flags"] = flag_match(entry, features, prediction)
        enriched.append(entry)
    annotate_opportunities(enriched)
    write_json(run_dir / "predictions_enriched.json", enriched)
    if skipped_unpublishable:
        write_json(
            run_dir / "predictions_skipped_unpublishable.json",
            {
                "run_started_local": run_started_local.strftime("%Y-%m-%dT%H:%M:%S"),
                "publish_reference_local": publish_reference_local.strftime("%Y-%m-%dT%H:%M:%S"),
                "skipped": dict(skipped_unpublishable),
            },
        )
    compute_daily_calibration(master, run_dir)
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich latest daily snapshot with predictions and reliability flags.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    enriched = enrich(run_dir, Path(args.history))
    print(json.dumps({"run_dir": str(run_dir), "matches": len(enriched)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

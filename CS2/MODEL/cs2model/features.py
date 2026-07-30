"""Estado cronológico + construcción de features point-in-time.

Pieza central anti-fuga temporal (PROJECT.md §9). Un único `ChronologicalState`
se usa en DOS modos sin duplicar lógica:

  * Backtest: se itera el histórico en orden; para cada partido se EMITEN las
    features (con el estado anterior) y LUEGO se OBSERVA el resultado.
  * Vivo: se reconstruye el estado con todo el histórico y se emiten features
    para los partidos futuros, sin observarlos.

Como las features de un partido se calculan siempre antes de observar su
resultado, el "cero fuga" queda garantizado por el flujo, no por vigilancia
manual. Las features de contexto pre-game de HLTV (LAN/online, fase, avance,
eliminacion) se calculan siempre, pero el entrenamiento solo las activa cuando
hay cobertura cerrada suficiente.

Ratings:
  - Glicko-2 por periodos de rating semanales (rating + RD + sigma).
  - Elo plano como baseline/feature secundaria.

Importante sobre point-in-time de los players: el histórico de HLTV solo trae
resultados de serie, no stats de jugador fechadas. Por eso el MODELO se entrena
solo con features derivadas de resultados (Glicko/Elo/forma/h2h/actividad), que
son reconstruibles idénticamente en backtest y en vivo. Las stats de roster son
contexto que la web muestra, no entradas del modelo (evita leakage; PROJECT.md
§4.6, §9).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from .glicko2 import Glicko2, Rating, _Match
from .trueskill import TeamTrueSkill, TSRating
from .bayesian_bt import BayesianBradleyTerry, BTRating
from .kalman_rating import KalmanRating, KalmanTeamRating
from .compositional_bo3 import compositional_bo3_features
from .config import get_runtime_config

PERIOD_DAYS = 7           # periodo de rating semanal (alineado con ranking HLTV)
FORM_HALF_LIFE = 120.0    # días: vida media del decaimiento de la forma


# --- columnas de features --------------------------------------------------
# DIFF: antisimétricas (cambian de signo al intercambiar A<->B).
DIFF_COLUMNS = [
    "glicko_diff",
    "glicko_prob_centered",
    "glicko_rd_diff",
    "elo_diff",
    "elo_prob_centered",
    "matches_log_diff",
    "winrate_diff",
    "winrate_decay_diff",
    "last5_winrate_diff",
    "last10_winrate_diff",
    "last20_winrate_diff",
    "last30_winrate_diff",
    "winrate_trend_5v20_diff",
    "avg_score_diff",
    "last5_score_diff",
    "last10_score_diff",
    "last20_score_diff",
    "recency_advantage",
    "activity_2_diff",
    "activity_7_diff",
    "activity_14_diff",
    "activity_30_diff",
    "streak_diff",
    "opp_elo_last5_diff",
    "recent_opponent_elo_diff",
    "opp_elo_last20_diff",
    "format_matches_log_diff",
    "format_winrate_diff",
    "format_last5_winrate_diff",
    "format_last10_winrate_diff",
    "format_winrate_trend_5v20_diff",
    "format_avg_score_diff",
    "format_last5_score_diff",
    "format_last10_score_diff",
    "format_opp_elo_last5_diff",
    "format_recent_opponent_elo_diff",
    "h2h_winrate_centered",
    "format_h2h_winrate_centered",
]
STRENGTH_INTERACTION_DIFF_COLUMNS = [
    "strength_consensus_prob_centered",
    "strength_consensus_x_agreement",
]
DIFF_COLUMNS += STRENGTH_INTERACTION_DIFF_COLUMNS
# SYM: simétricas (iguales al intercambiar A<->B).
SYM_COLUMNS = [
    "glicko_rd_sum",
    "experience_min",
    "experience_total",
    "inactivity_min_days",
    "inactivity_max_days",
    "activity_2_total",
    "activity_7_total",
    "h2h_matches",
    "format_experience_min",
    "format_experience_total",
    "format_h2h_matches",
    "format_bo1",
    "format_bo3",
    "format_bo5",
]
STRENGTH_INTERACTION_SYM_COLUMNS = [
    "strength_margin_abs",
    "strength_signal_disagreement",
    "strength_signal_agreement",
]
SYM_COLUMNS += STRENGTH_INTERACTION_SYM_COLUMNS
STRENGTH_INTERACTION_COLUMNS = (
    STRENGTH_INTERACTION_DIFF_COLUMNS + STRENGTH_INTERACTION_SYM_COLUMNS
)
BASE_FEATURE_COLUMNS = [
    column
    for column in DIFF_COLUMNS + SYM_COLUMNS
    if column not in STRENGTH_INTERACTION_COLUMNS
]
FEATURE_COLUMNS = DIFF_COLUMNS + SYM_COLUMNS

# --- Familias de rating adicionales, AUTO-GATED por muestra en train.py -------
# Se calculan SIEMPRE (baratas, point-in-time) y solo entran al modelo cuando su
# columna *_available supera el umbral (AUTO_FEATURE_FAMILIES en train.py). Sin
# flags manuales: al acumularse muestra suficiente se activan solas.

# TrueSkill de equipo (online, causal). Herbrich et al. 2006.
TRUESKILL_DIFF_COLUMNS = ["trueskill_diff", "trueskill_prob_centered"]
TRUESKILL_SYM_COLUMNS = ["trueskill_available"]
TRUESKILL_FEATURE_COLUMNS = TRUESKILL_DIFF_COLUMNS + TRUESKILL_SYM_COLUMNS

# Rating consciente del MARGEN (Elo-MOV): escala la actualizacion por el margen
# de mapas (2-0 pesa mas que 2-1). Senal ortogonal al win-only (MOVDA 2025).
MOV_DIFF_COLUMNS = ["mov_diff", "mov_prob_centered"]
MOV_SYM_COLUMNS = ["mov_available"]
MOV_FEATURE_COLUMNS = MOV_DIFF_COLUMNS + MOV_SYM_COLUMNS

# Rating por JUGADOR (TrueSkill por jugador desde box score) agregado al equipo
# por su ultimo quinteto observado. La habilidad "viaja" con el jugador entre
# equipos (Skill Issues 2024; GRID: +3.6pp / +14.1pp en lineups nuevos).
PLAYER_RATING_DIFF_COLUMNS = [
    "player_skill_mean_diff", "player_skill_max_diff",
    "player_skill_min_diff", "player_skill_spread_diff",
]
PLAYER_RATING_SYM_COLUMNS = ["player_skill_available"]
PLAYER_RATING_FEATURE_COLUMNS = PLAYER_RATING_DIFF_COLUMNS + PLAYER_RATING_SYM_COLUMNS

# Strength of schedule explicito. No se limita a la media de Elo rival: mide
# tambien cuanto rindio cada equipo por encima/debajo de lo esperado contra
# esos rivales, conservando el Elo que tenian justo antes de cada encuentro.
SOS_DIFF_COLUMNS = [
    "sos_opp_elo_decay_diff",
    "sos_performance_residual_l10_diff",
    "sos_performance_residual_decay_diff",
]
SOS_SYM_COLUMNS = [
    "sos_available",
    "sos_matches_min",
    "sos_opponent_elo_std_sum",
]
SOS_FEATURE_COLUMNS = SOS_DIFF_COLUMNS + SOS_SYM_COLUMNS

BAYES_BT_DIFF_COLUMNS = ["bayesian_bt_mean_diff", "bayesian_bt_prob_centered"]
BAYES_BT_SYM_COLUMNS = ["bayesian_bt_available", "bayesian_bt_games_min", "bayesian_bt_uncertainty_sum"]
BAYES_BT_FEATURE_COLUMNS = BAYES_BT_DIFF_COLUMNS + BAYES_BT_SYM_COLUMNS

KALMAN_DIFF_COLUMNS = ["kalman_mean_diff", "kalman_prob_centered"]
KALMAN_SYM_COLUMNS = ["kalman_available", "kalman_games_min", "kalman_uncertainty_sum"]
KALMAN_FEATURE_COLUMNS = KALMAN_DIFF_COLUMNS + KALMAN_SYM_COLUMNS

BO3_COMPOSITIONAL_DIFF_COLUMNS = ["bo3_compositional_prob_centered"]
BO3_COMPOSITIONAL_SYM_COLUMNS = [
    "bo3_compositional_available",
    "bo3_map_probability_spread",
    "bo3_map_sample_min",
    "bo3_veto_confidence",
]
BO3_COMPOSITIONAL_FEATURE_COLUMNS = BO3_COMPOSITIONAL_DIFF_COLUMNS + BO3_COMPOSITIONAL_SYM_COLUMNS

# Meta-regime variables are symmetric calibration inputs. Map-pool state is
# reconstructed only from previously observed mapstats; patch metadata must be
# explicitly present in a pre-match source and is never guessed from results.
REGIME_FEATURE_COLUMNS = [
    "regime_available",
    "regime_patch_known",
    "regime_patch_age_log_days",
    "regime_patch_recent",
    "regime_map_pool_available",
    "regime_map_pool_size",
    "regime_map_pool_turnover",
]

MAP_ASSET_DIFF_COLUMNS = [
    "asset_map_winrate_diff",
    "asset_ct_round_winrate_diff",
    "asset_t_round_winrate_diff",
    "asset_rating_l5_diff",
    "asset_rating_l10_diff",
    "asset_rating_l20_diff",
    "asset_adr_l10_diff",
    "asset_kast_l10_diff",
    "asset_opening_diff_l10_diff",
]
MAP_ASSET_SYM_COLUMNS = [
    "asset_available",
    "asset_maps_played_min",
    "asset_maps_played_total",
]
MAP_ASSET_FEATURE_COLUMNS = MAP_ASSET_DIFF_COLUMNS + MAP_ASSET_SYM_COLUMNS

EVENT_HISTORY_DIFF_COLUMNS = [
    "event_matches_diff",
    "event_winrate_diff",
]
EVENT_HISTORY_SYM_COLUMNS = [
    "event_history_available",
    "event_experience_min",
    "event_experience_total",
]
EVENT_HISTORY_FEATURE_COLUMNS = EVENT_HISTORY_DIFF_COLUMNS + EVENT_HISTORY_SYM_COLUMNS

ANALYTICS_DIFF_COLUMNS = [
    "analytics_map_win_pct_diff",
    "analytics_first_pick_pct_diff",
    "analytics_first_ban_pct_diff",
    "analytics_map_played_diff",
    "analytics_insight_score_diff",
]
ANALYTICS_SYM_COLUMNS = [
    "analytics_available",
    "analytics_common_maps",
    "analytics_map_rows_total",
    "analytics_insight_total",
]
ANALYTICS_FEATURE_COLUMNS = ANALYTICS_DIFF_COLUMNS + ANALYTICS_SYM_COLUMNS

# Señales adicionales del Analytics Center. Se almacenan desde ahora, pero su
# familia tiene un umbral de cobertura propio en train.py para que una novedad
# de scraping no cambie el modelo con unas pocas decenas de casos.
ANALYTICS_EXTENDED_DIFF_COLUMNS = [
    "analytics_standin_advantage",
    "analytics_core_lineup_risk_advantage",
    "analytics_series_win_pct_diff",
    "analytics_map_handicap_margin_diff",
]
ANALYTICS_EXTENDED_SYM_COLUMNS = [
    "analytics_extended_available",
    "analytics_series_sample_min",
    "analytics_overtime_pct_abs_diff",
    "analytics_map_handicap_common_maps",
]
ANALYTICS_EXTENDED_FEATURE_COLUMNS = ANALYTICS_EXTENDED_DIFF_COLUMNS + ANALYTICS_EXTENDED_SYM_COLUMNS

# La alineacion anunciada en la pagina del partido es una fuente distinta del
# roster historico: identifica los cinco que HLTV esperaba antes del inicio.
# Se mantiene como familia propia para que su cobertura y su activacion se
# puedan auditar sin mezclarla con snapshots generales de perfil.
ANNOUNCED_LINEUP_DIFF_COLUMNS = [
    "announced_lineup_rating_diff",
    "announced_lineup_rating_top2_avg_diff",
    "announced_lineup_rating_bottom2_avg_diff",
    "announced_lineup_kpr_diff",
    "announced_lineup_kast_diff",
    "announced_lineup_adr_diff",
    "announced_lineup_multi_kill_rating_diff",
    "announced_lineup_round_swing_diff",
    "announced_lineup_standin_advantage",
]
ANNOUNCED_LINEUP_SYM_COLUMNS = [
    "announced_lineup_available",
    "announced_lineup_players_min",
    "announced_lineup_rating_coverage_min",
]
ANNOUNCED_LINEUP_FEATURE_COLUMNS = ANNOUNCED_LINEUP_DIFF_COLUMNS + ANNOUNCED_LINEUP_SYM_COLUMNS

# Metadatos comunes del evento. No inclinan por si solos A frente a B, pero un
# modelo no lineal puede usarlos para calibrar interacciones con contexto y
# fortaleza. Se activan con un umbral mas alto en train.py.
EVENT_METADATA_FEATURE_COLUMNS = [
    "event_metadata_available",
    "event_prize_pool_log",
    "event_teams_competing",
]

CONTEXT_STAGE_COLUMNS = [
    "context_stage_group",
    "context_stage_swiss",
    "context_stage_ro16",
    "context_stage_quarter",
    "context_stage_semi",
    "context_stage_final",
    "context_stage_playoff",
    "context_stage_qualifier",
    "context_stage_bracket",
    "context_stage_other",
]
CONTEXT_FEATURE_COLUMNS = [
    "context_available",
    "context_environment_known",
    "context_is_lan",
    "context_is_online",
    "context_stage_known",
    "context_high_stakes",
    "context_winner_advances",
    "context_loser_eliminated",
    "context_opening_match",
    "context_winners_match",
    "context_placement_match",
    "context_substitution_note",
    "context_bracket_upper",
    "context_bracket_lower",
] + CONTEXT_STAGE_COLUMNS

PLAYER_DIFF_COLUMNS = [
    "player_rating_diff",
    "player_rating_max_diff",
    "player_rating_top2_avg_diff",
    "player_rating_median_diff",
    "player_rating_bottom2_avg_diff",
    "player_rating_std_diff",
    "player_kpr_diff",
    "player_kast_diff",
    "player_adr_diff",
    "player_impact_diff",
    "player_round_swing_diff",
    "player_round_swing_top2_avg_diff",
    "player_opening_kpr_diff",
    "player_opening_kpr_top2_avg_diff",
]
PLAYER_SYM_COLUMNS = [
    "player_snapshot_available",
    "player_coverage_min",
    "player_maps_min",
    "player_maps_per_player_min",
]
PLAYER_FEATURE_COLUMNS = PLAYER_DIFF_COLUMNS + PLAYER_SYM_COLUMNS

RANKING_DIFF_COLUMNS = [
    "ranking_hltv_position_advantage",
    "ranking_hltv_points_diff",
    "ranking_valve_position_advantage",
    "ranking_valve_points_diff",
]
RANKING_SYM_COLUMNS = [
    "ranking_available",
    "ranking_sources",
    "ranking_age_days_max",
]
RANKING_FEATURE_COLUMNS = RANKING_DIFF_COLUMNS + RANKING_SYM_COLUMNS

ROSTER_DIFF_COLUMNS = [
    "roster_days_log_diff",
    "roster_size_diff",
    "roster_standin_risk_advantage",
]
ROSTER_SYM_COLUMNS = [
    "roster_available",
    "roster_days_min",
    "roster_size_min",
]
ROSTER_FEATURE_COLUMNS = ROSTER_DIFF_COLUMNS + ROSTER_SYM_COLUMNS

EXTENDED_DIFF_COLUMNS = (
    MAP_ASSET_DIFF_COLUMNS
    + EVENT_HISTORY_DIFF_COLUMNS
    + ANALYTICS_DIFF_COLUMNS
    + ANALYTICS_EXTENDED_DIFF_COLUMNS
    + ANNOUNCED_LINEUP_DIFF_COLUMNS
    + PLAYER_DIFF_COLUMNS
    + RANKING_DIFF_COLUMNS
    + ROSTER_DIFF_COLUMNS
    + TRUESKILL_DIFF_COLUMNS
    + MOV_DIFF_COLUMNS
    + PLAYER_RATING_DIFF_COLUMNS
    + SOS_DIFF_COLUMNS
    + BAYES_BT_DIFF_COLUMNS
    + KALMAN_DIFF_COLUMNS
    + BO3_COMPOSITIONAL_DIFF_COLUMNS
)


def _clean_team(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _context_bool(context: dict[str, Any], key: str) -> bool:
    value = context.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}
    return False


def _context_stage_flags(context: dict[str, Any]) -> dict[str, float]:
    flags = {col: 0.0 for col in CONTEXT_STAGE_COLUMNS}
    stage = str(context.get("stage") or "").strip().lower().replace("-", "_").replace(" ", "_")
    detail = " ".join(
        str(context.get(key) or "").lower()
        for key in ("stage_detail", "incentive_label", "raw_meta")
    )
    text = f"{stage} {detail}"

    if stage in {"group", "swiss", "ro16", "quarter", "semi", "final", "other"}:
        flags[f"context_stage_{stage}"] = 1.0
    if "group" in text:
        flags["context_stage_group"] = 1.0
    if "swiss" in text:
        flags["context_stage_swiss"] = 1.0
    if "round of 16" in text or "ro16" in text:
        flags["context_stage_ro16"] = 1.0
    if "quarter" in text:
        flags["context_stage_quarter"] = 1.0
    if "semi" in text:
        flags["context_stage_semi"] = 1.0
    if "final" in text:
        flags["context_stage_final"] = 1.0
    if "playoff" in text or "play-off" in text:
        flags["context_stage_playoff"] = 1.0
    if "qualifier" in text or "qualification" in text:
        flags["context_stage_qualifier"] = 1.0
    if "bracket" in text or context.get("bracket"):
        flags["context_stage_bracket"] = 1.0
    if stage and not any(flags.values()):
        flags["context_stage_other"] = 1.0
    return flags


def context_match_features(match: dict[str, Any]) -> dict[str, float]:
    """Same-match tournament context from HLTV Maps box.

    These are pre-game, symmetric features: LAN/online and tournament phase can
    affect upset rate, but training only enables them once coverage is large
    enough. Keeping them here makes live inference ready for that moment.
    """
    feats = {col: 0.0 for col in CONTEXT_FEATURE_COLUMNS}
    context = match.get("match_context") or match.get("tournament_context") or {}
    if not isinstance(context, dict) or not context:
        return feats

    env = str(context.get("environment") or "").strip().lower()
    if env not in {"lan", "online"}:
        if context.get("is_lan") is True:
            env = "lan"
        elif context.get("is_lan") is False:
            env = "online"
    feats["context_environment_known"] = 1.0 if env in {"lan", "online"} else 0.0
    feats["context_is_lan"] = 1.0 if env == "lan" else 0.0
    feats["context_is_online"] = 1.0 if env == "online" else 0.0

    stage_flags = _context_stage_flags(context)
    feats.update(stage_flags)
    feats["context_stage_known"] = 1.0 if any(stage_flags.values()) else 0.0
    feats["context_high_stakes"] = 1.0 if _context_bool(context, "high_stakes") else 0.0
    feats["context_winner_advances"] = 1.0 if _context_bool(context, "winner_advances") else 0.0
    feats["context_loser_eliminated"] = 1.0 if _context_bool(context, "loser_eliminated") else 0.0
    feats["context_opening_match"] = 1.0 if _context_bool(context, "opening_match") else 0.0
    feats["context_winners_match"] = 1.0 if _context_bool(context, "winners_match") else 0.0
    feats["context_placement_match"] = 1.0 if _context_bool(context, "placement_match") else 0.0
    feats["context_substitution_note"] = 1.0 if _context_bool(context, "has_substitution_note") else 0.0

    bracket = str(context.get("bracket") or "").strip().lower()
    feats["context_bracket_upper"] = 1.0 if bracket == "upper" else 0.0
    feats["context_bracket_lower"] = 1.0 if bracket == "lower" else 0.0
    signal_columns = [
        col for col in CONTEXT_FEATURE_COLUMNS
        if col not in {
            "context_available",
            "context_is_lan",
            "context_is_online",
            "context_environment_known",
            "context_stage_known",
        }
    ]
    meaningful = (
        feats["context_environment_known"] >= 0.5
        or feats["context_stage_known"] >= 0.5
        or any(feats[col] >= 0.5 for col in signal_columns)
    )
    feats["context_available"] = 1.0 if meaningful else 0.0
    return feats


def analytics_match_features(match: dict[str, Any]) -> dict[str, float]:
    """Point-in-time features from HLTV Betting Analytics for this match.

    These are same-match pre-game features, not rolling state. They are only
    used by training when coverage is high enough; otherwise they stay inert.
    """
    feats = {
        col: 0.0
        for col in (
            ANALYTICS_FEATURE_COLUMNS
            + ANALYTICS_EXTENDED_FEATURE_COLUMNS
            + BO3_COMPOSITIONAL_FEATURE_COLUMNS
        )
    }
    analytics = match.get("analytics") or {}
    if not isinstance(analytics, dict) or not analytics.get("available"):
        return feats

    t1 = _clean_team(match.get("team1") or match.get("team1_name"))
    t2 = _clean_team(match.get("team2") or match.get("team2_name"))
    if not t1 or not t2:
        t1 = _clean_team(match.get("team1_key"))
        t2 = _clean_team(match.get("team2_key"))

    feats.update(
        compositional_bo3_features(
            analytics,
            match.get("team1") or match.get("team1_name") or match.get("team1_key") or "",
            match.get("team2") or match.get("team2_name") or match.get("team2_key") or "",
            match.get("format") or "bo3",
        )
    )

    feats["analytics_available"] = 1.0
    rows_by_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in analytics.get("map_stats") or []:
        if not isinstance(row, dict):
            continue
        map_name = str(row.get("map") or "").strip().lower()
        team = _clean_team(row.get("team"))
        if not map_name or not team:
            continue
        rows_by_map[map_name][team] = row
        feats["analytics_map_rows_total"] += 1.0

    win_diffs: list[float] = []
    pick_diffs: list[float] = []
    ban_diffs: list[float] = []
    played_diffs: list[float] = []
    common_maps = 0
    for _map_name, teams in rows_by_map.items():
        r1 = teams.get(t1)
        r2 = teams.get(t2)
        if not r1 or not r2:
            continue
        common_maps += 1
        w1 = _safe_float(r1.get("win_pct"))
        w2 = _safe_float(r2.get("win_pct"))
        if w1 is not None and w2 is not None:
            win_diffs.append((w1 - w2) / 100.0)
        p1 = _safe_float(r1.get("first_pick_pct"))
        p2 = _safe_float(r2.get("first_pick_pct"))
        if p1 is not None and p2 is not None:
            pick_diffs.append((p1 - p2) / 100.0)
        b1 = _safe_float(r1.get("first_ban_pct"))
        b2 = _safe_float(r2.get("first_ban_pct"))
        if b1 is not None and b2 is not None:
            ban_diffs.append((b1 - b2) / 100.0)
        pl1 = _safe_float(r1.get("played"))
        pl2 = _safe_float(r2.get("played"))
        if pl1 is not None and pl2 is not None:
            played_diffs.append(pl1 - pl2)

    feats["analytics_common_maps"] = float(common_maps)
    feats["analytics_map_win_pct_diff"] = _avg(win_diffs)
    feats["analytics_first_pick_pct_diff"] = _avg(pick_diffs)
    feats["analytics_first_ban_pct_diff"] = _avg(ban_diffs)
    feats["analytics_map_played_diff"] = _avg(played_diffs)

    positive_terms = ("better ranked", "better form", "has won", "won ")
    negative_terms = ("worse ranked", "stand-in", "less than", "only played", "lost ")
    scores = {t1: 0.0, t2: 0.0}
    insight_total = 0
    for raw in analytics.get("insights") or []:
        line = str(raw or "")
        lower = line.lower()
        line_team = None
        if t1 and t1 in _clean_team(line):
            line_team = t1
        elif t2 and t2 in _clean_team(line):
            line_team = t2
        if not line_team:
            continue
        insight_total += 1
        if any(term in lower for term in positive_terms):
            scores[line_team] += 1.0
        if any(term in lower for term in negative_terms):
            scores[line_team] -= 1.0
    feats["analytics_insight_total"] = float(insight_total)
    feats["analytics_insight_score_diff"] = scores.get(t1, 0.0) - scores.get(t2, 0.0)

    def named_payload(payload: dict[str, Any], name: str) -> dict[str, Any]:
        for team, value in (payload or {}).items():
            if _clean_team(team) == name and isinstance(value, dict):
                return value
        return {}

    def named_list(payload: dict[str, Any], name: str) -> list[Any]:
        for team, value in (payload or {}).items():
            if _clean_team(team) == name and isinstance(value, list):
                return value
        return []

    def team_series(side: str, name: str) -> dict[str, Any]:
        value = (analytics.get("series_stats") or {}).get(side) or {}
        if isinstance(value, dict) and (_clean_team(value.get("team")) in {"", name}):
            return value
        for candidate in (analytics.get("series_stats") or {}).values():
            if isinstance(candidate, dict) and _clean_team(candidate.get("team")) == name:
                return candidate
        return {}

    standins = analytics.get("standins") or {}
    core_lineup = analytics.get("core_lineup") or {}
    standins1 = len(named_list(standins, t1))
    standins2 = len(named_list(standins, t2))
    core1 = named_payload(core_lineup, t1)
    core2 = named_payload(core_lineup, t2)
    # Positivo = ventaja para team1: menos sustituciones o menor alerta de core.
    feats["analytics_standin_advantage"] = float(standins2 - standins1)
    feats["analytics_core_lineup_risk_advantage"] = float(bool(core2)) - float(bool(core1))

    series1 = team_series("team1", t1)
    series2 = team_series("team2", t2)
    dist1 = series1.get("score_distribution") or {}
    dist2 = series2.get("score_distribution") or {}
    win1 = _safe_float(dist1.get("2_0_wins")) or 0.0
    win1 += _safe_float(dist1.get("2_1_wins")) or 0.0
    win2 = _safe_float(dist2.get("2_0_wins")) or 0.0
    win2 += _safe_float(dist2.get("2_1_wins")) or 0.0
    if series1 and series2:
        feats["analytics_series_win_pct_diff"] = (win1 - win2) / 100.0
        samples = [_safe_float(series1.get("matches")), _safe_float(series2.get("matches"))]
        if all(value is not None for value in samples):
            feats["analytics_series_sample_min"] = min(samples)
        overtime1 = _safe_float(series1.get("overtime_pct"))
        overtime2 = _safe_float(series2.get("overtime_pct"))
        if overtime1 is not None and overtime2 is not None:
            feats["analytics_overtime_pct_abs_diff"] = abs(overtime1 - overtime2) / 100.0

    handicap_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in analytics.get("map_handicap") or []:
        if not isinstance(row, dict):
            continue
        team = _clean_team(row.get("team"))
        map_name = str(row.get("map") or "").strip().lower()
        if team and map_name and map_name != "overall":
            handicap_rows[map_name][team] = row
    margins: list[float] = []
    for teams in handicap_rows.values():
        row1, row2 = teams.get(t1), teams.get(t2)
        if not row1 or not row2:
            continue
        lost1, won1 = _safe_float(row1.get("avg_rounds_lost_in_wins")), _safe_float(row1.get("avg_rounds_won_in_losses"))
        lost2, won2 = _safe_float(row2.get("avg_rounds_lost_in_wins")), _safe_float(row2.get("avg_rounds_won_in_losses"))
        if None not in {lost1, won1, lost2, won2}:
            margins.append((won1 - lost1) - (won2 - lost2))
    feats["analytics_map_handicap_common_maps"] = float(len(margins))
    feats["analytics_map_handicap_margin_diff"] = _avg(margins)
    feats["analytics_extended_available"] = 1.0 if (
        standins1 or standins2 or core1 or core2 or (series1 and series2) or margins
    ) else 0.0
    return feats


def _lineup_metric(players: list[dict[str, Any]], metric: str, aggregate: str = "avg") -> float:
    values = [_safe_float(player.get(metric)) for player in players]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    ordered = sorted(values)
    if aggregate == "top2_avg":
        selected = ordered[-min(2, len(ordered)) :]
        return _avg(selected)
    if aggregate == "bottom2_avg":
        selected = ordered[: min(2, len(ordered))]
        return _avg(selected)
    return _avg(values)


def announced_lineup_features(match: dict[str, Any]) -> dict[str, float]:
    """Five-versus-five features from the announced pre-match lineup.

    The caller must pass the snapshot that was captured before kick-off. This
    function deliberately does not fall back to a current roster: a missing
    announced lineup is safer than silently substituting post-match knowledge.
    """
    feats = {col: 0.0 for col in ANNOUNCED_LINEUP_FEATURE_COLUMNS}
    lineups = match.get("prematch_lineups") or match.get("announced_lineups") or {}
    if not isinstance(lineups, dict):
        return feats
    team1 = lineups.get("team1") or {}
    team2 = lineups.get("team2") or {}
    players1 = team1.get("players") if isinstance(team1, dict) else None
    players2 = team2.get("players") if isinstance(team2, dict) else None
    if not isinstance(players1, list) or not isinstance(players2, list):
        return feats
    players1 = [player for player in players1 if isinstance(player, dict)]
    players2 = [player for player in players2 if isinstance(player, dict)]
    ratings1 = [_safe_float(player.get("rating")) for player in players1]
    ratings2 = [_safe_float(player.get("rating")) for player in players2]
    ratings1 = [value for value in ratings1 if value is not None]
    ratings2 = [value for value in ratings2 if value is not None]

    feats["announced_lineup_players_min"] = float(min(len(players1), len(players2)))
    feats["announced_lineup_rating_coverage_min"] = float(min(len(ratings1), len(ratings2)))
    # Five announced players on both sides and four ratings per team protect
    # the feature family from partial/placeholder lineups.
    if len(players1) < 5 or len(players2) < 5 or len(ratings1) < 4 or len(ratings2) < 4:
        return feats

    feats["announced_lineup_available"] = 1.0
    feats["announced_lineup_rating_diff"] = _avg(ratings1) - _avg(ratings2)
    feats["announced_lineup_rating_top2_avg_diff"] = (
        _lineup_metric(players1, "rating", "top2_avg") - _lineup_metric(players2, "rating", "top2_avg")
    )
    feats["announced_lineup_rating_bottom2_avg_diff"] = (
        _lineup_metric(players1, "rating", "bottom2_avg") - _lineup_metric(players2, "rating", "bottom2_avg")
    )
    for source, target in (
        ("kpr", "announced_lineup_kpr_diff"),
        ("kast", "announced_lineup_kast_diff"),
        ("adr", "announced_lineup_adr_diff"),
        ("multi_kill_rating", "announced_lineup_multi_kill_rating_diff"),
        ("round_swing", "announced_lineup_round_swing_diff"),
    ):
        feats[target] = _lineup_metric(players1, source) - _lineup_metric(players2, source)
    standins1 = sum(1 for player in players1 if _context_bool(player, "is_standin"))
    standins2 = sum(1 for player in players2 if _context_bool(player, "is_standin"))
    # Positivo = ventaja para team1 porque tiene menos sustituciones anunciadas.
    feats["announced_lineup_standin_advantage"] = float(standins2 - standins1)
    return feats


def event_metadata_features(match: dict[str, Any]) -> dict[str, float]:
    """Point-in-time event size metadata captured with HLTV Analytics."""
    feats = {col: 0.0 for col in EVENT_METADATA_FEATURE_COLUMNS}
    metadata = match.get("event_metadata") or {}
    if not isinstance(metadata, dict):
        return feats
    prize_pool = _safe_float(metadata.get("prize_pool"))
    teams_competing = _safe_float(metadata.get("teams_competing"))
    if prize_pool is None and teams_competing is None:
        return feats
    feats["event_metadata_available"] = 1.0
    if prize_pool is not None and prize_pool >= 0:
        feats["event_prize_pool_log"] = math.log1p(prize_pool)
    if teams_competing is not None and teams_competing >= 0:
        feats["event_teams_competing"] = teams_competing
    return feats


def player_snapshot_features(match: dict[str, Any]) -> dict[str, float]:
    payload = match.get("player_snapshot_features")
    if isinstance(payload, dict):
        return {
            col: float(payload.get(col) or 0.0)
            for col in PLAYER_FEATURE_COLUMNS
        }
    return {col: 0.0 for col in PLAYER_FEATURE_COLUMNS}


def external_snapshot_features(match: dict[str, Any]) -> dict[str, float]:
    """Features externas que `dataio` unio respetando captured_at_utc."""
    out: dict[str, float] = {}
    for payload_key, columns in (
        ("ranking_snapshot_features", RANKING_FEATURE_COLUMNS),
        ("roster_snapshot_features", ROSTER_FEATURE_COLUMNS),
    ):
        payload = match.get(payload_key)
        for col in columns:
            try:
                out[col] = float((payload or {}).get(col) or 0.0)
            except (TypeError, ValueError):
                out[col] = 0.0
    return out


def _period_index(date_obj: datetime | None) -> int:
    if date_obj is None:
        return 0
    return date_obj.toordinal() // PERIOD_DAYS


def _laplace_winrate(wins: int, n: int) -> float:
    return (wins + 1.0) / (n + 2.0)


class ChronologicalState:
    """Acumula la verdad de los partidos en orden y emite features point-in-time."""

    def __init__(self, glicko_tau: float = 0.5, form_half_life: float = FORM_HALF_LIFE) -> None:
        self.glicko = Glicko2(tau=glicko_tau)
        # Vida media (dias) del decaimiento temporal de la forma. Tunable como
        # hiperparametro (PROJECT.md; Dixon-Coles time-weighting).
        self.form_half_life = float(form_half_life) if form_half_life and form_half_life > 0 else FORM_HALF_LIFE
        self.ratings: dict[str, Rating] = {}
        self.current_period: int | None = None
        self.period_buffer: dict[str, list[_Match]] = defaultdict(list)

        # Elo baseline
        self.elos: dict[str, float] = defaultdict(lambda: 1500.0)

        # TrueSkill de equipo (online, point-in-time). Se computa siempre; entra
        # al modelo por auto-gating de muestra (no por flag).
        self.trueskill = TeamTrueSkill()
        self.ts_ratings: dict[str, TSRating] = {}

        # B8: Bradley-Terry bayesiano con prior compartido (partial pooling).
        self.bayesian_bt = BayesianBradleyTerry()
        self.bt_ratings: dict[str, BTRating] = {}
        self.bt_last_date: dict[str, datetime] = {}

        # B9: alternativa en espacio de estados con deriva de habilidad.
        self.kalman = KalmanTeamRating()
        self.kalman_ratings: dict[str, KalmanRating] = {}
        self.kalman_last_date: dict[str, datetime] = {}

        # Rating consciente del margen (Elo-MOV) a nivel equipo.
        self.mov: dict[str, float] = defaultdict(lambda: 1500.0)

        # Rating por jugador (TrueSkill por jugador) + agregado del ultimo
        # quinteto observado por equipo (media/max/min/dispersion de mu).
        self.player_ts: dict[str, TSRating] = {}
        self.team_player_skill: dict[str, dict[str, float]] = {}

        # rolling de resultados (cada entrada lleva la fecha para decay)
        self.win_hist: dict[str, list[tuple[int, datetime | None]]] = defaultdict(list)
        self.diff_hist: dict[str, list[int]] = defaultdict(list)
        self.match_dates: dict[str, list[datetime]] = defaultdict(list)
        self.opp_elo_hist: dict[str, list[float]] = defaultdict(list)
        self.sos_hist: dict[str, list[tuple[float, float, datetime | None]]] = defaultdict(list)
        self.n_matches: dict[str, int] = defaultdict(int)
        self.streak: dict[str, int] = defaultdict(int)
        self.last_date: dict[str, datetime] = {}

        self.h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0})
        self.format_win_hist: dict[tuple[str, str], list[tuple[int, datetime | None]]] = defaultdict(list)
        self.format_diff_hist: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.format_opp_elo_hist: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.format_n_matches: dict[tuple[str, str], int] = defaultdict(int)
        self.format_h2h: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"n": 0})
        self.event_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0, "w": 0})
        self.asset_map_hist: dict[tuple[str, str], list[int]] = defaultdict(list)
        # Indice por equipo (todos los mapas concatenados) para evitar escanear
        # TODAS las claves (equipo,mapa) en cada llamada a _asset_map_winrate.
        self.asset_map_results_by_team: dict[str, list[int]] = defaultdict(list)
        self.asset_team_maps: dict[str, int] = defaultdict(int)
        self.asset_ct_rounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.asset_t_rounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.asset_rating_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_adr_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_kast_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_opening_diff_hist: dict[str, list[float]] = defaultdict(list)
        self.map_regime_observations: list[tuple[datetime, str]] = []

    def _map_pool_sets(self, as_of: datetime | None) -> tuple[set[str], set[str], int]:
        if as_of is None:
            return set(), set(), 0
        cfg = get_runtime_config().drift
        current_start = as_of - timedelta(days=cfg.regime_map_window_days)
        previous_start = current_start - timedelta(days=cfg.regime_previous_window_days)
        current: list[str] = []
        previous: list[str] = []
        for observed_at, map_name in self.map_regime_observations:
            if current_start <= observed_at < as_of:
                current.append(map_name)
            elif previous_start <= observed_at < current_start:
                previous.append(map_name)
        return set(current), set(previous), len(current)

    def map_pool_regime_label(self, as_of: datetime | None) -> str | None:
        current, _previous, observations = self._map_pool_sets(as_of)
        minimum = get_runtime_config().drift.regime_min_observed_maps
        if observations < minimum or not current:
            return None
        return "+".join(sorted(current))

    def regime_features(self, match: dict[str, Any]) -> dict[str, float]:
        feats = {column: 0.0 for column in REGIME_FEATURE_COLUMNS}
        as_of = match.get("date_obj")
        context = match.get("match_context") or {}
        patch_version = match.get("patch_version") or context.get("patch_version")
        patch_released_at = match.get("patch_released_at") or context.get("patch_released_at")
        feats["regime_patch_known"] = 1.0 if patch_version else 0.0
        if patch_version and patch_released_at and as_of is not None:
            if not isinstance(patch_released_at, datetime):
                try:
                    patch_released_at = datetime.fromisoformat(
                        str(patch_released_at).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError:
                    patch_released_at = None
            if patch_released_at is not None:
                age_days = max(0.0, (as_of - patch_released_at).total_seconds() / 86400.0)
                feats["regime_patch_age_log_days"] = math.log1p(age_days)
                feats["regime_patch_recent"] = float(age_days <= 14.0)

        current, previous, observations = self._map_pool_sets(as_of)
        minimum = get_runtime_config().drift.regime_min_observed_maps
        if observations >= minimum and current:
            feats["regime_map_pool_available"] = 1.0
            feats["regime_map_pool_size"] = float(len(current))
            union = current | previous
            feats["regime_map_pool_turnover"] = (
                1.0 - len(current & previous) / len(union) if previous and union else 0.0
            )
        feats["regime_available"] = float(
            feats["regime_patch_known"] >= 0.5 or feats["regime_map_pool_available"] >= 0.5
        )
        return feats

    # ---- gestión de ratings -------------------------------------------
    def _get_rating(self, key: str) -> Rating:
        r = self.ratings.get(key)
        if r is None:
            r = Rating()
            self.ratings[key] = r
        return r

    @staticmethod
    def _elapsed_days(last_date: datetime | None, as_of: datetime | None) -> float:
        if last_date is None or as_of is None:
            return 0.0
        return max(0.0, (as_of - last_date).total_seconds() / 86400.0)

    def _bt_asof(self, key: str, as_of: datetime | None) -> BTRating:
        rating = self.bt_ratings.get(key) or self.bayesian_bt.default()
        return self.bayesian_bt.evolve(
            rating, self._elapsed_days(self.bt_last_date.get(key), as_of)
        )

    def _kalman_asof(self, key: str, as_of: datetime | None) -> KalmanRating:
        rating = self.kalman_ratings.get(key) or self.kalman.default()
        return self.kalman.evolve(
            rating, self._elapsed_days(self.kalman_last_date.get(key), as_of)
        )

    def rating_asof(self, key: str, period: int) -> Rating:
        """Rating con la RD decaída hasta 'period' (sin mutar el estado)."""
        r = self.ratings.get(key)
        if r is None:
            return Rating()
        rr = r.copy()
        if r.last_period is not None:
            elapsed = max(0, period - r.last_period)
            if elapsed > 0:
                self.glicko._decay_rd(rr, elapsed)
        return rr

    def _flush_period(self, period: int) -> None:
        if not self.period_buffer:
            return
        for key, matches in self.period_buffer.items():
            self.ratings[key] = self.glicko.update(self._get_rating(key), matches, period)
        self.period_buffer = defaultdict(list)

    def _advance_to(self, period: int) -> None:
        if self.current_period is None:
            self.current_period = period
            return
        while self.current_period < period:
            self._flush_period(self.current_period)
            self.current_period += 1

    # ---- helpers de rolling -------------------------------------------
    def _winrate(self, key: str, last: int | None = None) -> float:
        hist = self.win_hist[key]
        hist = hist if last is None else hist[-last:]
        wins = sum(w for w, _ in hist)
        return _laplace_winrate(wins, len(hist))

    def _winrate_decay(self, key: str, as_of: datetime | None, last: int = 30) -> float:
        hist = self.win_hist[key][-last:]
        if not hist or as_of is None:
            return 0.5
        num = den = 0.0
        for won, date in hist:
            if date is None:
                w = 0.5
            else:
                age = max(0.0, (as_of - date).days)
                w = 0.5 ** (age / self.form_half_life)
            num += w * won
            den += w
        return (num + 0.5) / (den + 1.0) if den else 0.5

    def _avg_diff(self, key: str, last: int | None = None) -> float:
        hist = self.diff_hist[key]
        hist = hist if last is None else hist[-last:]
        return sum(hist) / len(hist) if hist else 0.0

    def _days_since(self, key: str, as_of: datetime | None, cap: int = 90) -> float:
        last = self.last_date.get(key)
        if last is None or as_of is None:
            return float(cap)
        return float(min(max((as_of - last).days, 0), cap))

    def _activity(self, key: str, as_of: datetime | None, days: int) -> float:
        if as_of is None:
            return 0.0
        # match_dates esta ordenado ascendente (append cronologico), asi que al
        # recorrer del mas reciente al mas antiguo el delta crece de forma
        # monotona: en cuanto supera la ventana, se corta. Resultado identico al
        # sum() original, pero O(partidos en ventana) en vez de O(historial).
        count = 0
        for d in reversed(self.match_dates[key]):
            delta = (as_of - d).days
            if delta > days:
                break
            if delta > 0:
                count += 1
        return float(count)

    def _avg_opp_elo(self, key: str, last: int = 10) -> float:
        hist = self.opp_elo_hist[key][-last:]
        return sum(hist) / len(hist) if hist else 1500.0

    def _sos_summary(self, key: str, as_of: datetime | None, last: int = 20) -> dict[str, float]:
        """Schedule strength and Elo-adjusted performance, strictly as-of."""
        hist = self.sos_hist[key][-last:]
        if not hist:
            return {"n": 0.0, "opp_elo_decay": 1500.0, "residual_l10": 0.0, "residual_decay": 0.0, "opp_elo_std": 0.0}
        weights: list[float] = []
        for _opp_elo, _residual, played_at in hist:
            if as_of is None or played_at is None:
                weights.append(1.0)
            else:
                age = max(0.0, (as_of - played_at).total_seconds() / 86400.0)
                weights.append(0.5 ** (age / self.form_half_life))
        weight_sum = sum(weights) or 1.0
        opponent_elos = [item[0] for item in hist]
        residuals = [item[1] for item in hist]
        opp_mean = sum(weight * value for weight, value in zip(weights, opponent_elos)) / weight_sum
        residual_decay = sum(weight * value for weight, value in zip(weights, residuals)) / weight_sum
        residual_l10_values = residuals[-10:]
        residual_l10 = sum(residual_l10_values) / len(residual_l10_values)
        plain_mean = sum(opponent_elos) / len(opponent_elos)
        opp_std = math.sqrt(sum((value - plain_mean) ** 2 for value in opponent_elos) / len(opponent_elos))
        return {
            "n": float(len(hist)),
            "opp_elo_decay": opp_mean,
            "residual_l10": residual_l10,
            "residual_decay": residual_decay,
            "opp_elo_std": opp_std,
        }

    def _format_key(self, key: str, fmt: str) -> tuple[str, str]:
        fmt = fmt if fmt in {"bo1", "bo3", "bo5"} else "bo3"
        return key, fmt

    def _format_winrate(self, key: str, fmt: str, last: int | None = None) -> float:
        hist = self.format_win_hist[self._format_key(key, fmt)]
        hist = hist if last is None else hist[-last:]
        wins = sum(w for w, _ in hist)
        return _laplace_winrate(wins, len(hist))

    def _format_avg_diff(self, key: str, fmt: str, last: int | None = None) -> float:
        hist = self.format_diff_hist[self._format_key(key, fmt)]
        hist = hist if last is None else hist[-last:]
        return sum(hist) / len(hist) if hist else 0.0

    def _format_avg_opp_elo(self, key: str, fmt: str, last: int = 10) -> float:
        hist = self.format_opp_elo_hist[self._format_key(key, fmt)][-last:]
        return sum(hist) / len(hist) if hist else 1500.0

    def _asset_map_winrate(self, key: str, last: int | None = None) -> float:
        # O(mapas del equipo) usando el indice por equipo, en vez de O(todas las
        # claves (equipo,mapa)). Con last=None (uso real) el resultado es identico
        # al de concatenar todas las listas por mapa de ese equipo.
        hist = self.asset_map_results_by_team.get(key, [])
        hist = hist if last is None else hist[-last:]
        return _laplace_winrate(sum(hist), len(hist))

    def _asset_round_winrate(self, key: str, side: str, last: int = 20) -> float:
        source = self.asset_ct_rounds if side == "ct" else self.asset_t_rounds
        hist = source[key][-last:]
        wins = sum(w for w, _total in hist)
        total = sum(total for _w, total in hist)
        return (wins + 1.0) / (total + 2.0) if total else 0.5

    def _asset_avg(self, source: dict[str, list[float]], key: str, last: int, default: float) -> float:
        hist = source[key][-last:]
        return sum(hist) / len(hist) if hist else default

    # ---- emisión de features ------------------------------------------
    def emit_features(
        self,
        a_key: str,
        b_key: str,
        date_obj: datetime | None,
        event: str = "",
        fmt: str = "bo3",
    ) -> dict[str, float]:
        fmt = fmt if fmt in {"bo1", "bo3", "bo5"} else "bo3"
        period = _period_index(date_obj)
        self._advance_to(period)

        ra = self.rating_asof(a_key, period)
        rb = self.rating_asof(b_key, period)
        glicko_prob = self.glicko.win_probability(ra, rb)

        elo_a, elo_b = self.elos[a_key], self.elos[b_key]
        elo_prob = 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))

        ts_a = self.ts_ratings.get(a_key) or self.trueskill.default()
        ts_b = self.ts_ratings.get(b_key) or self.trueskill.default()
        ts_prob = self.trueskill.win_probability(ts_a, ts_b)
        bt_a = self._bt_asof(a_key, date_obj)
        bt_b = self._bt_asof(b_key, date_obj)
        bt_prob = self.bayesian_bt.win_probability(bt_a, bt_b)
        kalman_a = self._kalman_asof(a_key, date_obj)
        kalman_b = self._kalman_asof(b_key, date_obj)
        kalman_prob = self.kalman.win_probability(kalman_a, kalman_b)

        pair = tuple(sorted((a_key, b_key)))
        pstats = self.h2h[pair]
        pn = pstats.get("n", 0)
        a_wins = pstats.get(a_key, 0)
        h2h_centered = ((a_wins + 1.0) / (pn + 2.0)) - 0.5
        fpstats = self.format_h2h[(pair[0], pair[1], fmt)]
        fpn = fpstats.get("n", 0)
        format_a_wins = fpstats.get(a_key, 0)
        format_h2h_centered = ((format_a_wins + 1.0) / (fpn + 2.0)) - 0.5

        ea = self.event_stats[(a_key, event)]
        eb = self.event_stats[(b_key, event)]
        fa = self._format_key(a_key, fmt)
        fb = self._format_key(b_key, fmt)
        a_last5_wr = self._winrate(a_key, 5)
        b_last5_wr = self._winrate(b_key, 5)
        a_last20_wr = self._winrate(a_key, 20)
        b_last20_wr = self._winrate(b_key, 20)
        a_fmt_last5_wr = self._format_winrate(a_key, fmt, 5)
        b_fmt_last5_wr = self._format_winrate(b_key, fmt, 5)
        a_fmt_last20_wr = self._format_winrate(a_key, fmt, 20)
        b_fmt_last20_wr = self._format_winrate(b_key, fmt, 20)
        days_a = self._days_since(a_key, date_obj)
        days_b = self._days_since(b_key, date_obj)
        activity_2_a = self._activity(a_key, date_obj, 2)
        activity_2_b = self._activity(b_key, date_obj, 2)
        activity_7_a = self._activity(a_key, date_obj, 7)
        activity_7_b = self._activity(b_key, date_obj, 7)
        winrate_decay_diff = self._winrate_decay(a_key, date_obj) - self._winrate_decay(b_key, date_obj)
        format_winrate_diff = self._format_winrate(a_key, fmt) - self._format_winrate(b_key, fmt)

        # Las señales están en una escala de probabilidad centrada comparable.
        # Su consenso representa separación de fuerza; dispersión y acuerdo
        # representan incertidumbre sin usar el resultado del partido.
        strength_signals = [
            glicko_prob - 0.5,
            elo_prob - 0.5,
            0.5 * winrate_decay_diff,
            0.5 * (a_last20_wr - b_last20_wr),
            0.5 * format_winrate_diff,
        ]
        strength_consensus = sum(strength_signals) / len(strength_signals)
        strength_disagreement = math.sqrt(
            sum((value - strength_consensus) ** 2 for value in strength_signals)
            / len(strength_signals)
        )
        signs = [1.0 if value > 0 else -1.0 if value < 0 else 0.0 for value in strength_signals]
        strength_agreement = abs(sum(signs)) / len(signs)

        # Rating consciente del margen (Elo-MOV).
        mov_a, mov_b = self.mov[a_key], self.mov[b_key]
        mov_prob = 1.0 / (1.0 + 10.0 ** (-(mov_a - mov_b) / 400.0))

        # Agregado de habilidad por jugador del ultimo quinteto observado (prior).
        ps_a = self.team_player_skill.get(a_key)
        ps_b = self.team_player_skill.get(b_key)
        ps_ok = bool(ps_a and ps_b)
        ps_mean = (ps_a["mean"] - ps_b["mean"]) if ps_ok else 0.0
        ps_max = (ps_a["max"] - ps_b["max"]) if ps_ok else 0.0
        ps_min = (ps_a["min"] - ps_b["min"]) if ps_ok else 0.0
        ps_spread = (ps_a["spread"] - ps_b["spread"]) if ps_ok else 0.0
        rating_ready = float(min(self.n_matches[a_key], self.n_matches[b_key]) >= 1)
        sos_a = self._sos_summary(a_key, date_obj)
        sos_b = self._sos_summary(b_key, date_obj)
        sos_matches_min = min(sos_a["n"], sos_b["n"])

        feats = {
            "glicko_diff": ra.rating - rb.rating,
            "glicko_prob_centered": glicko_prob - 0.5,
            "glicko_rd_diff": ra.rd - rb.rd,
            "glicko_rd_sum": ra.rd + rb.rd,
            "elo_diff": elo_a - elo_b,
            "elo_prob_centered": elo_prob - 0.5,
            "trueskill_diff": ts_a.mu - ts_b.mu,
            "trueskill_prob_centered": ts_prob - 0.5,
            "trueskill_available": rating_ready,
            "bayesian_bt_mean_diff": bt_a.mean - bt_b.mean,
            "bayesian_bt_prob_centered": bt_prob - 0.5,
            "bayesian_bt_available": float(min(bt_a.games, bt_b.games) >= 5),
            "bayesian_bt_games_min": float(min(bt_a.games, bt_b.games)),
            "bayesian_bt_uncertainty_sum": math.sqrt(bt_a.variance) + math.sqrt(bt_b.variance),
            "kalman_mean_diff": kalman_a.mean - kalman_b.mean,
            "kalman_prob_centered": kalman_prob - 0.5,
            "kalman_available": float(min(kalman_a.games, kalman_b.games) >= 5),
            "kalman_games_min": float(min(kalman_a.games, kalman_b.games)),
            "kalman_uncertainty_sum": math.sqrt(kalman_a.variance) + math.sqrt(kalman_b.variance),
            "mov_diff": mov_a - mov_b,
            "mov_prob_centered": mov_prob - 0.5,
            "mov_available": rating_ready,
            "player_skill_mean_diff": ps_mean,
            "player_skill_max_diff": ps_max,
            "player_skill_min_diff": ps_min,
            "player_skill_spread_diff": ps_spread,
            "player_skill_available": 1.0 if ps_ok else 0.0,
            "matches_log_diff": math.log1p(self.n_matches[a_key]) - math.log1p(self.n_matches[b_key]),
            "experience_min": float(min(self.n_matches[a_key], self.n_matches[b_key])),
            "experience_total": float(self.n_matches[a_key] + self.n_matches[b_key]),
            "winrate_diff": self._winrate(a_key) - self._winrate(b_key),
            "winrate_decay_diff": winrate_decay_diff,
            "last5_winrate_diff": a_last5_wr - b_last5_wr,
            "last10_winrate_diff": self._winrate(a_key, 10) - self._winrate(b_key, 10),
            "last20_winrate_diff": a_last20_wr - b_last20_wr,
            "last30_winrate_diff": self._winrate(a_key, 30) - self._winrate(b_key, 30),
            "winrate_trend_5v20_diff": (a_last5_wr - a_last20_wr) - (b_last5_wr - b_last20_wr),
            "avg_score_diff": self._avg_diff(a_key) - self._avg_diff(b_key),
            "last5_score_diff": self._avg_diff(a_key, 5) - self._avg_diff(b_key, 5),
            "last10_score_diff": self._avg_diff(a_key, 10) - self._avg_diff(b_key, 10),
            "last20_score_diff": self._avg_diff(a_key, 20) - self._avg_diff(b_key, 20),
            "recency_advantage": days_b - days_a,
            "inactivity_min_days": min(days_a, days_b),
            "inactivity_max_days": max(days_a, days_b),
            "activity_2_diff": activity_2_a - activity_2_b,
            "activity_2_total": activity_2_a + activity_2_b,
            "activity_7_diff": activity_7_a - activity_7_b,
            "activity_7_total": activity_7_a + activity_7_b,
            "activity_14_diff": self._activity(a_key, date_obj, 14) - self._activity(b_key, date_obj, 14),
            "activity_30_diff": self._activity(a_key, date_obj, 30) - self._activity(b_key, date_obj, 30),
            "streak_diff": float(self.streak[a_key] - self.streak[b_key]),
            "opp_elo_last5_diff": self._avg_opp_elo(a_key, 5) - self._avg_opp_elo(b_key, 5),
            "recent_opponent_elo_diff": self._avg_opp_elo(a_key) - self._avg_opp_elo(b_key),
            "opp_elo_last20_diff": self._avg_opp_elo(a_key, 20) - self._avg_opp_elo(b_key, 20),
            "sos_opp_elo_decay_diff": sos_a["opp_elo_decay"] - sos_b["opp_elo_decay"],
            "sos_performance_residual_l10_diff": sos_a["residual_l10"] - sos_b["residual_l10"],
            "sos_performance_residual_decay_diff": sos_a["residual_decay"] - sos_b["residual_decay"],
            "sos_available": float(sos_matches_min >= 5),
            "sos_matches_min": sos_matches_min,
            "sos_opponent_elo_std_sum": sos_a["opp_elo_std"] + sos_b["opp_elo_std"],
            "format_matches_log_diff": math.log1p(self.format_n_matches[fa]) - math.log1p(self.format_n_matches[fb]),
            "format_experience_min": float(min(self.format_n_matches[fa], self.format_n_matches[fb])),
            "format_experience_total": float(self.format_n_matches[fa] + self.format_n_matches[fb]),
            "format_winrate_diff": format_winrate_diff,
            "format_last5_winrate_diff": a_fmt_last5_wr - b_fmt_last5_wr,
            "format_last10_winrate_diff": self._format_winrate(a_key, fmt, 10) - self._format_winrate(b_key, fmt, 10),
            "format_winrate_trend_5v20_diff": (a_fmt_last5_wr - a_fmt_last20_wr) - (b_fmt_last5_wr - b_fmt_last20_wr),
            "format_avg_score_diff": self._format_avg_diff(a_key, fmt) - self._format_avg_diff(b_key, fmt),
            "format_last5_score_diff": self._format_avg_diff(a_key, fmt, 5) - self._format_avg_diff(b_key, fmt, 5),
            "format_last10_score_diff": self._format_avg_diff(a_key, fmt, 10) - self._format_avg_diff(b_key, fmt, 10),
            "format_opp_elo_last5_diff": self._format_avg_opp_elo(a_key, fmt, 5) - self._format_avg_opp_elo(b_key, fmt, 5),
            "format_recent_opponent_elo_diff": self._format_avg_opp_elo(a_key, fmt) - self._format_avg_opp_elo(b_key, fmt),
            "asset_maps_played_min": float(min(self.asset_team_maps[a_key], self.asset_team_maps[b_key])),
            "asset_maps_played_total": float(self.asset_team_maps[a_key] + self.asset_team_maps[b_key]),
            "asset_available": float(min(self.asset_team_maps[a_key], self.asset_team_maps[b_key]) >= 5),
            "asset_map_winrate_diff": self._asset_map_winrate(a_key) - self._asset_map_winrate(b_key),
            "asset_ct_round_winrate_diff": self._asset_round_winrate(a_key, "ct") - self._asset_round_winrate(b_key, "ct"),
            "asset_t_round_winrate_diff": self._asset_round_winrate(a_key, "t") - self._asset_round_winrate(b_key, "t"),
            "asset_rating_l5_diff": self._asset_avg(self.asset_rating_hist, a_key, 5, 1.0) - self._asset_avg(self.asset_rating_hist, b_key, 5, 1.0),
            "asset_rating_l10_diff": self._asset_avg(self.asset_rating_hist, a_key, 10, 1.0) - self._asset_avg(self.asset_rating_hist, b_key, 10, 1.0),
            "asset_rating_l20_diff": self._asset_avg(self.asset_rating_hist, a_key, 20, 1.0) - self._asset_avg(self.asset_rating_hist, b_key, 20, 1.0),
            "asset_adr_l10_diff": self._asset_avg(self.asset_adr_hist, a_key, 10, 75.0) - self._asset_avg(self.asset_adr_hist, b_key, 10, 75.0),
            "asset_kast_l10_diff": self._asset_avg(self.asset_kast_hist, a_key, 10, 70.0) - self._asset_avg(self.asset_kast_hist, b_key, 10, 70.0),
            "asset_opening_diff_l10_diff": self._asset_avg(self.asset_opening_diff_hist, a_key, 10, 0.0) - self._asset_avg(self.asset_opening_diff_hist, b_key, 10, 0.0),
            "event_matches_diff": math.log1p(ea["n"]) - math.log1p(eb["n"]),
            "event_winrate_diff": _laplace_winrate(ea["w"], ea["n"]) - _laplace_winrate(eb["w"], eb["n"]),
            "event_history_available": float(min(ea["n"], eb["n"]) >= 1),
            "event_experience_min": float(min(ea["n"], eb["n"])),
            "event_experience_total": float(ea["n"] + eb["n"]),
            "h2h_winrate_centered": h2h_centered,
            "h2h_matches": float(pn),
            "format_h2h_winrate_centered": format_h2h_centered,
            "format_h2h_matches": float(fpn),
            "format_bo1": float(fmt == "bo1"),
            "format_bo3": float(fmt == "bo3"),
            "format_bo5": float(fmt == "bo5"),
            "strength_consensus_prob_centered": strength_consensus,
            "strength_consensus_x_agreement": strength_consensus * strength_agreement,
            "strength_margin_abs": abs(strength_consensus),
            "strength_signal_disagreement": strength_disagreement,
            "strength_signal_agreement": strength_agreement,
        }
        return feats

    # ---- observación del resultado ------------------------------------
    def observe(self, match: dict[str, Any]) -> None:
        a, b = match["team1_key"], match["team2_key"]
        date_obj = match.get("date_obj")
        period = _period_index(date_obj)
        self._advance_to(period)

        a_won = bool(match["team1_win"])
        s1, s2 = int(match["score1"]), int(match["score2"])
        event = match.get("event") or ""
        fmt = match.get("format") or "bo3"
        fmt = fmt if fmt in {"bo1", "bo3", "bo5"} else "bo3"

        # Glicko: bufferiza con el rating del oponente decaído a este periodo.
        ra_snap = self.rating_asof(a, period)
        rb_snap = self.rating_asof(b, period)
        self._get_rating(a)
        self._get_rating(b)
        self.period_buffer[a].append(_Match(rb_snap, 1.0 if a_won else 0.0))
        self.period_buffer[b].append(_Match(ra_snap, 0.0 if a_won else 1.0))

        # Elo: actualización online inmediata.
        elo_a, elo_b = self.elos[a], self.elos[b]
        exp_a = 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))
        actual_a = 1.0 if a_won else 0.0
        k = 32.0
        self.elos[a] = elo_a + k * (actual_a - exp_a)
        self.elos[b] = elo_b + k * ((1.0 - actual_a) - (1.0 - exp_a))

        # TrueSkill: actualización online inmediata (ganador vs perdedor).
        ts_a = self.ts_ratings.get(a) or self.trueskill.default()
        ts_b = self.ts_ratings.get(b) or self.trueskill.default()
        if a_won:
            self.ts_ratings[a], self.ts_ratings[b] = self.trueskill.update(ts_a, ts_b)
        else:
            self.ts_ratings[b], self.ts_ratings[a] = self.trueskill.update(ts_b, ts_a)

        bt_a = self._bt_asof(a, date_obj)
        bt_b = self._bt_asof(b, date_obj)
        self.bt_ratings[a], self.bt_ratings[b] = self.bayesian_bt.update(bt_a, bt_b, a_won)
        if date_obj is not None:
            self.bt_last_date[a] = date_obj
            self.bt_last_date[b] = date_obj

        kalman_a = self._kalman_asof(a, date_obj)
        kalman_b = self._kalman_asof(b, date_obj)
        self.kalman_ratings[a], self.kalman_ratings[b] = self.kalman.update(kalman_a, kalman_b, a_won)
        if date_obj is not None:
            self.kalman_last_date[a] = date_obj
            self.kalman_last_date[b] = date_obj

        # Elo-MOV: mismo esquema Elo pero con multiplicador por margen de mapas
        # (formula estilo FiveThirtyEight; 2-0 pesa mas que 2-1).
        mov_a, mov_b = self.mov[a], self.mov[b]
        mov_exp_a = 1.0 / (1.0 + 10.0 ** (-(mov_a - mov_b) / 400.0))
        margin = abs(s1 - s2)
        winner_gap = (mov_a - mov_b) if a_won else (mov_b - mov_a)
        mov_mult = math.log(margin + 1.0) * (2.2 / (abs(winner_gap) * 0.001 + 2.2))
        mov_k = 32.0 * (mov_mult if mov_mult > 0 else 1.0)
        self.mov[a] = mov_a + mov_k * (actual_a - mov_exp_a)
        self.mov[b] = mov_b + mov_k * ((1.0 - actual_a) - (1.0 - mov_exp_a))

        # Rating por jugador (si el box score expone el quinteto de ambos equipos).
        self._observe_player_ratings(match, a, b, a_won)

        # rolling
        self.n_matches[a] += 1
        self.n_matches[b] += 1
        self.win_hist[a].append((int(a_won), date_obj))
        self.win_hist[b].append((int(not a_won), date_obj))
        self.diff_hist[a].append(s1 - s2)
        self.diff_hist[b].append(s2 - s1)
        self.opp_elo_hist[a].append(elo_b)
        self.opp_elo_hist[b].append(elo_a)
        self.sos_hist[a].append((elo_b, actual_a - exp_a, date_obj))
        self.sos_hist[b].append((elo_a, exp_a - actual_a, date_obj))
        fa = self._format_key(a, fmt)
        fb = self._format_key(b, fmt)
        self.format_n_matches[fa] += 1
        self.format_n_matches[fb] += 1
        self.format_win_hist[fa].append((int(a_won), date_obj))
        self.format_win_hist[fb].append((int(not a_won), date_obj))
        self.format_diff_hist[fa].append(s1 - s2)
        self.format_diff_hist[fb].append(s2 - s1)
        self.format_opp_elo_hist[fa].append(elo_b)
        self.format_opp_elo_hist[fb].append(elo_a)
        if date_obj is not None:
            self.match_dates[a].append(date_obj)
            self.match_dates[b].append(date_obj)
            self.last_date[a] = date_obj
            self.last_date[b] = date_obj
        self.streak[a] = (max(self.streak[a], 0) + 1) if a_won else (min(self.streak[a], 0) - 1)
        self.streak[b] = (max(self.streak[b], 0) + 1) if not a_won else (min(self.streak[b], 0) - 1)

        pair = tuple(sorted((a, b)))
        self.h2h[pair]["n"] = self.h2h[pair].get("n", 0) + 1
        winner = a if a_won else b
        self.h2h[pair][winner] = self.h2h[pair].get(winner, 0) + 1
        format_pair = (pair[0], pair[1], fmt)
        self.format_h2h[format_pair]["n"] = self.format_h2h[format_pair].get("n", 0) + 1
        self.format_h2h[format_pair][winner] = self.format_h2h[format_pair].get(winner, 0) + 1

        self.event_stats[(a, event)]["n"] += 1
        self.event_stats[(a, event)]["w"] += int(a_won)
        self.event_stats[(b, event)]["n"] += 1
        self.event_stats[(b, event)]["w"] += int(not a_won)
        self._observe_asset(match)

    def team_feature_row(self, key: str, date_obj: datetime | None) -> dict[str, Any]:
        """Features ABSOLUTAS de un equipo as-of la fecha (para `match_features`)."""
        period = _period_index(date_obj)
        r = self.rating_asof(key, period)
        return {
            "glicko_rating": round(r.rating, 2),
            "glicko_rd": round(r.rd, 2),
            "glicko_sigma": round(r.sigma, 5),
            "elo": round(self.elos.get(key, 1500.0), 2),
            "form_winrate_10": round(self._winrate(key, 10), 4),
            "form_winrate_20": round(self._winrate(key, 20), 4),
            "winrate_overall": round(self._winrate(key), 4),
            "avg_score_diff": round(self._avg_diff(key), 4),
            "recent_opp_elo": round(self._avg_opp_elo(key), 2),
            "streak": int(self.streak.get(key, 0)),
            "matches_played": int(self.n_matches.get(key, 0)),
            "days_since_last": int(self._days_since(key, date_obj)),
        }

    def _observe_player_ratings(self, match: dict[str, Any], a_key: str, b_key: str, a_won: bool) -> None:
        """Actualiza TrueSkill por jugador desde el box score y guarda el agregado
        del quinteto por equipo (media/max/min/dispersion de mu).

        Point-in-time: emit_features lee el agregado GUARDADO (de partidos previos);
        aqui se actualiza DESPUES de emitir. Aproximacion de juego por equipos: cada
        jugador se enfrenta al rating agregado del equipo rival (pre-update).
        """
        asset = match.get("asset")
        if not isinstance(asset, dict):
            return
        players_by_team: dict[str, set[str]] = defaultdict(set)
        for map_payload in asset.get("mapstats") or []:
            for stat in map_payload.get("player_stats") or []:
                if stat.get("side") != "total":
                    continue
                team_key = _clean_team(stat.get("team_name"))
                player = _clean_team(stat.get("player_name") or stat.get("nick"))
                if team_key in (a_key, b_key) and player:
                    players_by_team[team_key].add(player)
        team_a_players = players_by_team.get(a_key) or set()
        team_b_players = players_by_team.get(b_key) or set()
        if len(team_a_players) < 3 or len(team_b_players) < 3:
            return  # cobertura insuficiente para un rating por jugador fiable

        def _team_agg(players: set[str]) -> TSRating:
            rs = [self.player_ts.get(p) or self.trueskill.default() for p in players]
            mu = sum(r.mu for r in rs) / len(rs)
            sigma = math.sqrt(sum(r.sigma * r.sigma for r in rs)) / len(rs)
            return TSRating(mu, max(sigma, 1e-3))

        opp_for_a = _team_agg(team_b_players)   # rival de los jugadores de A (pre-update)
        opp_for_b = _team_agg(team_a_players)
        for p in team_a_players:
            r = self.player_ts.get(p) or self.trueskill.default()
            self.player_ts[p] = (self.trueskill.update(r, opp_for_a)[0] if a_won
                                 else self.trueskill.update(opp_for_a, r)[1])
        for p in team_b_players:
            r = self.player_ts.get(p) or self.trueskill.default()
            self.player_ts[p] = (self.trueskill.update(r, opp_for_b)[0] if not a_won
                                 else self.trueskill.update(opp_for_b, r)[1])
        for team_key, players in ((a_key, team_a_players), (b_key, team_b_players)):
            mus = [(self.player_ts.get(p) or self.trueskill.default()).mu for p in players]
            self.team_player_skill[team_key] = {
                "mean": sum(mus) / len(mus),
                "max": max(mus),
                "min": min(mus),
                "spread": max(mus) - min(mus),
            }

    def _observe_asset(self, match: dict[str, Any]) -> None:
        asset = match.get("asset")
        if not isinstance(asset, dict):
            return
        observed_at = match.get("date_obj")
        for map_payload in asset.get("mapstats") or []:
            info = map_payload.get("info") or {}
            map_name = info.get("map_name") or "Unknown"
            normalized_map = str(map_name).strip().lower()
            if (
                isinstance(observed_at, datetime)
                and normalized_map not in {"", "unknown", "tba"}
            ):
                self.map_regime_observations.append((observed_at, normalized_map))
            left = info.get("team_left") or {}
            right = info.get("team_right") or {}
            left_key = _clean_team(left.get("name"))
            right_key = _clean_team(right.get("name"))
            if not left_key or not right_key:
                continue
            left_score = _safe_int(left.get("score"))
            right_score = _safe_int(right.get("score"))
            if left_score is not None and right_score is not None and left_score != right_score:
                self.asset_map_hist[(left_key, map_name)].append(int(left_score > right_score))
                self.asset_map_hist[(right_key, map_name)].append(int(right_score > left_score))
                self.asset_map_results_by_team[left_key].append(int(left_score > right_score))
                self.asset_map_results_by_team[right_key].append(int(right_score > left_score))
                self.asset_team_maps[left_key] += 1
                self.asset_team_maps[right_key] += 1

            side_rounds = (info.get("breakdown") or {}).get("side_rounds") or {}
            left_side = side_rounds.get("left") or {}
            right_side = side_rounds.get("right") or {}
            left_ct = _safe_int(left_side.get("ct"))
            left_t = _safe_int(left_side.get("t"))
            right_ct = _safe_int(right_side.get("ct"))
            right_t = _safe_int(right_side.get("t"))
            if left_ct is not None and right_t is not None:
                self.asset_ct_rounds[left_key].append((left_ct, left_ct + right_t))
                self.asset_t_rounds[right_key].append((right_t, left_ct + right_t))
            if right_ct is not None and left_t is not None:
                self.asset_ct_rounds[right_key].append((right_ct, right_ct + left_t))
                self.asset_t_rounds[left_key].append((left_t, right_ct + left_t))

            by_team: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for stat in map_payload.get("player_stats") or []:
                if stat.get("side") != "total":
                    continue
                team_key = _clean_team(stat.get("team_name"))
                if not team_key:
                    continue
                for field in ("rating", "adr", "kast"):
                    value = _safe_float(stat.get(field))
                    if value is not None:
                        by_team[team_key][field].append(value)
                opening_kills = _safe_float(stat.get("opening_kills"))
                opening_deaths = _safe_float(stat.get("opening_deaths"))
                if opening_kills is not None and opening_deaths is not None:
                    by_team[team_key]["opening_diff"].append(opening_kills - opening_deaths)
            for team_key, fields in by_team.items():
                if fields.get("rating"):
                    self.asset_rating_hist[team_key].append(sum(fields["rating"]) / len(fields["rating"]))
                if fields.get("adr"):
                    self.asset_adr_hist[team_key].append(sum(fields["adr"]) / len(fields["adr"]))
                if fields.get("kast"):
                    self.asset_kast_hist[team_key].append(sum(fields["kast"]) / len(fields["kast"]))
                if fields.get("opening_diff"):
                    self.asset_opening_diff_hist[team_key].append(sum(fields["opening_diff"]))

    def rating_card(self, key: str, date_obj: datetime | None = None) -> dict[str, Any]:
        """Resumen del rating de un equipo para mostrar/exportar."""
        period = _period_index(date_obj) if date_obj is not None else (self.current_period or 0)
        r = self.rating_asof(key, period)
        return {
            "glicko_rating": round(r.rating, 1),
            "glicko_rd": round(r.rd, 1),
            "glicko_sigma": round(r.sigma, 4),
            "elo": round(self.elos.get(key, 1500.0), 1),
            "matches": self.n_matches.get(key, 0),
            "glicko_games": r.games,
        }


def build_training_frame(
    rows: list[dict[str, Any]],
    form_half_life: float = FORM_HALF_LIFE,
) -> tuple[list[dict[str, float]], list[int], list[dict[str, Any]], ChronologicalState]:
    """Itera el histórico cronológicamente y devuelve (X, y, meta, estado_final).

    Para cada partido: emite features con el estado previo, registra la etiqueta
    y luego observa el resultado. El estado final sirve para predecir en vivo.
    `form_half_life` (dias) controla el decaimiento temporal de la forma.
    """
    state = ChronologicalState(form_half_life=form_half_life)
    X: list[dict[str, float]] = []
    y: list[int] = []
    meta: list[dict[str, Any]] = []
    def emit(m: dict[str, Any]) -> None:
        match_context = m.get("match_context") or {}
        feats = state.emit_features(
            m["team1_key"], m["team2_key"], m.get("date_obj"), m.get("event") or "", m.get("format") or "bo3"
        )
        feats.update(state.regime_features(m))
        feats.update(analytics_match_features(m))
        feats.update(announced_lineup_features(m))
        feats.update(event_metadata_features(m))
        feats.update(context_match_features(m))
        feats.update(player_snapshot_features(m))
        feats.update(external_snapshot_features(m))
        X.append(feats)
        y.append(int(m["team1_win"]))
        meta.append(
            {
                "id": m["id"],
                "date": m["date"],
                "event": m.get("event"),
                "event_tier": m.get("event_tier"),
                "format": m.get("format"),
                "environment": match_context.get("environment"),
                "stage": match_context.get("stage"),
                "patch_version": m.get("patch_version") or match_context.get("patch_version"),
                "map_pool_regime": state.map_pool_regime_label(m.get("date_obj")),
                "datetime_precision": m.get("datetime_precision") or "exact",
                "team1": m["team1"],
                "team2": m["team2"],
                "team1_key": m["team1_key"],
                "team2_key": m["team2_key"],
            }
        )

    index = 0
    while index < len(rows):
        day = str(rows[index].get("date") or "")[:10]
        end = index + 1
        while end < len(rows) and str(rows[end].get("date") or "")[:10] == day:
            end += 1
        day_rows = rows[index:end]
        has_unknown_order = any(
            row.get("datetime_precision") == "date_only" for row in day_rows
        )
        if has_unknown_order:
            # No inventamos una hora para la semilla historica: todos los
            # partidos del dia se predicen con el estado al inicio del dia.
            for match in day_rows:
                emit(match)
            for match in day_rows:
                state.observe(match)
        else:
            for match in day_rows:
                emit(match)
                state.observe(match)
        index = end
    return X, y, meta, state

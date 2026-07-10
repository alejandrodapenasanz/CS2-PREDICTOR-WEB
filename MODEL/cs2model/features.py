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
from datetime import datetime
from typing import Any

from .glicko2 import Glicko2, Rating, _Match

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
FEATURE_COLUMNS = DIFF_COLUMNS + SYM_COLUMNS

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
    "player_rating_min_diff",
    "player_rating_std_diff",
    "player_rating_spread_diff",
    "player_star_gap_diff",
    "player_weak_link_gap_diff",
    "player_kpr_diff",
    "player_kast_diff",
    "player_adr_diff",
    "player_impact_diff",
    "player_round_swing_diff",
    "player_opening_kpr_diff",
]
PLAYER_SYM_COLUMNS = [
    "player_snapshot_available",
    "player_coverage_min",
    "player_maps_min",
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
    + PLAYER_DIFF_COLUMNS
    + RANKING_DIFF_COLUMNS
    + ROSTER_DIFF_COLUMNS
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
    feats = {col: 0.0 for col in ANALYTICS_FEATURE_COLUMNS}
    analytics = match.get("analytics") or {}
    if not isinstance(analytics, dict) or not analytics.get("available"):
        return feats

    t1 = _clean_team(match.get("team1") or match.get("team1_name"))
    t2 = _clean_team(match.get("team2") or match.get("team2_name"))
    if not t1 or not t2:
        t1 = _clean_team(match.get("team1_key"))
        t2 = _clean_team(match.get("team2_key"))

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

        # rolling de resultados (cada entrada lleva la fecha para decay)
        self.win_hist: dict[str, list[tuple[int, datetime | None]]] = defaultdict(list)
        self.diff_hist: dict[str, list[int]] = defaultdict(list)
        self.match_dates: dict[str, list[datetime]] = defaultdict(list)
        self.opp_elo_hist: dict[str, list[float]] = defaultdict(list)
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
        self.asset_team_maps: dict[str, int] = defaultdict(int)
        self.asset_ct_rounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.asset_t_rounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.asset_rating_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_adr_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_kast_hist: dict[str, list[float]] = defaultdict(list)
        self.asset_opening_diff_hist: dict[str, list[float]] = defaultdict(list)

    # ---- gestión de ratings -------------------------------------------
    def _get_rating(self, key: str) -> Rating:
        r = self.ratings.get(key)
        if r is None:
            r = Rating()
            self.ratings[key] = r
        return r

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
        return float(sum(0 < (as_of - d).days <= days for d in self.match_dates[key]))

    def _avg_opp_elo(self, key: str, last: int = 10) -> float:
        hist = self.opp_elo_hist[key][-last:]
        return sum(hist) / len(hist) if hist else 1500.0

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
        hist: list[int] = []
        for team_key, _map_name in list(self.asset_map_hist.keys()):
            if team_key == key:
                hist.extend(self.asset_map_hist[(team_key, _map_name)])
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

        feats = {
            "glicko_diff": ra.rating - rb.rating,
            "glicko_prob_centered": glicko_prob - 0.5,
            "glicko_rd_diff": ra.rd - rb.rd,
            "glicko_rd_sum": ra.rd + rb.rd,
            "elo_diff": elo_a - elo_b,
            "elo_prob_centered": elo_prob - 0.5,
            "matches_log_diff": math.log1p(self.n_matches[a_key]) - math.log1p(self.n_matches[b_key]),
            "experience_min": float(min(self.n_matches[a_key], self.n_matches[b_key])),
            "experience_total": float(self.n_matches[a_key] + self.n_matches[b_key]),
            "winrate_diff": self._winrate(a_key) - self._winrate(b_key),
            "winrate_decay_diff": self._winrate_decay(a_key, date_obj) - self._winrate_decay(b_key, date_obj),
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
            "format_matches_log_diff": math.log1p(self.format_n_matches[fa]) - math.log1p(self.format_n_matches[fb]),
            "format_experience_min": float(min(self.format_n_matches[fa], self.format_n_matches[fb])),
            "format_experience_total": float(self.format_n_matches[fa] + self.format_n_matches[fb]),
            "format_winrate_diff": self._format_winrate(a_key, fmt) - self._format_winrate(b_key, fmt),
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

        # rolling
        self.n_matches[a] += 1
        self.n_matches[b] += 1
        self.win_hist[a].append((int(a_won), date_obj))
        self.win_hist[b].append((int(not a_won), date_obj))
        self.diff_hist[a].append(s1 - s2)
        self.diff_hist[b].append(s2 - s1)
        self.opp_elo_hist[a].append(elo_b)
        self.opp_elo_hist[b].append(elo_a)
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

    def _observe_asset(self, match: dict[str, Any]) -> None:
        asset = match.get("asset")
        if not isinstance(asset, dict):
            return
        for map_payload in asset.get("mapstats") or []:
            info = map_payload.get("info") or {}
            map_name = info.get("map_name") or "Unknown"
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
    for m in rows:
        feats = state.emit_features(
            m["team1_key"], m["team2_key"], m.get("date_obj"), m.get("event") or "", m.get("format") or "bo3"
        )
        feats.update(analytics_match_features(m))
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
                "format": m.get("format"),
                "team1": m["team1"],
                "team2": m["team2"],
                "team1_key": m["team1_key"],
                "team2_key": m["team2_key"],
            }
        )
        state.observe(m)
    return X, y, meta, state

"""Entrenamiento y evaluación walk-forward del modelo de predicción de CS2.

Pipeline (PROJECT.md §5, §7, §8):
  1. Carga el histórico (solo era CS2) y construye features point-in-time
     (Glicko-2 + Elo + forma + h2h + actividad), cronológicamente.
  2. Validación walk-forward por semanas (ventana expansiva, nunca k-fold
     aleatorio): para cada semana de test se entrena solo con el pasado.
  3. Modelos: baselines (base-rate, Elo, Glicko) + regresión logística +
     LightGBM con calibración isotónica.
  4. Métricas probabilísticas: log loss, Brier, ROC-AUC, accuracy, ECE.
  5. Ajuste final sobre todo el histórico + calibrador isotónico sobre holdout
     reciente -> artefacto que carga producción.
  6. Importancia de features con SHAP (interpretabilidad).

Uso:
    python MODEL/train.py                         # fuente por defecto: BBDD/cs2.db
    python MODEL/train.py --raw <results_all.json> --warmup-weeks 10 --min-train 800
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# Windows PowerShell may expose CP1252 stdout. Keep verbose training output
# informative without letting one non-ASCII diagnostic abort a completed run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError):
        pass

from cs2model import dataio
from cs2model.features import (
    build_training_frame,
    FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    STRENGTH_INTERACTION_COLUMNS,
    DIFF_COLUMNS,
    ANALYTICS_FEATURE_COLUMNS,
    ANALYTICS_EXTENDED_FEATURE_COLUMNS,
    ANNOUNCED_LINEUP_FEATURE_COLUMNS,
    EVENT_METADATA_FEATURE_COLUMNS,
    CONTEXT_FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    MAP_ASSET_FEATURE_COLUMNS,
    EVENT_HISTORY_FEATURE_COLUMNS,
    RANKING_FEATURE_COLUMNS,
    ROSTER_FEATURE_COLUMNS,
    TRUESKILL_FEATURE_COLUMNS,
    MOV_FEATURE_COLUMNS,
    PLAYER_RATING_FEATURE_COLUMNS,
    SOS_FEATURE_COLUMNS,
    BAYES_BT_FEATURE_COLUMNS,
    KALMAN_FEATURE_COLUMNS,
    BO3_COMPOSITIONAL_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    EXTENDED_DIFF_COLUMNS,
    _period_index,
)
from cs2model.metrics import metric_dict, calibration_bins
from cs2model.artifacts import (
    ARTIFACT_PATH,
    ColumnSubsetEstimator,
    Component,
    ModelArtifact,
    ProbabilityColumnEstimator,
)
from cs2model.calibration import BetaCalibratedClassifier
from cs2model.diagnostics import feature_pruning_diagnostics, prune_exact_redundancies
from cs2model.rich_targets import (
    BO3_SCORE_CLASSES,
    evaluate_rich_target_walk_forward,
    fit_rich_target_model,
)
from cs2model.compositional_bo3 import MAP_POOL_MIN_ROWS
from cs2model.optuna_tuning import tune_logistic_c_purged
from cs2model.config import (
    DEFAULT_CONFIG_PATH,
    get_runtime_config,
    load_config,
    set_runtime_config,
)
from cs2model.reproducibility import experiment_manifest, set_global_determinism
from cs2model.economic import economic_backtest as run_economic_backtest
from cs2model.drift import build_drift_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = load_config()
DEFAULT_RAW = (
    ROOT / "SCRAPER" / "hltv-scraper-api" / "hltv_scraper" / "data" / "raw"
    / "history_10000_2026-06-28" / "results_all.json"
)
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
OUTPUT_DIR = ROOT / "MODEL" / "results"
MODEL_REGISTRY_DIR = ROOT / "MODEL" / "artifacts" / "registry"
ODDS_FEATURE_COLUMNS = [
    "opening_odds_prob_centered",
    "opening_odds_confidence",
    "opening_bookmaker_count_log",
]
EXTRA_DIFF_COLUMNS = {"opening_odds_prob_centered"}
# Las columnas DIFF de trueskill/mov/player-rating ya estan en
# EXTENDED_DIFF_COLUMNS, asi que augment() las niega correctamente.
ANALYTICS_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("analytics", 120)
ANALYTICS_EXTENDED_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("analytics_extended", 200)
# Una alineacion anunciada completa tiene varias variables correlacionadas; se
# exige una muestra cerrada mayor antes de dejar que altere produccion.
ANNOUNCED_LINEUP_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("announced_lineups", 200)
# Prize pool/tamano del evento son contexto de calibracion, no una ventaja de
# lado. Requieren mas eventos antes de permitir interacciones no lineales.
EVENT_METADATA_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("event_metadata", 300)
CONTEXT_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("context_total", 200)
CONTEXT_MIN_ENV_ROWS = DEFAULT_CONFIG.feature_thresholds.get("context_per_environment", 50)
PLAYER_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("player_snapshots", 200)
MAP_ASSET_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("map_box_scores", 200)
EVENT_HISTORY_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("event_history", 200)
RANKING_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("rankings", 200)
ROSTER_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("roster", 200)
# Ratings adicionales auto-gated. MOV y TrueSkill son reconstruibles de todo el
# histórico (umbral alto = entran con historia suficiente); el rating por jugador
# depende de cobertura de box score, umbral como el de player snapshots.
MOV_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("mov_rating", 800)
TRUESKILL_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("team_trueskill", 800)
PLAYER_RATING_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("player_rating", 200)
SOS_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("strength_of_schedule", 800)
BAYES_BT_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("bayesian_bradley_terry", 800)
KALMAN_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("kalman_state_space", 800)
BO3_COMPOSITIONAL_MIN_TRAIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get(
    "bo3_map_compositional", MAP_POOL_MIN_ROWS
)
RICH_TARGET_MIN_ROWS = DEFAULT_CONFIG.feature_thresholds.get("rich_target_total", 2000)
RICH_TARGET_MIN_CLASS_ROWS = DEFAULT_CONFIG.feature_thresholds.get("rich_target_per_class", 300)
ALL_ALGORITHMS = ("logistic", "lightgbm", "catboost", "xgboost", "random_forest")
DEFAULT_ALGORITHMS = ALL_ALGORITHMS
KIND_BY_ALGORITHM = {
    "logistic": "logistic",
    "lightgbm": "gbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "random_forest": "random_forest",
}
INDIVIDUAL_CANDIDATE = {
    "logistic": "logistic_cal",
    "gbm": "lightgbm_cal",
    "catboost": "catboost_cal",
    "xgboost": "xgboost_cal",
    "random_forest": "random_forest_cal",
}
SUPER_LEARNER_MIN_HISTORY = 500

AUTO_FEATURE_FAMILIES = (
    ("map_box_scores", MAP_ASSET_FEATURE_COLUMNS, "asset_available", MAP_ASSET_MIN_TRAIN_ROWS),
    ("event_history", EVENT_HISTORY_FEATURE_COLUMNS, "event_history_available", EVENT_HISTORY_MIN_TRAIN_ROWS),
    ("analytics", ANALYTICS_FEATURE_COLUMNS, "analytics_available", ANALYTICS_MIN_TRAIN_ROWS),
    ("analytics_extended", ANALYTICS_EXTENDED_FEATURE_COLUMNS, "analytics_extended_available", ANALYTICS_EXTENDED_MIN_TRAIN_ROWS),
    ("announced_lineups", ANNOUNCED_LINEUP_FEATURE_COLUMNS, "announced_lineup_available", ANNOUNCED_LINEUP_MIN_TRAIN_ROWS),
    ("event_metadata", EVENT_METADATA_FEATURE_COLUMNS, "event_metadata_available", EVENT_METADATA_MIN_TRAIN_ROWS),
    ("player_snapshots", PLAYER_FEATURE_COLUMNS, "player_snapshot_available", PLAYER_MIN_TRAIN_ROWS),
    ("rankings", RANKING_FEATURE_COLUMNS, "ranking_available", RANKING_MIN_TRAIN_ROWS),
    ("roster", ROSTER_FEATURE_COLUMNS, "roster_available", ROSTER_MIN_TRAIN_ROWS),
    ("mov_rating", MOV_FEATURE_COLUMNS, "mov_available", MOV_MIN_TRAIN_ROWS),
    ("team_trueskill", TRUESKILL_FEATURE_COLUMNS, "trueskill_available", TRUESKILL_MIN_TRAIN_ROWS),
    ("player_rating", PLAYER_RATING_FEATURE_COLUMNS, "player_skill_available", PLAYER_RATING_MIN_TRAIN_ROWS),
    ("strength_of_schedule", SOS_FEATURE_COLUMNS, "sos_available", SOS_MIN_TRAIN_ROWS),
    ("bayesian_bradley_terry", BAYES_BT_FEATURE_COLUMNS, "bayesian_bt_available", BAYES_BT_MIN_TRAIN_ROWS),
    ("kalman_state_space", KALMAN_FEATURE_COLUMNS, "kalman_available", KALMAN_MIN_TRAIN_ROWS),
    (
        "bo3_map_compositional",
        BO3_COMPOSITIONAL_FEATURE_COLUMNS,
        "bo3_compositional_available",
        BO3_COMPOSITIONAL_MIN_TRAIN_ROWS,
    ),
    (
        "regime",
        REGIME_FEATURE_COLUMNS,
        "regime_available",
        DEFAULT_CONFIG.feature_thresholds.get("regime", 200),
    ),
)


def _matrix(rows: list[dict[str, float]], cols: list[str]) -> np.ndarray:
    return np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)


def _recency_weights(periods_subset: np.ndarray, half_life_days: float) -> np.ndarray | None:
    """Pesos por recencia (Dixon-Coles): 0.5**(edad_dias/half_life). None si off.

    A1: el meta reciente pesa mas en el entrenamiento. Los periodos son semanales
    (PERIOD_DAYS=7), asi que la edad en dias = (periodo_max - periodo) * 7.
    """
    if not half_life_days or half_life_days <= 0:
        return None
    p = np.asarray(periods_subset, dtype=float)
    if p.size == 0:
        return None
    ages = (p.max() - p) * 7.0
    return np.clip(0.5 ** (ages / float(half_life_days)), 1e-3, 1.0)


def select_feature_columns(
    X_dicts: list[dict[str, float]],
    feature_profile: str = "error-aware",
    feature_thresholds: dict[str, int] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    thresholds = feature_thresholds or {}
    context_min_rows = int(thresholds.get("context_total", CONTEXT_MIN_TRAIN_ROWS))
    context_min_env_rows = int(
        thresholds.get("context_per_environment", CONTEXT_MIN_ENV_ROWS)
    )
    context_rows = sum(1 for row in X_dicts if (row.get("context_available") or 0.0) >= 0.5)
    lan_rows = sum(1 for row in X_dicts if (row.get("context_is_lan") or 0.0) >= 0.5)
    online_rows = sum(1 for row in X_dicts if (row.get("context_is_online") or 0.0) >= 0.5)
    stage_rows = sum(1 for row in X_dicts if (row.get("context_stage_known") or 0.0) >= 0.5)
    context_enabled = (
        context_rows >= context_min_rows
        and lan_rows >= context_min_env_rows
        and online_rows >= context_min_env_rows
    )
    use_strength_interactions = feature_profile == "error-aware"
    columns = list(FEATURE_COLUMNS if use_strength_interactions else BASE_FEATURE_COLUMNS)
    policies: dict[str, Any] = {
        "strength_interactions": {
            "available_rows": len(X_dicts),
            "min_rows": 0,
            "availability_column": "point_in_time_derived",
            "enabled": use_strength_interactions,
            "columns": list(STRENGTH_INTERACTION_COLUMNS) if use_strength_interactions else [],
            "activation": "feature_profile_ablation",
            "note": (
                "Point-in-time consensus/disagreement features enabled."
                if use_strength_interactions else
                "Core ablation: consensus/disagreement features disabled."
            ),
        }
    }
    for name, family_columns, availability_column, min_rows in AUTO_FEATURE_FAMILIES:
        min_rows = int(thresholds.get(name, min_rows))
        available_rows = sum(
            1 for row in X_dicts
            if (row.get(availability_column) or 0.0) >= 0.5
        )
        enabled = available_rows >= min_rows
        if enabled:
            columns += list(family_columns)
        policies[name] = {
            "available_rows": available_rows,
            "min_rows": min_rows,
            "availability_column": availability_column,
            "enabled": enabled,
            "columns": list(family_columns) if enabled else [],
            "activation": "automatic_at_training_time",
            "note": (
                f"{name} enabled automatically in training."
                if enabled else
                f"{name} stored but excluded until enough closed point-in-time rows exist."
            ),
        }
    if context_enabled:
        columns += list(CONTEXT_FEATURE_COLUMNS)
    policies["context"] = {
        "available_rows": context_rows,
        "lan_rows": lan_rows,
        "online_rows": online_rows,
        "stage_rows": stage_rows,
        "min_rows": context_min_rows,
        "min_env_rows": context_min_env_rows,
        "enabled": context_enabled,
        "columns": list(CONTEXT_FEATURE_COLUMNS) if context_enabled else [],
        "activation": "automatic_at_training_time",
        "note": (
            "Tournament context enabled automatically in training."
            if context_enabled else
            "Tournament context stored but excluded until total and LAN/online coverage pass their thresholds."
        ),
    }
    return columns, policies


def augment(X: np.ndarray, y: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Duplica el dataset intercambiando A<->B (niega columnas DIFF, invierte y).

    Enseña al modelo una frontera antisimétrica y elimina el sesgo de 'team1'.
    """
    diff_idx = [
        i for i, c in enumerate(cols)
        if c in DIFF_COLUMNS or c in EXTENDED_DIFF_COLUMNS or c in EXTRA_DIFF_COLUMNS
    ]
    X_rev = X.copy()
    X_rev[:, diff_idx] *= -1.0
    return np.vstack([X, X_rev]), np.concatenate([y, 1 - y])


# Features "ventaja de team1": a mayor valor, mas probable que gane team1. Se
# imponen como restricciones monotonas crecientes en los GBDT: reducen overfitting
# con ~7k series y mejoran la generalizacion (LightGBM monotone_constraints).
MONOTONE_INCREASING = {
    "glicko_prob_centered", "glicko_diff", "elo_prob_centered", "elo_diff",
    "winrate_diff", "winrate_decay_diff", "last5_winrate_diff", "last10_winrate_diff",
    "last20_winrate_diff", "last30_winrate_diff", "avg_score_diff", "last5_score_diff",
    "last10_score_diff", "last20_score_diff", "streak_diff",
    "format_winrate_diff", "format_avg_score_diff",
    "h2h_winrate_centered", "format_h2h_winrate_centered",
    "asset_map_winrate_diff", "asset_rating_l10_diff",
    "event_winrate_diff", "analytics_map_win_pct_diff",
    "announced_lineup_rating_diff", "announced_lineup_rating_top2_avg_diff",
    "announced_lineup_rating_bottom2_avg_diff", "announced_lineup_kpr_diff",
    "announced_lineup_kast_diff", "announced_lineup_adr_diff",
    "announced_lineup_multi_kill_rating_diff", "announced_lineup_round_swing_diff",
    "announced_lineup_standin_advantage",
    "player_rating_diff", "player_kpr_diff", "player_adr_diff", "player_impact_diff",
    "player_opening_kpr_diff",
    "ranking_hltv_position_advantage", "ranking_hltv_points_diff",
    "ranking_valve_position_advantage", "ranking_valve_points_diff",
    "roster_days_log_diff", "roster_standin_risk_advantage",
    "trueskill_diff", "trueskill_prob_centered",
    "mov_diff", "mov_prob_centered",
    "player_skill_mean_diff", "player_skill_max_diff", "player_skill_min_diff",
    "sos_performance_residual_l10_diff", "sos_performance_residual_decay_diff",
    "bayesian_bt_mean_diff", "bayesian_bt_prob_centered",
    "kalman_mean_diff", "kalman_prob_centered",
    "bo3_compositional_prob_centered",
}
CALIBRATION_METHODS = ("sigmoid", "isotonic", "beta")
RATING_CANDIDATE_COLUMNS = {
    "bayesian_bt_cal": "bayesian_bt_prob_centered",
    "kalman_cal": "kalman_prob_centered",
}


def _monotone_vector(cols: list[str]) -> list[int]:
    return [1 if c in MONOTONE_INCREASING else 0 for c in cols]


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except Exception:
        return False


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def parse_algorithms(raw: str) -> tuple[str, ...]:
    requested = [item.strip().lower().replace("-", "_") for item in raw.split(",") if item.strip()]
    if requested == ["all"]:
        requested = list(ALL_ALGORITHMS)
    unknown = sorted(set(requested) - set(ALL_ALGORITHMS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Algoritmos desconocidos: {', '.join(unknown)}")
    if not requested:
        raise argparse.ArgumentTypeError("Debes indicar al menos un algoritmo.")
    return tuple(dict.fromkeys(requested))


def enabled_algorithm_kinds(algorithms: tuple[str, ...], no_catboost: bool = False) -> tuple[str, ...]:
    enabled: list[str] = []
    for algorithm in algorithms:
        if algorithm == "catboost" and (no_catboost or not _catboost_available()):
            continue
        if algorithm == "xgboost" and not _xgboost_available():
            continue
        enabled.append(KIND_BY_ALGORITHM[algorithm])
    return tuple(enabled)


def chronological_holdout_indices(y: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Último bloque como holdout, buscando un corte con ambas clases."""
    n_rows = len(y)
    if n_rows < 20:
        raise ValueError("Se necesitan al menos 20 filas para un holdout temporal.")
    target = max(1, min(n_rows - 1, int(round(n_rows * (1.0 - fraction)))))
    radius = max(10, int(round(n_rows * 0.1)))
    candidates = sorted(
        range(max(1, target - radius), min(n_rows, target + radius + 1)),
        key=lambda split: abs(split - target),
    )
    for split in candidates:
        train_idx = np.arange(split)
        holdout_idx = np.arange(split, n_rows)
        if len(train_idx) and len(holdout_idx) and len(np.unique(y[train_idx])) == 2 and len(np.unique(y[holdout_idx])) == 2:
            return train_idx, holdout_idx
    raise ValueError("No existe un corte temporal con ambas clases en train y holdout.")


def make_lgbm(monotone: list[int] | None = None, verbose: bool = False):
    runtime = get_runtime_config()
    configured = runtime.estimators.get("lightgbm", {})
    try:
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=int(configured.get("n_estimators", 2000)),
            learning_rate=float(configured.get("learning_rate", 0.02)),
            num_leaves=int(configured.get("num_leaves", 31)),
            max_depth=-1,
            min_child_samples=int(configured.get("min_child_samples", 80)),
            subsample=float(configured.get("subsample", 0.8)),
            subsample_freq=1,
            colsample_bytree=float(configured.get("colsample_bytree", 0.8)),
            reg_lambda=float(configured.get("reg_lambda", 5.0)),
            reg_alpha=0.0,
            objective="binary",
            n_jobs=-1,
            random_state=runtime.random_seed,
            verbosity=1 if verbose else -1,
        )
        if monotone is not None and any(monotone):
            params["monotone_constraints"] = monotone
        return LGBMClassifier(**params)
    except ModuleNotFoundError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        kwargs = dict(
            max_iter=600,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=runtime.random_seed,
        )
        if monotone is not None and any(monotone):
            kwargs["monotonic_cst"] = monotone
        return HistGradientBoostingClassifier(**kwargs)


def make_catboost(monotone: list[int] | None = None, verbose: bool = False):
    """CatBoost calibrado suele mejorar el log loss (ordered boosting, arboles
    simetricos). Opcional: si no esta instalado devuelve None y se omite el
    candidato (literatura: XGBoost/CatBoost/LightGBM boosting comparison)."""
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None
    runtime = get_runtime_config()
    configured = runtime.estimators.get("catboost", {})
    params = dict(
        iterations=int(configured.get("iterations", 2000)),
        learning_rate=float(configured.get("learning_rate", 0.02)),
        depth=int(configured.get("depth", 5)),
        l2_leaf_reg=float(configured.get("l2_leaf_reg", 6.0)),
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=runtime.random_seed,
        allow_writing_files=False,
        verbose=100 if verbose else False,
    )
    if monotone is not None and any(monotone):
        params["monotone_constraints"] = list(monotone)
    return CatBoostClassifier(**params)


def make_xgboost(monotone: list[int] | None = None, verbose: bool = False):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    runtime = get_runtime_config()
    configured = runtime.estimators.get("xgboost", {})
    params = dict(
        n_estimators=int(configured.get("n_estimators", 700)),
        learning_rate=float(configured.get("learning_rate", 0.02)),
        max_depth=int(configured.get("max_depth", 4)),
        min_child_weight=float(configured.get("min_child_weight", 20.0)),
        subsample=float(configured.get("subsample", 0.8)),
        colsample_bytree=float(configured.get("colsample_bytree", 0.8)),
        reg_lambda=float(configured.get("reg_lambda", 5.0)),
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=runtime.random_seed,
        verbosity=1 if verbose else 0,
    )
    if monotone is not None and any(monotone):
        params["monotone_constraints"] = tuple(monotone)
    return XGBClassifier(**params)


def make_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    runtime = get_runtime_config()
    configured = runtime.estimators.get("random_forest", {})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(configured.get("n_estimators", 600)),
                    max_features="sqrt",
                    min_samples_leaf=int(configured.get("min_samples_leaf", 12)),
                    n_jobs=-1,
                    random_state=runtime.random_seed,
                ),
            ),
        ]
    )


def make_logistic(c_value: float = 0.5):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    runtime = get_runtime_config()
    configured = runtime.estimators.get("logistic", {})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=int(configured.get("max_iter", 2000)),
                    C=float(c_value),
                    random_state=runtime.random_seed,
                ),
            ),
        ]
    )


def _new_estimator(
    kind: str,
    cols: list[str],
    verbose: bool = False,
    estimator_params: dict[str, Any] | None = None,
):
    estimator_params = estimator_params or {}
    if kind == "gbm":
        return make_lgbm(_monotone_vector(cols), verbose=verbose)
    if kind == "catboost":
        return make_catboost(_monotone_vector(cols), verbose=verbose)
    if kind == "xgboost":
        return make_xgboost(_monotone_vector(cols), verbose=verbose)
    if kind == "random_forest":
        return make_random_forest()
    if kind == "logistic":
        return make_logistic(estimator_params.get("C", 0.5))
    raise ValueError(f"Tipo de estimador desconocido: {kind}")


def _weight_key(est) -> str:
    """Nombre del kwarg de peso: las Pipelines lo enrutan al paso final (model)."""
    from sklearn.pipeline import Pipeline
    return f"{est.steps[-1][0]}__sample_weight" if isinstance(est, Pipeline) else "sample_weight"


def _fit_base(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, cols: list[str],
              random_state: int = 0, verbose: bool = False,
              sample_weight: np.ndarray | None = None,
              estimator_params: dict[str, Any] | None = None):
    """Ajusta el base con augmentacion por simetria y, en GBDT, early stopping
    sobre un holdout interno por log loss (evita fijar n_estimators a mano).

    `sample_weight` (alineado a X_tr) pondera cada fila; augment lo duplica para
    las filas espejo A<->B. Si el estimador no acepta pesos, cae a sin pesos.
    """
    est = _new_estimator(kind, cols, verbose=verbose, estimator_params=estimator_params)

    def _augw(idx: np.ndarray):
        if sample_weight is None:
            return None
        w = np.asarray(sample_weight)[idx]
        return np.concatenate([w, w])  # augment apila [X, X_rev] -> [w, w]

    if kind in ("gbm", "catboost", "xgboost"):
        try:
            tr, va = chronological_holdout_indices(y_tr, 0.15)
            Xf, yf = augment(X_tr[tr], y_tr[tr], cols)
            wf = _augw(tr)
            wkw = {"sample_weight": wf} if wf is not None else {}
            if kind == "gbm":
                try:
                    import lightgbm as lgb

                    est.fit(
                        Xf, yf,
                        eval_set=[(X_tr[va], y_tr[va])],
                        eval_metric="binary_logloss",
                        callbacks=[
                            lgb.early_stopping(80, verbose=verbose),
                            lgb.log_evaluation(50 if verbose else 0),
                        ],
                        **wkw,
                    )
                    return est
                except Exception:
                    pass  # HistGB fallback (early stopping interno) o firma distinta
            elif kind == "catboost":
                est.fit(
                    Xf, yf,
                    eval_set=(X_tr[va], y_tr[va]),
                    use_best_model=True,
                    verbose=100 if verbose else False,
                    **wkw,
                )
                return est
            else:  # xgboost
                est.fit(
                    Xf, yf,
                    eval_set=[(X_tr[va], y_tr[va])],
                    verbose=50 if verbose else False,
                    **wkw,
                )
                return est
        except Exception:
            pass
        est = _new_estimator(
            kind, cols, verbose=verbose, estimator_params=estimator_params
        )  # fallback robusto sin early stopping
        Xf, yf = augment(X_tr, y_tr, cols)
        wf = _augw(np.arange(len(X_tr)))
        try:
            est.fit(Xf, yf, **({"sample_weight": wf} if wf is not None else {}))
        except Exception:
            est.fit(Xf, yf)
        return est

    # logistic / random_forest (Pipeline)
    Xf, yf = augment(X_tr, y_tr, cols)
    wf = _augw(np.arange(len(X_tr)))
    if wf is not None:
        try:
            est.fit(Xf, yf, **{_weight_key(est): wf})
            return est
        except Exception:
            est = _new_estimator(kind, cols, verbose=verbose, estimator_params=estimator_params)
    est.fit(Xf, yf)
    return est


def _make_calibrator(base, X_cal: np.ndarray, y_cal: np.ndarray, method: str):
    if method == "beta":
        return BetaCalibratedClassifier(base).fit(X_cal, y_cal)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)


def fit_calibrated_multi(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, cols: list[str],
                         methods: tuple[str, ...] = CALIBRATION_METHODS,
                         cal_frac: float = 0.2, random_state: int = 0,
                         verbose: bool = False, sample_weight: np.ndarray | None = None,
                         estimator_params: dict[str, Any] | None = None):
    """Ajusta el base UNA vez y devuelve (base, {metodo: estimador_calibrado}).

    El base se entrena sobre tr_idx (augmentado) y cada calibrador sobre cal_idx.
    Compartir el base hace barato comparar sigmoid/isotonica/beta por fold.
    `sample_weight` (recencia) se aplica solo al base; el calibrador se ajusta sin
    pesos sobre el holdout reciente (ya sesgado a lo actual por ser cronologico).
    """
    tr_idx, cal_idx = chronological_holdout_indices(y_tr, cal_frac)
    sw_tr = np.asarray(sample_weight)[tr_idx] if sample_weight is not None else None
    base = _fit_base(kind, X_tr[tr_idx], y_tr[tr_idx], cols,
                     random_state=random_state, verbose=verbose, sample_weight=sw_tr,
                     estimator_params=estimator_params)
    cals: dict[str, Any] = {}
    for m in methods:
        try:
            cals[m] = _make_calibrator(base, X_tr[cal_idx], y_tr[cal_idx], m)
        except Exception:
            continue
    return base, cals


def fit_calibrated(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, cols: list[str],
                   cal_frac: float = 0.2, random_state: int = 0, method: str = "sigmoid",
                   verbose: bool = False):
    """Compat de un solo metodo (usado por Model B): devuelve (base, calibrado)."""
    base, cals = fit_calibrated_multi(kind, X_tr, y_tr, cols, methods=(method,),
                                      cal_frac=cal_frac, random_state=random_state,
                                      verbose=verbose)
    return base, (cals.get(method) or cals.get("sigmoid"))


def candidate_specs(kinds: tuple[str, ...]) -> dict[str, tuple[list[str], str]]:
    """name -> (kinds, método de calibración).

    Además de los candidatos históricos, `super_learner_cal` combina sus
    probabilidades con pesos convexos aprendidos sobre predicciones OOS previas.
    Esto evita tanto el voto uniforme arbitrario como pesos negativos inestables.
    """
    specs: dict[str, tuple[list[str], str]] = {
        INDIVIDUAL_CANDIDATE[kind]: ([kind], "sigmoid")
        for kind in kinds
    }
    if {"logistic", "gbm"}.issubset(kinds):
        specs.update(
            {
                "ensemble_cal": (["logistic", "gbm"], "sigmoid"),
                "ensemble_iso": (["logistic", "gbm"], "isotonic"),
                "ensemble_beta": (["logistic", "gbm"], "beta"),
            }
        )
    if {"logistic", "gbm", "catboost"}.issubset(kinds):
        specs["ensemble3_cal"] = (["logistic", "gbm", "catboost"], "sigmoid")
    if len(kinds) >= 2:
        specs["super_learner_cal"] = (list(kinds), "sigmoid")
    return specs


def optimize_convex_weights(probabilities: np.ndarray, y: np.ndarray, l2: float = 0.002) -> np.ndarray:
    """Pesos no negativos que suman uno y minimizan log loss regularizado."""
    n_models = probabilities.shape[1]
    equal = np.full(n_models, 1.0 / n_models)
    if n_models == 1 or len(y) == 0:
        return equal
    try:
        from scipy.optimize import minimize

        clipped = np.clip(probabilities, 1e-4, 1 - 1e-4)

        def objective(weights: np.ndarray) -> float:
            blended = np.clip(clipped @ weights, 1e-4, 1 - 1e-4)
            logloss = -np.mean(y * np.log(blended) + (1.0 - y) * np.log(1.0 - blended))
            return float(logloss + l2 * np.sum((weights - equal) ** 2))

        result = minimize(
            objective,
            equal,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
            options={"maxiter": 200, "ftol": 1e-10},
        )
        weights = np.asarray(result.x, dtype=float)
        if result.success and np.isfinite(weights).all() and np.all(weights >= -1e-9):
            weights = np.clip(weights, 0.0, 1.0)
            total = weights.sum()
            if total > 0:
                return weights / total
    except Exception:
        pass
    return equal


def super_learner_weights_for_candidates(
    preds: dict[str, list[dict[str, Any]]],
    component_names: list[str] | tuple[str, ...],
    min_history: int = SUPER_LEARNER_MIN_HISTORY,
) -> np.ndarray:
    rows = [preds.get(name, []) for name in component_names]
    equal = np.full(len(component_names), 1.0 / len(component_names))
    if not rows or any(len(row) != len(rows[0]) for row in rows) or len(rows[0]) < min_history:
        return equal
    match_ids = [row["match_id"] for row in rows[0]]
    if any([row["match_id"] for row in series] != match_ids for series in rows[1:]):
        return equal
    probabilities = np.column_stack(
        [[float(row["prob_team1"]) for row in series] for series in rows]
    )
    y = np.asarray([int(row["actual"]) for row in rows[0]], dtype=float)
    return optimize_convex_weights(probabilities, y)


def super_learner_component_names(
    kinds: list[str] | tuple[str, ...],
    rating_candidates: list[str] | tuple[str, ...] = (),
) -> list[str]:
    return [INDIVIDUAL_CANDIDATE[kind] for kind in kinds] + list(rating_candidates)


def super_learner_weights(
    preds: dict[str, list[dict[str, Any]]],
    kinds: list[str] | tuple[str, ...],
    min_history: int = SUPER_LEARNER_MIN_HISTORY,
) -> np.ndarray:
    """Backward-compatible wrapper for estimator-kind component names."""
    return super_learner_weights_for_candidates(
        preds,
        super_learner_component_names(kinds),
        min_history=min_history,
    )


def fit_probability_column_estimator(
    X: np.ndarray,
    y: np.ndarray,
    column_index: int,
    sample_weight: np.ndarray | None = None,
) -> ProbabilityColumnEstimator:
    """Platt-calibrate a causal rating probability with A/B symmetry."""
    from sklearn.linear_model import LogisticRegression

    raw = np.clip(np.nan_to_num(X[:, column_index], nan=0.0) + 0.5, 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    X_fit = np.vstack([logits, -logits])
    y_fit = np.concatenate([y, 1 - y])
    weights = None if sample_weight is None else np.concatenate([sample_weight, sample_weight])
    estimator = LogisticRegression(
        max_iter=1000, C=1.0, random_state=get_runtime_config().random_seed
    )
    estimator.fit(X_fit, y_fit, sample_weight=weights)
    return ProbabilityColumnEstimator(
        column_index=column_index,
        slope=float(estimator.coef_[0, 0]),
        intercept=float(estimator.intercept_[0]),
    )


def _proba(est, X: np.ndarray) -> np.ndarray:
    return np.clip(est.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)


def walk_forward(
    X_all: np.ndarray,
    y_all: np.ndarray,
    periods: np.ndarray,
    meta: list[dict[str, Any]],
    cols: list[str],
    warmup_weeks: int,
    min_train: int,
    gap: int = 0,
    algorithm_kinds: tuple[str, ...] = ("logistic", "gbm"),
    verbose: bool = False,
    recency_half_life: float = 0.0,
    optuna_trials: int = 0,
    optuna_retune_periods: int = 26,
    tuning_report: dict[str, Any] | None = None,
    learner_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Walk-forward semanal. Devuelve predicciones por modelo/candidato.

    `gap` deja periodos de separacion entre el fin del entrenamiento y el test
    (endurece contra fuga temporal). Los candidatos vienen de `candidate_specs`
    y se eligen despues por menor log loss (seguro por construccion).
    """
    preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uniq_periods = sorted(set(periods.tolist()))
    test_periods = uniq_periods[warmup_weeks:]

    elo_idx = cols.index("elo_prob_centered")
    glicko_idx = cols.index("glicko_prob_centered")
    learner_cols = list(learner_cols or cols)
    learner_indices = [cols.index(column) for column in learner_cols]

    kinds = tuple(dict.fromkeys(algorithm_kinds))
    specs = candidate_specs(kinds)
    rating_candidates = {
        name: cols.index(column)
        for name, column in RATING_CANDIDATE_COLUMNS.items()
        if column in cols
    }
    current_logistic_c = 0.5
    tuning_events: list[dict[str, Any]] = []

    for wi, period in enumerate(test_periods):
        train_mask = periods < (period - gap)
        test_mask = periods == period
        if train_mask.sum() < min_train or len(np.unique(y_all[train_mask])) < 2:
            if verbose:
                print(
                    f"      fold {wi+1:03d}/{len(test_periods):03d} period={period}: "
                    f"SKIP train={int(train_mask.sum())} test={int(test_mask.sum())}",
                    flush=True,
                )
            continue
        X_tr_full, y_tr = X_all[train_mask], y_all[train_mask]
        X_te_full, y_te = X_all[test_mask], y_all[test_mask]
        X_tr = X_tr_full[:, learner_indices]
        X_te = X_te_full[:, learner_indices]
        te_meta = [meta[i] for i in np.where(test_mask)[0]]
        base_rate = float(np.mean(y_tr))
        sw_tr = _recency_weights(periods[train_mask], recency_half_life)
        if (
            "logistic" in kinds
            and optuna_trials > 0
            and (not tuning_events or wi % max(1, optuna_retune_periods) == 0)
        ):
            tuning = tune_logistic_c_purged(
                X_tr,
                y_tr,
                periods[train_mask],
                learner_cols,
                set(learner_cols) & (set(DIFF_COLUMNS) | set(EXTENDED_DIFF_COLUMNS) | set(EXTRA_DIFF_COLUMNS)),
                n_trials=optuna_trials,
                gap=max(1, gap),
                seed=seed + wi,
                verbose=verbose,
            )
            tuning["outer_period"] = int(period)
            tuning["outer_train_rows"] = int(len(X_tr))
            tuning_events.append(tuning)
            if tuning.get("enabled"):
                current_logistic_c = float(tuning["best_c"])
            if verbose:
                print(
                    f"        Optuna purged: {'ON' if tuning.get('enabled') else 'OFF'} "
                    f"C={current_logistic_c:.5f} reason={tuning.get('reason')}",
                    flush=True,
                )
        if verbose:
            print(
                f"      fold {wi+1:03d}/{len(test_periods):03d} period={period}: "
                f"train={len(X_tr)} test={len(X_te)} kinds={','.join(kinds)}",
                flush=True,
            )

        # baselines (sin ajuste)
        elo_p = np.clip(X_te_full[:, elo_idx] + 0.5, 1e-4, 1 - 1e-4)
        glicko_p = np.clip(X_te_full[:, glicko_idx] + 0.5, 1e-4, 1 - 1e-4)

        # Ajusta cada tipo de base una vez (con sus calibradores) y reusa.
        fitted: dict[str, Any] = {}
        for kind in kinds:
            try:
                if verbose:
                    print(f"        fitting {kind}...", flush=True)
                fitted[kind] = fit_calibrated_multi(
                    kind, X_tr, y_tr, learner_cols, random_state=seed + wi, verbose=verbose,
                    sample_weight=sw_tr,
                    estimator_params={"C": current_logistic_c} if kind == "logistic" else None,
                )
            except Exception:
                fitted[kind] = None
                if verbose:
                    print(f"        fitting {kind}: FAILED", flush=True)

        rating_arrays: dict[str, np.ndarray] = {}
        for rating_name, column_index in rating_candidates.items():
            try:
                rating_estimator = fit_probability_column_estimator(
                    X_tr_full, y_tr, column_index, sample_weight=sw_tr
                )
                rating_arrays[rating_name] = _proba(rating_estimator, X_te_full)
            except Exception:
                if verbose:
                    print(f"        fitting {rating_name}: FAILED", flush=True)

        def _cand_pred(name: str, spec: tuple[list[str], str]) -> np.ndarray | None:
            est_kinds, method = spec
            arrs = []
            for k in est_kinds:
                fk = fitted.get(k)
                if not fk:
                    return None
                est = fk[1].get(method) or fk[1].get("sigmoid")
                if est is None:
                    return None
                arrs.append(_proba(est, X_te))
            if not arrs:
                return None
            if name == "super_learner_cal":
                rating_names = [candidate for candidate in rating_candidates if candidate in rating_arrays]
                arrs.extend(rating_arrays[candidate] for candidate in rating_names)
                component_names = super_learner_component_names(est_kinds, rating_names)
                weights = super_learner_weights_for_candidates(preds, component_names)
                if verbose:
                    text_weights = ", ".join(
                        f"{kind}={weight:.3f}" for kind, weight in zip(component_names, weights)
                    )
                    print(f"        super learner weights: {text_weights}", flush=True)
                return np.clip(np.average(np.vstack(arrs), axis=0, weights=weights), 1e-4, 1 - 1e-4)
            return np.clip(np.mean(arrs, axis=0), 1e-4, 1 - 1e-4)

        cand: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            arr = _cand_pred(name, spec)
            if arr is not None:
                cand[name] = arr
        cand.update(rating_arrays)

        for i, m in enumerate(te_meta):
            common = {
                "match_id": m["id"], "date": m["date"], "event": m.get("event"),
                "team1": m["team1"], "team2": m["team2"], "actual": int(y_te[i]),
                "format": m.get("format"),
                "environment": m.get("environment"),
                "stage": m.get("stage"),
                "event_tier": m.get("event_tier"),
                "patch_version": m.get("patch_version"),
                "map_pool_regime": m.get("map_pool_regime"),
            }
            preds["base_rate"].append({**common, "prob_team1": base_rate})
            preds["elo"].append({**common, "prob_team1": float(elo_p[i])})
            preds["glicko"].append({**common, "prob_team1": float(glicko_p[i])})
            for name, arr in cand.items():
                preds[name].append({**common, "prob_team1": float(arr[i])})
    if tuning_report is not None:
        tuning_report.update({
            "enabled": any(event.get("enabled") for event in tuning_events),
            "automatic": True,
            "objective": "nested_purged_inner_cv_log_loss",
            "requested_trials_per_study": int(max(0, optuna_trials)),
            "retune_periods": int(max(1, optuna_retune_periods)),
            "inner_gap": int(max(1, gap)),
            "events": tuning_events,
            "last_best_c": current_logistic_c,
        })
    return preds


def summarize(preds: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    out = {}
    for model, rows in preds.items():
        if not rows:
            continue
        y = np.array([r["actual"] for r in rows])
        p = np.array([r["prob_team1"] for r in rows])
        out[model] = metric_dict(y, p)
    return out


def segmented_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Métricas por banda de COMPETITIVIDAD (confianza del modelo).

    Responde a la pregunta honesta: ¿el acierto viene de palizas obvias o hay
    skill en los partidos parejos? El log loss de un coinflip puro es ln(2)≈0.693.
    """
    if not rows:
        return []
    y = np.array([r["actual"] for r in rows])
    p = np.clip(np.array([r["prob_team1"] for r in rows]), 1e-9, 1 - 1e-9)
    conf = np.maximum(p, 1 - p)
    bands = [
        ("coinflip (<55%)", conf < 0.55),
        ("parejos (55-65%)", (conf >= 0.55) & (conf < 0.65)),
        ("claros (65-80%)", (conf >= 0.65) & (conf < 0.80)),
        ("palizas (>=80%)", conf >= 0.80),
    ]
    out = []
    for label, mask in bands:
        n = int(mask.sum())
        if n == 0:
            continue
        yy, pp = y[mask], p[mask]
        out.append({
            "band": label,
            "n": n,
            "share": round(n / len(y), 4),
            "accuracy": round(float(np.mean((pp >= 0.5) == yy)), 4),
            "log_loss": round(float(np.mean(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))), 4),
        })
    return out


def segment_calibration(rows: list[dict[str, Any]], key: str = "format", min_n: int = 30) -> dict[str, Any]:
    """A3: metricas + ECE por SEGMENTO (default: formato BO1/3/5).

    Un ECE global bueno puede ocultar descalibracion por subgrupo. Segmentos con
    <min_n se marcan como no concluyentes. (tier/LAN-online se añaden cuando esas
    columnas de contexto esten activas.)
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "unknown")].append(r)
    out: dict[str, Any] = {}
    for g, rs in groups.items():
        if g.strip().lower() in {"", "unknown", "none"}:
            out[g or "unknown"] = {"n": len(rs), "note": "segmento desconocido; excluido de conclusiones"}
            continue
        if len(rs) < min_n:
            out[g] = {"n": len(rs), "note": f"muestra <{min_n}; no concluyente"}
            continue
        y = np.array([x["actual"] for x in rs])
        p = np.array([x["prob_team1"] for x in rs])
        out[g] = metric_dict(y, p)
    return out


def segment_calibration_suite(rows: list[dict[str, Any]], min_n: int = 30) -> dict[str, Any]:
    """A3: calibration audit across every pre-match segment we can observe."""
    dimensions = {
        "by_format": "format",
        "by_environment": "environment",
        "by_stage": "stage",
        "by_event_tier": "event_tier",
    }
    suite: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for label, key in dimensions.items():
        known = [row for row in rows if str(row.get(key) or "").strip().lower() not in {"", "unknown", "none"}]
        suite[label] = segment_calibration(rows, key, min_n=min_n)
        coverage[key] = {
            "known_rows": len(known),
            "total_rows": len(rows),
            "coverage": round(len(known) / len(rows), 4) if rows else 0.0,
            "min_group_rows": min_n,
        }
    suite["coverage"] = coverage
    return suite


def favorite_accuracy_bands(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy y calibracion del favorito puro en franjas de 10 puntos."""
    bands = [
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.01, "90-100%"),
    ]
    output: list[dict[str, Any]] = []
    total_correct = 0
    for low, high, label in bands:
        selected = []
        for row in rows:
            probability = float(row["prob_team1"])
            confidence = max(probability, 1.0 - probability)
            if low <= confidence < high:
                selected.append((row, confidence))
        n = len(selected)
        correct = sum(
            1
            for row, _confidence in selected
            if (float(row["prob_team1"]) >= 0.5) == bool(row["actual"])
        )
        total_correct += correct
        accuracy = correct / n if n else None
        average_probability = (
            sum(confidence for _row, confidence in selected) / n if n else None
        )
        output.append(
            {
                "band": label,
                "low": low,
                "high": min(high, 1.0),
                "n": n,
                "correct": correct,
                "accuracy": accuracy,
                "average_predicted_probability": average_probability,
                "calibration_gap": (
                    accuracy - average_probability
                    if accuracy is not None and average_probability is not None
                    else None
                ),
                "representative": n >= 30,
            }
        )
    total = len(rows)
    return {
        "n": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else None,
        "bands": output,
        "method": "walk_forward_model_favorite_10_point_bands",
    }


def market_benchmark(rows: list[dict[str, Any]], preds: dict[str, list[dict[str, Any]]],
                     prod_model: str) -> dict[str, Any]:
    """Compara el modelo contra la probabilidad implícita del mercado (odds de apertura).

    Solo para partidos con odds guardadas (hoy: pocos, del pipeline diario). Batir
    la cuota de apertura en log loss es señal de valor (PROJECT.md §8.3). Con n
    pequeño es ILUSTRATIVO, no concluyente.
    """
    odds_by_id = {r["id"]: r for r in rows if r.get("opening_odds_t1") is not None}
    if not odds_by_id:
        return {"n": 0, "note": "Sin partidos con odds de apertura guardadas todavía."}
    prod = {p["match_id"]: p for p in preds.get(prod_model, preds.get("logistic_cal", []))}
    ym, pm, pmkt = [], [], []
    for mid, r in odds_by_id.items():
        pr = prod.get(mid)
        if pr is None:
            continue
        ym.append(int(r["team1_win"]))
        pm.append(pr["prob_team1"])
        pmkt.append(float(r["opening_odds_t1"]))
    if len(ym) < 5:
        return {"n": len(ym), "note": "Muy pocos partidos con odds para concluir (ilustrativo)."}
    ym = np.array(ym); pm = np.clip(np.array(pm), 1e-9, 1 - 1e-9); pmkt = np.clip(np.array(pmkt), 1e-9, 1 - 1e-9)
    ll = lambda p: float(np.mean(-(ym * np.log(p) + (1 - ym) * np.log(1 - p))))
    return {
        "n": int(len(ym)),
        "model_log_loss": round(ll(pm), 4),
        "market_log_loss": round(ll(pmkt), 4),
        "model_accuracy": round(float(np.mean((pm >= 0.5) == ym)), 4),
        "market_accuracy": round(float(np.mean((pmkt >= 0.5) == ym)), 4),
        "note": "Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.",
    }


def economic_backtest(rows: list[dict[str, Any]], prod_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return run_economic_backtest(
        rows, prod_rows, get_runtime_config().economic_backtest
    )


def save_model_registry(
    artifact: ModelArtifact,
    metrics: dict,
    shap_rows: list,
    segments: list,
    market: dict,
    model_b: dict,
    economic: dict,
) -> dict[str, str]:
    trained_at = artifact.metadata.get("trained_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = trained_at.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "Z")
    target = MODEL_REGISTRY_DIR / stamp
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "model.pkl"
    artifact.save(artifact_path)
    metadata = {
        "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact": str(artifact_path),
        "feature_columns": artifact.feature_columns,
        "metadata": artifact.metadata,
        "metrics": metrics,
        "segments": segments,
        "market": market,
        "model_b": model_b,
        "economic_backtest": {k: v for k, v in economic.items() if k != "bets"},
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "shap_importance.json").write_text(json.dumps(shap_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "experiment_manifest.json").write_text(
        json.dumps(artifact.metadata.get("reproducibility") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path = artifact.metadata.get("effective_config_path") or artifact.metadata.get("config_path")
    if config_path and Path(config_path).exists():
        shutil.copyfile(config_path, target / "config.yaml")
    latest = {"latest": stamp, "artifact": str(artifact_path), "metadata": str(target / "metadata.json")}
    (MODEL_REGISTRY_DIR / "latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"model_registry_dir": str(target), "registered_model": str(artifact_path)}


def shap_importance(
    model, X: np.ndarray, cols: list[str], sample: int = 2000, seed: int = 42
) -> list[dict[str, Any]]:
    try:
        import shap
        idx = np.random.RandomState(seed).choice(len(X), size=min(sample, len(X)), replace=False)
        Xs = X[idx]
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xs)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, -1]
        mean_abs = np.mean(np.abs(vals), axis=0)
        rows = [{"feature": c, "mean_abs_shap": float(v)} for c, v in zip(cols, mean_abs)]
        rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        return rows
    except Exception:
        return []


def _safe_probability(value: Any) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 < p < 1.0:
        return None
    return p


def add_odds_features(X_dicts: list[dict[str, float]], rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    for feats, row in zip(X_dicts, rows):
        new = dict(feats)
        market_p = _safe_probability(row.get("opening_odds_t1"))
        bookmaker_count = row.get("opening_bookmaker_count") or 0
        try:
            bookmaker_count = float(bookmaker_count)
        except (TypeError, ValueError):
            bookmaker_count = 0.0
        new["opening_odds_prob_centered"] = (market_p - 0.5) if market_p is not None else np.nan
        new["opening_odds_confidence"] = abs(market_p - 0.5) if market_p is not None else np.nan
        new["opening_bookmaker_count_log"] = math.log1p(max(bookmaker_count, 0.0))
        out.append(new)
    return out


def model_b_eval(
    X_b: np.ndarray,
    y_all: np.ndarray,
    periods: np.ndarray,
    meta: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    model_a_rows: list[dict[str, Any]],
    model_b_columns: list[str],
    min_train_odds: int = 120,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Evalua Model B (stats + opening odds) solo cuando hay muestra suficiente.

    Se entrena de forma walk-forward sobre partidos que tienen cuota de apertura
    point-in-time. La cuota de cierre nunca entra como feature.
    """
    odds_values = np.array([
        _safe_probability(row.get("opening_odds_t1")) if row.get("opening_odds_t1") is not None else None
        for row in rows
    ], dtype=object)
    odds_mask = np.array([value is not None for value in odds_values], dtype=bool)
    odds_n = int(odds_mask.sum())
    if odds_n < min_train_odds + 5:
        return {
            "available": False,
            "n_odds_rows": odds_n,
            "min_train_odds": min_train_odds,
            "note": (
                "Modelo B preparado, pero aun no hay suficientes partidos con odds "
                "de apertura para validacion walk-forward fiable."
            ),
        }

    model_a_by_id = {row["match_id"]: row["prob_team1"] for row in model_a_rows}
    pred_rows: dict[str, list[dict[str, Any]]] = {
        "market_opening": [],
        "model_a_same_matches": [],
        "model_b_stats_plus_opening_odds": [],
    }

    for wi, period in enumerate(sorted(set(periods[odds_mask].tolist()))):
        train_mask = (periods < period) & odds_mask
        test_mask = (periods == period) & odds_mask
        if train_mask.sum() < min_train_odds or test_mask.sum() == 0:
            continue
        class_counts = np.bincount(y_all[train_mask], minlength=2)
        if class_counts.min() < 4:
            continue

        _base, calibrated = fit_calibrated(
            "logistic",
            X_b[train_mask],
            y_all[train_mask],
            model_b_columns,
            random_state=random_seed + 10_000 + wi,
        )
        test_indices = np.where(test_mask)[0]
        model_b_p = _proba(calibrated, X_b[test_indices])

        for local_i, global_i in enumerate(test_indices):
            m = meta[global_i]
            common = {
                "match_id": m["id"],
                "date": m["date"],
                "event": m.get("event"),
                "team1": m["team1"],
                "team2": m["team2"],
                "actual": int(y_all[global_i]),
            }
            market_p = float(odds_values[global_i])
            pred_rows["market_opening"].append({**common, "prob_team1": market_p})
            if m["id"] in model_a_by_id:
                pred_rows["model_a_same_matches"].append({**common, "prob_team1": float(model_a_by_id[m["id"]])})
            pred_rows["model_b_stats_plus_opening_odds"].append({**common, "prob_team1": float(model_b_p[local_i])})

    metrics = summarize(pred_rows)
    if not metrics:
        return {
            "available": False,
            "n_odds_rows": odds_n,
            "min_train_odds": min_train_odds,
            "note": (
                "Hay odds guardadas, pero todavia no hay folds walk-forward con "
                "suficiente entrenamiento previo."
            ),
        }

    segments = {name: segmented_eval(rs) for name, rs in pred_rows.items() if rs}
    shap_rows: list[dict[str, Any]] = []
    try:
        final_gbm = make_lgbm()
        odds_indices = np.where(odds_mask)[0]
        X_aug, y_aug = augment(X_b[odds_indices], y_all[odds_indices], model_b_columns)
        final_gbm.fit(X_aug, y_aug)
        shap_rows = shap_importance(final_gbm, X_b[odds_indices], model_b_columns, sample=1000)
    except Exception:
        shap_rows = []

    return {
        "available": True,
        "n_odds_rows": odds_n,
        "n_eval": int(metrics.get("model_b_stats_plus_opening_odds", {}).get("n", 0)),
        "min_train_odds": min_train_odds,
        "metrics": metrics,
        "feature_columns": model_b_columns,
        "segments": segments,
        "shap_importance": shap_rows,
        "note": "Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.",
    }


def paired_significance(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]],
                        n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    """A4: ¿es real la diferencia de log loss entre dos modelos, o es ruido?

    Empareja por match_id, calcula la diferencia de log loss POR PARTIDO (a-b),
    y devuelve CI95 bootstrap + Wilcoxon + MDE. `significant`=True si el intervalo
    excluye 0 (a mejor que b => diferencia media negativa y ci_high<0).
    """
    a = {r["match_id"]: r for r in rows_a}
    b = {r["match_id"]: r for r in rows_b}
    ids = [i for i in a if i in b]
    if len(ids) < 30:
        return {"n": len(ids), "note": "muestra insuficiente (<30) para significancia"}

    def _ll(r: dict[str, Any]) -> float:
        p = min(max(float(r["prob_team1"]), 1e-9), 1 - 1e-9)
        y = int(r["actual"])
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    d = np.array([_ll(a[i]) - _ll(b[i]) for i in ids], dtype=float)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    try:
        from scipy.stats import wilcoxon
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        wp = None
    mde = 2.80 * sd / math.sqrt(len(d)) if len(d) else None
    return {
        "n": len(ids),
        "mean_logloss_diff": round(mean_d, 5),
        "ci95_low": round(lo, 5),
        "ci95_high": round(hi, 5),
        "wilcoxon_p": (round(wp, 5) if wp is not None else None),
        "sigma_d": round(sd, 5),
        "mde_95_80": (round(mde, 5) if mde is not None else None),
        "significant": bool(hi < 0),
        "note": "a-b<0 y ci_high<0 => 'a' baja el log loss de forma significativa vs 'b'.",
    }


def main() -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default=os.environ.get("CS2_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )
    config_args, _ = config_parser.parse_known_args()
    runtime_config = load_config(config_args.config)
    set_runtime_config(runtime_config)
    training_defaults = runtime_config.training

    parser = argparse.ArgumentParser(description="Entrena el modelo CS2 (Glicko-2 + LightGBM + calibración).")
    parser.add_argument("--config", default=str(config_args.config),
                        help="Configuracion YAML versionada; la CLI tiene prioridad.")
    parser.add_argument("--raw", default="",
                        help="Compatibilidad: results_all.json. Si se omite, entrena desde BBDD/cs2.db.")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="BBDD viva usada como fuente por defecto.")
    parser.add_argument(
        "--master",
        default=str(ROOT / "PIPELINE" / "master" / "matches.json"),
        help="Master diario opcional que se combina con --raw; usa un JSON vacio para aislar un experimento.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--warmup-weeks", type=int, default=training_defaults.warmup_weeks)
    parser.add_argument("--min-train", type=int, default=training_defaults.min_train_rows)
    parser.add_argument("--no-cs2-filter", action="store_true")
    parser.add_argument("--form-half-life", type=float, default=training_defaults.form_half_life_days,
                        help="Vida media (dias) del decaimiento de la forma. Tunable.")
    parser.add_argument("--wf-gap", type=int, default=training_defaults.walk_forward_gap_periods,
                        help="Periodos de separacion train->test en walk-forward (anti-fuga).")
    parser.add_argument("--recency-half-life", type=float, default=training_defaults.recency_half_life_days,
                        help="A1: vida media (dias) del peso por recencia en el learner "
                             "(sample_weight, Dixon-Coles). 0 desactiva. Activo por defecto.")
    parser.add_argument("--no-catboost", action="store_true",
                        help="No usar CatBoost como candidato aunque este instalado.")
    parser.add_argument(
        "--algorithms",
        type=parse_algorithms,
        default=DEFAULT_ALGORITHMS,
        help="Lista separada por comas: logistic,lightgbm,catboost,xgboost,random_forest o all.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=("core", "error-aware"),
        default=training_defaults.feature_profile,
        help="core para la ablacion; error-aware añade consenso, margen y desacuerdo point-in-time.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar progreso detallado por fold y logs internos de LightGBM/CatBoost.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Guarda el artefacto en --output-dir, sin sobrescribir MODEL/artifacts/model.pkl ni registry.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=training_defaults.optuna_trials,
        help="B10: trials por estudio interno purgado; activo automaticamente (0 desactiva).",
    )
    parser.add_argument(
        "--optuna-retune-weeks",
        type=int,
        default=training_defaults.optuna_retune_weeks,
        help="B10: frecuencia causal de reoptimizacion durante walk-forward.",
    )
    parser.add_argument("--seed", type=int, default=runtime_config.random_seed,
                        help="Semilla global registrada con el experimento.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke determinista rapido; omite diagnosticos costosos.")
    args = parser.parse_args()

    if args.seed != runtime_config.random_seed:
        runtime_config = replace(runtime_config, random_seed=int(args.seed))
        set_runtime_config(runtime_config)
    set_global_determinism(args.seed)

    enabled_kinds = enabled_algorithm_kinds(args.algorithms, no_catboost=args.no_catboost)
    if not enabled_kinds:
        raise SystemExit("No hay algoritmos disponibles. Instala dependencias o cambia --algorithms.")
    has_catboost = "catboost" in enabled_kinds
    has_xgboost = "xgboost" in enabled_kinds

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective_config_path = out / "config.effective.yaml"
    effective_config = runtime_config.as_dict()
    effective_config["cli_arguments"] = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in vars(args).items()
    }
    effective_config_path.write_text(
        yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("[1/6] Cargando histórico…", flush=True)
    master_path = Path(args.master)
    raw_source = args.raw or None
    rows = dataio.load_training_rows(
        raw_source,
        master_path if master_path.exists() else None,
        cs2_only=not args.no_cs2_filter,
        db_path=args.db,
    )
    if not rows and not raw_source and DEFAULT_RAW.exists():
        print("      BBDD sin filas entrenables; fallback legacy results_all.json", flush=True)
        raw_source = str(DEFAULT_RAW)
        rows = dataio.load_training_rows(
            raw_source,
            master_path if master_path.exists() else None,
            cs2_only=not args.no_cs2_filter,
            db_path=None,
        )
    if not rows:
        raise SystemExit("No hay filas de entrenamiento. Inicializa BBDD o pasa --raw <results_all.json>.")
    reproducibility = experiment_manifest(rows, runtime_config, ROOT, vars(args))
    reproducibility["random_seed"] = int(args.seed)
    (out / "experiment_manifest.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_daily = sum(1 for r in rows if r.get("opening_odds_t1") is not None)
    source_label = f"DB {args.db}" if not raw_source else f"raw {raw_source}"
    print(f"      fuente: {source_label}")
    print(f"      {len(rows)} series ({rows[0]['date']} -> {rows[-1]['date']}) · con odds guardadas: {n_daily}")

    print("[2/6] Construyendo features point-in-time…", flush=True)
    t0 = time.time()
    X_dicts, y_list, meta, state = build_training_frame(rows, form_half_life=args.form_half_life)
    selected_columns, feature_policies = select_feature_columns(
        X_dicts, args.feature_profile, runtime_config.feature_thresholds
    )
    selected_matrix = _matrix(X_dicts, selected_columns)
    model_columns, exact_pruning = prune_exact_redundancies(
        selected_matrix,
        selected_columns,
        protected={"elo_prob_centered", "glicko_prob_centered"},
    )
    feature_policies["feature_pruning"] = {
        "available_rows": len(X_dicts),
        "min_rows": 0,
        "enabled": bool(exact_pruning["input_columns"] != exact_pruning["output_columns"]),
        "columns": list(model_columns),
        "activation": "automatic_target_free_exact_redundancy_only",
        **exact_pruning,
    }
    candidate_only_columns = set(BAYES_BT_FEATURE_COLUMNS) | set(KALMAN_FEATURE_COLUMNS)
    learner_columns = [column for column in model_columns if column not in candidate_only_columns]
    learner_indices = [model_columns.index(column) for column in learner_columns]
    for family in ("bayesian_bradley_terry", "kalman_state_space"):
        feature_policies[family]["production_scope"] = (
            "standalone_and_super_learner_candidate_not_generic_learner_matrix"
        )
    analytics_policy = feature_policies["analytics"]
    context_policy = feature_policies["context"]
    player_policy = feature_policies["player_snapshots"]
    model_b_columns = list(learner_columns) + ODDS_FEATURE_COLUMNS
    X_all = _matrix(X_dicts, model_columns)
    X_model_b_dicts = add_odds_features(X_dicts, rows)
    X_model_b = _matrix(X_model_b_dicts, model_b_columns)
    y_all = np.array(y_list, dtype=int)
    periods = np.array([_period_index(r.get("date_obj")) for r in rows])
    print(f"      {len(X_all)} filas, {len(model_columns)} features en {time.time()-t0:.1f}s")
    print("      activacion automatica de extended features:")
    for family, policy in feature_policies.items():
        detail = f"{policy['available_rows']}/{policy['min_rows']}"
        if family == "context":
            detail += (
                f"; LAN={policy['lan_rows']}/{policy['min_env_rows']}"
                f"; online={policy['online_rows']}/{policy['min_env_rows']}"
            )
        print(f"        {family:18s} {'ON' if policy['enabled'] else 'OFF':3s} ({detail})")

    print("[3/6] Walk-forward semanal…", flush=True)
    print(f"      candidatos: {'con' if has_catboost else 'sin'} CatBoost · half_life={args.form_half_life:.0f}d · wf_gap={args.wf_gap}")
    t0 = time.time()
    optuna_report: dict[str, Any] = {}
    preds = walk_forward(
        X_all, y_all, periods, meta, model_columns, args.warmup_weeks,
        args.min_train, gap=args.wf_gap, algorithm_kinds=enabled_kinds,
        verbose=args.verbose, recency_half_life=args.recency_half_life,
        optuna_trials=max(0, args.optuna_trials),
        optuna_retune_periods=max(1, args.optuna_retune_weeks),
        tuning_report=optuna_report,
        learner_cols=learner_columns,
        seed=args.seed,
    )
    metrics = summarize(preds)
    feature_policies["optuna_purged_cv"] = {
        "available_rows": len(X_all),
        "min_rows": 400,
        "enabled": bool(optuna_report.get("enabled")),
        "columns": [],
        "activation": "automatic_nested_purged_cv_log_loss",
        "requested_trials_per_study": max(0, args.optuna_trials),
        "retune_weeks": max(1, args.optuna_retune_weeks),
        "events": len(optuna_report.get("events") or []),
        "note": (
            "Optuna reoptimiza solo con periodos anteriores y embargo interno >=1 semana."
            if optuna_report.get("enabled") else
            "Optuna no tuvo folds internos suficientes o no esta instalado."
        ),
    }
    n_test = max((int(item.get("n", 0)) for item in metrics.values()), default=0)
    print(f"      hecho en {time.time()-t0:.1f}s; n_test={n_test}")
    specs = candidate_specs(enabled_kinds)
    active_rating_candidates = [name for name in RATING_CANDIDATE_COLUMNS if name in metrics]
    model_order = ["base_rate", "elo", "glicko"] + active_rating_candidates + list(specs.keys())
    for model in model_order:
        m = metrics.get(model)
        if m:
            print(f"      {model:16s} acc={m['accuracy']:.4f} logloss={m['log_loss']:.4f} "
                  f"brier={m['brier']:.4f} auc={m['roc_auc']:.4f} ece={m['ece_10']:.4f}")
    print(
        "      Optuna purged CV: "
        f"{'ON' if optuna_report.get('enabled') else 'OFF'} "
        f"studies={len(optuna_report.get('events') or [])} "
        f"last_C={optuna_report.get('last_best_c', 0.5):.5f}"
    )

    # Selección del modelo de producción por menor log loss walk-forward.
    candidates = {
        key: metrics[key]
        for key in list(specs) + active_rating_candidates
        if key in metrics
    }
    if not candidates:
        raise SystemExit("Sin candidatos evaluables en walk-forward (revisa min-train/warmup).")
    best_name = min(candidates, key=lambda k: candidates[k]["log_loss"])
    print(f"      -> modelo de producción elegido por log loss: {best_name}")

    # A4: significancia estadistica del modelo de produccion (bootstrap+Wilcoxon
    # pareado sobre log loss por-partido) vs baseline Glicko y vs el 2o mejor.
    ranked = sorted(candidates, key=lambda k: candidates[k]["log_loss"])
    second_best = ranked[1] if len(ranked) > 1 else None
    significance = {
        "production_model": best_name,
        "walk_forward_gap": args.wf_gap,
        "vs_glicko_baseline": paired_significance(
            preds.get(best_name, []), preds.get("glicko", []), seed=args.seed
        ),
    }
    if second_best:
        significance["vs_second_best"] = {
            "model": second_best,
            **paired_significance(
                preds.get(best_name, []), preds.get(second_best, []), seed=args.seed
            ),
        }
    _sg = significance["vs_glicko_baseline"]
    if _sg.get("n"):
        print(f"      significancia vs glicko: delta_logloss={_sg.get('mean_logloss_diff')} "
              f"ci95=[{_sg.get('ci95_low')},{_sg.get('ci95_high')}] p={_sg.get('wilcoxon_p')} "
              f"-> {'SIGNIFICATIVO' if _sg.get('significant') else 'no concluyente'}")

    # Evaluación segmentada por competitividad + benchmark de mercado.
    segments = segmented_eval(preds.get(best_name, []))
    favorite_accuracy = favorite_accuracy_bands(preds.get(best_name, []))
    segment_cal = segment_calibration_suite(preds.get(best_name, []))
    print("      calibracion por segmento:")
    for dimension in ("by_format", "by_environment", "by_stage", "by_event_tier"):
        conclusive = [
            f"{group}:n={metrics_row['n']},ll={metrics_row['log_loss']:.3f},ece={metrics_row['ece_10']:.3f}"
            for group, metrics_row in segment_cal[dimension].items()
            if metrics_row.get("log_loss") is not None
        ]
        print(f"        {dimension}: " + (" | ".join(conclusive) if conclusive else "sin muestra concluyente"))
    rich_target_eval = evaluate_rich_target_walk_forward(
        X_all,
        rows,
        periods,
        model_columns,
        set(DIFF_COLUMNS) | set(EXTENDED_DIFF_COLUMNS) | set(EXTRA_DIFF_COLUMNS),
        warmup_weeks=args.warmup_weeks,
        min_train=max(1000, args.min_train),
        gap=args.wf_gap,
        recency_half_life=args.recency_half_life,
        min_rows=int(runtime_config.feature_thresholds.get("rich_target_total", RICH_TARGET_MIN_ROWS)),
        min_class_rows=int(
            runtime_config.feature_thresholds.get("rich_target_per_class", RICH_TARGET_MIN_CLASS_ROWS)
        ),
        random_seed=args.seed,
    )
    rich_target_predictions = rich_target_eval.pop("predictions", [])
    if rich_target_predictions:
        binary_by_id = {row["match_id"]: row for row in preds.get(best_name, [])}
        comparable_binary = [binary_by_id[row["match_id"]] for row in rich_target_predictions if row["match_id"] in binary_by_id]
        comparable_rich = [row for row in rich_target_predictions if row["match_id"] in binary_by_id]
        rich_target_eval["winner_metrics"] = summarize({"rich_target": comparable_rich}).get("rich_target", {})
        rich_target_eval["winner_vs_binary_significance"] = paired_significance(
            comparable_rich, comparable_binary, seed=args.seed
        )
    rich_target_eval["columns"] = list(BO3_SCORE_CLASSES) if rich_target_eval.get("enabled") else []
    feature_policies["rich_target_bo3_scoreline"] = rich_target_eval
    print(
        "      target BO3 scoreline: "
        f"{'ON' if rich_target_eval.get('enabled') else 'OFF'} "
        f"({rich_target_eval.get('available_rows', 0)}/{rich_target_eval.get('min_rows', RICH_TARGET_MIN_ROWS)}; "
        f"wf={rich_target_eval.get('n_walk_forward', 0)})"
    )
    print("      por competitividad:")
    for s in segments:
        print(f"        {s['band']:18s} n={s['n']:5d} ({s['share']*100:4.1f}%) acc={s['accuracy']:.3f} logloss={s['log_loss']:.3f}")
    market = market_benchmark(rows, preds, best_name)
    if market.get("n"):
        print(f"      mercado (n={market['n']}): modelo logloss={market.get('model_log_loss')} vs mercado={market.get('market_log_loss')}")
    model_b = model_b_eval(
        X_model_b, y_all, periods, meta, rows, preds.get(best_name, []), model_b_columns,
        random_seed=args.seed,
    )
    feature_policies["opening_odds_model_b"] = {
        "available_rows": int(model_b.get("n_odds_rows", n_daily)),
        "min_rows": int(model_b.get("min_train_odds", 120)),
        "enabled": bool(model_b.get("available")),
        "columns": list(ODDS_FEATURE_COLUMNS) if model_b.get("available") else [],
        "activation": "automatic_separate_model_b_evaluation",
        "production_scope": "benchmark_and_operational_market_layer_not_model_a",
        "note": model_b.get("note", ""),
    }
    economic = economic_backtest(rows, preds.get(best_name, []))
    drift_report = build_drift_report(
        preds.get(best_name, []), rows, runtime_config.drift
    )
    if model_b.get("available"):
        mb = model_b["metrics"].get("model_b_stats_plus_opening_odds", {})
        print(
            f"      Model B stats+odds (n={model_b.get('n_eval')}): "
            f"acc={mb.get('accuracy', float('nan')):.4f} logloss={mb.get('log_loss', float('nan')):.4f}"
        )
    else:
        print(f"      Model B stats+odds: {model_b.get('note')} (n_odds={model_b.get('n_odds_rows')})")
    print(
        f"      backtest económico: bets={economic['n_bets']} "
        f"roi={economic['roi_on_staked'] if economic['roi_on_staked'] is not None else 'NA'} "
        f"drawdown={economic['max_drawdown']:.3f}"
    )
    print(
        f"      drift: {drift_report['status']} "
        f"warnings={','.join(drift_report['warnings']) or 'none'} "
        f"clv_n={drift_report['clv'].get('n', 0)}"
    )

    print("[4/6] Ajuste final sobre todo el histórico…", flush=True)
    if best_name in RATING_CANDIDATE_COLUMNS:
        best_kinds, best_method = [], "sigmoid"
        final_component_names = [best_name]
    else:
        best_kinds, best_method = specs[best_name]
        final_component_names = super_learner_component_names(best_kinds)
        if best_name == "super_learner_cal":
            final_component_names += active_rating_candidates
    fitted_full: dict[str, Any] = {}
    final_tuning = tune_logistic_c_purged(
        X_all[:, learner_indices],
        y_all,
        periods,
        learner_columns,
        set(learner_columns) & (set(DIFF_COLUMNS) | set(EXTENDED_DIFF_COLUMNS) | set(EXTRA_DIFF_COLUMNS)),
        n_trials=max(0, args.optuna_trials),
        gap=max(1, args.wf_gap),
        seed=args.seed,
        verbose=args.verbose,
    ) if "logistic" in best_kinds else {"enabled": False, "reason": "logistic_not_in_production"}
    optuna_report["final_full_history_tuning"] = final_tuning
    final_logistic_c = float(final_tuning.get("best_c", optuna_report.get("last_best_c", 0.5)))
    final_fit_kinds = set(best_kinds) | (set() if args.smoke else {"gbm"})
    for kind in sorted(final_fit_kinds):  # gbm adicional solo para SHAP en entreno completo
        if args.verbose:
            print(f"      fitting final {kind}...", flush=True)
        fitted_full[kind] = fit_calibrated_multi(
            kind, X_all[:, learner_indices], y_all, learner_columns,
            cal_frac=training_defaults.final_calibration_fraction, random_state=args.seed,
            verbose=args.verbose,
            sample_weight=_recency_weights(periods, args.recency_half_life),
            estimator_params={"C": final_logistic_c} if kind == "logistic" else None,
        )
    _component_kind = {
        "logistic": "logistic",
        "gbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
        "random_forest": "random_forest",
    }
    if best_name == "super_learner_cal":
        component_weights = super_learner_weights_for_candidates(
            preds, final_component_names, min_history=1
        )
    else:
        component_weights = np.full(len(final_component_names), 1.0 / len(final_component_names))
    print(
        "      pesos finales: " + ", ".join(
            f"{kind}={weight:.3f}" for kind, weight in zip(final_component_names, component_weights)
        )
    )
    components = []
    kind_by_candidate = {value: key for key, value in INDIVIDUAL_CANDIDATE.items()}
    full_weights = _recency_weights(periods, args.recency_half_life)
    for component_name, weight in zip(final_component_names, component_weights):
        if component_name in RATING_CANDIDATE_COLUMNS:
            column_index = model_columns.index(RATING_CANDIDATE_COLUMNS[component_name])
            estimator = fit_probability_column_estimator(
                X_all, y_all, column_index, sample_weight=full_weights
            )
            components.append(Component(component_name, estimator, None, float(weight)))
            continue
        kind = kind_by_candidate[component_name]
        est = fitted_full[kind][1].get(best_method) or fitted_full[kind][1].get("sigmoid")
        wrapped = ColumnSubsetEstimator(estimator=est, column_indices=learner_indices)
        components.append(Component(_component_kind.get(kind, kind), wrapped, None, float(weight)))

    print("[5/6] Importancia SHAP…", flush=True)
    gbm_base = fitted_full.get("gbm", (None,))[0]
    shap_rows = [] if args.smoke else shap_importance(
        gbm_base, X_all[:, learner_indices], learner_columns, seed=args.seed
    )
    pruning_diagnostics = (
        {"available": False, "reason": "smoke_mode"}
        if args.smoke else feature_pruning_diagnostics(X_all, y_all, model_columns, shap_rows)
    )
    directional_columns = sorted(
        set(model_columns) & (set(DIFF_COLUMNS) | set(EXTENDED_DIFF_COLUMNS) | set(EXTRA_DIFF_COLUMNS))
    )
    rich_target_model = None
    if rich_target_eval.get("enabled"):
        rich_target_model = fit_rich_target_model(
            X_all,
            rows,
            model_columns,
            directional_columns,
            sample_weight=_recency_weights(periods, args.recency_half_life),
            random_state=args.seed,
        )

    print("[6/6] Guardando artefacto y resultados…", flush=True)
    artifact = ModelArtifact(
        feature_columns=list(model_columns),
        components=components,
        metadata={
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_source": str(raw_source or ""),
            "db_source": str(args.db) if not raw_source else "",
            "n_train_rows": int(len(X_all)),
            "date_min": rows[0]["date"],
            "date_max": rows[-1]["date"],
            "cs2_only": not args.no_cs2_filter,
            "reproducibility": reproducibility,
            "config_version": runtime_config.version,
            "config_sha256": runtime_config.sha256,
            "config_path": str(Path(args.config).resolve()),
            "effective_config_path": str(effective_config_path.resolve()),
            "random_seed": int(args.seed),
            "analytics_features": analytics_policy,
            "context_features": context_policy,
            "player_snapshot_features": player_policy,
            "map_box_score_features": feature_policies["map_box_scores"],
            "ranking_features": feature_policies["rankings"],
            "roster_features": feature_policies["roster"],
            "feature_policies": feature_policies,
            "feature_pruning_diagnostics": pruning_diagnostics,
            "rich_target_bo3_scoreline": rich_target_eval,
            "optuna_tuning": optuna_report,
            "walk_forward_metrics": metrics,
            "significance": significance,
            "segment_calibration": segment_cal,
            "segmented_eval": segments,
            "market_benchmark": market,
            "model_b": model_b,
            "economic_backtest": {k: v for k, v in economic.items() if k != "bets"},
            "drift_monitoring": drift_report,
            "production_model": best_name,
            "model": "Glicko-2 features + " + {
                "ensemble_cal": "LightGBM ⊕ Logística (Platt)",
                "ensemble_iso": "LightGBM ⊕ Logística (isotónica)",
                "ensemble_beta": "LightGBM ⊕ Logística (beta)",
                "ensemble3_cal": "LightGBM ⊕ Logística ⊕ CatBoost (Platt)",
                "logistic_cal": "Logística (Platt)",
                "lightgbm_cal": "LightGBM (Platt)",
                "catboost_cal": "CatBoost (Platt)",
                "xgboost_cal": "XGBoost (Platt)",
                "random_forest_cal": "Random Forest (Platt)",
                "super_learner_cal": "Super Learner temporal (pesos convexos, Platt)",
                "bayesian_bt_cal": "Bradley-Terry bayesiano (partial pooling, Platt)",
                "kalman_cal": "Kalman state-space (Platt)",
            }.get(best_name, best_name),
            "production_calibration": best_method,
            "production_components": [c.name for c in components],
            "production_component_weights": {c.name: float(c.weight) for c in components},
            "algorithm_request": list(args.algorithms),
            "algorithms_enabled": list(enabled_kinds),
            "feature_profile": args.feature_profile,
            "generic_learner_feature_columns": learner_columns,
            "candidate_only_feature_columns": sorted(candidate_only_columns & set(model_columns)),
            "inner_split": "chronological_holdout",
            "form_half_life_days": args.form_half_life,
            "recency_half_life_days": args.recency_half_life,
            "walk_forward_gap": args.wf_gap,
            "catboost_enabled": has_catboost,
            "xgboost_enabled": has_xgboost,
            "monotone_features": sorted(MONOTONE_INCREASING),
            "directional_feature_columns": directional_columns,
            "period_days": 7,
        },
        rich_target_estimator=rich_target_model,
        rich_target_classes=list(BO3_SCORE_CLASSES) if rich_target_model is not None else [],
    )
    favorite_accuracy.update(
        {
            "model": best_name,
            "trained_at": artifact.metadata["trained_at"],
        }
    )
    artifact.metadata["favorite_accuracy_bands"] = favorite_accuracy
    # predicciones y calibración del mejor modelo
    cal_rows = []
    for model, rs in preds.items():
        if not rs:
            continue
        y = np.array([r["actual"] for r in rs])
        p = np.array([r["prob_team1"] for r in rs])
        for b in calibration_bins(y, p, 10):
            cal_rows.append({"model": model, **b})

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "significance.json").write_text(json.dumps(significance, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "segment_calibration.json").write_text(json.dumps(segment_cal, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "favorite_accuracy_bands.json").write_text(
        json.dumps(favorite_accuracy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "shap_importance.json").write_text(json.dumps(shap_rows, indent=2), encoding="utf-8")
    (out / "feature_pruning.json").write_text(
        json.dumps(pruning_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "optuna_tuning.json").write_text(
        json.dumps(optuna_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "rich_target_bo3.json").write_text(
        json.dumps(rich_target_eval, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "rich_target_bo3_predictions.json").write_text(
        json.dumps(rich_target_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "model_b_eval.json").write_text(json.dumps(model_b, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "economic_backtest.json").write_text(json.dumps(economic, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "drift_report.json").write_text(
        json.dumps(drift_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "segmented_eval.json").write_text(
        json.dumps(
            {
                "segments": segments,
                "favorite_accuracy": favorite_accuracy,
                "market": market,
                "model_b": model_b,
                "economic_backtest": {k: v for k, v in economic.items() if k != "bets"},
                "drift": drift_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    import csv

    with open(out / "predictions_walkforward.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "match_id", "date", "event", "team1", "team2", "actual", "prob_team1",
            "format", "environment", "stage", "event_tier", "patch_version", "map_pool_regime",
        ])
        for model, rs in preds.items():
            for r in rs:
                w.writerow([
                    model, r["match_id"], r["date"], r.get("event"), r["team1"], r["team2"],
                    r["actual"], r["prob_team1"], r.get("format"), r.get("environment"),
                    r.get("stage"), r.get("event_tier"), r.get("patch_version"), r.get("map_pool_regime"),
                ])
    with open(out / "calibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "bin", "low", "high", "n", "avg_pred", "observed"])
        w.writeheader()
        w.writerows(cal_rows)

    artifact_output = out / "model.pkl" if args.no_promote else ARTIFACT_PATH
    if args.no_promote:
        registry = {
            "promoted": False,
            "registered_model": str(artifact_output),
            "model_registry_dir": str(out),
            "note": "--no-promote: verificacion/local, no toca artefacto de produccion.",
        }
    else:
        registry = save_model_registry(artifact, metrics, shap_rows, segments, market, model_b, economic)
    artifact.metadata["registry"] = registry
    artifact.save(artifact_output)
    _write_report(out, metrics, shap_rows, rows, artifact, segments, market, model_b, economic)
    print(f"      artefacto: {artifact_output}")
    print(f"      resultados: {out}")
    return 0


def _write_report(out: Path, metrics: dict, shap_rows: list, rows: list, artifact: ModelArtifact,
                  segments: list, market: dict, model_b: dict, economic: dict) -> None:
    def row(m, name):
        d = metrics.get(m)
        if not d:
            return ""
        return (f"| {name} | {d['n']} | {d['accuracy']:.4f} | {d['log_loss']:.4f} | "
                f"{d['brier']:.4f} | {d['roc_auc']:.4f} | {d['ece_10']:.4f} |\n")

    feature_policies = artifact.metadata.get("feature_policies") or {}
    policy_lines = [
        "| Familia | Estado | Cobertura | Umbral | Activacion |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for family, policy in feature_policies.items():
        coverage = str(policy.get("available_rows", 0))
        if family == "context":
            coverage += f" (LAN {policy.get('lan_rows', 0)}, online {policy.get('online_rows', 0)})"
        policy_lines.append(
            f"| {family} | {'ON' if policy.get('enabled') else 'OFF'} | {coverage} | "
            f"{policy.get('min_rows', 0)} | {policy.get('activation', 'automatic_at_training_time')} |\n"
        )
    lines = [
        "# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)\n",
        f"\nGenerado: {artifact.metadata['trained_at']}\n",
        f"Histórico: {rows[0]['date']} → {rows[-1]['date']} ({len(rows)} series, era CS2)\n",
        "\n## Politica de features opcionales\n\n",
        *policy_lines,
        "\n## Resultados walk-forward (ventana expansiva, paso semanal)\n\n",
        "| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
        row("base_rate", "Base rate (baseline)"),
        row("elo", "Elo (baseline)"),
        row("glicko", "Glicko-2 (baseline)"),
        row("bayesian_bt_cal", "Bradley-Terry bayesiano (Platt)"),
        row("kalman_cal", "Kalman state-space (Platt)"),
        row("logistic_cal", "Logística (Platt)"),
        row("lightgbm_cal", "LightGBM (Platt)"),
        row("catboost_cal", "CatBoost (Platt)"),
        row("xgboost_cal", "XGBoost (Platt)"),
        row("random_forest_cal", "Random Forest (Platt)"),
        row("ensemble_cal", "Ensemble LGBM⊕Log (Platt)"),
        row("ensemble_iso", "Ensemble (isotónica)"),
        row("ensemble_beta", "Ensemble (beta)"),
        row("ensemble3_cal", "**Ensemble +CatBoost (Platt)**"),
        row("super_learner_cal", "**Super Learner temporal (Platt)**"),
        f"\n> Modelo de producción elegido por menor log loss: **{artifact.metadata.get('production_model')}**.\n",
        "\n> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.\n",
        "> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.\n",
        "\n## Importancia de features (SHAP, |valor| medio)\n\n",
        "| Feature | mean |SHAP| |\n|---|---:|\n",
    ]
    for r in shap_rows[:15]:
        lines.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} |\n")

    pruning = artifact.metadata.get("feature_pruning_diagnostics") or {}
    exact_pruning = feature_policies.get("feature_pruning") or {}
    lines.append("\n## Poda y multicolinealidad (A5)\n\n")
    lines.append(
        f"- Poda automatica sin target: {exact_pruning.get('input_columns', 0)} -> "
        f"{exact_pruning.get('output_columns', len(artifact.feature_columns))} columnas.\n"
        f"- Constantes eliminadas: {len(exact_pruning.get('removed_constants') or [])}.\n"
        f"- Duplicados exactos o de signo opuesto eliminados: "
        f"{len(exact_pruning.get('removed_exact_redundancies') or [])}.\n"
    )
    if pruning.get("available"):
        candidates = pruning.get("review_drop_candidates") or []
        lines.append(
            f"- Diagnostico temporal: VIF + permutation importance en los ultimos "
            f"{pruning.get('holdout_rows', 0)} casos + RFE + contraste SHAP.\n"
            f"- Candidatas para revision: {', '.join(candidates) if candidates else 'ninguna'}.\n\n"
            "> Las sugerencias supervisadas son auditoria, no poda automatica: para retirar una "
            "feature deben ganar una validacion walk-forward anidada.\n"
        )
    else:
        lines.append(f"\n> Diagnostico no disponible: {pruning.get('reason', 'sin resultado')}.\n")

    segment_calibration_result = artifact.metadata.get("segment_calibration") or {}
    lines.append("\n## Calibracion por segmento (A3)\n\n")
    lines.append("| Dimension | Segmento | N | Accuracy | Log loss | Brier | ECE | Estado |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|\n")
    for dimension in ("by_format", "by_environment", "by_stage", "by_event_tier"):
        for segment, values in (segment_calibration_result.get(dimension) or {}).items():
            if values.get("note"):
                lines.append(
                    f"| {dimension.removeprefix('by_')} | {segment} | {values.get('n', 0)} | "
                    f"- | - | - | - | {values['note']} |\n"
                )
                continue
            lines.append(
                f"| {dimension.removeprefix('by_')} | {segment} | {values.get('n', 0)} | "
                f"{values.get('accuracy', 0.0):.4f} | {values.get('log_loss', 0.0):.4f} | "
                f"{values.get('brier', 0.0):.4f} | {values.get('ece_10', 0.0):.4f} | concluyente |\n"
            )
    lines.append("\nCobertura conocida: ")
    coverage_parts = []
    for dimension, values in (segment_calibration_result.get("coverage") or {}).items():
        coverage_parts.append(
            f"{dimension} {values.get('known_rows', 0)}/{values.get('total_rows', 0)} "
            f"({100 * float(values.get('coverage', 0.0)):.1f}%)"
        )
    lines.append(("; ".join(coverage_parts) if coverage_parts else "sin filas evaluables") + ".\n")

    rich_target = artifact.metadata.get("rich_target_bo3_scoreline") or {}
    lines.append("\n## Target BO3 enriquecido (A6)\n\n")
    lines.append(
        f"- Estado: {'ON' if rich_target.get('enabled') else 'OFF'}.\n"
        f"- Cobertura BO3: {rich_target.get('available_rows', 0)}/{rich_target.get('min_rows', 0)}; "
        f"minimo por clase: {rich_target.get('min_class_rows', 0)}.\n"
        f"- Clases: {rich_target.get('class_counts') or {}}.\n"
        f"- Scope: {rich_target.get('production_scope', 'scoreline auxiliar')}.\n"
    )
    if rich_target.get("n_walk_forward"):
        lines.append(
            f"- Walk-forward: n={rich_target.get('n_walk_forward')}; log loss multiclase="
            f"{rich_target.get('multiclass_log_loss')}; baseline empirico="
            f"{rich_target.get('empirical_baseline_log_loss')}; accuracy scoreline="
            f"{rich_target.get('scoreline_accuracy')}; accuracy ganador="
            f"{rich_target.get('winner_accuracy')}.\n"
        )
    lines.append(f"\n> {rich_target.get('note', 'pendiente de evaluar')}.\n")

    optuna_result = artifact.metadata.get("optuna_tuning") or {}
    map_policy = feature_policies.get("bo3_map_compositional") or {}
    lines.append("\n## Ratings y optimizacion B8-B11\n\n")
    lines.append(
        f"- B8 Bradley-Terry bayesiano: {'ON' if (feature_policies.get('bayesian_bradley_terry') or {}).get('enabled') else 'OFF'}; "
        "partial pooling gaussiano, incertidumbre predictiva y candidato calibrado.\n"
        f"- B9 Kalman state-space: {'ON' if (feature_policies.get('kalman_state_space') or {}).get('enabled') else 'OFF'}; "
        "deriva de estado, varianza y candidato calibrado.\n"
        f"- B10 Optuna purgado: {'ON' if optuna_result.get('enabled') else 'OFF'}; "
        f"estudios={len(optuna_result.get('events') or [])}, C final="
        f"{(optuna_result.get('final_full_history_tuning') or {}).get('best_c', optuna_result.get('last_best_c'))}.\n"
        f"- B11 BO3 composicional: {'ON' if map_policy.get('enabled') else 'OFF'}; "
        f"cobertura={map_policy.get('available_rows', 0)}/{map_policy.get('min_rows', 0)}.\n"
    )

    lines.append("\n## Acierto por competitividad (¿hay skill o son palizas?)\n\n")
    lines.append("| Banda (confianza) | N | % | Accuracy | Log loss |\n|---|---:|---:|---:|---:|\n")
    for s in segments:
        lines.append(f"| {s['band']} | {s['n']} | {s['share']*100:.1f}% | {s['accuracy']:.3f} | {s['log_loss']:.3f} |\n")
    lines.append("\n> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra "
                 "en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.\n")
    favorite_accuracy = artifact.metadata.get("favorite_accuracy_bands") or {}
    lines.append("\n## Accuracy del favorito por probabilidad predicha\n\n")
    lines.append("| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for band in favorite_accuracy.get("bands", []):
        accuracy = band.get("accuracy")
        average = band.get("average_predicted_probability")
        gap = band.get("calibration_gap")
        if accuracy is None or average is None or gap is None:
            lines.append(f"| {band['band']} | 0 | 0 | - | - | - |\n")
            continue
        lines.append(
            f"| {band['band']} | {band['correct']} | {band['n']} | "
            f"{accuracy:.4f} | {average:.4f} | {gap:+.4f} |\n"
        )
    lines.append(
        f"| **Total** | **{favorite_accuracy.get('correct', 0)}** | "
        f"**{favorite_accuracy.get('n', 0)}** | "
        f"**{favorite_accuracy.get('accuracy', 0.0):.4f}** | - | - |\n"
    )
    lines.append("\n## Benchmark de mercado (odds de apertura)\n\n")
    if market.get("n"):
        lines.append(f"Sobre {market['n']} partidos con odds guardadas (ILUSTRATIVO, n pequeño):\n\n")
        lines.append(f"- Modelo: log loss {market.get('model_log_loss')} · accuracy {market.get('model_accuracy')}\n")
        lines.append(f"- Mercado (odds apertura): log loss {market.get('market_log_loss')} · accuracy {market.get('market_accuracy')}\n")
        lines.append(f"\n> {market.get('note')}\n")
    else:
        lines.append(f"> {market.get('note','Aún sin odds suficientes.')} A medida que el pipeline diario "
                     "acumule odds, este benchmark se llena solo.\n")

    lines.append("\n## Modelo B (stats + odds de apertura)\n\n")
    if model_b.get("available"):
        lines.append(f"Validación walk-forward solo en partidos con odds de apertura (n_eval={model_b.get('n_eval')}):\n\n")
        lines.append("| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        labels = {
            "market_opening": "Mercado apertura",
            "model_a_same_matches": "Model A en mismos partidos",
            "model_b_stats_plus_opening_odds": "Model B stats + apertura",
        }
        for key, label in labels.items():
            d = (model_b.get("metrics") or {}).get(key)
            if not d:
                continue
            lines.append(
                f"| {label} | {d['n']} | {d['accuracy']:.4f} | {d['log_loss']:.4f} | "
                f"{d['brier']:.4f} | {d['roc_auc']:.4f} | {d['ece_10']:.4f} |\n"
            )
        b_segments = (model_b.get("segments") or {}).get("model_b_stats_plus_opening_odds") or []
        if b_segments:
            lines.append("\nBanda de competitividad para Model B:\n\n")
            lines.append("| Banda | N | % | Accuracy | Log loss |\n|---|---:|---:|---:|---:|\n")
            for s in b_segments:
                lines.append(f"| {s['band']} | {s['n']} | {s['share']*100:.1f}% | {s['accuracy']:.3f} | {s['log_loss']:.3f} |\n")
        b_shap = model_b.get("shap_importance") or []
        if b_shap:
            lines.append("\nSHAP de Model B (top 10):\n\n")
            lines.append("| Feature | mean |SHAP| |\n|---|---:|\n")
            for r in b_shap[:10]:
                lines.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} |\n")
        lines.append(f"\n> {model_b.get('note')}\n")
    else:
        lines.append(
            f"> {model_b.get('note')} n_odds={model_b.get('n_odds_rows')} "
            f"(mínimo de entrenamiento: {model_b.get('min_train_odds')}). "
            "No se calcula SHAP de Model B hasta que haya validación walk-forward fiable.\n"
        )

    lines.append("\n## Backtest economico\n\n")
    lines.append(
        f"- Apuestas simuladas: {economic.get('n_bets')}\n"
        f"- Hit rate: {economic.get('hit_rate')}\n"
        f"- ROI sobre stake: {economic.get('roi_on_staked')}\n"
        f"- Banca: {economic.get('initial_bankroll')} -> {economic.get('final_bankroll')}\n"
        f"- Profit sobre banca inicial: {economic.get('profit_fraction_start_bankroll')}\n"
        f"- Max drawdown: {economic.get('max_drawdown')}\n"
        f"- Vig medio de apertura: {economic.get('average_opening_vig')}\n"
        f"- CLV: n={(economic.get('clv') or {}).get('n')}; cobertura="
        f"{(economic.get('clv') or {}).get('coverage')}; media precio="
        f"{(economic.get('clv') or {}).get('mean_price_clv')}; mediana="
        f"{(economic.get('clv') or {}).get('median_price_clv')}\n"
        f"- Apuestas limitadas por stake/payout: "
        f"{(economic.get('limits') or {}).get('constrained_bets')}\n"
        f"- Staking: {economic.get('staking')}\n\n"
        f"> {economic.get('note')}\n"
    )

    drift = artifact.metadata.get("drift_monitoring") or {}
    lines.append("\n## Drift causal (C12)\n\n")
    lines.append(
        f"- Estado: **{drift.get('status', 'unknown')}**.\n"
        f"- Predicciones OOS cerradas: {drift.get('n_predictions', 0)}.\n"
        f"- Alertas: {', '.join(drift.get('warnings') or []) or 'ninguna'}.\n"
        f"- Log loss ventana actual: {(drift.get('log_loss') or {}).get('current_mean')}; "
        f"referencia: {(drift.get('log_loss') or {}).get('reference_mean')}.\n"
        f"- CLV evaluable: {(drift.get('clv') or {}).get('n', 0)} "
        f"({100 * float((drift.get('clv') or {}).get('coverage', 0.0)):.1f}%); "
        f"media: {(drift.get('clv') or {}).get('current_mean')}.\n\n"
        "> Page-Hinkley se aplica al log loss y a -CLV en orden temporal. Las cuotas de cierre "
        "solo auditan la ejecucion y nunca son features.\n"
    )

    registry = artifact.metadata.get("registry") or {}
    if registry:
        lines.append("\n## Model registry\n\n")
        lines.append(f"- Artefacto versionado: `{registry.get('registered_model')}`\n")
        lines.append(f"- Carpeta: `{registry.get('model_registry_dir')}`\n")

    lines.append(
        "\n## Notas metodológicas\n\n"
        "- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).\n"
        "- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).\n"
        "- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; "
        "se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.\n"
        "- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.\n"
        "- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.\n"
        "- Strength-of-schedule usa Elo del rival y rendimiento real menos esperado, ambos calculados estrictamente antes del partido.\n"
        "- El modelo A sigue prediciendo el ganador de la serie. El sidecar BO3 estima 0-2/1-2/2-1/2-0 "
        "y solo se publica si mejora el baseline multiclase en walk-forward. El modelo por mapa queda pendiente de mas mapstats point-in-time.\n"
    )
    (out / "REPORT.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

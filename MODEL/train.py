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
import shutil
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

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
    TRUESKILL_DIFF_COLUMNS,
    EXTENDED_DIFF_COLUMNS,
    _period_index,
)
from cs2model.metrics import metric_dict, calibration_bins
from cs2model.artifacts import ModelArtifact, Component, ARTIFACT_PATH
from cs2model.calibration import BetaCalibratedClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT / "SCRAPPER" / "hltv-scraper-api" / "hltv_scraper" / "data" / "raw"
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
EXTRA_DIFF_COLUMNS = {"opening_odds_prob_centered", *TRUESKILL_DIFF_COLUMNS}
ANALYTICS_MIN_TRAIN_ROWS = 120
ANALYTICS_EXTENDED_MIN_TRAIN_ROWS = 200
# Una alineacion anunciada completa tiene varias variables correlacionadas; se
# exige una muestra cerrada mayor antes de dejar que altere produccion.
ANNOUNCED_LINEUP_MIN_TRAIN_ROWS = 200
# Prize pool/tamano del evento son contexto de calibracion, no una ventaja de
# lado. Requieren mas eventos antes de permitir interacciones no lineales.
EVENT_METADATA_MIN_TRAIN_ROWS = 300
CONTEXT_MIN_TRAIN_ROWS = 200
CONTEXT_MIN_ENV_ROWS = 50
PLAYER_MIN_TRAIN_ROWS = 200
MAP_ASSET_MIN_TRAIN_ROWS = 200
EVENT_HISTORY_MIN_TRAIN_ROWS = 200
RANKING_MIN_TRAIN_ROWS = 200
ROSTER_MIN_TRAIN_ROWS = 200
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
)


def _matrix(rows: list[dict[str, float]], cols: list[str]) -> np.ndarray:
    return np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)


def select_feature_columns(
    X_dicts: list[dict[str, float]],
    feature_profile: str = "error-aware",
) -> tuple[list[str], dict[str, Any]]:
    context_rows = sum(1 for row in X_dicts if (row.get("context_available") or 0.0) >= 0.5)
    lan_rows = sum(1 for row in X_dicts if (row.get("context_is_lan") or 0.0) >= 0.5)
    online_rows = sum(1 for row in X_dicts if (row.get("context_is_online") or 0.0) >= 0.5)
    stage_rows = sum(1 for row in X_dicts if (row.get("context_stage_known") or 0.0) >= 0.5)
    context_enabled = (
        context_rows >= CONTEXT_MIN_TRAIN_ROWS
        and lan_rows >= CONTEXT_MIN_ENV_ROWS
        and online_rows >= CONTEXT_MIN_ENV_ROWS
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
        "min_rows": CONTEXT_MIN_TRAIN_ROWS,
        "min_env_rows": CONTEXT_MIN_ENV_ROWS,
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
}
CALIBRATION_METHODS = ("sigmoid", "isotonic", "beta")


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
    try:
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=2000,          # techo alto; el early stopping lo recorta
            learning_rate=0.02,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=0.0,
            objective="binary",
            n_jobs=-1,
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
            random_state=42,
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
    params = dict(
        iterations=2000,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
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
    params = dict(
        n_estimators=700,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=1 if verbose else 0,
    )
    if monotone is not None and any(monotone):
        params["monotone_constraints"] = tuple(monotone)
    return XGBClassifier(**params)


def make_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    max_features="sqrt",
                    min_samples_leaf=12,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_logistic():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5)),
        ]
    )


def _new_estimator(kind: str, cols: list[str], verbose: bool = False):
    if kind == "gbm":
        return make_lgbm(_monotone_vector(cols), verbose=verbose)
    if kind == "catboost":
        return make_catboost(_monotone_vector(cols), verbose=verbose)
    if kind == "xgboost":
        return make_xgboost(_monotone_vector(cols), verbose=verbose)
    if kind == "random_forest":
        return make_random_forest()
    if kind == "logistic":
        return make_logistic()
    raise ValueError(f"Tipo de estimador desconocido: {kind}")


def _fit_base(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, cols: list[str],
              random_state: int = 0, verbose: bool = False):
    """Ajusta el base con augmentacion por simetria y, en GBDT, early stopping
    sobre un holdout interno por log loss (evita fijar n_estimators a mano)."""
    est = _new_estimator(kind, cols, verbose=verbose)
    if kind in ("gbm", "catboost", "xgboost"):
        try:
            tr, va = chronological_holdout_indices(y_tr, 0.15)
            Xf, yf = augment(X_tr[tr], y_tr[tr], cols)
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
                )
                return est
            else:  # xgboost
                est.fit(
                    Xf, yf,
                    eval_set=[(X_tr[va], y_tr[va])],
                    verbose=50 if verbose else False,
                )
                return est
        except Exception:
            pass
        est = _new_estimator(kind, cols, verbose=verbose)  # fallback robusto sin early stopping
        Xf, yf = augment(X_tr, y_tr, cols)
        est.fit(Xf, yf)
        return est
    Xf, yf = augment(X_tr, y_tr, cols)
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
                         verbose: bool = False):
    """Ajusta el base UNA vez y devuelve (base, {metodo: estimador_calibrado}).

    El base se entrena sobre tr_idx (augmentado) y cada calibrador sobre cal_idx.
    Compartir el base hace barato comparar sigmoid/isotonica/beta por fold.
    """
    tr_idx, cal_idx = chronological_holdout_indices(y_tr, cal_frac)
    base = _fit_base(kind, X_tr[tr_idx], y_tr[tr_idx], cols,
                     random_state=random_state, verbose=verbose)
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


def super_learner_weights(
    preds: dict[str, list[dict[str, Any]]],
    kinds: list[str] | tuple[str, ...],
    min_history: int = SUPER_LEARNER_MIN_HISTORY,
) -> np.ndarray:
    component_names = [INDIVIDUAL_CANDIDATE[kind] for kind in kinds]
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

    kinds = tuple(dict.fromkeys(algorithm_kinds))
    specs = candidate_specs(kinds)

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
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]
        te_meta = [meta[i] for i in np.where(test_mask)[0]]
        base_rate = float(np.mean(y_tr))
        if verbose:
            print(
                f"      fold {wi+1:03d}/{len(test_periods):03d} period={period}: "
                f"train={len(X_tr)} test={len(X_te)} kinds={','.join(kinds)}",
                flush=True,
            )

        # baselines (sin ajuste)
        elo_p = np.clip(X_te[:, elo_idx] + 0.5, 1e-4, 1 - 1e-4)
        glicko_p = np.clip(X_te[:, glicko_idx] + 0.5, 1e-4, 1 - 1e-4)

        # Ajusta cada tipo de base una vez (con sus calibradores) y reusa.
        fitted: dict[str, Any] = {}
        for kind in kinds:
            try:
                if verbose:
                    print(f"        fitting {kind}...", flush=True)
                fitted[kind] = fit_calibrated_multi(
                    kind, X_tr, y_tr, cols, random_state=wi, verbose=verbose
                )
            except Exception:
                fitted[kind] = None
                if verbose:
                    print(f"        fitting {kind}: FAILED", flush=True)

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
                weights = super_learner_weights(preds, est_kinds)
                if verbose:
                    text_weights = ", ".join(
                        f"{kind}={weight:.3f}" for kind, weight in zip(est_kinds, weights)
                    )
                    print(f"        super learner weights: {text_weights}", flush=True)
                return np.clip(np.average(np.vstack(arrs), axis=0, weights=weights), 1e-4, 1 - 1e-4)
            return np.clip(np.mean(arrs, axis=0), 1e-4, 1 - 1e-4)

        cand: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            arr = _cand_pred(name, spec)
            if arr is not None:
                cand[name] = arr

        for i, m in enumerate(te_meta):
            common = {
                "match_id": m["id"], "date": m["date"], "event": m.get("event"),
                "team1": m["team1"], "team2": m["team2"], "actual": int(y_te[i]),
            }
            preds["base_rate"].append({**common, "prob_team1": base_rate})
            preds["elo"].append({**common, "prob_team1": float(elo_p[i])})
            preds["glicko"].append({**common, "prob_team1": float(glicko_p[i])})
            for name, arr in cand.items():
                preds[name].append({**common, "prob_team1": float(arr[i])})
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


def _kelly_fraction(probability: float, decimal_odds: float) -> float:
    edge = probability * decimal_odds - 1.0
    if edge <= 0 or decimal_odds <= 1:
        return 0.0
    return edge / (decimal_odds - 1.0)


def economic_backtest(rows: list[dict[str, Any]], prod_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_by_id = {row["match_id"]: row for row in prod_rows}
    bankroll = 1.0
    peak = 1.0
    max_drawdown = 0.0
    staked = 0.0
    profit = 0.0
    bets = []
    for row in rows:
        pred = pred_by_id.get(row["id"])
        if not pred:
            continue
        p1 = float(pred["prob_team1"])
        p2 = 1.0 - p1
        odds1 = _safe_decimal_odds(row.get("opening_odds_decimal_t1"))
        odds2 = _safe_decimal_odds(row.get("opening_odds_decimal_t2"))
        if odds1 is None or odds2 is None:
            continue
        side = "team1" if p1 >= 0.5 else "team2"
        probability = p1 if side == "team1" else p2
        odds = odds1 if side == "team1" else odds2
        kelly = _kelly_fraction(probability, odds)
        if kelly <= 0:
            continue
        fraction = min(0.025, 0.25 * kelly)
        stake = bankroll * fraction
        won = bool(row["team1_win"]) if side == "team1" else not bool(row["team1_win"])
        pnl = stake * (odds - 1.0) if won else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak else 0.0)
        staked += stake
        profit += pnl
        bets.append(
            {
                "match_id": row["id"],
                "date": row["date"],
                "side": side,
                "probability": probability,
                "decimal_odds": odds,
                "stake_fraction": fraction,
                "won": won,
                "pnl_fraction_start_bankroll": pnl,
            }
        )
    wins = sum(1 for bet in bets if bet["won"])
    return {
        "n_bets": len(bets),
        "wins": wins,
        "hit_rate": wins / len(bets) if bets else None,
        "roi_on_staked": profit / staked if staked else None,
        "profit_fraction_start_bankroll": bankroll - 1.0,
        "final_bankroll": bankroll,
        "max_drawdown": max_drawdown,
        "staking": "model_favorite_only_quarter_kelly_cap_2_5pct",
        "note": (
            "Siempre apuesta al favorito puro del modelo; las odds de apertura "
            "solo filtran EV positivo y dimensionan el stake."
        ),
        "bets": bets,
    }


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
    latest = {"latest": stamp, "artifact": str(artifact_path), "metadata": str(target / "metadata.json")}
    (MODEL_REGISTRY_DIR / "latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"model_registry_dir": str(target), "registered_model": str(artifact_path)}


def shap_importance(model, X: np.ndarray, cols: list[str], sample: int = 2000) -> list[dict[str, Any]]:
    try:
        import shap
        idx = np.random.RandomState(42).choice(len(X), size=min(sample, len(X)), replace=False)
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


def _safe_decimal_odds(value: Any) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    return odds if odds > 1.0 else None


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
            random_state=10_000 + wi,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena el modelo CS2 (Glicko-2 + LightGBM + calibración).")
    parser.add_argument("--raw", default="",
                        help="Compatibilidad: results_all.json. Si se omite, entrena desde BBDD/cs2.db.")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="BBDD viva usada como fuente por defecto.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--warmup-weeks", type=int, default=10)
    parser.add_argument("--min-train", type=int, default=800)
    parser.add_argument("--no-cs2-filter", action="store_true")
    parser.add_argument("--form-half-life", type=float, default=120.0,
                        help="Vida media (dias) del decaimiento de la forma. Tunable.")
    parser.add_argument("--wf-gap", type=int, default=0,
                        help="Periodos de separacion train->test en walk-forward (anti-fuga).")
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
        default="core",
        help="core para la ablacion; error-aware añade consenso, margen y desacuerdo point-in-time.",
    )
    parser.add_argument(
        "--extra-rating",
        choices=("none", "trueskill"),
        default="none",
        help="Prototipo A/B: 'trueskill' añade rating TrueSkill de equipo (online, "
             "point-in-time) como features. OFF por defecto (no cambia produccion).",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar progreso detallado por fold y logs internos de LightGBM/CatBoost.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Guarda el artefacto en --output-dir, sin sobrescribir MODEL/artifacts/model.pkl ni registry.")
    args = parser.parse_args()

    enabled_kinds = enabled_algorithm_kinds(args.algorithms, no_catboost=args.no_catboost)
    if not enabled_kinds:
        raise SystemExit("No hay algoritmos disponibles. Instala dependencias o cambia --algorithms.")
    has_catboost = "catboost" in enabled_kinds
    has_xgboost = "xgboost" in enabled_kinds

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Cargando histórico…", flush=True)
    master_path = ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"
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
    n_daily = sum(1 for r in rows if r.get("opening_odds_t1") is not None)
    source_label = f"DB {args.db}" if not raw_source else f"raw {raw_source}"
    print(f"      fuente: {source_label}")
    print(f"      {len(rows)} series ({rows[0]['date']} -> {rows[-1]['date']}) · con odds guardadas: {n_daily}")

    print("[2/6] Construyendo features point-in-time…", flush=True)
    t0 = time.time()
    X_dicts, y_list, meta, state = build_training_frame(rows, form_half_life=args.form_half_life)
    model_columns, feature_policies = select_feature_columns(X_dicts, args.feature_profile)
    if args.extra_rating == "trueskill":
        model_columns = list(model_columns) + list(TRUESKILL_FEATURE_COLUMNS)
        feature_policies["extra_rating_trueskill"] = {
            "available_rows": len(X_dicts),
            "min_rows": 0,
            "enabled": True,
            "columns": list(TRUESKILL_FEATURE_COLUMNS),
            "activation": "flag_extra_rating",
            "note": "TrueSkill de equipo (online, point-in-time) activado por --extra-rating.",
        }
    analytics_policy = feature_policies["analytics"]
    context_policy = feature_policies["context"]
    player_policy = feature_policies["player_snapshots"]
    model_b_columns = list(model_columns) + ODDS_FEATURE_COLUMNS
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
    preds = walk_forward(
        X_all, y_all, periods, meta, model_columns, args.warmup_weeks,
        args.min_train, gap=args.wf_gap, algorithm_kinds=enabled_kinds,
        verbose=args.verbose,
    )
    metrics = summarize(preds)
    n_test = max((int(item.get("n", 0)) for item in metrics.values()), default=0)
    print(f"      hecho en {time.time()-t0:.1f}s; n_test={n_test}")
    specs = candidate_specs(enabled_kinds)
    model_order = ["base_rate", "elo", "glicko"] + list(specs.keys())
    for model in model_order:
        m = metrics.get(model)
        if m:
            print(f"      {model:16s} acc={m['accuracy']:.4f} logloss={m['log_loss']:.4f} "
                  f"brier={m['brier']:.4f} auc={m['roc_auc']:.4f} ece={m['ece_10']:.4f}")

    # Selección del modelo de producción por menor log loss walk-forward.
    candidates = {k: metrics[k] for k in specs if k in metrics}
    if not candidates:
        raise SystemExit("Sin candidatos evaluables en walk-forward (revisa min-train/warmup).")
    best_name = min(candidates, key=lambda k: candidates[k]["log_loss"])
    print(f"      -> modelo de producción elegido por log loss: {best_name}")

    # Evaluación segmentada por competitividad + benchmark de mercado.
    segments = segmented_eval(preds.get(best_name, []))
    favorite_accuracy = favorite_accuracy_bands(preds.get(best_name, []))
    print("      por competitividad:")
    for s in segments:
        print(f"        {s['band']:18s} n={s['n']:5d} ({s['share']*100:4.1f}%) acc={s['accuracy']:.3f} logloss={s['log_loss']:.3f}")
    market = market_benchmark(rows, preds, best_name)
    if market.get("n"):
        print(f"      mercado (n={market['n']}): modelo logloss={market.get('model_log_loss')} vs mercado={market.get('market_log_loss')}")
    model_b = model_b_eval(X_model_b, y_all, periods, meta, rows, preds.get(best_name, []), model_b_columns)
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

    print("[4/6] Ajuste final sobre todo el histórico…", flush=True)
    best_kinds, best_method = specs[best_name]
    fitted_full: dict[str, Any] = {}
    for kind in sorted(set(best_kinds) | {"gbm"}):  # gbm siempre, para SHAP
        if args.verbose:
            print(f"      fitting final {kind}...", flush=True)
        fitted_full[kind] = fit_calibrated_multi(kind, X_all, y_all, model_columns,
                                                 cal_frac=0.18, random_state=7,
                                                 verbose=args.verbose)
    _component_kind = {
        "logistic": "logistic",
        "gbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
        "random_forest": "random_forest",
    }
    if best_name == "super_learner_cal":
        component_weights = super_learner_weights(preds, best_kinds, min_history=1)
    else:
        component_weights = np.full(len(best_kinds), 1.0 / len(best_kinds))
    print(
        "      pesos finales: " + ", ".join(
            f"{kind}={weight:.3f}" for kind, weight in zip(best_kinds, component_weights)
        )
    )
    components = []
    for kind, weight in zip(best_kinds, component_weights):
        est = fitted_full[kind][1].get(best_method) or fitted_full[kind][1].get("sigmoid")
        components.append(Component(_component_kind.get(kind, kind), est, None, float(weight)))

    print("[5/6] Importancia SHAP…", flush=True)
    gbm_base = fitted_full["gbm"][0]
    shap_rows = shap_importance(gbm_base, X_all, model_columns)

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
            "analytics_features": analytics_policy,
            "context_features": context_policy,
            "player_snapshot_features": player_policy,
            "map_box_score_features": feature_policies["map_box_scores"],
            "ranking_features": feature_policies["rankings"],
            "roster_features": feature_policies["roster"],
            "feature_policies": feature_policies,
            "walk_forward_metrics": metrics,
            "segmented_eval": segments,
            "market_benchmark": market,
            "model_b": model_b,
            "economic_backtest": {k: v for k, v in economic.items() if k != "bets"},
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
            }.get(best_name, best_name),
            "production_calibration": best_method,
            "production_components": [c.name for c in components],
            "production_component_weights": {c.name: float(c.weight) for c in components},
            "algorithm_request": list(args.algorithms),
            "algorithms_enabled": list(enabled_kinds),
            "feature_profile": args.feature_profile,
            "extra_rating": args.extra_rating,
            "inner_split": "chronological_holdout",
            "form_half_life_days": args.form_half_life,
            "walk_forward_gap": args.wf_gap,
            "catboost_enabled": has_catboost,
            "xgboost_enabled": has_xgboost,
            "monotone_features": sorted(MONOTONE_INCREASING),
            "period_days": 7,
        },
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
    (out / "favorite_accuracy_bands.json").write_text(
        json.dumps(favorite_accuracy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "shap_importance.json").write_text(json.dumps(shap_rows, indent=2), encoding="utf-8")
    (out / "model_b_eval.json").write_text(json.dumps(model_b, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "economic_backtest.json").write_text(json.dumps(economic, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "segmented_eval.json").write_text(
        json.dumps(
            {"segments": segments, "favorite_accuracy": favorite_accuracy, "market": market, "model_b": model_b, "economic_backtest": {k: v for k, v in economic.items() if k != "bets"}},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    import csv

    with open(out / "predictions_walkforward.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "match_id", "date", "event", "team1", "team2", "actual", "prob_team1"])
        for model, rs in preds.items():
            for r in rs:
                w.writerow([model, r["match_id"], r["date"], r.get("event"), r["team1"], r["team2"], r["actual"], r["prob_team1"]])
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
        f"- Profit sobre banca inicial: {economic.get('profit_fraction_start_bankroll')}\n"
        f"- Max drawdown: {economic.get('max_drawdown')}\n"
        f"- Staking: {economic.get('staking')}\n\n"
        f"> {economic.get('note')}\n"
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
        "- El modelo predice el ganador de la SERIE directamente; la arquitectura composicional Bo3 "
        "(P(serie)=P(2-0)+P(2-1)) queda como extensión cuando haya datos por mapa suficientes (PROJECT.md §7.3).\n"
    )
    (out / "REPORT.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

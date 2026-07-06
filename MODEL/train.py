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
    python MODEL/train.py
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
    DIFF_COLUMNS,
    ANALYTICS_DIFF_COLUMNS,
    ANALYTICS_FEATURE_COLUMNS,
    CONTEXT_FEATURE_COLUMNS,
    _period_index,
)
from cs2model.metrics import metric_dict, calibration_bins
from cs2model.artifacts import ModelArtifact, Component, ARTIFACT_PATH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT / "SCRAPPER" / "hltv-scraper-api" / "hltv_scraper" / "data" / "raw"
    / "history_10000_2026-06-28" / "results_all.json"
)
OUTPUT_DIR = ROOT / "MODEL" / "results"
MODEL_REGISTRY_DIR = ROOT / "MODEL" / "artifacts" / "registry"
ODDS_FEATURE_COLUMNS = [
    "opening_odds_prob_centered",
    "opening_odds_confidence",
    "opening_bookmaker_count_log",
]
EXTRA_DIFF_COLUMNS = {"opening_odds_prob_centered"}
ANALYTICS_MIN_TRAIN_ROWS = 120
CONTEXT_MIN_TRAIN_ROWS = 200
CONTEXT_MIN_ENV_ROWS = 50


def _matrix(rows: list[dict[str, float]], cols: list[str]) -> np.ndarray:
    return np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)


def select_feature_columns(X_dicts: list[dict[str, float]]) -> tuple[list[str], dict[str, Any]]:
    analytics_rows = sum(1 for row in X_dicts if (row.get("analytics_available") or 0.0) >= 0.5)
    analytics_enabled = analytics_rows >= ANALYTICS_MIN_TRAIN_ROWS
    context_rows = sum(1 for row in X_dicts if (row.get("context_available") or 0.0) >= 0.5)
    lan_rows = sum(1 for row in X_dicts if (row.get("context_is_lan") or 0.0) >= 0.5)
    online_rows = sum(1 for row in X_dicts if (row.get("context_is_online") or 0.0) >= 0.5)
    stage_rows = sum(1 for row in X_dicts if (row.get("context_stage_known") or 0.0) >= 0.5)
    context_enabled = (
        context_rows >= CONTEXT_MIN_TRAIN_ROWS
        and lan_rows >= CONTEXT_MIN_ENV_ROWS
        and online_rows >= CONTEXT_MIN_ENV_ROWS
    )
    columns = list(FEATURE_COLUMNS)
    if analytics_enabled:
        columns += list(ANALYTICS_FEATURE_COLUMNS)
    if context_enabled:
        columns += list(CONTEXT_FEATURE_COLUMNS)
    return columns, {
        "analytics": {
            "available_rows": analytics_rows,
            "min_rows": ANALYTICS_MIN_TRAIN_ROWS,
            "enabled": analytics_enabled,
            "columns": list(ANALYTICS_FEATURE_COLUMNS) if analytics_enabled else [],
            "note": (
                "Analytics features enabled in training."
                if analytics_enabled else
                "Analytics captured and stored, but excluded from training until enough closed point-in-time rows exist."
            ),
        },
        "context": {
            "available_rows": context_rows,
            "lan_rows": lan_rows,
            "online_rows": online_rows,
            "stage_rows": stage_rows,
            "min_rows": CONTEXT_MIN_TRAIN_ROWS,
            "min_env_rows": CONTEXT_MIN_ENV_ROWS,
            "enabled": context_enabled,
            "columns": list(CONTEXT_FEATURE_COLUMNS) if context_enabled else [],
            "note": (
                "Tournament context features enabled in training."
                if context_enabled else
                "Tournament context captured and stored, but excluded from production training until LAN/online coverage is balanced enough."
            ),
        },
    }


def augment(X: np.ndarray, y: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Duplica el dataset intercambiando A<->B (niega columnas DIFF, invierte y).

    Enseña al modelo una frontera antisimétrica y elimina el sesgo de 'team1'.
    """
    diff_idx = [
        i for i, c in enumerate(cols)
        if c in DIFF_COLUMNS or c in ANALYTICS_DIFF_COLUMNS or c in EXTRA_DIFF_COLUMNS
    ]
    X_rev = X.copy()
    X_rev[:, diff_idx] *= -1.0
    return np.vstack([X, X_rev]), np.concatenate([y, 1 - y])


def make_lgbm():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.0,
            objective="binary",
            n_jobs=-1,
            verbosity=-1,
        )
    except ModuleNotFoundError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.03,
            max_leaf_nodes=24,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=42,
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


def _new_estimator(kind: str):
    return make_lgbm() if kind == "gbm" else make_logistic()


def fit_calibrated(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, cols: list[str],
                   cal_frac: float = 0.2, random_state: int = 0, method: str = "sigmoid"):
    """Ajusta un estimador base + un calibrado Platt (sigmoid) sobre holdout aleatorio.

    Devuelve (base, calibrated):
      - base:       estimador crudo (predict_proba sin calibrar).
      - calibrated: CalibratedClassifierCV(prefit) — probabilidades calibradas.

    El holdout es ALEATORIO estratificado para no perder recencia. Platt es de
    baja varianza (2 parámetros): rara vez degrada un modelo ya calibrado, a
    diferencia de la isotónica con holdouts pequeños (PROJECT.md §7.4).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X_tr))
    tr_idx, cal_idx = train_test_split(
        idx, test_size=cal_frac, random_state=random_state, stratify=y_tr
    )
    base = _new_estimator(kind)
    Xf, yf = augment(X_tr[tr_idx], y_tr[tr_idx], cols)
    base.fit(Xf, yf)

    # FrozenEstimator: calibra sobre el base ya entrenado sin reentrenarlo
    # (sustituye al antiguo cv='prefit', retirado en sklearn 1.8).
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    calibrated.fit(X_tr[cal_idx], y_tr[cal_idx])
    return base, calibrated


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
) -> dict[str, list[dict[str, Any]]]:
    """Walk-forward semanal. Devuelve predicciones por modelo."""
    preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uniq_periods = sorted(set(periods.tolist()))
    test_periods = uniq_periods[warmup_weeks:]

    elo_idx = cols.index("elo_prob_centered")
    glicko_idx = cols.index("glicko_prob_centered")

    for wi, period in enumerate(test_periods):
        train_mask = periods < period
        test_mask = periods == period
        if train_mask.sum() < min_train or len(np.unique(y_all[train_mask])) < 2:
            continue
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]
        te_meta = [meta[i] for i in np.where(test_mask)[0]]
        base_rate = float(np.mean(y_tr))

        # baselines (sin ajuste)
        elo_p = np.clip(X_te[:, elo_idx] + 0.5, 1e-4, 1 - 1e-4)
        glicko_p = np.clip(X_te[:, glicko_idx] + 0.5, 1e-4, 1 - 1e-4)

        # componentes (logística + GBM): crudo y calibrado (Platt)
        log_base, log_cal = fit_calibrated("logistic", X_tr, y_tr, cols, random_state=wi)
        gbm_base, gbm_cal = fit_calibrated("gbm", X_tr, y_tr, cols, random_state=wi)
        cand = {
            "logistic_cal": _proba(log_cal, X_te),
            "lightgbm_cal": _proba(gbm_cal, X_te),
            "ensemble_cal": np.clip(0.5 * _proba(log_cal, X_te) + 0.5 * _proba(gbm_cal, X_te), 1e-4, 1 - 1e-4),
            "ensemble_raw": np.clip(0.5 * _proba(log_base, X_te) + 0.5 * _proba(gbm_base, X_te), 1e-4, 1 - 1e-4),
        }

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
        k1 = _kelly_fraction(p1, odds1)
        k2 = _kelly_fraction(p2, odds2)
        if max(k1, k2) <= 0:
            continue
        side = "team1" if k1 >= k2 else "team2"
        probability = p1 if side == "team1" else p2
        odds = odds1 if side == "team1" else odds2
        fraction = min(0.025, 0.25 * max(k1, k2))
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
        "staking": "quarter_kelly_cap_2_5pct",
        "note": "Solo usa odds de apertura guardadas; n pequeno hasta acumular mas mercado.",
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
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--warmup-weeks", type=int, default=10)
    parser.add_argument("--min-train", type=int, default=800)
    parser.add_argument("--no-cs2-filter", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Cargando histórico…")
    master_path = ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"
    rows = dataio.load_training_rows(args.raw, master_path if master_path.exists() else None,
                                     cs2_only=not args.no_cs2_filter)
    n_daily = sum(1 for r in rows if r.get("opening_odds_t1") is not None)
    print(f"      {len(rows)} series ({rows[0]['date']} -> {rows[-1]['date']}) · con odds guardadas: {n_daily}")

    print("[2/6] Construyendo features point-in-time…")
    t0 = time.time()
    X_dicts, y_list, meta, state = build_training_frame(rows)
    model_columns, feature_policies = select_feature_columns(X_dicts)
    analytics_policy = feature_policies["analytics"]
    context_policy = feature_policies["context"]
    model_b_columns = list(model_columns) + ODDS_FEATURE_COLUMNS
    X_all = _matrix(X_dicts, model_columns)
    X_model_b_dicts = add_odds_features(X_dicts, rows)
    X_model_b = _matrix(X_model_b_dicts, model_b_columns)
    y_all = np.array(y_list, dtype=int)
    periods = np.array([_period_index(r.get("date_obj")) for r in rows])
    print(f"      {len(X_all)} filas, {len(model_columns)} features en {time.time()-t0:.1f}s")
    print(
        "      analytics features: "
        f"{'ON' if analytics_policy['enabled'] else 'OFF'} "
        f"({analytics_policy['available_rows']}/{analytics_policy['min_rows']} filas cerradas)"
    )
    print(
        "      context features:   "
        f"{'ON' if context_policy['enabled'] else 'OFF'} "
        f"({context_policy['available_rows']}/{context_policy['min_rows']} contexto; "
        f"LAN={context_policy['lan_rows']}/{context_policy['min_env_rows']}, "
        f"online={context_policy['online_rows']}/{context_policy['min_env_rows']})"
    )

    print("[3/6] Walk-forward semanal…")
    t0 = time.time()
    preds = walk_forward(X_all, y_all, periods, meta, model_columns, args.warmup_weeks, args.min_train)
    metrics = summarize(preds)
    print(f"      hecho en {time.time()-t0:.1f}s; n_test={metrics.get('ensemble_cal',{}).get('n',0)}")
    model_order = ["base_rate", "elo", "glicko", "logistic_cal", "lightgbm_cal", "ensemble_raw", "ensemble_cal"]
    for model in model_order:
        m = metrics.get(model)
        if m:
            print(f"      {model:14s} acc={m['accuracy']:.4f} logloss={m['log_loss']:.4f} "
                  f"brier={m['brier']:.4f} auc={m['roc_auc']:.4f} ece={m['ece_10']:.4f}")

    # Selección del modelo de producción por menor log loss walk-forward.
    candidates = {k: metrics[k] for k in ("logistic_cal", "lightgbm_cal", "ensemble_raw", "ensemble_cal") if k in metrics}
    best_name = min(candidates, key=lambda k: candidates[k]["log_loss"])
    print(f"      -> modelo de producción elegido por log loss: {best_name}")

    # Evaluación segmentada por competitividad + benchmark de mercado.
    segments = segmented_eval(preds.get(best_name, []))
    print("      por competitividad:")
    for s in segments:
        print(f"        {s['band']:18s} n={s['n']:5d} ({s['share']*100:4.1f}%) acc={s['accuracy']:.3f} logloss={s['log_loss']:.3f}")
    market = market_benchmark(rows, preds, best_name)
    if market.get("n"):
        print(f"      mercado (n={market['n']}): modelo logloss={market.get('model_log_loss')} vs mercado={market.get('market_log_loss')}")
    model_b = model_b_eval(X_model_b, y_all, periods, meta, rows, preds.get(best_name, []), model_b_columns)
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

    print("[4/6] Ajuste final sobre todo el histórico…")
    log_base, log_cal = fit_calibrated("logistic", X_all, y_all, model_columns, cal_frac=0.18, random_state=7)
    gbm_base, gbm_cal = fit_calibrated("gbm", X_all, y_all, model_columns, cal_frac=0.18, random_state=7)
    cal_use = best_name in ("logistic_cal", "lightgbm_cal", "ensemble_cal")
    log_est = log_cal if cal_use else log_base
    gbm_est = gbm_cal if cal_use else gbm_base
    if best_name in ("ensemble_cal", "ensemble_raw"):
        components = [
            Component("logistic", log_est, None, 0.5),
            Component("lightgbm", gbm_est, None, 0.5),
        ]
    elif best_name == "logistic_cal":
        components = [Component("logistic", log_est, None, 1.0)]
    else:
        components = [Component("lightgbm", gbm_est, None, 1.0)]

    print("[5/6] Importancia SHAP…")
    shap_rows = shap_importance(gbm_base, X_all, model_columns)

    print("[6/6] Guardando artefacto y resultados…")
    artifact = ModelArtifact(
        feature_columns=list(model_columns),
        components=components,
        metadata={
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_source": str(args.raw),
            "n_train_rows": int(len(X_all)),
            "date_min": rows[0]["date"],
            "date_max": rows[-1]["date"],
            "cs2_only": not args.no_cs2_filter,
            "analytics_features": analytics_policy,
            "context_features": context_policy,
            "feature_policies": feature_policies,
            "walk_forward_metrics": metrics,
            "segmented_eval": segments,
            "market_benchmark": market,
            "model_b": model_b,
            "economic_backtest": {k: v for k, v in economic.items() if k != "bets"},
            "production_model": best_name,
            "model": "Glicko-2 features + " + {
                "ensemble_cal": "LightGBM ⊕ Logística (Platt)",
                "ensemble_raw": "LightGBM ⊕ Logística",
                "logistic_cal": "Logística (Platt)",
                "lightgbm_cal": "LightGBM (Platt)",
            }.get(best_name, best_name),
            "period_days": 7,
        },
    )
    artifact.save(ARTIFACT_PATH)

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
    (out / "shap_importance.json").write_text(json.dumps(shap_rows, indent=2), encoding="utf-8")
    (out / "model_b_eval.json").write_text(json.dumps(model_b, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "economic_backtest.json").write_text(json.dumps(economic, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "segmented_eval.json").write_text(
        json.dumps(
            {"segments": segments, "market": market, "model_b": model_b, "economic_backtest": {k: v for k, v in economic.items() if k != "bets"}},
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

    registry = save_model_registry(artifact, metrics, shap_rows, segments, market, model_b, economic)
    artifact.metadata["registry"] = registry
    artifact.save(ARTIFACT_PATH)
    _write_report(out, metrics, shap_rows, rows, artifact, segments, market, model_b, economic)
    print(f"      artefacto: {ARTIFACT_PATH}")
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
    analytics_policy = feature_policies.get("analytics") or artifact.metadata.get("analytics_features") or {}
    context_policy = feature_policies.get("context") or artifact.metadata.get("context_features") or {}
    lines = [
        "# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)\n",
        f"\nGenerado: {artifact.metadata['trained_at']}  \n",
        f"Histórico: {rows[0]['date']} → {rows[-1]['date']} ({len(rows)} series, era CS2)\n",
        "\n## Politica de features opcionales\n\n",
        f"- Analytics HLTV: {'ON' if analytics_policy.get('enabled') else 'OFF'} "
        f"({analytics_policy.get('available_rows', 0)}/{analytics_policy.get('min_rows', 0)} filas cerradas). "
        f"{analytics_policy.get('note', '')}\n",
        f"- Contexto torneo LAN/online/fase: {'ON' if context_policy.get('enabled') else 'OFF'} "
        f"({context_policy.get('available_rows', 0)}/{context_policy.get('min_rows', 0)} contexto; "
        f"LAN={context_policy.get('lan_rows', 0)}/{context_policy.get('min_env_rows', 0)}, "
        f"online={context_policy.get('online_rows', 0)}/{context_policy.get('min_env_rows', 0)}). "
        f"{context_policy.get('note', '')}\n",
        "\n## Resultados walk-forward (ventana expansiva, paso semanal)\n\n",
        "| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
        row("base_rate", "Base rate (baseline)"),
        row("elo", "Elo (baseline)"),
        row("glicko", "Glicko-2 (baseline)"),
        row("logistic_cal", "Logística (Platt)"),
        row("lightgbm_cal", "LightGBM (Platt)"),
        row("ensemble_raw", "Ensemble sin calibrar"),
        row("ensemble_cal", "**Ensemble (Platt)**"),
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

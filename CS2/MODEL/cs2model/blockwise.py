"""Block-wise missing + activacion data-driven + driver de comparacion anidado.

Responde empiricamente a "¿es el enfoque hibrido lo mejor?" comparando bajo el
mismo walk-forward ANIDADO (seleccion por log loss):

  (i)  indicators : un modelo sobre TODO con flags de disponibilidad.
  (ii) profiles   : submodelos por patron de disponibilidad de familias (routing).
  (iii) two_stage : base con features siempre disponibles + enriquecimiento que
                    solo actua donde esas features existen (gating por cobertura).

Ademas, la activacion de familias es DATA-DRIVEN (forward greedy por log loss OOS
en la CV interna), no por umbral fijo (analytics 154, rankings 209, ...).

Todas las decisiones (familias, hiperparametros via Optuna, calibrador) se toman
en el bucle interno con datos estrictamente pasados; el bucle externo solo mide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .metrics import metric_dict, calibration_bins
from .evaluation import EvalConfig, Family, _blocks, _recency_weights, _num
from .model_zoo import build_model, available_models
from .calibration_suite import select_calibrator, ALL_METHODS

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ===========================================================================
# Utilidades
# ===========================================================================
def _matrix(rows: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(cols)))
    return np.array([[_num(r.get(c)) for c in cols] for r in rows], dtype=float)


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(r["label"]) for r in rows], dtype=int)


def _periods(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(r["period"]) for r in rows], dtype=int)


def family_columns(base_cols: list[str], active: Sequence[Family]) -> list[str]:
    cols = list(base_cols)
    for fam in active:
        cols += [c for c in fam.columns if c not in cols]
        if fam.availability_column not in cols:
            cols.append(fam.availability_column)
    return cols


def _coverage_key(row: dict[str, Any], families: Sequence[Family]) -> tuple[int, ...]:
    return tuple(1 if (_num(row.get(f.availability_column)) or 0.0) >= 0.5 else 0 for f in families)


def _fit_predict(model_name: str, params: dict[str, Any], recency: float,
                 train_rows: list[dict[str, Any]], predict_rows: list[dict[str, Any]],
                 cols: list[str]) -> np.ndarray:
    if not train_rows or not predict_rows:
        return np.full(len(predict_rows), 0.5)
    X_tr = _matrix(train_rows, cols)
    y_tr = _labels(train_rows)
    if len(np.unique(y_tr)) < 2:
        return np.full(len(predict_rows), float(np.mean(y_tr)) if len(y_tr) else 0.5)
    sw = _recency_weights(_periods(train_rows), recency) if recency else None
    model = build_model(model_name, params)
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model.predict_proba(_matrix(predict_rows, cols))


def _inner_logloss(model_name: str, params: dict[str, Any], recency: float,
                   rows: list[dict[str, Any]], cols: list[str], cfg: EvalConfig) -> float:
    """Log loss OOS media en un walk-forward interno sobre `rows`."""
    per = _periods(rows)
    periods_sorted = sorted(set(per.tolist()))
    inner = _blocks(periods_sorted, warmup=cfg.inner_warmup, step=cfg.inner_step,
                    gap=cfg.gap_periods, window=cfg.window, train_width=cfg.train_width)
    if not inner:
        return np.inf
    ps, ys = [], []
    for tr_p, te_p in inner:
        tr = [rows[i] for i in range(len(rows)) if per[i] in tr_p]
        te = [rows[i] for i in range(len(rows)) if per[i] in te_p]
        if len(tr) < 20 or not te or len(np.unique(_labels(tr))) < 2:
            continue
        p = _fit_predict(model_name, params, recency, tr, te, cols)
        ps.extend(p.tolist()); ys.extend(_labels(te).tolist())
    if len(ps) < 20 or len(set(ys)) < 2:
        return np.inf
    return metric_dict(np.array(ys), np.clip(np.array(ps), 1e-9, 1 - 1e-9))["log_loss"]


# ===========================================================================
# Activacion DATA-DRIVEN de familias (forward greedy por log loss OOS)
# ===========================================================================
def select_families_cv(train_rows: list[dict[str, Any]], base_cols: list[str],
                       families: Sequence[Family], model_name: str, cfg: EvalConfig,
                       default_params: dict[str, Any] | None = None) -> tuple[list[Family], dict[str, Any]]:
    """Anade una familia solo si BAJA el log loss OOS interno. Sustituye umbrales fijos."""
    params = default_params or {}
    active: list[Family] = []
    remaining = list(families)
    base_ll = _inner_logloss(model_name, params, 0.0, train_rows, family_columns(base_cols, active), cfg)
    log = {"base_log_loss": base_ll, "steps": []}
    improved = True
    while improved and remaining:
        improved = False
        best_gain = 1e-9
        best_fam = None
        best_ll = base_ll
        for fam in remaining:
            cols = family_columns(base_cols, active + [fam])
            ll = _inner_logloss(model_name, params, 0.0, train_rows, cols, cfg)
            gain = base_ll - ll
            if gain > best_gain:
                best_gain, best_fam, best_ll = gain, fam, ll
        if best_fam is not None:
            active.append(best_fam)
            remaining.remove(best_fam)
            log["steps"].append({"family": best_fam.name, "log_loss": float(best_ll), "gain": float(best_gain)})
            base_ll = best_ll
            improved = True
    log["active"] = [f.name for f in active]
    return active, log


# ===========================================================================
# Optuna TPE en el bucle interno (half-life de recencia = hiperparametro)
# ===========================================================================
def tune_optuna(model_name: str, train_rows: list[dict[str, Any]], cols: list[str],
                cfg: EvalConfig, n_trials: int) -> tuple[dict[str, Any], float]:
    if not HAS_OPTUNA or n_trials <= 0:
        return {}, 0.0
    model_cls = build_model(model_name).__class__

    def objective(trial: Any) -> float:
        params = model_cls.optuna_space(trial)
        recency = trial.suggest_categorical("recency_half_life", [0.0, 90.0, 180.0, 365.0, 730.0])
        return _inner_logloss(model_name, params, recency, train_rows, cols, cfg)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    recency = best.pop("recency_half_life", 0.0)
    return best, float(recency)


# ===========================================================================
# Assemblies (block-wise)
# ===========================================================================
def assembly_fit_predict(mode: str, model_name: str, params: dict[str, Any], recency: float,
                         active: Sequence[Family], train_rows: list[dict[str, Any]],
                         predict_rows: list[dict[str, Any]], base_cols: list[str],
                         cfg: EvalConfig) -> np.ndarray:
    if mode == "indicators":
        cols = family_columns(base_cols, active)
        return _fit_predict(model_name, params, recency, train_rows, predict_rows, cols)

    if mode == "profiles":
        # Submodelo por patron de disponibilidad; routing por perfil, fallback a base.
        base_only_cols = list(base_cols)
        groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for r in train_rows:
            groups.setdefault(_coverage_key(r, active), []).append(r)
        preds = np.empty(len(predict_rows))
        # fallback global (solo base) entrenado con todo
        for i, r in enumerate(predict_rows):
            key = _coverage_key(r, active)
            grp = groups.get(key, [])
            # familias cubiertas en este perfil
            covered = [f for f, k in zip(active, key) if k == 1]
            cols = family_columns(base_cols, covered)
            if len(grp) >= 60 and len(np.unique(_labels(grp))) >= 2:
                preds[i] = _fit_predict(model_name, params, recency, grp, [r], cols)[0]
            else:
                preds[i] = _fit_predict(model_name, params, recency, train_rows, [r], base_only_cols)[0]
        return preds

    if mode == "two_stage":
        # Base con features siempre disponibles + enriquecido donde hay cobertura.
        base_cols_only = list(base_cols)
        full_cols = family_columns(base_cols, active)
        p_base = _fit_predict(model_name, params, recency, train_rows, predict_rows, base_cols_only)
        # modelo enriquecido entrenado SOLO con filas con cobertura total de `active`
        covered_train = [r for r in train_rows if all(k == 1 for k in _coverage_key(r, active))]
        out = p_base.copy()
        if len(covered_train) >= 60 and len(np.unique(_labels(covered_train))) >= 2:
            covered_idx = [i for i, r in enumerate(predict_rows)
                           if all(k == 1 for k in _coverage_key(r, active))]
            if covered_idx:
                p_full = _fit_predict(model_name, params, recency, covered_train,
                                      [predict_rows[i] for i in covered_idx], full_cols)
                for j, i in enumerate(covered_idx):
                    out[i] = p_full[j]   # gating: usa el enriquecido donde existe
        return out

    raise ValueError(f"assembly desconocido: {mode}")


# ===========================================================================
# Driver de comparacion anidado
# ===========================================================================
@dataclass
class ComparisonConfig:
    models: tuple[str, ...] = ("logistic_en", "lightgbm")
    assemblies: tuple[str, ...] = ("indicators", "two_stage")
    n_trials: int = 0
    cal_fraction: float = 0.25            # cola temporal del train para calibrar
    calibration_methods: tuple[str, ...] = ALL_METHODS


def _split_tail(rows: list[dict[str, Any]], fraction: float) -> tuple[list, list]:
    periods_sorted = sorted(set(_periods(rows).tolist()))
    if len(periods_sorted) < 4:
        return rows, []
    k = max(1, int(round(len(periods_sorted) * fraction)))
    val_periods = set(periods_sorted[-k:])
    head = [r for r in rows if int(r["period"]) not in val_periods]
    tail = [r for r in rows if int(r["period"]) in val_periods]
    return head, tail


def run_comparison(rows: list[dict[str, Any]], base_cols: list[str], families: Sequence[Family],
                   cfg: EvalConfig, comp: ComparisonConfig) -> dict[str, Any]:
    per = _periods(rows)
    periods_sorted = sorted(set(per.tolist()))
    outer = _blocks(periods_sorted, warmup=cfg.warmup_periods, step=cfg.outer_step,
                    gap=cfg.gap_periods, window=cfg.window, train_width=cfg.train_width)
    y_all = _labels(rows)

    combos = [(m, a) for m in comp.models for a in comp.assemblies]
    oos: dict[str, list[float]] = {f"{m}|{a}": [] for (m, a) in combos}
    oos.update({"elo": [], "market": []})
    oos_y: list[int] = []
    oos_meta: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    calib_choice: dict[str, list[str]] = {f"{m}|{a}": [] for (m, a) in combos}

    for fold_i, (tr_p, te_p) in enumerate(outer):
        tr_idx = [i for i in range(len(rows)) if per[i] in tr_p]
        te_idx = [i for i in range(len(rows)) if per[i] in te_p]
        if len(tr_idx) < cfg.warmup_periods or not te_idx or len(np.unique(y_all[tr_idx])) < 2:
            continue
        train_rows = [rows[i] for i in tr_idx]
        test_rows = [rows[i] for i in te_idx]
        head, tail = _split_tail(train_rows, comp.cal_fraction)

        fold_dec: dict[str, Any] = {"fold": fold_i, "train_rows": len(train_rows), "test_rows": len(test_rows)}
        # activacion + tuning por MODELO (compartido entre assemblies)
        per_model: dict[str, dict[str, Any]] = {}
        for m in comp.models:
            active, act_log = select_families_cv(train_rows, base_cols, families, m, cfg)
            cols = family_columns(base_cols, active)
            params, recency = tune_optuna(m, train_rows, cols, cfg, comp.n_trials)
            per_model[m] = {"active": active, "params": params, "recency": recency}
            fold_dec[m] = {"families": [f.name for f in active], "recency": recency,
                           "tuned": bool(params), "activation": act_log}

        for (m, a) in combos:
            info = per_model[m]
            active, params, recency = info["active"], info["params"], info["recency"]
            # 1) calibrador: entrena assembly en head, predice tail, selecciona
            if tail:
                p_tail = assembly_fit_predict(a, m, params, recency, active, head, tail, base_cols, cfg)
                y_tail = _labels(tail)
                # sub-split del tail para elegir metodo sin optimismo
                cut = len(tail) // 2
                if cut >= 10 and len(set(y_tail[:cut].tolist())) == 2 and len(set(y_tail[cut:].tolist())) == 2:
                    cal, _sc = select_calibrator(p_tail[:cut], y_tail[:cut], p_tail[cut:], y_tail[cut:],
                                                 methods=comp.calibration_methods)
                else:
                    cal, _sc = select_calibrator(p_tail, y_tail, p_tail, y_tail, methods=comp.calibration_methods)
            else:
                from .calibration_suite import fit_calibrator
                cal = fit_calibrator("identity", np.array([0.5]), np.array([0]))
            calib_choice[f"{m}|{a}"].append(cal.method)
            # 2) refit en TODO el train, predice test, aplica calibrador
            p_test = assembly_fit_predict(a, m, params, recency, active, train_rows, test_rows, base_cols, cfg)
            oos[f"{m}|{a}"].extend(cal.apply(p_test).tolist())

        # baselines + meta del test
        for r in test_rows:
            oos["elo"].append(_elo_prob(r))
            oos["market"].append(_num(r.get("opening_prob_t1")))
            oos_meta.append({k: _num(r.get(k)) for k in
                             ("opening_prob_t1", "opening_odds_t1", "opening_odds_t2", "closing_prob_t1", "closing_odds_t1", "closing_odds_t2")})
        oos_y.extend(y_all[te_idx].tolist())
        decisions.append(fold_dec)

    y = np.array(oos_y, dtype=int)
    result: dict[str, Any] = {"window": cfg.window, "n_eval": int(len(y)), "n_folds": len(decisions),
                              "models": {}, "calibration_bins": {}, "calibrator_choice": {},
                              "decisions": decisions, "available_models": available_models()}
    for name, preds in oos.items():
        p = np.array(preds, dtype=float)
        finite = np.isfinite(p)
        if finite.sum() == 0:
            result["models"][name] = {"n": 0}
            continue
        result["models"][name] = metric_dict(y[finite], np.clip(p[finite], 1e-9, 1 - 1e-9))
        if name in calib_choice:
            uniq, cnts = np.unique(calib_choice[name], return_counts=True)
            result["calibrator_choice"][name] = dict(zip(uniq.tolist(), cnts.tolist()))
    # calibracion del mejor combo por log loss
    combo_names = [f"{m}|{a}" for (m, a) in combos]
    ranked = sorted((n for n in combo_names if result["models"].get(n, {}).get("n", 0)),
                    key=lambda n: result["models"][n]["log_loss"])
    best = ranked[0] if ranked else None
    result["best_combo"] = best
    if best:
        pb = np.array(oos[best], dtype=float)
        fb = np.isfinite(pb)
        result["calibration_bins"][best] = calibration_bins(y[fb], np.clip(pb[fb], 1e-9, 1 - 1e-9))
        result["betting"] = betting_clv(y, pb, oos_meta)
    return result


def _elo_prob(row: dict[str, Any]) -> float:
    for key in ("elo_prob_centered", "glicko_prob_centered"):
        v = _num(row.get(key))
        if np.isfinite(v):
            return float(np.clip(v + 0.5, 1e-9, 1 - 1e-9))
    return 0.5


# ===========================================================================
# Apuestas: ROI + CLV vs APERTURA y vs CIERRE (F)
# ===========================================================================
def betting_clv(y: np.ndarray, p_model: np.ndarray, meta: list[dict[str, Any]]) -> dict[str, Any]:
    bets = 0; stake = 0.0; ret = 0.0; wins = 0
    clv_open_to_close: list[float] = []
    for i, m in enumerate(meta):
        po1, po2 = m.get("opening_odds_t1"), m.get("opening_odds_t2")
        if not (np.isfinite(po1) and np.isfinite(po2) and po1 > 1 and po2 > 1):
            continue
        pm = float(p_model[i]); side1 = pm >= 0.5
        odds = po1 if side1 else po2
        prob = pm if side1 else (1 - pm)
        if prob * odds - 1.0 <= 0:
            continue
        bets += 1; stake += 1.0
        won = (int(y[i]) == 1) if side1 else (int(y[i]) == 0)
        ret += (odds - 1.0) if won else -1.0
        wins += int(won)
        op, cp = m.get("opening_prob_t1"), m.get("closing_prob_t1")
        if np.isfinite(op) and np.isfinite(cp):
            open_side = op if side1 else (1 - op)
            close_side = cp if side1 else (1 - cp)
            clv_open_to_close.append(float(close_side - open_side))
    return {
        "subset_with_odds": int(sum(1 for m in meta if np.isfinite(m.get("opening_odds_t1")))),
        "bets": bets, "hit_rate": (wins / bets) if bets else None,
        "roi_on_stake": (ret / stake) if stake else None,
        "clv_vs_closing_n": len(clv_open_to_close),
        "clv_vs_closing_mean": (float(np.mean(clv_open_to_close)) if clv_open_to_close else None),
        "note": "lado = favorito del modelo con EV+ vs APERTURA; CLV = implicita cierre - apertura del lado apostado.",
    }

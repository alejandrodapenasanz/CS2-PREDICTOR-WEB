"""Harness de evaluacion honesto: walk-forward temporal ANIDADO sin leakage.

Objetivo de este modulo (ver docs/EVALUATION.md y docs/AUDIT.md L1/L2):

  * BUCLE EXTERNO  -> estimacion de rendimiento INSESGADA. Su ventana de test
    no influye en NINGUNA decision.
  * BUCLE INTERNO  -> TODAS las decisiones (hiperparametros, activacion de
    familias por cobertura, metodo de calibracion, eleccion de estimador),
    entrenadas SOLO con datos estrictamente pasados.
  * GAP/EMBARGO temporal entre train y test en ambos bucles.
  * Comparacion ventana EXPANSIVA vs DESLIZANTE.

El nucleo es numpy puro y trae un fitter logistico propio para poder validarse
sin sklearn/lightgbm. El fitter de PRODUCCION (LightGBM/CatBoost/super-learner)
se INYECTA (parametro `fitter`) en la maquina del usuario, sin cambiar el
protocolo anti-leakage. Reutiliza cs2model.metrics para las metricas.

Nada aqui decide sobre produccion: mide. El modelo de produccion se reentrena
aparte con todo el historico usando el mismo protocolo interno.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .metrics import metric_dict, calibration_bins


# ===========================================================================
# Errores
# ===========================================================================
class PointInTimeError(AssertionError):
    """Se lanza cuando una feature depende de informacion futura (leakage)."""


# ===========================================================================
# Fitter por defecto: regresion logistica numpy (deterministic, sin deps)
# ===========================================================================
@dataclass
class _StandardizedLogit:
    """Logistica L2 con estandarizacion e imputacion aprendidas SOLO en train."""
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    fill: np.ndarray  # valor de imputacion (media de train) por columna

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xi = _impute(np.asarray(X, dtype=float), self.fill)
        Z = (Xi - self.mean) / self.std
        logits = Z @ self.weights + self.bias
        return _sigmoid(logits)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _impute(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = X.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        cols = np.where(mask)[1]
        out[mask] = np.take(fill, cols)
    return out


def fit_logistic_np(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    iters: int = 300,
    lr: float = 0.5,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> _StandardizedLogit:
    """Logistica por descenso de gradiente. Determinista (sin aleatoriedad)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # columna todo-NaN -> se rellena con 0
        fill = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    Xi = _impute(X, fill)
    mean = Xi.mean(axis=0)
    std = Xi.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = (Xi - mean) / std

    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    sw = sw / (sw.mean() if sw.mean() else 1.0)
    for _ in range(iters):
        p = _sigmoid(Z @ w + b)
        g = (p - y) * sw
        grad_w = Z.T @ g / n + l2 * w / n
        grad_b = g.sum() / n
        w -= lr * grad_w
        b -= lr * grad_b
    return _StandardizedLogit(weights=w, bias=float(b), mean=mean, std=std, fill=fill)


#: Firma de un fitter enchufable: (X_tr, y_tr, cols, l2, sample_weight) -> objeto con predict_proba.
Fitter = Callable[..., Any]


def default_fitter(X: np.ndarray, y: np.ndarray, cols: Sequence[str], **kw: Any) -> _StandardizedLogit:
    return fit_logistic_np(X, y, l2=kw.get("l2", 1.0), sample_weight=kw.get("sample_weight"))


# ===========================================================================
# Calibracion (numpy puro): identity / Platt / isotonica (PAV)
# ===========================================================================
def _platt_fit(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    logit = fit_logistic_np(s, np.asarray(y, dtype=float), l2=0.0, iters=500, lr=0.5)
    a = float(logit.weights[0] / logit.std[0])
    b = float(logit.bias - logit.weights[0] * logit.mean[0] / logit.std[0])
    return a, b


def _isotonic_fit(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators. Devuelve (x_knots, y_knots) crecientes."""
    order = np.argsort(p, kind="mergesort")
    x = np.asarray(p, dtype=float)[order]
    yv = np.asarray(y, dtype=float)[order]
    w = np.ones_like(yv)
    # PAV
    ys = yv.copy()
    i = 0
    n = len(ys)
    while i < n - 1:
        if ys[i] > ys[i + 1] + 1e-12:
            new = (w[i] * ys[i] + w[i + 1] * ys[i + 1]) / (w[i] + w[i + 1])
            ys[i] = new
            ys[i + 1] = new
            w[i] += w[i + 1]
            # colapsa hacia atras para mantener monotonia
            j = i
            while j > 0 and ys[j - 1] > ys[j] + 1e-12:
                new = (w[j - 1] * ys[j - 1] + w[j] * ys[j]) / (w[j - 1] + w[j])
                ys[j - 1] = new
                ys[j] = new
                w[j - 1] += w[j]
                j -= 1
            i = max(j - 1, 0)
        else:
            i += 1
    return x, ys


@dataclass
class Calibrator:
    method: str = "identity"
    a: float = 1.0
    b: float = 0.0
    x_knots: np.ndarray | None = None
    y_knots: np.ndarray | None = None

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        if self.method == "identity":
            return p
        if self.method == "platt":
            return _sigmoid(self.a * _logit(p) + self.b)
        if self.method == "isotonic" and self.x_knots is not None:
            return np.clip(np.interp(p, self.x_knots, self.y_knots), 1e-9, 1 - 1e-9)
        return p


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "identity":
        return Calibrator("identity")
    if method == "platt":
        a, b = _platt_fit(_logit(np.clip(p, 1e-9, 1 - 1e-9)), y)
        return Calibrator("platt", a=a, b=b)
    if method == "isotonic":
        x, yk = _isotonic_fit(p, y)
        return Calibrator("isotonic", x_knots=x, y_knots=yk)
    raise ValueError(f"metodo de calibracion desconocido: {method}")


# ===========================================================================
# Config de evaluacion
# ===========================================================================
@dataclass
class EvalConfig:
    warmup_periods: int = 10          # periodos iniciales reservados al primer train
    gap_periods: int = 1              # embargo entre fin de train y test (ambos bucles)
    outer_step: int = 4               # ancho del bloque de test externo (semanas). 4 ~ mensual
    inner_step: int = 4               # ancho del bloque de test interno
    inner_warmup: int = 8
    window: str = "expanding"         # "expanding" | "sliding"
    train_width: int = 52             # ancho de la ventana deslizante (periodos)
    l2_grid: tuple[float, ...] = (0.25, 1.0, 4.0)          # hiperparametro tuneado en el inner
    calibration_methods: tuple[str, ...] = ("identity", "platt", "isotonic")
    recency_half_life: float = 0.0    # 0 = sin ponderado por recencia
    seed: int = 42


@dataclass
class Family:
    """Familia de features auto-activable por cobertura (decision del inner)."""
    name: str
    columns: tuple[str, ...]
    availability_column: str
    threshold: int


# ===========================================================================
# Utilidades temporales
# ===========================================================================
def _recency_weights(periods: np.ndarray, half_life: float) -> np.ndarray | None:
    if not half_life or half_life <= 0:
        return None
    p = np.asarray(periods, dtype=float)
    if p.size == 0:
        return None
    ages = (p.max() - p) * 7.0
    return np.clip(0.5 ** (ages / float(half_life)), 1e-3, 1.0)


def _blocks(
    periods_sorted: list[int],
    *,
    warmup: int,
    step: int,
    gap: int,
    window: str,
    train_width: int,
) -> list[tuple[set[int], set[int]]]:
    """Devuelve pares (periodos_train, periodos_test) respetando el gap.

    - expanding: train = todos los periodos < inicio_test - gap.
    - sliding:  train = [inicio_test - gap - train_width, inicio_test - gap).
    """
    out: list[tuple[set[int], set[int]]] = []
    start = warmup
    while start < len(periods_sorted):
        test_periods = set(periods_sorted[start:start + step])
        if not test_periods:
            break
        first_test = min(test_periods)
        train_hi = first_test - gap
        if window == "sliding":
            train_lo = first_test - gap - train_width
            train_periods = {p for p in periods_sorted if train_lo <= p < train_hi}
        else:
            train_periods = {p for p in periods_sorted if p < train_hi}
        if train_periods:
            out.append((train_periods, test_periods))
        start += step
    return out


# ===========================================================================
# Motor anidado
# ===========================================================================
def _matrix(rows: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    return np.array([[_num(r.get(c)) for c in cols] for r in rows], dtype=float)


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return np.nan


def _active_families(rows: list[dict[str, Any]], families: Sequence[Family]) -> list[Family]:
    """Decide que familias entran contando cobertura SOLO en `rows` (train)."""
    active = []
    for fam in families:
        cov = sum(1 for r in rows if (_num(r.get(fam.availability_column)) or 0.0) >= 0.5)
        if cov >= fam.threshold:
            active.append(fam)
    return active


def _columns_for(base_cols: list[str], active: Sequence[Family]) -> list[str]:
    cols = list(base_cols)
    for fam in active:
        cols += [c for c in fam.columns if c not in cols]
    return cols


def _inner_select(
    train_rows: list[dict[str, Any]],
    train_periods: np.ndarray,
    base_cols: list[str],
    families: Sequence[Family],
    cfg: EvalConfig,
    fitter: Fitter,
) -> dict[str, Any]:
    """BUCLE INTERNO: elige familias, l2 y calibrador con SOLO datos de train."""
    # Familias: decision por cobertura en train (pasado del outer).
    active = _active_families(train_rows, families)
    cols = _columns_for(base_cols, active)

    periods_sorted = sorted(set(int(p) for p in train_periods.tolist()))
    inner = _blocks(
        periods_sorted, warmup=cfg.inner_warmup, step=cfg.inner_step,
        gap=cfg.gap_periods, window=cfg.window, train_width=cfg.train_width,
    )
    y_all = np.array([int(r["label"]) for r in train_rows])
    X_all = _matrix(train_rows, cols)
    per = np.asarray(train_periods)

    best = {"log_loss": np.inf, "l2": cfg.l2_grid[0], "calibration": "identity"}
    if not inner:
        return {"columns": cols, "families": [f.name for f in active], **best}

    for l2 in cfg.l2_grid:
        # Predicciones OOS internas (sin calibrar) para elegir l2 y calibrador.
        oos_p: list[float] = []
        oos_y: list[int] = []
        for tr_p, te_p in inner:
            tr = np.array([p in tr_p for p in per])
            te = np.array([p in te_p for p in per])
            if tr.sum() < 20 or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            sw = _recency_weights(per[tr], cfg.recency_half_life)
            model = fitter(X_all[tr], y_all[tr], cols, l2=l2, sample_weight=sw)
            oos_p.extend(model.predict_proba(X_all[te]).tolist())
            oos_y.extend(y_all[te].tolist())
        if len(oos_p) < 20 or len(set(oos_y)) < 2:
            continue
        pa = np.array(oos_p)
        ya = np.array(oos_y)
        for method in cfg.calibration_methods:
            cal = fit_calibrator(method, pa, ya)
            ll = metric_dict(ya, cal.apply(pa))["log_loss"]
            if ll < best["log_loss"]:
                best = {"log_loss": float(ll), "l2": float(l2), "calibration": method}

    best["columns"] = cols
    best["families"] = [f.name for f in active]
    return best


def nested_walk_forward(
    rows: list[dict[str, Any]],
    base_cols: list[str],
    families: Sequence[Family],
    cfg: EvalConfig,
    *,
    fitter: Fitter = default_fitter,
) -> dict[str, Any]:
    """Walk-forward ANIDADO. Devuelve predicciones OOS externas + decisiones.

    `rows`: cada dict tiene al menos 'label' (0/1), 'period' (int semanal) y las
    columnas de features. Opcionalmente 'opening_prob_t1'/'closing_prob_t1' para
    el benchmark de mercado/apuestas.
    """
    periods = np.array([int(r["period"]) for r in rows])
    periods_sorted = sorted(set(periods.tolist()))
    outer = _blocks(
        periods_sorted, warmup=cfg.warmup_periods, step=cfg.outer_step,
        gap=cfg.gap_periods, window=cfg.window, train_width=cfg.train_width,
    )
    y_all = np.array([int(r["label"]) for r in rows])

    oos: dict[str, list[float]] = {"model": [], "elo": [], "market": []}
    oos_y: list[int] = []
    oos_meta: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for fold_i, (tr_p, te_p) in enumerate(outer):
        tr = np.array([p in tr_p for p in periods])
        te = np.array([p in te_p for p in periods])
        if tr.sum() < cfg.warmup_periods or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
            continue
        train_rows = [rows[i] for i in np.where(tr)[0]]
        test_rows = [rows[i] for i in np.where(te)[0]]

        # --- INNER: TODAS las decisiones con solo el pasado ---
        sel = _inner_select(train_rows, periods[tr], base_cols, families, cfg, fitter)
        cols = sel["columns"]
        decisions.append({
            "fold": fold_i, "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
            "train_period_max": int(periods[tr].max()), "test_period_min": int(periods[te].min()),
            "l2": sel["l2"], "calibration": sel["calibration"], "families": sel["families"],
        })

        # --- Refit en TODO el outer_train con la decision congelada ---
        X_tr = _matrix(train_rows, cols)
        X_te = _matrix(test_rows, cols)
        sw = _recency_weights(periods[tr], cfg.recency_half_life)
        model = fitter(X_tr, y_all[tr], cols, l2=sel["l2"], sample_weight=sw)
        p_tr = model.predict_proba(X_tr)
        cal = fit_calibrator(sel["calibration"], p_tr, y_all[tr])  # calibrador SOLO con train
        p_te = cal.apply(model.predict_proba(X_te))

        # --- Baselines apples-to-apples sobre el mismo test ---
        elo_te = np.array([_elo_prob(r) for r in test_rows])
        mkt_te = np.array([_num(r.get("opening_prob_t1")) for r in test_rows])

        oos["model"].extend(p_te.tolist())
        oos["elo"].extend(elo_te.tolist())
        oos["market"].extend(mkt_te.tolist())
        oos_y.extend(y_all[te].tolist())
        for r in test_rows:
            oos_meta.append({
                "period": int(r["period"]),
                "opening_prob_t1": _num(r.get("opening_prob_t1")),
                "opening_odds_t1": _num(r.get("opening_odds_t1")),
                "opening_odds_t2": _num(r.get("opening_odds_t2")),
                "closing_prob_t1": _num(r.get("closing_prob_t1")),
            })

    y = np.array(oos_y, dtype=int)
    result: dict[str, Any] = {
        "window": cfg.window,
        "n_eval": int(len(y)),
        "n_folds": len(decisions),
        "decisions": decisions,
        "models": {},
        "calibration": {},
        "baselines_present": ["elo", "market"],
    }
    for name, preds in oos.items():
        p = np.array(preds, dtype=float)
        finite = np.isfinite(p)
        if finite.sum() == 0:
            result["models"][name] = {"n": 0, "note": "sin predicciones finitas"}
            continue
        m = metric_dict(y[finite], np.clip(p[finite], 1e-9, 1 - 1e-9))
        result["models"][name] = m
        result["calibration"][name] = calibration_bins(y[finite], np.clip(p[finite], 1e-9, 1 - 1e-9))

    # 65% recomputado honesto: accuracy del favorito del modelo en el OOS externo.
    pm = np.array(oos["model"], dtype=float)
    result["recomputed_favorite_accuracy"] = float(np.mean((pm >= 0.5).astype(int) == y)) if len(y) else float("nan")
    result["betting"] = betting_metrics(y, pm, oos_meta)
    return result


def _elo_prob(row: dict[str, Any]) -> float:
    """Baseline Elo: usa la columna centrada si existe; si no, cae a 0.5."""
    for key in ("elo_prob_centered", "glicko_prob_centered"):
        v = _num(row.get(key))
        if np.isfinite(v):
            return float(np.clip(v + 0.5, 1e-9, 1 - 1e-9))
    return 0.5


# ===========================================================================
# Metrica de apuestas vs Model B (mercado): ROI + CLV sobre subconjunto con odds
# ===========================================================================
def betting_metrics(y: np.ndarray, p_model: np.ndarray, meta: list[dict[str, Any]]) -> dict[str, Any]:
    """ROI a stake plano y CLV sobre el subconjunto con cuotas de apertura.

    Apuesta al favorito del modelo cuando tiene EV+ contra la cuota de APERTURA
    (nunca la de cierre). CLV = prob implicita de cierre - de apertura del lado
    apostado (positivo = cerraste mejor que el mercado abrio).
    """
    bets = 0
    stake = 0.0
    ret = 0.0
    clv_vals: list[float] = []
    wins = 0
    for i, mrow in enumerate(meta):
        po1 = mrow.get("opening_odds_t1")
        po2 = mrow.get("opening_odds_t2")
        if not (np.isfinite(po1) and np.isfinite(po2) and po1 > 1 and po2 > 1):
            continue
        pm = float(p_model[i])
        side1 = pm >= 0.5
        odds = po1 if side1 else po2
        prob = pm if side1 else (1.0 - pm)
        ev = prob * odds - 1.0
        if ev <= 0:
            continue
        bets += 1
        stake += 1.0
        won = (int(y[i]) == 1) if side1 else (int(y[i]) == 0)
        if won:
            ret += (odds - 1.0)
            wins += 1
        else:
            ret -= 1.0
        # CLV: comparar prob implicita apertura vs cierre del lado apostado.
        op = mrow.get("opening_prob_t1")
        cp = mrow.get("closing_prob_t1")
        if np.isfinite(op) and np.isfinite(cp):
            open_side = op if side1 else (1.0 - op)
            close_side = cp if side1 else (1.0 - cp)
            clv_vals.append(float(close_side - open_side))
    return {
        "subset_with_odds": int(sum(1 for m in meta if np.isfinite(m.get("opening_odds_t1")))),
        "bets": bets,
        "hit_rate": (wins / bets) if bets else None,
        "roi_on_stake": (ret / stake) if stake else None,
        "clv_n": len(clv_vals),
        "clv_mean": (float(np.mean(clv_vals)) if clv_vals else None),
        "note": "stake plano; lado por favorito del modelo con EV+ vs apertura; cierre solo audita CLV.",
    }


# ===========================================================================
# Guardian anti-fuga temporal (tests que FALLAN si algo mira al futuro)
# ===========================================================================
def assert_point_in_time(
    rows: list[dict[str, Any]],
    build_features: Callable[[list[dict[str, Any]]], list[dict[str, float]]],
    feature_keys: Sequence[str],
    *,
    sample: int = 20,
    seed: int = 11,
    tol: float = 1e-9,
) -> None:
    """Verifica que las features de la fila i se reconstruyen con SOLO rows[:i+1].

    `build_features(subset)` debe devolver, para cada fila del subset, el dict de
    features emitido ANTES de observar esa fila. Comparamos la feature de la fila
    i calculada sobre todo el dataset contra la calculada sobre rows[:i+1]. Si
    difieren, hay leakage -> PointInTimeError.
    """
    full = build_features(rows)
    rng = np.random.default_rng(seed)
    n = len(rows)
    idxs = sorted(rng.choice(range(1, n), size=min(sample, n - 1), replace=False).tolist())
    for i in idxs:
        partial = build_features(rows[:i + 1])
        ref = full[i]
        got = partial[i]
        for k in feature_keys:
            a = _num(ref.get(k))
            b = _num(got.get(k))
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > tol:
                raise PointInTimeError(
                    f"FUGA: feature '{k}' de la fila {i} difiere al reconstruir solo con el pasado "
                    f"(completo={a}, pasado={b})."
                )

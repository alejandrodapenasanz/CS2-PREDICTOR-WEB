"""Suite de calibracion entrenada en un split TEMPORAL separado.

Compara Platt (sigmoide), isotonica, Beta (Kull et al.) y Venn-Abers (IVAP), y
selecciona por log loss en un holdout temporal. La salida calibrada de Model A
se cablea ANTES de la capa de EV.

Aviso: la isotonica necesita datos; con n pequeño puede sobreajustar. El selector
lo penaliza al medir en un holdout separado (no en el propio ajuste).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


@dataclass
class Calibrator:
    method: str
    _fn: Callable[[np.ndarray], np.ndarray]

    def apply(self, p: np.ndarray) -> np.ndarray:
        return _clip(self._fn(_clip(np.asarray(p, dtype=float))))


# --- ajustes individuales ---------------------------------------------------
def _fit_identity() -> Calibrator:
    return Calibrator("identity", lambda p: p)


def _fit_platt(p: np.ndarray, y: np.ndarray) -> Calibrator:
    z = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y, dtype=int))
    a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
    return Calibrator("platt", lambda q: 1 / (1 + np.exp(-(a * np.log(_clip(q) / (1 - _clip(q))) + b))))


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return Calibrator("isotonic", lambda q: iso.predict(np.asarray(q, dtype=float)))


def _fit_beta(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Beta calibration (Kull et al. 2017): logistica sobre [ln p, ln(1-p)]."""
    p = _clip(p)
    Z = np.column_stack([np.log(p), np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(Z, np.asarray(y, dtype=int))
    a, c = lr.coef_[0]
    d = float(lr.intercept_[0])

    def _apply(q: np.ndarray) -> np.ndarray:
        q = _clip(q)
        s = a * np.log(q) + c * np.log(1 - q) + d
        return 1 / (1 + np.exp(-s))

    return Calibrator("beta", _apply)


def _fit_venn_abers(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Venn-Abers inductivo (IVAP): p_cal = p1 / (1 - p0 + p1).

    Para cada score de test se ajusta isotonica anadiendo ese punto etiquetado 0
    (p0) y 1 (p1). Se deduplican scores para acotar el coste.
    """
    s_cal = np.asarray(p, dtype=float)
    y_cal = np.asarray(y, dtype=float)

    def _apply(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        uniq, inv = np.unique(q, return_inverse=True)
        out_u = np.empty_like(uniq)
        for j, s in enumerate(uniq):
            xs = np.append(s_cal, s)
            iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            p0 = float(iso0.fit(xs, np.append(y_cal, 0.0)).predict([s])[0])
            iso1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            p1 = float(iso1.fit(xs, np.append(y_cal, 1.0)).predict([s])[0])
            denom = (1 - p0 + p1)
            out_u[j] = (p1 / denom) if denom > 1e-12 else p1
        return out_u[inv]

    return Calibrator("venn_abers", _apply)


_FITTERS: dict[str, Callable[[np.ndarray, np.ndarray], Calibrator]] = {
    "identity": lambda p, y: _fit_identity(),
    "platt": _fit_platt,
    "isotonic": _fit_isotonic,
    "beta": _fit_beta,
    "venn_abers": _fit_venn_abers,
}

ALL_METHODS = ("identity", "platt", "isotonic", "beta", "venn_abers")


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> Calibrator:
    if method not in _FITTERS:
        raise ValueError(f"metodo desconocido: {method}")
    return _FITTERS[method](np.asarray(p, dtype=float), np.asarray(y, dtype=int))


def select_calibrator(
    p_fit: np.ndarray, y_fit: np.ndarray,
    p_val: np.ndarray, y_val: np.ndarray,
    methods: tuple[str, ...] = ALL_METHODS,
) -> tuple[Calibrator, dict[str, float]]:
    """Ajusta cada metodo en (p_fit,y_fit) y elige por log loss en (p_val,y_val)."""
    scores: dict[str, float] = {}
    best: Calibrator | None = None
    best_ll = np.inf
    for m in methods:
        try:
            cal = fit_calibrator(m, p_fit, y_fit)
            ll = _logloss(y_val, cal.apply(p_val))
        except Exception:
            continue
        scores[m] = ll
        if ll < best_ll:
            best_ll = ll
            best = cal
    if best is None:
        best = _fit_identity()
    return best, scores

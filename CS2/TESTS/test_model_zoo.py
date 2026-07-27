"""Tests del zoo de modelos, la suite de calibracion y el block-wise.

Rapidos (sin Optuna, modelos pequenos). Requiere numpy + scikit-learn; LightGBM
se testea si esta instalado. Valida la POLITICA DE MISSING (NaN nativo vs
impute+flag), que las calibraciones producen probabilidades validas y que el
driver anidado corre y elige un mejor combo.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

import numpy as np  # noqa: E402

from cs2model.model_zoo import available_models, build_model  # noqa: E402
from cs2model.calibration_suite import fit_calibrator, select_calibrator, ALL_METHODS  # noqa: E402
from cs2model.evaluation import EvalConfig, Family  # noqa: E402
from cs2model.blockwise import run_comparison, ComparisonConfig, select_families_cv, betting_clv  # noqa: E402

_SMALL = {
    "random_forest": {"n_estimators": 40},
    "extra_trees": {"n_estimators": 40},
    "hist_gb": {"max_iter": 60},
    "lightgbm": {"n_estimators": 60},
    "logistic_en": {},
    "xgboost": {"n_estimators": 60},
    "catboost": {"iterations": 60},
}


def _xy(n=160, d=4, seed=0, with_nan=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    p = 1 / (1 + np.exp(-(X @ w)))
    y = (rng.random(n) < p).astype(int)
    if with_nan:  # una columna con huecos (block-missing) + su flag implicito por NaN
        X[rng.random(n) < 0.4, d - 1] = np.nan
    return X, y


class ModelZooTests(unittest.TestCase):
    def test_available_models_present(self):
        av = available_models()
        for base in ("logistic_en", "random_forest", "extra_trees", "hist_gb"):
            self.assertIn(base, av)

    def test_every_model_fits_and_predicts_with_nan(self):
        X, y = _xy(with_nan=True)
        Xte, _ = _xy(n=40, seed=1, with_nan=True)
        for name in available_models():
            m = build_model(name, _SMALL.get(name, {}))
            m.fit(X, y)
            p = m.predict_proba(Xte)
            self.assertEqual(p.shape, (40,), f"{name}: shape")
            self.assertTrue(np.all((p > 0) & (p < 1)), f"{name}: probs fuera de (0,1)")
            self.assertFalse(np.any(np.isnan(p)), f"{name}: NaN en la prediccion")

    def test_missing_policy_declared(self):
        # arboles NaN-native declaran 'nan'; lineales/bosques 'impute'
        self.assertEqual(build_model("hist_gb").missing, "nan")
        self.assertEqual(build_model("logistic_en").missing, "impute")
        self.assertEqual(build_model("random_forest").missing, "impute")


class CalibrationSuiteTests(unittest.TestCase):
    def test_all_calibrators_produce_valid_probs(self):
        rng = np.random.default_rng(3)
        y = (rng.random(300) < 0.5).astype(int)
        p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 300), 1e-3, 1 - 1e-3)
        for method in ALL_METHODS:
            cal = fit_calibrator(method, p, y)
            q = cal.apply(p)
            self.assertTrue(np.all((q > 0) & (q < 1)), f"{method}: probs invalidas")
            self.assertEqual(q.shape, p.shape)

    def test_selector_returns_method_and_scores(self):
        rng = np.random.default_rng(4)
        y = (rng.random(400) < 0.5).astype(int)
        p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.25, 400), 1e-3, 1 - 1e-3)
        cal, scores = select_calibrator(p[:200], y[:200], p[200:], y[200:])
        self.assertIn(cal.method, ALL_METHODS)
        self.assertTrue(len(scores) >= 1)


def _synthetic_rows(seed=5, P=30, per=8):
    rng = random.Random(seed)
    T = 12
    strg = {t: rng.gauss(0, 1) for t in range(T)}
    rows = []
    for p in range(P):
        for _ in range(per):
            a, b = rng.sample(range(T), 2)
            d = strg[a] - strg[b]
            prob = 1 / (1 + math.exp(-d))
            lab = 1 if rng.random() < prob else 0
            cov = 1.0 if p >= P // 2 else 0.0
            rows.append({
                "label": lab, "period": p,
                "elo_prob_centered": prob - 0.5 + rng.gauss(0, 0.02),
                "noise1": rng.gauss(0, 1),
                "extra_feat": (d * 0.5 + rng.gauss(0, 0.3)) if cov else float("nan"),
                "extra_available": cov,
                "opening_prob_t1": prob, "opening_odds_t1": 1 / max(prob, .02),
                "opening_odds_t2": 1 / max(1 - prob, .02), "closing_prob_t1": prob,
            })
    return rows, ["elo_prob_centered", "noise1"], [Family("extra", ("extra_feat",), "extra_available", 0)]


class BlockwiseTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.base, self.fams = _synthetic_rows()
        self.cfg = EvalConfig(warmup_periods=8, gap_periods=1, outer_step=10, inner_step=8,
                              inner_warmup=5, window="expanding", train_width=20)

    def test_activation_is_data_driven(self):
        active, log = select_families_cv(self.rows[:160], self.base, self.fams, "logistic_en", self.cfg)
        self.assertIn("base_log_loss", log)
        self.assertIsInstance(active, list)  # puede entrar o no; la decision es por log loss OOS

    def test_comparison_runs_and_picks_best(self):
        comp = ComparisonConfig(models=("logistic_en",), assemblies=("indicators", "two_stage"), n_trials=0)
        res = run_comparison(self.rows, self.base, self.fams, self.cfg, comp)
        self.assertGreater(res["n_eval"], 0)
        self.assertIsNotNone(res["best_combo"])
        self.assertIn("elo", res["models"])
        self.assertIn("market", res["models"])
        for combo in ("logistic_en|indicators", "logistic_en|two_stage"):
            self.assertIn(combo, res["models"])

    def test_betting_clv_uses_opening_not_closing(self):
        y = np.array([1, 0, 1])
        p = np.array([0.7, 0.3, 0.8])
        base = {"opening_prob_t1": 0.6, "opening_odds_t1": 1.9, "opening_odds_t2": 2.1}
        a = betting_clv(y, p, [dict(base, closing_prob_t1=0.55)] * 3)
        b = betting_clv(y, p, [dict(base, closing_prob_t1=0.95)] * 3)
        self.assertEqual(a["bets"], b["bets"])
        self.assertEqual(a["roi_on_stake"], b["roi_on_stake"])           # el cierre no cambia la apuesta
        self.assertNotEqual(a["clv_vs_closing_mean"], b["clv_vs_closing_mean"])  # solo el CLV


if __name__ == "__main__":
    unittest.main()

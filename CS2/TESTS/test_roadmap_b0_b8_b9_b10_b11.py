from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.bayesian_bt import BayesianBradleyTerry
from cs2model.artifacts import ColumnSubsetEstimator, Component, ModelArtifact, ProbabilityColumnEstimator
from cs2model.compositional_bo3 import compositional_bo3_features, compositional_series_probability
from cs2model.features import (
    BAYES_BT_FEATURE_COLUMNS,
    BO3_COMPOSITIONAL_FEATURE_COLUMNS,
    KALMAN_FEATURE_COLUMNS,
    ChronologicalState,
)
from cs2model.kalman_rating import KalmanTeamRating
from cs2model.optuna_tuning import tune_logistic_c_purged
from train import select_feature_columns, super_learner_component_names, walk_forward


class BayesianBradleyTerryTests(unittest.TestCase):
    def test_partial_pooling_uncertainty_and_symmetry(self) -> None:
        model = BayesianBradleyTerry()
        a, b = model.default(), model.default()
        self.assertEqual(model.win_probability(a, b), 0.5)

        a, b = model.update(a, b, True)
        self.assertGreater(model.win_probability(a, b), 0.5)
        self.assertAlmostEqual(
            model.win_probability(a, b),
            1.0 - model.win_probability(b, a),
            places=12,
        )
        self.assertLess(a.variance, model.prior_variance)
        self.assertLess(abs(a.mean), 1.0)
        self.assertEqual((a.games, b.games), (1, 1))
        self.assertGreater(model.evolve(a, 70).variance, a.variance)


class KalmanRatingTests(unittest.TestCase):
    def test_state_update_reduces_uncertainty_and_is_symmetric(self) -> None:
        model = KalmanTeamRating()
        a, b = model.default(), model.default()
        a, b = model.update(a, b, True)
        self.assertGreater(model.win_probability(a, b), 0.5)
        self.assertAlmostEqual(
            model.win_probability(a, b),
            1.0 - model.win_probability(b, a),
            places=12,
        )
        self.assertLess(a.variance, model.prior_variance + model.process_variance)
        self.assertGreater(model.evolve(a, 70).variance, a.variance)


class ChronologicalAlternativeRatingTests(unittest.TestCase):
    @staticmethod
    def _match(index: int, team1: str = "alpha", team2: str = "beta") -> dict:
        date = datetime(2026, 1, 1) + timedelta(days=index)
        return {
            "id": str(index),
            "date": date.strftime("%Y-%m-%d"),
            "date_obj": date,
            "event": "Test",
            "format": "bo3",
            "team1": team1,
            "team2": team2,
            "team1_key": team1,
            "team2_key": team2,
            "score1": 2,
            "score2": 0,
            "team1_win": 1,
        }

    def test_b8_b9_features_are_point_in_time_and_antisymmetric(self) -> None:
        state = ChronologicalState()
        before = state.emit_features("alpha", "beta", datetime(2025, 12, 31))
        self.assertEqual(before["bayesian_bt_prob_centered"], 0.0)
        self.assertEqual(before["kalman_prob_centered"], 0.0)
        for index in range(6):
            state.observe(self._match(index))
        as_of = datetime(2026, 2, 1)
        forward = state.emit_features("alpha", "beta", as_of)
        reverse = state.emit_features("beta", "alpha", as_of)
        self.assertEqual(forward["bayesian_bt_available"], 1.0)
        self.assertEqual(forward["kalman_available"], 1.0)
        for column in ("bayesian_bt_mean_diff", "bayesian_bt_prob_centered", "kalman_mean_diff", "kalman_prob_centered"):
            self.assertAlmostEqual(forward[column], -reverse[column], places=12)


class CompositionalBO3Tests(unittest.TestCase):
    @staticmethod
    def _analytics() -> dict:
        rows = []
        values = {
            "mirage": ((65, 40, 60, 0), (45, 30, 5, 20)),
            "inferno": ((55, 30, 10, 10), (60, 35, 55, 0)),
            "nuke": ((58, 25, 5, 5), (52, 25, 5, 5)),
        }
        for map_name, sides in values.items():
            for team, (win, played, pick, ban) in zip(("Alpha", "Beta"), sides):
                rows.append({
                    "map": map_name,
                    "team": team,
                    "win_pct": win,
                    "played": played,
                    "first_pick_pct": pick,
                    "first_ban_pct": ban,
                })
        return {"available": True, "map_stats": rows}

    def test_formula_and_team_swap(self) -> None:
        self.assertAlmostEqual(compositional_series_probability(0.6, 0.6, 0.6), 0.648, places=12)
        analytics = self._analytics()
        forward = compositional_bo3_features(analytics, "Alpha", "Beta", "bo3")
        reverse = compositional_bo3_features(analytics, "Beta", "Alpha", "bo3")
        self.assertEqual(forward["bo3_compositional_available"], 1.0)
        self.assertAlmostEqual(
            forward["bo3_compositional_prob_centered"],
            -reverse["bo3_compositional_prob_centered"],
            places=12,
        )
        self.assertEqual(
            compositional_bo3_features(analytics, "Alpha", "Beta", "bo1")["bo3_compositional_available"],
            0.0,
        )


class AutomaticGatingAndWalkForwardTests(unittest.TestCase):
    def test_new_families_auto_gate_at_threshold(self) -> None:
        rows = [
            {
                "bayesian_bt_available": 1.0,
                "kalman_available": 1.0,
                "bo3_compositional_available": 1.0 if index < 500 else 0.0,
            }
            for index in range(900)
        ]
        columns, policies = select_feature_columns(rows)
        self.assertTrue(set(BAYES_BT_FEATURE_COLUMNS).issubset(columns))
        self.assertTrue(set(KALMAN_FEATURE_COLUMNS).issubset(columns))
        self.assertTrue(set(BO3_COMPOSITIONAL_FEATURE_COLUMNS).issubset(columns))
        self.assertTrue(policies["bayesian_bradley_terry"]["enabled"])
        self.assertTrue(policies["kalman_state_space"]["enabled"])
        self.assertTrue(policies["bo3_map_compositional"]["enabled"])

    def test_rating_candidates_are_evaluated_in_walk_forward(self) -> None:
        rng = np.random.RandomState(17)
        n = 240
        strength = rng.normal(size=n)
        probability = 1.0 / (1.0 + np.exp(-strength))
        y = (rng.uniform(size=n) < probability).astype(int)
        centered = probability - 0.5
        X = np.column_stack([centered, centered, centered, centered])
        columns = [
            "elo_prob_centered",
            "glicko_prob_centered",
            "bayesian_bt_prob_centered",
            "kalman_prob_centered",
        ]
        periods = np.repeat(np.arange(12), 20)
        meta = [
            {
                "id": str(index),
                "date": "2026-01-01",
                "event": "Test",
                "team1": "A",
                "team2": "B",
                "format": "bo3",
            }
            for index in range(n)
        ]
        predictions = walk_forward(
            X,
            y,
            periods,
            meta,
            columns,
            warmup_weeks=2,
            min_train=40,
            algorithm_kinds=("logistic",),
            optuna_trials=0,
        )
        self.assertGreater(len(predictions["bayesian_bt_cal"]), 0)
        self.assertEqual(len(predictions["bayesian_bt_cal"]), len(predictions["kalman_cal"]))

    def test_b8_b9_are_explicit_super_learner_members(self) -> None:
        names = super_learner_component_names(
            ("logistic", "gbm"),
            ("bayesian_bt_cal", "kalman_cal"),
        )
        self.assertEqual(
            names,
            ["logistic_cal", "lightgbm_cal", "bayesian_bt_cal", "kalman_cal"],
        )


class OptunaPurgedCVTests(unittest.TestCase):
    def test_optuna_uses_only_purged_inner_folds_and_log_loss(self) -> None:
        rng = np.random.RandomState(23)
        n = 360
        x = rng.normal(size=n)
        y = (x + rng.normal(scale=0.8, size=n) > 0).astype(int)
        X = np.column_stack([x, rng.normal(size=n)])
        periods = np.repeat(np.arange(18), 20)
        result = tune_logistic_c_purged(
            X,
            y,
            periods,
            ["signal", "noise"],
            {"signal"},
            n_trials=2,
            gap=1,
            max_inner_folds=3,
            min_train=120,
        )
        self.assertTrue(result["enabled"])
        self.assertEqual(result["objective"], "purged_inner_cv_log_loss")
        self.assertEqual(result["gap"], 1)
        self.assertEqual(result["inner_folds"], 3)
        self.assertTrue(math.isfinite(result["best_log_loss"]))
        for period in result["inner_periods"]:
            self.assertGreaterEqual(int(np.sum(periods < period - 1)), 120)


class MixedArtifactTests(unittest.TestCase):
    def test_generic_and_rating_components_share_one_artifact_safely(self) -> None:
        generic = ColumnSubsetEstimator(
            estimator=ProbabilityColumnEstimator(column_index=0),
            column_indices=[1],
        )
        rating = ProbabilityColumnEstimator(column_index=0)
        artifact = ModelArtifact(
            feature_columns=["rating_probability", "generic_probability"],
            components=[
                Component("generic", generic, None, 0.5),
                Component("rating", rating, None, 0.5),
            ],
        )
        probability = artifact.predict_proba_team1([
            {"rating_probability": 0.2, "generic_probability": 0.1}
        ])[0]
        self.assertAlmostEqual(probability, 0.65, places=12)


if __name__ == "__main__":
    unittest.main()

"""Pruebas offline de los baselines causales de ranking y mercado."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.baselines import (  # noqa: E402
    BaselineDataError,
    RankProbabilityBaseline,
    market_baseline_predictions,
    ranking_favorite_predictions,
)


class RankingBaselineTest(unittest.TestCase):
    """Comprueba favorito determinista y baseline probabilístico rank-only."""

    def test_ranking_favorite_marks_ties_and_missing_as_unavailable(
        self,
    ) -> None:
        """No fuerza un ganador cuando el ranking no lo determina."""

        frame = pd.DataFrame(
            {"rank_diff": [-10.0, 5.0, 0.0, np.nan]},
            index=[4, 5, 6, 7],
        )

        predictions = ranking_favorite_predictions(frame)

        self.assertEqual(predictions.predicted_y.loc[4], 1)
        self.assertEqual(predictions.predicted_y.loc[5], 0)
        self.assertTrue(pd.isna(predictions.predicted_y.loc[6]))
        self.assertTrue(pd.isna(predictions.predicted_y.loc[7]))
        self.assertEqual(predictions.probability_coverage, 0.0)
        self.assertEqual(predictions.decision_coverage, 0.5)

    def test_rank_probability_is_fitted_and_preserves_missing_coverage(
        self,
    ) -> None:
        """Ajusta en train y no imputa rankings ausentes en evaluación."""

        train = pd.DataFrame(
            {"rank_diff": [-100.0, -50.0, -10.0, 10.0, 50.0, 100.0]}
        )
        target = pd.Series([1, 1, 1, 0, 0, 0])
        baseline = RankProbabilityBaseline().fit(train, target)

        predictions = baseline.predict(
            pd.DataFrame({"rank_diff": [-20.0, 20.0, np.nan]})
        )

        self.assertGreater(
            float(predictions.probability_a.iloc[0]), 0.5
        )
        self.assertLess(
            float(predictions.probability_a.iloc[1]), 0.5
        )
        self.assertTrue(pd.isna(predictions.probability_a.iloc[2]))
        self.assertAlmostEqual(
            predictions.probability_coverage, 2.0 / 3.0
        )

    def test_rank_probability_requires_fit_and_both_classes(self) -> None:
        """Rechaza inferencia prematura y folds degenerados."""

        frame = pd.DataFrame({"rank_diff": [-1.0, 1.0]})
        with self.assertRaises(BaselineDataError):
            RankProbabilityBaseline().predict(frame)
        with self.assertRaises(BaselineDataError):
            RankProbabilityBaseline().fit(
                frame, pd.Series([1, 1])
            )


class MarketBaselineTest(unittest.TestCase):
    """Comprueba cobertura honesta de las probabilidades de-vigadas."""

    def test_market_uses_only_complete_valid_probability_pairs(self) -> None:
        """Conserva faltantes, probabilidades y empate sin inventar favorito."""

        frame = pd.DataFrame(
            {
                "market_probability_a": [0.60, np.nan, 0.50, 0.40],
                "market_probability_b": [0.40, np.nan, 0.50, np.nan],
            }
        )

        predictions = market_baseline_predictions(frame)

        self.assertEqual(
            list(predictions.probability_a.iloc[[0, 2]].astype(float)),
            [0.6, 0.5],
        )
        self.assertTrue(pd.isna(predictions.probability_a.iloc[1]))
        self.assertTrue(pd.isna(predictions.probability_a.iloc[3]))
        self.assertEqual(predictions.predicted_y.iloc[0], 1)
        self.assertTrue(pd.isna(predictions.predicted_y.iloc[2]))
        self.assertEqual(predictions.probability_coverage, 0.5)
        self.assertEqual(predictions.decision_coverage, 0.25)

    def test_market_rejects_present_malformed_probabilities(self) -> None:
        """Distingue ausencia legítima de un par presente incoherente."""

        malformed = pd.DataFrame(
            {
                "market_probability_a": [0.8],
                "market_probability_b": [0.3],
            }
        )

        with self.assertRaises(BaselineDataError):
            market_baseline_predictions(malformed)
        non_numeric = pd.DataFrame(
            {
                "market_probability_a": ["unknown"],
                "market_probability_b": [0.4],
            }
        )
        with self.assertRaises(BaselineDataError):
            market_baseline_predictions(non_numeric)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.artifacts import Component, ModelArtifact
from cs2model.odds import (
    devig_two_way,
    no_odds_features,
    resolve_opening_odds,
    validate_opening_odds,
)
from train import _architecture_report


class ConstantEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


class OddsContractTests(unittest.TestCase):
    def test_devigged_market_sums_to_one(self) -> None:
        normalized = devig_two_way(1.80, 2.10)
        self.assertIsNotNone(normalized)
        probability1, probability2, overround = normalized or (0.0, 0.0, 0.0)
        self.assertAlmostEqual(probability1 + probability2, 1.0, places=12)
        self.assertGreater(overround, 1.0)

    def test_opening_requires_coherent_pair_strictly_before_kickoff(self) -> None:
        market = {"team1_decimal": 1.80, "team2_decimal": 2.10}
        self.assertIsNotNone(
            validate_opening_odds(
                market,
                captured_at_utc="2026-08-24T15:59:59Z",
                kickoff_utc="2026-08-24T16:00:00Z",
                source="test",
            )
        )
        self.assertIsNone(
            validate_opening_odds(
                market,
                captured_at_utc="2026-08-24T16:00:00Z",
                kickoff_utc="2026-08-24T16:00:00Z",
                source="test",
            )
        )
        self.assertIsNone(
            validate_opening_odds(
                {"team1_decimal": 1.01, "team2_decimal": 1.01},
                captured_at_utc="2026-08-24T15:00:00Z",
                kickoff_utc="2026-08-24T16:00:00Z",
                source="test",
            )
        )

    def test_prior_opening_is_recovered_when_current_scrape_has_no_market(self) -> None:
        recovered = resolve_opening_odds(
            current_market=None,
            current_captured_at_utc="2026-08-24T12:00:00Z",
            stored_opening={
                "captured_at": "2026-08-23T18:00:00Z",
                "team1_decimal": 1.80,
                "team2_decimal": 2.10,
                "bookmaker_count": 2,
            },
            kickoff_utc="2026-08-24T16:00:00Z",
        )
        self.assertIsNotNone(recovered)
        self.assertTrue(recovered and recovered.recovered)
        self.assertEqual(recovered and recovered.source, "stored_opening")


class OddsRouterTests(unittest.TestCase):
    @staticmethod
    def _row(with_odds: bool) -> dict[str, float]:
        row = {"rating": 0.2}
        row.update(no_odds_features())
        if with_odds:
            row.update(
                {
                    "opening_odds_prob_centered": 0.12,
                    "opening_odds_confidence": 0.12,
                    "opening_bookmaker_count_log": 1.1,
                    "odds_available": 1.0,
                }
            )
        return row

    def test_two_model_router_uses_primary_only_for_valid_odds(self) -> None:
        artifact = ModelArtifact(
            feature_columns=["rating"],
            components=[Component("reserve", ConstantEstimator(0.40), None)],
            prediction_architecture="router_two_models",
            odds_feature_columns=[
                "rating",
                "opening_odds_prob_centered",
                "opening_odds_confidence",
                "opening_bookmaker_count_log",
                "odds_available",
            ],
            odds_components=[Component("primary", ConstantEstimator(0.70), None)],
        )
        probabilities = artifact.predict_proba_team1([self._row(True), self._row(False)])
        np.testing.assert_allclose(probabilities, [0.70, 0.40])
        self.assertEqual(artifact.prediction_regime(self._row(True)), "odds")
        self.assertEqual(artifact.prediction_regime(self._row(False)), "no_odds")

    def test_single_mixed_model_always_returns_a_prediction(self) -> None:
        artifact = ModelArtifact(
            feature_columns=["rating"],
            components=[Component("reserve", ConstantEstimator(0.40), None)],
            prediction_architecture="single_mixed_lgbm",
            mixed_feature_columns=[
                "rating",
                "opening_odds_prob_centered",
                "opening_odds_confidence",
                "opening_bookmaker_count_log",
                "odds_available",
            ],
            mixed_components=[Component("mixed", ConstantEstimator(0.60), None)],
        )
        probabilities = artifact.predict_proba_team1([self._row(True), self._row(False)])
        np.testing.assert_allclose(probabilities, [0.60, 0.60])

    def test_architecture_report_separates_both_regimes_on_common_support(
        self,
    ) -> None:
        common = [
            {
                "match_id": "1",
                "actual": 1,
                "prediction_regime": "odds",
            },
            {
                "match_id": "2",
                "actual": 0,
                "prediction_regime": "no_odds",
            },
        ]
        report = _architecture_report(
            {
                "router_two_models": [
                    {**common[0], "prob_team1": 0.80},
                    {**common[1], "prob_team1": 0.30},
                ],
                "single_mixed_lgbm": [
                    {**common[0], "prob_team1": 0.70},
                    {**common[1], "prob_team1": 0.40},
                ],
            }
        )

        self.assertEqual(report["n_common_holdout"], 2)
        self.assertEqual(report["no_odds_fraction"], 0.5)
        for architecture in ("router_two_models", "single_mixed_lgbm"):
            self.assertEqual(report["regimes"][architecture]["odds"]["n"], 1)
            self.assertEqual(report["regimes"][architecture]["no_odds"]["n"], 1)
            self.assertIn("calibration", report["regimes"][architecture]["odds"])


if __name__ == "__main__":
    unittest.main()

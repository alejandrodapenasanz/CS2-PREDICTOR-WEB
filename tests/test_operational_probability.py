import unittest
from datetime import datetime, timedelta

from DAILY_SNAPSHOTS.enrich_predictions import (
    decision_probability_team1,
    favorite_upset_summary,
    reliability_adjusted_probability,
    staking_recommendation,
)


class OperationalProbabilityTests(unittest.TestCase):
    def test_reliability_adjustment_shrinks_toward_coinflip(self):
        self.assertAlmostEqual(reliability_adjusted_probability(0.80, 0.25), 0.575)
        self.assertAlmostEqual(reliability_adjusted_probability(0.20, 0.25), 0.425)

    def test_decision_probability_uses_dynamic_market_prior_before_learning(self):
        decision = decision_probability_team1(0.35, 0.80, 0.25)
        self.assertEqual(decision["source"], "model_market_dynamic_prior")
        self.assertAlmostEqual(decision["market_weight"], 0.2375)
        self.assertAlmostEqual(decision["prob_team1"], 0.456875)

    def test_decision_probability_uses_learned_policy_when_available(self):
        decision = decision_probability_team1(
            0.35,
            0.80,
            0.25,
            policy={"available": True, "production_market_weight": 0.30, "source": "learned_expanding_walk_forward"},
        )
        self.assertEqual(decision["source"], "model_market_learned_walk_forward")
        self.assertAlmostEqual(decision["market_weight"], 0.30)
        self.assertAlmostEqual(decision["prob_team1"], 0.485)

    def test_staking_uses_decision_probability_not_raw_model_probability(self):
        entry = {
            "format": "bo3",
            "team1": {"name": "A"},
            "team2": {"name": "B"},
            "odds": {
                "available": True,
                "average": {
                    "team1_decimal": 1.8,
                    "team2_decimal": 1.9,
                    "team1_implied_prob_norm": 0.52,
                    "team2_implied_prob_norm": 0.48,
                },
            },
        }
        features = {
            "matches_team1": 30,
            "matches_team2": 30,
            "glicko_rd_team1": 80,
            "glicko_rd_team2": 80,
            "player_coverage_team1": 1.0,
            "player_coverage_team2": 1.0,
        }
        prediction = {
            "model_prob_team1": 0.80,
            "decision_prob_team1": 0.50,
            "decision_probability_source": "model_reliability_shrink",
        }
        market = {
            "available": True,
            "latest": {"team1_decimal": 1.8, "team2_decimal": 1.9},
            "consensus_label": "high",
        }
        result = staking_recommendation(entry, features, prediction, market, {"production_model": None})
        self.assertEqual(result["action"], "no_bet")
        self.assertEqual(result["reason_code"], "no_positive_ev")

    def test_favorite_upset_summary_counts_recent_favorite_losses(self):
        asof = datetime(2026, 7, 3, 18, 0)
        records = [
            {"date_obj": asof - timedelta(days=5), "team1_key": "a", "team2_key": "b", "favorite_key": "a", "favorite_lost": True, "favorite_source": "market_opening_or_snapshot_odds"},
            {"date_obj": asof - timedelta(days=15), "team1_key": "a", "team2_key": "c", "favorite_key": "a", "favorite_lost": False, "favorite_source": "market_opening_or_snapshot_odds"},
            {"date_obj": asof - timedelta(days=20), "team1_key": "d", "team2_key": "a", "favorite_key": "d", "favorite_lost": False, "favorite_source": "market_opening_or_snapshot_odds"},
            {"date_obj": asof - timedelta(days=30), "team1_key": "a", "team2_key": "f", "favorite_key": None, "favorite_lost": False, "favorite_source": "missing_market_odds"},
            {"date_obj": asof - timedelta(days=120), "team1_key": "a", "team2_key": "e", "favorite_key": "a", "favorite_lost": True, "favorite_source": "market_opening_or_snapshot_odds"},
        ]

        summary = favorite_upset_summary(records, "a", asof, window_days=90)

        self.assertEqual(summary["favorite_losses"], 1)
        self.assertEqual(summary["favorite_matches"], 2)
        self.assertEqual(summary["played_matches"], 4)
        self.assertEqual(summary["market_odds_matches"], 3)
        self.assertAlmostEqual(summary["favorite_loss_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

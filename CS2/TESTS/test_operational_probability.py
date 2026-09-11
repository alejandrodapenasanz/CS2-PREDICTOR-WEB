import unittest
import sqlite3
from datetime import datetime, timedelta

from PIPELINE.enrich_predictions import (
    build_favorite_upset_records,
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
            {
                "date_obj": asof - timedelta(days=5),
                "team1_key": "a",
                "team2_key": "b",
                "favorite_key": "a",
                "favorite_lost": True,
                "favorite_source": "market_opening_or_snapshot_odds",
            },
            {
                "date_obj": asof - timedelta(days=15),
                "team1_key": "a",
                "team2_key": "c",
                "favorite_key": "a",
                "favorite_lost": False,
                "favorite_source": "market_opening_or_snapshot_odds",
            },
            {
                "date_obj": asof - timedelta(days=20),
                "team1_key": "d",
                "team2_key": "a",
                "favorite_key": "d",
                "favorite_lost": False,
                "favorite_source": "market_opening_or_snapshot_odds",
            },
            {
                "date_obj": asof - timedelta(days=30),
                "team1_key": "a",
                "team2_key": "f",
                "favorite_key": None,
                "favorite_lost": False,
                "favorite_source": "missing_market_odds",
            },
            {
                "date_obj": asof - timedelta(days=120),
                "team1_key": "a",
                "team2_key": "e",
                "favorite_key": "a",
                "favorite_lost": True,
                "favorite_source": "market_opening_or_snapshot_odds",
            },
        ]

        summary = favorite_upset_summary(records, "a", asof, window_days=90)

        self.assertEqual(summary["favorite_losses"], 1)
        self.assertEqual(summary["favorite_matches"], 2)
        self.assertEqual(summary["played_matches"], 4)
        self.assertFalse(summary["odds_required"])
        self.assertEqual(summary["history_scope"], "rolling_window")
        self.assertAlmostEqual(summary["favorite_loss_rate"], 0.5)

    def test_model_favorite_history_counts_no_odds_and_requires_51_percent(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                hltv_id TEXT,
                name TEXT NOT NULL
            );
            CREATE TABLE prediction_ledger (
                ledger_id INTEGER PRIMARY KEY,
                hltv_match_id TEXT,
                kickoff_utc TEXT,
                prob_team1 REAL,
                actual_team1_win INTEGER,
                prediction_regime TEXT,
                team1_id INTEGER,
                team2_id INTEGER,
                ledger_status TEXT
            );
            INSERT INTO teams(team_id, hltv_id, name) VALUES
                (1, '101', 'Alpha'),
                (2, '202', 'Beta');
            INSERT INTO prediction_ledger(
                ledger_id, hltv_match_id, kickoff_utc, prob_team1,
                actual_team1_win, prediction_regime, team1_id, team2_id,
                ledger_status
            ) VALUES
                (1, 'm1', '2026-01-01T12:00:00Z', 0.70, 0, 'no_odds', 1, 2, 'evaluated'),
                (2, 'm2', '2026-02-01T12:00:00Z', 0.40, 0, 'no_odds', 1, 2, 'evaluated'),
                (3, 'm3', '2026-03-01T12:00:00Z', 0.505, 1, 'no_odds', 1, 2, 'evaluated'),
                (4, 'm4', '2026-04-01T12:00:00Z', 0.80, 0, 'odds', 1, 2, 'invalid');
            """
        )

        records = build_favorite_upset_records(conn)
        summary = favorite_upset_summary(
            records,
            "hltv:101",
            datetime(2026, 8, 1),
        )
        conn.close()

        self.assertEqual([record["match_id"] for record in records], ["m1", "m2"])
        self.assertEqual(records[0]["prediction_regime"], "no_odds")
        self.assertEqual(summary["favorite_losses"], 1)
        self.assertEqual(summary["favorite_matches"], 1)
        self.assertAlmostEqual(summary["favorite_loss_rate"], 1.0)
        self.assertFalse(summary["odds_required"])
        self.assertEqual(summary["history_scope"], "all_database_history")


if __name__ == "__main__":
    unittest.main()

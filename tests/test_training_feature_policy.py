from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.dataio import external_snapshot_features_asof
from cs2model.features import (
    ANALYTICS_FEATURE_COLUMNS,
    EVENT_HISTORY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MAP_ASSET_FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    RANKING_FEATURE_COLUMNS,
    ROSTER_FEATURE_COLUMNS,
)
from train import (
    ANALYTICS_MIN_TRAIN_ROWS,
    EVENT_HISTORY_MIN_TRAIN_ROWS,
    MAP_ASSET_MIN_TRAIN_ROWS,
    PLAYER_MIN_TRAIN_ROWS,
    RANKING_MIN_TRAIN_ROWS,
    ROSTER_MIN_TRAIN_ROWS,
    economic_backtest,
    favorite_accuracy_bands,
    select_feature_columns,
)


class TrainingFeaturePolicyTests(unittest.TestCase):
    def test_favorite_accuracy_bands_use_walk_forward_favorite(self) -> None:
        predictions = [
            {"prob_team1": 0.55, "actual": 1},
            {"prob_team1": 0.45, "actual": 0},
            {"prob_team1": 0.65, "actual": 1},
            {"prob_team1": 0.75, "actual": 0},
            {"prob_team1": 0.95, "actual": 1},
        ]

        result = favorite_accuracy_bands(predictions)

        self.assertEqual(result["n"], 5)
        self.assertEqual(result["correct"], 4)
        self.assertAlmostEqual(result["accuracy"], 0.8)
        self.assertEqual(result["bands"][0]["n"], 2)
        self.assertEqual(result["bands"][0]["correct"], 2)
        self.assertAlmostEqual(result["bands"][0]["average_predicted_probability"], 0.55)
        self.assertFalse(result["bands"][0]["representative"])

    def test_economic_backtest_never_bets_against_model_favorite(self) -> None:
        rows = [
            {
                "id": "favorite-team1",
                "date": "2026-01-01",
                "team1_win": True,
                "opening_odds_decimal_t1": 2.20,
                "opening_odds_decimal_t2": 1.60,
            },
            {
                "id": "favorite-team2",
                "date": "2026-01-02",
                "team1_win": False,
                "opening_odds_decimal_t1": 1.60,
                "opening_odds_decimal_t2": 2.20,
            },
        ]
        predictions = [
            {"match_id": "favorite-team1", "prob_team1": 0.60},
            {"match_id": "favorite-team2", "prob_team1": 0.40},
        ]

        result = economic_backtest(rows, predictions)

        self.assertEqual([bet["side"] for bet in result["bets"]], ["team1", "team2"])
        self.assertTrue(all(bet["probability"] >= 0.5 for bet in result["bets"]))
        self.assertEqual(
            result["staking"],
            "model_favorite_only_quarter_kelly_cap_2_5pct",
        )

    def test_analytics_features_are_gated_until_enough_closed_rows(self) -> None:
        rows = [{"analytics_available": 1.0} for _ in range(ANALYTICS_MIN_TRAIN_ROWS - 1)]
        columns, policy = select_feature_columns(rows)

        self.assertFalse(policy["analytics"]["enabled"])
        self.assertEqual(policy["analytics"]["available_rows"], ANALYTICS_MIN_TRAIN_ROWS - 1)
        self.assertEqual(columns, list(FEATURE_COLUMNS))
        self.assertFalse(any(c in columns for c in ANALYTICS_FEATURE_COLUMNS))

    def test_analytics_features_are_added_automatically_after_threshold(self) -> None:
        rows = [{"analytics_available": 1.0} for _ in range(ANALYTICS_MIN_TRAIN_ROWS)]
        columns, policy = select_feature_columns(rows)

        self.assertTrue(policy["analytics"]["enabled"])
        self.assertEqual(policy["analytics"]["available_rows"], ANALYTICS_MIN_TRAIN_ROWS)
        self.assertEqual(columns, list(FEATURE_COLUMNS) + list(ANALYTICS_FEATURE_COLUMNS))
        self.assertTrue(all(c in columns for c in ANALYTICS_FEATURE_COLUMNS))

    def test_player_features_are_gated_until_enough_closed_rows(self) -> None:
        rows = [{"player_snapshot_available": 1.0} for _ in range(PLAYER_MIN_TRAIN_ROWS - 1)]
        columns, policy = select_feature_columns(rows)

        self.assertFalse(policy["player_snapshots"]["enabled"])
        self.assertEqual(policy["player_snapshots"]["available_rows"], PLAYER_MIN_TRAIN_ROWS - 1)
        self.assertFalse(any(c in columns for c in PLAYER_FEATURE_COLUMNS))

    def test_player_features_are_added_automatically_after_threshold(self) -> None:
        rows = [{"player_snapshot_available": 1.0} for _ in range(PLAYER_MIN_TRAIN_ROWS)]
        columns, policy = select_feature_columns(rows)

        self.assertTrue(policy["player_snapshots"]["enabled"])
        self.assertEqual(policy["player_snapshots"]["available_rows"], PLAYER_MIN_TRAIN_ROWS)
        self.assertTrue(all(c in columns for c in PLAYER_FEATURE_COLUMNS))

    def test_every_supported_extended_family_uses_automatic_threshold(self) -> None:
        families = [
            ("map_box_scores", "asset_available", MAP_ASSET_MIN_TRAIN_ROWS, MAP_ASSET_FEATURE_COLUMNS),
            ("event_history", "event_history_available", EVENT_HISTORY_MIN_TRAIN_ROWS, EVENT_HISTORY_FEATURE_COLUMNS),
            ("rankings", "ranking_available", RANKING_MIN_TRAIN_ROWS, RANKING_FEATURE_COLUMNS),
            ("roster", "roster_available", ROSTER_MIN_TRAIN_ROWS, ROSTER_FEATURE_COLUMNS),
        ]
        for family, availability, threshold, feature_columns in families:
            with self.subTest(family=family, state="below"):
                columns, policy = select_feature_columns([{availability: 1.0} for _ in range(threshold - 1)])
                self.assertFalse(policy[family]["enabled"])
                self.assertFalse(any(col in columns for col in feature_columns))
            with self.subTest(family=family, state="ready"):
                columns, policy = select_feature_columns([{availability: 1.0} for _ in range(threshold)])
                self.assertTrue(policy[family]["enabled"])
                self.assertTrue(all(col in columns for col in feature_columns))
                self.assertEqual(policy[family]["activation"], "automatic_at_training_time")

    def test_ranking_and_roster_join_rejects_future_snapshots(self) -> None:
        past = datetime(2026, 1, 1, 10)
        future = datetime(2026, 1, 3, 10)
        store = {
            "rankings": {
                ("1", "hltv"): {
                    "dates": [past, future],
                    "rows": [
                        {"position": 10, "points": 500, "captured_at_utc": "2026-01-01T10:00:00Z"},
                        {"position": 1, "points": 9999, "captured_at_utc": "2026-01-03T10:00:00Z"},
                    ],
                },
                ("2", "hltv"): {
                    "dates": [past],
                    "rows": [{"position": 20, "points": 300, "captured_at_utc": "2026-01-01T10:00:00Z"}],
                },
            },
            "rosters": {
                "1": [{"player_id": i, "valid_from": "2025-12-01T00:00:00Z", "valid_to": None} for i in range(1, 6)],
                "2": [{"player_id": i, "valid_from": "2025-12-15T00:00:00Z", "valid_to": None} for i in range(6, 11)],
            },
        }

        result = external_snapshot_features_asof(store, "1", "2", "2026-01-02T10:00:00Z")
        ranking = result["ranking_snapshot_features"]
        roster = result["roster_snapshot_features"]

        self.assertEqual(ranking["ranking_hltv_position_advantage"], 10.0)
        self.assertEqual(ranking["ranking_hltv_points_diff"], 200.0)
        self.assertEqual(ranking["ranking_available"], 1.0)
        self.assertEqual(roster["roster_available"], 1.0)
        self.assertGreater(roster["roster_days_log_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()

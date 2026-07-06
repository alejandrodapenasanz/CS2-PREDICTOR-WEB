from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import ANALYTICS_FEATURE_COLUMNS, FEATURE_COLUMNS
from train import ANALYTICS_MIN_TRAIN_ROWS, select_feature_columns


class TrainingFeaturePolicyTests(unittest.TestCase):
    def test_analytics_features_are_gated_until_enough_closed_rows(self) -> None:
        rows = [{"analytics_available": 1.0} for _ in range(ANALYTICS_MIN_TRAIN_ROWS - 1)]
        columns, policy = select_feature_columns(rows)

        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["available_rows"], ANALYTICS_MIN_TRAIN_ROWS - 1)
        self.assertEqual(columns, list(FEATURE_COLUMNS))
        self.assertFalse(any(c in columns for c in ANALYTICS_FEATURE_COLUMNS))

    def test_analytics_features_are_added_automatically_after_threshold(self) -> None:
        rows = [{"analytics_available": 1.0} for _ in range(ANALYTICS_MIN_TRAIN_ROWS)]
        columns, policy = select_feature_columns(rows)

        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["available_rows"], ANALYTICS_MIN_TRAIN_ROWS)
        self.assertEqual(columns, list(FEATURE_COLUMNS) + list(ANALYTICS_FEATURE_COLUMNS))
        self.assertTrue(all(c in columns for c in ANALYTICS_FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from analyze_walkforward_errors import (
    benjamini_hochberg,
    orient_value,
    wilson_interval,
)


class WalkForwardErrorAuditTests(unittest.TestCase):
    def test_directional_features_are_oriented_to_favorite(self) -> None:
        directional = {"glicko_diff"}
        self.assertEqual(orient_value("glicko_diff", 25.0, 1, directional), 25.0)
        self.assertEqual(orient_value("glicko_diff", 25.0, -1, directional), -25.0)
        self.assertEqual(orient_value("glicko_rd_sum", 150.0, -1, directional), 150.0)

    def test_benjamini_hochberg_preserves_ordered_fdr(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, None])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertIsNone(adjusted[3])

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(45, 100)
        self.assertLess(low, 0.45)
        self.assertGreater(high, 0.45)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()

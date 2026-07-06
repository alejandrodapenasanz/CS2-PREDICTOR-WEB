from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import ChronologicalState


def match(fmt: str, idx: int, team1_win: bool = True) -> dict:
    return {
        "id": str(idx),
        "team1_key": "alpha",
        "team2_key": "beta",
        "team1": "Alpha",
        "team2": "Beta",
        "score1": 1 if fmt == "bo1" else 2,
        "score2": 0 if team1_win else 2,
        "team1_win": int(team1_win),
        "date": f"2026-01-{idx:02d}",
        "date_obj": datetime(2026, 1, idx),
        "event": "Format Cup",
        "format": fmt,
    }


class FormatFeatureTests(unittest.TestCase):
    def test_format_specific_history_does_not_leak_between_bo1_and_bo3(self) -> None:
        state = ChronologicalState()
        for idx in range(1, 4):
            state.observe(match("bo1", idx, team1_win=True))

        bo1 = state.emit_features("alpha", "beta", datetime(2026, 1, 10), "Format Cup", "bo1")
        bo3 = state.emit_features("alpha", "beta", datetime(2026, 1, 10), "Format Cup", "bo3")

        self.assertEqual(bo1["format_matches_log_diff"], 0.0)
        self.assertGreater(bo1["format_winrate_diff"], 0.0)
        self.assertGreater(bo1["format_avg_score_diff"], 0.0)
        self.assertGreater(bo1["format_h2h_matches"], 0.0)
        self.assertEqual(bo3["format_matches_log_diff"], 0.0)
        self.assertEqual(bo3["format_winrate_diff"], 0.0)
        self.assertEqual(bo3["format_avg_score_diff"], 0.0)
        self.assertEqual(bo3["format_h2h_matches"], 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model import dataio


def _base_record() -> dict:
    return {
        "status": "completed",
        "score": {"team1": 2, "team2": 1},
        "date": "2026-06-01",
        "event": "Leak Test Cup",
        "format": "bo3",
        "detail": {
            "match": {
                "team1": {"name": "Alpha", "id": "10"},
                "team2": {"name": "Beta", "id": "20"},
            }
        }
    }


class OddsPointInTimeTests(unittest.TestCase):
    def load_one(self, record: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            master = Path(tmp) / "matches.json"
            master.write_text(json.dumps({"1": record}), encoding="utf-8")
            return dataio.load_daily_completed(master)[0]

    def test_completed_daily_row_uses_opening_odds_not_closing(self) -> None:
        record = _base_record()
        record["opening_odds"] = {
            "captured_at": "2026-06-01T08:00:00Z",
            "team1_implied_prob_norm": 0.25,
            "team2_implied_prob_norm": 0.75,
            "bookmaker_count": 2,
        }
        record["closing_odds"] = {
            "captured_at": "2026-06-01T11:55:00Z",
            "team1_implied_prob_norm": 0.80,
            "team2_implied_prob_norm": 0.20,
            "bookmaker_count": 2,
        }

        row = self.load_one(record)

        self.assertEqual(row["opening_odds_t1"], 0.25)
        self.assertEqual(row["opening_odds_t2"], 0.75)
        self.assertEqual(row["opening_odds_captured_at"], "2026-06-01T08:00:00Z")

    def test_completed_daily_row_falls_back_to_first_odds_history_point(self) -> None:
        record = _base_record()
        record["odds_history"] = [
            {
                "captured_at": "2026-06-01T08:00:00Z",
                "team1_implied_prob_norm": 0.35,
                "team2_implied_prob_norm": 0.65,
                "bookmaker_count": 1,
            },
            {
                "captured_at": "2026-06-01T11:55:00Z",
                "team1_implied_prob_norm": 0.70,
                "team2_implied_prob_norm": 0.30,
                "bookmaker_count": 1,
            },
        ]

        row = self.load_one(record)

        self.assertEqual(row["opening_odds_t1"], 0.35)
        self.assertEqual(row["opening_odds_captured_at"], "2026-06-01T08:00:00Z")


if __name__ == "__main__":
    unittest.main()

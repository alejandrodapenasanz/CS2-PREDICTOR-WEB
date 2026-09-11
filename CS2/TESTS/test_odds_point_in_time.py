from __future__ import annotations

import json
import sqlite3
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

    def test_database_opening_accepts_same_day_prestart_and_rejects_later_prices(
        self,
    ) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE matches(
                match_id INTEGER PRIMARY KEY,
                datetime_utc TEXT,
                datetime_precision TEXT
            );
            CREATE TABLE odds(
                match_id INTEGER,
                bookmaker TEXT,
                captured_at_utc TEXT,
                market_type TEXT,
                odds_t1 REAL,
                odds_t2 REAL,
                prob_t1 REAL,
                prob_t2 REAL
            );
            INSERT INTO matches VALUES (1, '2026-06-01T16:00:00Z', 'exact');
            INSERT INTO odds VALUES
                (1, 'Broken', '2026-06-01T08:00:00Z', 'opening', 1.01, 1.01, 0.5, 0.5),
                (1, 'Book', '2026-06-01T09:00:00Z', 'opening', 1.80, 2.10, 0.9, 0.1),
                (1, 'Book', '2026-06-01T15:00:00Z', 'opening', 1.20, 4.50, 0.9, 0.1),
                (1, 'Book', '2026-06-01T16:01:00Z', 'opening', 4.50, 1.20, 0.1, 0.9);
            """
        )

        opening = dataio._opening_odds_by_match(connection)[1]
        expected = (1 / 1.80) / ((1 / 1.80) + (1 / 2.10))

        self.assertEqual(opening["captured_at"], "2026-06-01T09:00:00Z")
        self.assertAlmostEqual(opening["team1_implied_prob_norm"], expected)
        self.assertAlmostEqual(
            opening["team1_implied_prob_norm"]
            + opening["team2_implied_prob_norm"],
            1.0,
        )
        connection.close()

    def test_database_opening_requires_exact_kickoff(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE matches(
                match_id INTEGER PRIMARY KEY,
                datetime_utc TEXT,
                datetime_precision TEXT
            );
            CREATE TABLE odds(
                match_id INTEGER,
                bookmaker TEXT,
                captured_at_utc TEXT,
                market_type TEXT,
                odds_t1 REAL,
                odds_t2 REAL,
                prob_t1 REAL,
                prob_t2 REAL
            );
            INSERT INTO matches VALUES (1, '2026-06-01T00:00:00Z', 'date_only');
            INSERT INTO odds VALUES
                (1, 'Book', '2026-05-31T20:00:00Z', 'opening', 1.80, 2.10, 0.5, 0.5);
            """
        )

        self.assertEqual(dataio._opening_odds_by_match(connection), {})
        connection.close()


if __name__ == "__main__":
    unittest.main()

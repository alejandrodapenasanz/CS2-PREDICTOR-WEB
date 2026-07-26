from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MODEL"))

from DAILY_SNAPSHOTS.enrich_predictions import (
    flag_match,
    load_actual_lineup_store,
    roster_change_90d,
)


class RosterChange90dTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(
            """
            CREATE TABLE matches (
                match_id INTEGER PRIMARY KEY,
                hltv_match_id TEXT,
                datetime_utc TEXT,
                status TEXT
            );
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                hltv_id TEXT
            );
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                hltv_id TEXT,
                nick TEXT
            );
            CREATE TABLE match_lineups (
                match_id INTEGER,
                team_id INTEGER,
                player_id INTEGER,
                is_standin INTEGER
            );
            INSERT INTO teams VALUES (1, '100');
            """
        )
        for player_id in range(1, 11):
            self.conn.execute(
                "INSERT INTO players VALUES (?, ?, ?)",
                (player_id, str(player_id), f"p{player_id}"),
            )

    def tearDown(self) -> None:
        self.conn.close()

    def add_match(
        self,
        match_id: int,
        hltv_match_id: str,
        datetime_utc: str,
        players: tuple[int, ...] = (),
        status: str = "completed",
    ) -> None:
        self.conn.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (match_id, hltv_match_id, datetime_utc, status),
        )
        for player_id in players:
            self.conn.execute(
                "INSERT INTO match_lineups VALUES (?, 1, ?, 0)",
                (match_id, player_id),
            )

    @staticmethod
    def snapshot(
        player_ids: tuple[int, ...],
        *,
        standin_id: int | None = None,
        captured_at: str = "2026-07-26T12:00:00Z",
    ) -> dict:
        return {
            "id": "9002",
            "date": "2026-07-26",
            "hour": "19:00",
            "captured_at": captured_at,
            "prematch_lineups": {
                "team1": {
                    "hltv_team_id": "100",
                    "team_name": "Alpha",
                    "players": [
                        {
                            "hltv_player_id": str(player_id),
                            "nickname": f"p{player_id}",
                            "is_standin": player_id == standin_id,
                        }
                        for player_id in player_ids
                    ],
                }
            },
        }

    def summary(self, player_ids: tuple[int, ...], **snapshot_kwargs: object) -> dict:
        store = load_actual_lineup_store(self.conn)
        return roster_change_90d(
            self.snapshot(player_ids, **snapshot_kwargs),
            "team1",
            {"id": "100", "name": "Alpha"},
            store,
        )

    @staticmethod
    def flags_for(summary: dict) -> list[dict[str, str]]:
        entry = {
            "format": "bo3",
            "odds": {"available": True, "bookmaker_count": 3},
            "controls": {
                "roster_change_90d": {"team1": summary, "team2": {}},
                "roster_stability": {},
                "player_form": {
                    "team1": {"real_form_coverage": 1.0},
                    "team2": {"real_form_coverage": 1.0},
                },
                "calibration": {"all": {"n": 30}},
            },
        }
        features = {
            "matches_team1": 20,
            "matches_team2": 20,
            "elo_diff": 100,
            "event_volatility_label": "normal",
            "player_coverage_team1": 1.0,
            "player_coverage_team2": 1.0,
        }
        prediction = {
            "model_prob_team1": 0.60,
            "confidence": 0.60,
        }
        return flag_match(entry, features, prediction)

    def test_change_against_latest_actual_lineup_is_a_red_flag(self) -> None:
        self.add_match(1, "9001", "2026-07-20T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "9002", "2026-07-26T17:00:00Z", status="scheduled")

        result = self.summary((1, 2, 3, 4, 6))

        self.assertEqual(result["status"], "changed")
        self.assertTrue(result["red_flag"])
        self.assertEqual(result["latest_previous_match_id"], "9001")
        self.assertEqual(result["players_in"], [{"id": "6", "name": "p6"}])
        self.assertEqual(result["players_out"], [{"id": "5", "name": "p5"}])
        roster_flag = self.flags_for(result)[0]
        self.assertEqual(roster_flag["code"], "ROSTER_CHANGE_90D")
        self.assertEqual(roster_flag["level"], "danger")
        self.assertIn("p6", roster_flag["message"])
        self.assertIn("p5", roster_flag["message"])

    def test_same_five_player_ids_are_unchanged(self) -> None:
        self.add_match(1, "9001", "2026-07-20T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "9002", "2026-07-26T17:00:00Z", status="scheduled")

        result = self.summary((5, 4, 3, 2, 1))

        self.assertEqual(result["status"], "unchanged")
        self.assertFalse(result["red_flag"])
        self.assertNotIn(
            "ROSTER_CHANGE_90D",
            {flag["code"] for flag in self.flags_for(result)},
        )

    def test_old_future_and_incomplete_actual_lineups_are_not_evidence(self) -> None:
        self.add_match(1, "old", "2026-04-01T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "partial", "2026-07-20T17:00:00Z", (1, 2, 3, 4))
        self.add_match(3, "9002", "2026-07-26T17:00:00Z", status="scheduled")
        self.add_match(4, "future", "2026-07-27T17:00:00Z", (6, 7, 8, 9, 10))

        result = self.summary((1, 2, 3, 4, 6))

        self.assertEqual(result["status"], "no_recent_complete_history")
        self.assertEqual(result["previous_matches_compared"], 0)
        self.assertFalse(result["red_flag"])

    def test_incomplete_or_post_start_current_lineup_never_flags(self) -> None:
        self.add_match(1, "9001", "2026-07-20T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "9002", "2026-07-26T17:00:00Z", status="scheduled")

        incomplete = self.summary((1, 2, 3, 4))
        post_start = self.summary(
            (1, 2, 3, 4, 6),
            captured_at="2026-07-26T18:00:00Z",
        )

        self.assertEqual(incomplete["status"], "current_lineup_incomplete")
        self.assertFalse(incomplete["red_flag"])
        self.assertEqual(post_start["status"], "post_start_snapshot_rejected")
        self.assertFalse(post_start["red_flag"])

    def test_missing_announced_lineup_is_unknown_not_a_mismatch(self) -> None:
        self.add_match(1, "9001", "2026-07-20T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "9002", "2026-07-26T17:00:00Z", status="scheduled")
        store = load_actual_lineup_store(self.conn)
        snapshot = self.snapshot(())
        snapshot["prematch_lineups"] = {}

        result = roster_change_90d(
            snapshot,
            "team1",
            {"id": "100", "name": "Alpha"},
            store,
        )

        self.assertEqual(result["status"], "current_lineup_missing")
        self.assertFalse(result["red_flag"])

    def test_announced_standin_is_preserved_as_separate_warning(self) -> None:
        self.add_match(1, "9001", "2026-07-20T17:00:00Z", (1, 2, 3, 4, 5))
        self.add_match(2, "9002", "2026-07-26T17:00:00Z", status="scheduled")

        result = self.summary((1, 2, 3, 4, 6), standin_id=6)
        codes = {flag["code"] for flag in self.flags_for(result)}

        self.assertEqual(result["announced_standins"], [{"id": "6", "name": "p6", "is_standin": True}])
        self.assertIn("ROSTER_CHANGE_90D", codes)
        self.assertIn("STANDIN_RISK", codes)


if __name__ == "__main__":
    unittest.main()

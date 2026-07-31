from __future__ import annotations

import sys
import sqlite3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MODEL"))

from PIPELINE.enrich_predictions import (
    _announced_lineup,
    aggregate_player_stats,
    reliability_score,
    roster_summary,
)
from cs2model.dataio import _team_player_snapshot_summary
from cs2model.features import (
    ANNOUNCED_LINEUP_FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    announced_lineup_features,
)


def _snapshot(rating: float, maps: int) -> dict:
    return {
        "maps": maps,
        "stats": {
            "Rating 3.0": rating,
            "Round Swing": rating - 1.0,
            "Opening KPR": rating / 10,
        },
    }


class PlayerRosterAggregationTests(unittest.TestCase):
    def test_announced_lineup_features_preserve_star_depth_and_standin_signal(self) -> None:
        def player(rating: float, standin: bool = False) -> dict:
            return {
                "rating": rating,
                "kpr": rating / 2,
                "kast": rating * 70,
                "adr": rating * 75,
                "multi_kill_rating": rating,
                "round_swing": rating - 1,
                "is_standin": standin,
            }

        result = announced_lineup_features(
            {
                "prematch_lineups": {
                    "team1": {"players": [player(v) for v in (1.30, 1.10, 1.00, 0.90, 0.80)]},
                    "team2": {"players": [player(v, standin=(i == 0)) for i, v in enumerate((1.10, 1.00, 0.95, 0.90, 0.85))]},
                }
            }
        )

        self.assertEqual(result["announced_lineup_available"], 1.0)
        self.assertEqual(result["announced_lineup_players_min"], 5.0)
        self.assertEqual(result["announced_lineup_rating_coverage_min"], 5.0)
        self.assertAlmostEqual(result["announced_lineup_rating_top2_avg_diff"], 0.15)
        self.assertAlmostEqual(result["announced_lineup_rating_bottom2_avg_diff"], -0.025)
        self.assertEqual(result["announced_lineup_standin_advantage"], 1.0)
        self.assertTrue(set(result).issuperset(ANNOUNCED_LINEUP_FEATURE_COLUMNS))

    def test_partial_announced_lineups_do_not_activate_feature_family(self) -> None:
        result = announced_lineup_features(
            {
                "prematch_lineups": {
                    "team1": {"players": [{"rating": 1.0}] * 5},
                    "team2": {"players": [{"rating": 1.0}] * 4},
                }
            }
        )

        self.assertEqual(result["announced_lineup_available"], 0.0)
        self.assertEqual(result["announced_lineup_players_min"], 4.0)

    def test_distribution_separates_star_duo_and_weak_core(self) -> None:
        players = [
            {"name": f"p{idx}", "stats": _snapshot(rating, 12)}
            for idx, rating in enumerate((1.30, 1.10, 1.00, 0.90, 0.80), start=1)
        ]

        result = aggregate_player_stats(players)
        rating = result["metrics"]["rating"]

        self.assertEqual(result["covered_players"], 5)
        self.assertEqual(result["maps_min"], 12)
        self.assertAlmostEqual(rating["avg"], 1.02)
        self.assertAlmostEqual(rating["max"], 1.30)
        self.assertAlmostEqual(rating["top2_avg"], 1.20)
        self.assertAlmostEqual(rating["median"], 1.00)
        self.assertAlmostEqual(rating["bottom2_avg"], 0.85)

    def test_roster_uses_a_common_representative_window(self) -> None:
        profiles = {
            "1": {
                "squad": [
                    {"id": str(idx), "name": f"p{idx}"}
                    for idx in range(1, 6)
                ]
            }
        }
        player_stats = {}
        for idx in range(1, 6):
            windows = {"past6months": _snapshot(1.0 + idx / 100, 12)}
            if idx <= 3:
                windows["past3months"] = _snapshot(1.5, 10)
            player_stats[str(idx)] = {"windows": windows, "preferred": windows["past6months"]}

        result = roster_summary("1", profiles, player_stats)

        self.assertEqual(result["preferred_time_filter"], "past6months")
        self.assertEqual(result["distribution"]["covered_players"], 5)
        self.assertEqual(result["distribution"]["maps_min"], 12)
        self.assertAlmostEqual(result["avg"]["Rating 3.0"], 1.03)

    def test_complete_match_lineup_overrides_incomplete_team_profile(self) -> None:
        profiles = {
            "13613": {
                "squad": [
                    {"id": "14273", "name": "iDISBALANCE"},
                    {"id": "20197", "name": "faydett"},
                    {"id": "20948", "name": "lov1kus"},
                ]
            }
        }
        player_stats = {
            player_id: {
                "windows": {"past3months": _snapshot(rating, 20)},
                "preferred": _snapshot(rating, 20),
                "preferred_time_filter": "past3months",
            }
            for player_id, rating in (
                ("14273", 0.96),
                ("20197", 1.02),
                ("20948", 1.05),
            )
        }
        snapshot = {
            "prematch_lineups": {
                "team1": {
                    "hltv_team_id": "13613",
                    "players": [
                        {"hltv_player_id": "14273", "nickname": "iDISBALANCE"},
                        {"hltv_player_id": "20197", "nickname": "faydett"},
                        {"hltv_player_id": "20948", "nickname": "lov1kus"},
                        {
                            "hltv_player_id": "24726",
                            "nickname": "z3ndeR",
                            "rating": None,
                            "kast": None,
                        },
                        {
                            "hltv_player_id": "25111",
                            "nickname": "redzed",
                            "rating": 1.0,
                            "kast": 70.7,
                        },
                    ],
                }
            }
        }

        announced = _announced_lineup(snapshot, "team1", "13613")
        result = roster_summary("13613", profiles, player_stats, announced)
        players = {player["id"]: player for player in result["players"]}

        self.assertEqual(result["roster_source"], "prematch_announced_lineup")
        self.assertTrue(result["announced_lineup_complete"])
        self.assertEqual(set(players), {"14273", "20197", "20948", "24726", "25111"})
        self.assertAlmostEqual(result["coverage"], 3 / 5)
        self.assertEqual(result["distribution"]["covered_players"], 3)
        self.assertIsNone(players["24726"]["stats"])
        self.assertEqual(players["25111"]["stats"]["stats"]["Rating 3.0"], 1.0)
        self.assertEqual(players["25111"]["stats_source"], "HLTV.match_prematch_lineup")
        self.assertEqual(result["integrity"]["status"], "confirmed")
        self.assertTrue(result["integrity"]["announced_matches_rendered"])

    def test_partial_announced_lineup_is_not_lost_or_counted_as_full_coverage(self) -> None:
        profiles = {
            "200": {
                "squad": [
                    {"id": "1", "name": "one"},
                    {"id": "2", "name": "two"},
                    {"id": "3", "name": "three"},
                ]
            }
        }
        player_stats = {
            str(player_id): {
                "windows": {"past3months": _snapshot(1.0, 20)},
                "preferred": _snapshot(1.0, 20),
                "preferred_time_filter": "past3months",
            }
            for player_id in range(1, 5)
        }
        snapshot = {
            "prematch_lineups": {
                "team1": {
                    "hltv_team_id": "200",
                    "players": [
                        {"hltv_player_id": str(player_id), "nickname": f"p{player_id}"}
                        for player_id in range(1, 5)
                    ],
                }
            }
        }

        announced = _announced_lineup(snapshot, "team1", "200")
        result = roster_summary("200", profiles, player_stats, announced)

        self.assertEqual({player["id"] for player in result["players"]}, {"1", "2", "3", "4"})
        self.assertEqual(result["coverage"], 4 / 5)
        self.assertFalse(result["announced_lineup_complete"])
        self.assertEqual(result["integrity"]["status"], "announced_incomplete")
        self.assertEqual(result["integrity"]["announced_players"], 4)

    def test_reliability_penalizes_three_of_five_player_coverage(self) -> None:
        entry = {
            "format": "bo3",
            "data_quality": {"real_pre_match_snapshot": True},
            "team1": {"name": "Bebop"},
            "team2": {"name": "Falcons Force"},
        }
        base = {
            "matches_team1": 30,
            "matches_team2": 30,
            "glicko_rd_team1": 80,
            "glicko_rd_team2": 80,
            "player_coverage_team2": 1.0,
        }

        complete = reliability_score(entry, {**base, "player_coverage_team1": 1.0})
        incomplete = reliability_score(entry, {**base, "player_coverage_team1": 0.6})

        self.assertEqual(complete, 1.0)
        self.assertEqual(incomplete, 0.9)

    def test_database_join_uses_the_same_common_window_point_in_time(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE team_rosters (
                team_id INTEGER, player_id INTEGER, valid_from TEXT, valid_to TEXT
            );
            CREATE TABLE players (player_id INTEGER, hltv_id TEXT);
            CREATE TABLE player_stat_snapshots (
                player_stat_snapshot_id INTEGER PRIMARY KEY,
                hltv_player_id TEXT,
                captured_at_utc TEXT,
                time_filter TEXT,
                maps INTEGER,
                rating REAL,
                kpr REAL,
                kast REAL,
                adr REAL,
                impact REAL,
                round_swing REAL,
                opening_kpr REAL
            );
            """
        )
        for player_id in range(1, 6):
            conn.execute("INSERT INTO players VALUES (?, ?)", (player_id, str(player_id)))
            conn.execute(
                "INSERT INTO team_rosters VALUES (1, ?, '2025-01-01T00:00:00Z', NULL)",
                (player_id,),
            )
            values = (str(player_id), "2026-01-01T00:00:00Z", "past6months", 12, 1.0 + player_id / 100)
            conn.execute(
                "INSERT INTO player_stat_snapshots "
                "(hltv_player_id, captured_at_utc, time_filter, maps, rating, kpr, kast, adr, impact, round_swing, opening_kpr) "
                "VALUES (?, ?, ?, ?, ?, 0.70, 72, 75, 1.0, 0.0, 0.10)",
                values,
            )
            if player_id <= 3:
                conn.execute(
                    "INSERT INTO player_stat_snapshots "
                    "(hltv_player_id, captured_at_utc, time_filter, maps, rating, kpr, kast, adr, impact, round_swing, opening_kpr) "
                    "VALUES (?, '2026-01-01T00:00:00Z', 'past3months', 10, 1.50, 0.70, 72, 75, 1.0, 0.0, 0.10)",
                    (str(player_id),),
                )

        result = _team_player_snapshot_summary(conn, 1, "2026-01-10T00:00:00Z")

        self.assertEqual(result["time_filter"], "past6months")
        self.assertEqual(result["players"], 5)
        self.assertEqual(result["maps_per_player_min"], 12)
        self.assertAlmostEqual(result["metrics"]["rating"]["avg"], 1.03)
        conn.close()

    def test_training_vector_has_rank_features_without_exactly_redundant_gaps(self) -> None:
        expected = {
            "player_rating_max_diff",
            "player_rating_top2_avg_diff",
            "player_rating_median_diff",
            "player_rating_bottom2_avg_diff",
            "player_rating_std_diff",
            "player_round_swing_top2_avg_diff",
            "player_opening_kpr_top2_avg_diff",
        }
        redundant = {
            "player_rating_min_diff",
            "player_rating_spread_diff",
            "player_star_gap_diff",
            "player_weak_link_gap_diff",
        }

        self.assertTrue(expected.issubset(PLAYER_FEATURE_COLUMNS))
        self.assertFalse(redundant.intersection(PLAYER_FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import ChronologicalState, analytics_match_features


def asset_payload() -> dict:
    return {
        "mapstats": [
            {
                "info": {
                    "map_name": "Inferno",
                    "team_left": {"name": "Alpha", "score": 13},
                    "team_right": {"name": "Beta", "score": 7},
                    "breakdown": {
                        "side_rounds": {
                            "left": {"ct": 7, "t": 6},
                            "right": {"ct": 4, "t": 3},
                        }
                    },
                },
                "player_stats": [
                    {"side": "total", "team_name": "Alpha", "rating": 1.2, "adr": 82, "kast": 75, "opening_kills": 3, "opening_deaths": 1},
                    {"side": "total", "team_name": "Alpha", "rating": 1.1, "adr": 78, "kast": 72, "opening_kills": 2, "opening_deaths": 1},
                    {"side": "total", "team_name": "Beta", "rating": 0.9, "adr": 65, "kast": 62, "opening_kills": 1, "opening_deaths": 2},
                    {"side": "total", "team_name": "Beta", "rating": 0.8, "adr": 60, "kast": 60, "opening_kills": 0, "opening_deaths": 2},
                ],
            }
        ]
    }


class HltvAssetFeatureTests(unittest.TestCase):
    def test_asset_features_are_observed_only_after_match(self) -> None:
        state = ChronologicalState()
        before = state.emit_features("alpha", "beta", datetime(2026, 1, 1), "Cup", "bo1")

        self.assertEqual(before["asset_maps_played_total"], 0.0)
        self.assertEqual(before["asset_rating_l5_diff"], 0.0)

        state.observe(
            {
                "id": "1",
                "team1_key": "alpha",
                "team2_key": "beta",
                "score1": 1,
                "score2": 0,
                "team1_win": 1,
                "date": "2026-01-01",
                "date_obj": datetime(2026, 1, 1),
                "event": "Cup",
                "format": "bo1",
                "asset": asset_payload(),
            }
        )
        after = state.emit_features("alpha", "beta", datetime(2026, 1, 2), "Cup", "bo1")

        self.assertEqual(after["asset_maps_played_total"], 2.0)
        self.assertGreater(after["asset_map_winrate_diff"], 0.0)
        self.assertGreater(after["asset_rating_l5_diff"], 0.0)
        self.assertGreater(after["asset_adr_l10_diff"], 0.0)

    def test_analytics_features_are_directional_and_point_in_time_ready(self) -> None:
        feats = analytics_match_features(
            {
                "team1": "Alpha",
                "team2": "Beta",
                "team1_key": "alpha",
                "team2_key": "beta",
                "analytics": {
                    "available": True,
                    "map_stats": [
                        {"map": "Mirage", "team": "Alpha", "win_pct": 60, "first_pick_pct": 20, "first_ban_pct": 10, "played": 5},
                        {"map": "Mirage", "team": "Beta", "win_pct": 40, "first_pick_pct": 0, "first_ban_pct": 25, "played": 3},
                    ],
                    "insights": [
                        "Alpha is better ranked (#23)",
                        "Beta is worse ranked (#50)",
                    ],
                },
            }
        )

        self.assertEqual(feats["analytics_available"], 1.0)
        self.assertEqual(feats["analytics_common_maps"], 1.0)
        self.assertAlmostEqual(feats["analytics_map_win_pct_diff"], 0.20)
        self.assertAlmostEqual(feats["analytics_first_pick_pct_diff"], 0.20)
        self.assertAlmostEqual(feats["analytics_first_ban_pct_diff"], -0.15)
        self.assertEqual(feats["analytics_map_played_diff"], 2.0)
        self.assertGreater(feats["analytics_insight_score_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()

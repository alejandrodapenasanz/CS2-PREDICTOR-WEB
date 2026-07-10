from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))
sys.path.insert(0, str(ROOT / "tests"))

from tests_wallet_simulator import make_bet_candidates, make_recommended_bets


class WalletFavoritePolicyTests(unittest.TestCase):
    def test_recommended_bet_keeps_model_favorite(self) -> None:
        rows = [
            {
                "id": "match-1",
                "date": "2026-01-01",
                "event": "Test",
                "team1": "Alpha",
                "team2": "Bravo",
                "team1_win": False,
                "opening_odds_decimal_t1": 1.50,
                "opening_odds_decimal_t2": 2.20,
                "opening_odds_t1": 0.60,
                "opening_bookmaker_count": 3,
            }
        ]
        predictions = {"match-1": {"prob_team1": 0.40}}

        candidates = make_bet_candidates(rows, predictions, {"match-1": 1.0})
        recommended = make_recommended_bets(candidates, {})

        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["side"], "team2")
        self.assertEqual(recommended[0]["team"], "Bravo")
        self.assertGreaterEqual(recommended[0]["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()

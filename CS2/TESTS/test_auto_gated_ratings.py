"""Familias de rating auto-gated: MOV, TrueSkill de equipo y rating por jugador.

Valida (sin dataset real): correctitud basica, antisimetria A<->B de las columnas
DIFF (imprescindible para la augmentacion) y la ACTIVACION automatica por umbral
de muestra en train.select_feature_columns.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

from cs2model.features import (
    ChronologicalState,
    MOV_FEATURE_COLUMNS,
    TRUESKILL_FEATURE_COLUMNS,
    PLAYER_RATING_FEATURE_COLUMNS,
)
import train


def _asset(team_a, team_b, players_a, players_b):
    ps = [{"side": "total", "team_name": team_a, "player_name": p} for p in players_a]
    ps += [{"side": "total", "team_name": team_b, "player_name": p} for p in players_b]
    return {"mapstats": [{"info": {}, "player_stats": ps}]}


class MovRatingTests(unittest.TestCase):
    def test_margin_matters(self):
        # Un 2-0 debe mover el rating MOV mas que un 2-1 (mismo enfrentamiento).
        big = ChronologicalState(); small = ChronologicalState()
        d = datetime(2025, 1, 1)
        base = dict(team1_key="a", team2_key="b", date_obj=d, team1_win=1, event="e", format="bo3")
        big.observe({**base, "score1": 2, "score2": 0})
        small.observe({**base, "score1": 2, "score2": 1})
        self.assertGreater(big.mov["a"], small.mov["a"])
        self.assertLess(big.mov["b"], small.mov["b"])


class AntisymmetryTests(unittest.TestCase):
    def _state(self):
        st = ChronologicalState()
        d0 = datetime(2025, 1, 1)
        pa = [f"a{i}" for i in range(5)]
        pb = [f"b{i}" for i in range(5)]
        pc = [f"c{i}" for i in range(5)]
        # historia con box score para poblar rating por jugador y agregados
        for k in range(30):
            d = d0 + timedelta(days=k)
            st.observe({"team1_key":"alpha","team2_key":"beta","date_obj":d,"team1_win":1,
                        "score1":2,"score2":k%2,"event":"e","format":"bo3",
                        "asset":_asset("alpha","beta",pa,pb)})
            st.observe({"team1_key":"beta","team2_key":"gamma","date_obj":d,"team1_win":0,
                        "score1":0,"score2":2,"event":"e","format":"bo3",
                        "asset":_asset("beta","gamma",pb,pc)})
        return st

    def test_diff_columns_antisymmetric(self):
        st = self._state()
        d = datetime(2025, 3, 1)
        fab = st.emit_features("alpha", "beta", d)
        fba = st.emit_features("beta", "alpha", d)
        for col in ["mov_diff","mov_prob_centered",
                    "player_skill_mean_diff","player_skill_max_diff",
                    "player_skill_min_diff","player_skill_spread_diff"]:
            self.assertAlmostEqual(fab[col], -fba[col], places=9, msg=col)
        # player_skill debe estar disponible tras observar box score de ambos
        self.assertEqual(fab["player_skill_available"], 1.0)
        self.assertEqual(fab["trueskill_available"], 1.0)
        self.assertEqual(fab["mov_available"], 1.0)


class AutoGatingTests(unittest.TestCase):
    def test_families_activate_at_threshold(self):
        # Suficientes filas -> mov y trueskill (umbral 800) y player_rating (200) ON.
        rows = []
        for i in range(900):
            rows.append({"mov_available":1.0,"trueskill_available":1.0,
                         "player_skill_available":1.0 if i < 250 else 0.0})
        cols, pol = train.select_feature_columns(rows, feature_profile="core")
        self.assertTrue(set(MOV_FEATURE_COLUMNS).issubset(cols))
        self.assertTrue(set(TRUESKILL_FEATURE_COLUMNS).issubset(cols))
        self.assertTrue(set(PLAYER_RATING_FEATURE_COLUMNS).issubset(cols))
        self.assertTrue(pol["mov_rating"]["enabled"])
        self.assertTrue(pol["player_rating"]["enabled"])

    def test_families_off_below_threshold(self):
        rows = [{"mov_available":1.0,"trueskill_available":1.0,"player_skill_available":0.0}
                for _ in range(100)]
        cols, pol = train.select_feature_columns(rows, feature_profile="core")
        self.assertFalse(set(MOV_FEATURE_COLUMNS).issubset(cols))       # 100 < 800
        self.assertFalse(pol["player_rating"]["enabled"])              # 0 < 200
        self.assertFalse(pol["mov_rating"]["enabled"])


if __name__ == "__main__":
    unittest.main()

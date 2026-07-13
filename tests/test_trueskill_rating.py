"""Correctitud del prototipo TrueSkill (equipo, online, point-in-time).

Sin dataset real: se valida con datos sinteticos de fuerza latente conocida.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

from cs2model.trueskill import TeamTrueSkill, TSRating
from cs2model.features import ChronologicalState


class TrueSkillEngineTests(unittest.TestCase):
    def test_win_probability_symmetry_and_bounds(self):
        ts = TeamTrueSkill()
        a = TSRating(30.0, 4.0)
        b = TSRating(20.0, 4.0)
        p = ts.win_probability(a, b)
        q = ts.win_probability(b, a)
        self.assertGreater(p, 0.5)
        self.assertAlmostEqual(p + q, 1.0, places=9)  # antisimetria exacta
        self.assertTrue(0.0 < p < 1.0)
        self.assertAlmostEqual(ts.win_probability(a, a), 0.5, places=9)

    def test_winner_mu_rises_loser_falls_sigma_shrinks(self):
        ts = TeamTrueSkill()
        w0 = ts.default()
        l0 = ts.default()
        w1, l1 = ts.update(w0, l0)
        self.assertGreater(w1.mu, w0.mu)
        self.assertLess(l1.mu, l0.mu)
        self.assertLess(w1.sigma, w0.sigma)  # menos incertidumbre tras observar
        self.assertLess(l1.sigma, l0.sigma)

    def test_recovers_latent_ranking(self):
        # 8 equipos con fuerza latente; el ganador se sortea por Bradley-Terry.
        random.seed(7)
        strengths = {f"t{i}": (i - 3.5) for i in range(8)}  # t7 mas fuerte
        ts = TeamTrueSkill()
        ratings = {k: ts.default() for k in strengths}
        for _ in range(4000):
            a, b = random.sample(list(strengths), 2)
            pa = 1.0 / (1.0 + math.exp(-(strengths[a] - strengths[b])))
            if random.random() < pa:
                ratings[a], ratings[b] = ts.update(ratings[a], ratings[b])
            else:
                ratings[b], ratings[a] = ts.update(ratings[b], ratings[a])
        order_true = [k for k, _ in sorted(strengths.items(), key=lambda kv: kv[1])]
        order_est = [k for k, _ in sorted(ratings.items(), key=lambda kv: kv[1].mu)]
        # correlacion de rango alta (permitimos algun swap adyacente por ruido)
        inversions = sum(
            1
            for i in range(len(order_true))
            for j in range(i + 1, len(order_true))
            if order_est.index(order_true[i]) > order_est.index(order_true[j])
        )
        self.assertLessEqual(inversions, 3)


class TrueSkillFeatureIntegrationTests(unittest.TestCase):
    def _observe_history(self) -> ChronologicalState:
        random.seed(3)
        st = ChronologicalState()
        teams = [f"t{i}" for i in range(6)]
        base = datetime(2025, 1, 1)
        for k in range(300):
            a, b = random.sample(teams, 2)
            d = base + timedelta(days=k // 2)
            aw = random.random() < 0.5
            st.observe({
                "team1_key": a, "team2_key": b, "date_obj": d,
                "team1_win": int(aw), "score1": 2 if aw else 0,
                "score2": 0 if aw else 2, "event": "e", "format": "bo3",
            })
        return st

    def test_emit_includes_trueskill_and_is_antisymmetric(self):
        st = self._observe_history()
        d = datetime(2025, 6, 1)
        fab = st.emit_features("t0", "t1", d)
        fba = st.emit_features("t1", "t0", d)
        self.assertIn("trueskill_diff", fab)
        self.assertIn("trueskill_prob_centered", fab)
        # DIFF antisimetrico: intercambiar A<->B niega el valor (clave para augment)
        self.assertAlmostEqual(fab["trueskill_diff"], -fba["trueskill_diff"], places=9)
        self.assertAlmostEqual(
            fab["trueskill_prob_centered"], -fba["trueskill_prob_centered"], places=9
        )


if __name__ == "__main__":
    unittest.main()

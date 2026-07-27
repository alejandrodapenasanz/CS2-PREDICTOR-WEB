"""C16: auditoria de fuga temporal exhaustiva.

Garantiza que las features rolling emitidas para un partido dependen SOLO de
partidos ANTERIORES: reconstruye un estado limpio con rows[:i] y compara su
emision con la que produjo build_training_frame (que emite antes de observar).
Si alguna feature dependiera del futuro, los valores diferirian.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

from cs2model.features import ChronologicalState, build_training_frame


def _synthetic_rows(n: int = 600, seed: int = 11):
    random.seed(seed)
    teams = [f"t{i}" for i in range(24)]
    strengths = {t: random.gauss(0, 1) for t in teams}
    base = datetime(2024, 1, 1)
    rows = []
    for k in range(n):
        a, b = random.sample(teams, 2)
        d = base + timedelta(hours=k * 8)
        pa = 1.0 / (1.0 + math.exp(-(strengths[a] - strengths[b])))
        aw = 1 if random.random() < pa else 0
        fmt = random.choice(["bo1", "bo3", "bo3", "bo5"])
        mx = {"bo1": 1, "bo3": 2, "bo5": 3}[fmt]
        s1, s2 = (mx, random.randint(0, mx - 1)) if aw else (random.randint(0, mx - 1), mx)
        rows.append({
            "id": str(k), "date": d.strftime("%Y-%m-%d"), "date_obj": d,
            "event": f"e{k % 10}", "format": fmt,
            "team1": a, "team2": b, "team1_key": a, "team2_key": b,
            "score1": s1, "score2": s2, "team1_win": aw,
        })
    return rows


class LeakageAuditTests(unittest.TestCase):
    def test_emit_features_use_only_prior_matches(self):
        rows = _synthetic_rows()
        X_full, _y, _meta, _state = build_training_frame(rows)
        # Muestra de indices; para cada uno, estado fresco con SOLO rows[:i].
        random.seed(1)
        sample = random.sample(range(60, len(rows)), 20)
        for i in sample:
            st = ChronologicalState()
            for j in range(i):
                st.observe(rows[j])
            r = rows[i]
            fresh = st.emit_features(
                r["team1_key"], r["team2_key"], r.get("date_obj"),
                r.get("event") or "", r.get("format") or "bo3",
            )
            for key, value in fresh.items():
                ref = X_full[i].get(key)
                self.assertIsNotNone(ref, f"feature {key} ausente en build_training_frame[{i}]")
                self.assertAlmostEqual(
                    float(value), float(ref), places=9,
                    msg=f"FUGA: feature '{key}' del partido {i} difiere al reconstruir solo con el pasado",
                )


if __name__ == "__main__":
    unittest.main()

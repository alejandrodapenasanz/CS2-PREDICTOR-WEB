"""C16: auditoria de fuga temporal exhaustiva.

Garantiza que las features rolling emitidas para un partido dependen SOLO de
partidos ANTERIORES: reconstruye un estado limpio con rows[:i] y compara su
emision con la que produjo build_training_frame (que emite antes de observar).
Si alguna feature dependiera del futuro, los valores diferirian.
"""

from __future__ import annotations

import math
import random
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

from cs2model.dataio import (
    _iso_date,
    _latest_analytics_by_match_asof,
    _opening_odds_by_match,
    _prematch_lineups_by_match_asof,
    _team_player_snapshot_summary,
    load_training_rows_from_db,
)
from cs2model.features import ChronologicalState, build_training_frame


ROOT = Path(__file__).resolve().parents[1]
LIVE_DB = ROOT / "BBDD" / "cs2.db"


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


@unittest.skipUnless(LIVE_DB.exists(), "cs2.db real no disponible en este entorno")
class RealDatabaseLeakageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_training_rows_from_db(LIVE_DB)
        cls.X_full, _y, _meta, _state = build_training_frame(cls.rows)

    def test_enriched_features_are_stable_when_rebuilt_from_prefix(self):
        """Cada familia enriquecida debe ser idéntica sin filas futuras."""
        self.assertGreater(len(self.rows), 100)
        availability_columns = (
            "analytics_available",
            "analytics_extended_available",
            "announced_lineup_available",
            "player_snapshot_available",
            "ranking_available",
            "roster_available",
            "asset_available",
            "event_history_available",
            "context_available",
        )
        sample_indices: set[int] = {len(self.rows) - 1}
        for column in availability_columns:
            covered = [
                index for index, features in enumerate(self.X_full)
                if float(features.get(column, 0.0) or 0.0) > 0.0
            ]
            if covered:
                sample_indices.add(covered[len(covered) // 2])
                sample_indices.add(covered[-1])

        for index in sorted(sample_indices):
            prefix_X, _y, _meta, _state = build_training_frame(self.rows[: index + 1])
            rebuilt = prefix_X[-1]
            reference = self.X_full[index]
            self.assertEqual(set(rebuilt), set(reference))
            for column, expected in reference.items():
                actual = rebuilt[column]
                self.assertAlmostEqual(
                    float(actual),
                    float(expected),
                    places=9,
                    msg=(
                        f"FUGA L2: '{column}' del partido {self.rows[index]['id']} "
                        "cambia al eliminar el futuro"
                    ),
                )

    def test_asof_selectors_never_cross_match_start(self):
        """Los selectores pueden ignorar fotos futuras sin borrarlas de la BBDD."""
        with sqlite3.connect(LIVE_DB) as conn:
            conn.row_factory = sqlite3.Row
            matches = conn.execute(
                """
                SELECT match_id, team1_id, team2_id, datetime_utc
                FROM matches
                WHERE datetime_utc IS NOT NULL
                ORDER BY datetime_utc
                """
            ).fetchall()
            match_dates = {
                int(row["match_id"]): _iso_date(row["datetime_utc"])
                for row in matches
            }
            opening = _opening_odds_by_match(conn)
            analytics = _latest_analytics_by_match_asof(conn)
            lineups = _prematch_lineups_by_match_asof(conn, matches)

            selected_sources = {
                "opening odds": {
                    match_id: payload.get("captured_at")
                    for match_id, payload in opening.items()
                },
                "analytics": {
                    match_id: payload.get("captured_at")
                    for match_id, payload in analytics.items()
                },
                "announced lineups": {
                    match_id: payload.get("captured_at")
                    for match_id, payload in lineups.items()
                },
            }
            for source, selected in selected_sources.items():
                for match_id, captured_at in selected.items():
                    captured = _iso_date(captured_at)
                    self.assertIsNotNone(captured, f"{source}: falta captured_at en {match_id}")
                    self.assertLessEqual(
                        captured,
                        match_dates[match_id],
                        f"{source}: snapshot futuro seleccionado para {match_id}",
                    )

            sampled = matches[:: max(1, len(matches) // 100)]
            for match in sampled:
                match_dt = match_dates[int(match["match_id"])]
                for team_column in ("team1_id", "team2_id"):
                    summary = _team_player_snapshot_summary(
                        conn,
                        int(match[team_column]),
                        str(match["datetime_utc"]),
                    )
                    selected_at = _iso_date(summary.get("captured_at_max"))
                    if selected_at is not None:
                        self.assertLessEqual(
                            selected_at,
                            match_dt,
                            f"player stats: snapshot futuro seleccionado para {match['match_id']}",
                        )


if __name__ == "__main__":
    unittest.main()

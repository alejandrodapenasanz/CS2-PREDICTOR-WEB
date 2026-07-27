from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.config import EconomicConfig, load_config
from cs2model.dataio import _closing_odds_by_match, _opening_odds_by_match
from cs2model.drift import PageHinkley, build_drift_report
from cs2model.economic import closing_line_value, devig_two_way, economic_backtest
from cs2model.features import ChronologicalState, REGIME_FEATURE_COLUMNS
from cs2model.reproducibility import dataset_fingerprint
from train import select_feature_columns


def _asset(map_name: str, left: str = "Alpha", right: str = "Beta") -> dict:
    return {
        "mapstats": [
            {
                "info": {
                    "map_name": map_name,
                    "team_left": {"name": left, "score": 13},
                    "team_right": {"name": right, "score": 8},
                }
            }
        ]
    }


def _completed(index: int, when: datetime, map_name: str) -> dict:
    return {
        "id": str(index),
        "date": when.date().isoformat(),
        "date_obj": when,
        "event": "Regime Cup",
        "format": "bo3",
        "team1": "Alpha",
        "team2": "Beta",
        "team1_key": "alpha",
        "team2_key": "beta",
        "score1": 2,
        "score2": 0,
        "team1_win": 1,
        "asset": _asset(map_name),
    }


class ConfigurationAndReproducibilityTests(unittest.TestCase):
    def test_default_config_is_versioned_and_hash_is_stable(self) -> None:
        first = load_config(ROOT / "MODEL" / "config.yaml")
        second = load_config(ROOT / "MODEL" / "config.yaml")
        self.assertEqual(first.version, 1)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(len(first.sha256), 64)
        self.assertEqual(first.feature_thresholds["regime"], 200)

    def test_custom_threshold_changes_auto_activation_without_code_change(self) -> None:
        rows = [{"regime_available": 1.0} for _ in range(3)]
        columns, policy = select_feature_columns(rows, feature_thresholds={"regime": 3})
        self.assertTrue(policy["regime"]["enabled"])
        self.assertTrue(all(column in columns for column in REGIME_FEATURE_COLUMNS))

    def test_dataset_fingerprint_is_deterministic_and_order_sensitive(self) -> None:
        rows = [
            {"id": "1", "date": "2026-01-01", "team1_key": "a", "team2_key": "b", "team1_win": 1},
            {"id": "2", "date": "2026-01-02", "team1_key": "c", "team2_key": "d", "team1_win": 0},
        ]
        self.assertEqual(dataset_fingerprint(rows), dataset_fingerprint([dict(row) for row in rows]))
        self.assertNotEqual(dataset_fingerprint(rows), dataset_fingerprint(list(reversed(rows))))


class RegimeAndDriftTests(unittest.TestCase):
    def test_map_pool_regime_uses_only_prior_observations(self) -> None:
        state = ChronologicalState()
        start = datetime(2026, 1, 1)
        before = state.regime_features({"date_obj": start})
        self.assertEqual(before["regime_map_pool_available"], 0.0)
        maps = ["mirage", "inferno", "nuke", "ancient", "dust2"]
        for index in range(20):
            state.observe(_completed(index, start + timedelta(days=index), maps[index % len(maps)]))
        as_of = start + timedelta(days=21)
        after = state.regime_features({"date_obj": as_of})
        self.assertEqual(after["regime_map_pool_available"], 1.0)
        self.assertEqual(after["regime_map_pool_size"], 5.0)
        self.assertEqual(state.map_pool_regime_label(as_of), "ancient+dust2+inferno+mirage+nuke")

    def test_patch_is_unknown_unless_prematch_metadata_supplies_it(self) -> None:
        state = ChronologicalState()
        as_of = datetime(2026, 2, 1)
        unknown = state.regime_features({"date_obj": as_of})
        known = state.regime_features(
            {
                "date_obj": as_of,
                "patch_version": "test-patch",
                "patch_released_at": "2026-01-25T00:00:00Z",
            }
        )
        self.assertEqual(unknown["regime_patch_known"], 0.0)
        self.assertEqual(known["regime_patch_known"], 1.0)
        self.assertEqual(known["regime_patch_recent"], 1.0)

    def test_page_hinkley_and_report_detect_loss_shift(self) -> None:
        detector = PageHinkley(delta=0.001, threshold=2.0)
        stream = [0.1] * 100 + [2.0] * 40
        self.assertTrue(any(detector.update(value) for value in stream))

        predictions = []
        rows = []
        for index in range(140):
            good_period = index < 100
            predictions.append(
                {
                    "match_id": str(index),
                    "date": f"2026-01-{index % 28 + 1:02d}",
                    "actual": 1 if good_period else 0,
                    "prob_team1": 0.9,
                    "environment": "online",
                }
            )
            rows.append({"id": str(index)})
        config = replace(
            load_config(ROOT / "MODEL" / "config.yaml").drift,
            rolling_window=40,
            reference_window=80,
            min_samples=20,
            page_hinkley_threshold=2.0,
        )
        report = build_drift_report(predictions, rows, config)
        self.assertEqual(report["status"], "warning")
        self.assertIn("page_hinkley_log_loss", report["warnings"])
        self.assertEqual(report["clv"]["coverage"], 0.0)


class EconomicBacktestTests(unittest.TestCase):
    def test_devig_and_clv_have_expected_direction(self) -> None:
        market = devig_two_way(1.80, 2.10)
        self.assertAlmostEqual(market["fair_prob_team1"] + market["fair_prob_team2"], 1.0)
        self.assertGreater(market["overround"], 0.0)
        self.assertGreater(closing_line_value(2.20, 1.90), 0.0)
        self.assertLess(closing_line_value(1.90, 2.20), 0.0)

    def test_closing_odds_are_audit_only_and_limits_apply(self) -> None:
        base = {
            "id": "1",
            "date": "2026-01-01",
            "team1_win": True,
            "opening_odds_decimal_t1": 2.20,
            "opening_odds_decimal_t2": 1.70,
            "closing_odds_decimal_t1": 1.90,
            "closing_odds_decimal_t2": 2.05,
        }
        predictions = [{"match_id": "1", "prob_team1": 0.60}]
        config = EconomicConfig(max_stake_amount=10.0, max_profit_amount=100.0)
        first = economic_backtest([base], predictions, config)
        changed_close = {**base, "closing_odds_decimal_t1": 3.00, "closing_odds_decimal_t2": 1.45}
        second = economic_backtest([changed_close], predictions, config)
        self.assertEqual(first["bets"][0]["side"], "team1")
        self.assertEqual(first["bets"][0]["stake"], second["bets"][0]["stake"])
        self.assertEqual(first["bets"][0]["pnl"], second["bets"][0]["pnl"])
        self.assertLessEqual(first["bets"][0]["stake"], 10.0)
        self.assertGreater(first["clv"]["mean_price_clv"], 0.0)
        self.assertLess(second["clv"]["mean_price_clv"], 0.0)

    def test_db_loader_selects_first_open_and_last_valid_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "odds.db"
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.executescript(
                """
                CREATE TABLE matches(match_id INTEGER PRIMARY KEY, datetime_utc TEXT);
                CREATE TABLE odds(
                    match_id INTEGER, bookmaker TEXT, captured_at_utc TEXT,
                    market_type TEXT, odds_t1 REAL, odds_t2 REAL, prob_t1 REAL, prob_t2 REAL
                );
                INSERT INTO matches VALUES(1, '2026-01-01T12:00:00Z');
                INSERT INTO odds VALUES(1,'book','2026-01-01T08:00:00Z','opening',2.2,1.7,.44,.56);
                INSERT INTO odds VALUES(1,'book','2026-01-01T09:00:00Z','opening',2.0,1.8,.47,.53);
                INSERT INTO odds VALUES(1,'book','2026-01-01T10:00:00Z','closing',1.9,2.0,.51,.49);
                INSERT INTO odds VALUES(1,'book','2026-01-01T11:55:00Z','closing',1.8,2.1,.54,.46);
                INSERT INTO odds VALUES(1,'book','2026-01-01T12:05:00Z','closing',1.5,2.5,.62,.38);
                """
            )
            opening = _opening_odds_by_match(conn)[1]
            closing = _closing_odds_by_match(conn)[1]
            conn.close()
        self.assertEqual(opening["captured_at"], "2026-01-01T08:00:00Z")
        self.assertEqual(opening["team1_decimal"], 2.2)
        self.assertEqual(closing["captured_at"], "2026-01-01T11:55:00Z")
        self.assertEqual(closing["team1_decimal"], 1.8)


if __name__ == "__main__":
    unittest.main()


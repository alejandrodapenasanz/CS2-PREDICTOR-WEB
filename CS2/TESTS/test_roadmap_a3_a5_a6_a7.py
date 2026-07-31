from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.artifacts import ModelArtifact
from cs2model.diagnostics import feature_pruning_diagnostics, prune_exact_redundancies
from cs2model.features import SOS_FEATURE_COLUMNS, ChronologicalState, build_training_frame
from cs2model.rich_targets import (
    BO3_SCORE_CLASSES,
    bo3_score_class,
    evaluate_rich_target_walk_forward,
    fit_rich_target_model,
    rich_target_policy,
)
from train import SOS_MIN_TRAIN_ROWS, _write_report, segment_calibration_suite, select_feature_columns


def _match(match_id: str, date: datetime, team1: str, team2: str, team1_win: bool) -> dict:
    return {
        "id": match_id,
        "date": date.strftime("%Y-%m-%d"),
        "date_obj": date,
        "event": "Test Cup",
        "format": "bo3",
        "team1": team1,
        "team2": team2,
        "team1_key": team1.lower(),
        "team2_key": team2.lower(),
        "score1": 2 if team1_win else 0,
        "score2": 0 if team1_win else 2,
        "team1_win": int(team1_win),
    }


class RoadmapA3A5A6A7Tests(unittest.TestCase):
    def test_a3_segment_suite_reports_context_coverage_and_small_groups(self) -> None:
        rows = []
        for index in range(40):
            rows.append({
                "actual": index % 2,
                "prob_team1": 0.65 if index % 2 else 0.35,
                "format": "bo3",
                "environment": "online" if index < 30 else "lan",
                "stage": "group" if index < 30 else "final",
                "event_tier": "A" if index < 30 else None,
            })

        report = segment_calibration_suite(rows, min_n=20)

        self.assertEqual(report["by_environment"]["online"]["n"], 30)
        self.assertIn("note", report["by_environment"]["lan"])
        self.assertEqual(report["coverage"]["environment"]["known_rows"], 40)
        self.assertEqual(report["coverage"]["event_tier"]["known_rows"], 30)
        self.assertIn("note", report["by_event_tier"]["unknown"])

    def test_a3_context_is_propagated_to_walkforward_metadata(self) -> None:
        row = _match("1", datetime(2026, 1, 1), "Alpha", "Beta", True)
        row["match_context"] = {"environment": "lan", "stage": "final"}
        row["event_tier"] = "S"

        _X, _y, meta, _state = build_training_frame([row])

        self.assertEqual(meta[0]["environment"], "lan")
        self.assertEqual(meta[0]["stage"], "final")
        self.assertEqual(meta[0]["event_tier"], "S")

    def test_a5_exact_pruning_is_target_free_and_diagnostics_are_temporal(self) -> None:
        rng = np.random.RandomState(7)
        signal = rng.normal(size=400)
        noise = rng.normal(size=400)
        X = np.column_stack([signal, signal.copy(), np.ones(400), noise])
        columns = ["signal", "signal_duplicate", "constant", "noise"]
        y = (signal + rng.normal(scale=0.2, size=400) > 0).astype(int)

        kept, pruning = prune_exact_redundancies(X, columns)
        diagnostics = feature_pruning_diagnostics(
            X[:, [columns.index(column) for column in kept]],
            y,
            kept,
            shap_rows=[{"feature": "signal", "mean_abs_shap": 1.0}],
            permutation_repeats=2,
        )

        self.assertEqual(kept, ["signal", "noise"])
        self.assertEqual(pruning["removed_constants"], ["constant"])
        self.assertEqual(pruning["removed_exact_redundancies"][0]["feature"], "signal_duplicate")
        self.assertTrue(diagnostics["available"])
        self.assertLess(diagnostics["train_rows"], diagnostics["n_rows"])
        self.assertEqual({row["feature"] for row in diagnostics["features"]}, set(kept))

    def test_a6_bo3_target_auto_gate_and_symmetric_artifact(self) -> None:
        rng = np.random.RandomState(11)
        rows: list[dict] = []
        values: list[list[float]] = []
        periods: list[int] = []
        score_by_class = ((0, 2), (1, 2), (2, 1), (2, 0))
        for index in range(600):
            class_index = index % 4
            score1, score2 = score_by_class[class_index]
            rows.append({
                "id": str(index),
                "date": f"2026-01-{1 + (index // 200):02d}",
                "format": "bo3",
                "score1": score1,
                "score2": score2,
                "team1_win": int(class_index >= 2),
            })
            values.append([(-1.5 + class_index) + rng.normal(scale=0.25)])
            periods.append(index // 20)
        X = np.asarray(values, dtype=float)
        periods_array = np.asarray(periods, dtype=int)

        self.assertEqual(bo3_score_class(rows[0]), 0)
        self.assertTrue(rich_target_policy(rows, min_rows=500, min_class_rows=100)["coverage_ready"])
        evaluation = evaluate_rich_target_walk_forward(
            X,
            rows,
            periods_array,
            ["strength"],
            {"strength"},
            warmup_weeks=2,
            min_train=80,
            min_rows=500,
            min_class_rows=100,
            recency_half_life=0,
        )
        self.assertTrue(evaluation["enabled"])
        self.assertLess(evaluation["multiclass_log_loss"], evaluation["empirical_baseline_log_loss"])

        estimator = fit_rich_target_model(X, rows, ["strength"], {"strength"})
        artifact = ModelArtifact(
            feature_columns=["strength"],
            components=[],
            metadata={"directional_feature_columns": ["strength"]},
            rich_target_estimator=estimator,
            rich_target_classes=list(BO3_SCORE_CLASSES),
        )
        positive, negative = artifact.predict_series_score_distribution([{"strength": 1.0}, {"strength": -1.0}])
        self.assertAlmostEqual(sum(positive.values()), 1.0, places=8)
        self.assertAlmostEqual(positive["2-0"], negative["0-2"], places=8)
        self.assertAlmostEqual(positive["2-1"], negative["1-2"], places=8)

    def test_a7_strength_of_schedule_is_causal_symmetric_and_auto_gated(self) -> None:
        state = ChronologicalState(form_half_life=60.0)
        start = datetime(2026, 1, 1)
        for index in range(5):
            date = start + timedelta(days=index)
            state.elos["alpha"] = 1500.0
            state.elos[f"strong{index}"] = 1800.0
            state.observe(_match(f"a{index}", date, "Alpha", f"Strong{index}", True))
            state.elos["beta"] = 1500.0
            state.elos[f"weak{index}"] = 1200.0
            state.observe(_match(f"b{index}", date + timedelta(hours=1), "Beta", f"Weak{index}", False))

        as_of = start + timedelta(days=6)
        forward = state.emit_features("alpha", "beta", as_of, "Future", "bo3")
        reverse = state.emit_features("beta", "alpha", as_of, "Future", "bo3")

        self.assertEqual(forward["sos_available"], 1.0)
        self.assertGreater(forward["sos_opp_elo_decay_diff"], 0.0)
        self.assertGreater(forward["sos_performance_residual_decay_diff"], 0.0)
        for column in SOS_FEATURE_COLUMNS[:3]:
            self.assertAlmostEqual(forward[column], -reverse[column], places=8)
        self.assertEqual(forward["sos_matches_min"], reverse["sos_matches_min"])

        columns, policy = select_feature_columns([{"sos_available": 1.0} for _ in range(SOS_MIN_TRAIN_ROWS)])
        self.assertTrue(policy["strength_of_schedule"]["enabled"])
        self.assertTrue(all(column in columns for column in SOS_FEATURE_COLUMNS))

    def test_training_report_renders_new_roadmap_sections(self) -> None:
        artifact = ModelArtifact(
            feature_columns=["elo_diff"],
            components=[],
            metadata={
                "trained_at": "2026-07-13T00:00:00Z",
                "production_model": "test",
                "feature_policies": {
                    "feature_pruning": {
                        "available_rows": 100,
                        "min_rows": 0,
                        "enabled": False,
                        "input_columns": 1,
                        "output_columns": 1,
                    }
                },
                "feature_pruning_diagnostics": {
                    "available": True,
                    "holdout_rows": 20,
                    "review_drop_candidates": [],
                },
                "segment_calibration": {
                    "by_format": {
                        "bo3": {
                            "n": 30,
                            "accuracy": 0.6,
                            "log_loss": 0.65,
                            "brier": 0.23,
                            "ece_10": 0.03,
                        }
                    },
                    "by_environment": {},
                    "by_stage": {},
                    "by_event_tier": {},
                    "coverage": {
                        "format": {"known_rows": 30, "total_rows": 30, "coverage": 1.0}
                    },
                },
                "rich_target_bo3_scoreline": {
                    "enabled": True,
                    "available_rows": 2200,
                    "min_rows": 2000,
                    "min_class_rows": 300,
                    "class_counts": {"0-2": 500, "1-2": 500, "2-1": 500, "2-0": 700},
                    "production_scope": "scoreline_props_auxiliary_not_winner_model_a",
                    "note": "auxiliary scoreline model enabled",
                },
            },
        )
        rows = [{"date": "2026-01-01"}, {"date": "2026-07-13"}]
        with TemporaryDirectory() as tmp:
            _write_report(
                Path(tmp),
                {},
                [],
                rows,
                artifact,
                [],
                {"note": "sin odds"},
                {"available": False, "note": "sin odds", "n_odds_rows": 0, "min_train_odds": 100},
                {},
            )
            report = (Path(tmp) / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Calibracion por segmento (A3)", report)
        self.assertIn("Poda y multicolinealidad (A5)", report)
        self.assertIn("Target BO3 enriquecido (A6)", report)


if __name__ == "__main__":
    unittest.main()

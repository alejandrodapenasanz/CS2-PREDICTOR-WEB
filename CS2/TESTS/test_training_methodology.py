from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    DIFF_COLUMNS,
    STRENGTH_INTERACTION_COLUMNS,
    SYM_COLUMNS,
    ChronologicalState,
)
from train import (  # noqa: E402
    causal_candidate_policy,
    candidate_specs,
    chronological_holdout_indices,
    optimize_convex_weights,
    parse_algorithms,
    select_feature_columns,
)


def match(idx: int, team1_win: bool) -> dict:
    return {
        "id": str(idx),
        "team1_key": "alpha",
        "team2_key": "beta",
        "team1": "Alpha",
        "team2": "Beta",
        "score1": 2 if team1_win else 0,
        "score2": 0 if team1_win else 2,
        "team1_win": int(team1_win),
        "date": f"2026-01-{idx:02d}",
        "date_obj": datetime(2026, 1, idx),
        "event": "Test Cup",
        "format": "bo3",
    }


class TrainingMethodologyTests(unittest.TestCase):
    def test_causal_candidate_policy_cannot_see_current_period_labels(self) -> None:
        predictions = {"candidate_a": [], "candidate_b": []}
        for period in range(6):
            for index in range(20):
                actual = (period + index) % 2
                common = {
                    "match_id": f"{period}-{index}",
                    "date": f"2026-02-{period + 1:02d}",
                    "period": period,
                    "actual": actual,
                }
                predictions["candidate_a"].append(
                    {**common, "prob_team1": 0.75 if actual else 0.25}
                )
                predictions["candidate_b"].append(
                    {**common, "prob_team1": 0.25 if actual else 0.75}
                )

        _rows_a, report_a = causal_candidate_policy(
            predictions, ("candidate_a", "candidate_b"), min_history=20
        )
        changed = {
            name: [dict(row) for row in rows]
            for name, rows in predictions.items()
        }
        target_period = 4
        for rows in changed.values():
            for row in rows:
                if row["period"] == target_period:
                    row["actual"] = 1 - row["actual"]
        _rows_b, report_b = causal_candidate_policy(
            changed, ("candidate_a", "candidate_b"), min_history=20
        )

        decision_a = {
            row["period"]: row["selected_candidate"]
            for row in report_a["decisions"]
        }
        decision_b = {
            row["period"]: row["selected_candidate"]
            for row in report_b["decisions"]
        }
        self.assertEqual(decision_a[target_period], decision_b[target_period])

    def test_chronological_holdout_never_mixes_future_rows_into_training(self) -> None:
        y = np.array(([0, 1] * 30), dtype=int)
        train_idx, holdout_idx = chronological_holdout_indices(y, 0.2)

        self.assertLess(train_idx[-1], holdout_idx[0])
        self.assertEqual(set(np.unique(y[train_idx])), {0, 1})
        self.assertEqual(set(np.unique(y[holdout_idx])), {0, 1})

    def test_convex_super_learner_weights_are_valid_and_favor_lower_log_loss(self) -> None:
        y = np.array([0, 1] * 100, dtype=float)
        good = np.where(y == 1, 0.82, 0.18)
        bad = np.where(y == 1, 0.35, 0.65)
        weights = optimize_convex_weights(np.column_stack([good, bad]), y)

        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertGreater(weights[0], weights[1])

    def test_error_aware_strength_features_preserve_team_swap_symmetry(self) -> None:
        state = ChronologicalState()
        for idx, won in enumerate((True, True, False, True, True), start=1):
            state.observe(match(idx, won))

        alpha_beta = state.emit_features("alpha", "beta", datetime(2026, 1, 10), "Test Cup", "bo3")
        beta_alpha = state.emit_features("beta", "alpha", datetime(2026, 1, 10), "Test Cup", "bo3")

        for column in STRENGTH_INTERACTION_COLUMNS:
            with self.subTest(column=column):
                if column in DIFF_COLUMNS:
                    self.assertAlmostEqual(alpha_beta[column], -beta_alpha[column])
                elif column in SYM_COLUMNS:
                    self.assertAlmostEqual(alpha_beta[column], beta_alpha[column])
                else:
                    self.fail(f"{column} debe ser direccional o simetrica")

    def test_core_profile_is_an_explicit_ablation_without_strength_interactions(self) -> None:
        core_columns, core_policy = select_feature_columns([], "core")
        enhanced_columns, enhanced_policy = select_feature_columns([], "error-aware")

        self.assertEqual(core_columns, list(BASE_FEATURE_COLUMNS))
        self.assertFalse(core_policy["strength_interactions"]["enabled"])
        self.assertTrue(enhanced_policy["strength_interactions"]["enabled"])
        self.assertTrue(all(column in enhanced_columns for column in STRENGTH_INTERACTION_COLUMNS))
        self.assertFalse(any(column in core_columns for column in STRENGTH_INTERACTION_COLUMNS))

    def test_all_algorithm_library_contains_individual_and_super_learner_candidates(self) -> None:
        algorithms = parse_algorithms("all")
        specs = candidate_specs(tuple({
            "logistic": "logistic",
            "lightgbm": "gbm",
            "catboost": "catboost",
            "xgboost": "xgboost",
            "random_forest": "random_forest",
        }[algorithm] for algorithm in algorithms))

        self.assertIn("logistic_cal", specs)
        self.assertIn("lightgbm_cal", specs)
        self.assertIn("catboost_cal", specs)
        self.assertIn("xgboost_cal", specs)
        self.assertIn("random_forest_cal", specs)
        self.assertIn("super_learner_cal", specs)


if __name__ == "__main__":
    unittest.main()

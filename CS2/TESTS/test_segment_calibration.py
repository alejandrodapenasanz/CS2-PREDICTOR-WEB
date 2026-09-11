from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import (
    SEGMENT_INTERACTION_DIFF_COLUMNS,
    SEGMENT_INTERACTION_FEATURE_COLUMNS,
    build_training_frame,
    segment_interaction_features,
)
from cs2model.segment_calibration import (
    elo_gap_bin,
    segment_calibration,
    segment_calibration_suite,
)
from evaluate_live_ledger import evaluate
from train import candidate_feature_columns


def _rows(n: int, probability: float, *, observed_rate: float = 0.5) -> list[dict]:
    positives = round(n * observed_rate)
    return [
        {
            "actual": int(index < positives),
            "prob_team1": probability,
            "elo_diff": (-1 if index % 2 else 1) * 125.0,
            "stage": "final",
            "environment": "lan",
            "format": "bo3",
        }
        for index in range(n)
    ]


def test_elo_gap_bins_use_absolute_point_in_time_difference() -> None:
    assert elo_gap_bin(-49.999) == "0-50"
    assert elo_gap_bin(50.0) == "50-100"
    assert elo_gap_bin(-199.9) == "100-200"
    assert elo_gap_bin(300.0) == "300+"
    assert elo_gap_bin(None) is None


def test_segment_report_distinguishes_material_gap_from_sampling_noise() -> None:
    calibrated = segment_calibration(_rows(200, 0.5), "stage", min_n=100)
    biased = segment_calibration(_rows(200, 0.8), "stage", min_n=100)
    small = segment_calibration(_rows(20, 0.8), "stage", min_n=100)

    assert calibrated["final"]["status"] == "compatible_with_sampling_noise"
    assert biased["final"]["status"] == "deviation_detected"
    assert biased["final"]["calibration_gap"] == -0.3
    assert biased["final"]["reliability"]
    assert small["final"]["status"] == "insufficient_sample"
    assert "sin muestra suficiente" in small["final"]["note"]


def test_segment_suite_includes_requested_dimensions_and_coverage() -> None:
    report = segment_calibration_suite(_rows(120, 0.5), min_n=100)

    assert report["by_elo_gap"]["100-200"]["n"] == 120
    assert report["by_stage"]["final"]["n"] == 120
    assert report["by_environment"]["lan"]["n"] == 120
    assert report["coverage"]["elo_gap_bin"]["coverage"] == 1.0


def test_segment_interactions_are_optional_and_preserve_team_swap_symmetry() -> None:
    forward = segment_interaction_features(
        {
            "elo_diff": 125.0,
            "context_is_lan": 1.0,
            "context_stage_final": 1.0,
        }
    )
    reverse = segment_interaction_features(
        {
            "elo_diff": -125.0,
            "context_is_lan": 1.0,
            "context_stage_final": 1.0,
        }
    )
    for column in SEGMENT_INTERACTION_DIFF_COLUMNS:
        assert forward[column] == -reverse[column]
    assert forward["segment_interactions_available"] == 1.0
    assert reverse["segment_interactions_available"] == 1.0

    default_columns = candidate_feature_columns("core", {})
    experiment_columns = candidate_feature_columns("core", {}, include_segment_interactions=True)
    assert not any(column in default_columns for column in SEGMENT_INTERACTION_FEATURE_COLUMNS)
    assert all(column in experiment_columns for column in SEGMENT_INTERACTION_FEATURE_COLUMNS)


def test_future_results_cannot_change_prior_segment_interactions() -> None:
    def match(match_id: str, date: str, winner: int) -> dict:
        return {
            "id": match_id,
            "date": date,
            "date_obj": datetime.fromisoformat(date),
            "event": "Temporal Cup",
            "format": "bo3",
            "team1": "Alpha",
            "team2": "Beta",
            "team1_key": "alpha",
            "team2_key": "beta",
            "score1": 2 if winner else 0,
            "score2": 0 if winner else 2,
            "team1_win": winner,
            "match_context": {"environment": "lan", "stage": "group"},
        }

    history = [match("1", "2026-01-01", 1), match("2", "2026-01-02", 0)]
    future = match("3", "2026-02-01", 1)
    prefix_features, _labels, _meta, _state = build_training_frame(history)
    full_features, _labels, _meta, _state = build_training_frame([*history, future])

    for index in range(len(history)):
        for column in SEGMENT_INTERACTION_FEATURE_COLUMNS:
            assert prefix_features[index][column] == full_features[index][column]


def test_live_ledger_report_is_read_only_and_current_sample_cannot_decide() -> None:
    with TemporaryDirectory() as temporary:
        db_path = Path(temporary) / "live.db"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            """
            CREATE TABLE matches(
                match_id INTEGER PRIMARY KEY,
                environment TEXT,
                stage TEXT,
                best_of INTEGER
            );
            CREATE TABLE prediction_ledger(
                ledger_id INTEGER PRIMARY KEY,
                match_id INTEGER,
                artifact_sha256 TEXT,
                model_version TEXT,
                prediction_correct INTEGER,
                realized_log_loss REAL,
                realized_brier REAL,
                actual_team1_win INTEGER,
                prob_team1 REAL,
                features_json TEXT,
                ledger_status TEXT,
                result_filled_at_utc TEXT
            );
            """
        )
        for index in range(120):
            actual = index % 2
            probability = 0.5
            connection.execute(
                "INSERT INTO matches VALUES (?, 'online', 'group', 3)",
                (index + 1,),
            )
            connection.execute(
                "INSERT INTO prediction_ledger VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    index + 1,
                    index + 1,
                    "artifact-a",
                    "v1",
                    int((probability >= 0.5) == bool(actual)),
                    -(actual * math.log(probability) + (1 - actual) * math.log(1 - probability)),
                    (probability - actual) ** 2,
                    actual,
                    probability,
                    json.dumps({"elo_diff": 75.0}),
                    "evaluated",
                    "2026-01-01T00:00:00Z",
                ),
            )
        connection.commit()
        before = connection.total_changes
        connection.close()

        report = evaluate(db_path)

        assert report["segment_calibration"]["by_elo_gap"]["50-100"]["n"] == 120
        assert not report["segment_calibration"]["live_decision_policy"]["decision_allowed"]
        check = sqlite3.connect(db_path)
        assert check.execute("SELECT COUNT(*) FROM prediction_ledger").fetchone()[0] == 120
        assert check.total_changes == 0
        check.close()
        assert before == 240

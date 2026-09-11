from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "CS2" / "MODEL" / "results" / "favorite_accuracy_bands.json"
WEB_PATH = REPO_ROOT / "WEB" / "index.html"
BUILD_WEB_PATH = REPO_ROOT / "WEB" / "build_web.py"


def _load_build_web():
    spec = importlib.util.spec_from_file_location("build_web_accuracy_test", BUILD_WEB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_favorite_accuracy_bands_are_consistent_walk_forward_results() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    bands = payload["bands"]

    assert payload["method"].startswith("walk_forward")
    assert payload["n"] == sum(int(band["n"]) for band in bands)
    assert payload["correct"] == sum(int(band["correct"]) for band in bands)
    assert abs(payload["accuracy"] - payload["correct"] / payload["n"]) < 1e-12
    assert all(0.0 <= float(band["accuracy"]) <= 1.0 for band in bands)
    assert all(0.5 <= float(band["average_predicted_probability"]) <= 1.0 for band in bands)


def test_accuracy_timeline_deduplicates_and_builds_all_requested_windows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions_walkforward.csv"
    fields = ["model", "match_id", "date", "actual", "prob_team1"]
    rows = [
        {
            "model": "nested_model_policy",
            "match_id": str(index),
            "date": f"2026-01-{index:02d}",
            "actual": index % 2,
            "prob_team1": 0.8,
        }
        for index in range(1, 11)
    ]
    rows.append(dict(rows[-1]))
    rows.append(
        {
            "model": "other_model",
            "match_id": "ignored",
            "date": "2026-01-10",
            "actual": 1,
            "prob_team1": 0.9,
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    timeline = _load_build_web().build_favorite_accuracy_timeline(
        path,
        "nested_model_policy",
    )

    assert timeline["available_start"] == "2026-01-01"
    assert timeline["available_end"] == "2026-01-10"
    assert set(timeline["windows"]) == {"7d", "1m", "3m", "6m", "1y"}
    assert timeline["windows"]["7d"]["start"] == "2026-01-04"
    assert timeline["windows"]["7d"]["n"] == 7
    assert timeline["windows"]["7d"]["correct"] == 3
    assert len(timeline["windows"]["7d"]["points"]) == 7
    assert timeline["windows"]["1y"]["n"] == 10


def test_current_timeline_uses_all_causal_policy_predictions() -> None:
    predictions_path = REPO_ROOT / "CS2" / "MODEL" / "results" / "predictions_walkforward.csv"
    module = _load_build_web()
    timeline = module.build_favorite_accuracy_timeline(
        predictions_path,
        "nested_model_policy",
    )

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["match_id"]: row for row in csv.DictReader(handle) if row["model"] == "nested_model_policy"}
    expected_correct = sum((float(row["prob_team1"]) >= 0.5) == (int(row["actual"]) == 1) for row in rows.values())
    expected_dates = [row["date"][:10] for row in rows.values()]
    assert timeline["windows"]["1y"]["n"] == len(rows)
    assert timeline["windows"]["1y"]["correct"] == expected_correct
    assert timeline["available_start"] == min(expected_dates)
    assert timeline["available_end"] == max(expected_dates)


def _create_ledger_fixture(path: Path, rows: list[tuple]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE prediction_ledger(
                ledger_id INTEGER PRIMARY KEY,
                kickoff_utc TEXT NOT NULL,
                result_filled_at_utc TEXT,
                prob_team1 REAL NOT NULL,
                ledger_status TEXT NOT NULL,
                actual_team1_win INTEGER,
                prediction_correct INTEGER,
                prediction_regime TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO prediction_ledger(
                ledger_id, kickoff_utc, result_filled_at_utc, prob_team1,
                ledger_status, actual_team1_win, prediction_correct,
                prediction_regime
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def test_live_ledger_fixture_builds_calibration_and_rolling_accuracy(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cs2.db"
    rows = [
        (1, "2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z", 0.1, "evaluated", 0, 1, "no_odds"),
        (2, "2026-01-02T10:00:00Z", "2026-01-02T12:00:00Z", 0.2, "evaluated", 1, 0, "odds"),
        (3, "2026-01-03T10:00:00Z", "2026-01-03T12:00:00Z", 0.7, "evaluated", 1, 1, "odds"),
        (4, "2026-01-04T10:00:00Z", "2026-01-04T12:00:00Z", 0.8, "evaluated", 0, 0, "no_odds"),
        (5, "2026-01-05T10:00:00Z", "2026-01-05T12:00:00Z", 0.9, "evaluated", 1, 1, None),
        (6, "2026-01-06T10:00:00Z", None, 0.9, "open", 1, 1, "odds"),
    ]
    _create_ledger_fixture(db_path, rows)

    payload = _load_build_web().build_cs2_ledger_performance(db_path, rolling_window=3)

    assert payload["status"] == "available"
    assert payload["evaluated_count"] == 5
    assert payload["unlabeled_regime_count"] == 1
    assert payload["regime_breakdown_available"] is True
    all_scope = payload["scopes"]["all"]
    assert all_scope["n"] == 5
    assert all_scope["correct"] == 3
    assert abs(all_scope["accuracy"] - 0.6) < 1e-12
    assert sum(bucket["n"] for bucket in all_scope["calibration"]["bins"]) == 5
    assert all(bucket["n"] > 0 for bucket in all_scope["calibration"]["bins"])
    assert all_scope["rolling_accuracy"]["latest"] == {
        "ledger_id": 5,
        "at_utc": "2026-01-05T12:00:00Z",
        "window_start_utc": "2026-01-03T12:00:00Z",
        "n": 3,
        "correct": 2,
        "accuracy": 2 / 3,
    }
    assert payload["scopes"]["odds"]["n"] == 2
    assert payload["scopes"]["no_odds"]["n"] == 2


def test_empty_ledger_publishes_no_chart_data(tmp_path: Path) -> None:
    db_path = tmp_path / "cs2.db"
    _create_ledger_fixture(db_path, [])

    payload = _load_build_web().build_cs2_ledger_performance(db_path)

    assert payload["status"] == "not_computed"
    assert payload["evaluated_count"] == 0
    assert payload["scopes"]["all"]["accuracy"] is None
    assert payload["scopes"]["all"]["calibration"]["bins"] == []
    assert payload["scopes"]["all"]["rolling_accuracy"]["points"] == []


def test_model_view_renders_live_ledger_charts_and_hides_empty_state() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    render_model = html.split("function renderModel(){", 1)[1].split(
        "function drawFavoriteAccuracy(){",
        1,
    )[0]

    assert 'id="liveAccuracyCanvas"' in html
    assert 'id="liveAccuracyStats"' in html
    assert 'id="liveAccuracyTooltip"' in html
    assert 'id="calibrationBucketRows"' in html
    assert 'data-ledger-scope="odds"' in html
    assert 'data-ledger-scope="no_odds"' in html
    assert "drawLiveAccuracy();" in render_model
    assert "drawLiveCalibration();" in render_model
    assert "DATA.model?.live_performance" in html
    assert "setCs2LiveVisibility(hasLive)" in html
    assert 'classList.toggle("hidden",!visible)' in html
    assert (
        "favorite_accuracy_timeline"
        not in BUILD_WEB_PATH.read_text(encoding="utf-8")
        .split("def model_payload", 1)[1]
        .split("def _month_start", 1)[0]
    )


def test_model_view_exposes_odds_architecture_and_per_match_regime() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    module_source = BUILD_WEB_PATH.read_text(encoding="utf-8")

    assert '"odds_architectures": odds_architectures' in module_source
    assert 'MODEL_ROOT / "results" / "odds_architecture_eval.json"' in module_source
    assert "DATA.model?.odds_architectures" in html
    assert "p.prediction_regime" in html
    assert "p.prediction_architecture" in html
    assert "p.opening_odds_recovered" in html

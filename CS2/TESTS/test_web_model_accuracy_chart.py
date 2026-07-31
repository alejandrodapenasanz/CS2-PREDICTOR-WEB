from __future__ import annotations

import csv
import importlib.util
import json
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
    assert all(
        0.5 <= float(band["average_predicted_probability"]) <= 1.0
        for band in bands
    )


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
    predictions_path = (
        REPO_ROOT / "CS2" / "MODEL" / "results" / "predictions_walkforward.csv"
    )
    module = _load_build_web()
    timeline = module.build_favorite_accuracy_timeline(
        predictions_path,
        "nested_model_policy",
    )

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["match_id"]: row
            for row in csv.DictReader(handle)
            if row["model"] == "nested_model_policy"
        }
    expected_correct = sum(
        (float(row["prob_team1"]) >= 0.5) == (int(row["actual"]) == 1)
        for row in rows.values()
    )
    expected_dates = [row["date"][:10] for row in rows.values()]
    assert timeline["windows"]["1y"]["n"] == len(rows)
    assert timeline["windows"]["1y"]["correct"] == expected_correct
    assert timeline["available_start"] == min(expected_dates)
    assert timeline["available_end"] == max(expected_dates)


def test_model_view_renders_temporal_accuracy_chart() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    render_model = html.split("function renderModel(){", 1)[1].split(
        "function drawCalibration(){",
        1,
    )[0]

    assert 'id="accuracyCanvas"' in html
    assert 'id="accuracyStats"' in html
    assert 'id="accuracyTooltip"' in html
    for window in ("7d", "1m", "3m", "6m", "1y"):
        assert f'data-accuracy-range="{window}"' in html
    assert "drawFavoriteAccuracy();" in render_model
    assert "DATA.model?.favorite_accuracy_timeline" in render_model
    assert "point.accuracy" in render_model
    assert "point.date" in render_model
    assert "windowData.points" in render_model

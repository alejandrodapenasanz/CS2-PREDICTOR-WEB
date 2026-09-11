"""Offline acquisition, append-only persistence and strict availability regressions."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from BBDD.round_history_store import ingest_archives
from PIPELINE.round_history import parse_round_history
from cs2model.pistols import load_pistol_store, pistol_features_asof, team_summary
from cs2model.pistols import PISTOL_COLUMNS, PISTOL_DIFF_COLUMNS

URL = "https://www.hltv.org/stats/matches/mapstatsid/200001/alpha-vs-beta"


def map_html(*, overtime: bool = False, half_length: int = 12) -> str:
    """Generate the inspected HLTV icon/score/separator contract, including multi-OT."""
    halves = [[0] * (half_length // 2) + [1] * (half_length - half_length // 2)]
    if overtime:
        halves += [[1] * (half_length // 2) + [0] * (half_length - half_length // 2), [0] * 3, [1] * 3, [0] * 3, [0]]
    else:
        halves += [[1] * 2 + [0] * (half_length + 1 - half_length // 2)]
    sides = ["ct", "t", "t", "ct", "ct", "t"]
    scores = [0, 0]
    rendered: list[list[str]] = []
    for index, winners in enumerate(halves):
        pair = ["<div class='round-history-bar'></div>"] * 2
        for winner in winners:
            scores[winner] += 1
            for team in (0, 1):
                side = sides[index] if team == 0 else ("t" if sides[index] == "ct" else "ct")
                icon = side + "_win.svg" if team == winner else "emptyHistory.svg"
                score = f"{scores[0]}-{scores[1]}" if team == winner else ""
                pair[team] += f"<img class='round-history-outcome' src='/img/static/scoreboard/{icon}' title='{score}'>"
        # HLTV pads all unplayed positions at the end of a half.
        target = half_length if index < 2 else 3
        for team in (0, 1):
            pair[team] += (
                "<img class='round-history-outcome' src='/img/static/scoreboard/emptyHistory.svg' title=''>"
                * (target - len(winners))
            )
        rendered.append(pair)
    stamp = int(datetime(2026, 7, 1, 10, tzinfo=UTC).timestamp() * 1000)
    html = f"<a class='match-page-link' href='/matches/1000001/alpha-vs-beta'></a><div class='match-info-box'><span data-unix='{stamp}'></span>Mirage"
    for side, team, name, score in zip(("left", "right"), (101, 102), ("Alpha", "Beta"), scores, strict=True):
        html += f"<div class='team-{side}'><a href='/stats/teams/{team}/team'>{name}</a><div class='bold'>{score}</div></div>"
    html += "</div>"
    for start, stop in ((0, 2), (2, len(rendered))):
        if start == stop:
            continue
        html += "<div class='round-history-con" + (" round-history-overtime" if start else "") + "'>"
        for team, name in enumerate(("Alpha", "Beta")):
            html += f"<div class='round-history-team-row'><img class='round-history-team' title='{name}'>"
            html += "".join(pair[team] for pair in rendered[start:stop]) + "</div>"
        html += "</div>"
    return html


@pytest.mark.parametrize("half_length", [12, 15])
@pytest.mark.parametrize("overtime", [False, True])
def test_round_parser_scores_format_and_pistols(half_length: int, overtime: bool) -> None:
    """MR12/MR15 and multiple overtimes never create extra pistols."""
    payload = parse_round_history(map_html(overtime=overtime, half_length=half_length), URL)
    assert payload["format"] == f"mr{half_length}"
    assert [r["round_number"] for r in payload["rounds"] if r["is_pistol"]] == [1, half_length + 1]
    assert len(payload["rounds"]) == sum(t["score"] for t in payload["teams"])
    assert not any(r["overtime"] and r["is_pistol"] for r in payload["rounds"])


@pytest.mark.parametrize(
    "before,after", [("title='1-0'", "title='9-0'"), ("ct_win.svg", "unknown.svg"), ("title='Alpha'", "title='Beta'")]
)
def test_round_parser_rejects_corruption(before: str, after: str) -> None:
    """Unknown semantics, shifted scores and orientation cannot enter storage."""
    with pytest.raises(ValueError):
        parse_round_history(map_html().replace(before, after, 1), URL)


def fixture_database() -> sqlite3.Connection:
    """Create a minimal isolated source database with a sacred sentinel."""
    c = sqlite3.connect(":memory:")
    c.executescript(
        "CREATE TABLE teams(team_id INTEGER PRIMARY KEY,hltv_id TEXT);"
        "CREATE TABLE matches(match_id INTEGER PRIMARY KEY,hltv_match_id TEXT,team1_id INTEGER,team2_id INTEGER);"
        "CREATE TABLE maps(map_id INTEGER PRIMARY KEY,match_id INTEGER,hltv_mapstats_id TEXT);"
        "CREATE TABLE prediction_ledger(evidence TEXT); INSERT INTO prediction_ledger VALUES('untouched');"
        "INSERT INTO teams VALUES(1,'101'),(2,'102'); INSERT INTO matches VALUES(1,'1000001',1,2);"
        "INSERT INTO maps VALUES(1,1,'200001');"
    )
    c.execute("PRAGMA foreign_keys=ON")
    return c


def write_archive(root: Path, *, captured: str = "2026-07-02T10:00:00Z") -> Path:
    """Write a fixture capture with an independently verified content hash."""
    path = root / "raw_html/mapstats/200001.html.gz"
    path.parent.mkdir(parents=True)
    html = map_html()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)
    path.with_suffix(".gz.json").write_text(
        json.dumps({"url": URL, "sha256": hashlib.sha256(html.encode()).hexdigest(), "captured_at": captured}),
        encoding="utf-8",
    )
    return path


def test_ingestion_idempotent_readonly_and_asof(tmp_path: Path) -> None:
    """No writes in dry-run; repeat ingest deduplicates and same-day capture is excluded."""
    c = fixture_database()
    write_archive(tmp_path)
    dry = ingest_archives(c, tmp_path)
    assert dry["counts"]["valid_captures"] == 1
    assert not c.execute("SELECT 1 FROM sqlite_master WHERE name='map_round_sources'").fetchone()
    assert ingest_archives(c, tmp_path, apply=True)["counts"]["valid_captures"] == 1
    assert ingest_archives(c, tmp_path, apply=True)["counts"]["already_stored"] == 1
    assert c.execute("SELECT evidence FROM prediction_ledger").fetchall() == [("untouched",)]
    store = load_pistol_store(c)
    assert team_summary(store, "101", "2026-07-02")["n"] == 0
    summary = team_summary(store, "101", "2026-07-03")
    assert summary["n"] == 2
    assert summary["rates"]["win"] == 0.5
    assert summary["rates"]["convert2"] == 1.0
    future = deepcopy(store)
    future["teams"]["101"] += [
        {**r, "played_at": "2026-07-04", "captured_at": "2026-07-05"} for r in store["teams"]["101"]
    ]
    assert pistol_features_asof(store, "101", "102", "2026-07-03") == pistol_features_asof(
        future, "101", "102", "2026-07-03"
    )
    assert team_summary(future, "101", "2026-07-03") == summary
    late = deepcopy(store)
    late["teams"]["101"] += [{**r, "captured_at": "2026-07-03T00:00:00Z"} for r in store["teams"]["101"]]
    assert team_summary(late, "101", "2026-07-03") == summary
    reverse = pistol_features_asof(store, "102", "101", "2026-07-03")
    assert reverse["pistol_sample_total"] == 4
    c.close()


def test_sample_flags_and_swap_symmetry(tmp_path: Path) -> None:
    """Low coverage cannot create a strong rate advantage; swapping teams is coherent."""
    c = fixture_database()
    write_archive(tmp_path)
    ingest_archives(c, tmp_path, apply=True)
    store = load_pistol_store(c)
    assert pistol_features_asof(store, "101", "102", "2026-07-03")["pistol_available"] == 0
    for team in store["teams"]:
        store["teams"][team] *= 6
    a = pistol_features_asof(store, "101", "102", "2026-07-03")
    b = pistol_features_asof(store, "102", "101", "2026-07-03")
    assert a["pistol_available"] == 1
    for column in PISTOL_COLUMNS:
        assert a[column] == (-b[column] if column in PISTOL_DIFF_COLUMNS else b[column])
    c.close()


def test_pistols_are_candidate_only_and_served_through_same_join() -> None:
    """Current model columns stay unchanged; training and serving call one as-of reader."""
    from cs2model.features import FEATURE_COLUMNS, external_snapshot_features
    from cs2model.dataio import external_snapshot_features_asof
    from run_pistol_ablation import calibration_fraction
    import numpy as np

    assert not set(PISTOL_COLUMNS) & set(FEATURE_COLUMNS)
    joined = external_snapshot_features_asof({}, "101", "102", "2026-07-03")
    features = external_snapshot_features(joined)
    assert {c: features[c] for c in PISTOL_COLUMNS} == joined["pistol_snapshot_features"]
    dates = [f"2026-07-{day:02d}" for day in range(1, 11) for _ in range(4)]
    _, split = calibration_fraction(dates, np.tile([0, 1], 20), 0.18)
    assert split["fit_end"] < split["calibration_start"]


def test_hash_mismatch_keeps_original_and_does_not_ingest(tmp_path: Path) -> None:
    """Corrupt source provenance stays quarantined, never redated as valid."""
    c = fixture_database()
    path = write_archive(tmp_path)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(map_html() + "tampered")
    report = ingest_archives(c, tmp_path, apply=True)
    assert report["quarantine_reasons"] == {"source_hash_mismatch": 1}
    assert c.execute("SELECT COUNT(*) FROM map_round_sources").fetchone()[0] == 0
    assert path.exists()
    c.close()


def test_offline_round_pipeline_prediction_gate_and_future_capture(tmp_path: Path) -> None:
    """Archive -> DB -> as-of features -> serialized model -> gate, without network."""
    import numpy as np
    from cs2model.artifacts import Component, ModelArtifact, ProbabilityColumnEstimator, load_artifact
    from cs2model.dataio import external_snapshot_features_asof
    from cs2model.features import external_snapshot_features
    from cs2model.promotion import assert_same_artifact_same_cohort_metrics, compare_candidate_to_incumbent

    c = fixture_database()
    write_archive(tmp_path / "early")
    ingest_archives(c, tmp_path / "early", apply=True)

    def feature_row() -> dict[str, float]:
        return external_snapshot_features(
            external_snapshot_features_asof({"pistols": load_pistol_store(c)}, "101", "102", "2026-07-03")
        )

    before = feature_row()
    artifact = ModelArtifact(
        feature_columns=PISTOL_COLUMNS,
        components=[Component("fixture", ProbabilityColumnEstimator(0), None)],
        metadata={"date_max": "2026-07-02"},
    )
    probabilities = artifact.predict_proba_team1([before, before])
    assert np.all(np.isfinite(probabilities))
    assert before["pistol_available"] == 0  # Still predicts with insufficient history.
    artifact_path = tmp_path / "candidate.pkl"
    artifact.save(artifact_path)
    restored = load_artifact(artifact_path)
    assert restored is not None
    assert_same_artifact_same_cohort_metrics(restored, [before, before], [0, 1], probabilities)
    cohort = [{"match_id": str(i), "date": "2026-07-03", "actual": i, "features": before} for i in (0, 1)]
    predictions = [{**row, "prob_team1": float(probabilities[i])} for i, row in enumerate(cohort)]
    decision = compare_candidate_to_incumbent(restored, predictions, cohort, min_samples=2)
    assert not decision.promote  # The same artifact cannot cause churn.
    write_archive(tmp_path / "later", captured="2026-07-09T10:00:00Z")
    ingest_archives(c, tmp_path / "later", apply=True)
    assert c.execute("SELECT COUNT(*) FROM map_round_sources").fetchone()[0] == 2
    assert feature_row() == before
    np.testing.assert_array_equal(restored.predict_proba_team1([feature_row(), feature_row()]), probabilities)
    assert c.execute("SELECT evidence FROM prediction_ledger").fetchall() == [("untouched",)]
    c.close()


def test_truncated_gzip_is_quarantined_without_breaking_ingestion(tmp_path: Path) -> None:
    """Interrupted archive writes cannot bring down the next ordinary ingestion."""
    c = fixture_database()
    path = write_archive(tmp_path)
    path.write_bytes(path.read_bytes()[:20])
    report = ingest_archives(c, tmp_path, apply=True)
    assert report["counts"]["quarantined"] == 1
    assert c.execute("SELECT COUNT(*) FROM map_round_sources").fetchone()[0] == 0
    c.close()

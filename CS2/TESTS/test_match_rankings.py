"""Match-page rankings reach retraining and serving with the same causal recipe."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from BBDD.ranking_store import ensure_schema
from cs2model.dataio import external_snapshot_features_asof, load_external_feature_store
from cs2model.features import build_training_frame, external_snapshot_features
from cs2model.match_rankings import (
    MATCH_RANKING_COLUMNS,
    MATCH_RANKING_DIFF_COLUMNS,
    MATCH_RANKING_SYM_COLUMNS,
    load_match_ranking_store,
    match_ranking_payload_asof,
)
from PIPELINE.enrich_predictions import model_external_features_for_order
import train


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY)")
    database.execute("CREATE TABLE teams(team_id INTEGER PRIMARY KEY)")
    ensure_schema(database)
    try:
        yield database
    finally:
        database.close()


def insert(
    conn: sqlite3.Connection,
    team: str,
    position: int,
    *,
    kind: str = "hltv",
    edition: str = "2026-09-07",
    captured: str = "2026-09-08T12:00:00Z",
) -> None:
    index = conn.execute("SELECT COUNT(*) FROM match_team_ranking_observations").fetchone()[0] + 1
    conn.execute(
        """INSERT INTO match_team_ranking_observations
           (hltv_team_id,hltv_match_id,ranking_type,position,ranking_date,captured_at_utc,
            ranking_url,match_url,source_file,source_sha256)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            team,
            str(2_390_000 + index),
            kind,
            position,
            edition,
            captured,
            f"https://www.hltv.org/ranking/{team}",
            f"https://www.hltv.org/matches/{index}",
            f"fixture/{index}.gz",
            f"{index:064x}",
        ),
    )


def payload(conn: sqlite3.Connection, day: str = "2026-09-09T20:00:00") -> dict:
    return external_snapshot_features_asof(load_external_feature_store(conn), "1", "2", day)


def test_capture_during_d_is_not_available_even_before_kickoff(conn: sqlite3.Connection) -> None:
    insert(conn, "1", 40)
    insert(conn, "2", 117)
    same_day = payload(conn, "2026-09-08T23:59:59")
    assert same_day["match_ranking_features"]["match_ranking_available"] == 0
    later = payload(conn)
    assert later["match_ranking_features"]["match_ranking_hltv_position_advantage"] == 77
    assert later["match_ranking_features"]["match_ranking_edition_age_days_max"] == 2
    source = later["match_ranking_evidence"]["sources"]["hltv"]
    assert source["edition"] == "2026-09-07"
    assert len(source["source_sha256"]) == 2


def test_missing_table_and_unknown_timestamps_remain_missing(conn: sqlite3.Connection) -> None:
    insert(conn, "1", 10, captured="2026-09-08T10:00:00")
    insert(conn, "2", 20)
    # UTC timezone is mandatory; no silent conversion of a local timestamp.
    assert payload(conn)["match_ranking_features"]["match_ranking_available"] == 0
    conn.execute("DROP TABLE match_team_ranking_observations")
    assert load_match_ranking_store(conn) == {}
    assert payload(conn)["match_ranking_features"]["match_ranking_available"] == 0


def test_latest_edition_wins_not_latest_download_and_no_mixed_editions(conn: sqlite3.Connection) -> None:
    insert(conn, "1", 10)
    insert(conn, "2", 20)
    # A freshly captured old match must not replace a more recent edition.
    insert(conn, "1", 150, edition="2026-09-01", captured="2026-09-09T10:00:00Z")
    assert payload(conn, "2026-09-10")["match_ranking_features"]["match_ranking_hltv_position_advantage"] == 10
    insert(conn, "1", 5, edition="2026-09-09", captured="2026-09-09T11:00:00Z")
    result = payload(conn, "2026-09-10")
    assert result["match_ranking_features"]["match_ranking_available"] == 0
    assert result["match_ranking_evidence"]["sources"]["hltv"]["reason"] == "different_editions"


def test_future_results_captures_and_conflicts_leave_asof_features_unchanged(conn: sqlite3.Connection) -> None:
    insert(conn, "1", 10)
    insert(conn, "2", 20)
    before = payload(conn)

    def training_rows() -> list[dict]:
        return [
            {
                "id": "match",
                "date": "2026-09-09",
                "date_obj": datetime(2026, 9, 9),
                "team1": "a",
                "team2": "b",
                "team1_key": "a",
                "team2_key": "b",
                "event": "fixture",
                "format": "bo3",
                "score1": 2,
                "score2": 0,
                "team1_win": 1,
                **payload(conn),
            }
        ]

    old_frame = build_training_frame(training_rows())[0]
    insert(conn, "1", 99, captured="2026-09-09T10:00:00Z")
    insert(conn, "2", 1, edition="2026-09-10", captured="2026-09-10T10:00:00Z")
    assert payload(conn) == before
    rows = training_rows()
    rows.append(
        {
            **rows[0],
            "id": "future",
            "date": "2026-09-11",
            "date_obj": datetime(2026, 9, 11),
            "score1": 0,
            "score2": 2,
            "team1_win": 0,
        }
    )
    assert build_training_frame(rows)[0][0] == old_frame[0]
    # Once the conflict is actually known, this edition becomes unusable.
    conflict = payload(conn, "2026-09-10")
    assert conflict["match_ranking_features"]["match_ranking_available"] == 0
    assert conflict["match_ranking_evidence"]["sources"]["hltv"]["team1_status"] == "conflicting_positions_in_edition"


def test_repeated_captures_do_not_refresh_age_or_modify_legacy_inputs(conn: sqlite3.Connection) -> None:
    insert(conn, "1", 10)
    insert(conn, "2", 20)
    before = payload(conn, "2026-09-12")
    insert(conn, "1", 10, captured="2026-09-10T10:00:00Z")
    assert payload(conn, "2026-09-12") == before
    assert before["ranking_snapshot_features"]["ranking_available"] == 0
    assert before["match_ranking_features"]["match_ranking_available"] == 1


def test_training_and_live_serving_swap_teams_identically(conn: sqlite3.Connection) -> None:
    for kind, ranks in (("hltv", (40, 117)), ("valve", (49, 108))):
        insert(conn, "1", ranks[0], kind=kind)
        insert(conn, "2", ranks[1], kind=kind)
    training = external_snapshot_features(payload(conn))
    serving = model_external_features_for_order(training)
    reverse = model_external_features_for_order(training, reverse=True)
    reverse_source = match_ranking_payload_asof(load_match_ranking_store(conn), "2", "1", datetime(2026, 9, 9))
    for column in MATCH_RANKING_COLUMNS:
        assert serving[column] == training[column]
        assert reverse[column] == reverse_source["match_ranking_features"][column]
    matrix = np.array([[training[column] for column in MATCH_RANKING_COLUMNS]])
    augmented, labels = train.augment(matrix, np.array([1]), MATCH_RANKING_COLUMNS)
    assert labels.tolist() == [1, 0]
    assert augmented[1].tolist() == [reverse[column] for column in MATCH_RANKING_COLUMNS]


def test_retrain_auto_family_is_eligible_but_never_forced() -> None:
    assert train.MATCH_RANKING_MIN_TRAIN_ROWS == 200
    assert set(MATCH_RANKING_COLUMNS) <= set(train.candidate_feature_columns("core"))
    for count, enabled in ((199, False), (200, True)):
        columns, policies = train.select_feature_columns([{"match_ranking_available": 1.0}] * count)
        assert policies["match_rankings"]["enabled"] is enabled
        assert all((column in columns) is enabled for column in MATCH_RANKING_COLUMNS)


@pytest.mark.parametrize("informative", [True, False])
def test_temporal_feature_gate_accepts_signal_rejects_noise_without_future_labels(informative: bool) -> None:
    """Synthetic proof of wiring, NOT a claim of real-world ranking accuracy."""
    periods = np.repeat(np.arange(16), 20)
    labels = np.tile([0, 1], len(periods) // 2)
    columns = train.candidate_feature_columns("core")
    rows = []
    for label in labels:
        row = dict.fromkeys(columns, 0.0)
        row["match_ranking_available"] = 1.0
        row["match_ranking_hltv_available"] = 1.0
        row["match_ranking_hltv_position_advantage"] = float(20 * (2 * label - 1)) if informative else 0.0
        rows.append(row)
    thresholds = {name: 100_000 for name, *_ in train.fold_local_family_specs()}
    thresholds["match_rankings"] = 200

    def evaluate(target: np.ndarray) -> tuple[list[str], dict]:
        return train.select_fold_local_features(
            rows,
            target,
            periods,
            periods < 14,
            columns,
            ["elo_prob_centered", "glicko_prob_centered"],
            feature_thresholds=thresholds,
            validation_periods=4,
            min_validation_rows=40,
            min_validation_available_rows=20,
            min_log_loss_gain=0.001,
            seed=42,
        )

    selected, report = evaluate(labels)
    family = report["families"]["match_rankings"]
    assert family["available_rows"] == 280
    assert family["validation_available_rows"] == 80
    assert family["enabled"] is informative
    assert all((column in selected) is informative for column in MATCH_RANKING_COLUMNS)
    changed = labels.copy()
    changed[periods >= 14] = 1 - changed[periods >= 14]
    assert evaluate(changed) == (selected, report)


def test_zero_coverage_report_stays_wait() -> None:
    _, policies = train.select_feature_columns([dict.fromkeys(MATCH_RANKING_COLUMNS, 0.0)] * 500)
    assert policies["match_rankings"]["available_rows"] == 0
    assert not policies["match_rankings"]["enabled"]
    assert "excluded" in policies["match_rankings"]["note"]

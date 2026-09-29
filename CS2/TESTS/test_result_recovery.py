"""Offline result recovery: frozen predictions, maps, retries and isolation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from BBDD import ingest, result_store
from PIPELINE.result_recovery import parse_result_html, run_daily
import test_integrity_ledger_and_feature_gates as ledger_fixtures


@pytest.fixture
def database():
    fixture = ledger_fixtures.PredictionLedgerTests()
    fixture.setUp()
    result_store.ensure_schema(fixture.conn)
    yield fixture
    fixture.tearDown()


def result(**overrides):
    return {
        "hltv_match_id": "99",
        "team1_hltv_id": "1",
        "team2_hltv_id": "2",
        "source_url": "https://www.hltv.org/matches/99/alpha-beta",
        "obtained_at_utc": "2026-01-02T15:00:00Z",
        "status": "finished",
        "score_t1": 2,
        "score_t2": 1,
        "best_of": 3,
        "maps": [],
        **overrides,
    }


def maps():
    return [
        {"map_number": n, "map_name": name, "rounds_t1": a, "rounds_t2": b, "overtime": ot}
        for n, name, a, b, ot in [(1, "Mirage", 13, 9, None), (2, "Inferno", 7, 13, None), (3, "Nuke", 16, 14, 1)]
    ]


def test_partial_complete_repeat_non_degradation_and_no_feature_feedback(database):
    conn = database.conn
    # Model inputs are entirely unchanged, including same-day rows/availability.
    before = {
        t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")]
        for t in ("matches", "maps", "raw_results", "raw_snapshots", "match_features")
    }
    with conn:
        assert result_store.persist(conn, result())["new_results"] == 1
    assert conn.execute("SELECT complete FROM result_status").fetchone()[0] == 0
    with conn:
        assert result_store.persist(conn, result(maps=maps()))["maps_added"] == 3
        assert result_store.persist(conn, result(maps=maps()))["maps_added"] == 0
        result_store.persist(conn, result())
        result_store.persist(conn, result(status="live", score_t1=None, score_t2=None))
    assert tuple(conn.execute("SELECT status,complete,score_t1,score_t2 FROM result_status").fetchone()) == (
        "finished",
        1,
        2,
        1,
    )
    assert conn.execute("SELECT COUNT(*) FROM result_maps").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM result_evidence").fetchone()[0] == 2
    assert before == {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")] for t in before}
    assert (
        result_store.plan(conn, {"99": {"link": result()["source_url"]}}, now="2026-01-03T00:00:00Z")["selected"] == []
    )


def test_settlement_uses_frozen_probability_and_market_upset(database):
    conn = database.conn
    match, match_id = database._match_and_id()
    prediction = database._row("2026-01-02T10:00:00Z", 0.7)
    prediction["prediction"]["odds_prob_team1"] = 0.6
    ingest.upsert_prediction_ledger(conn, prediction, match_id=match_id, match_row=match, model_version="model@v1")
    frozen = conn.execute("SELECT prediction_json FROM prediction_ledger").fetchone()[0]
    with conn:
        result_store.persist(conn, result(score_t1=0, score_t2=2))
        assert ingest.finalize_prediction_ledger(conn, "2026-01-02T16:00:00Z") == 1
        assert ingest.finalize_prediction_ledger(conn, "2026-01-02T17:00:00Z") == 0
    row = dict(conn.execute("SELECT * FROM result_settlement_v1").fetchone())
    assert (row["prediction_correct"], row["market_upset"], row["market_favorite"], row["actual_winner"]) == (
        0,
        1,
        1,
        2,
    )
    assert row["predicted_probability_team1"] == 0.7
    assert row["frozen_prediction_json"] == frozen


def test_cancelled_never_settles_and_conflicts_preserve_result(database):
    conn = database.conn
    match, match_id = database._match_and_id()
    ingest.upsert_prediction_ledger(
        conn, database._row("2026-01-02T10:00:00Z", 0.7), match_id=match_id, match_row=match, model_version="model@v1"
    )
    with conn:
        result_store.persist(conn, result(status="cancelled", score_t1=None, score_t2=None))
        assert ingest.finalize_prediction_ledger(conn, "2026-01-02T16:00:00Z") == 0
    with pytest.raises(ValueError, match="Conflicting terminal"):
        with conn:
            result_store.persist(conn, result(maps=maps()))
    assert conn.execute("SELECT status FROM result_status").fetchone()[0] == "cancelled"


def fake_client(tmp_path: Path, *, fail: bool = False):
    from PIPELINE import start

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    html = '<div class="countdown">Match over</div>Best of 3' + "".join(
        f'<div class="mapholder"><span class="mapname">{m["map_name"]}</span>'
        f'<div class="results-left"><span class="results-team-score">{m["rounds_t1"]}</span></div>'
        f'<div class="results-right"><span class="results-team-score">{m["rounds_t2"]}</span></div></div>'
        for m in maps()
    )
    return SimpleNamespace(
        now_utc=lambda: "2026-01-03T16:00:00Z",
        FETCH_MIN_INTERVAL=1,
        fetch_html=Mock(side_effect=TimeoutError("temporary") if fail else None, return_value=html),
        save_raw_html=lambda *args: {"captured_at": "2026-01-03T16:00:00Z"},
        write_json=write,
        parse_match_detail_html=lambda _: {
            "match": {"team1": {"id": "1", "score": "2"}, "team2": {"id": "2", "score": "1"}}
        },
        parse_int=start.parse_int,
        hltv_url=start.hltv_url,
        log=lambda *a, **kw: None,
        looks_like_cf_or_waf_problem=lambda exc: False,
        FetchBudgetExceeded=start.FetchBudgetExceeded,
        FetchSuppressedError=start.FetchSuppressedError,
    )


def test_daily_failure_then_retry_with_bo3_html(database, tmp_path):
    master = {"99": {"link": result()["source_url"]}}
    database.conn.commit()
    client = fake_client(tmp_path, fail=True)
    report = run_daily(database.db_path, master, tmp_path, client=client)
    assert report["errors"] == 1
    assert run_daily(database.db_path, master, tmp_path, client=client)["reviewed"] == 0
    client = fake_client(tmp_path)
    client.now_utc = lambda: "2026-01-04T16:00:00Z"
    report = run_daily(database.db_path, master, tmp_path, client=client)
    assert (report["errors"], report["maps_added"], report["coverage"]["pending"]) == (0, 3, 0)
    assert run_daily(database.db_path, master, tmp_path, client=client)["reviewed"] == 0


def test_live_html_numeric_score_is_not_terminal(tmp_path):
    client = fake_client(tmp_path)
    parsed = parse_result_html(
        '<div class="countdown">LIVE</div>',
        {"entity_key": "99", "url": result()["source_url"]},
        "2026-01-03T16:00:00Z",
        client=client,
    )
    assert parsed["status"] == "live"
    assert parsed["score_t1"] is None


def test_reversed_stable_team_ids_are_normalized_without_changing_evidence(database):
    reversed_maps = [{**m, "rounds_t1": m["rounds_t2"], "rounds_t2": m["rounds_t1"]} for m in maps()]
    with database.conn:
        result_store.persist(
            database.conn, result(team1_hltv_id="2", team2_hltv_id="1", score_t1=1, score_t2=2, maps=reversed_maps)
        )
    row = dict(database.conn.execute("SELECT * FROM result_details_v1").fetchone())
    assert (row["score_t1"], row["score_t2"], row["winner_team_id"], row["loser_team_id"], row["complete"]) == (
        2,
        1,
        1,
        2,
        1,
    )
    evidence = json.loads(database.conn.execute("SELECT payload_json FROM result_evidence").fetchone()[0])
    assert evidence["source_payload"]["team1_hltv_id"] == "2"

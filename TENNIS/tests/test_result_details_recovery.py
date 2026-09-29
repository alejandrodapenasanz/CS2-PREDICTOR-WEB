"""Result details and automatic recovery without network or model mutations."""

from datetime import UTC, datetime, timedelta
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from src.operations import OperationsStore
from src.operations.result_details import coverage, parse_sets, pending, replay_details
from src.operations.result_recovery import recover_results, profile_observations
from tests.test_operations_store import prediction_row, result_row, temporary_store


NOW = datetime(2026, 7, 31, 14, tzinfo=UTC)


def test_partial_settled_match_enriched_three_sets_and_never_degraded():
    with temporary_store() as store:
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        frozen = tuple(store.connection.execute("SELECT * FROM predictions").fetchone())
        global_row = {
            **result_row(
                first_sets=1, second_sets=2, winner_slug="beta-player", winner_side="player_2"
            ),
            "sets_score": "1-2",
        }
        first = store.reconcile_observations("r1", pd.DataFrame([global_row]))
        assert first.settlements_inserted == 1
        assert len(pending(store.connection, NOW)) == 1
        detailed = {**global_row, "sets_score": "6-4 3-6 6-7(5)"}
        store.reconcile_observations("r2", pd.DataFrame([detailed]))
        store.reconcile_observations("r2", pd.DataFrame([detailed]))
        store.reconcile_observations("r3", pd.DataFrame([global_row]))
        assert replay_details(store.connection)["sets_added"] == 0
        sets = store.connection.execute("SELECT * FROM result_sets ORDER BY set_number").fetchall()
        assert len(sets) == 3
        assert (sets[2]["tiebreak_player_1"], sets[2]["tiebreak_player_2"]) == (5, None)
        assert len(pending(store.connection, NOW)) == 0
        assert coverage(store.connection, NOW)["with_detailed_sets"] == 1
        assert tuple(store.connection.execute("SELECT * FROM predictions").fetchone()) == frozen
        assert len(store.load_settlements()) == 1
        view = dict(store.connection.execute("SELECT * FROM result_settlement_v1").fetchone())
        assert (view["prediction_correct"], view["market_upset"], view["actual_winner"]) == (
            0,
            1,
            "beta-player",
        )


def test_cancel_walkover_retirement_kept_without_settlements():
    for status, score in (("cancelled", ""), ("walkover", "W/O"), ("retirement", "6-4 2-1 RET")):
        with temporary_store() as store:
            store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
            row = {
                **result_row(),
                "status": status,
                "sets_score": score,
                "player_1_sets_won": None,
                "player_2_sets_won": None,
            }
            store.reconcile_observations("r", pd.DataFrame([row]))
            assert not len(store.load_settlements())
            detail = dict(store.connection.execute("SELECT * FROM result_details").fetchone())
            assert detail["status"] == status
            assert detail["complete"] == (status != "retirement")
            assert store.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


def test_winner_oriented_ratio_sets_are_reversed_once():
    with temporary_store() as store:
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        row = {
            **result_row(
                first_sets=0, second_sets=2, winner_slug="beta-player", winner_side="player_2"
            ),
            "sets_score": "6-4 7-6(3)",
            "result_evidence": "tennisratio_mapped_terminal_result",
        }
        store.reconcile_observations("r", pd.DataFrame([row]))
        sets = store.connection.execute("SELECT * FROM result_sets ORDER BY set_number").fetchall()
        assert [(s["player_1_games"], s["player_2_games"]) for s in sets] == [(4, 6), (6, 7)]
        assert sets[1]["tiebreak_player_1"] == 3
        assert sets[1]["tiebreak_player_2"] is None


def test_conflicting_detail_is_quarantined_not_overwritten():
    with temporary_store() as store:
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        store.reconcile_observations("r1", pd.DataFrame([result_row()]))
        previous = [tuple(r) for r in store.connection.execute("SELECT * FROM result_sets")]
        store.reconcile_observations(
            "r2", pd.DataFrame([{**result_row(), "sets_score": "6-0 6-0"}])
        )
        assert [tuple(r) for r in store.connection.execute("SELECT * FROM result_sets")] == previous
        assert (
            store.connection.execute("SELECT COUNT(*) FROM result_detail_conflicts").fetchone()[0]
            == 1
        )
        assert len(store.load_settlements()) == 1


def test_retry_after_temporary_failure_and_skip_complete(tmp_path):
    # OperationsStore intentionally restricts paths to TENNIS or VAULT.
    from src.config import TESTS_DIR
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=TESTS_DIR) as directory:
        db = Path(directory) / "operations.sqlite3"
        with OperationsStore(db, clock=lambda: NOW) as store:
            store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        failing = Mock(side_effect=RuntimeError("temporary source failure"))
        args = dict(
            database_path=db,
            clock=lambda: NOW,
            max_profiles=0,
            max_dates=1,
            ratio_loader=lambda *a, **kw: (),
            result_refresher=failing,
        )
        report = recover_results(**args)
        assert report["errors"] and report["pending"] == 1
        assert recover_results(**args)["reviewed"] == 0
        snap = SimpleNamespace(
            matches=pd.DataFrame([result_row()]),
            match_date=NOW.date() - timedelta(days=1),
            retrieved_at_utc=NOW,
            source_url="https://www.tennisexplorer.com/matches/",
            snapshot_sha256="a" * 64,
            html_path=None,
            metadata_path=None,
        )
        args.update(clock=lambda: NOW + timedelta(days=1), result_refresher=lambda *a, **kw: snap)
        report = recover_results(**args)
        assert not report["errors"]
        assert (report["sets_added"], report["settled"], report["pending"]) == (2, 1, 0)
        assert recover_results(**args)["reviewed"] == 0


def test_super_tiebreak_and_aggregate_not_invented():
    assert parse_sets("2-1") == []
    rows = parse_sets("6-4 3-6 [10-8]")
    assert rows[2]["score_kind"] == "match_tiebreak_points"
    assert parse_sets("7-6(4)")[0]["tiebreak_player_1"] is None


def test_real_profile_parser_exact_mapping_without_guessing():
    from src.tennisratio.parser import parse_profile_html

    html = (Path(__file__).parent / "fixtures/tennisratio/AliceAlpha.html").read_bytes()
    url = "https://www.tennisratio.com/players/AliceAlpha.html"
    parsed = parse_profile_html(html, source_url=url)
    payload = SimpleNamespace(
        content=html,
        source_url=url,
        retrieved_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
        sha256="b" * 64,
    )
    target = {
        "source_match_id": "tennisratio:1",
        "match_date": "2026-08-20",
        "gender": "M",
        "player_1_slug": "tennisratio:AliceAlpha",
        "player_2_slug": "tennisratio:BobBeta",
    }
    rows = profile_observations(parsed, payload, [target])
    assert len(rows) == 1
    assert rows[0]["sets_score"] == "6-4 3-6 6-2"
    assert rows[0]["player_1_sets_won"] == 2
    assert rows[0]["winner_slug"] == "tennisratio:AliceAlpha"
    assert (
        profile_observations(
            parsed, payload, [target, {**target, "source_match_id": "tennisratio:2"}]
        )
        == []
    )


def test_result_capture_today_preserves_old_prediction_and_feature_audit():
    with temporary_store() as store:
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        before = {
            t: [tuple(r) for r in store.connection.execute(f"SELECT * FROM {t}")]
            for t in ("predictions", "player_statistics", "matches")
        }
        row = result_row(observed_at_utc=NOW.isoformat())
        store.reconcile_observations("r", pd.DataFrame([row]))
        assert before == {
            t: [tuple(r) for r in store.connection.execute(f"SELECT * FROM {t}")] for t in before
        }
        detail = dict(store.connection.execute("SELECT * FROM result_details").fetchone())
        assert detail["event_date"] == "2026-07-30"
        assert datetime.fromisoformat(detail["obtained_at_utc"]).date() == NOW.date()


def test_newly_recovered_result_is_excluded_by_real_elo_availability_cutoff():
    """Existing Explorer handoff may consume source results only from D+1."""
    import sqlite3
    import gc
    from src.elo.operational import load_explorer_operational_results
    from datetime import date

    with temporary_store() as store:
        database = Path(store.connection.execute("PRAGMA database_list").fetchone()[2])
        mapping = database.parent / "mapping.sqlite3"
        with closing(sqlite3.connect(mapping)) as conn, conn:
            conn.execute("CREATE TABLE player_mappings(gender TEXT,slug TEXT,player_id INTEGER)")
            conn.executemany(
                "INSERT INTO player_mappings VALUES ('M',?,?)",
                [("alpha-player", 101), ("beta-player", 202)],
            )
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        kwargs = dict(
            operations_database_path=database,
            player_mapping_database_path=mapping,
            cutoff_date=date(2026, 7, 1),
            as_of_date=NOW.date(),
        )
        before = load_explorer_operational_results(**kwargs)
        store.reconcile_observations(
            "r", pd.DataFrame([result_row(observed_at_utc=NOW.isoformat())])
        )
        after = load_explorer_operational_results(**kwargs)
        assert before == after
        assert (
            len(
                load_explorer_operational_results(
                    **{**kwargs, "as_of_date": NOW.date() + timedelta(days=1)}
                ).results
            )
            == 1
        )
        # The existing reader uses sqlite's transaction context manager, not
        # closing(). Collect its released read-only handles before Windows
        # removes the fixture directory; no production reader is changed here.
        gc.collect()


def test_live_set_grows_to_final_without_duplicate_set():
    with temporary_store() as store:
        store.register_prediction_run("p", pd.DataFrame([prediction_row()]))
        live = {
            **result_row(),
            "status": "live",
            "sets_score": "4-2",
            "winner_side": None,
            "winner_slug": None,
            "player_1_sets_won": None,
            "player_2_sets_won": None,
        }
        store.reconcile_observations("r1", pd.DataFrame([live]))
        store.reconcile_observations("r2", pd.DataFrame([result_row()]))
        assert coverage(store.connection, NOW)["with_detailed_sets"] == 1
        assert store.connection.execute("SELECT COUNT(*) FROM result_sets").fetchone()[0] == 2

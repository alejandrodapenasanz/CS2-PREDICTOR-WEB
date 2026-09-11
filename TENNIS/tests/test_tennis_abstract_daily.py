"""Offline daily acquisition, provenance, retention and temporal regressions."""

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from unittest.mock import Mock
import zlib

import pytest

from src.responsible_http import HttpResponse, RateLimitedError, WafBlockedError
from src.tennis_abstract_access import ORIGIN
from src.tennis_abstract_daily import (
    InventoryPlayer,
    acquisition_status,
    inventory_player,
    update_daily,
)
from src.tennis_abstract_history_audit import HISTORY_HEADER, fetch_player_history
from src.tennis_abstract_store import TennisAbstractStore, acquisition_lock
from src.tennis_abstract_browser import TennisAbstractPlayerTimeout

START = datetime(2026, 9, 6, 12, tzinfo=UTC)


def player(key="JannikSinner", gender="M", rank=1):
    """Use a source-owned legacy key with dated inventory evidence."""

    return InventoryPlayer(
        gender,
        key,
        key,
        ORIGIN + "/cgi-bin/player.cgi?p=" + key,
        ORIGIN + "/reports/atp_elo_ratings.html",
        START.isoformat(),
        rank,
    )


def page(points="60", source_date="20260831"):
    """Keep a known header and a literal history row, without invented event dates."""

    row = [""] * 48
    row[0], row[1], row[23] = source_date, "US Open", points
    return (
        "<script>var matchhead="
        + repr(HISTORY_HEADER)
        + ";var matchmx="
        + repr([row])
        + ";</script>"
    ).encode()


def client_for(*responses):
    """Inject static responses through the actual history parser."""

    client = Mock()
    client.get.side_effect = [
        item if isinstance(item, Exception) else HttpResponse(200, item, {}, ORIGIN)
        for item in responses
    ]
    return client


def test_daily_refresh_is_persistent_idempotent_and_full_inventory_by_default(tmp_path):
    """A second run fetches nothing and does not duplicate facts or snapshots."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player(), player("CarlosAlcaraz", rank=2)]
    client = client_for(page(), page("70"))
    first = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert first["status"] == "completed"
    assert first["refreshed"] == first["stored_players"] == 2
    assert first["pending"] == first["model_ready_rows"] == 0
    again = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert again["status"] == "already_refreshed"
    assert again["unique_source_rows"] == 2
    assert client.get.call_count == 2
    assert not first["production_changed"]


def test_partial_limit_resumes_without_starving_remaining_players(tmp_path):
    """A test sample is visibly partial, and the next run resumes the unvisited tail."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player(), player("CarlosAlcaraz", rank=2)]
    client = client_for(page(), page())
    first = update_daily(
        store_path=path, players=players, client=client, clock=lambda: START, max_profiles=1
    )
    assert first["status"] == "partial" and first["pending"] == 1
    second = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert second["status"] == "completed" and second["already_refreshed"] == 1
    assert client.get.call_args_list[-1].args[0].endswith("p=CarlosAlcaraz")


def test_future_captures_and_raw_rotation_do_not_change_prior_evidence(tmp_path):
    """Only two raw versions remain; all distinct facts and causal snapshots survive."""

    store = TennisAbstractStore(tmp_path / "tennis_abstract.sqlite3")
    try:
        for day_offset in range(3):
            at = START + timedelta(days=day_offset)
            history = fetch_player_history(
                client_for(page(str(60 + day_offset))), "M", "JannikSinner"
            )
            store.save(history, captured_at=at, inventory={"fixture": True})
            if day_offset == 0:
                before = store.snapshot_before("M", "JannikSinner", date(2026, 9, 7))
                assert store.snapshot_before("M", "JannikSinner", START.date()) is None
        assert before == store.snapshot_before("M", "JannikSinner", date(2026, 9, 7))
        assert before["rows"][0][23] == "60"
        assert before["exact_match_date"] is None and not before["model_ready"]
        assert before["date_semantics"] == "tournament_date_not_verified_match_date"
        assert store.connection.execute("SELECT count(*) FROM ta_raw_documents").fetchone()[0] == 2
        assert store.connection.execute("SELECT count(*) FROM ta_rows").fetchone()[0] == 3
        assert store.connection.execute("SELECT count(*) FROM ta_snapshots").fetchone()[0] == 3
        for row in store.connection.execute("SELECT content_zlib FROM ta_raw_documents"):
            assert zlib.decompress(row[0]).startswith(b"<script>")
    finally:
        store.close()


def test_reversion_to_old_content_is_a_new_observation_not_a_new_fact(tmp_path):
    """A -> B -> A restores A prospectively without rewriting either earlier date."""

    store = TennisAbstractStore(tmp_path / "tennis_abstract.sqlite3")
    try:
        for offset, points in enumerate(("60", "61", "60")):
            store.save(
                fetch_player_history(client_for(page(points)), "M", "JannikSinner"),
                captured_at=START + timedelta(days=offset),
                inventory={},
            )
        assert store.snapshot_before("M", "JannikSinner", date(2026, 9, 8))["rows"][0][23] == "61"
        assert store.snapshot_before("M", "JannikSinner", date(2026, 9, 9))["rows"][0][23] == "60"
        assert store.connection.execute("SELECT count(*) FROM ta_snapshots").fetchone()[0] == 2
    finally:
        store.close()


def test_failed_refresh_keeps_last_good_and_persists_host_pause(tmp_path):
    """A WAF never erases a good player snapshot, nor retries via a new process."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player()]
    update_daily(store_path=path, players=players, client=client_for(page()), clock=lambda: START)
    tomorrow = START + timedelta(days=1)
    blocked = WafBlockedError("challenge", HttpResponse(403, b"blocked", {}, ORIGIN))
    client = client_for(blocked)
    result = update_daily(store_path=path, players=players, client=client, clock=lambda: tomorrow)
    assert result["status"] == "failed" and result["stored_players"] == 1
    assert result["pending"] == 1 and result["failure"]["status_code"] == 403
    again = update_daily(store_path=path, players=players, client=client, clock=lambda: tomorrow)
    assert again["status"] == "deferred"
    assert client.get.call_count == 1


def test_completed_capture_uses_clock_after_network_not_start_of_run(tmp_path):
    """A fetch crossing midnight is unavailable on the new civil date too."""

    path = tmp_path / "tennis_abstract.sqlite3"
    moments = iter([START, START + timedelta(days=1), START + timedelta(days=1)])
    update_daily(
        store_path=path, players=[player()], client=client_for(page()), clock=lambda: next(moments)
    )
    store = TennisAbstractStore(path)
    try:
        assert store.snapshot_before("M", "JannikSinner", date(2026, 9, 7)) is None
        assert store.snapshot_before("M", "JannikSinner", date(2026, 9, 8)) is not None
    finally:
        store.close()


@pytest.mark.parametrize(
    "url",
    [
        ORIGIN + "/cgi-bin/player.cgi?p=123/Carlos-Alcaraz",
        ORIGIN + "/cgi-bin/player.cgi?p=JannikSinner&p=Other",
        "https://other.test/cgi-bin/player.cgi?p=JannikSinner",
        ORIGIN + "/cgi-bin/wplayer.cgi?p=JannikSinner",
    ],
)
def test_inventory_never_guesses_or_crosses_identity_scopes(url):
    """Do not build keys from display names, wrong-gender pages or new ID formats."""

    with pytest.raises(ValueError):
        inventory_player({"gender": "M", "player_url": url})


def test_acquisition_lock_is_exclusive_and_released(tmp_path):
    """Distinct source-store commits do not release the process-wide refresh lock."""

    path = tmp_path / "tennis_abstract.sqlite3"
    with acquisition_lock(path):
        with pytest.raises(RuntimeError, match="already running"):
            with acquisition_lock(path):
                pytest.fail("Concurrent acquisition acquired the writer lock")
    with acquisition_lock(path):
        pass


def test_production_database_filename_is_rejected(tmp_path):
    """An accidental operational DB path cannot create acquisition tables there."""

    with pytest.raises(ValueError, match="dedicated"):
        TennisAbstractStore(tmp_path / "tennis.sqlite3")
    assert not (tmp_path / "tennis.sqlite3").exists()


def test_exhausted_429_persists_retry_after_and_does_not_request_next_player(tmp_path):
    """The service preserves HTTP-layer cooldown instead of retrying another URL."""

    path = tmp_path / "tennis_abstract.sqlite3"
    deadline = START + timedelta(hours=1)
    blocked = RateLimitedError(ORIGIN, deadline.timestamp(), fresh_response=True)
    client = client_for(page(), blocked)
    players = [player(), player("CarlosAlcaraz", rank=2), player("Other", rank=3)]
    result = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert result["status"] == "deferred" and result["refreshed"] == 1
    assert result["pending"] == 2
    assert datetime.fromisoformat(result["failure"]["retry_at_utc"]) == deadline
    next_client = client_for(page())
    repeated = update_daily(
        store_path=path, players=players, client=next_client, clock=lambda: START
    )
    assert repeated["status"] == "deferred"
    next_client.get.assert_not_called()
    assert client.get.call_count == 2


def test_real_abstract_client_recovers_plain_429_and_keeps_scoped_robots(tmp_path, monkeypatch):
    """Exercise the common Scrapling route, replacing only wire I/O and time."""

    from src import tennis_abstract_access as access
    from src.responsible_http import build_http_client
    from tests.test_responsible_http import _Clock, _Response
    from tests.test_tennisratio_rate_limit import _SequenceTransport

    clock = _Clock()
    url = ORIGIN + "/jsmatches/ArynaSabalenka.js"
    robots = ORIGIN + "/robots.txt"
    wire = _SequenceTransport(
        {
            robots: [_Response(200, b"User-agent: *\nDisallow: /jsmatches/\n", {}, robots)],
            url: [
                _Response(429, b"Too many requests", {"Retry-After": "40"}, url),
                _Response(200, page(), {}, url),
            ],
        }
    )

    def factory(**kwargs):
        """Retain production host settings and operator scope, with a fixture wire."""

        return build_http_client(**kwargs, transport=wire, clock=clock.time, sleeper=clock.sleep)

    monkeypatch.setattr(access, "build_http_client", factory)
    client = access.TennisAbstractAcquisitionClient(cache_root=tmp_path)
    try:
        # The explicitly authorized data route need not fetch robots implicitly;
        # exercise the ordinary robots path too without changing the exception.
        assert b"Disallow: /jsmatches/" in client.get(robots).content
        assert client.get(url).content == page()
        assert client.get(url).from_cache
        assert wire.calls == [robots, url, url]
        assert sum(clock.sleeps) >= 40
    finally:
        client.close()


def test_schema_error_never_replaces_a_successful_snapshot(tmp_path):
    """Malformed HTML is reported, not silently transformed into empty stats."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player()]
    update_daily(store_path=path, players=players, client=client_for(page()), clock=lambda: START)
    result = update_daily(
        store_path=path,
        players=players,
        client=client_for(b"<html>wrong</html>"),
        clock=lambda: START + timedelta(days=1),
    )
    assert result["status"] == "failed" and result["pending"] == 1
    assert result["unique_source_rows"] == 1
    assert result["latest_rows_with_serve_points"] == 1


def test_incomplete_rows_are_preserved_unmapped_while_other_players_continue(tmp_path):
    """The operator-approved short-row quarantine never pads cells or loses evidence."""

    from src.tennis_abstract_history_audit import literal_array

    path = tmp_path / "tennis_abstract.sqlite3"
    full = literal_array(page().decode(), "matchmx")[0]
    short = full[:27]
    body = (
        "<script>var matchhead="
        + repr(HISTORY_HEADER)
        + ";var matchmx="
        + repr([full, short])
        + ";</script>"
    ).encode()
    players = [player("ElenaRybakina", "F"), player("Other", rank=2)]
    client = client_for(body, page())
    report = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert report["status"] == "completed" and report["refreshed"] == 2
    assert report["latest_quarantined_rows"] == 1
    assert report["latest_player_rows"] == 2
    store = TennisAbstractStore(path)
    try:
        saved = store.snapshot_before("F", "ElenaRybakina", date(2026, 9, 7))
        assert saved["rows"] == [full]
        assert saved["quarantined_evidence"][0]["cells"] == short
        assert saved["quarantined_evidence"][0]["source_row_index"] == 1
        assert saved["quarantined_evidence"][0]["width"] == 27
        assert saved["model_ready"] is False
        # Re-reading after a future capture leaves both valid and quarantined evidence stable.
        store.save(
            fetch_player_history(client_for(page("99")), "F", "ElenaRybakina"),
            captured_at=START + timedelta(days=2),
            inventory={},
        )
        assert store.snapshot_before("F", "ElenaRybakina", date(2026, 9, 7)) == saved
    finally:
        store.close()


def test_parser_revision_can_resume_schema_pause_but_never_waf_or_429():
    """A new approved schema contract is not a transport pause override."""

    from src.tennis_abstract_daily import pause_applies
    from src.tennis_abstract_store import utc_text

    pause = {"retry_at_utc": utc_text(START + timedelta(days=1)), "type": "ValueError"}
    assert not pause_applies(pause, utc_text(START))
    for kind in ("WafBlockedError", "RateLimitedError", "RobotsDisallowedError"):
        assert pause_applies({**pause, "type": kind}, utc_text(START))
    assert pause_applies({**pause, "parser_contract_version": 2}, utc_text(START))


def test_status_does_not_create_missing_store_or_start_network(tmp_path, monkeypatch):
    """A first status check is strictly read-only, including absent directories."""

    factory = Mock(side_effect=AssertionError("Status must not create a network client"))
    monkeypatch.setattr("src.tennis_abstract_daily.TennisAbstractAcquisitionClient", factory)
    path = tmp_path / "not-created" / "tennis_abstract.sqlite3"
    result = acquisition_status(store_path=path, players=[player()], clock=lambda: START)
    assert result["status"] == "not_started"
    assert result["never_acquired"] == result["pending_today"] == 1
    assert not path.parent.exists()
    factory.assert_not_called()


def test_status_sees_committed_progress_during_acquisition_without_writes(tmp_path):
    """Live counters are not the stale final report, and the source file is unchanged."""

    import hashlib

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player(), player("CarlosAlcaraz", rank=2)]
    update_daily(
        store_path=path, players=players[:1], client=client_for(page()), clock=lambda: START
    )
    store = TennisAbstractStore(path)
    try:
        store.save(
            fetch_player_history(client_for(page("70")), "M", "CarlosAlcaraz"),
            captured_at=START + timedelta(minutes=1),
            inventory={},
        )
    finally:
        store.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with acquisition_lock(path):
        result = acquisition_status(store_path=path, players=players, clock=lambda: START)
    assert result["status"] == "up_to_date"
    assert result["refreshed_today"] == result["stored_inventory_players"] == 2
    assert result["pending_today"] == result["never_acquired"] == 0
    assert result["last_run"]["inventory_players"] == 1
    assert result["last_success_utc"] == (START + timedelta(minutes=1)).isoformat(
        timespec="microseconds"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_status_separates_today_inventory_and_all_source_coverage(tmp_path):
    """Old or out-of-inventory snapshots must not falsely complete today's inventory."""

    path = tmp_path / "tennis_abstract.sqlite3"
    update_daily(
        store_path=path,
        players=[player(), player("Extra")],
        client=client_for(page(), page()),
        clock=lambda: START,
    )
    tomorrow = START + timedelta(days=1)
    result = acquisition_status(
        store_path=path,
        players=[player(), player("CarlosAlcaraz")],
        clock=lambda: tomorrow,
    )
    assert result["status"] == "pending"
    assert result["stored_players"] == 2
    assert result["stored_inventory_players"] == result["never_acquired"] == 1
    assert result["refreshed_today"] == 0 and result["pending_today"] == 2
    assert result["model_ready_rows"] == 0 and not result["production_changed"]


def test_status_reports_server_pause_without_resetting_it(tmp_path):
    """A status query never acts as an explicit browser retry or clears Retry-After."""

    path = tmp_path / "tennis_abstract.sqlite3"
    deadline = START + timedelta(hours=1)
    update_daily(
        store_path=path,
        players=[player()],
        client=client_for(RateLimitedError(ORIGIN, deadline.timestamp(), fresh_response=True)),
        clock=lambda: START,
    )
    result = acquisition_status(store_path=path, players=[player()], clock=lambda: START)
    assert result["status"] == "deferred"
    assert datetime.fromisoformat(result["pause"]["retry_at_utc"]) == deadline
    again = acquisition_status(store_path=path, players=[player()], clock=lambda: START)
    assert result == again


def test_status_store_enforces_sqlite_read_only(tmp_path):
    """The read-only source connection rejects accidental writes at the SQLite layer."""

    import sqlite3

    path = tmp_path / "tennis_abstract.sqlite3"
    TennisAbstractStore(path).close()
    store = TennisAbstractStore(path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.set_state("unexpected", True)
    finally:
        store.close()


def test_isolated_timeout_keeps_snapshot_continues_and_defers_only_that_player(tmp_path):
    """One timeout cannot stop other players, erase evidence or cause an immediate retry."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player(), player("CarlosAlcaraz", rank=2)]
    update_daily(
        store_path=path, players=players[:1], client=client_for(page()), clock=lambda: START
    )
    tomorrow = START + timedelta(days=1)
    # Explicitly make the previously acquired player first in this fixture's scheduling.
    store = TennisAbstractStore(path)
    store.record_failure("M", "CarlosAlcaraz", START + timedelta(hours=1), "fixture")
    before = store.snapshot_before("M", "JannikSinner", date(2026, 9, 7))
    store.close()
    failed = TennisAbstractPlayerTimeout(ORIGIN + "/cgi-bin/player-classic.cgi?p=JannikSinner")
    client = client_for(failed, page("70"))
    result = update_daily(store_path=path, players=players, client=client, clock=lambda: tomorrow)
    assert client.get.call_count == 2
    assert result["status"] == "partial" and result["pending"] == 1
    assert result["refreshed"] == 1 and result["failure"] is None
    assert result["player_timeouts"][0]["player"] == "M:JannikSinner"
    status = acquisition_status(store_path=path, players=players, clock=lambda: tomorrow)
    assert status["pause"] is None and len(status["deferred_player_timeouts"]) == 1
    later_client = client_for(page("80"))
    again = update_daily(
        store_path=path, players=players, client=later_client, clock=lambda: tomorrow
    )
    assert again["skipped_player_timeouts"] == 1
    later_client.get.assert_not_called()
    recovered = update_daily(
        store_path=path,
        players=players,
        client=later_client,
        clock=lambda: tomorrow + timedelta(minutes=31),
    )
    assert recovered["status"] == "completed"
    store = TennisAbstractStore(path, read_only=True)
    try:
        assert store.state("player_timeouts") == {}
        assert store.snapshot_before("M", "JannikSinner", date(2026, 9, 7)) == before
    finally:
        store.close()


def test_consecutive_timeouts_stop_with_short_local_pause(tmp_path):
    """Three navigation failures bound a possible outage; no fourth player is requested."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [player(key, rank=i) for i, key in enumerate(("One", "Two", "Three", "Four"), 1)]
    client = client_for(*(TennisAbstractPlayerTimeout(ORIGIN) for _ in range(3)), page())
    result = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert result["status"] == "deferred"
    assert result["failure"]["type"] == "ConsecutivePlayerTimeouts"
    assert datetime.fromisoformat(result["failure"]["retry_at_utc"]) == START + timedelta(
        minutes=15
    )
    assert client.get.call_count == 3 and result["pending"] == 4
    repeated = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert repeated["status"] == "deferred" and client.get.call_count == 3


def test_success_resets_timeout_streak_but_waf_still_stops(tmp_path):
    """Interleaved healthy pages permit progress; a subsequent WAF remains host-wide."""

    path = tmp_path / "tennis_abstract.sqlite3"
    players = [
        player(key, rank=i) for i, key in enumerate(("One", "Two", "Three", "Four", "Five"), 1)
    ]
    timeout = TennisAbstractPlayerTimeout(ORIGIN)
    blocked = WafBlockedError("challenge", HttpResponse(403, b"blocked", {}, ORIGIN))
    client = client_for(timeout, page(), timeout, blocked, page())
    result = update_daily(store_path=path, players=players, client=client, clock=lambda: START)
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "WafBlockedError"
    assert client.get.call_count == 4 and result["refreshed"] == 1


def test_nonconsecutive_timeouts_do_not_trigger_outage_pause(tmp_path):
    """Two failures, success, then two failures must not count as four consecutive failures."""

    players = [
        player(key, rank=i)
        for i, key in enumerate(("One", "Two", "Three", "Four", "Five", "Six"), 1)
    ]
    timeout = TennisAbstractPlayerTimeout(ORIGIN)
    client = client_for(timeout, timeout, page(), timeout, timeout, page())
    result = update_daily(
        store_path=tmp_path / "tennis_abstract.sqlite3",
        players=players,
        client=client,
        clock=lambda: START,
    )
    assert result["status"] == "partial" and result["failure"] is None
    assert result["refreshed"] == 2 and len(result["player_timeouts"]) == 4
    assert client.get.call_count == 6

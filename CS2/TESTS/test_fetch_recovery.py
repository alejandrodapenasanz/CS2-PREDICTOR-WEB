"""Offline maintenance/menu regressions. Never connect to production or HLTV."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from unittest.mock import Mock

import pytest

from BBDD import build_db, ingest
from BBDD.fetch_recovery import persist_recovery, plan_recovery, protect_history
from BBDD.recovery_references import verified_reference
from PIPELINE import fetch_recovery_worker as worker
from PIPELINE.operation_lock import operation_lock

REPO = Path(__file__).resolve().parents[2]
NOW = "2026-09-27T12:00:00Z"


@pytest.fixture
def db(tmp_path):
    conn = build_db.connect_live_db(tmp_path / "fixture.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def issue(conn, key, status="partial", captured="2026-09-26T10:00:00Z", kind="match_analytics"):
    ingest.upsert_fetch_state(conn, kind, key, status, fetched_at=captured)
    conn.commit()


def record(key):
    return {"id": key, "link": f"/matches/{key}/alpha-vs-beta", "date": "2026-09-28"}


def test_plan_respects_cooldown_no_sample_and_budget(db):
    issue(db, "1")
    issue(db, "2", "blocked", "2026-09-27T11:50:00Z")
    issue(db, "3", "not_found")
    issue(db, "4")
    issue(db, "5")
    plan = plan_recovery(db, {k: record(k) for k in ("1", "2", "3", "4")}, now=NOW, limit=1)
    assert [r["entity_key"] for r in plan["selected"]] == ["1"]
    assert {r["deferred_reason"] for r in plan["deferred"]} == {"limite", "cooldown", "sin_url_verificable"}
    assert plan["issue_count"] == 4 and plan["no_sample_count"] == 1


def test_false_ok_detail_is_detected_without_mutating_during_audit(db, tmp_path):
    issue(db, "1", "ok", kind="match_detail")
    payload = {**record("1"), "captured_at": "2026-09-26T10:00:00Z", "detail": {"source": "upcoming_row_fallback"}}
    build_db.insert_raw_snapshot(
        db.cursor(),
        kind="match_snapshot",
        run_id="old",
        path=tmp_path / "one.json",
        payload=payload,
        captured_at=payload["captured_at"],
        hltv_match_id="1",
    )
    db.commit()
    plan = plan_recovery(db, {}, now=NOW, limit=10)
    assert len(plan["corrections"]) == 1 and plan["selected"][0]["last_status"] == "partial"
    assert db.execute("SELECT last_status FROM fetch_state").fetchone()[0] == "ok"


@pytest.mark.parametrize("error,expected", [("403 Cloudflare", "blocked"), ("timeout", "error"), (None, "partial")])
def test_ingest_does_not_label_fallback_as_ok(db, tmp_path, error, expected):
    save(
        tmp_path / "match_snapshots/1.json",
        {"id": "1", "captured_at": NOW, "html_error": error, "detail": {"source": "upcoming_row_fallback"}},
    )
    ingest.update_fetch_state_from_run(db, tmp_path)
    assert db.execute("SELECT last_status FROM fetch_state").fetchone()[0] == expected


def test_real_ingest_is_additive_idempotent_and_retains_capture_time(db, tmp_path):
    issue(db, "7", kind="player_stats")
    # The full production schema is used, including the protected table.
    before = [tuple(r) for r in db.execute("SELECT * FROM prediction_ledger")]
    player = {
        "id": "7",
        "name": "Player",
        "maps": 10,
        "captured_at": NOW,
        "time_filter": "past3months",
        "stats": {k: "1.1" for k in ingest.PLAYER_CORE_STATS},
    }
    save(
        tmp_path / "player_compare_stats_7.json", {"year": 2026, "captured_at": NOW, "results": [{"players": [player]}]}
    )
    save(
        tmp_path / "results.json",
        {
            "results": [
                {
                    "entity_type": "player_stats",
                    "entity_key": "7",
                    "attempted": True,
                    "captured_at": NOW,
                    "status": "ok",
                }
            ]
        },
    )
    persist_recovery(db, {"corrections": []}, tmp_path)
    first = db.execute("SELECT COUNT(*) FROM player_stat_snapshots").fetchone()[0]
    persist_recovery(db, {"corrections": []}, tmp_path)
    assert db.execute("SELECT COUNT(*) FROM player_stat_snapshots").fetchone()[0] == first == 1
    assert db.execute("SELECT captured_at_utc FROM player_stat_snapshots").fetchone()[0] == NOW
    assert db.execute("SELECT last_status FROM fetch_state").fetchone()[0] == "ok"
    assert [tuple(r) for r in db.execute("SELECT * FROM prediction_ledger")] == before
    # Availability is current, never the date of an old match or old manifest.
    assert (
        db.execute("SELECT COUNT(*) FROM player_stat_snapshots WHERE captured_at_utc < '2026-09-26'").fetchone()[0] == 0
    )


def test_older_report_does_not_replace_newer_fetch_state(db, tmp_path):
    issue(db, "1", "blocked", captured=NOW)
    save(
        tmp_path / "results.json",
        {
            "results": [
                {
                    "entity_type": "match_analytics",
                    "entity_key": "1",
                    "attempted": True,
                    "captured_at": "2026-09-26T00:00:00Z",
                    "status": "ok",
                }
            ]
        },
    )
    persist_recovery(db, {"corrections": []}, tmp_path)
    assert db.execute("SELECT last_status FROM fetch_state").fetchone()[0] == "blocked"


def test_sql_guard_denies_ledger_writes_and_destructive_operations():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE prediction_ledger(id INTEGER)")
    conn.execute("INSERT INTO prediction_ledger VALUES(1)")
    conn.commit()
    conn.set_authorizer(protect_history)
    for sql in (
        "INSERT INTO prediction_ledger VALUES(2)",
        "UPDATE prediction_ledger SET id=2",
        "DELETE FROM prediction_ledger",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql)
    assert conn.execute("SELECT id FROM prediction_ledger").fetchone() == (1,)
    conn.close()


def test_worker_stops_after_waf_and_preserves_pending(tmp_path, monkeypatch):
    acquire = Mock(side_effect=RuntimeError("403 Cloudflare challenge"))
    monkeypatch.setattr(worker, "acquire", acquire)
    plan = {"selected": [{"entity_type": "match_detail", "entity_key": str(n)} for n in range(3)]}
    report = worker.run_plan(plan, tmp_path)
    assert acquire.call_count == 1 and len(report["results"]) == 1
    assert report["results"][0]["status"] == "blocked"
    assert report["stopped_reason"] == "blocked"
    assert len(report["unattempted"]) == 2
    assert (tmp_path / "results.json").exists()


def test_worker_uses_real_parser_with_fixture_and_current_availability(tmp_path, monkeypatch):
    from PIPELINE import start

    monkeypatch.setattr(start, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(start, "now_utc", lambda: NOW)
    fetch = Mock(
        return_value='<html><body><div class="stats-row"><span>Maps played</span><span>0</span></div></body></html>'
    )
    monkeypatch.setattr(start, "fetch_html", fetch)
    status, _ = worker.acquire(
        {
            "entity_type": "player_stats",
            "entity_key": "7",
            "url": "/player/7/player",
            "player": {"id": "7", "name": "Player", "slug": "player", "link": "/player/7/player"},
        },
        tmp_path / "run",
    )
    assert status == "not_found"
    payload = json.loads((tmp_path / "run/player_compare_stats_7.json").read_text())
    assert payload["results"][0]["players"][0]["captured_at"] == NOW
    assert fetch.call_args.kwargs["max_attempts"] == 2
    with pytest.raises(ValueError):
        worker.safe_url("https://example.com/matches/1/test")


def test_exclusive_lock_released_after_exception(tmp_path):
    path = tmp_path / "operation.lock"
    with operation_lock(path):
        with pytest.raises(RuntimeError):
            with operation_lock(path):
                pass
    with operation_lock(path):
        pass


def load_root_module(path):
    spec = importlib.util.spec_from_file_location("menu_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_menu_option_one_invokes_requested_script_and_returns_to_menu(monkeypatch):
    menu = load_root_module(REPO / "main.py")
    answers = iter(["bad", "1", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    run = Mock(return_value=subprocess.CompletedProcess([], 2))
    monkeypatch.setattr(menu.subprocess, "run", run)
    assert menu.main() == 0
    assert run.call_args.args[0] == [sys.executable, str(REPO / "scripts/soluciona_errores.py")]


def test_wrapper_forwards_only_to_component_entrypoint(monkeypatch):
    wrapper = load_root_module(REPO / "scripts/soluciona_errores.py")
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    run = Mock(return_value=subprocess.CompletedProcess([], 2))
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    assert wrapper.main(["--solo-diagnostico", "--limite", "5"]) == 2
    command = run.call_args.args[0]
    assert "VAULT" in command[0] and command[1].endswith("repair_fetch_errors.py")
    assert command[-3:] == ["--limit", "5", "--diagnose"]


def test_start_ps1_and_maintenance_share_lock():
    text = (REPO / "CS2/start.ps1").read_text(encoding="utf-8")
    assert "PIPELINE\\operation.lock" in text and "[IO.FileShare]::None" in text


def archive(conn, tmp_path, kind, payload, captured=NOW):
    build_db.insert_raw_snapshot(
        conn.cursor(),
        kind=kind,
        run_id="fixture",
        path=tmp_path / f"{kind}.json",
        payload=payload,
        captured_at=captured,
    )
    conn.commit()


@pytest.mark.parametrize(
    "kind,payload",
    [
        (
            "match_assets",
            {
                "mapstats": [
                    {
                        "player_stats": [
                            {"player_hltv_id": "7", "player_name": "Seven", "player_link": "/stats/players/7/seven"}
                        ]
                    }
                ]
            },
        ),
        ("team_profile", {"profile": {"players": [{"id": "7", "name": "Seven", "link": "/player/7/seven"}]}}),
        (
            "match_snapshot",
            {"prematch_lineups": [{"players": [{"id": "7", "name": "Seven", "link": "/player/7/seven"}]}]},
        ),
        ("player_compare_stats", {"results": [{"players": [{"id": "7", "name": "Seven", "link": "/player/7/seven"}]}]}),
    ],
)
def test_first_failed_player_download_recovers_observed_reference(db, tmp_path, kind, payload):
    issue(db, "7", kind="player_stats")
    archive(db, tmp_path, kind, payload)
    plan = plan_recovery(db, {}, now=NOW, limit=0)
    assert len(plan["selected"]) == 1 and not plan["deferred"]
    player = plan["selected"][0]["player"]
    assert player["id"] == "7" and player["slug"] == "seven"
    assert player["provenance"]["kind"] == kind
    assert db.execute("SELECT COUNT(*) FROM player_stat_snapshots").fetchone()[0] == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/player/7/test",
        "/player/8/test",
        "/matches/7/test",
        "//example.com/player/7/test",
        "http://www.hltv.org/player/7/test",
    ],
)
def test_reference_rejects_foreign_host_or_wrong_identity(url):
    assert verified_reference(url, "player", "7") is None


def test_malformed_new_reference_does_not_hide_valid_old_reference(db, tmp_path):
    issue(db, "7", kind="player_stats")
    archive(db, tmp_path, "team_profile", {"id": "7", "link": "/player/7/real"}, "2026-09-25T00:00:00Z")
    archive(
        db,
        tmp_path,
        "match_assets",
        {"player_hltv_id": "7", "player_name": "fake", "player_link": "/stats/players/8/wrong"},
    )
    plan = plan_recovery(db, {}, now=NOW, limit=0)
    assert plan["selected"][0]["player"]["slug"] == "real"


def test_plan_covers_every_supported_entity_and_does_not_default_to_50(db, tmp_path):
    kinds = (
        "match_analytics",
        "match_assets",
        "match_detail",
        "player_stats",
        "ranking_hltv",
        "ranking_valve",
        "team_profile",
    )
    for kind in kinds:
        issue(db, "7", kind=kind)
    archive(
        db,
        tmp_path,
        "match_assets",
        {
            "player_hltv_id": "7",
            "player_link": "/stats/players/7/p",
            "team_hltv_id": "7",
            "team_link": "/stats/teams/7/t",
        },
    )
    master = {"7": record("7")}
    for key in range(10, 70):
        issue(db, str(key))
        master[str(key)] = record(str(key))
    plan = plan_recovery(db, master, now=NOW, limit=0)
    assert len(plan["selected"]) == 67
    assert not plan["deferred"]
    assert set(plan["by_entity"]) == set(kinds)


def test_budget_does_not_mark_unattempted_entities_as_failed(db, tmp_path, monkeypatch):
    from PIPELINE import start

    issue(db, "7", kind="player_stats")
    original = tuple(db.execute("SELECT * FROM fetch_state").fetchone())
    monkeypatch.setattr(worker, "acquire", Mock(side_effect=start.FetchBudgetExceeded("budget")))
    plan = {"corrections": [], "selected": [{"entity_type": "player_stats", "entity_key": "7"}]}
    report = worker.run_plan(plan, tmp_path)
    assert not report["results"][0]["attempted"]
    assert len(report["unattempted"]) == 1
    persist_recovery(db, plan, tmp_path)
    assert tuple(db.execute("SELECT * FROM fetch_state").fetchone()) == original


def test_individual_parse_failure_does_not_skip_other_issues(tmp_path, monkeypatch):
    acquire = Mock(side_effect=[ValueError("malformed"), ("ok", "recovered")])
    monkeypatch.setattr(worker, "acquire", acquire)
    report = worker.run_plan(
        {"selected": [{"entity_type": "player_stats", "entity_key": str(n)} for n in range(2)]}, tmp_path
    )
    assert [r["status"] for r in report["results"]] == ["error", "ok"]
    assert report["stopped_reason"] is None


def test_real_maintenance_entrypoint_recovers_first_player_and_publishes(tmp_path, monkeypatch):
    """Offline E2E: real plan -> parser -> guarded ingest -> web dispatch."""
    from PIPELINE import repair_fetch_errors as repair, start

    state = tmp_path / "state"
    database = state / "BBDD/cs2.db"
    database.parent.mkdir(parents=True)
    conn = build_db.connect_live_db(database)
    issue(conn, "7", kind="player_stats", captured="2020-01-01T00:00:00Z")
    archive(conn, tmp_path, "match_assets", {"player_hltv_id": "7", "player_link": "/stats/players/7/seven"})
    conn.close()
    monkeypatch.setattr(repair, "STATE", state)
    python = (
        state / "SCRAPER/hltv-scraper-api/.venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(start, "DATA_ROOT", state)
    monkeypatch.setattr(start, "now_utc", lambda: NOW)
    monkeypatch.setattr(repair, "utcnow", lambda: NOW)
    stats = {k: "1.1" for k in ingest.PLAYER_CORE_STATS}
    stats["Maps played"] = "20"
    html = (
        "<html><body>"
        + "".join(f'<div class="stats-row"><span>{k}</span><span>{v}</span></div>' for k, v in stats.items())
        + "</body></html>"
    )
    monkeypatch.setattr(start, "fetch_html", Mock(return_value=html))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "fetch_recovery_worker.py" in command[1]:
            plan_path = Path(command[-1])
            result = worker.run_plan(json.loads(plan_path.read_text(encoding="utf-8")), plan_path.parent)
            assert result["stopped_reason"] is None
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(repair.subprocess, "run", run)
    assert repair.execute(diagnose=False, limit=0) == 0
    summary = json.loads((state / "PIPELINE/maintenance/latest.json").read_text(encoding="utf-8"))
    assert summary["statuses"] == {"ok": 1} and summary["remaining"] == 0
    assert summary["attempted"] == 1 and not summary["pending"]
    assert calls[-1][1].endswith("build_web.py")
    conn = sqlite3.connect(database)
    assert conn.execute("SELECT captured_at_utc FROM player_stat_snapshots").fetchone()[0] == NOW
    assert conn.execute("SELECT COUNT(*) FROM prediction_ledger").fetchone()[0] == 0
    conn.close()

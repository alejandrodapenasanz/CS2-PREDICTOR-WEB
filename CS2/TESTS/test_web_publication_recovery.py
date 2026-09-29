"""Offline regressions for failed acquisition vs a genuinely empty CS2 agenda."""

from __future__ import annotations

import importlib.util
import json
import hashlib
from pathlib import Path
import sys

import pytest


def web_module():
    """Load the shared presentation adapter without production I/O."""

    path = Path(__file__).resolve().parents[2] / "WEB" / "build_web.py"
    spec = importlib.util.spec_from_file_location("web_publication_recovery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_run(root: Path, name: str, matches: list[dict], *, failed: bool = False) -> Path:
    """Create public artifacts only; no database or Telegram state is involved."""

    run = root / "runs" / name
    run.mkdir(parents=True)
    manifest = {
        "run_id": name,
        "started_at": f"{name}T06:00:00Z",
        "finished_at": f"{name}T07:00:00Z",
        "steps": {"upcoming": {"count": len(matches)}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "predictions_enriched.json").write_text(json.dumps(matches), encoding="utf-8")
    if failed:
        (run / "logs").mkdir()
        (run / "logs" / "upcoming_error.log.json").write_text('{"logs":"timeout"}', encoding="utf-8")
    return run


def test_failed_zero_match_run_restores_previous_cards_without_rewriting_probabilities(tmp_path, monkeypatch):
    """No new Telegram opportunities never means deleting existing web predictions."""

    web = web_module()
    upcoming = {"id": "future", "date": "2099-01-01", "prediction": {"decision_prob_team1": 0.75}}
    past = {"id": "past", "date": "2000-01-01"}
    completed = {"id": "done", "date": "2099-01-01", "status": "completed"}
    prior = fixture_run(tmp_path, "2026-09-24", [upcoming, past, completed])
    broken = fixture_run(tmp_path, "2026-09-25", [], failed=True)
    original = (prior / "predictions_enriched.json").read_bytes()
    monkeypatch.setattr(web, "configure_sport_root", lambda root: None)
    monkeypatch.setattr(web, "DAILY_ROOT", tmp_path)
    monkeypatch.setattr(web, "WEB_ROOT", tmp_path / "web")
    monkeypatch.setattr(web, "model_payload", lambda: {})
    monkeypatch.setattr(web, "database_payload", lambda: {})
    monkeypatch.setattr(web, "pending_cleanup_payload", lambda: {})
    monkeypatch.setattr(web, "tennis_payload", lambda: {"matches": [{"id": "tennis-current"}]})
    monkeypatch.setattr(sys, "argv", ["build_web.py", "--run-dir", str(broken)])

    assert web.main() == 0
    javascript = (web.WEB_ROOT / "data.js").read_text(encoding="utf-8")
    payload = json.loads(javascript.split("=", 1)[1].strip().removesuffix(";"))
    assert payload["matches"] == [upcoming]
    assert payload["tennis"]["matches"] == [{"id": "tennis-current"}]
    assert payload["publication"]["status"] == "fallback"
    assert payload["runDir"] == str(prior)
    assert payload["webFilter"]["sourceMatches"] == 3
    source = payload["freshness"]["components"]["CS2"]["sources"][0]
    assert source["last_success_at_utc"] == "2026-09-24T07:00:00Z"
    assert source["last_attempt_status"] == "failed"
    assert source["fallback_used"] is True
    assert (prior / "predictions_enriched.json").read_bytes() == original


def test_valid_empty_agenda_does_not_resurrect_old_matches(tmp_path):
    """Only failed/incomplete artifacts fall back, not successful empty agendas."""

    web = web_module()
    fixture_run(tmp_path, "2026-09-24", [{"id": "old"}])
    empty = fixture_run(tmp_path, "2026-09-25", [])
    selected, matches, publication = web.select_cs2_publication(empty)
    assert selected == empty
    assert matches == []
    assert publication["status"] == "current"


@pytest.mark.parametrize("invalid", ["missing", "truncated", "incomplete", "empty_predictions"])
def test_incomplete_publication_uses_only_prior_completed_run(tmp_path, invalid):
    """Partial writes, missing files and uncompleted runs cannot replace working cards."""

    web = web_module()
    prior = fixture_run(tmp_path, "2026-09-24", [{"id": "good"}])
    broken = fixture_run(tmp_path, "2026-09-25", [{"id": "new"}])
    predictions = broken / "predictions_enriched.json"
    if invalid == "missing":
        predictions.unlink()
    elif invalid == "truncated":
        predictions.write_text("[", encoding="utf-8")
    elif invalid == "empty_predictions":
        predictions.write_text("[]", encoding="utf-8")
    else:
        (broken / "manifest.json").write_text("{}", encoding="utf-8")
    fixture_run(tmp_path, "2026-09-26", [{"id": "future-run"}])
    selected, matches, _ = web.select_cs2_publication(broken)
    assert selected == prior
    assert matches == [{"id": "good"}]


def test_no_good_run_does_not_replace_existing_dashboard(tmp_path, monkeypatch):
    """Fail before publication instead of manufacturing a blank multi-sport dashboard."""

    web = web_module()
    broken = fixture_run(tmp_path, "2026-09-25", [], failed=True)
    output = tmp_path / "data.js"
    output.write_text("previous dashboard", encoding="utf-8")
    monkeypatch.setattr(web, "WEB_ROOT", tmp_path)
    monkeypatch.setattr(web, "configure_sport_root", lambda root: None)
    monkeypatch.setattr(sys, "argv", ["build_web.py", "--run-dir", str(broken)])
    with pytest.raises(RuntimeError, match="Sin cartelera CS2 válida"):
        web.main()
    assert output.read_text(encoding="utf-8") == "previous dashboard"


def test_atomic_replace_failure_preserves_previous_web_and_cleans_temporary_file(tmp_path, monkeypatch):
    """An interrupted publication never leaves half a JavaScript file."""

    web = web_module()
    output = tmp_path / "data.js"
    output.write_text("previous dashboard", encoding="utf-8")

    def denied(*args):
        """Simulate an OS failure at the atomic replacement boundary."""

        raise PermissionError("fixture")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        web.write_dashboard_payload({"matches": []}, output)
    assert output.read_text(encoding="utf-8") == "previous dashboard"
    assert list(tmp_path.iterdir()) == [output]


def test_legacy_two_of_seventy_one_is_not_considered_complete(tmp_path):
    """The exact production failure must trigger fallback, not a successful two-row update."""
    web = web_module()
    good = fixture_run(tmp_path, "2026-09-26", [{"id": "good"}])
    broken = fixture_run(tmp_path, "2026-09-27", [{"id": "1"}, {"id": "2"}])
    (broken / "predictions_skipped_unpublishable.json").write_text(
        json.dumps({"skipped": {"participants_unconfirmed": 68, "already_started": 1}})
    )
    selected, _, publication = web.select_cs2_publication(broken)
    assert selected == good and publication["status"] == "fallback"


@pytest.mark.parametrize("damage", [None, "missing_row", "hash", "duplicate", "missing_unavailable"])
def test_coverage_contract_keeps_unavailable_visible_and_rejects_truncation(tmp_path, damage):
    from PIPELINE.agenda_contract import coverage_report, unavailable_entry

    web = web_module()
    predictions = [{"id": "1", "prediction": {"model_prob_team1": 0.70}}]
    run = fixture_run(tmp_path, "2026-09-27", predictions)
    unavailable = [unavailable_entry({"id": "2", "date": "2099-01-01"}, "participants_unconfirmed")]
    coverage = coverage_report(
        [{"id": "1"}, {"id": "2"}, {"id": "3"}], predictions, unavailable, {"3": "already_started"}
    )
    unavailable_path = run / "agenda_unavailable.json"
    unavailable_path.write_text(json.dumps(unavailable))
    coverage["prediction_sha256"] = hashlib.sha256((run / "predictions_enriched.json").read_bytes()).hexdigest()
    coverage["unavailable_sha256"] = hashlib.sha256(unavailable_path.read_bytes()).hexdigest()
    if damage == "missing_row":
        coverage["source_ids"].append("missing")
    elif damage == "hash":
        coverage["prediction_sha256"] = "wrong"
    elif damage == "duplicate":
        coverage["excluded"]["1"] = "already_started"
    elif damage == "missing_unavailable":
        unavailable_path.unlink()
    (run / "publication_coverage.json").write_text(json.dumps(coverage))
    if damage:
        with pytest.raises(ValueError):
            web.cs2_publication_data(run)
    else:
        assert web.cs2_publication_data(run) == predictions + unavailable


@pytest.mark.parametrize("expired", [False, True])
def test_zero_predictions_with_complete_accounting_is_a_valid_publication(tmp_path, expired):
    """All TBD or all started is valid, unlike an unexplained empty model output."""
    from PIPELINE.agenda_contract import coverage_report, unavailable_entry

    web = web_module()
    run = fixture_run(tmp_path, "2026-09-27", [])
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["steps"]["upcoming"]["count"] = 1
    manifest_path.write_text(json.dumps(manifest))
    unavailable = [] if expired else [unavailable_entry({"id": "1"}, "participants_unconfirmed")]
    excluded = {"1": "already_started"} if expired else {}
    coverage = coverage_report([{"id": "1"}], [], unavailable, excluded)
    unavailable_path = run / "agenda_unavailable.json"
    unavailable_path.write_text(json.dumps(unavailable))
    coverage["prediction_sha256"] = hashlib.sha256((run / "predictions_enriched.json").read_bytes()).hexdigest()
    coverage["unavailable_sha256"] = hashlib.sha256(unavailable_path.read_bytes()).hexdigest()
    (run / "publication_coverage.json").write_text(json.dumps(coverage))
    assert web.cs2_publication_data(run) == unavailable

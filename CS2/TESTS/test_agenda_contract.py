"""Regression for the 71 acquired / 2 displayed agenda of 27 September."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from PIPELINE.agenda_contract import (
    agenda_identities,
    coverage_report,
    load_identity_evidence,
    recover_snapshot_identity,
    unavailable_entry,
    AgendaCoverageError,
    validate_agenda_rows,
)
from PIPELINE.enrich_predictions import is_snapshot_publishable, parse_match_datetime, write_json
from PIPELINE.start import detail_from_upcoming_row, parse_upcoming_matches_fallback


HTML = """<html><body><div class="matches-list-section"><div class="matches-list-headline">2026-09-27</div>
<div class="match-zone-wrapper"><div class="match-wrapper" data-match-id="42" team1="11" team2="22">
<a class="match-info" href="/matches/42/alpha-vs-beta"></a><div class="match-time">18:00</div>
<div class="team1"><div class="match-teamname">Alpha</div></div>
<div class="team2"><div class="match-teamname">Beta</div></div></div></div></div></body></html>"""


def snapshot() -> dict:
    """A real listing with names but IDs discarded by the old parser."""
    row = {
        "date": "2026-09-27",
        "hour": "18:00",
        "link": "/matches/42/alpha-vs-beta",
        "team1": {"name": "Alpha"},
        "team2": {"name": "Beta"},
    }
    return {
        "id": "42",
        "captured_at": "2026-09-27T08:00:00Z",
        **row,
        "upcoming_row": row,
        "detail": detail_from_upcoming_row(row),
        "status": "pending",
    }


def metadata() -> dict:
    """Original source availability, not the time of replay."""
    return {
        "captured_at": "2026-09-27T07:34:48Z",
        "sha256": hashlib.sha256(HTML.encode()).hexdigest(),
        "url": "https://www.hltv.org/matches",
        "source_file": "raw_html/upcoming_page/matches.html.gz",
    }


def test_ids_are_extracted_from_listing_without_detail_page():
    """Both acquisition paths preserve explicit team attributes."""
    rows = parse_upcoming_matches_fallback(HTML)
    assert rows[0]["team1"]["id"] == "11" and rows[0]["team2"]["id"] == "22"
    original = snapshot()
    before = json.dumps(original, sort_keys=True)
    fixed = recover_snapshot_identity(original, agenda_identities(HTML), metadata())
    assert is_snapshot_publishable(fixed, parse_match_datetime("2026-09-27", "10:00")) == (True, "")
    assert fixed["detail"]["match"]["team1"]["id"] == "11"
    assert json.dumps(original, sort_keys=True) == before  # No rewrite of the captured snapshot.
    assert fixed["data_quality"]["identity_recovered_from_agenda"]["captured_at"] == metadata()["captured_at"]


def test_partial_parser_or_unrecognized_html_cannot_be_published():
    rows = parse_upcoming_matches_fallback(HTML)
    validate_agenda_rows(HTML, rows)
    with pytest.raises(AgendaCoverageError, match="parcial"):
        validate_agenda_rows(HTML, [])
    with pytest.raises(AgendaCoverageError, match="estructura"):
        validate_agenda_rows("<html><body>Service unavailable</body></html>", [])


@pytest.mark.parametrize("change", ["future", "wrong_match", "wrong_name", "conflicting_id"])
def test_identity_recovery_is_exact_and_asof(change):
    """Never use a later observation, another fixture or a fuzzy name match."""
    row, meta = snapshot(), metadata()
    if change == "future":
        meta["captured_at"] = "2026-09-28T00:00:00Z"
    elif change == "wrong_match":
        row["id"] = "43"
    elif change == "wrong_name":
        row["detail"]["match"]["team1"]["name"] = "Different team"
    else:
        row["detail"]["match"]["team1"]["id"] = "999"
    fixed = recover_snapshot_identity(row, agenda_identities(HTML), meta)
    assert is_snapshot_publishable(fixed, parse_match_datetime("2026-09-27", "10:00"))[0] is False


def test_append_future_rows_does_not_change_recovered_identity():
    original = recover_snapshot_identity(snapshot(), agenda_identities(HTML), metadata())
    future = HTML.replace('data-match-id="42"', 'data-match-id="43"').replace('team1="11"', 'team1="99"')
    assert recover_snapshot_identity(snapshot(), agenda_identities(HTML + future), metadata()) == original


def test_conflicting_duplicate_identity_is_not_usable():
    assert "42" not in agenda_identities(HTML + HTML.replace('team1="11"', 'team1="99"'))


def test_archived_evidence_hash_is_verified(tmp_path):
    folder = tmp_path / "raw_html" / "upcoming_page"
    folder.mkdir(parents=True)
    path = folder / "matches.html.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(HTML)
    path.with_suffix(".gz.json").write_text(json.dumps(metadata()))
    assert load_identity_evidence(tmp_path)[0]["42"]["team2"]["id"] == "22"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(HTML + "corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_identity_evidence(tmp_path)


def test_every_acquired_match_is_accounted_for_without_inventing_probabilities():
    sources = [{"id": str(i)} for i in range(71)]
    predictions = sources[:2]
    assert not coverage_report(sources, predictions, [], {})["complete"]
    unavailable = [unavailable_entry(row, "participants_unconfirmed") for row in sources[2:]]
    assert coverage_report(sources, predictions, unavailable, {})["complete"]
    assert "model_prob_team1" not in unavailable[0]["prediction"]
    assert "favorite" not in unavailable[0]["prediction"]
    assert unavailable[0]["prediction"]["opportunity_eligible"] is False
    assert coverage_report([], [], [], {})["complete"]
    assert not coverage_report(sources, predictions + predictions, unavailable, {})["complete"]
    assert not coverage_report([{"id": "42"}], [], [], {"42": "participants_unconfirmed"})["complete"]


def test_expired_unresolved_matches_do_not_resurrect_on_dashboard():
    assert is_snapshot_publishable(snapshot(), parse_match_datetime("2026-09-28", "10:00")) == (
        False,
        "already_started",
    )


def test_unavailable_dashboard_rows_have_no_favorite_or_fake_fifty_percent():
    html = (Path(__file__).resolve().parents[2] / "WEB" / "index.html").read_text(encoding="utf-8")
    assert "if(!hasPrediction(m)) return" in html.split("function renderRows()", 1)[1]
    assert "if(!hasPrediction(m)) return" in html.split("function detailHTML(m)", 1)[1]
    assert "const predicted = list.filter(hasPrediction)" in html
    assert "M.filter(m=>hasPrediction(m)&&" in html


def test_failed_prediction_artifact_write_preserves_last_complete_file(tmp_path, monkeypatch):
    path = tmp_path / "predictions_enriched.json"
    original = '[{"id":"42"}]'
    path.write_text(original)

    def denied(*args):
        raise PermissionError("fixture")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        write_json(path, [{"id": "new"}])
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]

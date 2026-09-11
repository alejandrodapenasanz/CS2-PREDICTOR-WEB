"""Offline tests for canonical CS2 selection and Telegram HTML formatting."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from cs2_telegram.formatter import (
    TelegramMessageTooLongError,
    format_daily_messages,
    render_messages,
)
from cs2_telegram.models import DailyPredictions, PredictionPick
from cs2_telegram.report import render_daily_report, write_daily_report
from cs2_telegram.source import (
    PredictionSourceError,
    PredictionValidationError,
    load_daily_predictions,
    load_latest_daily_picks,
    load_upcoming_opportunities,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "predictions_enriched.json"
RUN_ID = "fixture-run"


def _fixture_rows() -> list[dict[str, object]]:
    """Return an independent copy of the network-free prediction fixture."""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_project(tmp_path: Path, rows: object) -> Path:
    """Create the canonical manifest/run layout below a temporary project root."""

    manifest_dir = tmp_path / "CS2" / "PIPELINE" / "master"
    run_dir = tmp_path / "CS2" / "PIPELINE" / "runs" / RUN_ID
    manifest_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"last_run_id": RUN_ID}),
        encoding="utf-8",
    )
    (run_dir / "predictions_enriched.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )
    return tmp_path


def test_loads_target_date_separates_best_and_orders_other_matches(tmp_path: Path) -> None:
    """Select only the requested date, one best pick, and time-ordered others."""

    root = _write_project(tmp_path, _fixture_rows())

    daily = load_daily_predictions(root, "2026-08-11")

    assert daily.source_run_id == RUN_ID
    assert daily.target_date == date(2026, 8, 11)
    assert daily.best is not None
    assert daily.best.match_id == "match-best"
    assert daily.best.hltv_url == ("https://www.hltv.org/matches/1002/northern-lights-vs-crimson-five")
    assert [pick.match_id for pick in daily.opportunities] == ["match-best"]
    assert [pick.match_id for pick in daily.others] == ["match-early", "match-late"]
    assert daily.others[1].hltv_url == ("https://www.hltv.org/matches/1001/alpha-vs-beta?event=7")
    assert daily.total == 3
    assert load_latest_daily_picks(root, date(2026, 8, 11)) == daily


def test_formats_exactly_two_safe_english_html_messages(tmp_path: Path) -> None:
    """Render appealing labels, escaped names, percentages, and stable ordering."""

    daily = load_daily_predictions(_write_project(tmp_path, _fixture_rows()), "2026-08-11")

    messages = format_daily_messages(daily)

    assert len(messages) == 2
    best_message, others_message = messages
    assert "<b>BEST OPPORTUNITIES</b>" in best_message
    assert "<b>Match:</b> Northern Lights vs Crimson Five" in best_message
    assert "<b>Winner:</b> Northern Lights" in best_message
    assert "<b>Win probability:</b> 73.46%" in best_message
    assert "Adjusted win probability &gt; 65%" in best_message
    assert "<b>Confidence:</b> HIGH" in best_message
    assert "Alpha &amp; Co" in others_message
    assert "Beta &lt;Academy&gt;" in others_message
    assert "<b>Confidence:</b> MEDIUM" in others_message
    assert "<b>Confidence:</b> LOW" in others_message
    assert "Alpha &amp; Co" not in best_message
    assert "Model estimates — not guarantees." in best_message
    assert "Early Birds" in others_message
    assert "Model estimates — not guarantees." in others_message
    assert render_messages(daily) == messages
    assert all(len(message) <= 4096 for message in messages)


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("decision_confidence", 0.49, "between 0.5 and 1"),
        ("decision_confidence", True, "must be numeric"),
        ("decision_favorite", "Unknown Team", "must equal team1.name or team2.name"),
    ],
)
def test_rejects_invalid_decision_fields(
    tmp_path: Path,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    """Reject unsafe decision values rather than publishing invented data."""

    rows = _fixture_rows()
    rows[0]["prediction"][field] = value  # type: ignore[index]
    root = _write_project(tmp_path, rows)

    with pytest.raises(PredictionValidationError, match=error_fragment):
        load_daily_predictions(root, "2026-08-11")


def test_rejects_fallback_model_prediction(tmp_path: Path) -> None:
    """Reject a target-day prediction explicitly produced by a fallback model."""

    rows = _fixture_rows()
    rows[0]["model_trace"]["is_fallback"] = True  # type: ignore[index]

    with pytest.raises(PredictionValidationError, match="fallback-model"):
        load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")


@pytest.mark.parametrize(
    "invalid_link",
    [
        "http://www.hltv.org/matches/1002/a-vs-b",
        "https://hltv.org/matches/1002/a-vs-b",
        "https://www.hltv.org.evil.example/matches/1002/a-vs-b",
        "https://user@www.hltv.org/matches/1002/a-vs-b",
        "https://www.hltv.org:443/matches/1002/a-vs-b",
        "https://www.hltv.org/matches/1002/a-vs-b#details",
        "https://www.hltv.org/events/1002/not-a-match",
        "/matches/1002/a vs b",
    ],
)
def test_rejects_noncanonical_or_unsafe_hltv_links(
    tmp_path: Path,
    invalid_link: str,
) -> None:
    """Accept only credential-free canonical HTTPS match URLs on the HLTV host."""

    rows = _fixture_rows()
    rows[1]["link"] = invalid_link

    with pytest.raises(PredictionValidationError, match="HLTV"):
        load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")


def test_accepts_multiple_qualified_opportunities(tmp_path: Path) -> None:
    """Include every target-day row qualified for the opportunity tab."""

    rows = _fixture_rows()
    rows[0]["prediction"]["is_best_opportunity"] = True  # type: ignore[index]
    rows[0]["prediction"]["decision_confidence"] = 0.68  # type: ignore[index]

    daily = load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")
    assert {pick.match_id for pick in daily.opportunities} == {"match-best", "match-late"}


def test_rejects_duplicate_match_ids(tmp_path: Path) -> None:
    """Reject duplicate target-day matches before they can be published twice."""

    rows = _fixture_rows()
    rows[2]["id"] = rows[0]["id"]

    with pytest.raises(PredictionValidationError, match="duplicate match id"):
        load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("opportunity_eligible", False, "opportunity_eligible=true"),
        ("opportunity_rank_day", 2, "opportunity_rank_day=1"),
    ],
)
def test_rejects_inconsistent_best_opportunity_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    """Reject a best flag that conflicts with optional upstream eligibility metadata."""

    rows = _fixture_rows()
    rows[1]["prediction"][field] = value  # type: ignore[index]

    with pytest.raises(PredictionValidationError, match=error_fragment):
        load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")


def test_empty_date_has_two_honest_status_messages(tmp_path: Path) -> None:
    """Describe empty best/other selections honestly instead of fabricating picks."""

    daily = load_daily_predictions(_write_project(tmp_path, _fixture_rows()), "2026-08-20")

    best_message, others_message = format_daily_messages(daily)

    assert daily.total == 0
    assert "No qualifying best opportunity is available today." in best_message
    assert "No other validated predictions are available today." in others_message


def test_missing_manifest_is_reported_clearly(tmp_path: Path) -> None:
    """Expose a missing canonical manifest as an actionable source error."""

    with pytest.raises(PredictionSourceError, match="manifest not found"):
        load_daily_predictions(tmp_path, "2026-08-11")


def test_rejects_message_over_telegram_limit() -> None:
    """Raise clearly rather than silently truncating an oversized Telegram message."""

    long_name = "A" * 4100
    pick = PredictionPick(
        match_id="oversized",
        match_date=date(2026, 8, 11),
        hltv_url="https://www.hltv.org/matches/9999/oversized-match",
        hour="10:00",
        team1=long_name,
        team2="Short Team",
        winner=long_name,
        win_probability=0.75,
        is_best_opportunity=True,
    )
    daily = DailyPredictions(
        target_date=date(2026, 8, 11),
        source_run_id=RUN_ID,
        opportunities=(pick,),
        others=(),
    )

    with pytest.raises(TelegramMessageTooLongError, match="4096"):
        format_daily_messages(daily)


def test_daily_report_contains_all_upcoming_opportunities_and_plain_cta(tmp_path: Path) -> None:
    """Render current and future qualified opportunities in one report."""

    root = _write_project(tmp_path / "project", _fixture_rows())
    _, picks = load_upcoming_opportunities(root, "2026-08-11")
    report = render_daily_report(picks)

    assert "Date: 2026-08-11" in report
    assert "URL: https://www.hltv.org/matches/1002/northern-lights-vs-crimson-five" in report
    assert "Winner: Northern Lights" in report
    assert "Match: Northern Lights vs Crimson Five" in report
    assert "Win probability: 73.46%" in report
    assert (
        "This prediction was calculated using my predictive Machine Learning and "
        "Artificial Intelligence model as part of my Data Science PhD thesis."
    ) in report
    assert "Machine Learning" in report
    assert "Artificial Intelligence" in report
    assert ("See all model predictions for free on Telegram:\n@cs2DailyPicks\n") in report
    assert "https://t.me/cs2DailyPicks" not in report
    assert report.index("URL:") < report.index("Winner:") < report.index("Match:") < report.index("Win probability:")
    assert "Alpha & Co" not in report
    assert "Confidence: HIGH" in report
    assert "Confidence: LOW" in report
    assert "Date: 2026-08-12" in report
    assert "Tomorrow One vs Tomorrow Two" in report
    assert "🔥" not in report

    report_root = tmp_path / "telegram"
    report_root.mkdir()
    destination = report_root / "daily_report.txt"
    destination.write_text("obsolete report\n", encoding="utf-8")

    written_path = write_daily_report(picks, report_root)

    assert written_path == destination.resolve()
    assert destination.read_text(encoding="utf-8") == report
    assert list(report_root.glob(".daily_report.*.tmp")) == []


def test_daily_report_states_absence_when_no_new_opportunities() -> None:
    """Report an empty incremental selection explicitly."""

    report = render_daily_report(())

    assert "Status: No new qualifying opportunities are available." in report
    assert "Winner:" not in report


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.60, False), (0.649999, False), (0.65, False), (0.650001, True), (0.75, True)],
)
def test_opportunity_threshold_is_strict_and_shared_with_report(
    tmp_path: Path, probability: float, expected: bool
) -> None:
    """Filter exported adjusted probability before rounding, for any date."""

    rows = _fixture_rows()
    for index in (0, 3):
        rows[index]["prediction"]["decision_confidence"] = probability  # type: ignore[index]
        rows[index]["prediction"]["confidence"] = 0.99  # type: ignore[index]
    root = _write_project(tmp_path, rows)
    daily = load_daily_predictions(root, "2026-08-11")
    _, upcoming = load_upcoming_opportunities(root, "2026-08-11")
    assert ("match-late" in {pick.match_id for pick in daily.opportunities}) is expected
    assert ("match-late" in {pick.match_id for pick in daily.others}) is not expected
    assert ("different-date" in {pick.match_id for pick in upcoming}) is expected
    assert daily.total == 3


def test_high_probability_does_not_override_upstream_ineligibility(tmp_path: Path) -> None:
    """Retain CS2 roster/data gates even for an apparently strong favourite."""

    rows = _fixture_rows()
    rows[0]["prediction"]["decision_confidence"] = 0.95  # type: ignore[index]
    rows[0]["prediction"]["opportunity_eligible"] = False  # type: ignore[index]
    root = _write_project(tmp_path, rows)
    daily = load_daily_predictions(root, "2026-08-11")
    _, upcoming = load_upcoming_opportunities(root, "2026-08-11")
    assert "match-late" in {pick.match_id for pick in daily.others}
    assert "match-late" not in {pick.match_id for pick in upcoming}


@pytest.mark.parametrize("level", ["high", "medium", "low", None])
def test_confidence_uses_exported_level_not_win_probability(tmp_path: Path, level: str | None) -> None:
    """Show honest uncertainty on both posts and reports, including legacy nulls."""

    rows = _fixture_rows()
    for row in rows:
        row["prediction"]["estimate_confidence_level"] = level  # type: ignore[index]
    root = _write_project(tmp_path, rows)
    daily = load_daily_predictions(root, "2026-08-11")
    best, others = format_daily_messages(daily)
    expected = level.upper() if level else "NOT AVAILABLE"
    assert f"<b>Confidence:</b> {expected}" in best
    assert others.count(f"<b>Confidence:</b> {expected}") == 2
    assert f"Confidence: {expected}" in render_daily_report(daily.opportunities)


def test_legacy_missing_confidence_is_explicit(tmp_path: Path) -> None:
    """Do not invent a confidence level for a legacy prediction without it."""

    rows = _fixture_rows()
    del rows[1]["prediction"]["estimate_confidence_level"]  # type: ignore[attr-defined]
    daily = load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")
    assert "<b>Confidence:</b> NOT AVAILABLE" in format_daily_messages(daily)[0]


@pytest.mark.parametrize("level", ["", "certain", "<b>high</b>", True, 0.9, {}])
def test_invalid_confidence_fails_explicitly(tmp_path: Path, level: object) -> None:
    """Reject malformed exported confidence instead of guessing or injecting HTML."""

    rows = _fixture_rows()
    rows[1]["prediction"]["estimate_confidence_level"] = level  # type: ignore[index]
    with pytest.raises(PredictionValidationError, match="estimate_confidence_level"):
        load_daily_predictions(_write_project(tmp_path, rows), "2026-08-11")


def test_future_rows_do_not_change_current_day_selection_or_confidence(tmp_path: Path) -> None:
    """Appending a future export never alters today's picks, levels or messages."""

    rows = _fixture_rows()
    baseline = load_daily_predictions(_write_project(tmp_path / "before", rows[:3]), "2026-08-11")
    extended = load_daily_predictions(_write_project(tmp_path / "after", rows), "2026-08-11")
    assert baseline == extended
    assert format_daily_messages(baseline) == format_daily_messages(extended)

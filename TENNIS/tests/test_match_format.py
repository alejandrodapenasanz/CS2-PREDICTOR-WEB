"""Serving metadata, original raw recovery, and strict daily cutoff regressions."""

from datetime import UTC, date, datetime, timedelta
import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.daily_pipeline.pipeline import _attach_feature_agenda, predict_mapped_matches
from src.daily_pipeline.format_report import match_format_report, match_format_summary
from src.features.match_format import normalize_round, resolve_match_format
from src.tennisratio.agenda_context import recover_agenda_context
from src.tennisratio.parser import parse_agenda_html
from tests.test_daily_pipeline import _FakeModel, _feature_context, _mapped_frame
from tests.test_tennisratio_service import _FixtureSession, _write_sackmann_masters
from src.tennisratio import load_agenda_as_of, refresh_tennisratio
from src.tennisratio.agenda_context import restore_agenda_context


D = date(2026, 8, 22)
CAPTURE = datetime(2026, 8, 21, 8, tzinfo=UTC)
FIXTURE = Path(__file__).parent / "fixtures/tennisratio/atp-matches.html"


def _html(round_name: str = "Round of 64") -> bytes:
    """Adapt the inspected fixture with the real header/round markup."""

    html = FIXTURE.read_text(encoding="utf-8").replace(
        'data-level="ATP"', 'data-level="Grand Slams"'
    )
    html = html.replace(
        '<span class="tournament-title">Test Open</span>',
        f'<span class="tournament-title">US Open</span><span class="tournament-meta">{round_name} · 2 matches</span>',
    )
    return html.replace("12:30</span>", f"12:30</span><span>{round_name}</span>").encode()


def _parse(raw: bytes, captured: datetime = CAPTURE) -> pd.DataFrame:
    """Parse only schedule evidence, without a result source or network."""

    return parse_agenda_html(
        raw,
        gender="M",
        source_url="https://www.tennisratio.com/atp-matches.html",
        retrieved_at_utc=captured,
    )


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("Final", "F"),
        ("Semi-finals", "SF"),
        ("Quarter-final", "QF"),
        ("Round of 128", "R128"),
        ("Round of 64", "R64"),
        ("Round of 32", "R32"),
        ("Round of 16", "R16"),
        ("Qualifying round 1", "Q1"),
        ("Round robin", "RR"),
        ("First round", None),
        ("Qualification", None),
        ("Round of 48", None),
    ],
)
def test_round_maps_to_original_sackmann_codes(raw, code) -> None:
    """Preserve Sackmann categories, never guess a draw size from an ordinal."""

    assert normalize_round(raw) == code
    if code:
        assert normalize_round(code) == code


@pytest.mark.parametrize(
    ("gender", "level", "round_raw", "expected"),
    [
        ("M", "Grand Slams", "Round of 64", 5),
        ("F", "Grand Slams", "Final", 3),
        ("M", "ATP", "Final", 3),
        ("F", "WTA", "Round of 32", 3),
        ("M", "Challengers", "Round of 16", 3),
        ("F", "Futures", "Round of 32", 3),
        ("M", "Grand Slams", "Qualifying round 3", None),
        ("M", "Grand Slams", None, None),
        ("M", None, "Final", None),
    ],
)
def test_best_of_requires_supported_level_gender_and_draw(
    gender, level, round_raw, expected
) -> None:
    """A male Slam is five only with main-draw evidence; unknown is not three."""

    result = resolve_match_format(
        match_date=D,
        captured_at_utc=CAPTURE,
        gender=gender,
        source_family="tennisratio",
        tournament_level=level,
        tournament="US Open",
        round_raw=round_raw,
        round_evidence="match",
    )
    assert result.best_of == expected
    assert result.best_of_reason


@pytest.mark.parametrize(
    "capture", [None, datetime(2026, 8, 22, tzinfo=UTC), datetime(2026, 8, 23, tzinfo=UTC)]
)
def test_same_day_future_or_unproved_capture_is_not_a_feature(capture) -> None:
    """Even a known draw format cannot bypass the civil-day cutoff."""

    result = resolve_match_format(
        match_date=D,
        captured_at_utc=capture,
        gender="M",
        source_family="tennisratio",
        tournament_level="Grand Slams",
        tournament="US Open",
        round_raw="R64",
        round_evidence="match",
    )
    assert result.best_of is None and result.round is None
    assert result.best_of_reason == result.round_reason == "no_pre_date_evidence"


def test_source_preserves_slam_label_and_round_for_both_layouts() -> None:
    """Compact rows inherit only their exact unconflicted enclosing group."""

    frame = _parse(_html())
    assert frame["tournament_level_source"].tolist() == ["Grand Slams"] * 2
    assert frame["round_source"].tolist() == ["Round of 64"] * 2
    assert frame["round_evidence"].tolist() == ["match_and_group", "tournament_group"]


def test_conflicting_group_does_not_propagate_to_compact_rows() -> None:
    """A conflicting featured round invalidates the group's propagation."""

    frame = _parse(_html().replace(b"<span>Round of 64</span>", b"<span>Final</span>"))
    assert frame["round_source"].isna().all()
    assert frame["round_evidence"].eq("conflict").all()


def test_recovery_is_read_only_hash_checked_and_keeps_original_capture(tmp_path) -> None:
    """A retained original snapshot can recover dropped fields, not new facts."""

    raw = _html()
    frame = _parse(raw)
    old = frame.drop(columns=["tournament_level_source", "round_source", "round_evidence"]).to_dict(
        orient="records"
    )
    sha = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "raw.gz"
    path.write_bytes(gzip.compress(raw))
    recovered = recover_agenda_context(old, {sha: path})
    assert recovered[0]["round_source"] == "Round of 64"
    assert recovered[0]["retrieved_at_utc"] == old[0]["retrieved_at_utc"]
    assert "round_source" not in old[0]
    path.write_bytes(gzip.compress(raw + b"tamper"))
    assert "round_source" not in recover_agenda_context(old, {sha: path})[0]


def test_append_future_does_not_change_frozen_context() -> None:
    """Changing a later presentation snapshot never replaces pre-D context."""

    frozen = _parse(_html())
    future = _parse(_html("Final"), CAPTURE + timedelta(days=2))
    before = _attach_feature_agenda(frozen, frozen)
    after = _attach_feature_agenda(future, frozen)
    columns = [c for c in before if c.startswith("feature_")]
    pd.testing.assert_frame_equal(before[columns], after[columns])


def test_daily_pipeline_really_passes_metadata_to_model_and_removes_flags() -> None:
    """Regression for the original hard-coded None serving bug."""

    frame = _mapped_frame().iloc[[0]].copy()
    frame["source_match_id"] = "tennisratio:M:1001"
    frame["feature_tournament_level_source"] = "Grand Slams"
    frame["feature_tournament"] = "US Open"
    frame["feature_round_source"] = "Round of 64"
    frame["feature_round_evidence"] = "match"
    model = _FakeModel()
    original_predict = model.predict

    def predict(inputs, **kwargs):
        """Inspect actual serving inputs before delegating deterministic scores."""
        assert inputs["best_of"].eq(5).all()
        assert inputs["round"].eq("R64").all()
        return original_predict(inputs, **kwargs)

    model.predict = predict
    result = predict_mapped_matches(
        frame,
        match_date=date(2024, 2, 1),
        prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
        models={"M": model},
        feature_context=_feature_context(),
        excluded_player_keys=frozenset(),
    )
    predicted = result.loc[result["prediction_status"].eq("predicted")]
    assert len(predicted) > 0
    assert predicted["best_of"].eq(5).all()
    assert predicted["round"].eq("R64").all()
    assert not predicted["confidence_flags"].str.contains("best_of_missing|round_missing").any()
    report = match_format_report(result)
    assert report["best_of"]["populated_pct"] == 100.0
    assert report["round"]["missing_flag_rows"] == 0
    assert len(match_format_summary(result)) == 3


def test_missing_metadata_keeps_nulls_and_existing_model_flags() -> None:
    """No evidence leaves actual model inputs null, with explicit output flags."""

    frame = _mapped_frame().iloc[[0]].copy()
    frame["source_match_id"] = "tennisratio:M:1001"
    result = predict_mapped_matches(
        frame,
        match_date=date(2024, 2, 1),
        prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
        models={"M": _FakeModel()},
        feature_context=_feature_context(),
        excluded_player_keys=frozenset(),
    )
    report = match_format_report(result)
    for field in ("best_of", "round"):
        assert result[field].isna().all()
        assert report[field]["missing_flag_rows"] == report[field]["missing"] == 1
    empty = match_format_report(result.iloc[0:0])
    assert empty["best_of"]["populated_pct"] is None


def test_qualifying_title_cannot_turn_a_qualifying_final_into_slam_main_final() -> None:
    """A conflicting qualifying title cannot produce the five-set main final."""

    result = resolve_match_format(
        match_date=D,
        captured_at_utc=CAPTURE,
        gender="M",
        source_family="tennisratio",
        tournament_level="Grand Slams",
        tournament="US Open Qualies",
        round_raw="Final",
        round_evidence="match",
    )
    assert result.best_of is None and result.round is None


def test_published_agenda_as_of_remains_stable_after_future_publication(tmp_path) -> None:
    """Exercise the real published-batch reader, parser and restoration adapter."""

    class Session(_FixtureSession):
        """Serve a changed future round through the existing fixture client."""

        round_name = "Round of 64"

        def get(self, url, **kwargs):
            """Retain source identities while changing only future metadata."""
            response = super().get(url, **kwargs)
            if url.endswith("/atp-matches.html"):
                response.content = _html(self.round_name)
            return response

    database = tmp_path / "ratio.sqlite3"
    root = _write_sackmann_masters(tmp_path)
    session = Session()
    kwargs = dict(
        raw_dir=tmp_path / "raw",
        database_path=database,
        sackmann_raw_dir=root,
        session=session,
        request_delay_seconds=0.0,
    )
    refresh_tennisratio(now_utc=CAPTURE, **kwargs)
    before = restore_agenda_context(
        load_agenda_as_of(D, match_date=D, database_path=database), database
    )
    assert before.loc[before["gender"].eq("M"), "round_source"].eq("Round of 64").all()
    session.round_name = "Final"
    refresh_tennisratio(now_utc=CAPTURE + timedelta(days=1), **kwargs)
    after = restore_agenda_context(
        load_agenda_as_of(D, match_date=D, database_path=database), database
    )
    pd.testing.assert_frame_equal(before, after)

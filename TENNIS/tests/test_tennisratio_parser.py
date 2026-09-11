"""Contract tests for strict TennisRatio public document parsers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.tennisratio import AGENDA_AUDIT_COLUMNS, AGENDA_OUTPUT_COLUMNS
from src.tennisratio.client import authorized_url
from src.tennisratio.parser import (
    SCHEDULE_CONTEXT_COLUMNS,
    parse_agenda_html,
    parse_player_sitemap,
    parse_profile_html,
    parse_robots_txt,
)
from src.tennisratio.service import _path_disallowed
from src.tennisratio.types import TennisRatioHttpError, TennisRatioSchemaError


FIXTURES = Path(__file__).parent / "fixtures" / "tennisratio"
OBSERVED = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)


def test_agenda_parses_featured_and_compact_rows_with_namespaced_ids() -> None:
    """Both inspected layouts produce the shared agenda plus audit columns."""

    frame = parse_agenda_html(
        (FIXTURES / "atp-matches.html").read_bytes(),
        gender="M",
        source_url="https://www.tennisratio.com/atp-matches.html",
        retrieved_at_utc=OBSERVED,
    )

    assert (
        tuple(frame.columns)
        == AGENDA_OUTPUT_COLUMNS + AGENDA_AUDIT_COLUMNS + SCHEDULE_CONTEXT_COLUMNS
    )
    assert frame["source_match_id"].tolist() == ["tennisratio:M:1001", "tennisratio:M:1002"]
    assert frame["player_1_slug"].tolist() == [
        "tennisratio:AliceAlpha",
        "tennisratio:AliceAlpha",
    ]
    assert frame["scheduled_time"].tolist() == ["12:30", "15:00"]
    assert frame["scheduled_start_utc"].tolist() == [
        datetime(2026, 8, 22, 12, 30, tzinfo=UTC),
        datetime(2026, 8, 22, 15, 0, tzinfo=UTC),
    ]
    assert str(frame["scheduled_start_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["retrieved_at_utc"].dtype) == "datetime64[ns, UTC]"


def test_agenda_fails_if_structured_utc_evidence_disappears() -> None:
    """Markup drift cannot silently produce an invented schedule time."""

    html = (FIXTURES / "atp-matches.html").read_text(encoding="utf-8")
    malformed = html.replace(' class="match-time-display" data-utc="2026-08-22T12:30:00Z"', "")
    with pytest.raises(TennisRatioSchemaError, match="match-time-display"):
        parse_agenda_html(
            malformed,
            gender="M",
            source_url="https://www.tennisratio.com/atp-matches.html",
            retrieved_at_utc=OBSERVED,
        )


def test_agenda_preserves_tbd_time_without_inventing_intraday_order() -> None:
    """A source ``data-utc=None`` keeps the real date and a nullable time."""

    html = (FIXTURES / "atp-matches.html").read_text(encoding="utf-8")
    tbd = html.replace("2026-08-22T15:00:00Z", "None")
    frame = parse_agenda_html(
        tbd,
        gender="M",
        source_url="https://www.tennisratio.com/atp-matches.html",
        retrieved_at_utc=OBSERVED,
    )

    assert frame.loc[1, "match_date"].date().isoformat() == "2026-08-22"
    assert bool(frame.loc[1:1, "scheduled_time"].isna().all())
    assert bool(frame.loc[1:1, "scheduled_start_utc"].isna().all())
    assert frame.loc[1, "status_evidence"] == "tennisratio:agenda_time_tbd"


def test_profile_normalizes_nan_and_winner_score_perspective() -> None:
    """Known source NaN becomes null and losing-profile scores are reversed."""

    alice = parse_profile_html(
        (FIXTURES / "AliceAlpha.html").read_bytes(),
        source_url="https://www.tennisratio.com/players/AliceAlpha.html",
    )
    bob = parse_profile_html(
        (FIXTURES / "BobBeta.html").read_bytes(),
        source_url="https://www.tennisratio.com/players/BobBeta.html",
    )

    assert alice.player_payload["ovr"] == {"aces_avg": None}
    assert alice.matches[0].score_winner_perspective == "6-4 3-6 6-2"
    stats = alice.matches[0].stats
    assert stats is not None
    assert stats.first_serve_accuracy_pct == 62.5
    assert (stats.breakpoints_saved, stats.breakpoints_faced) == (5, 7)
    assert (stats.service_games_won, stats.service_games_played) == (10, 12)
    assert stats.return_second_serve_points_won_pct == 58.0
    assert bob.matches[0].score_winner_perspective == "6-4 3-6 6-2"
    assert (bob.matches[0].winner_sets_won, bob.matches[0].loser_sets_won) == (2, 1)


def test_profile_normalizes_source_surface_case_missing_and_infinite_odds() -> None:
    """Keep valid history without inventing a missing surface or nonfinite odd."""

    html = (FIXTURES / "AliceAlpha.html").read_text(encoding="utf-8")
    lowercase = parse_profile_html(
        html.replace('"surface":"Hard"', '"surface":"clay"'),
        source_url="https://www.tennisratio.com/players/AliceAlpha.html",
    )
    missing = parse_profile_html(
        html.replace('"surface":"Hard"', '"surface":""'),
        source_url="https://www.tennisratio.com/players/AliceAlpha.html",
    )
    infinite = parse_profile_html(
        html.replace('"player_odd":1.8', '"player_odd":Infinity'),
        source_url="https://www.tennisratio.com/players/AliceAlpha.html",
    )

    assert lowercase.matches[0].surface == "Clay"
    assert missing.matches[0].surface == ""
    assert infinite.matches[0].player_odd is None


def test_profile_still_rejects_unknown_populated_surface() -> None:
    """Do not coerce a genuinely unknown surface into a canonical category."""

    html = (FIXTURES / "AliceAlpha.html").read_text(encoding="utf-8")
    with pytest.raises(TennisRatioSchemaError, match="Unsupported profile match surface"):
        parse_profile_html(
            html.replace('"surface":"Hard"', '"surface":"Ice"'),
            source_url="https://www.tennisratio.com/players/AliceAlpha.html",
        )


def test_profile_rejects_impossible_match_statistics() -> None:
    """Schema drift cannot turn an impossible percentage or fraction into a feature."""

    html = (FIXTURES / "AliceAlpha.html").read_text(encoding="utf-8")
    with pytest.raises(TennisRatioSchemaError, match="between 0 and 100"):
        parse_profile_html(
            html.replace('"first_serve_accuracy":62.5', '"first_serve_accuracy":120'),
            source_url="https://www.tennisratio.com/players/AliceAlpha.html",
        )
    with pytest.raises(TennisRatioSchemaError, match="numerator cannot exceed"):
        parse_profile_html(
            html.replace('"breakpoints_saved_count":"5/7"', '"breakpoints_saved_count":"8/7"'),
            source_url="https://www.tennisratio.com/players/AliceAlpha.html",
        )


def test_sitemap_and_robots_are_strictly_scoped() -> None:
    """The inventory is namespaced and foreign profile hosts are rejected."""

    inventory = parse_player_sitemap((FIXTURES / "sitemap-players.xml").read_bytes())
    assert len(inventory) == 5
    assert inventory[0].source_player_key == "tennisratio:AliceAlpha"
    assert "/api/" in parse_robots_txt((FIXTURES / "robots.txt").read_bytes())

    foreign = (
        (FIXTURES / "sitemap-players.xml")
        .read_text(encoding="utf-8")
        .replace(
            "https://www.tennisratio.com/players/AliceAlpha.html",
            "https://example.com/players/AliceAlpha.html",
        )
    )
    with pytest.raises(TennisRatioSchemaError, match="Unauthorized profile URL"):
        parse_player_sitemap(foreign)


def test_robots_query_wildcards_do_not_block_public_pages() -> None:
    """A query-only rule cannot be widened to every path beginning with slash."""

    rules = parse_robots_txt((FIXTURES / "robots.txt").read_bytes())
    assert "/*?q=" in rules
    assert not _path_disallowed("/atp-matches.html", rules)
    assert not _path_disallowed("/players/Example.html", rules)
    assert _path_disallowed("/search?q=alcaraz", rules)
    assert _path_disallowed("/api/private", rules)


def test_robots_consecutive_user_agents_keep_wildcard_group_rules() -> None:
    """Several User-agent lines in one group retain wildcard directives."""

    policy = """User-agent: ExampleBot
User-agent: *
Disallow: /api/
User-agent: OtherBot
Disallow: /
"""
    assert parse_robots_txt(policy) == frozenset({"/api/"})


def test_http_allowlist_accepts_profiles_but_never_api_or_query_urls() -> None:
    """The only dynamic GET family is a canonical public player profile."""

    profile = "https://www.tennisratio.com/players/AliceAlpha.html"
    assert authorized_url(profile) == profile
    for rejected in ("/api/player/AliceAlpha", f"{profile}?private=1"):
        with pytest.raises(TennisRatioHttpError, match="outside the approved"):
            authorized_url(rejected)

"""Offline daily acquisition budget and starvation regressions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest

from scripts import update_tennisratio as cli
from src.tennisratio.service import _select_profiles
from src.tennisratio.store import TennisRatioStore
from src.tennisratio.types import SitemapPlayer


@dataclass
class _Report:
    """Minimal JSON-serializable updater result without any I/O."""

    changed: bool = False


@pytest.mark.parametrize(
    "arguments,expected",
    [([], 50), (["--full-inventory"], None), (["--max-profiles", "0"], 0)],
)
def test_real_cli_bounds_extras_unless_explicit_full_inventory(monkeypatch, arguments, expected):
    """Both launchers call this entrypoint; default acquisition cannot crawl everything."""

    refresh = Mock(return_value=_Report())
    monkeypatch.setattr(cli, "refresh_tennisratio", refresh)
    assert cli.main(arguments) == 0
    assert refresh.call_args.kwargs["max_profiles"] == expected


def _player(key: str) -> SitemapPlayer:
    """Build an inspected-style profile identity for selection only."""

    return SitemapPlayer(
        source_url=f"https://www.tennisratio.com/players/{key}.html",
        source_player_key=f"tennisratio:{key}",
        profile_slug=key,
        lastmod=None,
    )


def test_agenda_goes_first_and_fifty_extras_do_not_starve_the_remaining_players():
    """Daily sitemap churn cannot cause the same alphabetical fifty to win forever."""

    visible = _player("ZToday")
    extras = [_player(f"Extra{index:03}") for index in range(80)]
    arguments = {"visible_entries": {visible.source_url: visible}, "max_profiles": 50}
    first = _select_profiles(extras + [visible], **arguments)
    assert first[0] == visible
    assert len(first) == 51
    attempted = {entry.source_url: "2026-09-24T10:00:00Z" for entry in first}
    second = _select_profiles(extras + [visible], attempted_at=attempted, **arguments)
    assert second[0] == visible
    assert len(second) == 51
    assert second[1:31] == extras[50:]
    assert (
        len(
            _select_profiles(
                extras, visible_entries=arguments["visible_entries"], max_profiles=None
            )
        )
        == 81
    )


def test_budget_remembers_published_successes_and_failures_only(tmp_path: Path) -> None:
    """A failed profile rotates too; unpublished partial batches remain retryable."""

    database = tmp_path / "source.sqlite3"
    store = TennisRatioStore(database)
    observed = datetime(2026, 9, 24, 10, tzinfo=UTC)
    good, failed, unfinished = (_player(key) for key in ("Good", "Failed", "Unfinished"))
    for entry, batch in ((good, "published"), (unfinished, "partial")):
        store.record_profile_sync(
            batch_id=batch,
            entry=entry,
            source_sha256="a" * 64,
            first_seen_at_utc=observed,
            status="accepted",
        )
    store.record_profile_failure(
        batch_id="published",
        source_url=failed.source_url,
        source_sha256=None,
        source_player_key=failed.source_player_key,
        first_seen_at_utc=observed,
        reason="fixture acquisition failure",
    )
    assert store.profile_attempt_times() == {}
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO refresh_batches VALUES (
                'published', '2026-09-24', '2026-09-24T10:00:00Z', 'published',
                0, 0, 0, 0, 0, 0, 0, ?
            )
            """,
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO published_batches VALUES ('published', ?, ?, ?)",
            ("2026-09-24T10:01:00Z", str(tmp_path / "manifest.json"), "b" * 64),
        )
    assert store.profile_attempt_times() == {
        good.source_url: "2026-09-24T10:00:00Z",
        failed.source_url: "2026-09-24T10:00:00Z",
    }

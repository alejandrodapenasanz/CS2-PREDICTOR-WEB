"""Offline regressions for dated ranking badges, immutable storage and as-of."""

from __future__ import annotations

from datetime import UTC, date, datetime
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from BBDD import ranking_store
from PIPELINE.ranking_badges import parse_ranking_badges

HTML = """
<div class="teamRanking">
<a href="/ranking/teams/2026/september/7/11446" class="ranking-badge hltv-ranking"><span>HLTV: </span>#40</a>
<a href="/valve-ranking/teams/2026/september/12?teamId=11446" class="ranking-badge vrs-ranking"><span>VRS: </span>#49</a>
</div>
"""


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep capture validation independent of the CI machine's real clock."""

    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, tzinfo=UTC)

    monkeypatch.setattr(ranking_store, "datetime", FixtureClock)


def capture(root: Path, day: str, *, corrupt: bool = False) -> None:
    """Create one hashed snapshot fixture; never reads real private state."""
    path = root / "raw_html/match_snapshot/2390001.html.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(HTML)
    path.with_suffix(".gz.json").write_text(
        json.dumps(
            {
                "captured_at": day + "T10:00:00Z",
                "url": "https://www.hltv.org/matches/2390001/a-vs-b",
                "sha256": "bad" if corrupt else hashlib.sha256(HTML.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def database() -> sqlite3.Connection:
    """Disposable schema with representative core rows, no production tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY,hltv_match_id TEXT UNIQUE)")
    conn.execute("CREATE TABLE teams(team_id INTEGER PRIMARY KEY,hltv_id INTEGER UNIQUE)")
    conn.execute("INSERT INTO matches VALUES (1,'2390001')")
    conn.execute("INSERT INTO teams VALUES (1,11446)")
    return conn


def test_badges_parse_both_sources_and_exact_edition() -> None:
    """No points or Elo are inferred from an ordinal rank."""
    rows = parse_ranking_badges(HTML)
    assert [(r["ranking_type"], r["position"], r["ranking_date"]) for r in rows] == [
        ("hltv", 40, "2026-09-07"),
        ("valve", 49, "2026-09-12"),
    ]
    assert all(r["hltv_team_id"] == "11446" and "points" not in r for r in rows)


def test_ambiguous_badge_and_untrusted_links_not_guessed() -> None:
    """Reject conflicting ranks, undated links and external origins."""
    assert parse_ranking_badges(HTML + HTML.replace("#40", "#41"))[0]["ranking_type"] == "valve"
    assert parse_ranking_badges(HTML.replace('href="/', 'href="https://other.test/')) == []
    assert parse_ranking_badges(HTML.replace("2026/september/7/", ""))[0]["ranking_type"] == "valve"


def test_stored_badges_are_additive_idempotent_and_future_stable(tmp_path: Path) -> None:
    """Neither capture during D nor appending future matches can leak into D."""
    capture(tmp_path / "first", "2026-09-12")
    conn = database()
    try:
        report = ranking_store.ingest_rankings(conn, tmp_path / "first", apply=True)
        assert report["counts"]["inserted"] == 2
        assert not ranking_store.observations_asof(conn, date(2026, 9, 12))
        before = ranking_store.observations_asof(conn, date(2026, 9, 13))
        assert len(before) == 2
        again = ranking_store.ingest_rankings(conn, tmp_path / "first", apply=True)
        assert again["counts"]["already_stored"] == 2
        # Noon is later than this fixture, but still part of the forbidden day D.
        capture(tmp_path / "future", "2026-09-13")
        ranking_store.ingest_rankings(conn, tmp_path / "future", apply=True)
        assert ranking_store.observations_asof(conn, date(2026, 9, 13)) == before
        assert conn.execute("SELECT match_id,team_id FROM match_team_ranking_observations LIMIT 1").fetchone() == (1, 1)
    finally:
        conn.close()


def test_invalid_hash_is_quarantined_without_writing(tmp_path: Path) -> None:
    """A malformed archive cannot manufacture a ranking observation."""
    capture(tmp_path, "2026-09-12", corrupt=True)
    conn = database()
    try:
        report = ranking_store.ingest_rankings(conn, tmp_path, apply=True)
        assert report["quarantine_reasons"] == {"source_hash_mismatch": 1}
        assert not ranking_store.observations_asof(conn, date(2026, 9, 13))
    finally:
        conn.close()

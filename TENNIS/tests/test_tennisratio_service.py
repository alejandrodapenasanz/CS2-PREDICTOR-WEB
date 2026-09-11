"""End-to-end fixture tests for UTC-daily last-good TennisRatio refreshes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

import src.tennisratio.service as tennisratio_service
from src.tennisratio import (
    load_active_agenda,
    load_agenda_as_of,
    load_identity_resolutions,
    load_mapped_match_stats,
    load_mapped_rankings,
    load_mapped_results,
    refresh_tennisratio,
)
from src.tennisratio.service import (
    _daily_lock,
    _prune_directory,
    _write_immutable_manifest,
    _write_raw_snapshot,
)
from src.tennisratio.store import TennisRatioStore
from src.tennisratio.types import (
    TennisRatioLockError,
    TennisRatioSchemaError,
    TennisRatioStoreError,
    TennisRatioBlockedError,
)


FIXTURES = Path(__file__).parent / "fixtures" / "tennisratio"
BASE_URL = "https://www.tennisratio.com"


def test_exhausted_429_stops_profiles_and_preserves_published_batch(tmp_path, monkeypatch):
    """A blocked host never publishes a truncated refresh or advances last_good."""
    sackmann_root = _write_sackmann_masters(tmp_path)
    raw_dir = tmp_path / "raw"
    database = tmp_path / "tennisratio.sqlite3"
    arguments = dict(
        raw_dir=raw_dir,
        database_path=database,
        request_delay_seconds=0,
        sackmann_raw_dir=sackmann_root,
    )
    refresh_tennisratio(
        **arguments,
        now_utc=datetime(2026, 8, 22, 10, tzinfo=UTC),
        session=_FixtureSession(),
    )
    before = (raw_dir / "last_good.json").read_bytes()
    before_agenda = load_active_agenda(date(2026, 8, 24), database_path=database)
    original_get = tennisratio_service.TennisRatioClient.get
    stopped_calls = []
    closed = []

    def blocked_get(self, path, **kwargs):
        """Simulate exhausted common-client retries at the first profile."""
        stopped_calls.append(path)
        if "/players/" in path:
            raise TennisRatioBlockedError("HTTP 429: retries exhausted")
        return original_get(self, path, **kwargs)

    def close(self):
        """Assert the batch releases the owned session on exceptional exit."""
        closed.append(True)

    monkeypatch.setattr(tennisratio_service.TennisRatioClient, "get", blocked_get)
    monkeypatch.setattr(tennisratio_service.TennisRatioClient, "close", close)
    with pytest.raises(TennisRatioBlockedError, match="429"):
        refresh_tennisratio(
            **arguments,
            now_utc=datetime(2026, 8, 23, 10, tzinfo=UTC),
            session=_FixtureSession(),
        )
    assert len([url for url in stopped_calls if "/players/" in url]) == 1
    assert closed == [True]
    assert (raw_dir / "last_good.json").read_bytes() == before
    after_agenda = load_active_agenda(date(2026, 8, 24), database_path=database)
    assert before_agenda.equals(after_agenda)
    assert not TennisRatioStore(database).has_published_date(date(2026, 8, 23))


@dataclass
class _FixtureResponse:
    """Minimal response accepted by the restricted client."""

    url: str
    content: bytes
    headers: dict[str, str]
    status_code: int = 200


class _FixtureSession:
    """Serve only explicitly registered public fixture URLs."""

    def __init__(
        self,
        *,
        conflicting_bob: bool = False,
        blocked_robots: bool = False,
        future_alice_version: bool = False,
        retired_agenda: bool = False,
        invalid_alice_category: str | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.conflicting_bob = conflicting_bob
        self.blocked_robots = blocked_robots
        self.future_alice_version = future_alice_version
        self.retired_agenda = retired_agenda
        self.invalid_alice_category = invalid_alice_category

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> _FixtureResponse:
        """Return deterministic bytes while asserting GET safety options."""

        assert headers["Accept"]
        assert timeout[0] > 0
        assert allow_redirects is False
        self.calls.append(url)
        filename = {
            f"{BASE_URL}/robots.txt": "robots.txt",
            f"{BASE_URL}/atp-matches.html": "atp-matches.html",
            f"{BASE_URL}/wta-matches.html": "wta-matches.html",
            f"{BASE_URL}/sitemap-players.xml": "sitemap-players.xml",
            f"{BASE_URL}/players/AliceAlpha.html": "AliceAlpha.html",
            f"{BASE_URL}/players/BobBeta.html": "BobBeta.html",
            f"{BASE_URL}/players/ClaraClay.html": "ClaraClay.html",
            f"{BASE_URL}/players/DonnaDrop.html": "DonnaDrop.html",
            f"{BASE_URL}/players/EvanExtra.html": "EvanExtra.html",
        }[url]
        content = (FIXTURES / filename).read_bytes()
        if self.retired_agenda and filename in {"atp-matches.html", "wta-matches.html"}:
            content = (
                b'<html><div class="matches-date-group" data-date="2026-08-24"></div>'
                b"<!-- tournament-group --></html>"
            )
        if self.blocked_robots and filename == "robots.txt":
            content += b"Disallow: /players/\n"
        if self.conflicting_bob and filename == "BobBeta.html":
            content = content.replace(b'"score":"4-6 6-3 2-6"', b'"score":"4-6 6-3 0-6"')
        if self.future_alice_version and filename == "AliceAlpha.html":
            content = content.replace(b'"rank":10', b'"rank":9').replace(
                b'"first_serve_accuracy":62.5', b'"first_serve_accuracy":65.0'
            )
        if self.invalid_alice_category is not None and filename == "AliceAlpha.html":
            content = content.replace(
                b'"category":"ATP"',
                f'"category":"{self.invalid_alice_category}"'.encode(),
            )
        if filename == "robots.txt":
            content_type = "text/plain"
        elif filename.endswith(".xml"):
            content_type = "application/xml"
        else:
            content_type = "text/html; charset=utf-8"
        return _FixtureResponse(url=url, content=content, headers={"Content-Type": content_type})


def test_refresh_is_daily_offline_causal_and_bilaterally_deduplicated(tmp_path: Path) -> None:
    """A full first sync publishes once and exposes no same-day feature facts."""

    sackmann_root = _write_sackmann_masters(tmp_path)
    raw_dir = tmp_path / "raw"
    database = tmp_path / "tennisratio.sqlite3"
    session = _FixtureSession()
    observed = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)

    report = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=observed,
        request_delay_seconds=0.0,
        session=session,  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann_root,
    )

    assert report.status == "published"
    assert report.agenda_rows == 3
    assert report.profiles_succeeded == 5
    assert len(session.calls) == 9
    agenda = load_active_agenda(date(2026, 8, 22), database_path=database)
    assert len(agenda) == 3
    assert agenda["source_url"].str.startswith(BASE_URL).all()
    assert agenda["snapshot_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert len(load_identity_resolutions(database_path=database)) == 5
    assert load_identity_resolutions(as_of_date=date(2026, 8, 22), database_path=database) == ()
    assert len(load_identity_resolutions(as_of_date=date(2026, 8, 23), database_path=database)) == 5
    assert load_agenda_as_of(
        date(2026, 8, 22), match_date=date(2026, 8, 22), database_path=database
    ).empty
    causal_agenda = load_agenda_as_of(
        date(2026, 8, 23), match_date=date(2026, 8, 22), database_path=database
    )
    assert len(causal_agenda) == 3
    assert causal_agenda["snapshot_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert load_mapped_rankings(date(2026, 8, 22), database_path=database) == ()
    assert load_mapped_match_stats(date(2026, 8, 22), database_path=database) == ()
    assert (
        load_mapped_results(
            date(2026, 8, 22),
            gender="M",
            base_cutoff_by_gender={"M": date(2026, 8, 19)},
            database_path=database,
        )
        == ()
    )

    rankings = load_mapped_rankings(date(2026, 8, 23), database_path=database)
    assert {item.sackmann_player_id for item in rankings} == {1, 2, 3, 4, 5}
    results = load_mapped_results(
        date(2026, 8, 23),
        gender="M",
        base_cutoff_by_gender={"M": date(2026, 8, 19)},
        database_path=database,
    )
    assert len(results) == 1
    assert results[0].score == "6-4 3-6 6-2"
    assert (results[0].winner_sets_won, results[0].loser_sets_won) == (2, 1)
    assert results[0].first_seen_at_utc.date() == results[0].available_date
    match_stats = load_mapped_match_stats(date(2026, 8, 23), database_path=database)
    assert len(match_stats) == 1
    assert match_stats[0].stats.first_serve_accuracy_pct == 62.5
    with sqlite3.connect(database) as connection:
        exposed = connection.execute(
            "SELECT first_serve_accuracy_pct, breakpoints_saved "
            "FROM match_statistics_v1 WHERE source_player_key = ?",
            ("tennisratio:AliceAlpha",),
        ).fetchone()
    assert exposed == (62.5, "5/7")

    second = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=observed,
        request_delay_seconds=0.0,
        session=session,  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann_root,
    )
    assert second.status == "already_refreshed"
    assert len(session.calls) == 9


def test_publication_date_is_part_of_every_causal_availability_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facts captured before D stay hidden when their batch is published on D."""

    database = tmp_path / "sidecar.sqlite3"
    original_mark_published = TennisRatioStore.mark_published
    delayed_publication = datetime(2026, 8, 23, 9, 0, tzinfo=UTC)

    def mark_published_later(
        self: TennisRatioStore,
        *,
        batch_id: str,
        published_at_utc: datetime,
        manifest_path: Path,
        manifest_sha256: str,
    ) -> None:
        assert published_at_utc.date() == date(2026, 8, 22)
        original_mark_published(
            self,
            batch_id=batch_id,
            published_at_utc=delayed_publication,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )

    monkeypatch.setattr(TennisRatioStore, "mark_published", mark_published_later)
    refresh_tennisratio(
        raw_dir=tmp_path / "raw",
        database_path=database,
        now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=_write_sackmann_masters(tmp_path),
    )

    assert load_identity_resolutions(as_of_date=date(2026, 8, 23), database_path=database) == ()
    assert load_mapped_rankings(date(2026, 8, 23), database_path=database) == ()
    assert load_agenda_as_of(
        date(2026, 8, 23), match_date=date(2026, 8, 22), database_path=database
    ).empty
    assert (
        load_mapped_results(
            date(2026, 8, 23),
            gender="M",
            base_cutoff_by_gender={"M": date(2026, 8, 19)},
            database_path=database,
        )
        == ()
    )

    assert len(load_identity_resolutions(as_of_date=date(2026, 8, 24), database_path=database)) == 4
    assert len(load_mapped_rankings(date(2026, 8, 24), database_path=database)) == 4
    assert (
        len(
            load_mapped_results(
                date(2026, 8, 24),
                gender="M",
                base_cutoff_by_gender={"M": date(2026, 8, 19)},
                database_path=database,
            )
        )
        == 1
    )


def test_same_day_second_refresh_remaps_updated_sackmann_without_any_get(tmp_path: Path) -> None:
    """A changed local master publishes a causal remap despite today's HTTP gate."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    atp_master = sackmann / "atp" / "atp_players.csv"
    complete_master = atp_master.read_text(encoding="utf-8")
    without_alice = "\n".join(
        line for line in complete_master.splitlines() if ",Alice,Alpha," not in line
    )
    atp_master.write_text(without_alice + "\n", encoding="utf-8")
    observed = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)

    first = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=observed,
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert all(
        item.source_player_key != "tennisratio:AliceAlpha"
        for item in load_identity_resolutions(database_path=database)
    )
    with sqlite3.connect(database) as connection:
        alice_facts_before = connection.execute(
            "SELECT COUNT(*) FROM player_observations WHERE source_player_key = ?",
            ("tennisratio:AliceAlpha",),
        ).fetchone()[0]

    atp_master.write_text(complete_master, encoding="utf-8")
    no_network = _FixtureSession()
    second = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=no_network,  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )

    assert second.status == "already_refreshed"
    assert second.identity_remaps == 1
    assert second.changed is True
    assert second.canonical_fingerprint != first.canonical_fingerprint
    assert no_network.calls == []
    assert load_identity_resolutions(as_of_date=date(2026, 8, 22), database_path=database) == ()
    identities = load_identity_resolutions(as_of_date=date(2026, 8, 23), database_path=database)
    alice = next(item for item in identities if item.source_player_key == "tennisratio:AliceAlpha")
    assert alice.sackmann_player_id == 1
    assert alice.first_seen_at_utc.date() == date(2026, 8, 22)
    assert any(
        item.source_player_key == "tennisratio:AliceAlpha"
        for item in load_mapped_rankings(date(2026, 8, 23), database_path=database)
    )
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM player_observations WHERE source_player_key = ?",
                ("tennisratio:AliceAlpha",),
            ).fetchone()[0]
            == alice_facts_before
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM identity_remap_observations").fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM identity_remap_batches").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE identity_remap_observations SET candidate_count = 99")

    third_transport = _FixtureSession()
    third = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 13, 0, tzinfo=UTC),
        session=third_transport,  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert third.identity_remaps == 0
    assert third.changed is False
    assert third.canonical_fingerprint == second.canonical_fingerprint
    assert third_transport.calls == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM identity_remap_batches").fetchone()[0] == 1


def test_active_agenda_never_resurrects_rows_absent_from_latest_global_batch(
    tmp_path: Path,
) -> None:
    """Current presentation uses only the newest publication, even when empty."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert len(load_active_agenda(date(2026, 8, 22), database_path=database)) == 3

    retired = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(retired_agenda=True),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert retired.agenda_rows == 0
    assert load_active_agenda(date(2026, 8, 22), database_path=database).empty
    assert (
        len(
            load_agenda_as_of(
                date(2026, 8, 25),
                match_date=date(2026, 8, 22),
                database_path=database,
            )
        )
        == 3
    )


def test_failed_profile_versions_preserve_last_parsed_good_raw(tmp_path: Path) -> None:
    """Good A remains active across malformed B/C while failed raws are audit-only."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    store = TennisRatioStore(database)
    refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    good = store.latest_good_profile_snapshot("tennisratio:AliceAlpha")
    assert good is not None and good.compressed_path.is_file()

    failed_b = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(invalid_alice_category="BROKEN_B"),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert failed_b.profiles_failed == 1
    manifest_b = json.loads((raw_dir / "last_good.json").read_text(encoding="utf-8"))
    alice_b = next(
        item
        for item in manifest_b["profiles"]
        if item["source_player_key"] == good.source_player_key
    )
    failed_b_path = Path(alice_b["attempted_compressed_path"])
    assert alice_b["status"] == "failed"
    assert alice_b["sha256"] == good.source_sha256
    assert alice_b["attempted_sha256"] != good.source_sha256
    assert Path(alice_b["compressed_path"]) == good.compressed_path
    assert failed_b_path.is_file()

    failed_c = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(invalid_alice_category="BROKEN_C"),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert failed_c.profiles_failed == 1
    manifest_c = json.loads((raw_dir / "last_good.json").read_text(encoding="utf-8"))
    alice_c = next(
        item
        for item in manifest_c["profiles"]
        if item["source_player_key"] == good.source_player_key
    )
    profile_blobs = list(good.compressed_path.parent.glob("*.raw.gz"))
    assert Path(alice_c["compressed_path"]) == good.compressed_path
    assert Path(alice_c["attempted_compressed_path"]).is_file()
    assert good.compressed_path.is_file()
    assert not failed_b_path.exists()
    assert len(profile_blobs) == 2
    assert any(
        item.source_player_key == good.source_player_key
        for item in load_identity_resolutions(as_of_date=date(2026, 8, 26), database_path=database)
    )
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM quarantine_events
            WHERE source_player_key = ? AND reason = 'profile_acquisition_failure'
            """,
                (good.source_player_key,),
            ).fetchone()[0]
            == 2
        )
        pruned = connection.execute(
            "SELECT artifact_path FROM retention_events WHERE artifact_kind = 'profile_raw'"
        ).fetchall()
    assert str(failed_b_path.resolve()) in {str(row[0]) for row in pruned}


def test_robots_block_stops_before_any_other_get(tmp_path: Path) -> None:
    """A changed policy is enforced immediately after the sole robots GET."""

    session = _FixtureSession(blocked_robots=True)
    with pytest.raises(TennisRatioSchemaError, match="player profiles"):
        refresh_tennisratio(
            raw_dir=tmp_path / "raw",
            database_path=tmp_path / "sidecar.sqlite3",
            now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
            request_delay_seconds=0.0,
            max_profiles=0,
            session=session,  # type: ignore[arg-type]
            sackmann_raw_dir=_write_sackmann_masters(tmp_path),
        )
    assert session.calls == [f"{BASE_URL}/robots.txt"]


def test_future_profile_version_cannot_change_historical_as_of_queries(tmp_path: Path) -> None:
    """Appending a later observation preserves rankings/results already knowable at D."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    historical_rankings = load_mapped_rankings(date(2026, 8, 23), database_path=database)
    historical_identities = load_identity_resolutions(
        as_of_date=date(2026, 8, 23), database_path=database
    )
    historical_agenda = load_agenda_as_of(
        date(2026, 8, 23), match_date=date(2026, 8, 22), database_path=database
    )
    historical_results = load_mapped_results(
        date(2026, 8, 23),
        gender="M",
        base_cutoff_by_gender={"M": date(2026, 8, 19)},
        database_path=database,
    )
    historical_stats = load_mapped_match_stats(
        date(2026, 8, 23), gender="M", database_path=database
    )

    refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
        force=True,
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(future_alice_version=True),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert load_mapped_rankings(date(2026, 8, 23), database_path=database) == historical_rankings
    assert (
        load_identity_resolutions(as_of_date=date(2026, 8, 23), database_path=database)
        == historical_identities
    )
    assert load_agenda_as_of(
        date(2026, 8, 23), match_date=date(2026, 8, 22), database_path=database
    ).equals(historical_agenda)
    assert (
        load_mapped_results(
            date(2026, 8, 23),
            gender="M",
            base_cutoff_by_gender={"M": date(2026, 8, 19)},
            database_path=database,
        )
        == historical_results
    )
    assert (
        load_mapped_match_stats(date(2026, 8, 23), gender="M", database_path=database)
        == historical_stats
    )
    current_stats = load_mapped_match_stats(date(2026, 8, 25), gender="M", database_path=database)
    assert current_stats[0].stats.first_serve_accuracy_pct == 65.0


def test_bilateral_conflict_is_quarantined_and_not_exposed(tmp_path: Path) -> None:
    """Contradictory winner-oriented scores are persisted as quarantine evidence."""

    database = tmp_path / "sidecar.sqlite3"
    refresh_tennisratio(
        raw_dir=tmp_path / "raw",
        database_path=database,
        now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(conflicting_bob=True),  # type: ignore[arg-type]
        sackmann_raw_dir=_write_sackmann_masters(tmp_path),
    )
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM quarantine_events WHERE reason='bilateral_match_conflict'"
        ).fetchone()[0]
    assert count == 1
    assert (
        load_mapped_results(
            date(2026, 8, 23),
            gender="M",
            base_cutoff_by_gender={"M": date(2026, 8, 19)},
            database_path=database,
        )
        == ()
    )


def test_late_failed_batch_is_invisible_and_retry_can_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile facts staged before a core failure never leak into public reads."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    original_complete = TennisRatioStore.complete_batch

    def fail_complete(self: TennisRatioStore, **kwargs: object) -> None:
        raise TennisRatioStoreError("fixture late failure")

    monkeypatch.setattr(TennisRatioStore, "complete_batch", fail_complete)
    with pytest.raises(TennisRatioStoreError, match="late failure"):
        refresh_tennisratio(
            raw_dir=raw_dir,
            database_path=database,
            now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
            request_delay_seconds=0.0,
            max_profiles=0,
            session=_FixtureSession(),  # type: ignore[arg-type]
            sackmann_raw_dir=sackmann,
        )
    assert load_identity_resolutions(database_path=database) == ()
    assert load_mapped_rankings(date(2026, 8, 23), database_path=database) == ()
    assert not (raw_dir / "last_good.json").exists()

    monkeypatch.setattr(TennisRatioStore, "complete_batch", original_complete)
    retry = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 11, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=_FixtureSession(),  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert retry.status == "published"
    assert len(load_identity_resolutions(database_path=database)) == 4


def test_published_manifest_repairs_last_good_after_pointer_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DB-published immutable manifest repairs a missing pointer before any GET."""

    database = tmp_path / "sidecar.sqlite3"
    raw_dir = tmp_path / "raw"
    sackmann = _write_sackmann_masters(tmp_path)
    original_atomic_write = tennisratio_service._atomic_write

    def fail_last_good(path: Path, content: bytes) -> None:
        if path.name == "last_good.json":
            raise OSError("fixture pointer crash")
        original_atomic_write(path, content)

    monkeypatch.setattr(tennisratio_service, "_atomic_write", fail_last_good)
    with pytest.raises(OSError, match="pointer crash"):
        refresh_tennisratio(
            raw_dir=raw_dir,
            database_path=database,
            now_utc=datetime(2026, 8, 22, 10, 0, tzinfo=UTC),
            request_delay_seconds=0.0,
            max_profiles=0,
            session=_FixtureSession(),  # type: ignore[arg-type]
            sackmann_raw_dir=sackmann,
        )
    assert not (raw_dir / "last_good.json").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM published_batches").fetchone()[0] == 1

    monkeypatch.setattr(tennisratio_service, "_atomic_write", original_atomic_write)
    no_network = _FixtureSession()
    report = refresh_tennisratio(
        raw_dir=raw_dir,
        database_path=database,
        now_utc=datetime(2026, 8, 22, 11, 0, tzinfo=UTC),
        request_delay_seconds=0.0,
        max_profiles=0,
        session=no_network,  # type: ignore[arg-type]
        sackmann_raw_dir=sackmann,
    )
    assert report.status == "already_refreshed"
    assert report.manifest_path.is_file()
    assert no_network.calls == []


def test_old_lock_recovers_but_live_current_day_lock_is_never_stolen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery preserves a same-day owner whose PID is still alive."""

    raw_dir = tmp_path / "raw"
    lock_dir = raw_dir / ".refresh.lock"
    lock_dir.mkdir(parents=True)
    owner = {
        "pid": 123,
        "utc_date": "2026-08-21",
        "acquired_at_utc": "2026-08-21T10:00:00Z",
    }
    (lock_dir / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    with _daily_lock(raw_dir, date(2026, 8, 22)):
        assert lock_dir.is_dir()
    assert not lock_dir.exists()

    lock_dir.mkdir()
    owner["utc_date"] = "2026-08-22"
    owner["acquired_at_utc"] = "2026-08-22T10:00:00Z"
    (lock_dir / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    monkeypatch.setattr(tennisratio_service, "_pid_is_alive", lambda _: True)
    with pytest.raises(TennisRatioLockError, match="current day"):
        with _daily_lock(raw_dir, date(2026, 8, 22)):
            raise AssertionError("Current-day lock was stolen.")
    (lock_dir / "owner.json").unlink()
    lock_dir.rmdir()


def test_current_day_lock_with_dead_owner_is_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed unattended refresh cannot block every later daily run."""

    raw_dir = tmp_path / "raw"
    lock_dir = raw_dir / ".refresh.lock"
    lock_dir.mkdir(parents=True)
    owner = {
        "pid": 123,
        "utc_date": "2026-08-22",
        "acquired_at_utc": "2026-08-22T10:00:00Z",
    }
    (lock_dir / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    monkeypatch.setattr(tennisratio_service, "_pid_is_alive", lambda _: False)

    with _daily_lock(raw_dir, date(2026, 8, 22)):
        replacement = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
        assert replacement["pid"] == os.getpid()
    assert not lock_dir.exists()


def test_raw_snapshot_retention_keeps_only_two_versions(tmp_path: Path) -> None:
    """Raw object retention is bounded by logical resource, not by whole source."""

    raw_dir = tmp_path / "raw"
    for index in range(3):
        content = f"version-{index}".encode()
        _write_raw_snapshot(
            raw_dir,
            "agenda_atp",
            content,
            hashlib.sha256(content).hexdigest(),
            datetime(2026, 8, 20 + index, tzinfo=UTC),
        )
    assert len(list((raw_dir / "snapshots" / "agenda_atp").glob("*.raw.gz"))) == 2


def test_retention_protects_last_good_during_failures_and_bounds_manifests(
    tmp_path: Path,
) -> None:
    """A->B->C->A keeps active core/profile evidence and exactly two versions."""

    raw_dir = tmp_path / "raw"
    store = TennisRatioStore(tmp_path / "retention.sqlite3")
    active_content = b"active"
    active_core = _write_raw_snapshot(
        raw_dir,
        "agenda_atp",
        active_content,
        hashlib.sha256(active_content).hexdigest(),
        datetime(2026, 8, 20, tzinfo=UTC),
    )
    active_profile = _write_raw_snapshot(
        raw_dir,
        "profile/AliceAlpha",
        active_content,
        hashlib.sha256(active_content).hexdigest(),
        datetime(2026, 8, 20, tzinfo=UTC),
    )
    (raw_dir / "last_good.json").write_text(
        json.dumps(
            {
                "sources": [{"compressed_path": str(active_core.resolve())}],
                "profiles": [{"compressed_path": str(active_profile.resolve())}],
            }
        ),
        encoding="utf-8",
    )
    for resource, active in (
        ("agenda_atp", active_core),
        ("profile/AliceAlpha", active_profile),
    ):
        for label in (b"failed-one", b"failed-two"):
            _write_raw_snapshot(
                raw_dir,
                resource,
                label,
                hashlib.sha256(label).hexdigest(),
                datetime(2026, 8, 21, tzinfo=UTC),
                on_pruned=lambda paths, kind=resource: store.record_retention_events(
                    batch_id="retention-fixture",
                    artifact_kind=kind,
                    paths=paths,
                    pruned_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
                ),
            )
        reappeared = _write_raw_snapshot(
            raw_dir,
            resource,
            active_content,
            hashlib.sha256(active_content).hexdigest(),
            datetime(2026, 8, 22, tzinfo=UTC),
        )
        blobs = list((raw_dir / "snapshots" / Path(resource)).glob("*.raw.gz"))
        assert reappeared == active
        assert active in blobs
        assert len(blobs) == 2

    with sqlite3.connect(store.database_path) as connection:
        events = connection.execute(
            "SELECT artifact_kind, artifact_path, reason FROM retention_events"
        ).fetchall()
    assert {row[0] for row in events} == {"agenda_atp", "profile/AliceAlpha"}
    assert all(row[2] == "retention_limit_2" for row in events)
    assert all(not Path(row[1]).exists() for row in events)

    manifest_dir = raw_dir / "manifests"
    for index in range(3):
        current = _write_immutable_manifest(raw_dir, f"batch-{index}", b"{}\n")
        _prune_directory(manifest_dir, keep=2, protected={current})
    assert len(list(manifest_dir.glob("*.json"))) == 2


def _write_sackmann_masters(tmp_path: Path) -> Path:
    """Create minimal exact-name/DOB Sackmann masters for an isolated store."""

    root = tmp_path / "sackmann"
    header = "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
    atp = (
        header
        + "1,Alice,Alpha,R,19900101,ESP,,\n"
        + "2,Bob,Beta,L,19920202,USA,,\n"
        + "5,Evan,Extra,R,19950505,CAN,,\n"
    )
    wta = header + "3,Clara,Clay,R,19930303,FRA,,\n4,Donna,Drop,R,19940404,GBR,,\n"
    (root / "atp").mkdir(parents=True)
    (root / "wta").mkdir(parents=True)
    (root / "atp" / "atp_players.csv").write_text(atp, encoding="utf-8")
    (root / "wta" / "wta_players.csv").write_text(wta, encoding="utf-8")
    return root

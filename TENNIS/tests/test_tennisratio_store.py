"""Append-only, quarantine and gender-isolation tests for the sidecar."""

from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
from pathlib import Path
import sqlite3

from src.tennisratio.store import TennisRatioStore
from src.tennisratio.types import HttpPayload, IdentityDecision, ParsedProfile, ProfileMatch


OBSERVED = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)


def test_match_conflict_lookup_has_dedicated_source_key_index(tmp_path: Path) -> None:
    """Keep per-match conflict checks indexed as the profile history grows."""

    database = tmp_path / "sidecar.sqlite3"
    TennisRatioStore(database)

    with sqlite3.connect(database) as connection:
        columns_by_index = {
            str(index_row[1]): tuple(
                str(column_row[2])
                for column_row in connection.execute(
                    f"PRAGMA index_info({index_row[1]!r})"
                ).fetchall()
            )
            for index_row in connection.execute(
                "PRAGMA index_list('match_observations')"
            ).fetchall()
        }

    assert columns_by_index["match_source_key_idx"] == ("source_match_key",)


def test_duplicate_profile_match_is_ingested_once(tmp_path: Path) -> None:
    """Treat a verbatim repeated source row as one immutable observation."""

    database = tmp_path / "sidecar.sqlite3"
    store = TennisRatioStore(database)
    content = b"profile-with-duplicate-match"
    sha = hashlib.sha256(content).hexdigest()
    url = "https://www.tennisratio.com/players/DuplicatePlayer.html"
    snapshot_id = store.register_snapshot(
        HttpPayload(
            source_url=url,
            content=content,
            content_type="text/html",
            retrieved_at_utc=OBSERVED,
            sha256=sha,
        ),
        batch_id="batch",
        resource_kind="profile",
        compressed_path=tmp_path / "profile.raw.gz",
    )
    match = ProfileMatch(
        effective_date=date(2026, 8, 20),
        tournament="Fixture Open",
        tour_level="ATP",
        surface="Hard",
        round="R32",
        rival_name="Rival Player",
        rival_source_key="tennisratio:RivalPlayer",
        result="Win",
        score="6-4 6-4",
        score_winner_perspective="6-4 6-4",
        winner_sets_won=2,
        loser_sets_won=0,
        player_odd=None,
        rival_odd=None,
        raw_payload={"duplicate": True},
    )
    profile = ParsedProfile(
        source_url=url,
        source_player_key="tennisratio:DuplicatePlayer",
        profile_slug="DuplicatePlayer",
        gender="M",
        player_name="Duplicate Player",
        dob=None,
        rank=123,
        country=None,
        hand=None,
        matches=(match, match),
        player_payload={"id": "DuplicatePlayer"},
    )

    assert store.record_profile(
        batch_id="batch",
        snapshot_id=snapshot_id,
        source_sha256=sha,
        first_seen_at_utc=OBSERVED,
        effective_date=date(2026, 8, 21),
        sitemap_lastmod=date(2026, 8, 21),
        profile=profile,
        decision=IdentityDecision(
            status="mapped",
            sackmann_player_id=999,
            method="fixture",
            candidate_count=1,
        ),
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_observations").fetchone()[0] == 1


def test_identity_latest_is_partitioned_by_gender_and_tables_are_immutable(tmp_path: Path) -> None:
    """The same source key cannot collapse ATP/WTA universes and facts reject UPDATE."""

    database = tmp_path / "sidecar.sqlite3"
    store = TennisRatioStore(database)
    for gender, player_id, suffix in (("M", 101, "Man"), ("F", 202, "Woman")):
        content = f"profile-{suffix}".encode()
        sha = hashlib.sha256(content).hexdigest()
        url = f"https://www.tennisratio.com/players/Shared{suffix}.html"
        payload = HttpPayload(
            source_url=url,
            content=content,
            content_type="text/html",
            retrieved_at_utc=OBSERVED,
            sha256=sha,
        )
        snapshot_id = store.register_snapshot(
            payload,
            batch_id="batch",
            resource_kind="profile",
            compressed_path=tmp_path / f"{suffix}.gz",
        )
        profile = ParsedProfile(
            source_url=url,
            source_player_key="tennisratio:Shared",
            profile_slug=f"Shared{suffix}",
            gender=gender,  # type: ignore[arg-type]
            player_name=f"Shared {suffix}",
            dob=None,
            rank=player_id,
            country=None,
            hand=None,
            matches=(),
            player_payload={"id": f"Shared{suffix}"},
        )
        store.record_profile(
            batch_id="batch",
            snapshot_id=snapshot_id,
            source_sha256=sha,
            first_seen_at_utc=OBSERVED,
            effective_date=date(2026, 8, 21),
            sitemap_lastmod=date(2026, 8, 21),
            profile=profile,
            decision=IdentityDecision(
                status="mapped",
                sackmann_player_id=player_id,
                method="fixture",
                candidate_count=1,
            ),
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO refresh_batches VALUES (
                'batch', '2026-08-22', '2026-08-22T10:00:00Z', 'published',
                0, 0, 0, 0, 0, 0, 0, ?
            )
            """,
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO published_batches VALUES ('batch', ?, ?, ?)",
            ("2026-08-22T10:01:00Z", str(tmp_path / "manifest.json"), "b" * 64),
        )

    identities = store.load_identity_resolutions()
    assert {(item.gender, item.sackmann_player_id) for item in identities} == {
        ("M", 101),
        ("F", 202),
    }
    assert store.load_identity_resolutions(as_of_date=date(2026, 8, 22)) == ()
    assert len(store.load_identity_resolutions(as_of_date=date(2026, 8, 23))) == 2
    with sqlite3.connect(database) as connection:
        try:
            connection.execute("UPDATE player_observations SET ranking = 1")
        except sqlite3.IntegrityError as exc:
            assert "append-only" in str(exc)
        else:
            raise AssertionError("Immutable player observation accepted UPDATE.")

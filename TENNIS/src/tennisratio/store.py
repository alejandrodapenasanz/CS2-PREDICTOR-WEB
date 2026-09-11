"""Append-only SQLite sidecar for TennisRatio provenance and mapped facts.

This database is intentionally separate from the sacred operational database.
All source facts, identity decisions and quarantine events are immutable; SQL
triggers reject updates and deletes.  A published-batch event is appended only
after its immutable manifest is durable; the mutable ``last_good`` pointer is
written afterward and can be repaired from that published manifest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Final, Iterator, cast

import pandas as pd
from unidecode import unidecode

from .types import (
    AGENDA_AUDIT_COLUMNS,
    AGENDA_OUTPUT_COLUMNS,
    Gender,
    GoodProfileSnapshot,
    HttpPayload,
    IdentityDecision,
    IdentityRemapCandidate,
    IdentityStatus,
    MappedIdentity,
    MappedMatchStats,
    MappedRanking,
    MappedResult,
    ParsedProfile,
    ProfileMatchStats,
    SitemapPlayer,
    TennisRatioStoreError,
)


_SCHEMA_VERSION: Final[int] = 3
_IMMUTABLE_TABLES: Final[tuple[str, ...]] = (
    "refresh_batches",
    "published_batches",
    "source_snapshots",
    "agenda_observations",
    "profile_inventory_observations",
    "profile_sync_observations",
    "player_observations",
    "match_observations",
    "identity_resolutions",
    "identity_remap_observations",
    "identity_remap_batches",
    "quarantine_events",
    "retention_events",
)
_NORMALIZE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


class TennisRatioStore:
    """Own the additive TennisRatio sidecar and its causal read APIs."""

    def __init__(self, database_path: Path) -> None:
        """Initialize or validate the lateral database schema."""

        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_snapshot(
        self,
        payload: HttpPayload,
        *,
        batch_id: str,
        resource_kind: str,
        compressed_path: Path,
    ) -> str:
        """Append one unique source URL/SHA snapshot and return its stable id."""

        snapshot_id = _stable_id("snapshot", payload.source_url, payload.sha256)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO source_snapshots (
                    snapshot_id, batch_id, resource_kind, source_url,
                    first_seen_at_utc, sha256, byte_size, content_type,
                    compressed_path, http_etag, http_last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    batch_id,
                    resource_kind,
                    payload.source_url,
                    _iso_instant(payload.retrieved_at_utc),
                    payload.sha256,
                    len(payload.content),
                    payload.content_type,
                    str(Path(compressed_path).resolve()),
                    payload.etag,
                    payload.last_modified,
                ),
            )
        return snapshot_id

    def has_published_date(self, utc_date: date) -> bool:
        """Return whether an immutable last-good batch exists for that UTC day."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM published_batches AS published
                JOIN refresh_batches AS batch USING (batch_id)
                WHERE batch.utc_date = ?
                LIMIT 1
                """,
                (utc_date.isoformat(),),
            ).fetchone()
        return row is not None

    def published_report_row(self, utc_date: date) -> sqlite3.Row | None:
        """Return the latest persisted report fields for an already-run date."""

        with self._connect() as connection:
            return connection.execute(
                """
                SELECT batch.*, published.manifest_path, published.manifest_sha256
                FROM published_batches AS published
                JOIN refresh_batches AS batch USING (batch_id)
                WHERE batch.utc_date = ?
                ORDER BY published.published_at_utc DESC
                LIMIT 1
                """,
                (utc_date.isoformat(),),
            ).fetchone()

    def latest_published_manifest_row(self) -> sqlite3.Row | None:
        """Return the newest published immutable manifest reference."""

        with self._connect() as connection:
            return connection.execute(
                """
                SELECT published.*, batch.utc_date, batch.status
                FROM published_batches AS published
                JOIN refresh_batches AS batch USING (batch_id)
                ORDER BY published.published_at_utc DESC
                LIMIT 1
                """
            ).fetchone()

    def inventory_is_empty(self) -> bool:
        """Return whether no sitemap inventory version has ever been accepted."""

        with self._connect() as connection:
            value = connection.execute(
                """
                SELECT COUNT(*)
                FROM profile_inventory_observations AS inventory
                JOIN published_batches AS published USING (batch_id)
                """
            ).fetchone()[0]
        return int(value) == 0

    def inventory_version_seen(self, entry: SitemapPlayer) -> bool:
        """Check the immutable ``(loc,lastmod)`` delta key."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM profile_inventory_observations AS inventory
                JOIN published_batches AS published USING (batch_id)
                WHERE source_url = ? AND COALESCE(lastmod, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (entry.source_url, _iso_date(entry.lastmod)),
            ).fetchone()
        return row is not None

    def profile_version_succeeded(self, entry: SitemapPlayer) -> bool:
        """Return whether a sitemap version has a validated good profile fact."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM profile_sync_observations AS sync
                JOIN published_batches AS published USING (batch_id)
                WHERE source_url = ?
                  AND COALESCE(lastmod, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (entry.source_url, _iso_date(entry.lastmod)),
            ).fetchone()
        return row is not None

    def record_profile_sync(
        self,
        *,
        batch_id: str,
        entry: SitemapPlayer,
        source_sha256: str,
        first_seen_at_utc: datetime,
        status: str,
    ) -> None:
        """Record a successful ``(loc,lastmod)+SHA`` delta decision append-only."""

        if status not in {"accepted", "unchanged"}:
            raise ValueError("Profile sync status must be accepted or unchanged.")
        observation_id = _stable_id(
            "profile-sync",
            batch_id,
            entry.source_url,
            _iso_date(entry.lastmod) or "missing",
            source_sha256,
        )
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO profile_sync_observations (
                    observation_id, batch_id, source_url, source_player_key,
                    lastmod, source_sha256, status, first_seen_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    batch_id,
                    entry.source_url,
                    entry.source_player_key,
                    _iso_date(entry.lastmod),
                    source_sha256,
                    status,
                    _iso_instant(first_seen_at_utc),
                ),
            )

    def latest_good_profile_sha(self, source_player_key: str) -> str | None:
        """Return the newest successfully parsed profile SHA, ignoring failures."""

        snapshot = self.latest_good_profile_snapshot(source_player_key)
        return None if snapshot is None else snapshot.source_sha256

    def latest_good_profile_snapshot(
        self,
        source_player_key: str,
    ) -> GoodProfileSnapshot | None:
        """Return the newest published parsed profile and its immutable raw path."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT observation.source_player_key, observation.source_sha256,
                       snapshot.compressed_path
                FROM player_observations AS observation
                JOIN published_batches AS published USING (batch_id)
                JOIN source_snapshots AS snapshot USING (snapshot_id)
                WHERE source_player_key = ?
                ORDER BY published.published_at_utc DESC,
                         observation.first_seen_at_utc DESC,
                         observation.observation_id DESC
                LIMIT 1
                """,
                (source_player_key,),
            ).fetchone()
        if row is None:
            return None
        return GoodProfileSnapshot(
            source_player_key=str(row["source_player_key"]),
            source_sha256=str(row["source_sha256"]),
            compressed_path=Path(str(row["compressed_path"])).resolve(),
        )

    def record_profile(
        self,
        *,
        batch_id: str,
        snapshot_id: str,
        source_sha256: str,
        first_seen_at_utc: datetime,
        effective_date: date,
        sitemap_lastmod: date | None,
        profile: ParsedProfile,
        decision: IdentityDecision,
    ) -> bool:
        """Append one independently transactional profile and all its matches.

        Returns ``False`` when the exact profile URL/SHA was already accepted;
        no duplicate facts or identity decisions are appended in that case.
        """

        observed = _iso_instant(first_seen_at_utc)
        observation_id = _stable_id("player", batch_id, profile.source_player_key, source_sha256)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT 1 FROM player_observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            if existing is not None:
                return False
            connection.execute(
                """
                INSERT INTO player_observations (
                    observation_id, batch_id, snapshot_id, source_player_key,
                    gender, player_name, dob, ranking, country, hand,
                    effective_date, sitemap_lastmod, first_seen_at_utc, source_url,
                    source_sha256, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    batch_id,
                    snapshot_id,
                    profile.source_player_key,
                    profile.gender,
                    profile.player_name,
                    _iso_date(profile.dob),
                    profile.rank,
                    profile.country,
                    profile.hand,
                    effective_date.isoformat(),
                    _iso_date(sitemap_lastmod),
                    observed,
                    profile.source_url,
                    source_sha256,
                    _json(profile.player_payload),
                ),
            )
            resolution_id = _stable_id("resolution", observation_id, decision.status)
            connection.execute(
                """
                INSERT INTO identity_resolutions (
                    resolution_id, player_observation_id, source_player_key,
                    gender, sackmann_player_id, status, method,
                    candidate_count, first_seen_at_utc, source_url, source_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolution_id,
                    observation_id,
                    profile.source_player_key,
                    profile.gender,
                    decision.sackmann_player_id,
                    decision.status,
                    decision.method,
                    decision.candidate_count,
                    observed,
                    profile.source_url,
                    source_sha256,
                ),
            )
            if decision.status != "mapped":
                self._insert_quarantine(
                    connection,
                    batch_id=batch_id,
                    source_url=profile.source_url,
                    source_sha256=source_sha256,
                    source_player_key=profile.source_player_key,
                    reason=f"identity_{decision.status}",
                    observed=observed,
                    payload={
                        "method": decision.method,
                        "candidate_count": decision.candidate_count,
                        "player_name": profile.player_name,
                        "dob": _iso_date(profile.dob),
                        "gender": profile.gender,
                    },
                )
            for item in profile.matches:
                source_match_key = _profile_match_key(profile, item)
                conflict = _match_conflict_payload(
                    connection,
                    profile=profile,
                    item=item,
                    source_match_key=source_match_key,
                )
                if conflict is not None:
                    self._insert_quarantine(
                        connection,
                        batch_id=batch_id,
                        source_url=profile.source_url,
                        source_sha256=source_sha256,
                        source_player_key=profile.source_player_key,
                        reason="bilateral_match_conflict",
                        observed=observed,
                        payload=conflict,
                    )
                match_observation_id = _stable_id(
                    "match",
                    batch_id,
                    source_match_key,
                    profile.source_player_key,
                    source_sha256,
                )
                duplicate = connection.execute(
                    "SELECT 1 FROM match_observations WHERE observation_id = ?",
                    (match_observation_id,),
                ).fetchone()
                if duplicate is not None:
                    # A profile can repeat the same historical row verbatim.
                    # The stable observation already preserves that evidence;
                    # a contradictory duplicate was quarantined just above.
                    continue
                connection.execute(
                    """
                    INSERT INTO match_observations (
                        observation_id, batch_id, snapshot_id, source_match_key,
                        source_player_key, rival_source_key, gender,
                        effective_date, tournament, tour_level, surface,
                        round_name, player_name, rival_name, result,
                        score_profile_perspective, score_winner_perspective,
                        winner_sets_won, loser_sets_won, player_odd, rival_odd,
                        first_seen_at_utc, source_url, source_sha256, payload_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        match_observation_id,
                        batch_id,
                        snapshot_id,
                        source_match_key,
                        profile.source_player_key,
                        item.rival_source_key,
                        profile.gender,
                        item.effective_date.isoformat(),
                        item.tournament,
                        item.tour_level,
                        item.surface,
                        item.round,
                        profile.player_name,
                        item.rival_name,
                        item.result,
                        item.score,
                        item.score_winner_perspective,
                        item.winner_sets_won,
                        item.loser_sets_won,
                        item.player_odd,
                        item.rival_odd,
                        observed,
                        profile.source_url,
                        source_sha256,
                        _json(item.raw_payload),
                    ),
                )
                if item.rival_source_key is None:
                    self._insert_quarantine(
                        connection,
                        batch_id=batch_id,
                        source_url=profile.source_url,
                        source_sha256=source_sha256,
                        source_player_key=profile.source_player_key,
                        reason="match_rival_without_profile_key",
                        observed=observed,
                        payload={
                            "source_match_key": source_match_key,
                            "rival_name": item.rival_name,
                            "effective_date": item.effective_date.isoformat(),
                        },
                    )
        return True

    def identity_remap_candidates(self) -> tuple[IdentityRemapCandidate, ...]:
        """Return latest published profile facts with their current decisions."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT observation.*
                FROM player_observations AS observation
                JOIN published_batches AS published USING (batch_id)
                ORDER BY published.published_at_utc,
                         observation.first_seen_at_utc,
                         observation.observation_id
                """
            ).fetchall()
        latest_profiles = {
            (cast(Gender, str(row["gender"])), str(row["source_player_key"])): row for row in rows
        }
        latest_resolutions = {
            (cast(Gender, str(row["gender"])), str(row["source_player_key"])): row
            for row in self._resolution_rows(as_of_date=None, gender=None)
        }
        candidates: list[IdentityRemapCandidate] = []
        for key, row in latest_profiles.items():
            resolution = latest_resolutions.get(key)
            if resolution is None:
                raise TennisRatioStoreError(
                    f"Published profile {key!r} has no published identity decision."
                )
            dob_raw = row["dob"]
            candidates.append(
                IdentityRemapCandidate(
                    player_observation_id=str(row["observation_id"]),
                    gender=key[0],
                    source_player_key=key[1],
                    player_name=str(row["player_name"]),
                    dob=None if dob_raw is None else date.fromisoformat(str(dob_raw)),
                    source_url=str(row["source_url"]),
                    source_sha256=str(row["source_sha256"]),
                    current_status=cast(IdentityStatus, str(resolution["status"])),
                    current_sackmann_player_id=None
                    if resolution["sackmann_player_id"] is None
                    else int(resolution["sackmann_player_id"]),
                    current_method=str(resolution["method"]),
                    current_candidate_count=int(resolution["candidate_count"]),
                )
            )
        return tuple(sorted(candidates, key=lambda item: (item.gender, item.source_player_key)))

    def record_identity_remap(
        self,
        *,
        batch_id: str,
        candidate: IdentityRemapCandidate,
        decision: IdentityDecision,
        mapping_source_sha256: str,
        first_seen_at_utc: datetime,
    ) -> bool:
        """Stage one changed mapping decision without rewriting its profile fact."""

        previous = (
            candidate.current_status,
            candidate.current_sackmann_player_id,
            candidate.current_method,
            candidate.current_candidate_count,
        )
        current = (
            decision.status,
            decision.sackmann_player_id,
            decision.method,
            decision.candidate_count,
        )
        if current == previous:
            return False
        if re.fullmatch(r"[0-9a-f]{64}", mapping_source_sha256) is None:
            raise ValueError("mapping_source_sha256 must be lowercase SHA-256.")
        observed = _iso_instant(first_seen_at_utc)
        remap_id = _stable_id(
            "identity-remap",
            batch_id,
            candidate.gender,
            candidate.source_player_key,
            decision.status,
            str(decision.sackmann_player_id),
            decision.method,
            str(decision.candidate_count),
            mapping_source_sha256,
        )
        with self._transaction() as connection:
            source_exists = connection.execute(
                """
                SELECT 1
                FROM player_observations AS observation
                JOIN published_batches AS published USING (batch_id)
                WHERE observation.observation_id = ?
                """,
                (candidate.player_observation_id,),
            ).fetchone()
            if source_exists is None:
                raise TennisRatioStoreError("Identity remap source is not published.")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO identity_remap_observations (
                    remap_id, batch_id, player_observation_id,
                    source_player_key, gender, sackmann_player_id,
                    status, method, candidate_count, first_seen_at_utc,
                    source_url, source_sha256, mapping_source_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    remap_id,
                    batch_id,
                    candidate.player_observation_id,
                    candidate.source_player_key,
                    candidate.gender,
                    decision.sackmann_player_id,
                    decision.status,
                    decision.method,
                    decision.candidate_count,
                    observed,
                    candidate.source_url,
                    candidate.source_sha256,
                    mapping_source_sha256,
                ),
            )
            inserted = cursor.rowcount > 0
            if inserted and decision.status != "mapped":
                self._insert_quarantine(
                    connection,
                    batch_id=batch_id,
                    source_url=candidate.source_url,
                    source_sha256=candidate.source_sha256,
                    source_player_key=candidate.source_player_key,
                    reason=f"identity_remap_{decision.status}",
                    observed=observed,
                    payload={
                        "previous_status": candidate.current_status,
                        "previous_sackmann_player_id": candidate.current_sackmann_player_id,
                        "status": decision.status,
                        "method": decision.method,
                        "candidate_count": decision.candidate_count,
                        "mapping_source_sha256": mapping_source_sha256,
                    },
                )
        return inserted

    def publish_identity_remap_batch(
        self,
        *,
        batch_id: str,
        observed_at_utc: datetime,
        published_at_utc: datetime,
        master_sha256_by_gender: Mapping[Gender, str],
        remaps: int,
        canonical_fingerprint: str,
    ) -> None:
        """Atomically expose a GET-free remap batch after all events are staged."""

        if remaps < 0:
            raise ValueError("remaps cannot be negative.")
        if re.fullmatch(r"[0-9a-f]{64}", canonical_fingerprint) is None:
            raise ValueError("canonical_fingerprint must be lowercase SHA-256.")
        for gender in ("M", "F"):
            digest = master_sha256_by_gender.get(cast(Gender, gender))
            if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Missing valid {gender} Sackmann master SHA-256.")
        with self._transaction() as connection:
            refresh_collision = connection.execute(
                "SELECT 1 FROM refresh_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if refresh_collision is not None:
                raise TennisRatioStoreError("Remap batch id collides with a refresh batch.")
            connection.execute(
                """
                INSERT INTO identity_remap_batches (
                    batch_id, observed_at_utc, published_at_utc,
                    atp_master_sha256, wta_master_sha256, remaps,
                    canonical_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    _iso_instant(observed_at_utc),
                    _iso_instant(published_at_utc),
                    master_sha256_by_gender["M"],
                    master_sha256_by_gender["F"],
                    remaps,
                    canonical_fingerprint,
                ),
            )

    def record_profile_failure(
        self,
        *,
        batch_id: str,
        source_url: str,
        source_sha256: str | None,
        source_player_key: str,
        first_seen_at_utc: datetime,
        reason: str,
    ) -> None:
        """Append a per-profile failure without rolling back valid profiles."""

        with self._transaction() as connection:
            self._insert_quarantine(
                connection,
                batch_id=batch_id,
                source_url=source_url,
                source_sha256=source_sha256,
                source_player_key=source_player_key,
                reason="profile_acquisition_failure",
                observed=_iso_instant(first_seen_at_utc),
                payload={"error": reason[:2_000]},
            )

    def record_retention_events(
        self,
        *,
        batch_id: str,
        artifact_kind: str,
        paths: Sequence[Path],
        pruned_at_utc: datetime,
    ) -> None:
        """Append explicit tombstones for files removed by the retention contract."""

        if not paths:
            return
        observed = _iso_instant(pruned_at_utc)
        with self._transaction() as connection:
            for path in paths:
                resolved = str(Path(path).resolve())
                event_id = _stable_id("retention", batch_id, artifact_kind, resolved)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO retention_events (
                        event_id, batch_id, artifact_kind, artifact_path,
                        reason, pruned_at_utc
                    ) VALUES (?, ?, ?, ?, 'retention_limit_2', ?)
                    """,
                    (event_id, batch_id, artifact_kind, resolved, observed),
                )

    def complete_batch(
        self,
        *,
        batch_id: str,
        utc_date: date,
        retrieved_at_utc: datetime,
        status: str,
        agenda: pd.DataFrame,
        agenda_snapshot_ids: Mapping[str, str],
        sitemap_snapshot_id: str,
        inventory: Sequence[SitemapPlayer],
        inventory_source_sha256: str,
        canonical_fingerprint: str,
        profiles_selected: int,
        profiles_attempted: int,
        profiles_succeeded: int,
        profiles_failed: int,
        profiles_unchanged: int,
    ) -> None:
        """Atomically append validated agenda, inventory delta and batch summary."""

        if status not in {"published", "published_with_profile_failures"}:
            raise TennisRatioStoreError(f"Invalid completed batch status {status!r}.")
        observed = _iso_instant(retrieved_at_utc)
        records = agenda.to_dict(orient="records")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO refresh_batches (
                    batch_id, utc_date, retrieved_at_utc, status, agenda_rows,
                    inventory_entries, profiles_selected, profiles_attempted,
                    profiles_succeeded, profiles_failed, profiles_unchanged,
                    canonical_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    utc_date.isoformat(),
                    observed,
                    status,
                    len(records),
                    len(inventory),
                    profiles_selected,
                    profiles_attempted,
                    profiles_succeeded,
                    profiles_failed,
                    profiles_unchanged,
                    canonical_fingerprint,
                ),
            )
            for record in records:
                source_url = str(record["source_url"])
                snapshot_id = agenda_snapshot_ids[source_url]
                source_match_id = str(record["source_match_id"])
                observation_id = _stable_id("agenda", batch_id, source_match_id)
                connection.execute(
                    """
                    INSERT INTO agenda_observations (
                        observation_id, batch_id, snapshot_id, source_match_id,
                        match_date, first_seen_at_utc, source_url,
                        source_sha256, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        batch_id,
                        snapshot_id,
                        source_match_id,
                        _date_from_value(record["match_date"]).isoformat(),
                        observed,
                        source_url,
                        str(record["snapshot_sha256"]),
                        _json({key: _json_value(value) for key, value in record.items()}),
                    ),
                )
            for entry in inventory:
                exists = connection.execute(
                    """
                    SELECT 1
                    FROM profile_inventory_observations AS inventory
                    LEFT JOIN published_batches AS published USING (batch_id)
                    WHERE source_url = ?
                      AND COALESCE(lastmod, '') = COALESCE(?, '')
                      AND (inventory.batch_id = ? OR published.batch_id IS NOT NULL)
                    LIMIT 1
                    """,
                    (entry.source_url, _iso_date(entry.lastmod), batch_id),
                ).fetchone()
                if exists is not None:
                    continue
                observation_id = _stable_id(
                    "inventory",
                    batch_id,
                    entry.source_url,
                    _iso_date(entry.lastmod) or "missing",
                )
                connection.execute(
                    """
                    INSERT INTO profile_inventory_observations (
                        observation_id, batch_id, snapshot_id, source_url,
                        source_player_key, profile_slug, lastmod,
                        first_seen_at_utc, source_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        batch_id,
                        sitemap_snapshot_id,
                        entry.source_url,
                        entry.source_player_key,
                        entry.profile_slug,
                        _iso_date(entry.lastmod),
                        observed,
                        inventory_source_sha256,
                    ),
                )

    def mark_published(
        self,
        *,
        batch_id: str,
        published_at_utc: datetime,
        manifest_path: Path,
        manifest_sha256: str,
    ) -> None:
        """Append the event that makes a completed batch visible to readers."""

        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO published_batches (
                    batch_id, published_at_utc, manifest_path, manifest_sha256
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    batch_id,
                    _iso_instant(published_at_utc),
                    str(Path(manifest_path).resolve()),
                    manifest_sha256,
                ),
            )

    def canonical_fingerprint(self, agenda: pd.DataFrame, *, batch_id: str) -> str:
        """Hash current canonical source state without acquisition timestamps."""

        agenda_state = sorted(
            (
                str(row["source_match_id"]),
                _date_from_value(row["match_date"]).isoformat(),
                str(row["player_1_slug"]),
                str(row["player_2_slug"]),
                str(row["snapshot_sha256"]),
            )
            for row in agenda.to_dict(orient="records")
        )
        with self._connect() as connection:
            profile_state = connection.execute(
                """
                SELECT gender, source_player_key, source_sha256
                FROM (
                    SELECT gender, source_player_key, source_sha256,
                           ROW_NUMBER() OVER (
                               PARTITION BY gender, source_player_key
                               ORDER BY first_seen_at_utc DESC, observation_id DESC
                           ) AS position
                    FROM player_observations AS observation
                    WHERE observation.batch_id = ?
                       OR EXISTS (
                           SELECT 1 FROM published_batches AS published
                           WHERE published.batch_id = observation.batch_id
                       )
                )
                WHERE position = 1
                ORDER BY gender, source_player_key
                """,
                (batch_id,),
            ).fetchall()
        payload = {
            "agenda": agenda_state,
            "profiles": [(str(row[0]), str(row[1]), str(row[2])) for row in profile_state],
            "identities": self._fingerprint_identity_state(staging_batch_id=batch_id),
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

    def current_canonical_fingerprint(self) -> str:
        """Hash the latest globally published agenda, profiles and identities."""

        with self._connect() as connection:
            latest_batch = connection.execute(
                """
                SELECT batch_id FROM published_batches
                ORDER BY published_at_utc DESC, batch_id DESC
                LIMIT 1
                """
            ).fetchone()
            if latest_batch is None:
                agenda_state: list[tuple[str, str, str, str, str]] = []
            else:
                agenda_rows = connection.execute(
                    """
                    SELECT source_match_id, match_date, source_sha256, payload_json
                    FROM agenda_observations
                    WHERE batch_id = ?
                    ORDER BY source_match_id
                    """,
                    (str(latest_batch["batch_id"]),),
                ).fetchall()
                agenda_state = []
                for row in agenda_rows:
                    payload = cast(dict[str, object], json.loads(str(row["payload_json"])))
                    agenda_state.append(
                        (
                            str(row["source_match_id"]),
                            str(row["match_date"]),
                            str(payload["player_1_slug"]),
                            str(payload["player_2_slug"]),
                            str(row["source_sha256"]),
                        )
                    )
            profile_rows = connection.execute(
                """
                SELECT gender, source_player_key, source_sha256
                FROM (
                    SELECT observation.gender, observation.source_player_key,
                           observation.source_sha256,
                           ROW_NUMBER() OVER (
                               PARTITION BY observation.gender, observation.source_player_key
                               ORDER BY published.published_at_utc DESC,
                                        observation.first_seen_at_utc DESC,
                                        observation.observation_id DESC
                           ) AS position
                    FROM player_observations AS observation
                    JOIN published_batches AS published USING (batch_id)
                )
                WHERE position = 1
                ORDER BY gender, source_player_key
                """
            ).fetchall()
        payload = {
            "agenda": sorted(agenda_state),
            "profiles": [(str(row[0]), str(row[1]), str(row[2])) for row in profile_rows],
            "identities": self._fingerprint_identity_state(staging_batch_id=None),
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

    def canonical_fingerprint_with_staged_remaps(self, batch_id: str) -> str:
        """Hash current published source state plus one unpublished remap batch."""

        with self._connect() as connection:
            latest_batch = connection.execute(
                """
                SELECT batch_id FROM published_batches
                ORDER BY published_at_utc DESC, batch_id DESC
                LIMIT 1
                """
            ).fetchone()
            if latest_batch is None:
                agenda_state: list[tuple[str, str, str, str, str]] = []
            else:
                rows = connection.execute(
                    "SELECT payload_json FROM agenda_observations WHERE batch_id = ?",
                    (str(latest_batch["batch_id"]),),
                ).fetchall()
                agenda_state = []
                for row in rows:
                    payload = cast(dict[str, object], json.loads(str(row["payload_json"])))
                    agenda_state.append(
                        (
                            str(payload["source_match_id"]),
                            _date_from_value(payload["match_date"]).isoformat(),
                            str(payload["player_1_slug"]),
                            str(payload["player_2_slug"]),
                            str(payload["snapshot_sha256"]),
                        )
                    )
            profile_rows = connection.execute(
                """
                SELECT gender, source_player_key, source_sha256
                FROM (
                    SELECT observation.gender, observation.source_player_key,
                           observation.source_sha256,
                           ROW_NUMBER() OVER (
                               PARTITION BY observation.gender, observation.source_player_key
                               ORDER BY published.published_at_utc DESC,
                                        observation.first_seen_at_utc DESC,
                                        observation.observation_id DESC
                           ) AS position
                    FROM player_observations AS observation
                    JOIN published_batches AS published USING (batch_id)
                )
                WHERE position = 1
                ORDER BY gender, source_player_key
                """
            ).fetchall()
        payload = {
            "agenda": sorted(agenda_state),
            "profiles": [(str(row[0]), str(row[1]), str(row[2])) for row in profile_rows],
            "identities": self._fingerprint_identity_state(staging_batch_id=batch_id),
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

    def _fingerprint_identity_state(
        self,
        *,
        staging_batch_id: str | None,
    ) -> list[tuple[str, str, str, int | None, int, str]]:
        """Return latest decision signatures, optionally overlaying staged rows."""

        latest: dict[tuple[str, str], tuple[str, str, str, int | None, int, str]] = {}
        for row in self._resolution_rows(as_of_date=None, gender=None):
            key = (str(row["gender"]), str(row["source_player_key"]))
            latest[key] = (
                key[0],
                key[1],
                str(row["status"]),
                None if row["sackmann_player_id"] is None else int(row["sackmann_player_id"]),
                int(row["candidate_count"]),
                str(row["source_sha256"]),
            )
        if staging_batch_id is not None:
            with self._connect() as connection:
                remaps = connection.execute(
                    """
                    SELECT remap_id AS event_id, gender, source_player_key,
                           status, sackmann_player_id, candidate_count,
                           source_sha256, first_seen_at_utc
                    FROM identity_remap_observations
                    WHERE batch_id = ?
                    ORDER BY first_seen_at_utc, remap_id
                    """,
                    (staging_batch_id,),
                ).fetchall()
                initial = connection.execute(
                    """
                    SELECT resolution.resolution_id AS event_id,
                           resolution.gender, resolution.source_player_key,
                           resolution.status, resolution.sackmann_player_id,
                           resolution.candidate_count, resolution.source_sha256,
                           resolution.first_seen_at_utc
                    FROM identity_resolutions AS resolution
                    JOIN player_observations AS observation
                      ON observation.observation_id = resolution.player_observation_id
                    WHERE observation.batch_id = ?
                    ORDER BY resolution.first_seen_at_utc, resolution.resolution_id
                    """,
                    (staging_batch_id,),
                ).fetchall()
            for row in (*remaps, *initial):
                key = (str(row["gender"]), str(row["source_player_key"]))
                latest[key] = (
                    key[0],
                    key[1],
                    str(row["status"]),
                    None if row["sackmann_player_id"] is None else int(row["sackmann_player_id"]),
                    int(row["candidate_count"]),
                    str(row["source_sha256"]),
                )
        return [latest[key] for key in sorted(latest)]

    def load_active_agenda(self, match_date: date) -> pd.DataFrame:
        """Load one date from the newest published last-good batch, offline."""

        from .parser import empty_agenda_dataframe  # Avoid parser/store import cycle.

        with self._connect() as connection:
            batch = connection.execute(
                """
                SELECT batch_id
                FROM published_batches
                ORDER BY published_at_utc DESC, batch_id DESC
                LIMIT 1
                """
            ).fetchone()
            if batch is None:
                return empty_agenda_dataframe()
            rows = connection.execute(
                """
                SELECT payload_json FROM agenda_observations
                WHERE batch_id = ? AND match_date = ?
                ORDER BY source_match_id
                """,
                (str(batch[0]), match_date.isoformat()),
            ).fetchall()
        payloads = [cast(dict[str, object], json.loads(str(row[0]))) for row in rows]
        columns = AGENDA_OUTPUT_COLUMNS + AGENDA_AUDIT_COLUMNS
        frame = pd.DataFrame.from_records(payloads, columns=columns)
        from .parser import _agenda_frame  # Centralizes the public dtype contract.

        return _agenda_frame(frame.to_dict(orient="records"))

    def load_mapped_rankings(
        self,
        as_of_date: date,
        *,
        gender: Gender | None = None,
    ) -> tuple[MappedRanking, ...]:
        """Return latest rankings with effective and available dates both ``< D``."""

        _validate_gender(gender)
        observations = self._eligible_player_rows(as_of_date, gender=gender)
        resolutions = self._latest_resolutions(as_of_date, gender=gender)
        latest: dict[tuple[Gender, str], sqlite3.Row] = {}
        for row in observations:
            row_gender = cast(Gender, str(row["gender"]))
            source_key = str(row["source_player_key"])
            key = (row_gender, source_key)
            previous = latest.get(key)
            rank_key = (
                str(row["effective_date"]),
                str(row["first_seen_at_utc"]),
                str(row["observation_id"]),
            )
            if previous is None or rank_key > (
                str(previous["effective_date"]),
                str(previous["first_seen_at_utc"]),
                str(previous["observation_id"]),
            ):
                latest[key] = row
        output: list[MappedRanking] = []
        for (row_gender, source_key), row in latest.items():
            resolution = resolutions.get((row_gender, source_key))
            if resolution is None or resolution["status"] != "mapped" or row["ranking"] is None:
                continue
            source_seen = max(
                _parse_instant(str(row["first_seen_at_utc"])),
                _parse_instant(str(row["batch_published_at_utc"])),
            )
            mapping_seen = max(
                _parse_instant(str(resolution["first_seen_at_utc"])),
                _parse_instant(str(resolution["observation_first_seen_at_utc"])),
                _parse_instant(str(resolution["observation_batch_published_at_utc"])),
                _parse_instant(str(resolution["resolution_batch_published_at_utc"])),
            )
            available = max(source_seen, mapping_seen).date()
            if available >= as_of_date:
                continue
            output.append(
                MappedRanking(
                    gender=row_gender,
                    sackmann_player_id=int(resolution["sackmann_player_id"]),
                    source_player_key=source_key,
                    player_name=str(row["player_name"]),
                    rank=int(row["ranking"]),
                    effective_date=date.fromisoformat(str(row["effective_date"])),
                    available_date=available,
                    first_seen_at_utc=max(source_seen, mapping_seen),
                    source_url=str(row["source_url"]),
                    source_sha256=str(row["source_sha256"]),
                )
            )
        return tuple(sorted(output, key=lambda item: (item.gender, item.sackmann_player_id)))

    def load_identity_resolutions(
        self,
        *,
        as_of_date: date | None = None,
        gender: Gender | None = None,
    ) -> tuple[MappedIdentity, ...]:
        """Return the latest unique mapped identity for each source profile.

        Passing ``as_of_date=D`` applies a strict civil-date cutoff to the
        resolution, its source observation and publication batch.  Passing
        ``None`` returns current presentation state only; callers must never
        use that shortcut for historical serving or feature construction.
        """

        _validate_gender(gender)
        rows = self._resolution_rows(as_of_date=as_of_date, gender=gender)
        latest = {
            (cast(Gender, str(row["gender"])), str(row["source_player_key"])): row for row in rows
        }
        output = [
            MappedIdentity(
                gender=row_gender,
                source_player_key=key,
                sackmann_player_id=int(row["sackmann_player_id"]),
                first_seen_at_utc=max(
                    _parse_instant(str(row["first_seen_at_utc"])),
                    _parse_instant(str(row["observation_first_seen_at_utc"])),
                    _parse_instant(str(row["observation_batch_published_at_utc"])),
                    _parse_instant(str(row["resolution_batch_published_at_utc"])),
                ),
                source_url=str(row["source_url"]),
                source_sha256=str(row["source_sha256"]),
            )
            for (row_gender, key), row in latest.items()
            if row["status"] == "mapped" and row["sackmann_player_id"] is not None
        ]
        return tuple(sorted(output, key=lambda item: (item.gender, item.source_player_key)))

    def load_agenda_as_of(
        self,
        as_of_date: date,
        *,
        match_date: date | None = None,
    ) -> pd.DataFrame:
        """Return latest per-match agenda evidence published strictly before D."""

        from .parser import _agenda_frame, empty_agenda_dataframe

        query = """
            SELECT agenda.source_match_id, agenda.payload_json,
                   agenda.first_seen_at_utc, published.published_at_utc,
                   agenda.observation_id
            FROM agenda_observations AS agenda
            JOIN published_batches AS published USING (batch_id)
            WHERE substr(agenda.first_seen_at_utc, 1, 10) < ?
              AND substr(published.published_at_utc, 1, 10) < ?
        """
        parameters: list[object] = [as_of_date.isoformat(), as_of_date.isoformat()]
        if match_date is not None:
            query += " AND agenda.match_date = ?"
            parameters.append(match_date.isoformat())
        query += (
            " ORDER BY published.published_at_utc, agenda.first_seen_at_utc, agenda.observation_id"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        if not rows:
            return empty_agenda_dataframe()
        latest = {str(row["source_match_id"]): row for row in rows}
        payloads = [
            cast(dict[str, object], json.loads(str(latest[key]["payload_json"])))
            for key in sorted(latest)
        ]
        return _agenda_frame(payloads)

    def load_mapped_match_stats(
        self,
        as_of_date: date,
        *,
        gender: Gender | None = None,
    ) -> tuple[MappedMatchStats, ...]:
        """Return latest per-player match stats strictly known before ``D``.

        Repeated profile snapshots do not reset availability when the stat
        payload is unchanged.  A changed payload is a new version and becomes
        usable only after that revision was first observed and published.
        """

        from .parser import _parse_profile_match_stats

        _validate_gender(gender)
        resolutions = self._latest_resolutions(as_of_date, gender=gender)
        query = """
            SELECT observation.*,
                   published.published_at_utc AS batch_published_at_utc
            FROM match_observations AS observation
            JOIN published_batches AS published USING (batch_id)
            WHERE observation.effective_date < ?
              AND substr(observation.first_seen_at_utc, 1, 10) < ?
              AND substr(published.published_at_utc, 1, 10) < ?
        """
        parameters: list[object] = [as_of_date.isoformat()] * 3
        if gender is not None:
            query += " AND observation.gender = ?"
            parameters.append(gender)
        query += (
            " ORDER BY observation.first_seen_at_utc, "
            "observation.batch_id, observation.observation_id"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        versions: dict[
            tuple[Gender, str, str],
            list[tuple[sqlite3.Row, ProfileMatchStats, datetime]],
        ] = defaultdict(list)
        for row in rows:
            payload = cast(dict[str, object], json.loads(str(row["payload_json"])))
            stats = _parse_profile_match_stats(payload)
            if stats is None:
                continue
            row_gender = cast(Gender, str(row["gender"]))
            key = (
                row_gender,
                str(row["source_player_key"]),
                str(row["source_match_key"]),
            )
            source_seen = max(
                _parse_instant(str(row["first_seen_at_utc"])),
                _parse_instant(str(row["batch_published_at_utc"])),
            )
            versions[key].append((row, stats, source_seen))

        output: list[MappedMatchStats] = []
        for (row_gender, source_key, _), observations in versions.items():
            row, stats_object, _ = observations[-1]
            resolution = resolutions.get((row_gender, source_key))
            if resolution is None or resolution["status"] != "mapped":
                continue
            first_stat_seen = min(
                seen for _, candidate_stats, seen in observations if candidate_stats == stats_object
            )
            mapping_seen = max(
                _parse_instant(str(resolution["first_seen_at_utc"])),
                _parse_instant(str(resolution["observation_first_seen_at_utc"])),
                _parse_instant(str(resolution["observation_batch_published_at_utc"])),
                _parse_instant(str(resolution["resolution_batch_published_at_utc"])),
            )
            available = max(first_stat_seen, mapping_seen)
            if available.date() >= as_of_date:
                continue
            output.append(
                MappedMatchStats(
                    gender=row_gender,
                    sackmann_player_id=int(resolution["sackmann_player_id"]),
                    source_player_key=source_key,
                    effective_date=date.fromisoformat(str(row["effective_date"])),
                    available_date=available.date(),
                    first_seen_at_utc=available,
                    surface=str(row["surface"]),
                    tour_level=str(row["tour_level"]),
                    stats=stats_object,
                    source_url=str(row["source_url"]),
                    source_sha256=str(row["source_sha256"]),
                )
            )
        return tuple(
            sorted(
                output,
                key=lambda item: (
                    item.gender,
                    item.sackmann_player_id,
                    item.effective_date,
                    item.source_sha256,
                ),
            )
        )

    def load_mapped_results(
        self,
        as_of_date: date,
        *,
        base_cutoff_by_gender: Mapping[str, date],
        gender: Gender | None = None,
    ) -> tuple[MappedResult, ...]:
        """Return deduplicated results strictly available before ``as_of_date``.

        Results at or before the caller's Sackmann cutoff are excluded so an
        overlay cannot double-count its immutable historical base.
        """

        _validate_gender(gender)
        required_genders: tuple[Gender, ...] = ("M", "F") if gender is None else (gender,)
        for required in required_genders:
            if required not in base_cutoff_by_gender or not isinstance(
                base_cutoff_by_gender[required], date
            ):
                raise ValueError(
                    "base_cutoff_by_gender must contain a date for every requested gender."
                )
        resolutions = self._latest_resolutions(as_of_date, gender=gender)
        query = """
            SELECT observation.*,
                   published.published_at_utc AS batch_published_at_utc
            FROM match_observations AS observation
            JOIN published_batches AS published USING (batch_id)
            WHERE observation.effective_date < ?
              AND substr(observation.first_seen_at_utc, 1, 10) < ?
              AND substr(published.published_at_utc, 1, 10) < ?
        """
        parameters: list[object] = [as_of_date.isoformat()] * 3
        if gender is not None:
            query += " AND observation.gender = ? AND observation.effective_date > ?"
            parameters.extend((gender, base_cutoff_by_gender[gender].isoformat()))
        else:
            query += (
                " AND ((observation.gender = 'M' "
                "AND observation.effective_date > ?) "
                "OR (observation.gender = 'F' "
                "AND observation.effective_date > ?))"
            )
            parameters.extend(
                (
                    base_cutoff_by_gender["M"].isoformat(),
                    base_cutoff_by_gender["F"].isoformat(),
                )
            )
        query += (
            " ORDER BY observation.effective_date, observation.source_match_key, "
            "observation.first_seen_at_utc"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        candidates: dict[str, list[MappedResult]] = defaultdict(list)
        for row in rows:
            row_gender = cast(Gender, str(row["gender"]))
            effective = date.fromisoformat(str(row["effective_date"]))
            if effective <= base_cutoff_by_gender[row_gender]:
                continue
            subject_key = str(row["source_player_key"])
            rival_raw = row["rival_source_key"]
            if rival_raw is None:
                continue
            rival_key = str(rival_raw)
            subject_resolution = resolutions.get((row_gender, subject_key))
            rival_resolution = resolutions.get((row_gender, rival_key))
            if (
                subject_resolution is None
                or rival_resolution is None
                or subject_resolution["status"] != "mapped"
                or rival_resolution["status"] != "mapped"
            ):
                continue
            match_seen = max(
                _parse_instant(str(row["first_seen_at_utc"])),
                _parse_instant(str(row["batch_published_at_utc"])),
            )
            subject_seen = max(
                _parse_instant(str(subject_resolution["first_seen_at_utc"])),
                _parse_instant(str(subject_resolution["observation_first_seen_at_utc"])),
                _parse_instant(str(subject_resolution["observation_batch_published_at_utc"])),
                _parse_instant(str(subject_resolution["resolution_batch_published_at_utc"])),
            )
            rival_seen = max(
                _parse_instant(str(rival_resolution["first_seen_at_utc"])),
                _parse_instant(str(rival_resolution["observation_first_seen_at_utc"])),
                _parse_instant(str(rival_resolution["observation_batch_published_at_utc"])),
                _parse_instant(str(rival_resolution["resolution_batch_published_at_utc"])),
            )
            available_instant = max(match_seen, subject_seen, rival_seen)
            if available_instant.date() >= as_of_date:
                continue
            subject_id = int(subject_resolution["sackmann_player_id"])
            rival_id = int(rival_resolution["sackmann_player_id"])
            if str(row["result"]) == "Win":
                winner_id, loser_id = subject_id, rival_id
                winner_key, loser_key = subject_key, rival_key
                winner_name, loser_name = str(row["player_name"]), str(row["rival_name"])
            else:
                winner_id, loser_id = rival_id, subject_id
                winner_key, loser_key = rival_key, subject_key
                winner_name, loser_name = str(row["rival_name"]), str(row["player_name"])
            candidates[str(row["source_match_key"])].append(
                MappedResult(
                    canonical_match_id=str(row["source_match_key"]),
                    gender=row_gender,
                    effective_date=effective,
                    available_date=available_instant.date(),
                    first_seen_at_utc=available_instant,
                    tournament=str(row["tournament"]),
                    tour_level=str(row["tour_level"]),
                    surface=str(row["surface"]),
                    round=str(row["round_name"]),
                    winner_sackmann_id=winner_id,
                    loser_sackmann_id=loser_id,
                    winner_source_key=winner_key,
                    loser_source_key=loser_key,
                    winner_name=winner_name,
                    loser_name=loser_name,
                    score=None
                    if row["score_winner_perspective"] is None
                    else str(row["score_winner_perspective"]),
                    winner_sets_won=None
                    if row["winner_sets_won"] is None
                    else int(row["winner_sets_won"]),
                    loser_sets_won=None
                    if row["loser_sets_won"] is None
                    else int(row["loser_sets_won"]),
                    source_url=str(row["source_url"]),
                    source_sha256=str(row["source_sha256"]),
                )
            )

        result: list[MappedResult] = []
        for source_match_key, group in candidates.items():
            signatures = {
                (
                    item.gender,
                    item.effective_date,
                    _normalize(item.tournament),
                    item.tour_level,
                    item.surface,
                    _normalize(item.round),
                    item.winner_sackmann_id,
                    item.loser_sackmann_id,
                )
                for item in group
            }
            scores = {item.score for item in group if item.score is not None}
            if len(signatures) != 1 or len(scores) > 1:
                continue
            selected = min(
                group,
                key=lambda item: (
                    item.available_date,
                    item.first_seen_at_utc,
                    item.source_url,
                ),
            )
            if selected.score is None and scores:
                selected = next(item for item in group if item.score is not None)
            if selected.canonical_match_id != source_match_key:
                raise TennisRatioStoreError("Canonical result key changed during deduplication.")
            result.append(selected)
        return tuple(
            sorted(
                result,
                key=lambda item: (item.effective_date, item.canonical_match_id),
            )
        )

    def _eligible_player_rows(
        self,
        as_of_date: date,
        *,
        gender: Gender | None,
    ) -> list[sqlite3.Row]:
        """Read player facts satisfying both strict daily cutoffs."""

        query = """
            SELECT observation.*,
                   published.published_at_utc AS batch_published_at_utc
            FROM player_observations AS observation
            JOIN published_batches AS published USING (batch_id)
            WHERE observation.effective_date < ?
              AND substr(observation.first_seen_at_utc, 1, 10) < ?
              AND substr(published.published_at_utc, 1, 10) < ?
        """
        parameters: list[object] = [as_of_date.isoformat()] * 3
        if gender is not None:
            query += " AND observation.gender = ?"
            parameters.append(gender)
        with self._connect() as connection:
            return list(connection.execute(query, parameters).fetchall())

    def _latest_resolutions(
        self,
        as_of_date: date,
        *,
        gender: Gender | None,
    ) -> dict[tuple[Gender, str], sqlite3.Row]:
        """Select the latest identity decision that was available before D."""

        rows = self._resolution_rows(as_of_date=as_of_date, gender=gender)
        return {
            (cast(Gender, str(row["gender"])), str(row["source_player_key"])): row for row in rows
        }

    def _resolution_rows(
        self,
        *,
        as_of_date: date | None,
        gender: Gender | None,
    ) -> list[sqlite3.Row]:
        """Union initial and remapped decisions through their publication gates."""

        _validate_gender(gender)
        query = """
            WITH resolution_events AS (
                SELECT resolution.resolution_id AS resolution_id,
                       resolution.player_observation_id AS player_observation_id,
                       resolution.source_player_key AS source_player_key,
                       resolution.gender AS gender,
                       resolution.sackmann_player_id AS sackmann_player_id,
                       resolution.status AS status,
                       resolution.method AS method,
                       resolution.candidate_count AS candidate_count,
                       resolution.first_seen_at_utc AS first_seen_at_utc,
                       resolution.source_url AS source_url,
                       resolution.source_sha256 AS source_sha256,
                       NULL AS mapping_source_sha256,
                       0 AS event_priority,
                       observation.first_seen_at_utc AS observation_first_seen_at_utc,
                       observation_published.published_at_utc
                           AS observation_batch_published_at_utc,
                       observation_published.published_at_utc
                           AS resolution_batch_published_at_utc
                FROM identity_resolutions AS resolution
                JOIN player_observations AS observation
                  ON observation.observation_id = resolution.player_observation_id
                JOIN published_batches AS observation_published
                  ON observation_published.batch_id = observation.batch_id

                UNION ALL

                SELECT remap.remap_id AS resolution_id,
                       remap.player_observation_id AS player_observation_id,
                       remap.source_player_key AS source_player_key,
                       remap.gender AS gender,
                       remap.sackmann_player_id AS sackmann_player_id,
                       remap.status AS status,
                       remap.method AS method,
                       remap.candidate_count AS candidate_count,
                       remap.first_seen_at_utc AS first_seen_at_utc,
                       remap.source_url AS source_url,
                       remap.source_sha256 AS source_sha256,
                       remap.mapping_source_sha256 AS mapping_source_sha256,
                       1 AS event_priority,
                       observation.first_seen_at_utc AS observation_first_seen_at_utc,
                       observation_published.published_at_utc
                           AS observation_batch_published_at_utc,
                       COALESCE(
                           refresh_remap_published.published_at_utc,
                           standalone_remap_published.published_at_utc
                       )
                           AS resolution_batch_published_at_utc
                FROM identity_remap_observations AS remap
                JOIN player_observations AS observation
                  ON observation.observation_id = remap.player_observation_id
                JOIN published_batches AS observation_published
                  ON observation_published.batch_id = observation.batch_id
                LEFT JOIN published_batches AS refresh_remap_published
                  ON refresh_remap_published.batch_id = remap.batch_id
                LEFT JOIN identity_remap_batches AS standalone_remap_published
                  ON standalone_remap_published.batch_id = remap.batch_id
                WHERE refresh_remap_published.batch_id IS NOT NULL
                   OR standalone_remap_published.batch_id IS NOT NULL
            )
            SELECT * FROM resolution_events
        """
        parameters: list[object] = []
        conditions: list[str] = []
        if as_of_date is not None:
            conditions.extend(
                (
                    "substr(first_seen_at_utc, 1, 10) < ?",
                    "substr(observation_first_seen_at_utc, 1, 10) < ?",
                    "substr(observation_batch_published_at_utc, 1, 10) < ?",
                    "substr(resolution_batch_published_at_utc, 1, 10) < ?",
                )
            )
            parameters.extend([as_of_date.isoformat()] * 4)
        if gender is not None:
            conditions.append("gender = ?")
            parameters.append(gender)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += (
            " ORDER BY resolution_batch_published_at_utc, first_seen_at_utc, "
            "event_priority, resolution_id"
        )
        with self._connect() as connection:
            return list(connection.execute(query, parameters).fetchall())

    def _insert_quarantine(
        self,
        connection: sqlite3.Connection,
        *,
        batch_id: str,
        source_url: str,
        source_sha256: str | None,
        source_player_key: str | None,
        reason: str,
        observed: str,
        payload: dict[str, object],
    ) -> None:
        """Append one structured quarantine event inside an existing transaction."""

        event_id = _stable_id(
            "quarantine",
            batch_id,
            source_url,
            source_sha256 or "missing",
            source_player_key or "missing",
            reason,
            _json(payload),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO quarantine_events (
                event_id, batch_id, source_url, source_sha256,
                source_player_key, reason, first_seen_at_utc, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                batch_id,
                source_url,
                source_sha256,
                source_player_key,
                reason,
                observed,
                _json(payload),
            ),
        )

    def _initialize(self) -> None:
        """Create the versioned schema and immutable-table triggers."""

        with self._connect() as connection:
            connection.executescript(_SCHEMA_SQL)
            row = connection.execute(
                "SELECT version FROM schema_versions ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_versions(version, applied_at_utc) VALUES (?, ?)",
                    (_SCHEMA_VERSION, _iso_instant(datetime.now(UTC))),
                )
            elif 1 <= int(row[0]) < _SCHEMA_VERSION:
                connection.execute(
                    "INSERT INTO schema_versions(version, applied_at_utc) VALUES (?, ?)",
                    (_SCHEMA_VERSION, _iso_instant(datetime.now(UTC))),
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise TennisRatioStoreError(f"Unsupported TennisRatio schema version {row[0]}.")
            for table in _IMMUTABLE_TABLES:
                connection.executescript(_immutable_trigger_sql(table))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one configured SQLite connection."""

        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Open an immediate transaction and roll it back on every exception."""

        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise


def _profile_match_key(profile: ParsedProfile, item: Any) -> str:
    """Build an orientation-independent match id from public source identities."""

    rival_key = item.rival_source_key or f"unmapped-name:{_normalize(item.rival_name)}"
    players = sorted((profile.source_player_key, rival_key))
    digest = hashlib.sha256(
        "|".join(
            (
                profile.gender,
                item.effective_date.isoformat(),
                _normalize(item.tournament),
                _normalize(item.round),
                players[0],
                players[1],
            )
        ).encode("utf-8")
    ).hexdigest()
    return f"tennisratio-result:{digest}"


def _match_conflict_payload(
    connection: sqlite3.Connection,
    *,
    profile: ParsedProfile,
    item: Any,
    source_match_key: str,
) -> dict[str, object] | None:
    """Detect contradictory unilateral/bilateral evidence before publication."""

    if item.rival_source_key is None:
        return None
    if item.result == "Win":
        incoming_winner = profile.source_player_key
        incoming_loser = item.rival_source_key
    else:
        incoming_winner = item.rival_source_key
        incoming_loser = profile.source_player_key
    incoming_core = (
        profile.gender,
        item.effective_date.isoformat(),
        _normalize(item.tournament),
        item.tour_level,
        item.surface,
        _normalize(item.round),
        incoming_winner,
        incoming_loser,
    )
    existing_rows = connection.execute(
        "SELECT * FROM match_observations WHERE source_match_key = ?",
        (source_match_key,),
    ).fetchall()
    for row in existing_rows:
        existing_subject = str(row["source_player_key"])
        existing_rival_raw = row["rival_source_key"]
        if existing_rival_raw is None:
            continue
        existing_rival = str(existing_rival_raw)
        if str(row["result"]) == "Win":
            existing_winner, existing_loser = existing_subject, existing_rival
        else:
            existing_winner, existing_loser = existing_rival, existing_subject
        existing_core = (
            str(row["gender"]),
            str(row["effective_date"]),
            _normalize(str(row["tournament"])),
            str(row["tour_level"]),
            str(row["surface"]),
            _normalize(str(row["round_name"])),
            existing_winner,
            existing_loser,
        )
        existing_score = (
            None
            if row["score_winner_perspective"] is None
            else str(row["score_winner_perspective"])
        )
        scores_conflict = (
            existing_score is not None
            and item.score_winner_perspective is not None
            and existing_score != item.score_winner_perspective
        )
        if existing_core != incoming_core or scores_conflict:
            return {
                "source_match_key": source_match_key,
                "existing_observation_id": str(row["observation_id"]),
                "existing_subject": existing_subject,
                "incoming_subject": profile.source_player_key,
                "existing_winner": existing_winner,
                "incoming_winner": incoming_winner,
                "existing_score_winner_perspective": existing_score,
                "incoming_score_winner_perspective": item.score_winner_perspective,
            }
    return None


def _stable_id(namespace: str, *parts: str) -> str:
    """Create a deterministic opaque id for append-only idempotence."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"tennisratio-{namespace}:{digest}"


def _normalize(value: str) -> str:
    """Apply exact accent/punctuation normalization, never fuzzy matching."""

    normalized = _NORMALIZE_RE.sub(" ", unidecode(value).casefold())
    return " ".join(normalized.split())


def _json(value: object) -> str:
    """Serialize deterministic, finite JSON for hashes and audit payloads."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: object) -> object:
    """Convert pandas/datetime nullable values to standard JSON scalars."""

    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
        return value.isoformat()
    if isinstance(value, datetime):
        return _iso_instant(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _date_from_value(value: object) -> date:
    """Normalize a DataFrame date cell to a calendar date."""

    parsed = pd.Timestamp(value)
    return parsed.date()


def _iso_date(value: date | None) -> str | None:
    """Serialize an optional calendar date."""

    return None if value is None else value.isoformat()


def _iso_instant(value: datetime) -> str:
    """Serialize an aware datetime canonically in UTC."""

    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_instant(value: str) -> datetime:
    """Parse a canonical stored UTC timestamp."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise TennisRatioStoreError("Stored timestamp is unexpectedly naive.")
    return parsed.astimezone(UTC)


def _validate_gender(gender: Gender | None) -> None:
    """Validate an optional public gender selector."""

    if gender not in {None, "M", "F"}:
        raise ValueError("gender must be 'M', 'F' or None.")


def _immutable_trigger_sql(table: str) -> str:
    """Create update/delete blockers for one append-only table."""

    return f"""
    CREATE TRIGGER IF NOT EXISTS {table}_reject_update
    BEFORE UPDATE ON {table}
    BEGIN
        SELECT RAISE(ABORT, '{table} is append-only');
    END;
    CREATE TRIGGER IF NOT EXISTS {table}_reject_delete
    BEFORE DELETE ON {table}
    BEGIN
        SELECT RAISE(ABORT, '{table} is append-only');
    END;
    """


_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_batches (
    batch_id TEXT PRIMARY KEY,
    utc_date TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('published', 'published_with_profile_failures')),
    agenda_rows INTEGER NOT NULL CHECK(agenda_rows >= 0),
    inventory_entries INTEGER NOT NULL CHECK(inventory_entries >= 0),
    profiles_selected INTEGER NOT NULL CHECK(profiles_selected >= 0),
    profiles_attempted INTEGER NOT NULL CHECK(profiles_attempted >= 0),
    profiles_succeeded INTEGER NOT NULL CHECK(profiles_succeeded >= 0),
    profiles_failed INTEGER NOT NULL CHECK(profiles_failed >= 0),
    profiles_unchanged INTEGER NOT NULL CHECK(profiles_unchanged >= 0),
    canonical_fingerprint TEXT NOT NULL CHECK(length(canonical_fingerprint) = 64)
);

CREATE TABLE IF NOT EXISTS published_batches (
    batch_id TEXT PRIMARY KEY REFERENCES refresh_batches(batch_id),
    published_at_utc TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64)
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    resource_kind TEXT NOT NULL,
    source_url TEXT NOT NULL,
    first_seen_at_utc TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    byte_size INTEGER NOT NULL CHECK(byte_size > 0),
    content_type TEXT NOT NULL,
    compressed_path TEXT NOT NULL,
    http_etag TEXT,
    http_last_modified TEXT,
    UNIQUE(source_url, sha256)
);

CREATE TABLE IF NOT EXISTS agenda_observations (
    observation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES refresh_batches(batch_id),
    snapshot_id TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    source_match_id TEXT NOT NULL,
    match_date TEXT NOT NULL,
    first_seen_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    payload_json TEXT NOT NULL,
    UNIQUE(batch_id, source_match_id)
);

CREATE TABLE IF NOT EXISTS profile_inventory_observations (
    observation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES refresh_batches(batch_id),
    snapshot_id TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    source_url TEXT NOT NULL,
    source_player_key TEXT NOT NULL,
    profile_slug TEXT NOT NULL,
    lastmod TEXT,
    first_seen_at_utc TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    CHECK(length(source_player_key) > 0)
);

CREATE TABLE IF NOT EXISTS player_observations (
    observation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    source_player_key TEXT NOT NULL,
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    player_name TEXT NOT NULL,
    dob TEXT,
    ranking INTEGER CHECK(ranking IS NULL OR ranking > 0),
    country TEXT,
    hand TEXT,
    effective_date TEXT NOT NULL,
    sitemap_lastmod TEXT,
    first_seen_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_sync_observations (
    observation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_player_key TEXT NOT NULL,
    lastmod TEXT,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    status TEXT NOT NULL CHECK(status IN ('accepted', 'unchanged')),
    first_seen_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_observations (
    observation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES source_snapshots(snapshot_id),
    source_match_key TEXT NOT NULL,
    source_player_key TEXT NOT NULL,
    rival_source_key TEXT,
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    effective_date TEXT NOT NULL,
    tournament TEXT NOT NULL,
    tour_level TEXT NOT NULL,
    surface TEXT NOT NULL,
    round_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    rival_name TEXT NOT NULL,
    result TEXT NOT NULL CHECK(result IN ('Win', 'Lose')),
    score_profile_perspective TEXT,
    score_winner_perspective TEXT,
    winner_sets_won INTEGER,
    loser_sets_won INTEGER,
    player_odd REAL,
    rival_odd REAL,
    first_seen_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    payload_json TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS match_statistics_v1 AS
SELECT
    observation_id,
    batch_id,
    source_match_key,
    source_player_key,
    rival_source_key,
    gender,
    effective_date,
    surface,
    tour_level,
    first_seen_at_utc,
    source_url,
    source_sha256,
    json_extract(payload_json, '$.rival_rank') AS rival_rank,
    json_extract(payload_json, '$.first_serve_accuracy') AS first_serve_accuracy_pct,
    json_extract(payload_json, '$.first_serve_points') AS first_serve_points_won_pct,
    json_extract(payload_json, '$.second_serve_points') AS second_serve_points_won_pct,
    json_extract(payload_json, '$.aces') AS aces,
    json_extract(payload_json, '$.double_faults') AS double_faults,
    NULLIF(json_extract(payload_json, '$.breakpoints_saved_count'), '') AS breakpoints_saved,
    NULLIF(json_extract(payload_json, '$.service_games_won'), '') AS service_games_won,
    json_extract(payload_json, '$.return_1st_serve_points') AS return_first_serve_points_won_pct,
    json_extract(payload_json, '$.return_2nd_serve_points') AS return_second_serve_points_won_pct,
    NULLIF(json_extract(payload_json, '$.breakpoints_converted_count'), '') AS breakpoints_converted,
    NULLIF(json_extract(payload_json, '$.return_games_won'), '') AS return_games_won,
    json_extract(payload_json, '$.serve_pressure_all') AS serve_pressure_points,
    json_extract(payload_json, '$.serve_pressure_won') AS serve_pressure_points_won,
    json_extract(payload_json, '$.return_pressure_all') AS return_pressure_points,
    json_extract(payload_json, '$.return_pressure_won') AS return_pressure_points_won
FROM match_observations;

CREATE TABLE IF NOT EXISTS identity_resolutions (
    resolution_id TEXT PRIMARY KEY,
    player_observation_id TEXT NOT NULL REFERENCES player_observations(observation_id),
    source_player_key TEXT NOT NULL,
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    sackmann_player_id INTEGER,
    status TEXT NOT NULL CHECK(status IN ('mapped', 'unmapped', 'ambiguous')),
    method TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0),
    first_seen_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    CHECK((status = 'mapped' AND sackmann_player_id IS NOT NULL) OR
          (status != 'mapped' AND sackmann_player_id IS NULL))
);

CREATE TABLE IF NOT EXISTS identity_remap_observations (
    remap_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    player_observation_id TEXT NOT NULL REFERENCES player_observations(observation_id),
    source_player_key TEXT NOT NULL,
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    sackmann_player_id INTEGER,
    status TEXT NOT NULL CHECK(status IN ('mapped', 'unmapped', 'ambiguous')),
    method TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0),
    first_seen_at_utc TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    mapping_source_sha256 TEXT NOT NULL CHECK(length(mapping_source_sha256) = 64),
    CHECK((status = 'mapped' AND sackmann_player_id IS NOT NULL) OR
          (status != 'mapped' AND sackmann_player_id IS NULL))
);

CREATE TABLE IF NOT EXISTS identity_remap_batches (
    batch_id TEXT PRIMARY KEY,
    observed_at_utc TEXT NOT NULL,
    published_at_utc TEXT NOT NULL,
    atp_master_sha256 TEXT NOT NULL CHECK(length(atp_master_sha256) = 64),
    wta_master_sha256 TEXT NOT NULL CHECK(length(wta_master_sha256) = 64),
    remaps INTEGER NOT NULL CHECK(remaps >= 0),
    canonical_fingerprint TEXT NOT NULL CHECK(length(canonical_fingerprint) = 64)
);

CREATE TABLE IF NOT EXISTS quarantine_events (
    event_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT,
    source_player_key TEXT,
    reason TEXT NOT NULL,
    first_seen_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retention_events (
    event_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    reason TEXT NOT NULL CHECK(reason = 'retention_limit_2'),
    pruned_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS agenda_date_batch_idx
    ON agenda_observations(match_date, batch_id);
CREATE INDEX IF NOT EXISTS player_key_available_idx
    ON player_observations(source_player_key, effective_date, first_seen_at_utc);
CREATE INDEX IF NOT EXISTS match_effective_available_idx
    ON match_observations(effective_date, first_seen_at_utc, source_match_key);
CREATE INDEX IF NOT EXISTS match_source_key_idx
    ON match_observations(source_match_key);
CREATE INDEX IF NOT EXISTS resolution_key_available_idx
    ON identity_resolutions(source_player_key, first_seen_at_utc);
CREATE INDEX IF NOT EXISTS remap_key_available_idx
    ON identity_remap_observations(source_player_key, first_seen_at_utc);
"""

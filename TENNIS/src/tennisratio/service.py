"""UTC-daily orchestration and causal public APIs for TennisRatio.

The service publishes only after robots, both agendas and the official player
sitemap have validated.  Every profile is then handled as an independent
transaction: one malformed profile is quarantined and cannot roll back valid
profiles or the last known good observation for that player.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager
import ctypes
import csv
from datetime import UTC, date, datetime
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Final, Iterator, cast
import uuid

import pandas as pd
import requests
from unidecode import unidecode

from .client import TennisRatioClient
from .parser import (
    parse_agenda_html,
    parse_player_sitemap,
    parse_profile_html,
    parse_robots_txt,
    profile_slug_from_url,
    source_player_key,
)
from .store import TennisRatioStore
from .types import (
    ATTRIBUTION,
    LICENSE_URL,
    Gender,
    HttpPayload,
    IdentityDecision,
    IdentityRemapCandidate,
    IdentityRemapReport,
    MappedIdentity,
    MappedMatchStats,
    MappedRanking,
    MappedResult,
    RefreshReport,
    RefreshStatus,
    SitemapPlayer,
    TennisRatioError,
    TennisRatioBlockedError,
    TennisRatioLockError,
    TennisRatioSchemaError,
    TennisRatioStoreError,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw" / "tennisratio"
DEFAULT_DB_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "tennisratio.sqlite3"
DEFAULT_SACKMANN_RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
LAST_GOOD_FILENAME: Final[str] = "last_good.json"
RAW_RETENTION: Final[int] = 2

_AGENDA_PATHS: Final[dict[Gender, str]] = {
    "M": "/atp-matches.html",
    "F": "/wta-matches.html",
}
_CORE_PATHS: Final[tuple[str, ...]] = (
    "/robots.txt",
    "/atp-matches.html",
    "/wta-matches.html",
    "/sitemap-players.xml",
)
_PLAYER_FILES: Final[dict[Gender, tuple[str, str]]] = {
    "M": ("atp", "atp_players.csv"),
    "F": ("wta", "wta_players.csv"),
}
_NORMALIZE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def refresh_tennisratio(
    *,
    raw_dir: Path | None = None,
    database_path: Path | None = None,
    now_utc: datetime | None = None,
    force: bool = False,
    full_inventory: bool | None = None,
    request_delay_seconds: float = 1.0,
    max_profiles: int | None = None,
    session: requests.Session | None = None,
    sackmann_raw_dir: Path | None = None,
) -> RefreshReport:
    """Acquire and atomically publish at most one successful batch per UTC day.

    Args:
        raw_dir: Snapshot root.  Defaults to ``data/raw/tennisratio``.
        database_path: Lateral SQLite path, never the sacred operations DB.
        now_utc: Injectable aware clock value for deterministic tests.  The CLI
            does not expose it and therefore always uses the real UTC clock.
        force: Permit a second validated publication on the same UTC day.
        full_inventory: Fetch every sitemap profile.  ``None`` enables it only
            when no successful inventory has been recorded yet.
        request_delay_seconds: Polite delay between profile GETs.
        max_profiles: Optional diagnostic cap on non-agenda inventory profiles;
            all players visible in either agenda are always included.
        session: Optional requests-compatible fixture session.
        sackmann_raw_dir: Root containing ``atp/atp_players.csv`` and the WTA
            equivalent.  It exists for isolated tests and deployments.

    Returns:
        A deterministic summary containing the canonical state fingerprint.

    Raises:
        TennisRatioError: On core acquisition/schema/publication failure.  A
            single profile failure is instead quarantined and reported.
    """

    if request_delay_seconds < 0.0:
        raise ValueError("request_delay_seconds cannot be negative.")
    if max_profiles is not None and max_profiles < 0:
        raise ValueError("max_profiles cannot be negative.")
    root = Path(raw_dir or DEFAULT_RAW_DIR).resolve()
    db_path = Path(database_path or DEFAULT_DB_PATH).resolve()
    sackmann_root = Path(sackmann_raw_dir or DEFAULT_SACKMANN_RAW_DIR).resolve()
    observed = _normalized_now(now_utc)
    utc_date = observed.date()
    root.mkdir(parents=True, exist_ok=True)
    try:
        store = TennisRatioStore(db_path)
    except sqlite3.Error as exc:
        raise TennisRatioStoreError(f"Cannot initialize TennisRatio sidecar {db_path}.") from exc
    _reconcile_last_good(store, root)

    with _daily_lock(root, utc_date), ExitStack() as resources:
        if not force and store.has_published_date(utc_date):
            remap_report = _publish_identity_remap(
                store=store,
                mapper=_SackmannMapper(sackmann_root),
                observed_at_utc=_clock_value(now_utc),
            )
            return _already_refreshed_report(
                store,
                utc_date,
                root,
                db_path,
                canonical_fingerprint=remap_report.canonical_fingerprint,
                identity_remaps=remap_report.remaps,
            )
        batch_id = f"{observed.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
        client = resources.enter_context(
            closing(
                TennisRatioClient(
                    session=session,
                    cache_root=root.parent / "_http_cache",
                    clock=lambda: _clock_value(now_utc),
                )
            )
        )
        payloads: dict[str, HttpPayload] = {}
        snapshot_ids: dict[str, str] = {}
        snapshot_paths: dict[str, Path] = {}
        for path in _CORE_PATHS:
            payload = client.get(
                path,
                retrieved_at_utc=_clock_value(now_utc),
                cache_mode="refresh" if force else "default",
            )
            resource_kind = _resource_kind(path)
            snapshot_path = _write_raw_snapshot(
                root,
                resource_kind,
                payload.content,
                payload.sha256,
                payload.retrieved_at_utc,
                on_pruned=lambda paths, kind=resource_kind: store.record_retention_events(
                    batch_id=batch_id,
                    artifact_kind=kind,
                    paths=paths,
                    pruned_at_utc=_clock_value(now_utc),
                ),
            )
            snapshot_id = store.register_snapshot(
                payload,
                batch_id=batch_id,
                resource_kind=resource_kind,
                compressed_path=snapshot_path,
            )
            payloads[path] = payload
            snapshot_ids[payload.source_url] = snapshot_id
            snapshot_paths[payload.source_url] = snapshot_path
            if path == "/robots.txt":
                _validate_robots_policy(parse_robots_txt(payload.content))

        agenda_frames = [
            parse_agenda_html(
                payloads[path].content,
                gender=gender,
                source_url=payloads[path].source_url,
                retrieved_at_utc=payloads[path].retrieved_at_utc,
                snapshot_sha256=payloads[path].sha256,
            )
            for gender, path in _AGENDA_PATHS.items()
        ]
        agenda = pd.concat(agenda_frames, ignore_index=True, sort=False)
        inventory = parse_player_sitemap(payloads["/sitemap-players.xml"].content)
        inventory_by_url = {entry.source_url: entry for entry in inventory}
        visible_urls = _visible_profile_urls(agenda)
        visible_entries = {
            url: inventory_by_url.get(url, _entry_from_profile_url(url)) for url in visible_urls
        }
        initial_sync = store.inventory_is_empty()
        fetch_all = initial_sync if full_inventory is None else full_inventory
        if fetch_all:
            delta_entries = list(inventory)
        else:
            delta_entries = [
                entry for entry in inventory if not store.profile_version_succeeded(entry)
            ]
        selected = _select_profiles(
            delta_entries,
            visible_entries=visible_entries,
            max_profiles=max_profiles,
        )
        mapper = _SackmannMapper(sackmann_root)
        identity_remaps = _stage_identity_remaps(
            store=store,
            mapper=mapper,
            batch_id=batch_id,
            observed_at_utc=_clock_value(now_utc),
        )

        attempted = 0
        succeeded = 0
        failed = 0
        unchanged = 0
        profile_manifest: list[dict[str, object]] = []
        for index, entry in enumerate(selected):
            attempted += 1
            previous_good = store.latest_good_profile_snapshot(entry.source_player_key)
            profile_payload = None
            profile_snapshot_path: Path | None = None
            try:
                profile_payload = client.get(
                    entry.source_url,
                    retrieved_at_utc=_clock_value(now_utc),
                    cache_mode="refresh" if force else "default",
                )
                resource_kind = f"profile/{entry.profile_slug}"
                profile_snapshot_path = _write_raw_snapshot(
                    root,
                    resource_kind,
                    profile_payload.content,
                    profile_payload.sha256,
                    profile_payload.retrieved_at_utc,
                    additional_protected=set()
                    if previous_good is None
                    else {previous_good.compressed_path},
                    on_pruned=lambda paths: store.record_retention_events(
                        batch_id=batch_id,
                        artifact_kind="profile_raw",
                        paths=paths,
                        pruned_at_utc=_clock_value(now_utc),
                    ),
                )
                profile_snapshot_id = store.register_snapshot(
                    profile_payload,
                    batch_id=batch_id,
                    resource_kind="profile",
                    compressed_path=profile_snapshot_path,
                )
                previous_sha = None if previous_good is None else previous_good.source_sha256
                if previous_sha == profile_payload.sha256:
                    store.record_profile_sync(
                        batch_id=batch_id,
                        entry=entry,
                        source_sha256=profile_payload.sha256,
                        first_seen_at_utc=profile_payload.retrieved_at_utc,
                        status="unchanged",
                    )
                    unchanged += 1
                    profile_manifest.append(
                        _profile_manifest_entry(
                            entry,
                            profile_payload.sha256,
                            "unchanged",
                            None,
                            profile_snapshot_path,
                        )
                    )
                    continue
                profile = parse_profile_html(
                    profile_payload.content,
                    source_url=profile_payload.source_url,
                )
                if (
                    entry.lastmod is not None
                    and entry.lastmod > profile_payload.retrieved_at_utc.date()
                ):
                    raise TennisRatioSchemaError(
                        f"Sitemap lastmod {entry.lastmod} is after retrieval date."
                    )
                effective_date = entry.lastmod or profile_payload.retrieved_at_utc.date()
                decision = mapper.resolve(profile.gender, profile.player_name, profile.dob)
                inserted = store.record_profile(
                    batch_id=batch_id,
                    snapshot_id=profile_snapshot_id,
                    source_sha256=profile_payload.sha256,
                    first_seen_at_utc=profile_payload.retrieved_at_utc,
                    effective_date=effective_date,
                    sitemap_lastmod=entry.lastmod,
                    profile=profile,
                    decision=decision,
                )
                if inserted:
                    store.record_profile_sync(
                        batch_id=batch_id,
                        entry=entry,
                        source_sha256=profile_payload.sha256,
                        first_seen_at_utc=profile_payload.retrieved_at_utc,
                        status="accepted",
                    )
                    succeeded += 1
                    profile_manifest.append(
                        _profile_manifest_entry(
                            entry,
                            profile_payload.sha256,
                            "accepted",
                            None,
                            profile_snapshot_path,
                        )
                    )
                else:
                    store.record_profile_sync(
                        batch_id=batch_id,
                        entry=entry,
                        source_sha256=profile_payload.sha256,
                        first_seen_at_utc=profile_payload.retrieved_at_utc,
                        status="unchanged",
                    )
                    unchanged += 1
                    profile_manifest.append(
                        _profile_manifest_entry(
                            entry,
                            profile_payload.sha256,
                            "unchanged",
                            None,
                            profile_snapshot_path,
                        )
                    )
            except TennisRatioBlockedError:
                # No recorrer cientos de perfiles durante un bloqueo del host.
                # No publicar un lote truncado ni reemplazar last_good.
                raise
            except (TennisRatioError, ValueError, OSError) as exc:
                failed += 1
                failure_sha = None if profile_payload is None else profile_payload.sha256
                store.record_profile_failure(
                    batch_id=batch_id,
                    source_url=entry.source_url,
                    source_sha256=failure_sha,
                    source_player_key=entry.source_player_key,
                    first_seen_at_utc=_clock_value(now_utc),
                    reason=f"{type(exc).__name__}: {exc}",
                )
                profile_manifest.append(
                    _profile_manifest_entry(
                        entry,
                        None if previous_good is None else previous_good.source_sha256,
                        "failed",
                        f"{type(exc).__name__}: {exc}"[:2_000],
                        None if previous_good is None else previous_good.compressed_path,
                        attempted_sha256=failure_sha,
                        attempted_compressed_path=profile_snapshot_path,
                    )
                )
            finally:
                if index + 1 < len(selected) and request_delay_seconds:
                    time.sleep(request_delay_seconds)

        fingerprint = store.canonical_fingerprint(agenda, batch_id=batch_id)
        status = "published_with_profile_failures" if failed else "published"
        core_retrieved = max(payload.retrieved_at_utc for payload in payloads.values())
        store.complete_batch(
            batch_id=batch_id,
            utc_date=utc_date,
            retrieved_at_utc=core_retrieved,
            status=status,
            agenda=agenda,
            agenda_snapshot_ids=snapshot_ids,
            sitemap_snapshot_id=snapshot_ids[payloads["/sitemap-players.xml"].source_url],
            inventory=inventory,
            inventory_source_sha256=payloads["/sitemap-players.xml"].sha256,
            canonical_fingerprint=fingerprint,
            profiles_selected=len(selected),
            profiles_attempted=attempted,
            profiles_succeeded=succeeded,
            profiles_failed=failed,
            profiles_unchanged=unchanged,
        )
        manifest = _build_manifest(
            batch_id=batch_id,
            utc_date=utc_date,
            retrieved_at_utc=core_retrieved,
            status=status,
            agenda_rows=len(agenda),
            inventory_entries=len(inventory),
            canonical_fingerprint=fingerprint,
            core_payloads=payloads,
            snapshot_paths=snapshot_paths,
            profiles=profile_manifest,
            identity_remaps=identity_remaps,
            master_sha256_by_gender=mapper.master_sha256_by_gender,
        )
        manifest_bytes = _json_bytes(manifest)
        immutable_manifest = _write_immutable_manifest(root, batch_id, manifest_bytes)
        last_good = root / LAST_GOOD_FILENAME
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        store.mark_published(
            batch_id=batch_id,
            published_at_utc=_clock_value(now_utc),
            manifest_path=immutable_manifest,
            manifest_sha256=manifest_sha,
        )
        _atomic_write(last_good, manifest_bytes)
        pruned_manifests = _prune_directory(
            immutable_manifest.parent, keep=RAW_RETENTION, protected={immutable_manifest}
        )
        store.record_retention_events(
            batch_id=batch_id,
            artifact_kind="batch_manifest",
            paths=pruned_manifests,
            pruned_at_utc=_clock_value(now_utc),
        )
        return RefreshReport(
            status=cast(RefreshStatus, status),
            utc_date=utc_date,
            retrieved_at_utc=core_retrieved,
            already_refreshed=False,
            agenda_rows=len(agenda),
            profiles_selected=len(selected),
            profiles_attempted=attempted,
            profiles_succeeded=succeeded,
            profiles_failed=failed,
            profiles_unchanged=unchanged,
            inventory_entries=len(inventory),
            canonical_fingerprint=fingerprint,
            manifest_path=last_good,
            database_path=db_path,
            identity_remaps=identity_remaps,
        )


def remap_tennisratio_identities(
    *,
    raw_dir: Path | None = None,
    database_path: Path | None = None,
    sackmann_raw_dir: Path | None = None,
    now_utc: datetime | None = None,
) -> IdentityRemapReport:
    """Reevaluate published identities against local Sackmann masters without HTTP.

    The changed decisions are staged append-only and become visible together
    through a dedicated publication event.  This entrypoint is intended to run
    immediately after Sackmann sources update, including when the daily HTTP
    refresh has already published.
    """

    root = Path(raw_dir or DEFAULT_RAW_DIR).resolve()
    database = Path(database_path or DEFAULT_DB_PATH).resolve()
    sackmann_root = Path(sackmann_raw_dir or DEFAULT_SACKMANN_RAW_DIR).resolve()
    observed = _normalized_now(now_utc)
    root.mkdir(parents=True, exist_ok=True)
    try:
        store = TennisRatioStore(database)
    except sqlite3.Error as exc:
        raise TennisRatioStoreError(f"Cannot initialize TennisRatio sidecar {database}.") from exc
    with _daily_lock(root, observed.date()):
        return _publish_identity_remap(
            store=store,
            mapper=_SackmannMapper(sackmann_root),
            observed_at_utc=_clock_value(now_utc),
        )


def load_active_agenda(
    match_date: date,
    *,
    database_path: Path | None = None,
) -> pd.DataFrame:
    """Load a published schedule date from local last-good state without network."""

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_active_agenda(
            match_date
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load TennisRatio active agenda.") from exc


def load_identity_resolutions(
    *,
    database_path: Path | None = None,
    as_of_date: date | None = None,
    gender: Gender | None = None,
) -> tuple[MappedIdentity, ...]:
    """Load exact unique identities, optionally with strict ``< D`` availability.

    Omitting ``as_of_date`` is reserved for current agenda presentation.  A
    historical serving or feature caller must always pass its prediction date.
    """

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_identity_resolutions(
            as_of_date=as_of_date, gender=gender
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load TennisRatio identity resolutions.") from exc


def load_agenda_as_of(
    as_of_date: date,
    *,
    match_date: date | None = None,
    database_path: Path | None = None,
) -> pd.DataFrame:
    """Load latest per-match agenda evidence published strictly before ``D``."""

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_agenda_as_of(
            as_of_date, match_date=match_date
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load causal TennisRatio agenda.") from exc


def load_mapped_rankings(
    as_of_date: date,
    *,
    gender: Gender | None = None,
    database_path: Path | None = None,
) -> tuple[MappedRanking, ...]:
    """Load causal uniquely mapped rankings; both source dates must be ``< D``."""

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_mapped_rankings(
            as_of_date, gender=gender
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load TennisRatio mapped rankings.") from exc


def load_mapped_results(
    as_of_date: date,
    *,
    base_cutoff_by_gender: Mapping[str, date],
    gender: Gender | None = None,
    database_path: Path | None = None,
) -> tuple[MappedResult, ...]:
    """Load causal mapped results newer than the caller's immutable base cutoff."""

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_mapped_results(
            as_of_date,
            base_cutoff_by_gender=base_cutoff_by_gender,
            gender=gender,
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load TennisRatio mapped results.") from exc


def load_mapped_match_stats(
    as_of_date: date,
    *,
    gender: Gender | None = None,
    database_path: Path | None = None,
) -> tuple[MappedMatchStats, ...]:
    """Load mapped service/return observations strictly available before ``D``."""

    try:
        return TennisRatioStore(Path(database_path or DEFAULT_DB_PATH)).load_mapped_match_stats(
            as_of_date,
            gender=gender,
        )
    except sqlite3.Error as exc:
        raise TennisRatioStoreError("Cannot load TennisRatio mapped match stats.") from exc


class _SackmannMapper:
    """Resolve exact gender/name/DOB identities from immutable Sackmann masters."""

    def __init__(self, raw_dir: Path) -> None:
        self._candidates: dict[tuple[Gender, str], list[tuple[int, date | None]]] = {}
        self.master_sha256_by_gender: dict[Gender, str] = {}
        for gender, (directory, filename) in _PLAYER_FILES.items():
            path = raw_dir / directory / filename
            if not path.is_file():
                raise TennisRatioSchemaError(f"Missing Sackmann player master: {path}.")
            try:
                self.master_sha256_by_gender[gender] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise TennisRatioSchemaError(
                    f"Cannot read Sackmann player master: {path}."
                ) from exc
            gender_candidates = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"player_id", "name_first", "name_last", "dob"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise TennisRatioSchemaError(f"Invalid Sackmann player header: {path}.")
                for row in reader:
                    player_id_text = (row.get("player_id") or "").strip()
                    full_name = " ".join(
                        part
                        for part in (
                            (row.get("name_first") or "").strip(),
                            (row.get("name_last") or "").strip(),
                        )
                        if part
                    )
                    if not player_id_text.isdigit() or not full_name:
                        continue
                    dob = _parse_sackmann_dob((row.get("dob") or "").strip())
                    key = (gender, _normalize_name(full_name))
                    self._candidates.setdefault(key, []).append((int(player_id_text), dob))
                    gender_candidates += 1
            if gender_candidates == 0:
                raise TennisRatioSchemaError(f"Empty Sackmann player master: {path}.")

    def resolve(self, gender: Gender, name: str, dob: date | None) -> IdentityDecision:
        """Return a mapping only for one exact admissible candidate."""

        candidates = list(self._candidates.get((gender, _normalize_name(name)), []))
        if dob is not None:
            candidates = [candidate for candidate in candidates if candidate[1] == dob]
            method = "exact_gender_normalized_name_dob"
        else:
            method = "exact_gender_normalized_name"
        if len(candidates) == 1:
            return IdentityDecision(
                status="mapped",
                sackmann_player_id=candidates[0][0],
                method=method,
                candidate_count=1,
            )
        return IdentityDecision(
            status="unmapped" if not candidates else "ambiguous",
            sackmann_player_id=None,
            method=method,
            candidate_count=len(candidates),
        )


def _stage_identity_remaps(
    *,
    store: TennisRatioStore,
    mapper: _SackmannMapper,
    batch_id: str,
    observed_at_utc: datetime,
    candidates: Sequence[IdentityRemapCandidate] | None = None,
) -> int:
    """Stage only changed identity decisions for a future publication event."""

    typed_candidates = store.identity_remap_candidates() if candidates is None else candidates
    changed = 0
    for candidate in typed_candidates:
        decision = mapper.resolve(candidate.gender, candidate.player_name, candidate.dob)
        if store.record_identity_remap(
            batch_id=batch_id,
            candidate=candidate,
            decision=decision,
            mapping_source_sha256=mapper.master_sha256_by_gender[candidate.gender],
            first_seen_at_utc=observed_at_utc,
        ):
            changed += 1
    return changed


def _publish_identity_remap(
    *,
    store: TennisRatioStore,
    mapper: _SackmannMapper,
    observed_at_utc: datetime,
) -> IdentityRemapReport:
    """Publish changed local-master resolutions as one all-or-nothing batch."""

    batch_id = f"remap-{observed_at_utc.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
    candidates = store.identity_remap_candidates()
    remaps = _stage_identity_remaps(
        store=store,
        mapper=mapper,
        batch_id=batch_id,
        observed_at_utc=observed_at_utc,
        candidates=candidates,
    )
    if remaps:
        fingerprint = store.canonical_fingerprint_with_staged_remaps(batch_id)
        store.publish_identity_remap_batch(
            batch_id=batch_id,
            observed_at_utc=observed_at_utc,
            published_at_utc=observed_at_utc,
            master_sha256_by_gender=mapper.master_sha256_by_gender,
            remaps=remaps,
            canonical_fingerprint=fingerprint,
        )
    else:
        fingerprint = store.current_canonical_fingerprint()
    return IdentityRemapReport(
        batch_id=batch_id,
        observed_at_utc=observed_at_utc,
        published_at_utc=observed_at_utc,
        candidates=len(candidates),
        remaps=remaps,
        master_sha256_by_gender=dict(mapper.master_sha256_by_gender),
        canonical_fingerprint=fingerprint,
        published=remaps > 0,
    )


def _select_profiles(
    delta_entries: Sequence[SitemapPlayer],
    *,
    visible_entries: Mapping[str, SitemapPlayer],
    max_profiles: int | None,
) -> list[SitemapPlayer]:
    """Keep every visible player, then fill deterministic sitemap delta slots."""

    selected = dict(visible_entries)
    extras = [
        entry
        for entry in sorted(delta_entries, key=lambda item: item.source_url)
        if entry.source_url not in selected
    ]
    if max_profiles is not None:
        extras = extras[:max_profiles]
    for entry in extras:
        selected[entry.source_url] = entry
    return sorted(selected.values(), key=lambda item: item.source_url)


def _visible_profile_urls(agenda: pd.DataFrame) -> set[str]:
    """Collect both participants of all server-rendered agenda matches."""

    if agenda.empty:
        return set()
    values = pd.concat((agenda["player_1_href"], agenda["player_2_href"]), ignore_index=True)
    urls = {str(value) for value in values.dropna().tolist()}
    for url in urls:
        profile_slug_from_url(url)
    return urls


def _entry_from_profile_url(source_url: str) -> SitemapPlayer:
    """Create a conservative no-lastmod entry for a visible non-inventory profile."""

    slug = profile_slug_from_url(source_url)
    return SitemapPlayer(
        source_url=source_url,
        source_player_key=source_player_key(slug),
        profile_slug=slug,
        lastmod=None,
    )


def _write_raw_snapshot(
    raw_dir: Path,
    resource_kind: str,
    content: bytes,
    sha256: str,
    retrieved_at_utc: datetime,
    additional_protected: set[Path] | None = None,
    on_pruned: Callable[[Sequence[Path]], None] | None = None,
) -> Path:
    """Write a deterministic gzip snapshot while retaining parsed-good evidence."""

    directory = raw_dir / "snapshots" / Path(resource_kind)
    directory.mkdir(parents=True, exist_ok=True)
    validated_additional: set[Path] = set()
    for protected_path in additional_protected or set():
        resolved = Path(protected_path).resolve()
        if resolved.parent != directory.resolve() or not resolved.is_file():
            raise TennisRatioStoreError(
                f"Published parsed-good snapshot is missing or misplaced: {resolved}."
            )
        validated_additional.add(resolved)
    # Identical bytes are one immutable object even when observed on another
    # day; this keeps every DB compressed_path valid while avoiding copies.
    path = directory / f"{sha256}.raw.gz"
    if not path.exists():
        temporary = directory / f".{path.name}.{uuid.uuid4().hex}.part"
        try:
            with temporary.open("xb") as raw_handle:
                with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
                    compressed.write(content)
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    protected = _last_good_protected_paths(raw_dir, directory)
    protected.update(validated_additional)
    protected.add(path.resolve())
    pruned = _prune_directory(directory, keep=RAW_RETENTION, protected=protected)
    if on_pruned is not None and pruned:
        on_pruned(pruned)
    return path


def _write_immutable_manifest(raw_dir: Path, batch_id: str, content: bytes) -> Path:
    """Write one immutable batch manifest and return its resolved path."""

    path = raw_dir / "manifests" / f"{batch_id}.json"
    if path.exists():
        raise TennisRatioStoreError(f"Immutable manifest already exists: {path}.")
    _atomic_write(path, content)
    return path


def _atomic_write(path: Path, content: bytes) -> None:
    """Durably replace one file using a temporary sibling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.part"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prune_directory(
    directory: Path,
    *,
    keep: int,
    protected: set[Path],
) -> tuple[Path, ...]:
    """Retain newest immutable files while never deleting the just-written file."""

    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    retained = {path.resolve() for path in protected if path.is_file()}
    for path in files:
        if len(retained) >= keep:
            break
        retained.add(path.resolve())
    pruned: list[Path] = []
    for path in files:
        if path.resolve() not in retained:
            path.unlink()
            pruned.append(path.resolve())
    return tuple(pruned)


def _last_good_protected_paths(raw_dir: Path, resource_dir: Path) -> set[Path]:
    """Read validated active-manifest paths for one logical resource directory."""

    manifest_path = raw_dir / LAST_GOOD_FILENAME
    if not manifest_path.is_file():
        return set()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TennisRatioStoreError("Cannot validate current last_good before pruning.") from exc
    if not isinstance(payload, dict):
        raise TennisRatioStoreError("Current last_good manifest must be a JSON object.")
    candidates: list[object] = []
    for section in ("sources", "profiles"):
        entries = payload.get(section, [])
        if not isinstance(entries, list):
            raise TennisRatioStoreError(f"last_good.{section} must be a list.")
        candidates.extend(entries)
    snapshot_root = (raw_dir / "snapshots").resolve()
    target_directory = resource_dir.resolve()
    protected: set[Path] = set()
    for entry in candidates:
        if not isinstance(entry, dict):
            raise TennisRatioStoreError("last_good snapshot entry must be an object.")
        raw_path = entry.get("compressed_path")
        if raw_path is None:
            continue
        if not isinstance(raw_path, str):
            raise TennisRatioStoreError("last_good compressed_path must be text or null.")
        resolved = Path(raw_path).resolve()
        if snapshot_root not in resolved.parents:
            raise TennisRatioStoreError("last_good compressed_path escaped the snapshot root.")
        if resolved.parent == target_directory:
            if not resolved.is_file():
                raise TennisRatioStoreError(
                    f"last_good references a missing active snapshot: {resolved}."
                )
            protected.add(resolved)
    return protected


@contextmanager
def _daily_lock(raw_dir: Path, utc_date: date) -> Iterator[None]:
    """Use exclusive directory creation as a cross-platform interprocess lock."""

    resolved_root = raw_dir.resolve()
    lock_dir = (resolved_root / ".refresh.lock").resolve()
    if lock_dir.parent != resolved_root:
        raise TennisRatioLockError("Refresh lock escaped the configured raw directory.")
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        if not _recover_stale_lock(lock_dir, utc_date):
            raise TennisRatioLockError(
                f"Another TennisRatio refresh owns {lock_dir} for the current day."
            ) from exc
        try:
            lock_dir.mkdir()
        except FileExistsError as retry_exc:
            raise TennisRatioLockError(
                f"Another TennisRatio refresh acquired {lock_dir} during stale recovery."
            ) from retry_exc
    owner = lock_dir / "owner.json"
    try:
        _atomic_write(
            owner,
            _json_bytes(
                {
                    "pid": os.getpid(),
                    "utc_date": utc_date.isoformat(),
                    "acquired_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
            ),
        )
        yield
    finally:
        owner.unlink(missing_ok=True)
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _pid_is_alive(pid: int) -> bool:
    """Return false only when the OS proves that ``pid`` no longer exists."""

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        # PROCESS_QUERY_LIMITED_INFORMATION; no mutation/termination rights.
        handle = open_process(0x1000, 0, pid)
        if handle:
            close_handle(handle)
            return True
        # ERROR_INVALID_PARAMETER is the documented result for a missing PID.
        return ctypes.get_last_error() != 87
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _recover_stale_lock(lock_dir: Path, utc_date: date) -> bool:
    """Recover a validated old-day lock or a same-day dead owner.

    Current/future locks with a live or unverifiable PID are never stolen.
    Recovery also requires the directory to contain exactly its owner record.
    """

    owner = (lock_dir / "owner.json").resolve()
    if owner.parent != lock_dir.resolve() or not owner.is_file():
        return False
    try:
        payload = json.loads(owner.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    owner_date_raw = payload.get("utc_date")
    owner_pid = payload.get("pid")
    acquired = payload.get("acquired_at_utc")
    if (
        not isinstance(owner_date_raw, str)
        or not isinstance(owner_pid, int)
        or isinstance(owner_pid, bool)
        or owner_pid <= 0
        or not isinstance(acquired, str)
    ):
        return False
    try:
        owner_date = date.fromisoformat(owner_date_raw)
        acquired_at = datetime.fromisoformat(acquired.replace("Z", "+00:00"))
    except ValueError:
        return False
    if (
        acquired_at.tzinfo is None
        or acquired_at.astimezone(UTC).date() != owner_date
        or owner_date > utc_date
        or (owner_date == utc_date and _pid_is_alive(owner_pid))
    ):
        return False
    try:
        entries = list(lock_dir.iterdir())
        if entries != [owner]:
            return False
        owner.unlink()
        lock_dir.rmdir()
    except OSError:
        return False
    return True


def _build_manifest(
    *,
    batch_id: str,
    utc_date: date,
    retrieved_at_utc: datetime,
    status: str,
    agenda_rows: int,
    inventory_entries: int,
    canonical_fingerprint: str,
    core_payloads: Mapping[str, HttpPayload],
    snapshot_paths: Mapping[str, Path],
    profiles: Sequence[dict[str, object]],
    identity_remaps: int,
    master_sha256_by_gender: Mapping[Gender, str],
) -> dict[str, object]:
    """Build the attributed, machine-readable last-good publication."""

    sources: list[dict[str, object]] = []
    for path in _CORE_PATHS:
        payload = core_payloads[path]
        sources.append(
            {
                "resource": _resource_kind(path),
                "source_url": payload.source_url,
                "sha256": payload.sha256,
                "retrieved_at_utc": payload.retrieved_at_utc.isoformat().replace("+00:00", "Z"),
                "compressed_path": str(snapshot_paths[payload.source_url]),
            }
        )
    return {
        "schema_version": 2,
        "batch_id": batch_id,
        "utc_date": utc_date.isoformat(),
        "status": status,
        "retrieved_at_utc": retrieved_at_utc.isoformat().replace("+00:00", "Z"),
        "canonical_fingerprint": canonical_fingerprint,
        "agenda_rows": agenda_rows,
        "inventory_entries": inventory_entries,
        "identity_remaps": identity_remaps,
        "sackmann_master_sha256": {
            "M": master_sha256_by_gender["M"],
            "F": master_sha256_by_gender["F"],
        },
        "license": LICENSE_URL,
        "attribution": ATTRIBUTION,
        "acquisition_contract": "public GET only; /api/ forbidden",
        "sources": sources,
        "profiles": list(profiles),
    }


def _profile_manifest_entry(
    entry: SitemapPlayer,
    sha256: str | None,
    status: str,
    error: str | None,
    compressed_path: Path | None,
    attempted_sha256: str | None = None,
    attempted_compressed_path: Path | None = None,
) -> dict[str, object]:
    """Create one per-profile manifest outcome."""

    return {
        "source_url": entry.source_url,
        "source_player_key": entry.source_player_key,
        "sitemap_lastmod": None if entry.lastmod is None else entry.lastmod.isoformat(),
        "sha256": sha256,
        "status": status,
        "error": error,
        "compressed_path": None if compressed_path is None else str(compressed_path.resolve()),
        "attempted_sha256": attempted_sha256,
        "attempted_compressed_path": None
        if attempted_compressed_path is None
        else str(attempted_compressed_path.resolve()),
    }


def _already_refreshed_report(
    store: TennisRatioStore,
    utc_date: date,
    raw_dir: Path,
    database_path: Path,
    *,
    canonical_fingerprint: str | None = None,
    identity_remaps: int = 0,
) -> RefreshReport:
    """Rehydrate an idempotent once-per-day response from the published event."""

    row = store.published_report_row(utc_date)
    if row is None:
        raise TennisRatioStoreError("Published date disappeared during report load.")
    return RefreshReport(
        status="already_refreshed",
        utc_date=utc_date,
        retrieved_at_utc=datetime.fromisoformat(
            str(row["retrieved_at_utc"]).replace("Z", "+00:00")
        ),
        already_refreshed=True,
        agenda_rows=int(row["agenda_rows"]),
        profiles_selected=int(row["profiles_selected"]),
        profiles_attempted=int(row["profiles_attempted"]),
        profiles_succeeded=int(row["profiles_succeeded"]),
        profiles_failed=int(row["profiles_failed"]),
        profiles_unchanged=int(row["profiles_unchanged"]),
        inventory_entries=int(row["inventory_entries"]),
        canonical_fingerprint=canonical_fingerprint or str(row["canonical_fingerprint"]),
        manifest_path=raw_dir / LAST_GOOD_FILENAME,
        database_path=database_path,
        identity_remaps=identity_remaps,
    )


def _reconcile_last_good(store: TennisRatioStore, raw_dir: Path) -> None:
    """Repair a stale pointer only from the newest DB-published immutable manifest."""

    row = store.latest_published_manifest_row()
    if row is None:
        return
    immutable = Path(str(row["manifest_path"])).resolve()
    manifests_dir = (raw_dir / "manifests").resolve()
    if immutable.parent != manifests_dir or not immutable.is_file():
        raise TennisRatioStoreError(
            "Newest published TennisRatio batch has no valid immutable manifest."
        )
    try:
        content = immutable.read_bytes()
    except OSError as exc:
        raise TennisRatioStoreError("Cannot read newest immutable manifest.") from exc
    if hashlib.sha256(content).hexdigest() != str(row["manifest_sha256"]):
        raise TennisRatioStoreError("Newest immutable manifest SHA does not match SQLite.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TennisRatioStoreError("Newest immutable manifest is invalid JSON.") from exc
    if not isinstance(payload, dict) or payload.get("batch_id") != str(row["batch_id"]):
        raise TennisRatioStoreError("Newest immutable manifest contradicts its batch id.")
    last_good = raw_dir / LAST_GOOD_FILENAME
    if last_good.is_file():
        try:
            if hashlib.sha256(last_good.read_bytes()).hexdigest() == str(row["manifest_sha256"]):
                return
        except OSError as exc:
            raise TennisRatioStoreError("Cannot validate current last_good pointer.") from exc
    _atomic_write(last_good, content)


def _validate_robots_policy(disallowed: frozenset[str]) -> None:
    """Ensure every path used by this adapter remains outside disallowed rules."""

    for path in _CORE_PATHS[1:]:
        if _path_disallowed(path, disallowed):
            raise TennisRatioSchemaError(f"robots.txt now disallows approved path {path}.")
    if _path_disallowed("/players/Example.html", disallowed):
        raise TennisRatioSchemaError("robots.txt now disallows public player profiles.")


def _path_disallowed(path: str, rules: frozenset[str]) -> bool:
    """Apply anchored prefix/wildcard rules without widening ``/*?...`` to ``/``."""

    for rule in rules:
        anchored = rule.endswith("$")
        body = rule[:-1] if anchored else rule
        pattern = re.escape(body).replace(r"\*", ".*")
        suffix = "$" if anchored else ""
        if pattern and re.match(f"^{pattern}{suffix}", path):
            return True
    return False


def _resource_kind(path: str) -> str:
    """Map an approved core path to a stable snapshot directory name."""

    return {
        "/robots.txt": "robots",
        "/atp-matches.html": "agenda_atp",
        "/wta-matches.html": "agenda_wta",
        "/sitemap-players.xml": "sitemap_players",
    }[path]


def _normalized_now(value: datetime | None) -> datetime:
    """Use a real aware UTC time unless a deterministic test value is supplied."""

    observed = datetime.now(UTC) if value is None else value
    if observed.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware.")
    return observed.astimezone(UTC)


def _clock_value(fixed: datetime | None) -> datetime:
    """Return a fresh acquisition instant or the injected deterministic clock."""

    return _normalized_now(fixed)


def _normalize_name(value: str) -> str:
    """Normalize exact names without fuzzy distance or initials."""

    normalized = _NORMALIZE_RE.sub(" ", unidecode(value).casefold())
    result = " ".join(normalized.split())
    if not result:
        raise TennisRatioSchemaError("Player name is empty after exact normalization.")
    return result


def _parse_sackmann_dob(value: str) -> date | None:
    """Parse optional Sackmann YYYYMMDD birth dates conservatively."""

    if not re.fullmatch(r"[0-9]{8}", value):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _json_bytes(value: object) -> bytes:
    """Serialize deterministic UTF-8 JSON with a final newline."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

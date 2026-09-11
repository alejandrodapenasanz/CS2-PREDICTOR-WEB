"""Dedicated, deduplicated Tennis Abstract acquisition store, not operational data.

Immutable source rows and snapshot manifests retain first observation time.
Only progress/status and the two-version raw response cache are mutable.
No source tournament date is exposed as an exact match date or model feature.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import zlib

from .config import PROJECT_ROOT
from .tennis_abstract_history_audit import PlayerHistory

DEFAULT_STORE = PROJECT_ROOT / "data" / "processed" / "tennis_abstract.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ta_rows (
    row_hash TEXT PRIMARY KEY, cells_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ta_snapshots (
    snapshot_id TEXT PRIMARY KEY, gender TEXT NOT NULL, player_key TEXT NOT NULL,
    first_captured_at_utc TEXT NOT NULL, manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ta_snapshots_asof
    ON ta_snapshots(gender, player_key, first_captured_at_utc);
CREATE TABLE IF NOT EXISTS ta_sightings (
    gender TEXT NOT NULL, player_key TEXT NOT NULL, captured_at_utc TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES ta_snapshots(snapshot_id),
    PRIMARY KEY(gender,player_key,captured_at_utc)
);
CREATE TABLE IF NOT EXISTS ta_checks (
    gender TEXT NOT NULL, player_key TEXT NOT NULL, last_success_utc TEXT,
    last_attempt_utc TEXT NOT NULL, snapshot_id TEXT REFERENCES ta_snapshots(snapshot_id),
    error TEXT, PRIMARY KEY(gender, player_key)
);
CREATE TABLE IF NOT EXISTS ta_raw_documents (
    url TEXT NOT NULL, sha256 TEXT NOT NULL, captured_at_utc TEXT NOT NULL,
    content_zlib BLOB NOT NULL, PRIMARY KEY(url, sha256)
);
CREATE TABLE IF NOT EXISTS ta_state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
"""


def utc_text(value: datetime) -> str:
    """Reject naive instants and use one sortable UTC representation."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An aware capture clock is required.")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def canonical_json(value: object) -> str:
    """Serialize evidence stably, including original non-ASCII player names."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: bytes) -> str:
    """Fingerprint source bytes or canonical manifests without stochastic state."""

    return hashlib.sha256(value).hexdigest()


@contextmanager
def acquisition_lock(path: Path) -> Iterator[None]:
    """Hold an OS-released SQLite writer lock, independent of committed progress."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path.with_suffix(".refresh.sqlite3"), timeout=0)) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Another Tennis Abstract acquisition is already running.") from exc
        try:
            yield
        finally:
            lock.rollback()


class TennisAbstractStore:
    """Persist source-only evidence without opening the sacred operational DB."""

    def __init__(self, path: Path = DEFAULT_STORE, *, read_only: bool = False) -> None:
        """Require a dedicated source file; status readers cannot create or change it."""

        self.path = path.resolve()
        if self.path.name != DEFAULT_STORE.name:
            raise ValueError("Use a dedicated tennis_abstract.sqlite3 source store.")
        if read_only:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.connection.executescript(SCHEMA)

    def close(self) -> None:
        """Release the sidecar connection."""

        self.connection.close()

    def state(self, key: str) -> Any:
        """Read replaceable acquisition status, never a feature value."""

        row = self.connection.execute(
            "SELECT value_json FROM ta_state WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def set_state(self, key: str, value: object) -> None:
        """Persist a resumable status checkpoint atomically."""

        with self.connection:
            self.connection.execute(
                "INSERT INTO ta_state VALUES (?,?) ON CONFLICT(key) "
                "DO UPDATE SET value_json=excluded.value_json",
                (key, canonical_json(value)),
            )

    def checks(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return successful refresh and attempt times for fair, resumable scheduling."""

        return {
            (row["gender"], row["player_key"]): dict(row)
            for row in self.connection.execute("SELECT * FROM ta_checks")
        }

    def record_failure(self, gender: str, key: str, at: datetime, error: str) -> None:
        """Keep the last successful snapshot when a new request fails."""

        with self.connection:
            self.connection.execute(
                "INSERT INTO ta_checks (gender,player_key,last_attempt_utc,error) VALUES (?,?,?,?) "
                "ON CONFLICT(gender,player_key) DO UPDATE SET "
                "last_attempt_utc=excluded.last_attempt_utc,error=excluded.error",
                (gender, key, utc_text(at), error),
            )

    def save(self, history: PlayerHistory, *, captured_at: datetime, inventory: dict) -> str:
        """Deduplicate facts; preserve first-seen availability and two raw versions."""

        at = utc_text(captured_at)
        gender, key = str(history.report["gender"]), str(history.report["player_key"])
        cells = [canonical_json(row) for row in history.rows]
        hashes = [digest(row.encode("utf-8")) for row in cells]
        quarantined_cells = [canonical_json(row) for _, row in history.quarantined_rows]
        quarantined_hashes = [digest(row.encode("utf-8")) for row in quarantined_cells]
        quarantine = [
            {
                "source_row_index": index,
                "row_hash": row_hash,
                "width": len(row),
                "reason": "fewer_than_44_fields_no_mapping",
            }
            for (index, row), row_hash in zip(
                history.quarantined_rows, quarantined_hashes, strict=True
            )
        ]
        # Ignore changing HTML/UI bytes when the inspected data has not changed.
        # Preserve the existing fingerprint when there are no quarantined rows.
        identity: list = [gender, key, history.header, hashes]
        if quarantine:
            identity.append(quarantine)
        snapshot_id = digest(canonical_json(identity).encode("utf-8"))
        manifest = {
            **history.report,
            "inventory": inventory,
            "header": history.header,
            "row_hashes": hashes,
            "quarantine": quarantine,
            "documents": [{"url": url, "sha256": digest(body)} for url, body in history.documents],
            "exact_match_date": None,
            "canonical_player_id": None,
            "model_ready": False,
        }
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO ta_rows VALUES (?,?)",
                zip(hashes + quarantined_hashes, cells + quarantined_cells, strict=True),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO ta_snapshots VALUES (?,?,?,?,?)",
                (snapshot_id, gender, key, at, canonical_json(manifest)),
            )
            previous = self.connection.execute(
                "SELECT snapshot_id FROM ta_checks WHERE gender=? AND player_key=?", (gender, key)
            ).fetchone()
            if previous is None or previous[0] != snapshot_id:
                self.connection.execute(
                    "INSERT INTO ta_sightings VALUES (?,?,?,?)", (gender, key, at, snapshot_id)
                )
            self.connection.execute(
                "INSERT INTO ta_checks VALUES (?,?,?,?,?,NULL) ON CONFLICT(gender,player_key) "
                "DO UPDATE SET last_success_utc=excluded.last_success_utc,"
                "last_attempt_utc=excluded.last_attempt_utc,snapshot_id=excluded.snapshot_id,error=NULL",
                (gender, key, at, at, snapshot_id),
            )
            for url, body in history.documents:
                self.connection.execute(
                    "INSERT INTO ta_raw_documents VALUES (?,?,?,?) ON CONFLICT(url,sha256) "
                    "DO UPDATE SET captured_at_utc=excluded.captured_at_utc",
                    (url, digest(body), at, zlib.compress(body)),
                )
                self.connection.execute(
                    "DELETE FROM ta_raw_documents WHERE url=? AND sha256 NOT IN "
                    "(SELECT sha256 FROM ta_raw_documents WHERE url=? "
                    "ORDER BY captured_at_utc DESC,sha256 DESC LIMIT 2)",
                    (url, url),
                )
        return snapshot_id

    def snapshot_before(self, gender: str, key: str, day: date) -> dict | None:
        """Read strictly prior acquisition evidence, NOT approved model features."""

        cutoff = utc_text(datetime.combine(day, datetime.min.time(), tzinfo=UTC))
        row = self.connection.execute(
            "SELECT s.*,o.captured_at_utc FROM ta_sightings o "
            "JOIN ta_snapshots s USING(snapshot_id) WHERE o.gender=? AND o.player_key=? "
            "AND o.captured_at_utc<? ORDER BY o.captured_at_utc DESC LIMIT 1",
            (gender, key, cutoff),
        ).fetchone()
        if row is None:
            return None
        manifest = json.loads(row["manifest_json"])
        manifest["first_captured_at_utc"] = row["first_captured_at_utc"]
        manifest["observed_at_utc"] = row["captured_at_utc"]
        manifest["rows"] = [
            json.loads(
                self.connection.execute(
                    "SELECT cells_json FROM ta_rows WHERE row_hash=?", (row_hash,)
                ).fetchone()[0]
            )
            for row_hash in manifest["row_hashes"]
        ]
        manifest["quarantined_evidence"] = [
            {
                **item,
                "cells": json.loads(
                    self.connection.execute(
                        "SELECT cells_json FROM ta_rows WHERE row_hash=?", (item["row_hash"],)
                    ).fetchone()[0]
                ),
            }
            for item in manifest.get("quarantine", [])
        ]
        return manifest

    def coverage(self) -> dict[str, int]:
        """Count acquired players/facts without claiming historical training coverage."""

        rows = self.connection.execute(
            "SELECT s.manifest_json FROM ta_checks c JOIN ta_snapshots s USING(snapshot_id)"
        ).fetchall()
        reports = [json.loads(row[0]) for row in rows]
        return {
            "stored_players": len(reports),
            "latest_player_rows": sum(report["rows"] for report in reports),
            "latest_rows_with_serve_points": sum(
                report["rows_with_serve_points"] for report in reports
            ),
            "latest_quarantined_rows": sum(report.get("quarantined_rows", 0) for report in reports),
            "unique_source_rows": self.connection.execute(
                "SELECT count(*) FROM ta_rows"
            ).fetchone()[0],
            "model_ready_rows": 0,
        }

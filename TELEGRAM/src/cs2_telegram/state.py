"""Fail-closed SQLite publication ledger for CS2 Telegram messages.

Each channel/date/kind tuple can be reserved once.  A reservation is persisted
before network I/O; ambiguous or failed sends remain blocked so an unattended
rerun can never duplicate a message.  Database triggers make payloads and sent
records immutable, including against accidental deletes.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Literal
import uuid


PublicationKind = Literal["best", "others"]
PublicationStatus = Literal["pending", "sent", "failed"]
_VALID_KINDS = frozenset({"best", "others"})


class PublicationStateError(RuntimeError):
    """Base exception for publication-ledger safety failures."""


class PayloadConflictError(PublicationStateError):
    """Report an attempt to replace a reserved or published payload."""


class AmbiguousPublicationError(PublicationStateError):
    """Block a retry whose preceding delivery was not safely resolved."""


class InvalidStateTransitionError(PublicationStateError):
    """Report an invalid or stale publication state transition."""


@dataclass(frozen=True)
class PublicationRecord:
    """Immutable snapshot of one publication-ledger row."""

    channel_id: str
    target_date: str
    kind: PublicationKind
    source_run_id: str
    payload_sha256: str
    payload_text: str = field(repr=False)
    status: PublicationStatus = "pending"
    reservation_id: str = field(default="", repr=False)
    reserved_at_utc: str = ""
    sent_at_utc: str | None = None
    message_id: int | None = None
    failure_code: str | None = None


@dataclass(frozen=True)
class Reservation:
    """Result of reserving a key, including whether this call created it."""

    record: PublicationRecord
    created: bool


def payload_digest(payload_text: str) -> str:
    """Return a deterministic SHA-256 digest for exact payload comparison."""

    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    """Return an unambiguous, timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _normalize_target_date(target_date: date | str) -> str:
    """Normalize a date-like publication key to strict ISO format."""

    if isinstance(target_date, datetime):
        normalized = target_date.date()
    elif isinstance(target_date, date):
        normalized = target_date
    elif isinstance(target_date, str):
        try:
            normalized = date.fromisoformat(target_date.strip())
        except ValueError:
            raise ValueError("target_date must use YYYY-MM-DD format.") from None
    else:
        raise TypeError("target_date must be a date or YYYY-MM-DD string.")
    return normalized.isoformat()


def _normalize_key(channel_id: str, target_date: date | str, kind: str) -> tuple[str, str, PublicationKind]:
    """Validate and normalize a publication ledger primary key."""

    normalized_channel = channel_id.strip()
    if not normalized_channel:
        raise ValueError("channel_id must not be empty.")
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _VALID_KINDS:
        raise ValueError("kind must be either 'best' or 'others'.")
    return normalized_channel, _normalize_target_date(target_date), normalized_kind  # type: ignore[return-value]


class PublishStateStore:
    """Manage idempotent publication reservations in an on-disk SQLite file."""

    def __init__(self, database_path: str | Path) -> None:
        """Create/open the database and install immutable-state safeguards."""

        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a configured connection with mapping-style result rows."""

        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        """Create the publication ledger and defensive triggers idempotently."""

        schema = """
        CREATE TABLE IF NOT EXISTS publications (
            channel_id TEXT NOT NULL,
            target_date TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('best', 'others')),
            source_run_id TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_text TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')),
            reservation_id TEXT NOT NULL,
            reserved_at_utc TEXT NOT NULL,
            sent_at_utc TEXT,
            message_id INTEGER,
            failure_code TEXT,
            PRIMARY KEY (channel_id, target_date, kind)
        );

        CREATE TABLE IF NOT EXISTS report_exports (
            match_id TEXT PRIMARY KEY,
            match_date TEXT,
            hltv_url TEXT,
            source_run_id TEXT NOT NULL,
            reported_at_utc TEXT NOT NULL
        );

        CREATE TRIGGER IF NOT EXISTS report_exports_update_guard
        BEFORE UPDATE ON report_exports
        BEGIN
            SELECT RAISE(ABORT, 'report export records are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS report_exports_delete_guard
        BEFORE DELETE ON report_exports
        BEGIN
            SELECT RAISE(ABORT, 'report export records are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS publications_identity_immutable
        BEFORE UPDATE ON publications
        WHEN NEW.channel_id <> OLD.channel_id
          OR NEW.target_date <> OLD.target_date
          OR NEW.kind <> OLD.kind
          OR NEW.source_run_id <> OLD.source_run_id
          OR NEW.payload_sha256 <> OLD.payload_sha256
          OR NEW.payload_text <> OLD.payload_text
          OR NEW.reservation_id <> OLD.reservation_id
          OR NEW.reserved_at_utc <> OLD.reserved_at_utc
        BEGIN
            SELECT RAISE(ABORT, 'publication identity and payload are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS publications_transition_guard
        BEFORE UPDATE ON publications
        WHEN NOT (OLD.status = 'pending' AND NEW.status IN ('sent', 'failed'))
        BEGIN
            SELECT RAISE(ABORT, 'invalid publication state transition');
        END;

        CREATE TRIGGER IF NOT EXISTS publications_delete_guard
        BEFORE DELETE ON publications
        BEGIN
            SELECT RAISE(ABORT, 'publication records are immutable');
        END;
        """
        with closing(self._connect()) as connection, connection:
            connection.executescript(schema)

    def reported_match_ids(self) -> set[str]:
        """Return every match identifier already emitted in a daily report."""

        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT match_id FROM report_exports").fetchall()
        return {str(row["match_id"]) for row in rows}

    def mark_reported(
        self,
        *,
        match_id: str,
        source_run_id: str,
        match_date: date | str | None = None,
        hltv_url: str | None = None,
    ) -> None:
        """Persist one immutable report export, idempotently by match ID."""

        normalized_id = str(match_id).strip()
        normalized_run = source_run_id.strip()
        if not normalized_id or not normalized_run:
            raise ValueError("match_id and source_run_id must not be empty.")
        normalized_date = _normalize_target_date(match_date) if match_date is not None else None
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO report_exports (
                    match_id, match_date, hltv_url, source_run_id, reported_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_id, normalized_date, hltv_url, normalized_run, _utc_now()),
            )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> PublicationRecord:
        """Convert a database row into a typed immutable record."""

        return PublicationRecord(
            channel_id=row["channel_id"],
            target_date=row["target_date"],
            kind=row["kind"],
            source_run_id=row["source_run_id"],
            payload_sha256=row["payload_sha256"],
            payload_text=row["payload_text"],
            status=row["status"],
            reservation_id=row["reservation_id"],
            reserved_at_utc=row["reserved_at_utc"],
            sent_at_utc=row["sent_at_utc"],
            message_id=row["message_id"],
            failure_code=row["failure_code"],
        )

    def get(self, channel_id: str, target_date: date | str, kind: str) -> PublicationRecord | None:
        """Return a publication record without modifying it."""

        channel, iso_date, normalized_kind = _normalize_key(channel_id, target_date, kind)
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT * FROM publications
                WHERE channel_id = ? AND target_date = ? AND kind = ?
                """,
                (channel, iso_date, normalized_kind),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def assert_publishable(
        self,
        channel_id: str,
        target_date: date | str,
        kind: str,
        payload_text: str,
    ) -> PublicationRecord | None:
        """Preflight one payload and fail before any unrelated message is sent.

        A matching sent row is publishable as an idempotent no-op.  Pending and
        failed rows are intentionally blocked because Telegram delivery may have
        occurred even when the local process did not receive confirmation.
        """

        record = self.get(channel_id, target_date, kind)
        if record is None:
            return None
        if record.payload_sha256 != payload_digest(payload_text):
            raise PayloadConflictError("A different payload already exists for this channel/date/kind.")
        if record.status != "sent":
            raise AmbiguousPublicationError("This publication key is fail-closed after an unconfirmed attempt.")
        return record

    def reserve(
        self,
        channel_id: str,
        target_date: date | str,
        kind: str,
        source_run_id: str,
        payload_text: str,
    ) -> Reservation:
        """Atomically reserve a payload before the corresponding network call."""

        channel, iso_date, normalized_kind = _normalize_key(channel_id, target_date, kind)
        normalized_run_id = source_run_id.strip()
        if not normalized_run_id:
            raise ValueError("source_run_id must not be empty.")
        if not payload_text or not payload_text.strip():
            raise ValueError("payload_text must not be empty.")

        digest = payload_digest(payload_text)
        reservation_id = uuid.uuid4().hex
        reserved_at = _utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM publications
                WHERE channel_id = ? AND target_date = ? AND kind = ?
                """,
                (channel, iso_date, normalized_kind),
            ).fetchone()
            if row is not None:
                record = self._record_from_row(row)
                if record.payload_sha256 != digest:
                    raise PayloadConflictError("A different payload already exists for this channel/date/kind.")
                if record.status != "sent":
                    raise AmbiguousPublicationError("This publication key is fail-closed after an unconfirmed attempt.")
                connection.commit()
                return Reservation(record=record, created=False)

            connection.execute(
                """
                INSERT INTO publications (
                    channel_id, target_date, kind, source_run_id,
                    payload_sha256, payload_text, status, reservation_id,
                    reserved_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    channel,
                    iso_date,
                    normalized_kind,
                    normalized_run_id,
                    digest,
                    payload_text,
                    reservation_id,
                    reserved_at,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        record = PublicationRecord(
            channel_id=channel,
            target_date=iso_date,
            kind=normalized_kind,
            source_run_id=normalized_run_id,
            payload_sha256=digest,
            payload_text=payload_text,
            status="pending",
            reservation_id=reservation_id,
            reserved_at_utc=reserved_at,
        )
        return Reservation(record=record, created=True)

    def mark_sent(self, reservation: Reservation, *, message_id: int) -> PublicationRecord:
        """Finalize a pending reservation after a confirmed Telegram response."""

        if not reservation.created:
            return reservation.record
        if not isinstance(message_id, int):
            raise TypeError("message_id must be an integer.")
        sent_at = _utc_now()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE publications
                SET status = 'sent', sent_at_utc = ?, message_id = ?
                WHERE channel_id = ? AND target_date = ? AND kind = ?
                  AND reservation_id = ? AND status = 'pending'
                """,
                (
                    sent_at,
                    message_id,
                    reservation.record.channel_id,
                    reservation.record.target_date,
                    reservation.record.kind,
                    reservation.record.reservation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidStateTransitionError("The publication reservation is stale or no longer pending.")
        completed = self.get(
            reservation.record.channel_id,
            reservation.record.target_date,
            reservation.record.kind,
        )
        if completed is None:
            raise InvalidStateTransitionError("The sent publication record disappeared.")
        return completed

    def mark_failed(self, reservation: Reservation, *, failure_code: str = "delivery_unconfirmed") -> PublicationRecord:
        """Permanently fail-close a pending reservation after send uncertainty."""

        if not reservation.created:
            raise InvalidStateTransitionError("An existing sent publication cannot be marked failed.")
        normalized_code = failure_code.strip() or "delivery_unconfirmed"
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE publications
                SET status = 'failed', failure_code = ?
                WHERE channel_id = ? AND target_date = ? AND kind = ?
                  AND reservation_id = ? AND status = 'pending'
                """,
                (
                    normalized_code,
                    reservation.record.channel_id,
                    reservation.record.target_date,
                    reservation.record.kind,
                    reservation.record.reservation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidStateTransitionError("The publication reservation is stale or no longer pending.")
        failed = self.get(
            reservation.record.channel_id,
            reservation.record.target_date,
            reservation.record.kind,
        )
        if failed is None:
            raise InvalidStateTransitionError("The failed publication record disappeared.")
        return failed

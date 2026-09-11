"""Coordinate exactly two idempotent CS2 Telegram publications per day.

The caller supplies fully rendered ``best`` and ``others`` messages plus the
immutable source run identifier.  The publisher reserves each database key
before network I/O and never retries uncertain deliveries automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .client import TelegramSendResult
from .state import PublishStateStore


class MessageClient(Protocol):
    """Describe the transport behavior needed by the daily publisher."""

    def send_message(self, chat_id: str, text: str, *, parse_mode: str | None = None) -> TelegramSendResult:
        """Send one message and return a confirmed Telegram message identifier."""


class PublicationDeliveryError(RuntimeError):
    """Report a fail-closed delivery without exposing transport secrets."""


@dataclass(frozen=True)
class MessagePublishOutcome:
    """Describe the durable outcome for one of the two daily messages."""

    kind: str
    status: str
    message_id: int | None = None


@dataclass(frozen=True)
class PublishOutcome:
    """Describe the combined result of the best and others publications."""

    target_date: str
    source_run_id: str
    best: MessagePublishOutcome
    others: MessagePublishOutcome
    dry_run: bool


def _iso_target_date(target_date: date | str) -> str:
    """Return a strict ISO target date for the public outcome object."""

    if isinstance(target_date, date):
        return target_date.isoformat()
    try:
        return date.fromisoformat(target_date.strip()).isoformat()
    except (AttributeError, ValueError):
        raise ValueError("target_date must be a date or YYYY-MM-DD string.") from None


def _deliver_one(
    *,
    client: MessageClient,
    store: PublishStateStore,
    channel_id: str,
    target_date: date | str,
    kind: str,
    source_run_id: str,
    message: str,
) -> MessagePublishOutcome:
    """Reserve, deliver and finalize exactly one Telegram publication."""

    reservation = store.reserve(
        channel_id,
        target_date,
        kind,
        source_run_id,
        message,
    )
    if not reservation.created:
        return MessagePublishOutcome(
            kind=kind,
            status="already_sent",
            message_id=reservation.record.message_id,
        )

    try:
        result = client.send_message(channel_id, message, parse_mode="HTML")
    except Exception:
        try:
            store.mark_failed(reservation, failure_code="delivery_unconfirmed")
        except Exception:
            # The original pending row still blocks automatic retries.  Do not
            # replace the sanitized error with a database traceback.
            pass
        raise PublicationDeliveryError(
            f"Telegram {kind} delivery was not confirmed; automatic retry is blocked."
        ) from None

    try:
        store.mark_sent(reservation, message_id=result.message_id)
    except Exception:
        # Telegram confirmed the send, while local finalization was uncertain.
        # Keeping the reservation pending is deliberately fail-closed.
        raise PublicationDeliveryError(
            f"Telegram confirmed {kind}, but local finalization failed; retry is blocked."
        ) from None
    return MessagePublishOutcome(kind=kind, status="sent", message_id=result.message_id)


def publish_daily_messages(
    *,
    client: MessageClient,
    store: PublishStateStore | None,
    channel_id: str,
    target_date: date | str,
    source_run_id: str,
    best_message: str,
    others_message: str,
    dry_run: bool = False,
) -> PublishOutcome:
    """Publish exactly ``best`` then ``others`` with durable idempotency.

    Both payloads are preflighted before any send.  This prevents a conflict or
    an ambiguous prior attempt for the second message from allowing a new first
    message to be delivered.  A dry run validates only in-memory inputs,
    performs no state call and makes no network call; callers may pass
    ``store=None`` to prove that it cannot create a database.
    """

    iso_date = _iso_target_date(target_date)
    normalized_run_id = source_run_id.strip()
    if not normalized_run_id:
        raise ValueError("source_run_id must not be empty.")
    if not best_message or not best_message.strip():
        raise ValueError("best_message must not be empty.")
    if not others_message or not others_message.strip():
        raise ValueError("others_message must not be empty.")

    if dry_run:
        return PublishOutcome(
            target_date=iso_date,
            source_run_id=normalized_run_id,
            best=MessagePublishOutcome(kind="best", status="dry_run"),
            others=MessagePublishOutcome(kind="others", status="dry_run"),
            dry_run=True,
        )

    if store is None:
        raise ValueError("store is required unless dry_run=True.")

    store.assert_publishable(channel_id, iso_date, "best", best_message)
    store.assert_publishable(channel_id, iso_date, "others", others_message)

    best_outcome = _deliver_one(
        client=client,
        store=store,
        channel_id=channel_id,
        target_date=iso_date,
        kind="best",
        source_run_id=normalized_run_id,
        message=best_message,
    )
    others_outcome = _deliver_one(
        client=client,
        store=store,
        channel_id=channel_id,
        target_date=iso_date,
        kind="others",
        source_run_id=normalized_run_id,
        message=others_message,
    )
    return PublishOutcome(
        target_date=iso_date,
        source_run_id=normalized_run_id,
        best=best_outcome,
        others=others_outcome,
        dry_run=False,
    )

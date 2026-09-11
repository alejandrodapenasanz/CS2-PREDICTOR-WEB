"""Minimal HTTPS client for the Telegram Bot API ``sendMessage`` endpoint.

Only the Python standard library is used.  Transport failures are deliberately
wrapped in sanitized exceptions so a URL containing the bot token can never be
printed by callers or test runners.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable
from urllib import parse, request


TELEGRAM_TEXT_LIMIT = 4096


class TelegramAPIError(RuntimeError):
    """Represent an unconfirmed or rejected Telegram delivery safely."""


@dataclass(frozen=True)
class TelegramSendResult:
    """Confirmed identifiers returned by Telegram after sending a message."""

    chat_id: str
    message_id: int
    date_unix: int | None = None


def _urlopen_transport(http_request: request.Request, *, timeout: float) -> Any:
    """Open a Telegram HTTPS request using the standard-library transport."""

    return request.urlopen(http_request, timeout=timeout)


def _read_transport_response(response: Any) -> bytes:
    """Read bytes from a urllib-compatible response object."""

    if isinstance(response, bytes):
        return response
    if isinstance(response, str):
        return response.encode("utf-8")
    if hasattr(response, "__enter__"):
        with response as opened:
            return opened.read()
    if hasattr(response, "read"):
        return response.read()
    raise TypeError("Transport returned an unsupported response type.")


class TelegramClient:
    """Send plain or Telegram-formatted messages through the official Bot API."""

    def __init__(
        self,
        bot_token: str,
        *,
        timeout_seconds: float = 20.0,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        """Create a client with an injectable transport for offline tests."""

        if not bot_token or not bot_token.strip():
            raise ValueError("A non-empty Telegram bot token is required.")
        if timeout_seconds <= 0:
            raise ValueError("Telegram request timeout must be greater than zero.")
        self._bot_token = bot_token.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport or _urlopen_transport

    def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
        disable_notification: bool = False,
    ) -> TelegramSendResult:
        """Send one message and return only after Telegram confirms delivery.

        Network and response errors deliberately discard their original message
        and traceback chain because urllib errors may include the secret-bearing
        request URL.  Callers must treat every such failure as ambiguous.
        """

        normalized_chat_id = chat_id.strip()
        if not normalized_chat_id:
            raise ValueError("Telegram chat_id must not be empty.")
        if not text or not text.strip():
            raise ValueError("Telegram message text must not be empty.")
        if len(text) > TELEGRAM_TEXT_LIMIT:
            raise ValueError(f"Telegram message exceeds the {TELEGRAM_TEXT_LIMIT}-character limit.")

        fields: dict[str, str] = {
            "chat_id": normalized_chat_id,
            "text": text,
            "disable_notification": "true" if disable_notification else "false",
        }
        if parse_mode:
            fields["parse_mode"] = parse_mode
        encoded_body = parse.urlencode(fields).encode("utf-8")
        endpoint = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        http_request = request.Request(
            endpoint,
            data=encoded_body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "User-Agent": "CS2-Predictor-Telegram/1.0",
            },
            method="POST",
        )

        try:
            raw_response = _read_transport_response(self._transport(http_request, timeout=self._timeout_seconds))
        except Exception:
            raise TelegramAPIError("Telegram delivery failed before a response could be confirmed.") from None

        try:
            payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TelegramAPIError("Telegram returned an invalid response.") from None

        if not isinstance(payload, dict) or payload.get("ok") is not True:
            error_code = payload.get("error_code") if isinstance(payload, dict) else None
            suffix = f" (error_code={error_code})" if isinstance(error_code, int) else ""
            raise TelegramAPIError(f"Telegram rejected the message{suffix}.")

        result = payload.get("result")
        message_id = result.get("message_id") if isinstance(result, dict) else None
        date_unix = result.get("date") if isinstance(result, dict) else None
        if not isinstance(message_id, int):
            raise TelegramAPIError("Telegram response did not contain a confirmed message identifier.")
        return TelegramSendResult(
            chat_id=normalized_chat_id,
            message_id=message_id,
            date_unix=date_unix if isinstance(date_unix, int) else None,
        )

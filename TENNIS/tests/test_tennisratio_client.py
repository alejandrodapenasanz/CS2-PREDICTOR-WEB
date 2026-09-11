"""Offline contracts for the restricted TennisRatio Scrapling client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from src.tennis_explorer.transport import ScraplingTransportError
from src.tennisratio.client import TennisRatioClient
from src.tennisratio.types import TennisRatioHttpError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _Response:
    """Minimal normalized Scrapling response used by the client."""

    status_code: int
    content: bytes
    headers: dict[str, str]


class _Session:
    """Record one requests-like GET or raise a configured transport error."""

    def __init__(self, result: _Response | BaseException) -> None:
        """Store the deterministic response and initialize the call log."""

        self.result = result
        self.calls: list[tuple[str, dict[str, str], tuple[float, float], bool]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> _Response:
        """Record the strict GET contract and return the configured result."""

        self.calls.append((url, dict(headers), timeout, allow_redirects))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_default_client_uses_scrapling_and_accepts_lowercase_headers() -> None:
    """The production path constructs Scrapling while fixtures remain injectable."""

    session = _Session(
        _Response(
            status_code=200,
            content=b"User-agent: *\nDisallow: /api/\n",
            headers={"content-type": "text/plain; charset=utf-8"},
        )
    )
    observed = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)
    with (
        TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary,
        mock.patch(
            "src.tennisratio.client.ScraplingHttpSession",
            return_value=session,
        ) as factory,
    ):
        payload = TennisRatioClient(cache_root=Path(temporary)).get(
            "/robots.txt", retrieved_at_utc=observed
        )

    factory.assert_called_once_with(
        browser_impersonation=True,
        stealthy_headers=True,
        retry_attempts=3,
        retry_delay_seconds=1.0,
    )
    assert payload.content_type == "text/plain"
    assert payload.retrieved_at_utc == observed
    assert len(session.calls) == 1
    assert session.calls[0][0] == "https://www.tennisratio.com/robots.txt"
    assert session.calls[0][3] is False


def test_scrapling_transport_error_is_exposed_as_domain_error() -> None:
    """A curl/Scrapling failure is handled by the TennisRatio CLI contract."""

    session = _Session(ScraplingTransportError("offline"))
    with pytest.raises(TennisRatioHttpError, match="Scrapling GET failed"):
        TennisRatioClient(session=session).get(
            "/robots.txt",
            retrieved_at_utc=datetime(2026, 8, 24, 8, 0, tzinfo=UTC),
        )

    assert len(session.calls) == 1

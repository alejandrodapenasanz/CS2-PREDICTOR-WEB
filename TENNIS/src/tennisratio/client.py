"""Restricted Scrapling client for the approved public TennisRatio surface.

Only GET is implemented.  The allow-list deliberately excludes ``/api/`` and
rejects redirects so a server-side change cannot silently expand authority.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urljoin, urlsplit

import requests

from ..responsible_http import (
    DEFAULT_CACHE_ROOT,
    RateLimitedError,
    ResponsibleHttpClient,
    WafBlockedError,
    build_http_client,
)
from ..tennis_explorer.transport import ScraplingHttpSession, ScraplingTransportError
from .types import HttpPayload, TENNISRATIO_BASE_URL, TennisRatioBlockedError, TennisRatioHttpError


_STATIC_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/robots.txt",
        "/atp-matches.html",
        "/wta-matches.html",
        "/sitemap-players.xml",
    }
)
_PROFILE_PATH: Final[re.Pattern[str]] = re.compile(r"^/players/[A-Za-z0-9_-]+\.html$")
_MIN_BYTES: Final[int] = 16
_MAX_BYTES: Final[int] = 32 * 1024 * 1024


class _Response(Protocol):
    """Minimal requests-compatible response used by Scrapling and fixtures."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _Session(Protocol):
    """Minimal requests-compatible session; only GET is authorized."""

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> _Response:
        """Issue one HTTP GET."""


class TennisRatioClient:
    """Fetch validated bytes through Scrapling from a hard public allow-list."""

    def __init__(
        self,
        *,
        session: requests.Session | _Session | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a client without issuing network requests."""

        if connect_timeout_seconds <= 0.0 or read_timeout_seconds <= 0.0:
            raise ValueError("HTTP timeouts must be positive.")
        self._session = cast(
            _Session,
            session
            or build_http_client(
                transport_kind="scrapling",
                transport=ScraplingHttpSession(
                    browser_impersonation=True,
                    stealthy_headers=True,
                    retry_attempts=3,
                    retry_delay_seconds=1.0,
                ),
                cache_root=cache_root,
                robots_user_agent="*",
                detect_waf=True,
                browser_impersonation=True,
                stealthy_headers=True,
                request_retry_attempts=3,
                request_retry_delay_seconds=1.0,
            ),
        )
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)
        self._clock = clock

    def get(
        self,
        path_or_url: str,
        *,
        retrieved_at_utc: datetime,
        cache_mode: Literal["default", "refresh", "no_store"] = "default",
    ) -> HttpPayload:
        """Perform one non-redirecting GET and validate status/type/body."""

        source_url = authorized_url(path_or_url)
        if retrieved_at_utc.tzinfo is None:
            raise ValueError("retrieved_at_utc must be timezone-aware.")
        observed = retrieved_at_utc.astimezone(UTC)
        try:
            request = {
                "headers": {"Accept": _accept_header(source_url)},
                "timeout": self._timeout,
                "allow_redirects": False,
            }
            if isinstance(self._session, ResponsibleHttpClient):
                response = self._session.get(
                    source_url,
                    **request,
                    cache_mode=cache_mode,
                )
            else:
                response = self._session.get(source_url, **request)
        except (RateLimitedError, WafBlockedError) as exc:
            raise TennisRatioBlockedError(
                f"TennisRatio detenido; último lote válido conservado: {exc}"
            ) from exc
        except (requests.RequestException, ScraplingTransportError) as exc:
            raise TennisRatioHttpError(f"Scrapling GET failed for {source_url}.") from exc
        except OSError as exc:
            raise TennisRatioHttpError(f"Transport failed for {source_url}.") from exc
        if response.status_code != 200:
            raise TennisRatioHttpError(
                f"GET {source_url} returned HTTP {response.status_code}; redirects are forbidden."
            )
        response_url = getattr(response, "url", source_url)
        if authorized_url(response_url) != source_url:
            raise TennisRatioHttpError(
                f"Response URL {response_url!r} differs from requested URL {source_url!r}."
            )
        content = bytes(response.content)
        if not (_MIN_BYTES <= len(content) <= _MAX_BYTES):
            raise TennisRatioHttpError(
                f"Response size {len(content)} outside validated bounds for {source_url}."
            )
        content_type = (
            _header_value(response.headers, "Content-Type").split(";", 1)[0].strip().lower()
        )
        _validate_content_type(source_url, content_type)
        _validate_body_marker(source_url, content)
        if self._clock is not None:
            completed_at = self._clock()
            if completed_at.tzinfo is None:
                raise ValueError("HTTP completion clock must be timezone-aware.")
            observed = max(observed, completed_at.astimezone(UTC))
        return HttpPayload(
            source_url=source_url,
            content=content,
            content_type=content_type,
            retrieved_at_utc=observed,
            sha256=hashlib.sha256(content).hexdigest(),
            etag=_optional_header(response.headers, "ETag"),
            last_modified=_optional_header(response.headers, "Last-Modified"),
        )

    def close(self) -> None:
        """Cierra el cliente común cuando esta instancia es su propietaria."""

        close = getattr(self._session, "close", None)
        if callable(close):
            close()


def authorized_url(path_or_url: str) -> str:
    """Return a canonical URL if and only if it belongs to the approved GET set."""

    if not isinstance(path_or_url, str) or not path_or_url.strip():
        raise TennisRatioHttpError("Requested path/URL must be non-empty text.")
    source_url = urljoin(f"{TENNISRATIO_BASE_URL}/", path_or_url.strip())
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "www.tennisratio.com"
        or parsed.query
        or parsed.fragment
        or (parsed.path not in _STATIC_PATHS and _PROFILE_PATH.fullmatch(parsed.path) is None)
    ):
        raise TennisRatioHttpError(
            f"URL is outside the approved TennisRatio GET set: {source_url!r}."
        )
    return source_url


def _accept_header(source_url: str) -> str:
    """Select a narrow response content type for the authorized resource."""

    path = urlsplit(source_url).path
    if path == "/robots.txt":
        return "text/plain"
    if path == "/sitemap-players.xml":
        return "application/xml,text/xml;q=0.9"
    return "text/html"


def _validate_content_type(source_url: str, content_type: str) -> None:
    """Reject HTML error bodies masquerading as XML/text and vice versa."""

    path = urlsplit(source_url).path
    if path == "/robots.txt":
        allowed = {"text/plain"}
    elif path == "/sitemap-players.xml":
        allowed = {"application/xml", "text/xml", "application/octet-stream"}
    else:
        allowed = {"text/html", "application/xhtml+xml"}
    if content_type not in allowed:
        raise TennisRatioHttpError(f"Unexpected Content-Type {content_type!r} for {source_url}.")


def _validate_body_marker(source_url: str, content: bytes) -> None:
    """Apply inexpensive response-family markers before a strict parser runs."""

    lowered = content[:262_144].lower()
    path = urlsplit(source_url).path
    if path == "/robots.txt":
        marker_ok = b"user-agent" in lowered
    elif path == "/sitemap-players.xml":
        marker_ok = b"<urlset" in lowered and b"<loc>" in lowered
    elif path.startswith("/players/"):
        marker_ok = b"window.playerdata" in lowered and b"window.matchesdata" in content.lower()
    else:
        marker_ok = b"matches-date-group" in lowered and b"tournament-group" in lowered
    if not marker_ok:
        raise TennisRatioHttpError(f"Response body markers are invalid for {source_url}.")


def _header_value(headers: Mapping[str, str], name: str) -> str:
    """Read one HTTP header without depending on backend key casing."""

    expected = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == expected:
            return str(value)
    return ""


def _optional_header(headers: Mapping[str, str], name: str) -> str | None:
    """Normalize an optional HTTP audit header."""

    value = _header_value(headers, name).strip()
    return value or None

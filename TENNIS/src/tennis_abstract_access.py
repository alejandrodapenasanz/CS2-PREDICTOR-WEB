"""Scrapling access for explicitly authorized Tennis Abstract acquisition.

HTTP is the first choice; the operator-authorized browser can recover a
challenge once, retaining the common client, robots scope and host pacing.
Downloaded history literals are never executed by the data parser.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import parse_qsl, urlsplit

from .config import PROJECT_ROOT
from .responsible_http import (
    DEFAULT_CACHE_ROOT,
    HttpResponse,
    RateLimitedError,
    ResponsibleHttpClient,
    WafBlockedError,
    build_http_client,
    _retry_after_seconds,
)


ACCESS_CONFIG = PROJECT_ROOT / "config" / "tennis_abstract_access.json"
ORIGIN = "https://www.tennisabstract.com"
_PROFILE_PATHS = {
    "/cgi-bin/player.cgi",
    "/cgi-bin/wplayer.cgi",
    "/cgi-bin/player-classic.cgi",
    "/cgi-bin/wplayer-classic.cgi",
}
_STATIC_PATHS = {
    "/",
    "/robots.txt",
    "/mwplayerlist.js",
    "/reports/atp_elo_ratings.html",
    "/reports/wta_elo_ratings.html",
}
_PLAYER_KEY = re.compile(r"(?:[0-9]+/)?[A-Za-z][A-Za-z0-9_-]*")
_DATA_PATH = re.compile(r"/(?:jsfrags|jsmatches)/[A-Za-z][A-Za-z0-9_-]*\.js")
_RANK_PATHS = {"/jsplayers/curr_rank_atp.js", "/jsplayers/curr_rank_wta.js"}
ACCESS_RECOVERY_VERSION = 1


def browser_recovery_authorized(config_path: Path = ACCESS_CONFIG) -> bool:
    """Require the explicit dated operator opt-in, independently of data-route access."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (
        config.get("operator_authorization_confirmed") is True
        and config.get("browser_recovery_enabled") is True
        and bool(config.get("browser_authorized_on"))
    )


def validate_acquisition_url(url: str) -> str:
    """Accept only inspected player/data routes on the exact HTTPS origin."""

    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.tennisabstract.com"
        or parsed.fragment
        or any(char.isspace() for char in url)
        or "\\" in url
    ):
        raise ValueError("Tennis Abstract URL outside the authorized origin.")
    if parsed.path in _PROFILE_PATHS:
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if len(query) == 1 and query[0][0] == "p" and _PLAYER_KEY.fullmatch(query[0][1]):
            return url
    elif not parsed.query and (
        parsed.path in _STATIC_PATHS
        or _DATA_PATH.fullmatch(parsed.path)
        or parsed.path in _RANK_PATHS
    ):
        return url
    raise ValueError("Tennis Abstract URL outside the inspected acquisition paths.")


def operator_data_exception(url: str, *, confirmed: bool) -> bool:
    """Apply the declared permission only to narrow data URLs, never whole hosts."""

    if not confirmed:
        return False
    try:
        validate_acquisition_url(url)
    except ValueError:
        return False
    path = urlsplit(url).path
    return bool(_DATA_PATH.fullmatch(path) or path in _RANK_PATHS)


class TennisAbstractAcquisitionClient:
    """Fetch authorized static documents with the common Scrapling transport."""

    def __init__(
        self,
        *,
        config_path: Path = ACCESS_CONFIG,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        start_in_browser: bool = False,
    ) -> None:
        """Load explicit operator scope before constructing a network client."""

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported Tennis Abstract authorization schema.")
        confirmed = config.get("operator_authorization_confirmed")
        if not isinstance(confirmed, bool):
            raise ValueError("operator_authorization_confirmed must be boolean.")
        if confirmed and not (config.get("confirmed_on") and config.get("authorization_basis")):
            raise ValueError("Tennis Abstract permission needs dated operator provenance.")
        self._client = build_http_client(
            transport_kind="scrapling",
            cache_root=cache_root,
            detect_waf=True,
            browser_impersonation=True,
            stealthy_headers=True,
            request_retry_attempts=1,
            robots_exception=lambda url: operator_data_exception(url, confirmed=confirmed),
        )
        self._blocked = False
        self._cache_root = cache_root
        self._confirmed = confirmed
        self._browser_enabled = browser_recovery_authorized(config_path)
        self._config = config
        self._browser_used = False
        self._browser_client: ResponsibleHttpClient | None = None
        if start_in_browser:
            if not self._browser_enabled:
                self._client.close()
                raise ValueError("Browser recovery requires dated operator authorization.")
            self._activate_browser()

    def _activate_browser(self) -> None:
        """Switch once to a persistent browser through the SAME HTTP policy layer."""

        from .tennis_abstract_browser import TennisAbstractBrowserTransport

        if self._browser_used:
            raise ValueError("Browser recovery already attempted in this session.")
        self._browser_used = True
        transport = TennisAbstractBrowserTransport(
            headless=self._config.get("browser_headless", False),
            timeout_seconds=float(self._config.get("browser_timeout_seconds", 90)),
        )
        self._browser_client = build_http_client(
            transport_kind="scrapling",
            transport=transport,
            cache_root=self._cache_root,
            detect_waf=True,
            robots_exception=lambda url: operator_data_exception(url, confirmed=self._confirmed),
        )

    def get(self, url: str) -> HttpResponse:
        """Validate every result; recover a challenge once, never a server rate-limit."""

        validate_acquisition_url(url)
        if self._blocked:
            raise ValueError("Tennis Abstract acquisition stopped after WAF/rate-limit.")
        try:
            active = self._browser_client or self._client
            response = active.get(url, timeout=(10.0, 30.0))
        except WafBlockedError as exc:
            retry_after = _retry_after_seconds(
                next(
                    (
                        value
                        for key, value in exc.response.headers.items()
                        if key.lower() == "retry-after"
                    ),
                    None,
                ),
                now=time.time(),
            )
            if retry_after is not None and retry_after > 0:
                self._blocked = True
                raise RateLimitedError(
                    url, time.time() + retry_after, fresh_response=False
                ) from exc
            if not self._browser_enabled or self._browser_used:
                self._blocked = True
                raise
            logging.getLogger(__name__).warning(
                "Tennis Abstract: challenge detectado; recuperacion autorizada con navegador."
            )
            self._activate_browser()
            try:
                assert self._browser_client is not None
                response = self._browser_client.get(url, timeout=(10.0, 30.0))
            except Exception:
                self._blocked = True
                raise
        except RateLimitedError:
            self._blocked = True
            raise
        if response.status_code != 200:
            raise ValueError(f"Tennis Abstract GET returned {response.status_code}: {url}")
        if not 16 <= len(response.content) <= 32 * 1024 * 1024:
            raise ValueError("Tennis Abstract document outside inspected size bounds.")
        return response

    def close(self) -> None:
        """Release the common client without mutating any domain database."""

        self._client.close()
        if self._browser_client is not None:
            self._browser_client.close()

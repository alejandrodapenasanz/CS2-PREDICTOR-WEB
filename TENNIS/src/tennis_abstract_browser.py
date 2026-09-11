"""Operator-authorized, bounded Scrapling browser transport for Tennis Abstract.

Source requests still go through ResponsibleHttpClient. A dedicated Chrome
profile retains server-issued clearance; source payloads are captured from
network responses, not from a mutated DOM or the initial challenge headers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import logging
from pathlib import Path
import random
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .config import PROJECT_ROOT
from .responsible_http import (
    HttpResponse,
    ResponsibleHttpError,
    _looks_like_waf,
    _retry_after_seconds,
)

PROFILE_ROOT = PROJECT_ROOT / "data" / "raw" / "_browser_profiles" / "tennis_abstract"
_PAGE_ASSETS = {
    "/blue/style.css",
    "/style.css",
    "/favicon.ico",
    "/navbar.js",
    "/jquery-1.7.1-min.js",
    "/jquery.ui.core.js",
    "/jquery.ui.position.js",
    "/jquery.ui.widget.js",
    "/jquery.ui.autocomplete.js",
    "/jquery.tablesorter.js",
}


class TennisAbstractPlayerTimeout(ResponsibleHttpError):
    """A verified navigation timeout without observed WAF or server cooldown evidence."""

    def __init__(self, url: str) -> None:
        """Keep only the validated source URL; do not expose browser tokens or internals."""

        self.url = url
        super().__init__(f"Timeout de navegacion de una ficha Tennis Abstract: {url}")


def is_navigation_timeout(error: Exception) -> bool:
    """Recognize the installed browser's Page.goto timeout, never a solver/startup deadline."""

    return (
        type(error).__name__ == "TimeoutError"
        and type(error).__module__ in {"patchright._impl._errors", "playwright._impl._errors"}
        and str(error).startswith("Page.goto:")
    )


def request_allowed(url: str, target: str, *, method: str, main_frame: bool) -> bool:
    """Allow the requested document and narrow browser/challenge dependencies only."""

    parsed, expected = urlsplit(url), urlsplit(target)
    if (
        parsed.scheme != "https"
        or parsed.netloc
        not in {"www.tennisabstract.com", "tennisabstract.com", "challenges.cloudflare.com"}
        or parsed.fragment
        or "\\" in url
        or any(char.isspace() for char in url)
    ):
        return False
    if main_frame:
        # Cloudflare may navigate through a temporary query token on the SAME page.
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query)
            if not key.startswith("__cf_chl_")
        ]
        return (
            method == "GET"
            and parsed.netloc == expected.netloc
            and parsed.path == expected.path
            and query == parse_qsl(expected.query)
        )
    if parsed.path.startswith("/cdn-cgi/challenge-platform/"):
        return method in {"GET", "POST"}
    if parsed.netloc == "challenges.cloudflare.com":
        return method == "GET" and parsed.path.startswith("/turnstile/")
    if method != "GET" or parsed.path.startswith(("/jsmatches/", "/jsfrags/", "/jsplayers/")):
        # Player data is fetched explicitly by the common client, with its own pacing.
        return False
    return parsed.path in _PAGE_ASSETS


class TennisAbstractBrowserTransport:
    """Adapt one persistent async Scrapling browser to the common sync GET protocol."""

    def __init__(
        self,
        *,
        profile_root: Path = PROFILE_ROOT,
        headless: bool = False,
        timeout_seconds: float = 90.0,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Configure lazy browser startup; never install binaries or use a personal profile."""

        if not 10 <= timeout_seconds <= 180:
            raise ValueError("Browser timeout must be between 10 and 180 seconds.")
        self._profile = profile_root.resolve()
        self._headless = headless
        self._timeout = timeout_seconds
        self._factory = session_factory
        self._session: Any = None
        self._runner = asyncio.Runner()
        self._closed = False

    async def _start(self) -> None:
        """Open a project-owned Chrome profile using the declared Scrapling dependency."""

        if self._session is not None:
            return
        if self._factory is None:
            from scrapling.fetchers import AsyncStealthySession

            self._factory = AsyncStealthySession
        self._profile.mkdir(parents=True, exist_ok=True)
        random.seed(42)
        logging.getLogger(__name__).warning(
            "Tennis Abstract: abriendo Chrome %s para resolver el challenge. "
            "Puede intervenir en la pantalla si se solicita; limite total %.0fs.",
            "visible" if not self._headless else "sin ventana",
            self._timeout,
        )
        self._session = self._factory(
            headless=self._headless,
            real_chrome=True,
            user_data_dir=str(self._profile),
            solve_cloudflare=True,
            timeout=30000,
            retries=1,
            google_search=False,
            load_dom=False,
            network_idle=False,
            max_pages=1,
            additional_args={"service_workers": "block"},
            extra_flags=["--disk-cache-size=10485760"],
        )
        await self._session.__aenter__()

    async def _fetch(self, url: str, headers: dict[str, str]) -> HttpResponse:
        """Capture final document bytes while bounding browser navigation and requests."""

        await self._start()
        last_response: Any = None
        captured: HttpResponse | None = None
        server_limit: HttpResponse | None = None
        hook_failure: Exception | None = None
        protection_seen = False
        body_verified = False
        last_request: dict[str, float] = {}
        host_locks: dict[str, asyncio.Lock] = {}

        async def setup(page: Any) -> None:
            """Gate browser dependencies and capture the actual final navigation response."""

            nonlocal hook_failure

            async def guard(route: Any) -> None:
                """Deny unrelated URLs/methods and pace browser dependencies per host."""

                request = route.request
                main = request.is_navigation_request() and request.frame == page.main_frame
                if server_limit or not request_allowed(
                    request.url, url, method=request.method, main_frame=main
                ):
                    await route.abort()
                    return
                host = urlsplit(request.url).netloc
                async with host_locks.setdefault(host, asyncio.Lock()):
                    await asyncio.sleep(
                        max(0.0, 1.0 - (time.monotonic() - last_request.get(host, 0.0)))
                    )
                    last_request[host] = time.monotonic()
                    await route.continue_()

            async def record(response: Any) -> None:
                """Keep source navigation evidence; stop immediately on server Retry-After."""

                nonlocal last_response, server_limit, protection_seen, body_verified
                request = response.request
                if not request.is_navigation_request() or request.frame != page.main_frame:
                    return
                last_response = response
                body_verified = False
                headers = {key.lower(): value for key, value in response.headers.items()}
                delay = _retry_after_seconds(headers.get("retry-after"), now=time.time())
                protection_seen |= (
                    response.status in {401, 403, 429}
                    or headers.get("cf-mitigated", "").strip().casefold() == "challenge"
                    or (delay is not None and delay > 0)
                )
                if response.status == 429 and delay is not None and delay > 0:
                    server_limit = HttpResponse(429, b"", headers, url)
                    await page.close()
                    return
                try:
                    body = await response.body()
                except Exception:
                    # An unknown/incomplete body cannot prove a timeout is not a challenge.
                    return
                protection_seen |= _looks_like_waf(
                    HttpResponse(int(response.status), body, headers, url)
                )
                if last_response is response:
                    body_verified = True

            try:
                await page.route("**/*", guard)
                page.on("response", record)
            except Exception as exc:
                hook_failure = exc
                await page.close()

        async def capture(page: Any) -> None:
            """Do not reuse the first challenge's headers or serialize rendered HTML."""

            nonlocal captured, hook_failure
            try:
                if last_response is None or last_response.url != url or page.url != url:
                    raise ValueError("Browser did not finish on the exact requested source URL.")
                captured = HttpResponse(
                    int(last_response.status),
                    await last_response.body(),
                    await last_response.all_headers(),
                    url,
                )
            except Exception as exc:
                hook_failure = exc

        try:
            await self._session.fetch(
                url, page_setup=setup, page_action=capture, extra_headers=headers
            )
        except Exception as exc:
            if server_limit is not None:
                return server_limit
            if (
                is_navigation_timeout(exc)
                and not protection_seen
                and hook_failure is None
                and (last_response is None or body_verified)
            ):
                raise TennisAbstractPlayerTimeout(url) from exc
            raise
        if server_limit is not None:
            return server_limit
        if hook_failure is not None or captured is None:
            raise RuntimeError(
                "Browser response hooks did not produce verified source bytes."
            ) from hook_failure
        return captured

    def get(
        self,
        url: str,
        **kwargs: Any,
    ) -> HttpResponse:
        """Execute one bounded browser fetch; the common client validates/caches the result."""

        from .tennis_abstract_access import validate_acquisition_url

        validate_acquisition_url(url)
        if set(kwargs) - {"headers", "timeout", "allow_redirects"}:
            raise ValueError("Unsupported browser transport options.")
        headers = kwargs.get("headers") or {}
        if not isinstance(headers, Mapping):
            raise ValueError("Browser transport headers must be a mapping.")
        if self._closed or kwargs.get("allow_redirects", False):
            raise ValueError("Closed browser or redirects not allowed.")
        try:
            return self._runner.run(self._bounded_fetch(url, dict(headers)))
        except Exception as exc:
            # Do not expose cookies, temporary challenge tokens or browser internals.
            raise RuntimeError(f"Tennis Abstract browser failed ({type(exc).__name__}).") from exc

    async def _bounded_fetch(self, url: str, headers: dict[str, str]) -> HttpResponse:
        """Cancel even a solver loop; Scrapling's per-operation timeout alone is insufficient."""

        return await asyncio.wait_for(self._fetch(url, headers), timeout=self._timeout)

    async def _close(self) -> None:
        """Close only this dedicated session and leave its persistent cookies private."""

        if self._session is not None:
            await asyncio.wait_for(self._session.__aexit__(None, None, None), timeout=15)

    def close(self) -> None:
        """Release the browser and event loop even after a timed-out challenge."""

        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run(self._close())
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Tennis Abstract: fallo al cerrar la sesion dedicada (%s).", type(exc).__name__
            )
        finally:
            self._runner.close()

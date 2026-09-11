"""Offline contracts for the explicitly authorized Tennis Abstract browser recovery."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.responsible_http import (
    HttpResponse,
    RateLimitedError,
    ResponsibleHttpError,
    WafBlockedError,
)
from src.tennis_abstract_access import ORIGIN, TennisAbstractAcquisitionClient
from src.tennis_abstract_browser import (
    TennisAbstractBrowserTransport,
    TennisAbstractPlayerTimeout,
    request_allowed,
)
from src.tennis_abstract_daily import can_recover_local_waf_pause, isolated_player_timeout

URL = ORIGIN + "/cgi-bin/wplayer-classic.cgi?p=JessicaPegula"
BODY = b"<html><script>var matchmx = [['original source bytes']];</script></html>"


class FakeResponse:
    """Provide network bytes and final headers independent of Scrapling's DOM response."""

    def __init__(self, page, *, status=200, headers=None):
        """Attach the exact requested document navigation."""

        self.url = page.url
        self.status = status
        self.headers = headers or {"content-type": "text/html"}
        self.request = SimpleNamespace(frame=page.main_frame, is_navigation_request=lambda: True)

    async def body(self):
        """Return real source bytes, not a serialized browser DOM."""

        return BODY

    async def all_headers(self):
        """Return only this response's headers."""

        return self.headers


class FakePage:
    """Model the small, public async page API used by the adapter."""

    def __init__(self, url):
        """Set the expected final URL and a stable main frame."""

        self.url = url
        self.main_frame = object()
        self.closed = False

    async def route(self, pattern, callback):
        """Record the browser request guard."""

        self.guard = callback

    def on(self, event, callback):
        """Record the response callback."""

        self.record = callback

    async def close(self):
        """Close this page only."""

        self.closed = True


class FakeSession:
    """Emulate challenge -> valid navigation without network or browser installation."""

    def __init__(self, *, rate_limit=False, stuck=False, **options):
        """Record configuration and select deterministic error branches."""

        self.options = options
        self.rate_limit = rate_limit
        self.stuck = stuck
        self.started = 0
        self.closed = 0

    async def __aenter__(self):
        """Count persistent session starts."""

        self.started += 1
        return self

    async def __aexit__(self, *args):
        """Count session cleanup."""

        self.closed += 1

    async def fetch(self, url, *, page_setup, page_action, extra_headers):
        """Supply final headers, or simulate Retry-After or an infinite solver loop."""

        self.received_headers = extra_headers
        if self.stuck:
            await asyncio.Event().wait()
        page = FakePage(url)
        await page_setup(page)
        if self.rate_limit:
            await page.record(FakeResponse(page, status=429, headers={"retry-after": "3600"}))
            assert page.closed
            raise RuntimeError("Browser page closed by the Retry-After guard")
        await page.record(FakeResponse(page, status=403, headers={"cf-mitigated": "challenge"}))
        await page.record(FakeResponse(page))
        await page_action(page)
        # Simulate the upstream wrapper retaining initial challenge metadata.
        return SimpleNamespace(
            status=200, body=b"modified DOM", headers={"cf-mitigated": "challenge"}
        )


def fetch(transport):
    """Use the actual common transport call signature."""

    return transport.get(URL, headers={}, timeout=(10, 30), allow_redirects=False)


def test_final_network_bytes_and_headers_not_initial_challenge_are_used(tmp_path):
    """A recovered response must not be rejected due to stale challenge headers."""

    session = FakeSession()
    factory = Mock(return_value=session)
    transport = TennisAbstractBrowserTransport(profile_root=tmp_path, session_factory=factory)
    try:
        for _ in range(2):
            result = fetch(transport)
            assert result.content == BODY and result.status_code == 200
            assert "cf-mitigated" not in result.headers
        assert session.started == 1
        assert factory.call_args.kwargs["solve_cloudflare"] is True
        assert factory.call_args.kwargs["headless"] is False
        assert factory.call_args.kwargs["retries"] == 1
        assert factory.call_args.kwargs["user_data_dir"] == str(tmp_path.resolve())
    finally:
        transport.close()
    assert session.closed == 1


def test_total_deadline_cancels_solver_and_closes_only_its_session(tmp_path):
    """Timeout applies around the entire fetch, not only individual browser calls."""

    session = FakeSession(stuck=True)
    transport = TennisAbstractBrowserTransport(
        profile_root=tmp_path, session_factory=lambda **kw: session
    )
    transport._timeout = 0.01
    try:
        with pytest.raises(RuntimeError, match="TimeoutError"):
            fetch(transport)
    finally:
        transport.close()
    assert session.closed == 1


def test_browser_stops_on_server_retry_after_before_solving(tmp_path):
    """A genuine server delay survives the browser path and reaches the common client."""

    session = FakeSession(rate_limit=True)
    transport = TennisAbstractBrowserTransport(
        profile_root=tmp_path, session_factory=lambda **kw: session
    )
    try:
        result = fetch(transport)
        assert result.status_code == 429
        assert result.headers["retry-after"] == "3600"
    finally:
        transport.close()


def test_common_robots_headers_are_supported_and_not_silently_dropped(tmp_path):
    """The common robots fetch must work before an ordinary player navigation."""

    session = FakeSession()
    transport = TennisAbstractBrowserTransport(
        profile_root=tmp_path, session_factory=lambda **kw: session
    )
    try:
        headers = {"User-Agent": "CS2-Predictor-TENNIS/1.0", "Accept": "text/plain"}
        result = transport.get(ORIGIN + "/robots.txt", headers=headers, timeout=(10, 30))
        assert result.status_code == 200
        assert session.received_headers == headers
        fetch(transport)
        assert session.received_headers == {}
    finally:
        transport.close()


@pytest.mark.parametrize(
    "url,method,main,allowed",
    [
        (URL, "GET", True, True),
        (URL + "&__cf_chl_tk=temporary", "GET", True, True),
        (URL + "&private=1", "GET", True, False),
        (ORIGIN + "/private.js", "GET", False, False),
        (ORIGIN + "/jsmatches/OtherPlayer.js", "GET", False, False),
        (ORIGIN + "/jquery-1.7.1-min.js", "GET", False, True),
        (ORIGIN + "/cgi-bin/player.cgi?p=Other", "GET", True, False),
        ("https://other.test/collect", "POST", False, False),
        ("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/test", "POST", False, True),
    ],
)
def test_browser_cannot_expand_to_unrelated_data_or_domains(url, method, main, allowed):
    """Challenge dependencies do not authorize unrelated crawling or arbitrary writes."""

    assert request_allowed(url, URL, method=method, main_frame=main) is allowed


def access_config(tmp_path):
    """Create explicit source and browser authorization in an isolated fixture."""

    path = tmp_path / "access.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operator_authorization_confirmed": True,
                "confirmed_on": "2026-09-06",
                "authorization_basis": "fixture",
                "browser_recovery_enabled": True,
                "browser_authorized_on": "2026-09-06",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_http_challenge_switches_once_and_keeps_common_client_policy(tmp_path):
    """All source fetches still use the shared robots/cache/rate-limit layer."""

    static, browser = Mock(), Mock()
    static.get.side_effect = WafBlockedError("challenge", HttpResponse(403, b"blocked", {}, URL))
    browser.get.return_value = HttpResponse(200, BODY, {}, URL)
    with patch(
        "src.tennis_abstract_access.build_http_client", side_effect=[static, browser]
    ) as build:
        client = TennisAbstractAcquisitionClient(
            config_path=access_config(tmp_path), cache_root=tmp_path
        )
        try:
            assert client.get(URL).content == BODY
            assert client.get(URL).content == BODY
            assert static.get.call_count == 1 and browser.get.call_count == 2
            assert build.call_args.kwargs["detect_waf"] is True
            assert "transport" in build.call_args.kwargs
        finally:
            client.close()
        browser.close.assert_called_once()


def test_challenge_retry_after_never_launches_browser(tmp_path):
    """A server cooldown takes priority over operator browser recovery."""

    with patch("src.tennis_abstract_access.build_http_client") as build:
        build.return_value.get.side_effect = WafBlockedError(
            "challenge", HttpResponse(429, b"challenge", {"Retry-After": "3600"}, URL)
        )
        client = TennisAbstractAcquisitionClient(
            config_path=access_config(tmp_path), cache_root=tmp_path
        )
        try:
            with pytest.raises(RateLimitedError):
                client.get(URL)
            assert build.call_count == 1
        finally:
            client.close()


def test_local_pause_migration_is_single_use_and_not_a_server_cooldown_bypass(monkeypatch):
    """Old local WAF pause can use newly authorized recovery once; server limits cannot."""

    monkeypatch.setattr("src.tennis_abstract_daily.browser_recovery_authorized", lambda: True)
    assert can_recover_local_waf_pause({"type": "WafBlockedError"})
    assert not can_recover_local_waf_pause(
        {"type": "WafBlockedError", "access_recovery_version": 1}
    )
    assert not can_recover_local_waf_pause({"type": "RateLimitedError"})
    assert not can_recover_local_waf_pause(
        {"type": "WafBlockedError", "server_retry_after_utc": "later"}
    )


def test_explicit_local_retry_keeps_server_rate_limits():
    """Even the explicit operator recovery flag is not a Retry-After override."""

    from src.tennis_abstract_daily import can_explicitly_retry_local_failure

    assert can_explicitly_retry_local_failure({"type": "ValueError"})
    assert can_explicitly_retry_local_failure({"type": "WafBlockedError"})
    assert not can_explicitly_retry_local_failure({"type": "RateLimitedError"})
    assert not can_explicitly_retry_local_failure(
        {
            "type": "WafBlockedError",
            "server_retry_after_utc": "later",
        }
    )


@pytest.mark.parametrize(
    "evidence", ["no_response", "valid_body", "waf_headers", "waf_body", "unknown_body"]
)
def test_only_verified_navigation_timeout_is_player_local(tmp_path, evidence):
    """A page timeout is isolated only without observed challenge or unknown response bytes."""

    navigation_timeout = type(
        "TimeoutError", (Exception,), {"__module__": "patchright._impl._errors"}
    )

    class TimeoutSession(FakeSession):
        """Emit controlled navigation evidence then the vendor's Page.goto timeout."""

        async def fetch(self, url, *, page_setup, page_action, extra_headers):
            """Do not contact a server or require a browser package in this fixture."""

            page = FakePage(url)
            await page_setup(page)
            if evidence != "no_response":
                response = FakeResponse(page)
                if evidence == "waf_headers":
                    response.headers["cf-mitigated"] = "challenge"

                async def body():
                    """Distinguish ordinary bytes, a challenge body and an unreadable document."""

                    if evidence == "unknown_body":
                        raise RuntimeError("Response bytes unavailable")
                    return b"<title>Just a moment...</title>" if evidence == "waf_body" else BODY

                response.body = body
                await page.record(response)
            raise navigation_timeout("Page.goto: Timeout 60000ms exceeded.")

    transport = TennisAbstractBrowserTransport(
        profile_root=tmp_path, session_factory=lambda **kw: TimeoutSession()
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            fetch(transport)
        wrapped = ResponsibleHttpError("Common HTTP transport wrapper")
        wrapped.__cause__ = caught.value
        timeout = isolated_player_timeout(wrapped)
        assert (timeout is not None) is (evidence in {"no_response", "valid_body"})
        if timeout:
            assert timeout.url == URL
    finally:
        transport.close()


def test_wrapper_text_and_policy_errors_are_never_classified_as_local_timeout():
    """Do not infer timeouts from generic error strings or erase a policy failure."""

    assert isolated_player_timeout(ResponsibleHttpError("Page.goto: Timeout")) is None
    protected = WafBlockedError("challenge", HttpResponse(403, b"blocked", {}, URL))
    protected.__cause__ = TennisAbstractPlayerTimeout(URL)
    assert isolated_player_timeout(protected) is None


def test_total_solver_deadline_is_not_a_player_local_timeout(tmp_path):
    """An overall solver/startup deadline must still stop the batch conservatively."""

    transport = TennisAbstractBrowserTransport(
        profile_root=tmp_path, session_factory=lambda **kw: FakeSession(stuck=True)
    )
    transport._timeout = 0.01
    try:
        with pytest.raises(RuntimeError) as caught:
            fetch(transport)
        wrapped = ResponsibleHttpError("Common HTTP transport wrapper")
        wrapped.__cause__ = caught.value
        assert isolated_player_timeout(wrapped) is None
    finally:
        transport.close()

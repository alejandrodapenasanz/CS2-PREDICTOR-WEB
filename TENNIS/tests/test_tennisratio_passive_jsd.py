"""Scrapling JSD regression: valid schedules pass; real WAF still stops."""

from datetime import UTC, datetime
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.responsible_http import HttpResponse, _looks_like_waf, build_http_client
from src.tennisratio.client import TennisRatioClient
from src.tennisratio.parser import parse_agenda_html
from src.tennisratio.types import TennisRatioHttpError, TennisRatioSchemaError
from tests.test_responsible_http import _Clock, _Response, _Transport, _settings


URL = "https://www.tennisratio.com/atp-matches.html"
ROBOTS = "https://www.tennisratio.com/robots.txt"
JSD = b"<script>var a=document.createElement('script');a.src='/cdn-cgi/challenge-platform/scripts/jsd/main.js';</script>"
FIXTURE = Path(__file__).parent / "fixtures/tennisratio/atp-matches.html"


def _client(tmp_path, content, *, status=200, extra_headers=None):
    """Build the actual TennisRatio Scrapling route over a recorded transport."""

    transport = _Transport(
        {
            ROBOTS: _Response(
                200, b"User-agent: *\nDisallow:\n", {"Content-Type": "text/plain"}, ROBOTS
            ),
            URL: _Response(
                status,
                content,
                {"Content-Type": "text/html; charset=utf-8", **(extra_headers or {})},
                URL,
            ),
        }
    )
    clock = _Clock()

    def factory(**kwargs):
        """Keep the real common-client factory, injecting only clock/config."""
        return build_http_client(
            **kwargs, settings=_settings(), clock=clock.time, sleeper=clock.sleep
        )

    with (
        patch("src.tennisratio.client.ScraplingHttpSession", return_value=transport) as scrapling,
        patch("src.tennisratio.client.build_http_client", side_effect=factory),
    ):
        client = TennisRatioClient(cache_root=tmp_path)
    scrapling.assert_called_once()
    return client, transport, clock


def test_real_scrapling_route_accepts_unchanged_schedule_with_passive_jsd_and_caches(
    tmp_path,
) -> None:
    """A complete schedule plus the observed injected script is not a block."""

    body = FIXTURE.read_bytes().replace(b"</body>", JSD + b"</body>")
    client, transport, clock = _client(tmp_path, body)
    try:
        first = client.get(URL, retrieved_at_utc=datetime(2026, 8, 21, 8, tzinfo=UTC))
        second = client.get(URL, retrieved_at_utc=datetime(2026, 8, 21, 9, tzinfo=UTC))
        assert first.content == second.content == body
        assert first.sha256 == hashlib.sha256(body).hexdigest()
        parsed = parse_agenda_html(
            first.content, gender="M", source_url=URL, retrieved_at_utc=first.retrieved_at_utc
        )
        assert len(parsed) == 2
        assert transport.calls == [ROBOTS, URL]
        assert clock.sleeps == [pytest.approx(1.0)]
    finally:
        client.close()


@pytest.mark.parametrize(
    ("status", "headers", "marker"),
    [
        (403, {}, b""),
        (429, {}, b""),
        (200, {"cF-mItIgAtEd": "challenge"}, b""),
        (200, {}, b"<title>Just a moment...</title>"),
        (200, {}, b"<div class='cf-chl-container'>Wait</div>"),
        (
            200,
            {},
            b"<script src='/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1'></script>",
        ),
        (200, {}, b"<script src='/cdn-cgi/challenge-platform/unknown'></script>"),
        (200, {}, b"<div class='cf-turnstile'></div>"),
        (200, {}, b"Enable JavaScript and cookies to continue"),
    ],
)
def test_passive_jsd_never_overrides_strong_waf_signals(tmp_path, status, headers, marker) -> None:
    """Even schedule-shaped markup cannot cancel a real WAF indicator."""

    body = FIXTURE.read_bytes() + JSD + marker
    client, transport, _ = _client(tmp_path, body, status=status, extra_headers=headers)
    try:
        for _attempt in range(2):
            with pytest.raises(TennisRatioHttpError):
                client.get(URL, retrieved_at_utc=datetime(2026, 8, 21, tzinfo=UTC))
        assert transport.calls == [ROBOTS, URL, URL]
    finally:
        client.close()


@pytest.mark.parametrize(
    "path",
    [
        b"/cdn-cgi/challenge-platform/scripts/jsd/main.js",
        b"/cdn-cgi/challenge-platform/h/g/scripts/jsd/main.js",
        b"/cdn-cgi/challenge-platform/scripts/jsd/api.js?onload=ready",
    ],
)
def test_only_known_jsd_urls_are_passive_in_successful_html(path) -> None:
    """Recognize narrow JSD URL forms without excluding neighboring paths."""

    body = b"<html><script src='" + path + b"'></script></html>"
    assert not _looks_like_waf(HttpResponse(200, body, {"Content-Type": "text/html"}, URL))
    assert _looks_like_waf(HttpResponse(503, body, {"Content-Type": "text/html"}, URL))
    assert _looks_like_waf(HttpResponse(200, body, {"Content-Type": "text/plain"}, URL))
    similar = body.replace(b".js", b".js-block")
    assert _looks_like_waf(HttpResponse(200, similar, {"Content-Type": "text/html"}, URL))


def test_passive_script_without_schedule_is_not_accepted_as_match_data() -> None:
    """JSD does not grant schema validity or fabricate a usable agenda."""

    with pytest.raises(TennisRatioSchemaError):
        parse_agenda_html(
            b"<html>" + JSD + b"</html>",
            gender="M",
            source_url=URL,
            retrieved_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
        )

"""Offline recovery of 429 through the production TennisRatio HTTP route."""

from datetime import UTC, datetime
from email.utils import format_datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.responsible_http import (
    HttpClientSettings,
    RateLimitedError,
    ResponsibleHttpError,
    _retry_after_seconds,
    build_http_client,
)
from src.tennisratio.client import TennisRatioClient
from src.tennisratio.types import TennisRatioBlockedError
from tests.test_responsible_http import _Clock, _Response


ORIGIN = "https://www.tennisratio.com"
ROBOTS = ORIGIN + "/robots.txt"
URL = ORIGIN + "/atp-matches.html"
FIXTURE = Path(__file__).parent / "fixtures/tennisratio/atp-matches.html"


class _SequenceTransport:
    """Serve a finite sequence per URL; unexpected requests fail the test."""

    def __init__(self, routes):
        """Record routes, request sequence and session close."""
        self.routes = {url: list(responses) for url, responses in routes.items()}
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        """Return the next recorded response without performing network I/O."""
        self.calls.append(url)
        return self.routes[url].pop(0)

    def close(self):
        """Record that the same transport was closed."""
        self.closed = True


def _response(url=URL, status=200, *, headers=None, content=None):
    """Provide legitimate content, a plain rate-limit, or a supplied challenge."""
    if content is None:
        content = (
            b"User-agent: *\nDisallow: /private/\n"
            if url == ROBOTS
            else FIXTURE.read_bytes()
            if status == 200
            else b"Too many requests"
        )
    return _Response(
        status,
        content,
        {
            "Content-Type": "text/plain" if url == ROBOTS else "text/html",
            **(headers or {}),
        },
        url,
    )


def _client(tmp_path, transport, clock):
    """Use real source/config/common-client logic, replacing only wire and time."""

    def sleep(seconds):
        """Backoff releases the interprocess lock and sleeps in bounded chunks."""
        assert seconds <= 30
        if seconds >= 30:
            assert not list(tmp_path.rglob("request.lock"))
        clock.sleep(seconds)

    def factory(**kwargs):
        """Inject a deterministic epoch clock while keeping the production config."""
        return build_http_client(**kwargs, clock=clock.time, sleeper=sleep)

    with (
        patch("src.tennisratio.client.ScraplingHttpSession", return_value=transport),
        patch("src.tennisratio.client.build_http_client", side_effect=factory),
    ):
        return TennisRatioClient(
            cache_root=tmp_path,
            clock=lambda: datetime.fromtimestamp(clock.time(), UTC),
        )


@pytest.mark.parametrize("header", ["45", "http-date"])
def test_retry_after_recovers_same_url_then_caches(tmp_path, header, caplog):
    """Both Retry-After encodings are honored; the 429 body is never cached."""
    clock = _Clock()
    if header == "http-date":
        header = format_datetime(datetime.fromtimestamp(clock.now + 46, UTC), usegmt=True)
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=429, headers={"rEtRy-AfTeR": header}), _response()],
        }
    )
    client = _client(tmp_path, transport, clock)
    observed = datetime.fromtimestamp(clock.now, UTC)
    first = client.get(URL, retrieved_at_utc=observed)
    second = client.get(URL, retrieved_at_utc=observed)
    client.close()
    assert transport.calls == [ROBOTS, URL, URL]
    assert transport.closed
    assert sum(clock.sleeps) == pytest.approx(46)
    assert first.content == second.content == FIXTURE.read_bytes()
    assert first.retrieved_at_utc.timestamp() == clock.now
    assert "reintento 2/3" in caplog.text
    state = json.loads(next(tmp_path.rglob("rate_limit.json")).read_text())
    assert state["streak"] == 0


def test_three_429_stop_and_cooldown_survives_new_client(tmp_path):
    """Three responses consume 30+60s, then all host URLs are deferred."""
    clock = _Clock()
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=429) for _ in range(3)],
        }
    )
    observed = datetime.fromtimestamp(clock.now, UTC)
    client = _client(tmp_path, transport, clock)
    with pytest.raises(TennisRatioBlockedError, match="429"):
        client.get(URL, retrieved_at_utc=observed)
    assert sum(clock.sleeps) == pytest.approx(91)
    client.close()
    restarted_transport = _SequenceTransport({})
    restarted = _client(tmp_path, restarted_transport, clock)
    for url in (ORIGIN + "/players/Other.html", URL):
        with pytest.raises(TennisRatioBlockedError, match="429"):
            restarted.get(url, retrieved_at_utc=observed, cache_mode="refresh")
    assert restarted_transport.calls == []
    assert transport.calls == [ROBOTS, URL, URL, URL]
    assert not list(tmp_path.rglob("request.lock"))
    assert len(list(tmp_path.rglob("rate_limit.json"))) == 1


def test_long_retry_after_is_not_capped_or_slept(tmp_path):
    """A server-requested hour is persisted; it never becomes an early retry."""
    clock = _Clock()
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=429, headers={"Retry-After": "3600"})],
        }
    )
    client = _client(tmp_path, transport, clock)
    client.get(ROBOTS, retrieved_at_utc=datetime.fromtimestamp(clock.now, UTC))
    with pytest.raises(TennisRatioBlockedError) as error:
        client.get(URL, retrieved_at_utc=datetime.fromtimestamp(clock.now, UTC))
    cause = error.value.__cause__
    assert isinstance(cause, RateLimitedError)
    assert cause.retry_at - clock.now == 3600
    assert transport.calls == [ROBOTS, URL]
    assert clock.sleeps == [1]
    # The valid cached robots body can still be read, even during cooldown.
    assert client.get(ROBOTS, retrieved_at_utc=datetime.fromtimestamp(clock.now, UTC)).content
    assert transport.calls == [ROBOTS, URL]


def test_robots_429_recovers_before_requesting_page(tmp_path):
    """Implicit robots fetch shares rate-limit policy and never bypasses it."""
    clock = _Clock()
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS, 429, content=b"Too many requests"), _response(ROBOTS)],
            URL: [_response()],
        }
    )
    client = _client(tmp_path, transport, clock)
    client.get(URL, retrieved_at_utc=datetime.fromtimestamp(clock.now, UTC))
    assert transport.calls == [ROBOTS, ROBOTS, URL]
    assert clock.sleeps == [30, 1]


@pytest.mark.parametrize(
    "status,headers,body",
    [
        (429, {"CF-Mitigated": "challenge"}, b"Wait"),
        (429, {}, b"<title>Just a moment...</title>"),
        (403, {}, b"Denied"),
        (200, {}, b"Enable JavaScript and cookies to continue"),
    ],
)
def test_real_waf_never_enters_429_retry(tmp_path, status, headers, body):
    """An actual challenge/403 still stops after one response, including 429."""
    clock = _Clock()
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=status, headers=headers, content=body)],
        }
    )
    client = _client(tmp_path, transport, clock)
    with pytest.raises(TennisRatioBlockedError):
        client.get(URL, retrieved_at_utc=datetime.fromtimestamp(clock.now, UTC))
    assert transport.calls == [ROBOTS, URL]
    assert clock.sleeps == [1]


def test_recovery_crossing_midnight_never_backdates_capture(tmp_path):
    """Data received on D remains unavailable before D despite starting on D-1."""
    clock = _Clock()
    clock.now = datetime(2026, 9, 5, 23, 59, 50, tzinfo=UTC).timestamp()
    observed = datetime.fromtimestamp(clock.now, UTC)
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=429), _response()],
        }
    )
    payload = _client(tmp_path, transport, clock).get(URL, retrieved_at_utc=observed)
    assert payload.retrieved_at_utc.date() > observed.date()
    assert payload.retrieved_at_utc.timestamp() == clock.now


@pytest.mark.parametrize("value", [None, "", "nan", "inf", "bad date"])
def test_invalid_retry_after_falls_back_to_finite_backoff(value):
    """Malformed server hints cannot break deterministic retry decisions."""
    assert _retry_after_seconds(value, now=10000) is None


def test_429_config_is_scoped_to_authorized_sources():
    """Ratio and Abstract share bounded recovery; other hosts remain unchanged."""
    assert HttpClientSettings.load(host="www.tennisratio.com").rate_limit_attempts == 3
    assert HttpClientSettings.load(host="www.tennisabstract.com").rate_limit_attempts == 3
    for host in ("www.tennisexplorer.com", "example.test"):
        assert HttpClientSettings.load(host=host).rate_limit_attempts == 1
    with pytest.raises(ResponsibleHttpError):
        HttpClientSettings(rate_limit_attempts=True).validate()


def test_cooldown_isolated_by_host_and_recovery_after_expiry(tmp_path):
    """A paused Ratio host does not block another source and recovers after expiry."""
    clock = _Clock()
    transport = _SequenceTransport(
        {
            ROBOTS: [_response(ROBOTS)],
            URL: [_response(status=429, headers={"Retry-After": "3600"}), _response()],
        }
    )
    client = _client(tmp_path, transport, clock)
    observed = datetime.fromtimestamp(clock.now, UTC)
    with pytest.raises(TennisRatioBlockedError):
        client.get(URL, retrieved_at_utc=observed)
    other_url = "https://example.test/robots.txt"
    other_transport = _SequenceTransport({other_url: [_response(other_url)]})
    other = build_http_client(
        transport_kind="scrapling",
        transport=other_transport,
        cache_root=tmp_path,
        clock=clock.time,
        sleeper=clock.sleep,
    )
    assert other.get(other_url).status_code == 200
    assert other_transport.calls == [other_url]
    clock.now += 3600
    assert client.get(URL, retrieved_at_utc=observed).content == FIXTURE.read_bytes()
    state = json.loads(next(tmp_path.rglob("rate_limit.json")).read_text())
    assert state["streak"] == 0

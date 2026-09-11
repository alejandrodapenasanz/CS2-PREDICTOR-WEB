from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_daily_start_module():
    spec = importlib.util.spec_from_file_location("daily_start_cf_refresh", ROOT / "PIPELINE" / "start.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_operator_pipeline_config_uses_visible_stealth() -> None:
    config = (ROOT / "PIPELINE" / "pipeline.config.psd1").read_text(encoding="utf-8")
    assert "HLTV_STEALTH_HEADLESS            = '0'" in config


def test_stats_challenge_hands_off_to_visible_stealth_without_backoff(monkeypatch) -> None:
    daily = load_daily_start_module()
    attempts: list[str] = []
    stealth_calls: list[str] = []
    sleeps: list[float] = []

    daily.SCRAPLING_ENABLED = True
    daily.SCRAPLING_STEALTH_ENABLED = True
    daily.SCRAPLING_STEALTH_HEADLESS = False
    daily.SCRAPLING_TIER1_ATTEMPTS = 3
    daily.SCRAPLING_STEALTH_MAX_SOLVES = 6
    daily._STEALTH_SOLVES = 0
    daily._ScraplingFetcher = object()
    daily._ScraplingStealthy = object()

    monkeypatch.setattr(daily, "_apply_ca_env", lambda: None)
    monkeypatch.setattr(daily, "ensure_fetch_budget", lambda url: None)
    monkeypatch.setattr(daily, "wait_for_domain_cooldown", lambda: None)
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda interval: None)
    monkeypatch.setattr(daily, "register_fetch_soft_block", lambda *args: None)
    monkeypatch.setattr(daily, "register_fetch_success", lambda *args: None)
    monkeypatch.setattr(daily, "fetch_cache_put", lambda *args: None)
    monkeypatch.setattr(daily, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: sleeps.append(seconds))

    def blocked_get(url, cookies, timeout):
        attempts.append(url)
        return 403, "<html><title>Just a moment</title></html>"

    def solved_get(url):
        stealth_calls.append(url)
        return 200, "<html>real stats</html>"

    monkeypatch.setattr(daily, "_scrapling_impersonate_get", blocked_get)
    monkeypatch.setattr(daily, "_scrapling_stealth_solve", solved_get)

    url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    html = daily._scrapling_fetch(
        url,
        {"cf_clearance": "stale"},
        timeout=5,
        base=5.0,
        max_wait=1200.0,
        interval=0.0,
    )

    assert html == "<html>real stats</html>"
    assert attempts == [url]
    assert stealth_calls == [url]
    assert sleeps == []


def test_stats_requests_block_refreshes_once_then_suppresses_without_sleep(monkeypatch, tmp_path) -> None:
    daily = load_daily_start_module()
    request_calls: list[str] = []
    refresh_calls: list[tuple[str, str]] = []
    quarantined: list[tuple[str, str]] = []
    sleeps: list[float] = []

    class BlockedResponse:
        status_code = 403
        text = "<html><title>Just a moment</title></html>"
        headers: dict[str, str] = {}

    class BlockedSession:
        def get(self, url, **kwargs):
            request_calls.append(url)
            return BlockedResponse()

    daily.CF_SESSION = tmp_path / "missing_cf_session.json"
    daily.CF_REFRESH_ENABLED = True
    daily._CF_SESSION_REFRESHED_THIS_RUN = False
    monkeypatch.setattr(daily, "_scrapling_fetch", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily, "http_session", lambda: BlockedSession())
    monkeypatch.setattr(daily, "ensure_fetch_budget", lambda url: None)
    monkeypatch.setattr(daily, "wait_for_domain_cooldown", lambda: None)
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda interval: None)
    monkeypatch.setattr(daily, "register_fetch_soft_block", lambda *args: None)
    monkeypatch.setattr(daily, "check_url_quarantine", lambda url: None)
    monkeypatch.setattr(daily, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        daily,
        "maybe_refresh_cf_session_once",
        lambda reason, url: refresh_calls.append((reason, url)) or False,
    )
    monkeypatch.setattr(
        daily,
        "quarantine_url",
        lambda url, reason: quarantined.append((url, reason)),
    )

    url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    with pytest.raises(daily.FetchSuppressedError, match="se omite esta URL"):
        daily.fetch_html(url, max_attempts=10, min_interval=0.0, use_cache=False)

    assert request_calls == [url]
    assert refresh_calls == [("requests_cloudflare_challenge", url)]
    assert quarantined and quarantined[0][0] == url
    assert sleeps == []


def test_unchanged_cf_session_is_not_reported_as_refreshed(monkeypatch, tmp_path) -> None:
    daily = load_daily_start_module()
    session = tmp_path / "cf_session.json"
    session.write_text('{"cf_clearance":"stale","user_agent":"ua"}', encoding="utf-8")
    helper = tmp_path / "hltv_scraper" / "grab_cf.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# fixture", encoding="utf-8")

    daily.CF_SESSION = session
    daily.SCRAPY_ROOT = tmp_path
    daily.CF_REFRESH_ENABLED = True
    daily._CF_SESSION_REFRESHED_THIS_RUN = False
    monkeypatch.setattr(daily, "run_cmd", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(daily, "log", lambda *args, **kwargs: None)

    assert daily.maybe_refresh_cf_session_once("test", "https://www.hltv.org/stats") is False


@pytest.mark.parametrize("failure", ["launch", "invalid_json"])
def test_refresh_failure_does_not_resume_http_retry_loop(monkeypatch, tmp_path, failure) -> None:
    daily = load_daily_start_module()
    helper = tmp_path / "hltv_scraper" / "grab_cf.py"
    helper.parent.mkdir()
    helper.touch()
    daily.CF_SESSION = tmp_path / "cf_session.json"
    daily.SCRAPY_ROOT = tmp_path
    daily.CF_REFRESH_ENABLED = True

    def failed_helper(*args, **kwargs):
        if failure == "launch":
            raise OSError("Cannot start browser")
        daily.CF_SESSION.write_text("incomplete JSON", encoding="utf-8")
        return True, ""

    monkeypatch.setattr(daily, "run_cmd", failed_helper)
    monkeypatch.setattr(daily, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily.time, "sleep", lambda _: pytest.fail("Must not sleep"))
    url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    with pytest.raises(daily.FetchSuppressedError):
        daily._refresh_blocked_stats_or_suppress(url, 5, 0.0, source="requests", reason="cloudflare_challenge")
    with pytest.raises(daily.FetchSuppressedError):
        daily.check_url_quarantine(url)
    assert daily._FETCH_STATS["cf_session_refresh_attempts"] == 1


def test_refresh_opens_exact_url_and_fetches_valid_html_once(monkeypatch, tmp_path) -> None:
    daily = load_daily_start_module()
    helper = tmp_path / "hltv_scraper" / "grab_cf.py"
    helper.parent.mkdir()
    helper.touch()
    daily.CF_SESSION = tmp_path / "cf_session.json"
    daily.SCRAPY_ROOT = tmp_path
    daily.CF_REFRESH_ENABLED = True
    commands: list[list[str]] = []
    requests: list[str] = []

    def renew(command, *args, **kwargs):
        commands.append(command)
        daily.CF_SESSION.write_text('{"cf_clearance":"new","user_agent":"ua"}', encoding="utf-8")
        return True, ""

    def fetch(url, cookies, timeout):
        assert cookies == {"cf_clearance": "new"}
        requests.append(url)
        return 200, "<html>real stats</html>"

    monkeypatch.setattr(daily, "run_cmd", renew)
    monkeypatch.setattr(daily, "_scrapling_impersonate_get", fetch)
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda _: None)
    monkeypatch.setattr(daily, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily.time, "sleep", lambda _: pytest.fail("Must not sleep"))
    url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    result = daily._refresh_blocked_stats_or_suppress(url, 5, 0.0, source="requests", reason="cloudflare_challenge")
    assert result == "<html>real stats</html>"
    assert commands[0][-2:] == ["--url", url]
    assert requests == [url]
    assert daily.fetch_cache_get(url) == result
    assert daily.maybe_refresh_cf_session_once("second_attempt", url) is False
    assert len(commands) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_server_failure_keep_backoff(monkeypatch, tmp_path, status) -> None:
    daily = load_daily_start_module()
    daily.CF_SESSION = tmp_path / "missing.json"
    sleeps: list[float] = []
    responses = iter(
        [
            SimpleNamespace(status_code=status, text="service unavailable", headers={"Retry-After": "60"}),
            SimpleNamespace(status_code=200, text="<html>real stats</html>", headers={}),
        ]
    )
    monkeypatch.setattr(daily, "_scrapling_fetch", lambda *args: None)
    monkeypatch.setattr(daily, "http_session", lambda: SimpleNamespace(get=lambda *a, **kw: next(responses)))
    monkeypatch.setattr(daily, "wait_for_domain_cooldown", lambda: None)
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda _: None)
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(daily, "maybe_refresh_cf_session_once", lambda *a: pytest.fail("Not a browser challenge"))
    result = daily.fetch_html("/stats/matches/mapstatsid/1/a-vs-b", max_attempts=2, use_cache=False)
    assert result == "<html>real stats</html>"
    assert len(sleeps) == 1 and sleeps[0] >= 60


def test_blocked_mapstats_preserves_partial_asset_and_returns(monkeypatch, tmp_path) -> None:
    daily = load_daily_start_module()
    map_url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    match_html = f'<html><a href="{map_url}">Stats</a></html>'
    calls: list[str] = []

    def get(url, **kwargs):
        calls.append(url)
        if "/matches/123/" in url:
            return SimpleNamespace(status_code=200, text=match_html, headers={})
        return SimpleNamespace(status_code=403, text="Just a moment", headers={})

    daily.CF_SESSION = tmp_path / "missing.json"
    daily.CF_REFRESH_ENABLED = False
    monkeypatch.setattr(daily, "_scrapling_fetch", lambda *a: None)
    monkeypatch.setattr(daily, "http_session", lambda: SimpleNamespace(get=get))
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda _: None)
    monkeypatch.setattr(daily, "save_raw_html", lambda *a: {"source_file": "fixture.html"})
    monkeypatch.setattr(daily, "load_persisted_raw_html", lambda *a: None)
    monkeypatch.setattr(daily, "db_mark_fetch_state", lambda *a: None)
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: None if seconds == 0 else pytest.fail("Long sleep"))
    output = tmp_path / "run" / "assets" / "123" / "assets.json"
    result = daily.scrape_match_assets("/matches/123/a-vs-b", output, delay=0)
    assert len(result["errors"]) == 1 and result["mapstats"] == []
    assert json.loads(output.read_text(encoding="utf-8"))["errors"] == result["errors"]
    assert len(calls) == 2
    with pytest.raises(daily.FetchSuppressedError):
        daily.fetch_html(map_url, use_cache=False)
    assert len(calls) == 2

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
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


@pytest.mark.parametrize("fallback_ok", [False, True])
def test_failed_upcoming_never_promotes_empty_run_even_when_results_were_downloaded(
    tmp_path, monkeypatch, fallback_ok
) -> None:
    """A timed-out/empty fallback must fail before replacing the master publication pointer."""

    daily = load_daily_start_module()
    master = tmp_path / "master.json"
    pointer = tmp_path / "manifest.json"
    master.write_text("{}", encoding="utf-8")
    pointer.write_text('{"last_run_id":"previous"}', encoding="utf-8")
    monkeypatch.setattr(daily, "MASTER_MATCHES", master)
    monkeypatch.setattr(daily, "MASTER_MANIFEST", pointer)
    monkeypatch.setattr(daily, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(daily, "run_id", lambda: "failed-run")
    monkeypatch.setattr(daily, "db_pending_match_ids", lambda: set())
    monkeypatch.setattr(daily, "db_pending_match_info", lambda: {})
    monkeypatch.setattr(daily, "scrape_recent_results", lambda *args, **kwargs: [{"id": "result"}])
    monkeypatch.setattr(daily, "update_pending_matches", lambda *args, **kwargs: pytest.fail("Agenda must run first"))
    monkeypatch.setattr(daily, "run_spider", lambda *args, **kwargs: (fallback_ok, "timeout or empty response"))

    def blocked(*args, **kwargs):
        """Simulate the production WAF failure without using the network."""

        raise RuntimeError("fixture WAF")

    monkeypatch.setattr(daily, "fetch_html", blocked)
    monkeypatch.setattr(sys, "argv", ["start.py", "--skip-warmup", "--skip-same-day-recovery", "--allow-empty-scrape"])
    with pytest.raises(RuntimeError, match="HLTV upcoming"):
        daily.main()
    assert json.loads(pointer.read_text())["last_run_id"] == "previous"
    failed_run = tmp_path / "runs" / "failed-run"
    manifest = json.loads((failed_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"]["recent_results"]["count"] == 1
    assert manifest["steps"]["upcoming"]["status"] == "failed"
    assert manifest["failed_at"]
    assert not (failed_run / "predictions_enriched.json").exists()


def test_operator_pipeline_config_uses_direct_visible_refresh(monkeypatch) -> None:
    config = (ROOT / "PIPELINE" / "pipeline.config.psd1").read_text(encoding="utf-8")
    assert "HLTV_SOLVE_CLOUDFLARE            = '0'" in config
    assert "HLTV_AUTO_REFRESH_CF_ON_BLOCK    = '1'" in config
    assert "HLTV_STEALTH_HEADLESS            = '0'" in config
    monkeypatch.delenv("HLTV_SOLVE_CLOUDFLARE", raising=False)
    monkeypatch.delenv("HLTV_AUTO_REFRESH_CF_ON_BLOCK", raising=False)
    daily = load_daily_start_module()
    assert not daily.SCRAPLING_STEALTH_ENABLED
    assert daily.CF_REFRESH_ENABLED


def test_explicit_legacy_stealth_remains_opt_in(monkeypatch) -> None:
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


@pytest.mark.parametrize("path", ["/stats/matches/mapstatsid/1/a-vs-b", "/matches/1/a-vs-b"])
def test_requests_block_refreshes_once_then_suppresses_without_sleep(monkeypatch, tmp_path, path) -> None:
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
    daily.SCRAPLING_STEALTH_ENABLED = False
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

    url = "https://www.hltv.org" + path
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
        daily._refresh_blocked_url_or_suppress(url, 5, 0.0, source="requests", reason="cloudflare_challenge")
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
    result = daily._refresh_blocked_url_or_suppress(url, 5, 0.0, source="requests", reason="cloudflare_challenge")
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


@pytest.fixture
def interactive_daily(monkeypatch, tmp_path):
    """Use the real routing/session checks without opening a browser or network."""
    monkeypatch.delenv("HLTV_SOLVE_CLOUDFLARE", raising=False)
    monkeypatch.setenv("HLTV_AUTO_REFRESH_CF_ON_BLOCK", "1")
    daily = load_daily_start_module()
    daily.CF_SESSION = tmp_path / "cf_session.json"
    daily.SCRAPY_ROOT = tmp_path
    helper = tmp_path / "hltv_scraper" / "grab_cf.py"
    helper.parent.mkdir()
    helper.touch()
    daily.SCRAPLING_ENABLED = True
    daily._ScraplingFetcher = object()
    daily._ScraplingStealthy = object()
    commands: list[list[str]] = []

    def renew(command, *args, **kwargs):
        commands.append(command)
        daily.CF_SESSION.write_text('{"cf_clearance":"fixture-new","user_agent":"fixture-ua"}', encoding="utf-8")
        return True, ""

    monkeypatch.setattr(daily, "run_cmd", renew)
    monkeypatch.setattr(daily, "_apply_ca_env", lambda: None)
    monkeypatch.setattr(daily, "wait_for_domain_cooldown", lambda: None)
    monkeypatch.setattr(daily, "polite_fetch_wait", lambda _: None)
    monkeypatch.setattr(daily, "log", lambda *a, **kw: None)
    monkeypatch.setattr(daily.time, "sleep", lambda _: pytest.fail("No long retry after a browser challenge"))
    monkeypatch.setattr(daily, "_scrapling_stealth_solve", lambda _: pytest.fail("Stealth must not open by default"))
    monkeypatch.setattr(daily, "http_session", lambda: pytest.fail("Must not repeat a blocked URL via HTTP fallback"))
    return daily, commands


@pytest.mark.parametrize("path", ["/stats/matches/mapstatsid/1/a-vs-b", "/matches/1/a-vs-b", "/team/1/example"])
@pytest.mark.parametrize("blocked", [(403, "Just a moment"), (200, "Checking your browser"), (403, "Forbidden")])
def test_scrapling_challenge_opens_grab_cf_directly(interactive_daily, monkeypatch, path, blocked) -> None:
    daily, commands = interactive_daily
    requests: list[str] = []
    url = "https://www.hltv.org" + path

    def fetch(target, cookies, timeout):
        requests.append(target)
        if len(requests) == 1:
            return blocked
        assert cookies == {"cf_clearance": "fixture-new"}
        return 200, "<html>real HLTV data</html>"

    monkeypatch.setattr(daily, "_scrapling_impersonate_get", fetch)
    result = daily.fetch_html(url, max_attempts=10, min_interval=0.0)

    assert result == "<html>real HLTV data</html>"
    assert len(commands) == 1 and commands[0][-2:] == ["--url", url]
    assert Path(commands[0][1]).name == "grab_cf.py"
    assert requests == [url, url]
    assert daily._STEALTH_SOLVES == 0
    assert daily._FETCH_STATS["cf_session_refresh_successes"] == 1
    assert daily.fetch_html(url) == result  # Cache does not reopen the browser.
    assert len(commands) == 1 and requests == [url, url]


@pytest.mark.parametrize("failure", ["helper_failed", "still_challenged", "already_attempted", "disabled"])
def test_direct_refresh_failure_stops_without_stealth_or_retry_loop(interactive_daily, monkeypatch, failure) -> None:
    daily, commands = interactive_daily
    if failure == "helper_failed":
        monkeypatch.setattr(daily, "run_cmd", lambda *a, **kw: (False, "Browser closed"))
    elif failure == "already_attempted":
        daily._CF_SESSION_REFRESHED_THIS_RUN = True
    elif failure == "disabled":
        daily.CF_REFRESH_ENABLED = False
    requests: list[str] = []

    def blocked(target, cookies, timeout):
        requests.append(target)
        return 403, "Just a moment"

    monkeypatch.setattr(daily, "_scrapling_impersonate_get", blocked)
    url = "https://www.hltv.org/matches/1/a-vs-b"
    with pytest.raises(daily.FetchSuppressedError, match="se omite esta URL"):
        daily.fetch_html(url, max_attempts=10, min_interval=0.0, use_cache=False)
    expected_requests = 2 if failure == "still_challenged" else 1
    assert len(requests) == expected_requests
    assert daily._STEALTH_SOLVES == 0
    assert daily._FETCH_STATS["cf_session_refresh_attempts"] <= 1
    assert len(commands) <= 1
    with pytest.raises(daily.FetchSuppressedError):
        daily.fetch_html(url, use_cache=False)
    assert len(requests) == expected_requests


@pytest.mark.parametrize("status", [429, 503])
def test_scrapling_rate_limit_and_server_failure_do_not_open_browser(interactive_daily, monkeypatch, status) -> None:
    daily, commands = interactive_daily
    responses = iter([(status, "service unavailable"), (200, "<html>real data</html>")])
    sleeps: list[float] = []
    monkeypatch.setattr(daily, "_scrapling_impersonate_get", lambda *args: next(responses))
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: sleeps.append(seconds))
    result = daily.fetch_html("/stats/matches/mapstatsid/1/a-vs-b", min_interval=0.0, use_cache=False)
    assert result == "<html>real data</html>"
    assert commands == [] and daily._STEALTH_SOLVES == 0
    assert len(sleeps) == 1 and sleeps[0] > 0


def test_regular_html_does_not_open_a_browser(interactive_daily, monkeypatch) -> None:
    daily, commands = interactive_daily
    monkeypatch.setattr(daily, "_scrapling_impersonate_get", lambda *args: (200, "<html>real data</html>"))
    assert daily.fetch_html("/matches/1/a-vs-b", min_interval=0.0, use_cache=False) == "<html>real data</html>"
    assert commands == [] and daily._STEALTH_SOLVES == 0


def test_browser_capture_is_used_without_retrying_blocked_http(interactive_daily, monkeypatch):
    """Regression: the visible browser worked but HTTP still returned 403."""
    daily, _ = interactive_daily
    url = "https://www.hltv.org/matches"
    html = '<html><body><div class="matches-list-section">real agenda</div></body></html>'
    calls = []

    def renew(command, *args, **kwargs):
        daily.CF_SESSION.write_text('{"cf_clearance":"new","user_agent":"ua"}', encoding="utf-8")
        output = Path(command[command.index("--capture-output") + 1])
        output.write_text(json.dumps({"url": url, "html": html, "sha256": hashlib.sha256(html.encode()).hexdigest()}))
        return True, ""

    def blocked(*args):
        calls.append(args[0])
        return 403, "Just a moment"

    monkeypatch.setattr(daily, "run_cmd", renew)
    monkeypatch.setattr(daily, "_scrapling_impersonate_get", blocked)
    assert daily.fetch_html(url, min_interval=0, use_cache=False) == html
    assert calls == [url]
    assert daily.fetch_diagnostics()["interactive_browser_successes"] == 1
    assert not list(daily.CF_SESSION.parent.glob("cf_capture_*"))


@pytest.mark.parametrize("reason", ["wrong_url", "wrong_hash", "challenge", "partial_load"])
def test_invalid_browser_capture_is_never_accepted(interactive_daily, monkeypatch, reason):
    daily, _ = interactive_daily
    url = "https://www.hltv.org/matches"
    daily.fetch_cache_put(url, "<html><body>old cached agenda</body></html>")
    html = "<html><body>Just a moment</body></html>" if reason == "challenge" else "<html><body>real page</body></html>"
    if reason == "partial_load":
        html = "<html><head>page loading</head></html>"

    def renew(command, *args, **kwargs):
        daily.CF_SESSION.write_text('{"cf_clearance":"new","user_agent":"ua"}', encoding="utf-8")
        output = Path(command[command.index("--capture-output") + 1])
        output.write_text(
            json.dumps(
                {
                    "url": url + "/wrong" if reason == "wrong_url" else url,
                    "html": html,
                    "sha256": "invalid" if reason == "wrong_hash" else hashlib.sha256(html.encode()).hexdigest(),
                }
            )
        )
        return True, ""

    monkeypatch.setattr(daily, "run_cmd", renew)
    monkeypatch.setattr(daily, "_scrapling_impersonate_get", lambda *args: (403, "Just a moment"))
    with pytest.raises(daily.FetchSuppressedError):
        daily.fetch_html(url, min_interval=0, use_cache=False)
    assert daily.fetch_cache_get(url) is None


@pytest.mark.parametrize("error_name", ["FetchSuppressedError", "FetchBudgetExceeded"])
def test_upcoming_deferred_does_not_start_another_scrapy_retry_loop(tmp_path, monkeypatch, error_name):
    daily = load_daily_start_module()

    def deferred(*args, **kwargs):
        raise getattr(daily, error_name)("fixture access deferred")

    monkeypatch.setattr(daily, "fetch_html", deferred)
    monkeypatch.setattr(daily, "run_spider", lambda *args, **kwargs: pytest.fail("No 600-second retry loop"))
    with pytest.raises(RuntimeError, match="preserving the previous published run"):
        daily.scrape_upcoming(tmp_path)
    assert not (tmp_path / "raw" / "upcoming_matches.json").exists()


def test_scrapling_reuses_the_browser_user_agent(tmp_path, monkeypatch):
    daily = load_daily_start_module()
    daily.CF_SESSION = tmp_path / "session.json"
    daily.CF_SESSION.write_text('{"cf_clearance":"secret","user_agent":"actual-browser-ua"}')
    captured = {}

    def get(url, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status=200, html_content="<html>real</html>")

    monkeypatch.setattr(daily, "_ScraplingFetcher", SimpleNamespace(get=get))
    assert daily._scrapling_impersonate_get("https://www.hltv.org/matches", {"cf_clearance": "secret"}, 5)[0] == 200
    assert captured["headers"]["User-Agent"] == "actual-browser-ua"
    assert captured["cookies"] == {"cf_clearance": "secret"}

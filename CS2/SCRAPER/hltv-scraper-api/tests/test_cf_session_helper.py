"""Interactive session helper behavior without network or a real browser."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

HELPER_PATH = Path(__file__).resolve().parents[1] / "hltv_scraper" / "hltv_scraper" / "grab_cf.py"


@pytest.fixture
def helper(tmp_path):
    spec = importlib.util.spec_from_file_location("cf_helper_test", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CF_SESSION_FILE = tmp_path / "cf_session.json"
    return module


@pytest.mark.parametrize("outcome", ["success", "timeout", "navigation_failure"])
def test_browser_closes_and_only_success_replaces_session(helper, monkeypatch, outcome):
    helper.CF_SESSION_FILE.write_text('{"cf_clearance":"previous"}', encoding="utf-8")
    original = helper.CF_SESSION_FILE.read_bytes()
    cookie = SimpleNamespace(name="cf_clearance", value="new")
    url = "https://www.hltv.org/stats/matches/mapstatsid/1/a-vs-b"
    tab = SimpleNamespace(
        evaluate=AsyncMock(
            side_effect=lambda expression: {
                "window.location.href": url,
                "document.readyState": "complete",
                "navigator.userAgent": "test-ua",
            }[expression]
        ),
        get_content=AsyncMock(return_value="<html><title>HLTV</title><body>Real page</body></html>"),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=tab),
        cookies=SimpleNamespace(get_all=AsyncMock(return_value=[cookie] if outcome == "success" else [])),
        stop=Mock(),
    )
    start = AsyncMock(return_value=browser)
    monkeypatch.setattr(helper.uc, "start", start)
    monkeypatch.setattr(helper.asyncio, "sleep", AsyncMock())
    if outcome == "navigation_failure":
        browser.get.side_effect = RuntimeError("navigation failed")
        with pytest.raises(RuntimeError, match="navigation failed"):
            asyncio.run(helper.grab_cf_session(url))
    else:
        assert asyncio.run(helper.grab_cf_session(url)) is (outcome == "success")
    start.assert_awaited_once_with(headless=False)
    browser.get.assert_awaited_once_with(url)
    browser.stop.assert_called_once()
    if outcome == "success":
        assert json.loads(helper.CF_SESSION_FILE.read_text(encoding="utf-8")) == {
            "cf_clearance": "new",
            "user_agent": "test-ua",
        }
    else:
        assert helper.CF_SESSION_FILE.read_bytes() == original


def test_unrelated_url_is_rejected_before_browser_launch(helper, monkeypatch):
    start = AsyncMock()
    monkeypatch.setattr(helper.uc, "start", start)
    with pytest.raises(argparse.ArgumentTypeError):
        asyncio.run(helper.grab_cf_session("https://example.org/stats"))
    start.assert_not_awaited()


@pytest.mark.parametrize("invalid", ["challenge", "redirect", "empty", "loading"])
def test_cookie_without_loaded_exact_page_is_not_success(helper, monkeypatch, tmp_path, invalid):
    """A cookie already present must not close the window before the page loads."""
    url = "https://www.hltv.org/matches"
    html = "<html><body>Just a moment</body></html>" if invalid == "challenge" else "<html><body>HLTV</body></html>"
    if invalid == "empty":
        html = ""
    tab = SimpleNamespace(
        evaluate=AsyncMock(
            side_effect=lambda expression: {
                "window.location.href": "https://www.hltv.org/" if invalid == "redirect" else url,
                "document.readyState": "loading" if invalid == "loading" else "complete",
            }[expression]
        ),
        get_content=AsyncMock(return_value=html),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=tab),
        cookies=SimpleNamespace(get_all=AsyncMock(return_value=[SimpleNamespace(name="cf_clearance", value="new")])),
        stop=Mock(),
    )
    monkeypatch.setattr(helper.uc, "start", AsyncMock(return_value=browser))
    monkeypatch.setattr(helper.asyncio, "sleep", AsyncMock())
    capture = tmp_path / "page.json"
    assert not asyncio.run(helper.grab_cf_session(url, capture))
    assert not capture.exists() and not helper.CF_SESSION_FILE.exists()
    browser.stop.assert_called_once()


def test_verified_browser_html_handoff(helper, monkeypatch, tmp_path):
    url = "https://www.hltv.org/matches"
    html = '<html><body><div class="matches-list-section">HLTV</div></body></html>'
    tab = SimpleNamespace(
        evaluate=AsyncMock(
            side_effect=lambda expression: {
                "window.location.href": url,
                "document.readyState": "complete",
                "navigator.userAgent": "ua",
            }[expression]
        ),
        get_content=AsyncMock(side_effect=["<html><head>partial load</head></html>", html]),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=tab),
        cookies=SimpleNamespace(get_all=AsyncMock(return_value=[SimpleNamespace(name="cf_clearance", value="secret")])),
        stop=Mock(),
    )
    monkeypatch.setattr(helper.uc, "start", AsyncMock(return_value=browser))
    monkeypatch.setattr(helper.asyncio, "sleep", AsyncMock())
    capture = tmp_path / "page.json"
    assert asyncio.run(helper.grab_cf_session(url, capture))
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["url"] == url and payload["html"] == html
    assert payload["sha256"] == hashlib.sha256(html.encode()).hexdigest()
    assert payload["captured_at"] and "secret" not in capture.read_text()
    assert tab.get_content.await_count == 2

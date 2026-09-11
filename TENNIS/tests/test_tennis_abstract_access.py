"""Offline regressions for the operator-scoped Tennis Abstract acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.responsible_http import HttpResponse, WafBlockedError, parse_robots_txt
from src.tennis_abstract_access import (
    ORIGIN,
    TennisAbstractAcquisitionClient,
    operator_data_exception,
    validate_acquisition_url,
)
from src.tennis_abstract_history_audit import (
    HISTORY_HEADER,
    audit_player,
    literal_array,
    summarize_history,
)


def _history() -> bytes:
    """Build two literal rows with tournament date, not invented match dates."""

    row = [""] * 44
    row[:12] = [
        "20260831",
        "US Open",
        "Hard",
        "G",
        "W",
        "1",
        "1",
        "",
        "R64",
        "6-1 6-1",
        "3",
        "Opponent",
    ]
    row[23] = "60"
    return ("var matchmx = " + repr([row, row]) + ";").encode()


@pytest.mark.parametrize(
    "path",
    [
        "/jsfrags/JannikSinner.js",
        "/jsmatches/ArynaSabalenka.js",
        "/jsmatches/ArynaSabalenkaCareer.js",
        "/jsplayers/curr_rank_atp.js",
    ],
)
def test_permission_is_explicit_and_does_not_change_general_robots(path: str) -> None:
    """The same Disallow still applies without this source's operator permission."""

    url = ORIGIN + path
    policy = parse_robots_txt(
        b"User-agent: *\nDisallow: /jsfrags/\nDisallow: /jsmatches/\nDisallow: /jsplayers/\n"
    )
    assert not policy.allows(url, user_agent="*")
    assert not operator_data_exception(url, confirmed=False)
    assert operator_data_exception(url, confirmed=True)


@pytest.mark.parametrize(
    "url",
    [
        "https://other.test/jsmatches/ArynaSabalenka.js",
        "https://www.tennisabstract.com.evil.test/jsmatches/ArynaSabalenka.js",
        "https://user@www.tennisabstract.com/jsmatches/ArynaSabalenka.js",
        ORIGIN + "/jsmatches/../private.js",
        ORIGIN + "/jsmatches/%2e%2e/private.js",
        ORIGIN + "/jsmatches/ArynaSabalenka.js?extra=1",
        ORIGIN + "/jsplayers/unknown.js",
        ORIGIN + "/cgi-bin/player.cgi?p=JannikSinner&p=Other",
        ORIGIN + "/private/",
        ORIGIN + "/jsmatches/ArynaSabalenka.js#fragment",
    ],
)
def test_scope_cannot_expand_to_arbitrary_urls(url: str) -> None:
    """Reject host tricks, traversal, extra queries and uninspected script paths."""

    assert not operator_data_exception(url, confirmed=True)
    with pytest.raises(ValueError):
        validate_acquisition_url(url)


def test_elo_and_profiles_do_not_gain_a_robots_exception() -> None:
    """Ordinary allowed pages continue through the default robots parser."""

    for suffix in ("/reports/atp_elo_ratings.html", "/cgi-bin/player.cgi?p=JannikSinner"):
        assert validate_acquisition_url(ORIGIN + suffix) == ORIGIN + suffix
        assert not operator_data_exception(ORIGIN + suffix, confirmed=True)


def test_client_uses_scrapling_common_pacing_cache_and_waf(tmp_path: Path) -> None:
    """Only the scoped permission changes; WAF remains an error, never a retry bypass."""

    config = tmp_path / "access.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operator_authorization_confirmed": True,
                "confirmed_on": "2026-09-05",
                "authorization_basis": "fixture",
            }
        ),
        encoding="utf-8",
    )
    with patch("src.tennis_abstract_access.build_http_client") as factory:
        client = TennisAbstractAcquisitionClient(config_path=config, cache_root=tmp_path)
        kwargs = factory.call_args.kwargs
        assert kwargs["transport_kind"] == "scrapling"
        assert kwargs["detect_waf"] is True
        assert kwargs["request_retry_attempts"] == 1
        url = ORIGIN + "/jsmatches/ArynaSabalenka.js"
        assert kwargs["robots_exception"](url)
        factory.return_value.get.side_effect = WafBlockedError(
            "blocked", HttpResponse(403, b"blocked", {}, url, False)
        )
        with pytest.raises(WafBlockedError):
            client.get(url)
        with pytest.raises(ValueError, match="stopped"):
            client.get(url)
        with pytest.raises(ValueError):
            client.get("https://other.test/private")
        assert factory.return_value.get.call_count == 1
        client.close()


def test_literal_history_is_measured_without_inventing_dates() -> None:
    """Preserve the source date semantics and refuse to declare model-ready rows."""

    report = summarize_history(_history())
    assert report["rows"] == 2
    assert report["max_source_date"] == "2026-08-31"
    assert report["rows_with_serve_points"] == 2
    assert report["date_semantics"] == "tournament_date_not_verified_match_date"
    assert report["model_integration_status"].startswith("quarantined")
    assert "match_date" not in report


@pytest.mark.parametrize(
    "text",
    [
        "var matchmx = [run_code()];",
        "var matchmx = [[[['x']]]];",
        "var matchmx = []; trailing",
        "var matchmx = [",
    ],
)
def test_nonliteral_or_empty_data_never_executes(text: str) -> None:
    """Fail closed on executable expressions, excessive nesting or truncated arrays."""

    with pytest.raises((ValueError, SyntaxError)):
        summarize_history(text.encode())


def test_missing_array_and_quoted_brackets_are_handled() -> None:
    """Parse data strings literally, including brackets and escape sequences."""

    assert literal_array("var x = 1;", "matchmx") is None
    assert literal_array("var matchmx = ['[quoted]', 'a\\'b',];", "matchmx") == ["[quoted]", "a'b"]


def test_audit_follows_only_history_link_not_browser_assets() -> None:
    """Download one linked matrix; never load UI JS, other players or invented URLs."""

    client = Mock()
    page = (
        "<script>var matchhead = " + repr(HISTORY_HEADER) + ";</script>"
        '<script src="/jsmatches/ArynaSabalenka.js"></script>'
        '<script src="/jquery.js"></script>'
    ).encode()
    client.get.side_effect = [Mock(content=page), Mock(content=_history())]
    report = audit_player(client, "F", "ArynaSabalenka")
    assert report["rows"] == 2
    assert [call.args[0] for call in client.get.call_args_list] == [
        ORIGIN + "/cgi-bin/wplayer-classic.cgi?p=ArynaSabalenka",
        ORIGIN + "/jsmatches/ArynaSabalenka.js",
    ]


def test_elo_downloader_also_uses_scrapling() -> None:
    """The existing startup Elo downloader now selects the same static transport."""

    from src.tennis_abstract_elo import build_http_session

    with patch("src.tennis_abstract_elo.build_http_client") as factory:
        build_http_session()
        assert factory.call_args.kwargs["transport_kind"] == "scrapling"
        assert factory.call_args.kwargs["detect_waf"] is True


def test_audit_cli_preserves_partial_results_and_stops_after_429(monkeypatch, capsys) -> None:
    """Report a rate-limit once, preserve prior evidence, and never request player three."""

    from scripts import audit_tennis_abstract as script

    url = ORIGIN + "/cgi-bin/wplayer-classic.cgi?p=CocoGauff"
    blocked = WafBlockedError("rate limit", HttpResponse(429, b"blocked", {}, url, False))
    audit = Mock(side_effect=[{"player_key": "JannikSinner", "rows": 565}, blocked])
    client = Mock()
    monkeypatch.setattr(script, "TennisAbstractAcquisitionClient", lambda: client)
    monkeypatch.setattr(script, "audit_player", audit)
    monkeypatch.setattr(
        script.sys,
        "argv",
        ["audit", "--player", "M:JannikSinner", "--player", "F:CocoGauff", "--player", "M:Other"],
    )
    assert script.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["players"] == [{"player_key": "JannikSinner", "rows": 565}]
    assert result["failure"]["status_code"] == 429
    assert result["production_changed"] is False
    assert audit.call_count == 2
    client.close.assert_called_once()

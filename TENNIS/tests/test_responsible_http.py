"""Pruebas offline del cliente HTTP consolidado y su política de acceso."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping
from unittest import mock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.responsible_http import (  # noqa: E402
    HttpClientSettings,
    ResponsibleHttpError,
    RobotsDisallowedError,
    WafBlockedError,
    authorized_matches_endpoint_exception,
    build_http_client,
    parse_robots_txt,
)
from src.tennis_abstract_elo import (  # noqa: E402
    TennisAbstractEloError,
    _validate_public_url,
)
from src.tennis_explorer.client import get_daily_matches  # noqa: E402
from src.tennis_explorer.types import TennisExplorerBlockedError  # noqa: E402
from src.tennisratio.client import TennisRatioClient  # noqa: E402
from src.tennisratio.types import TennisRatioHttpError  # noqa: E402


@dataclass
class _Response:
    """Respuesta requests-like enteramente en memoria."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str


class _Transport:
    """Transporte determinista que registra cada intento real."""

    def __init__(self, routes: Mapping[str, _Response]) -> None:
        """Copia las respuestas y prepara el registro de llamadas."""

        self.routes = dict(routes)
        self.calls: list[str] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Devuelve la fixture exacta o falla ante tráfico imprevisto."""

        del kwargs
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"GET offline no previsto: {url}")
        return self.routes[url]

    def close(self) -> None:
        """Registra el cierre del transporte."""

        self.closed = True


class _Clock:
    """Reloj epoch que avanza únicamente al dormir."""

    def __init__(self) -> None:
        """Empieza en un instante positivo y registra esperas."""

        self.now = 10_000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        """Devuelve el instante actual."""

        return self.now

    def sleep(self, seconds: float) -> None:
        """Avanza exactamente el intervalo solicitado."""

        self.sleeps.append(seconds)
        self.now += seconds


def _settings() -> HttpClientSettings:
    """Devuelve TTL amplios y un mínimo exacto de un segundo."""

    return HttpClientSettings(
        minimum_interval_seconds=1.0,
        robots_ttl_seconds=3_600,
        response_ttl_seconds=3_600,
        lock_timeout_seconds=5.0,
        lock_stale_seconds=60.0,
    )


def _response(url: str, content: bytes, status: int = 200) -> _Response:
    """Construye una respuesta HTML/texto mínima."""

    return _Response(status, content, {"Content-Type": "text/plain"}, url)


def test_real_robots_parser_applies_agent_specificity_and_allow_precedence() -> None:
    """La regla más larga gana y Allow gana un empate exacto."""

    policy = parse_robots_txt(
        """
        User-agent: *
        Disallow: /private/
        Allow: /private/public/

        User-agent: named-bot
        Disallow: /
        Allow: /reports/open$
        """
    )

    assert not policy.allows("https://example.test/private/a", user_agent="other")
    assert policy.allows("https://example.test/private/public/a", user_agent="other")
    assert policy.allows("https://example.test/reports/open", user_agent="named-bot/1")
    assert not policy.allows("https://example.test/reports/open/x", user_agent="named-bot/1")


def test_faster_host_override_requires_documented_api_or_allowlist_basis() -> None:
    """El punto de configuración no permite acelerar un host en silencio."""

    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        path = Path(temporary) / "http.json"
        payload = {
            "default_minimum_interval_seconds": 1.0,
            "host_overrides": {"api.example.test": {"minimum_interval_seconds": 0.25}},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ResponsibleHttpError, match="access_basis"):
            HttpClientSettings.load(path, host="api.example.test")

        payload["host_overrides"]["api.example.test"]["access_basis"] = "documented_api"
        path.write_text(json.dumps(payload), encoding="utf-8")
        settings = HttpClientSettings.load(path, host="api.example.test")

    assert settings.minimum_interval_seconds == pytest.approx(0.25)


def test_disallow_never_reaches_transport_and_cache_obeys_one_second() -> None:
    """Robots bloquea antes del GET; una respuesta fresca sale de caché."""

    robots_url = "https://example.test/robots.txt"
    allowed_url = "https://example.test/public/data"
    transport = _Transport(
        {
            robots_url: _response(
                robots_url,
                b"User-agent: *\nDisallow: /private/\nAllow: /private/public/\n",
            ),
            allowed_url: _response(allowed_url, b"public payload"),
        }
    )
    clock = _Clock()
    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        client = build_http_client(
            transport=transport,
            cache_root=Path(temporary),
            settings=_settings(),
            clock=clock.time,
            sleeper=clock.sleep,
        )
        with pytest.raises(RobotsDisallowedError):
            client.get("https://example.test/private/secret")
        first = client.get(allowed_url)
        second = client.get(allowed_url)

    assert transport.calls == [robots_url, allowed_url]
    assert clock.sleeps == [pytest.approx(1.0)]
    assert not first.from_cache
    assert second.from_cache
    assert first.content == second.content == b"public payload"


def test_tennis_explorer_exception_is_exact_and_no_other_url_bypasses_robots() -> None:
    """Solo la cadena canónica autorizada evita consultar robots."""

    canonical = "https://www.tennisexplorer.com/matches/?day=02&month=09&type=all&year=2026"
    reordered = "https://www.tennisexplorer.com/matches/?month=09&day=02&type=all&year=2026"
    robots_url = "https://www.tennisexplorer.com/robots.txt"
    transport = _Transport(
        {
            canonical: _response(canonical, b"<html>matches</html>"),
            robots_url: _response(robots_url, b"User-agent: *\nDisallow: /matches/\n"),
        }
    )
    clock = _Clock()
    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        client = build_http_client(
            transport=transport,
            cache_root=Path(temporary),
            settings=_settings(),
            robots_exception=authorized_matches_endpoint_exception,
            detect_waf=True,
            clock=clock.time,
            sleeper=clock.sleep,
        )
        assert client.get(canonical).status_code == 200
        with pytest.raises(RobotsDisallowedError):
            client.get(reordered)

    assert authorized_matches_endpoint_exception(canonical)
    assert not authorized_matches_endpoint_exception(reordered)
    assert transport.calls == [canonical, robots_url]


def test_waf_is_never_cached_or_bypassed() -> None:
    """Una señal WAF detiene cada intento y no produce una entrada cacheada."""

    url = "https://www.tennisexplorer.com/matches/?day=02&month=09&type=all&year=2026"
    response = _response(url, b"<title>Just a moment...</title>", status=403)
    transport = _Transport({url: response})
    clock = _Clock()
    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        client = build_http_client(
            transport=transport,
            cache_root=Path(temporary),
            settings=_settings(),
            robots_exception=authorized_matches_endpoint_exception,
            detect_waf=True,
            clock=clock.time,
            sleeper=clock.sleep,
        )
        with pytest.raises(WafBlockedError):
            client.get(url)
        with pytest.raises(WafBlockedError):
            client.get(url)

    assert transport.calls == [url, url]
    assert clock.sleeps == [pytest.approx(1.0)]


def test_per_host_lock_prevents_a_second_process_from_fetching() -> None:
    """Un lock vivo bloquea el transporte en vez de solapar dos GET del host."""

    url = "https://www.tennisexplorer.com/matches/?day=02&month=09&type=all&year=2026"
    transport = _Transport({url: _response(url, b"unused")})
    clock = _Clock()
    settings = HttpClientSettings(
        minimum_interval_seconds=1.0,
        robots_ttl_seconds=3_600,
        response_ttl_seconds=3_600,
        lock_timeout_seconds=0.1,
        lock_stale_seconds=60.0,
    )
    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        cache_root = Path(temporary)
        origin = "https://www.tennisexplorer.com"
        host_key = hashlib.sha256(origin.encode("utf-8")).hexdigest()
        lock_path = cache_root / "hosts" / host_key / "request.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text("another-process\n", encoding="ascii")
        client = build_http_client(
            transport=transport,
            cache_root=cache_root,
            settings=settings,
            robots_exception=authorized_matches_endpoint_exception,
            clock=clock.time,
            sleeper=clock.sleep,
        )

        with pytest.raises(ResponsibleHttpError, match="Timeout esperando lock"):
            client.get(url)

    assert transport.calls == []


def test_shared_client_preserves_tennis_explorer_persistent_waf_circuit() -> None:
    """La ruta real abre el cortacircuitos al recibir WAF del cliente común."""

    url = "https://www.tennisexplorer.com/matches/?day=02&month=09&type=all&year=2026"
    transport = _Transport(
        {
            url: _Response(
                403,
                b"<title>Just a moment...</title>",
                {"Content-Type": "text/html", "CF-Mitigated": "challenge"},
                url,
            )
        }
    )
    with (
        TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary,
        mock.patch(
            "src.tennis_explorer.client.ScraplingHttpSession",
            return_value=transport,
        ),
    ):
        raw_dir = Path(temporary) / "tennis_explorer"
        with pytest.raises(TennisExplorerBlockedError):
            get_daily_matches(
                date(2026, 9, 2),
                raw_dir=raw_dir,
                sleeper=lambda _: None,
                clock=lambda: datetime(2026, 9, 2, 8, tzinfo=UTC),
            )

        circuit = raw_dir / "policy" / "circuit_breaker.json"
        assert circuit.is_file()
        assert json.loads(circuit.read_text(encoding="utf-8"))["reason"] == "http_403"

    assert transport.calls == [url]


def test_tennisratio_obeys_common_robots_before_profile_fetch() -> None:
    """La allow-list de TennisRatio no puede saltarse un Disallow del host."""

    robots_url = "https://www.tennisratio.com/robots.txt"
    transport = _Transport(
        {
            robots_url: _response(
                robots_url,
                b"User-agent: *\nDisallow: /players/\n",
            )
        }
    )
    clock = _Clock()
    with TemporaryDirectory(dir=PROJECT_ROOT / "quality") as temporary:
        shared = build_http_client(
            transport_kind="scrapling",
            transport=transport,
            cache_root=Path(temporary),
            settings=_settings(),
            clock=clock.time,
            sleeper=clock.sleep,
        )
        client = TennisRatioClient(session=shared)
        with pytest.raises(TennisRatioHttpError, match="Scrapling GET failed"):
            client.get(
                "/players/Blocked.html",
                retrieved_at_utc=datetime(2026, 9, 2, tzinfo=UTC),
            )

    assert transport.calls == [robots_url]


def test_tennis_abstract_player_and_js_routes_are_rejected_before_http() -> None:
    """El adaptador solo acepta las dos páginas Elo públicas exactas."""

    for blocked in (
        "https://www.tennisabstract.com/cgi-bin/player.cgi?p=Example",
        "https://www.tennisabstract.com/jsplayers/example.js",
        "https://www.tennisabstract.com/jsmatches/example.js",
    ):
        with pytest.raises(TennisAbstractEloError):
            _validate_public_url("M", blocked)


def test_every_remote_source_builds_the_shared_client() -> None:
    """Impide reintroducir sesiones Requests independientes por fuente."""

    source_paths = (
        "src/sackmann_download.py",
        "src/match_charting_download.py",
        "src/tennis_abstract_elo.py",
        "src/scrapling_release.py",
        "src/tennis_explorer/client.py",
        "src/tennisratio/client.py",
    )
    for relative_path in source_paths:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "build_http_client(" in source, relative_path
        assert "requests.Session()" not in source, relative_path

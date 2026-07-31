"""Tests completamente offline del adaptador Scrapling no evasivo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_explorer.transport import (  # noqa: E402
    SCRAPLING_SESSION_OPTIONS,
    ScraplingHttpSession,
    ScraplingTransportError,
)


class FakeCurlError(Exception):
    """Simula localmente la excepción de transporte de ``curl_cffi``."""


@dataclass
class FakePage:
    """Representa los tres atributos públicos de una respuesta Scrapling."""

    status: int
    body: bytes
    headers: object


class FakeBackend:
    """Registra GET y devuelve o lanza el resultado configurado."""

    def __init__(self, result: FakePage | BaseException) -> None:
        """Guarda el resultado local y crea el registro de llamadas."""

        self.result = result
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
    ) -> FakePage:
        """Registra una única llamada y devuelve el resultado prefijado."""

        self.calls.append((url, dict(headers)))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeSessionManager:
    """Simula el context manager síncrono de ``FetcherSession``."""

    def __init__(self, backend: FakeBackend) -> None:
        """Conserva el backend y contadores de apertura y cierre."""

        self.backend = backend
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> FakeBackend:
        """Abre la sesión falsa exactamente una vez."""

        self.enter_count += 1
        return self.backend

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Registra el cierre sin recursos externos."""

        del exc_type, exc_value, traceback
        self.exit_count += 1


class RecordingFactory:
    """Registra la configuración exacta entregada a ``FetcherSession``."""

    def __init__(self, manager: FakeSessionManager) -> None:
        """Guarda el manager local y prepara una lista de invocaciones."""

        self.manager = manager
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeSessionManager:
        """Devuelve el manager tras copiar las opciones recibidas."""

        self.calls.append(dict(kwargs))
        return self.manager


class ScraplingTransportTest(unittest.TestCase):
    """Verifica configuración, adaptación, cierre y ausencia de reintentos."""

    def test_exact_non_evasive_configuration_and_response_adaptation(
        self,
    ) -> None:
        """Fija todas las opciones sensibles y adapta status/body/headers."""

        page = FakePage(
            status=200,
            body=bytearray(b"<html>ok</html>"),
            headers={"Content-Type": "text/html", "X-Test": 7},
        )
        backend = FakeBackend(page)
        manager = FakeSessionManager(backend)
        factory = RecordingFactory(manager)
        transport = ScraplingHttpSession(
            session_factory=factory,
            backend_error_types=(FakeCurlError,),
        )

        response = transport.get(
            "https://www.tennisexplorer.com/matches/",
            headers={"User-Agent": "fixed", "Accept": "text/html"},
            timeout=(10.0, 30.0),
            allow_redirects=False,
        )

        self.assertEqual(factory.calls, [dict(SCRAPLING_SESSION_OPTIONS)])
        self.assertEqual(
            factory.calls[0],
            {
                "impersonate": None,
                "stealthy_headers": False,
                "http3": False,
                "retries": 1,
                "follow_redirects": False,
                "timeout": 30,
                "verify": True,
            },
        )
        self.assertNotIn("proxy", factory.calls[0])
        self.assertNotIn("proxies", factory.calls[0])
        self.assertNotIn("proxy_rotator", factory.calls[0])
        self.assertEqual(manager.enter_count, 1)
        self.assertEqual(
            backend.calls,
            [
                (
                    "https://www.tennisexplorer.com/matches/",
                    {"User-Agent": "fixed", "Accept": "text/html"},
                )
            ],
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"<html>ok</html>")
        self.assertEqual(
            response.headers,
            {"Content-Type": "text/html", "X-Test": "7"},
        )

        transport.close()
        transport.close()
        self.assertEqual(manager.exit_count, 1)

    def test_context_manager_closes_session(self) -> None:
        """Cierra el manager una vez al abandonar un bloque ``with``."""

        backend = FakeBackend(FakePage(204, b"", {}))
        manager = FakeSessionManager(backend)
        with ScraplingHttpSession(
            session_factory=RecordingFactory(manager),
            backend_error_types=(FakeCurlError,),
        ) as transport:
            response = transport.get(
                "https://www.tennisexplorer.com/robots.txt",
                headers={"User-Agent": "fixed"},
                timeout=(10.0, 30.0),
                allow_redirects=False,
            )
            self.assertEqual(response.status_code, 204)

        self.assertEqual(manager.enter_count, 1)
        self.assertEqual(manager.exit_count, 1)
        with self.assertRaisesRegex(
            ScraplingTransportError,
            "cerrada",
        ):
            transport.get(
                "https://www.tennisexplorer.com/robots.txt",
                headers={"User-Agent": "fixed"},
                timeout=(10.0, 30.0),
                allow_redirects=False,
            )

    def test_curl_error_is_translated_without_second_attempt(self) -> None:
        """Traduce CurlError y comprueba una sola llamada al backend."""

        backend = FakeBackend(FakeCurlError("offline"))
        manager = FakeSessionManager(backend)
        factory = RecordingFactory(manager)
        transport = ScraplingHttpSession(
            session_factory=factory,
            backend_error_types=(FakeCurlError,),
        )

        with self.assertRaisesRegex(
            ScraplingTransportError,
            "única tentativa",
        ) as caught:
            transport.get(
                "https://www.tennisexplorer.com/matches/",
                headers={"User-Agent": "fixed"},
                timeout=(10.0, 30.0),
                allow_redirects=False,
            )

        self.assertIsInstance(caught.exception.__cause__, FakeCurlError)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(factory.calls[0]["retries"], 1)
        transport.close()

    def test_redirect_request_is_rejected_before_backend(self) -> None:
        """Impide que un llamador pueda relajar la política de redirects."""

        backend = FakeBackend(FakePage(200, b"ok", {}))
        manager = FakeSessionManager(backend)
        transport = ScraplingHttpSession(
            session_factory=RecordingFactory(manager),
            backend_error_types=(FakeCurlError,),
        )

        with self.assertRaisesRegex(
            ScraplingTransportError,
            "prohíbe seguir redirects",
        ):
            transport.get(
                "https://www.tennisexplorer.com/matches/",
                headers={"User-Agent": "fixed"},
                timeout=(10.0, 30.0),
                allow_redirects=True,
            )

        self.assertEqual(backend.calls, [])
        transport.close()

    def test_malformed_scrapling_response_fails_explicitly(self) -> None:
        """Rechaza un body textual en vez de inventar bytes o contenido."""

        backend = FakeBackend(
            FakePage(200, "not-bytes", {"Content-Type": "text/html"})  # type: ignore[arg-type]
        )
        transport = ScraplingHttpSession(
            session_factory=RecordingFactory(
                FakeSessionManager(backend)
            ),
            backend_error_types=(FakeCurlError,),
        )

        with self.assertRaisesRegex(
            ScraplingTransportError,
            "no son bytes",
        ):
            transport.get(
                "https://www.tennisexplorer.com/matches/",
                headers={"User-Agent": "fixed"},
                timeout=(10.0, 30.0),
                allow_redirects=False,
            )
        self.assertEqual(len(backend.calls), 1)
        transport.close()


if __name__ == "__main__":
    unittest.main()

"""Tests offline del comprobador de releases estables de Scrapling."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import importlib.util
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapling_release import (  # noqa: E402
    GITHUB_API_VERSION,
    PINNED_SCRAPLING_VERSION,
    REQUEST_TIMEOUT,
    SCRAPLING_LATEST_RELEASE_URL,
    USER_AGENT,
    ScraplingReleaseCheck,
    ScraplingReleaseCheckError,
    build_http_session,
    check_latest_scrapling_release,
)


RELEASE_URL = (
    "https://github.com/D4Vinci/Scrapling/releases/tag/v0.4.12"
)


class FakeResponse:
    """Representa una respuesta local compatible con el comprobador."""

    def __init__(
        self,
        status_code: int,
        *,
        payload: object | None = None,
        headers: dict[str, str] | None = None,
        json_error: ValueError | None = None,
    ) -> None:
        """Guarda estado, JSON y cabeceras sin realizar tráfico."""

        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {
            "Content-Type": "application/vnd.github+json; charset=utf-8"
        }
        self.json_error = json_error

    def json(self) -> object:
        """Devuelve el JSON prefijado o simula un documento inválido."""

        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    """Registra una única petición y devuelve una respuesta prefijada."""

    def __init__(
        self,
        responses: list[FakeResponse | requests.RequestException],
    ) -> None:
        """Inicializa la cola de respuestas y el registro de llamadas."""

        self.responses = list(responses)
        self.calls: list[
            tuple[str, dict[str, str], tuple[float, float], bool]
        ] = []
        self.closed = False

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> FakeResponse:
        """Consume exactamente una respuesta o un error de transporte."""

        self.calls.append(
            (url, dict(headers), timeout, allow_redirects)
        )
        if not self.responses:
            raise AssertionError("Se realizó un GET offline no previsto.")
        response = self.responses.pop(0)
        if isinstance(response, requests.RequestException):
            raise response
        return response

    def close(self) -> None:
        """Marca el cierre de la sesión falsa."""

        self.closed = True


def _release_payload(
    tag_name: str,
    *,
    draft: bool = False,
    prerelease: bool = False,
) -> dict[str, object]:
    """Construye un payload mínimo con la forma oficial de una release."""

    return {
        "tag_name": tag_name,
        "html_url": RELEASE_URL.replace("v0.4.12", tag_name),
        "draft": draft,
        "prerelease": prerelease,
        "published_at": date(2026, 7, 30).isoformat(),
    }


def _load_script_module() -> object:
    """Carga el script como módulo sin ejecutar su bloque ``__main__``."""

    script_path = PROJECT_ROOT / "scripts" / "check_scrapling_release.py"
    spec = importlib.util.spec_from_file_location(
        "check_scrapling_release_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("No se pudo crear el spec del script.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScraplingReleaseTests(unittest.TestCase):
    """Valida comparación, red mínima, errores y códigos del CLI."""

    def test_matching_latest_stable_release_returns_current(self) -> None:
        """Acepta el prefijo ``v`` y confirma el pin estable exacto."""

        session = FakeSession(
            [
                FakeResponse(
                    200,
                    payload=_release_payload("v0.4.12"),
                )
            ]
        )

        result = check_latest_scrapling_release(session=session)

        self.assertEqual(result.pinned_version, PINNED_SCRAPLING_VERSION)
        self.assertEqual(result.latest_version, "0.4.12")
        self.assertEqual(result.tag_name, "v0.4.12")
        self.assertFalse(result.update_available)
        self.assertFalse(session.closed)
        self.assertEqual(len(session.calls), 1)
        url, headers, timeout, allow_redirects = session.calls[0]
        self.assertEqual(url, SCRAPLING_LATEST_RELEASE_URL)
        self.assertEqual(headers["Accept"], "application/vnd.github+json")
        self.assertEqual(
            headers["X-GitHub-Api-Version"],
            GITHUB_API_VERSION,
        )
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(timeout, REQUEST_TIMEOUT)
        self.assertFalse(allow_redirects)

    def test_newer_stable_release_is_reported_without_updating(self) -> None:
        """Informa una release posterior y no efectúa una segunda petición."""

        session = FakeSession(
            [
                FakeResponse(
                    200,
                    payload=_release_payload("v0.4.13"),
                )
            ]
        )

        result = check_latest_scrapling_release(session=session)

        self.assertTrue(result.update_available)
        self.assertEqual(result.latest_version, "0.4.13")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.responses, [])

    def test_request_session_has_zero_retries(self) -> None:
        """Configura explícitamente ambos adaptadores con cero reintentos."""

        session = build_http_session()
        try:
            self.assertEqual(
                session.get_adapter("https://").max_retries.total,
                0,
            )
            self.assertEqual(
                session.get_adapter("http://").max_retries.total,
                0,
            )
        finally:
            session.close()

    def test_transport_error_is_not_retried(self) -> None:
        """Traduce el primer error de red y conserva un único GET."""

        session = FakeSession([requests.ConnectionError("offline")])

        with self.assertRaisesRegex(
            ScraplingReleaseCheckError,
            "no se reintentó",
        ):
            check_latest_scrapling_release(session=session)

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.responses, [])

    def test_invalid_http_json_and_release_contract_fail_clearly(self) -> None:
        """Rechaza HTTP, MIME, JSON, borradores, prereleases y tags dudosos."""

        cases = (
            (
                FakeResponse(403, payload={}),
                "HTTP 403",
            ),
            (
                FakeResponse(
                    200,
                    payload={},
                    headers={"Content-Type": "text/html"},
                ),
                "Content-Type",
            ),
            (
                FakeResponse(
                    200,
                    json_error=ValueError("bad json"),
                ),
                "JSON válido",
            ),
            (
                FakeResponse(
                    200,
                    payload=_release_payload("v0.4.12", draft=True),
                ),
                "borrador",
            ),
            (
                FakeResponse(
                    200,
                    payload=_release_payload(
                        "v0.4.13-rc1",
                        prerelease=True,
                    ),
                ),
                "prerelease",
            ),
            (
                FakeResponse(
                    200,
                    payload=_release_payload("release-main"),
                ),
                "versión estable numérica",
            ),
        )
        for response, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    ScraplingReleaseCheckError,
                    message,
                ):
                    check_latest_scrapling_release(
                        session=FakeSession([response])
                    )

    def test_remote_release_older_than_pin_is_an_error(self) -> None:
        """No interpreta una release remota anterior como estado actualizado."""

        session = FakeSession(
            [
                FakeResponse(
                    200,
                    payload=_release_payload("v0.4.11"),
                )
            ]
        )

        with self.assertRaisesRegex(
            ScraplingReleaseCheckError,
            "anterior al pin",
        ):
            check_latest_scrapling_release(session=session)

    def test_cli_exit_codes_and_messages(self) -> None:
        """Devuelve 0 al coincidir, 1 ante actualización y 2 ante error."""

        script = _load_script_module()
        current = ScraplingReleaseCheck(
            pinned_version="0.4.12",
            latest_version="0.4.12",
            tag_name="v0.4.12",
            release_url=RELEASE_URL,
        )
        newer = ScraplingReleaseCheck(
            pinned_version="0.4.12",
            latest_version="0.4.13",
            tag_name="v0.4.13",
            release_url=RELEASE_URL.replace("v0.4.12", "v0.4.13"),
        )

        stdout = StringIO()
        with (
            patch.object(
                script,
                "check_latest_scrapling_release",
                return_value=current,
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(script.main([]), 0)
        self.assertIn("coincide", stdout.getvalue())

        stderr = StringIO()
        with (
            patch.object(
                script,
                "check_latest_scrapling_release",
                return_value=newer,
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(script.main([]), 1)
        self.assertIn("NUEVA RELEASE ESTABLE", stderr.getvalue())

        stderr = StringIO()
        with (
            patch.object(
                script,
                "check_latest_scrapling_release",
                side_effect=ScraplingReleaseCheckError("sin API"),
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(script.main([]), 2)
        self.assertIn("ERROR: sin API", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

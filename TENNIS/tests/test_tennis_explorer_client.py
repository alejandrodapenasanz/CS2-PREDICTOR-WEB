"""Tests offline del cliente responsable y la caché de Tennis Explorer."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_explorer.client import (  # noqa: E402
    MIN_REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    USER_AGENT,
    build_daily_url,
    get_daily_matches,
    refresh_daily_results,
)
from src.tennis_explorer.parser import (  # noqa: E402
    empty_matches_dataframe,
)
from src.tennis_explorer.types import (  # noqa: E402
    TennisExplorerBlockedError,
    TennisExplorerCacheError,
    TennisExplorerHttpError,
    TennisExplorerSchemaError,
)
from src.tennis_explorer.transport import (  # noqa: E402
    ScraplingTransportError,
)


MATCH_DATE = date(2026, 7, 30)
NOW = datetime(2026, 7, 30, 10, 15, tzinfo=UTC)
DAILY_HTML = b"<!doctype html><html><body>daily matches</body></html>"


class FakeResponse:
    """Representa una respuesta HTTP local con los atributos consumidos."""

    def __init__(
        self,
        status_code: int,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Guarda estado, cuerpo y cabeceras sin ninguna conexión de red."""

        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    """Consume respuestas prefijadas y registra cada GET del cliente."""

    def __init__(
        self,
        responses: list[FakeResponse | ScraplingTransportError],
    ) -> None:
        """Inicializa la cola local, el registro y el indicador de cierre."""

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
        """Devuelve la siguiente respuesta y falla ante un GET imprevisto."""

        self.calls.append(
            (url, dict(headers), timeout, allow_redirects)
        )
        if not self.responses:
            raise AssertionError("El cliente hizo un GET no previsto.")
        response = self.responses.pop(0)
        if isinstance(response, ScraplingTransportError):
            raise response
        return response

    def close(self) -> None:
        """Marca el cierre sin liberar recursos externos."""

        self.closed = True


def _clock() -> datetime:
    """Devuelve un instante fijo y consciente de UTC para los tests."""

    return NOW


def _clock_at(value: datetime) -> Callable[[], datetime]:
    """Crea un reloj fijo inyectable para snapshots consecutivos."""

    def _read() -> datetime:
        """Devuelve el instante capturado por el reloj local."""

        return value

    return _read


def _one_row_frame() -> pd.DataFrame:
    """Crea un resultado mínimo para aislar el cliente del parser."""

    return pd.DataFrame(
        {"parser_marker": pd.Series(["validated"], dtype="string")}
    )


def _successful_responses() -> list[FakeResponse]:
    """Devuelve la única respuesta offline de una descarga inicial."""

    return [
        FakeResponse(
            200,
            content=DAILY_HTML,
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    ]


class TennisExplorerClientTest(unittest.TestCase):
    """Valida red mínima, política, integridad y parada segura del cliente."""

    def setUp(self) -> None:
        """Crea un directorio temporal exclusivamente dentro de TENNIS/tests."""

        self.temporary = TemporaryDirectory(
            prefix="te-client-",
            dir=PROJECT_ROOT / "tests",
        )
        self.raw_dir = Path(self.temporary.name) / "raw"

    def tearDown(self) -> None:
        """Elimina únicamente el directorio temporal creado por este test."""

        self.temporary.cleanup()

    def test_first_fetch_uses_one_request_then_cache_avoids_all_network(
        self,
    ) -> None:
        """Usa un GET la primera vez y cero GET para la fecha cacheada."""

        first_session = FakeSession(_successful_responses())
        delays: list[float] = []
        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ) as parser_mock:
            first = get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=first_session,
                sleeper=delays.append,
                clock=_clock,
            )

            cached_session = FakeSession([])
            cached_delays: list[float] = []
            second = get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=cached_session,
                sleeper=cached_delays.append,
                clock=_clock,
            )

        expected_url = build_daily_url(MATCH_DATE)
        self.assertEqual(
            [call[0] for call in first_session.calls],
            [expected_url],
        )
        self.assertEqual(
            delays,
            [MIN_REQUEST_DELAY_SECONDS],
        )
        self.assertEqual(cached_session.calls, [])
        self.assertEqual(cached_delays, [])
        self.assertEqual(parser_mock.call_count, 2)
        for _, headers, timeout, allow_redirects in first_session.calls:
            self.assertEqual(headers["User-Agent"], USER_AGENT)
            self.assertEqual(headers["Accept-Language"], "en-US,en;q=0.9")
            self.assertNotIn("Cache-Control", headers)
            self.assertEqual(timeout, REQUEST_TIMEOUT)
            self.assertFalse(allow_redirects)

        expected_hash = hashlib.sha256(DAILY_HTML).hexdigest()
        self.assertEqual(first.loc[0, "source_url"], expected_url)
        self.assertEqual(first.loc[0, "snapshot_sha256"], expected_hash)
        self.assertEqual(
            first.loc[0, "retrieved_at_utc"],
            pd.Timestamp("2026-07-30T10:15:00Z"),
        )
        pd.testing.assert_frame_equal(first, second)
        for column in ("source_url", "snapshot_sha256"):
            self.assertEqual(str(first[column].dtype), "string")
        self.assertEqual(
            str(first["retrieved_at_utc"].dtype),
            "datetime64[ns, UTC]",
        )

        html_path = (
            self.raw_dir / "daily" / "2026" / "2026-07-30.html"
        )
        metadata_path = html_path.with_name(
            "2026-07-30.metadata.json"
        )
        self.assertEqual(html_path.read_bytes(), DAILY_HTML)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["match_date"], "2026-07-30")
        self.assertEqual(metadata["sha256"], expected_hash)
        self.assertEqual(metadata["size_bytes"], len(DAILY_HTML))
        self.assertEqual(metadata["source_url"], expected_url)
        self.assertEqual(
            metadata["access_basis"],
            "user_reported_operator_authorization",
        )
        self.assertEqual(
            metadata["robots_policy"],
            "authorized_matches_endpoint_exception",
        )
        self.assertEqual(
            metadata["transport"],
            "scrapling_fetcher_session",
        )

    def test_result_refreshes_are_append_only_and_do_not_touch_daily_cache(
        self,
    ) -> None:
        """Permite varias observaciones, un GET cada una y cero sobrescrituras."""

        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )
        daily_path = (
            self.raw_dir / "daily" / "2026" / "2026-07-30.html"
        )
        daily_before = daily_path.read_bytes()

        snapshots = []
        moments = (
            datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
            datetime(2026, 7, 30, 15, 30, tzinfo=UTC),
            datetime(2026, 7, 31, 8, 0, tzinfo=UTC),
        )
        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ) as parser_mock:
            for moment in moments:
                session = FakeSession(_successful_responses())
                delays: list[float] = []
                snapshot = refresh_daily_results(
                    MATCH_DATE,
                    raw_dir=self.raw_dir,
                    session=session,
                    sleeper=delays.append,
                    clock=_clock_at(moment),
                )
                snapshots.append(snapshot)
                self.assertEqual(len(session.calls), 1)
                self.assertEqual(delays, [MIN_REQUEST_DELAY_SECONDS])

        self.assertEqual(parser_mock.call_count, 3)
        self.assertEqual(daily_path.read_bytes(), daily_before)
        self.assertEqual(len({item.html_path for item in snapshots}), 3)
        result_dir = (
            self.raw_dir
            / "results"
            / "2026"
            / "2026-07-30"
        )
        self.assertEqual(len(list(result_dir.glob("*.html"))), 3)
        self.assertEqual(len(list(result_dir.glob("*.metadata.json"))), 3)
        for snapshot, moment in zip(snapshots, moments, strict=True):
            self.assertEqual(snapshot.match_date, MATCH_DATE)
            self.assertEqual(snapshot.retrieved_at_utc, moment)
            self.assertTrue(snapshot.html_path.is_file())
            self.assertTrue(snapshot.metadata_path.is_file())
            self.assertEqual(
                snapshot.matches.loc[0, "snapshot_sha256"],
                snapshot.snapshot_sha256,
            )
            metadata = json.loads(
                snapshot.metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["artifact"],
                "tennis_explorer_results_snapshot",
            )
            self.assertEqual(
                metadata["snapshot_purpose"],
                "result_refresh",
            )

    def test_result_refresh_validates_schema_before_publishing_snapshot(
        self,
    ) -> None:
        """Un cambio HTML falla tras un GET y no deja snapshots parciales."""

        session = FakeSession(_successful_responses())
        with (
            patch(
                "src.tennis_explorer.client.parse_daily_matches_html",
                side_effect=TennisExplorerSchemaError(
                    "estructura inesperada"
                ),
            ),
            self.assertRaises(TennisExplorerSchemaError),
        ):
            refresh_daily_results(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=session,
                sleeper=lambda _: None,
                clock=_clock,
            )

        self.assertEqual(len(session.calls), 1)
        results_dir = self.raw_dir / "results"
        self.assertFalse(results_dir.exists())
        self.assertFalse(
            (
                self.raw_dir
                / "daily"
                / "2026"
                / "2026-07-30.html"
            ).exists()
        )

    def test_result_refresh_opens_circuit_on_waf_without_retry(self) -> None:
        """El refresco comparte la parada WAF y nunca publica el desafío."""

        session = FakeSession(
            [
                FakeResponse(
                    429,
                    content=b"Too many requests",
                    headers={"Content-Type": "text/html"},
                )
            ]
        )
        with self.assertRaises(TennisExplorerBlockedError):
            refresh_daily_results(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=session,
                sleeper=lambda _: None,
                clock=_clock,
            )

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.responses, [])
        self.assertTrue(
            (
                self.raw_dir
                / "policy"
                / "circuit_breaker.json"
            ).is_file()
        )
        self.assertFalse((self.raw_dir / "results").exists())

    def test_empty_result_refresh_keeps_snapshot_level_provenance(
        self,
    ) -> None:
        """Una jornada vacía conserva SHA, UTC y rutas fuera del DataFrame."""

        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=empty_matches_dataframe(),
        ):
            snapshot = refresh_daily_results(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )

        self.assertTrue(snapshot.matches.empty)
        self.assertEqual(
            snapshot.snapshot_sha256,
            hashlib.sha256(DAILY_HTML).hexdigest(),
        )
        self.assertEqual(snapshot.retrieved_at_utc, NOW)
        self.assertTrue(snapshot.html_path.is_file())
        self.assertTrue(snapshot.metadata_path.is_file())

    def test_default_path_builds_and_closes_scrapling_transport(self) -> None:
        """Usa el adaptador Scrapling cuando no se inyecta una sesión."""

        default_session = FakeSession(_successful_responses())
        with (
            patch(
                "src.tennis_explorer.client.ScraplingHttpSession",
                return_value=default_session,
            ) as transport_factory,
            patch(
                "src.tennis_explorer.client.parse_daily_matches_html",
                return_value=_one_row_frame(),
            ),
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                sleeper=lambda _: None,
                clock=_clock,
            )

        transport_factory.assert_called_once_with()
        self.assertEqual(len(default_session.calls), 1)
        self.assertTrue(default_session.closed)

    def test_corrupt_daily_cache_stops_without_redownload(self) -> None:
        """Rechaza un hash distinto y no intenta reparar la caché por red."""

        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )

        html_path = (
            self.raw_dir / "daily" / "2026" / "2026-07-30.html"
        )
        html_path.write_bytes(b"X" * len(DAILY_HTML))
        no_network = FakeSession([])
        with self.assertRaisesRegex(
            TennisExplorerCacheError,
            "SHA-256",
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=no_network,
                sleeper=lambda _: None,
                clock=_clock,
            )
        self.assertEqual(no_network.calls, [])

    def test_partial_cache_stops_without_redownload(self) -> None:
        """Considera corrupto un HTML sin metadata y conserva cero GET."""

        html_path = (
            self.raw_dir / "daily" / "2026" / "2026-07-30.html"
        )
        html_path.parent.mkdir(parents=True)
        html_path.write_bytes(DAILY_HTML)
        no_network = FakeSession([])

        with self.assertRaisesRegex(
            TennisExplorerCacheError,
            "incompleta",
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=no_network,
                clock=_clock,
            )
        self.assertEqual(no_network.calls, [])

    def test_authorized_exception_rejects_every_noncanonical_url(self) -> None:
        """Impide extender la excepción a otro host, ruta o consulta."""

        invalid_urls = (
            "http://www.tennisexplorer.com/matches/"
            "?day=30&month=07&type=all&year=2026",
            "https://tennisexplorer.com/matches/"
            "?day=30&month=07&type=all&year=2026",
            "https://www.tennisexplorer.com/player/example/",
            "https://www.tennisexplorer.com/matches/"
            "?day=30&month=07&type=all&year=2026&extra=1",
            "https://www.tennisexplorer.com/matches/"
            "?day=30&month=07&type=atp&year=2026",
        )
        for index, invalid_url in enumerate(invalid_urls):
            with self.subTest(index=index, url=invalid_url):
                no_network = FakeSession([])
                delays: list[float] = []
                with (
                    patch(
                        "src.tennis_explorer.client.build_daily_url",
                        return_value=invalid_url,
                    ),
                    self.assertRaisesRegex(
                        TennisExplorerHttpError,
                        "solo cubre",
                    ),
                ):
                    get_daily_matches(
                        MATCH_DATE,
                        raw_dir=self.raw_dir / f"scope-{index}",
                        session=no_network,
                        sleeper=delays.append,
                        clock=_clock,
                    )
                self.assertEqual(no_network.calls, [])
                self.assertEqual(delays, [])

    def test_each_uncached_date_waits_before_single_request(self) -> None:
        """Separa temporalmente las peticiones de dos fechas no cacheadas."""

        next_date = date(2026, 7, 31)
        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )

            next_session = FakeSession(
                [
                    FakeResponse(
                        200,
                        content=DAILY_HTML,
                        headers={"Content-Type": "text/html"},
                    )
                ]
            )
            delays: list[float] = []
            get_daily_matches(
                next_date,
                raw_dir=self.raw_dir,
                session=next_session,
                sleeper=delays.append,
                clock=_clock,
            )

        self.assertEqual(
            [call[0] for call in next_session.calls],
            [build_daily_url(next_date)],
        )
        self.assertEqual(delays, [MIN_REQUEST_DELAY_SECONDS])

    def test_existing_network_lock_fails_before_sleep_or_get(self) -> None:
        """Evita tráfico concurrente y conserva el lock para revisión."""

        lock_path = self.raw_dir / "policy" / "network.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text("existing execution\n", encoding="utf-8")
        no_network = FakeSession([])
        delays: list[float] = []

        with self.assertRaisesRegex(
            TennisExplorerHttpError,
            "lock abandonado",
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=no_network,
                sleeper=delays.append,
                clock=_clock,
            )

        self.assertEqual(no_network.calls, [])
        self.assertEqual(delays, [])
        self.assertTrue(lock_path.exists())

    def test_403_429_and_challenge_stop_without_retry(self) -> None:
        """Trata rate-limit y desafío Cloudflare como bloqueos terminales."""

        blocked_responses = (
            FakeResponse(
                403,
                content=b"Forbidden",
                headers={"Content-Type": "text/html"},
            ),
            FakeResponse(
                429,
                content=b"Too many requests",
                headers={"Content-Type": "text/html"},
            ),
            FakeResponse(
                200,
                content=(
                    b"<title>Just a moment...</title>"
                    b"<script src='/cdn-cgi/challenge-platform/x'></script>"
                ),
                headers={"Content-Type": "text/html"},
            ),
        )
        for index, blocked in enumerate(blocked_responses):
            with self.subTest(index=index, status=blocked.status_code):
                case_dir = self.raw_dir / f"blocked-{index}"
                session = FakeSession([blocked])
                with (
                    patch(
                        "src.tennis_explorer.client."
                        "parse_daily_matches_html"
                    ) as parser_mock,
                    self.assertRaises(
                        TennisExplorerBlockedError
                    ),
                ):
                    get_daily_matches(
                        MATCH_DATE,
                        raw_dir=case_dir,
                        session=session,
                        sleeper=lambda _: None,
                        clock=_clock,
                    )
                self.assertEqual(len(session.calls), 1)
                self.assertEqual(session.responses, [])
                parser_mock.assert_not_called()
                self.assertTrue(
                    (
                        case_dir
                        / "policy"
                        / "circuit_breaker.json"
                    ).exists()
                )
                self.assertFalse(
                    (
                        case_dir
                        / "daily"
                        / "2026"
                        / "2026-07-30.html"
                    ).exists()
                )

    def test_circuit_breaker_blocks_new_network_but_not_valid_cache(
        self,
    ) -> None:
        """Conserva la caché útil y corta futuras fechas tras un bloqueo."""

        cached_date = MATCH_DATE
        blocked_date = date(2026, 7, 31)
        later_date = date(2026, 8, 1)
        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=_one_row_frame(),
        ):
            cached_frame = get_daily_matches(
                cached_date,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )
            with self.assertRaises(TennisExplorerBlockedError):
                get_daily_matches(
                    blocked_date,
                    raw_dir=self.raw_dir,
                    session=FakeSession(
                        [
                            FakeResponse(
                                403,
                                content=b"Forbidden",
                                headers={"Content-Type": "text/html"},
                            )
                        ]
                    ),
                    sleeper=lambda _: None,
                    clock=_clock,
                )

            cached_session = FakeSession([])
            loaded_frame = get_daily_matches(
                cached_date,
                raw_dir=self.raw_dir,
                session=cached_session,
                sleeper=lambda _: None,
                clock=_clock,
            )

            stopped_delays: list[float] = []
            with (
                patch(
                    "src.tennis_explorer.client._build_http_session"
                ) as transport_builder,
                self.assertRaisesRegex(
                    TennisExplorerBlockedError,
                    "cortacircuitos",
                ),
            ):
                get_daily_matches(
                    later_date,
                    raw_dir=self.raw_dir,
                    sleeper=stopped_delays.append,
                    clock=_clock,
                )

        pd.testing.assert_frame_equal(cached_frame, loaded_frame)
        self.assertEqual(cached_session.calls, [])
        self.assertEqual(stopped_delays, [])
        transport_builder.assert_not_called()

    def test_redirect_and_unexpected_content_type_are_not_followed(
        self,
    ) -> None:
        """Falla ante redirect o JSON y nunca publica esos cuerpos."""

        invalid_responses = (
            FakeResponse(
                302,
                headers={"Location": "/blocked"},
            ),
            FakeResponse(
                200,
                content=b'{"error": "not html"}',
                headers={"Content-Type": "application/json"},
            ),
        )
        for index, invalid in enumerate(invalid_responses):
            with self.subTest(index=index):
                case_dir = self.raw_dir / f"invalid-{index}"
                session = FakeSession([invalid])
                with (
                    patch(
                        "src.tennis_explorer.client."
                        "parse_daily_matches_html"
                    ) as parser_mock,
                    self.assertRaises(TennisExplorerHttpError),
                ):
                    get_daily_matches(
                        MATCH_DATE,
                        raw_dir=case_dir,
                        session=session,
                        sleeper=lambda _: None,
                        clock=_clock,
                    )
                self.assertEqual(len(session.calls), 1)
                parser_mock.assert_not_called()
                self.assertFalse(
                    (
                        case_dir
                        / "daily"
                        / "2026"
                        / "2026-07-30.html"
                    ).exists()
                )

    def test_schema_is_validated_before_daily_cache_is_published(
        self,
    ) -> None:
        """No activa HTML ni metadata cuando el parser detecta un cambio."""

        session = FakeSession(_successful_responses())
        with (
            patch(
                "src.tennis_explorer.client.parse_daily_matches_html",
                side_effect=TennisExplorerSchemaError(
                    "estructura inesperada"
                ),
            ),
            self.assertRaises(TennisExplorerSchemaError),
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=session,
                sleeper=lambda _: None,
                clock=_clock,
            )

        daily_dir = self.raw_dir / "daily" / "2026"
        self.assertFalse((daily_dir / "2026-07-30.html").exists())
        self.assertFalse(
            (daily_dir / "2026-07-30.metadata.json").exists()
        )
        self.assertFalse(
            (daily_dir / "2026-07-30.html.part").exists()
        )
        self.assertFalse(
            (daily_dir / "2026-07-30.metadata.json.part").exists()
        )

    def test_day_without_matches_keeps_typed_empty_schema_and_audit(
        self,
    ) -> None:
        """Añade auditoría tipada sin alterar el esquema vacío del parser."""

        with patch(
            "src.tennis_explorer.client.parse_daily_matches_html",
            return_value=empty_matches_dataframe(),
        ):
            frame = get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=FakeSession(_successful_responses()),
                sleeper=lambda _: None,
                clock=_clock,
            )

        self.assertTrue(frame.empty)
        self.assertEqual(str(frame["match_date"].dtype), "datetime64[ns]")
        for column in ("source_url", "snapshot_sha256"):
            self.assertIn(column, frame.columns)
            self.assertEqual(str(frame[column].dtype), "string")
        self.assertIn("retrieved_at_utc", frame.columns)
        self.assertEqual(
            str(frame["retrieved_at_utc"].dtype),
            "datetime64[ns, UTC]",
        )

    def test_transport_error_is_not_retried(self) -> None:
        """Propaga un único fallo de red sin emitir una segunda petición."""

        session = FakeSession(
            [ScraplingTransportError("offline")]
        )
        with self.assertRaisesRegex(
            TennisExplorerHttpError,
            "única petición",
        ):
            get_daily_matches(
                MATCH_DATE,
                raw_dir=self.raw_dir,
                session=session,
                sleeper=lambda _: None,
                clock=_clock,
            )
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.responses, [])


if __name__ == "__main__":
    unittest.main()

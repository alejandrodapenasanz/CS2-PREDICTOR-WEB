"""Tests offline del descargador y parser Elo público de Tennis Abstract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
)
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_abstract_elo import (  # noqa: E402
    ELO_URLS,
    USER_AGENT,
    EloSchemaError,
    download_tennis_abstract_elo,
    load_elo_snapshot,
    parse_elo_html,
)


ATP_HTML = """\
<!doctype html>
<html>
<body>
<table width="600px"><tr><td>Navigation</td></tr></table>
<table width="1000px">
<tr><td>
<p><b>Current Elo ratings for the ATP tour</b>.</p>
<p><i>Updated weekly(ish). Last update: 2026-07-27</i></p>
</td></tr>
<tr><td>
<table id="reportable" class="tablesorter">
<thead><tr>
<th align="right">Elo&nbsp;Rank</th>
<th align="left">Player</th>
<th align="right">Age</th>
<th align="right">Elo</th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">hElo&nbsp;Rank</th>
<th align="right"><span title="Hard-court Elo rating">hElo</span></th>
<th align="right">cElo&nbsp;Rank</th>
<th align="right"><span title="Clay-court Elo rating">cElo</span></th>
<th align="right">gElo&nbsp;Rank</th>
<th align="right"><span title="Grass-court Elo rating">gElo</span></th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">Peak&nbsp;Elo</th>
<th align="right"><span title="When peak was achieved">Peak&nbsp;Month</span></th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">ATP&nbsp;Rank</th>
<th align="right"><span title="Rank difference">Log&nbsp;diff</span></th>
</tr></thead>
<tbody>
<tr>
<td align="right">1</td>
<td><a href="https://www.tennisabstract.com/cgi-bin/player.cgi?p=JannikSinner">
Jannik&nbsp;Sinner</a></td>
<td align="right">24.8</td><td align="right">2331.9</td><td></td>
<td align="right">1</td><td align="right">2269.3</td>
<td align="right">1</td><td align="right">2221.8</td>
<td align="right">1</td><td align="right">2135.5</td><td></td>
<td align="right">2339.8</td><td align="right">2026-05</td><td></td>
<td align="right">1</td><td align="right">0</td>
</tr>
<tr>
<td align="right">2</td>
<td><a href="https://www.tennisabstract.com/cgi-bin/player.cgi?p=UnknownAge">
Unknown&nbsp;Age</a></td>
<td align="right"></td><td align="right">2100.0</td><td></td>
<td align="right">2</td><td align="right">2050.0</td>
<td align="right">2</td><td align="right">2040.0</td>
<td align="right">2</td><td align="right">2030.0</td><td></td>
<td align="right">2110.0</td><td align="right">2026-01</td><td></td>
<td align="right"></td><td align="right"></td>
</tr>
</tbody></table>
</td></tr></table>
</body>
</html>
"""

WTA_HTML = """\
<!doctype html>
<html>
<body>
<p><b>Current Elo ratings for the WTA tour</b>.</p>
<p><i>Updated weekly(ish). Last update: 2026-07-27</i></p>
<table id="reportable" class="tablesorter">
<thead><tr>
<th align="right">Elo&nbsp;Rank</th>
<th align="left">Player</th>
<th align="right">Age</th>
<th align="right">Elo</th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">hElo&nbsp;Rank</th>
<th align="right"><span>hElo</span></th>
<th align="right">cElo&nbsp;Rank</th>
<th align="right"><span>cElo</span></th>
<th align="right">gElo&nbsp;Rank</th>
<th align="right"><span>gElo</span></th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">Peak&nbsp;Elo</th>
<th align="right"><span>Peak&nbsp;Month</span></th>
<th align="left">&nbsp;&nbsp;&nbsp;&nbsp;</th>
<th align="right">WTA&nbsp;Rank</th>
<th align="right"><span>Log&nbsp;diff</span></th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td><a href="https://www.tennisabstract.com/cgi-bin/wplayer.cgi?p=ArynaSabalenka">
Aryna&nbsp;Sabalenka</a></td>
<td align="right">28.1</td><td align="right">2209.4</td><td></td>
<td align="right">1</td><td align="right">2196.0</td>
<td align="right">1</td><td align="right">2097.9</td>
<td align="right">1</td><td align="right">1983.6</td><td></td>
<td align="right">2271.7</td><td align="right">2026-04</td><td></td>
<td align="right">1</td><td align="right">0</td>
</tr></tbody>
</table>
</body>
</html>
"""


class FakeResponse:
    """Representa una respuesta HTTP mínima y enteramente local."""

    def __init__(
        self,
        status_code: int,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Guarda status, bytes y headers usados por el descargador."""

        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    """Devuelve respuestas prefijadas y registra cada GET solicitado."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
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
        """Consume una respuesta y falla si aparece un GET no previsto."""

        self.calls.append((url, dict(headers), timeout, allow_redirects))
        if not self.responses:
            raise AssertionError("Se produjo un GET no previsto por el test.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        """Marca la sesión como cerrada sin acceder a la red."""

        self.closed = True


class TennisAbstractEloParserTest(unittest.TestCase):
    """Valida firma HTML, tipos, ausencias observadas y separación de género."""

    def test_atp_parser_returns_typed_dataframe(self) -> None:
        """Parsea ATP, preserva ausencias y asigna tipos nullable explícitos."""

        retrieved_at = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        frame = parse_elo_html(
            ATP_HTML,
            gender="M",
            retrieved_at_utc=retrieved_at,
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["gender"]), {"M"})
        self.assertEqual(frame.loc[0, "player_name"], "Jannik Sinner")
        self.assertEqual(str(frame["elo_rank"].dtype), "Int64")
        self.assertEqual(str(frame["elo"].dtype), "Float64")
        self.assertTrue(pd.isna(frame.loc[1, "age"]))
        self.assertTrue(pd.isna(frame.loc[1, "official_rank"]))
        self.assertTrue(is_datetime64_any_dtype(frame["rating_date"]))
        self.assertTrue(is_datetime64_any_dtype(frame["peak_month"]))
        self.assertIsInstance(
            frame["retrieved_at_utc"].dtype,
            pd.DatetimeTZDtype,
        )
        self.assertEqual(frame.loc[0, "rating_date"], pd.Timestamp("2026-07-27"))
        self.assertEqual(
            frame.loc[0, "retrieved_at_utc"],
            pd.Timestamp(retrieved_at),
        )

    def test_wta_parser_uses_independent_gender_and_wplayer_links(self) -> None:
        """Parsea WTA sin aceptar la cabecera ni los enlaces ATP."""

        frame = parse_elo_html(WTA_HTML.encode("utf-8"), gender="F")

        self.assertEqual(len(frame), 1)
        self.assertEqual(set(frame["gender"]), {"F"})
        self.assertEqual(frame.loc[0, "player_name"], "Aryna Sabalenka")
        self.assertIn("/cgi-bin/wplayer.cgi", frame.loc[0, "player_url"])
        self.assertTrue(pd.isna(frame.loc[0, "retrieved_at_utc"]))

    def test_parser_rejects_changed_header_instead_of_guessing(self) -> None:
        """Falla si Tennis Abstract renombra una columna de la tabla real."""

        changed_html = ATP_HTML.replace(
            "Peak&nbsp;Elo",
            "Peak&nbsp;Rating",
            1,
        )

        with self.assertRaises(EloSchemaError):
            parse_elo_html(changed_html, gender="M")

    def test_parser_rejects_wrong_gender_schema(self) -> None:
        """Impide cargar una tabla WTA dentro del universo masculino."""

        with self.assertRaises(EloSchemaError):
            parse_elo_html(WTA_HTML, gender="M")


class TennisAbstractEloDownloadTest(unittest.TestCase):
    """Valida snapshots, manifiesto, condicionales y límite de red offline."""

    def _temporary_raw_dir(self) -> TemporaryDirectory[str]:
        """Crea artefactos efímeros exclusivamente dentro de TENNIS/tests."""

        return TemporaryDirectory(
            prefix=".tennis-abstract-elo-",
            dir=Path(__file__).resolve().parent,
        )

    def _success_session(self) -> FakeSession:
        """Crea las dos respuestas ``200`` con validadores distintos."""

        return FakeSession(
            [
                FakeResponse(
                    200,
                    content=ATP_HTML.encode("utf-8"),
                    headers={
                        "Content-Type": "text/html; charset=UTF-8",
                        "ETag": '"atp-v1"',
                        "Last-Modified": "Mon, 27 Jul 2026 08:00:00 GMT",
                    },
                ),
                FakeResponse(
                    200,
                    content=WTA_HTML.encode("utf-8"),
                    headers={
                        "Content-Type": "text/html; charset=UTF-8",
                        "ETag": '"wta-v1"',
                        "Last-Modified": "Mon, 27 Jul 2026 08:00:00 GMT",
                    },
                ),
            ]
        )

    def test_download_versions_raw_html_metadata_and_manifest(self) -> None:
        """Guarda dos HTML raw con hash, headers, URL y fecha recuperada."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        session = self._success_session()
        with self._temporary_raw_dir() as directory:
            raw_dir = Path(directory) / "raw"
            report = download_tennis_abstract_elo(
                raw_dir=raw_dir,
                session=session,
                clock=lambda: now,
            )

            self.assertEqual(report.get_count, 2)
            self.assertEqual(report.downloaded_count, 2)
            self.assertEqual(
                [call[0] for call in session.calls],
                [ELO_URLS["M"], ELO_URLS["F"]],
            )
            for _, headers, _, allow_redirects in session.calls:
                self.assertEqual(headers["User-Agent"], USER_AGENT)
                self.assertNotIn("If-None-Match", headers)
                self.assertFalse(allow_redirects)
            for result in report.results:
                self.assertIsNotNone(result.snapshot_path)
                self.assertIsNotNone(result.metadata_path)
                assert result.snapshot_path is not None
                assert result.metadata_path is not None
                self.assertTrue(result.snapshot_path.exists())
                metadata = json.loads(
                    result.metadata_path.read_text(encoding="utf-8")
                )
                content = result.snapshot_path.read_bytes()
                self.assertEqual(
                    metadata["sha256"],
                    hashlib.sha256(content).hexdigest(),
                )
                self.assertEqual(metadata["retrieved_at_utc"], now.isoformat())
                self.assertEqual(metadata["url"], ELO_URLS[result.gender])
                self.assertIn("response_headers", metadata)

            manifest = json.loads(
                report.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["history"]), 2)
            self.assertTrue(manifest["operator_managed_frequency"])
            self.assertEqual(manifest["failure_backoff_hours"], 24)
            loaded = load_elo_snapshot(
                report.results[0].snapshot_path,
                metadata_path=report.results[0].metadata_path,
            )
            self.assertEqual(len(loaded), 2)

    def test_force_reparses_locally_and_frequency_is_operator_managed(self) -> None:
        """Comprueba que ``force`` consume cero GET durante el cooldown."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        with self._temporary_raw_dir() as directory:
            raw_dir = Path(directory) / "raw"
            first = download_tennis_abstract_elo(
                raw_dir=raw_dir,
                session=self._success_session(),
                clock=lambda: now,
            )
            self.assertEqual(first.get_count, 2)

            conditional = FakeSession(
                [
                    FakeResponse(304, headers={}),
                    FakeResponse(304, headers={}),
                ]
            )
            forced = download_tennis_abstract_elo(
                force=True,
                raw_dir=raw_dir,
                session=conditional,
                clock=lambda: now + timedelta(hours=1),
            )

            self.assertEqual(forced.get_count, 2)
            self.assertEqual(len(conditional.calls), 2)
            self.assertEqual(
                [result.outcome for result in forced.results],
                ["not_modified", "not_modified"],
            )
            self.assertEqual(len(forced.locally_validated), 2)

    def test_conditional_get_reuses_etag_and_last_modified(self) -> None:
        """Usa ETag y Last-Modified tras vencer la ventana, sin más de dos GET."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        with self._temporary_raw_dir() as directory:
            raw_dir = Path(directory) / "raw"
            first = download_tennis_abstract_elo(
                raw_dir=raw_dir,
                session=self._success_session(),
                clock=lambda: now,
            )
            conditional = FakeSession(
                [
                    FakeResponse(304, headers={"Date": "Fri, 31 Jul 2026"}),
                    FakeResponse(304, headers={"Date": "Fri, 31 Jul 2026"}),
                ]
            )
            second = download_tennis_abstract_elo(
                raw_dir=raw_dir,
                session=conditional,
                clock=lambda: now + timedelta(hours=24, seconds=1),
            )

            self.assertEqual(second.get_count, 2)
            self.assertEqual(
                [result.outcome for result in second.results],
                ["not_modified", "not_modified"],
            )
            self.assertEqual(
                conditional.calls[0][1]["If-None-Match"],
                '"atp-v1"',
            )
            self.assertEqual(
                conditional.calls[1][1]["If-None-Match"],
                '"wta-v1"',
            )
            for _, headers, _, allow_redirects in conditional.calls:
                self.assertEqual(
                    headers["If-Modified-Since"],
                    "Mon, 27 Jul 2026 08:00:00 GMT",
                )
                self.assertFalse(allow_redirects)
            self.assertEqual(
                second.results[0].snapshot_path,
                first.results[0].snapshot_path,
            )

    def test_429_is_not_retried_even_with_force_on_same_day(self) -> None:
        """El primer 429 detiene ATP/WTA y bloquea nuevos GET durante 24 h."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        session = FakeSession(
            [
                FakeResponse(429, headers={"Retry-After": "3600"}),
                FakeResponse(
                    200,
                    content=WTA_HTML.encode("utf-8"),
                    headers={"ETag": '"wta-v1"'},
                ),
            ]
        )
        with self._temporary_raw_dir() as directory:
            raw_dir = Path(directory) / "raw"
            first = download_tennis_abstract_elo(
                raw_dir=raw_dir,
                session=session,
                clock=lambda: now,
            )
            self.assertEqual(first.get_count, 1)
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(len(first.results), 1)
            self.assertEqual(first.results[0].outcome, "http_error")

            no_network = FakeSession([])
            second = download_tennis_abstract_elo(
                force=True,
                raw_dir=raw_dir,
                session=no_network,
                clock=lambda: now + timedelta(hours=2),
            )
            self.assertEqual(second.get_count, 0)
            self.assertEqual(len(no_network.calls), 0)
            self.assertEqual(
                [result.outcome for result in second.results],
                ["throttled", "throttled"],
            )
            self.assertEqual(len(second.locally_validated), 0)

    def test_network_exception_is_counted_once_without_retry(self) -> None:
        """Cuenta una excepción de red una vez y detiene el acceso inmediatamente."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        session = FakeSession(
            [
                requests.ConnectionError("offline"),
                FakeResponse(
                    200,
                    content=WTA_HTML.encode("utf-8"),
                    headers={},
                ),
            ]
        )
        with self._temporary_raw_dir() as directory:
            report = download_tennis_abstract_elo(
                raw_dir=Path(directory) / "raw",
                session=session,
                clock=lambda: now,
            )

        self.assertEqual(report.get_count, 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].outcome, "network_error")

    def test_schema_drift_preserves_raw_without_activating_latest(self) -> None:
        """Guarda un 200 cambiado, lo rechaza y no solicita la segunda página."""

        now = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
        changed_html = ATP_HTML.replace(
            "Peak&nbsp;Elo",
            "Peak&nbsp;Rating",
            1,
        )
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    content=changed_html.encode("utf-8"),
                    headers={"ETag": '"changed-atp"'},
                ),
                FakeResponse(
                    200,
                    content=WTA_HTML.encode("utf-8"),
                    headers={"ETag": '"wta-v1"'},
                ),
            ]
        )
        with self._temporary_raw_dir() as directory:
            report = download_tennis_abstract_elo(
                raw_dir=Path(directory) / "raw",
                session=session,
                clock=lambda: now,
            )

            self.assertEqual(report.get_count, 1)
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(report.results[0].outcome, "schema_error")
            snapshot_path = report.results[0].snapshot_path
            metadata_path = report.results[0].metadata_path
            self.assertIsNotNone(snapshot_path)
            self.assertIsNotNone(metadata_path)
            assert snapshot_path is not None
            assert metadata_path is not None
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(metadata_path.exists())

            manifest = json.loads(
                report.manifest_path.read_text(encoding="utf-8")
            )
            atp_record = manifest["sources"]["M"]
            self.assertNotIn("latest_snapshot", atp_record)
            self.assertEqual(
                atp_record["last_rejected_snapshot"],
                snapshot_path.relative_to(Path(directory) / "raw").as_posix(),
            )
            self.assertNotIn("F", manifest["sources"])

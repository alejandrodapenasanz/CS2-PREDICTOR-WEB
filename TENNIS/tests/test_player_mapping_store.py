"""Tests offline del almacén auditable de identidades de jugadores.

Todos los archivos SQLite se crean en directorios temporales contenidos bajo
``TENNIS/tests``. La suite no consulta la red ni usa los datos crudos locales.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.player_mapping.store import (  # noqa: E402
    MappingRecord,
    PlayerMappingStore,
    PlayerMappingStoreError,
)


TESTS_ROOT = Path(__file__).resolve().parent


def _temporary_test_directory() -> TemporaryDirectory[str]:
    """Crea un directorio temporal aislado dentro de ``TENNIS/tests``."""

    return TemporaryDirectory(prefix="player-mapping-store-", dir=TESTS_ROOT)


class _SequenceClock:
    """Devuelve instantes predeterminados y detecta lecturas inesperadas."""

    def __init__(self, *values: datetime) -> None:
        """Conserva los instantes en el orden exacto de consumo."""

        self._values = iter(values)

    def __call__(self) -> datetime:
        """Entrega el siguiente instante inyectado."""

        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("El store leyó el reloj demasiadas veces.") from exc


def _automatic_michelsen(
    store: PlayerMappingStore,
    *,
    player_id: int = 210506,
    visible_name: str = "Michelsen A.",
) -> MappingRecord:
    """Inserta el caso real de Alex Michelsen con valores controlados."""

    return store.put_automatic(
        gender="M",
        slug="michelsen-a98bb",
        player_id=player_id,
        visible_name=visible_name,
        sackmann_player_name="Alex Michelsen",
        sackmann_ioc="USA",
        first_resolved_date=date(2026, 7, 30),
    )


class PlayerMappingStoreSchemaTests(unittest.TestCase):
    """Valida esquema, journal tradicional y auditoría append-only."""

    def test_context_initializes_delete_journal_and_append_only_audit(
        self,
    ) -> None:
        """Crea ambas tablas y bloquea UPDATE/DELETE sobre la auditoría."""

        instant = datetime(2026, 7, 30, 10, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=lambda: instant,
            ) as store:
                _automatic_michelsen(store)

            connection = sqlite3.connect(db_path)
            try:
                journal_mode = connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()[0]
                tables = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    )
                }
                self.assertEqual(journal_mode.casefold(), "delete")
                self.assertTrue(
                    {"player_mappings", "mapping_audit"}.issubset(tables)
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "append-only",
                ):
                    connection.execute(
                        """
                        UPDATE mapping_audit
                        SET reason = 'alterado'
                        WHERE audit_id = 1
                        """
                    )
                connection.rollback()
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "append-only",
                ):
                    connection.execute(
                        "DELETE FROM mapping_audit WHERE audit_id = 1"
                    )
            finally:
                connection.rollback()
                connection.close()

    def test_store_requires_context_and_reopens_persisted_records(self) -> None:
        """Rechaza uso cerrado y conserva la asociación entre aperturas."""

        instant = datetime(2026, 7, 30, 10, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            closed_store = PlayerMappingStore(
                db_path,
                clock=lambda: instant,
            )
            with self.assertRaisesRegex(
                PlayerMappingStoreError,
                "bloque 'with'",
            ):
                closed_store.get("M", "michelsen-a98bb")

            with closed_store as store:
                inserted = _automatic_michelsen(store)
            with PlayerMappingStore(db_path) as reopened:
                loaded = reopened.get("M", "michelsen-a98bb")

            self.assertEqual(loaded, inserted)


class PlayerMappingStoreAutomaticTests(unittest.TestCase):
    """Comprueba inserción automática inmutable y separación por género."""

    def test_automatic_insert_never_changes_existing_mapping(self) -> None:
        """Un segundo candidato conflictivo devuelve la primera fila intacta."""

        first_time = datetime(2026, 7, 30, 10, tzinfo=UTC)
        second_time = datetime(2026, 7, 31, 10, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=_SequenceClock(first_time, second_time),
            ) as store:
                original = _automatic_michelsen(store)
                repeated = _automatic_michelsen(
                    store,
                    player_id=999999,
                    visible_name="Otro A.",
                )

                self.assertEqual(repeated, original)
                self.assertEqual(repeated.player_id, 210506)
                self.assertEqual(repeated.visible_name, "Michelsen A.")
                self.assertEqual(repeated.updated_at_utc, first_time)

            connection = sqlite3.connect(db_path)
            try:
                audit_rows = connection.execute(
                    """
                    SELECT
                        event_type,
                        old_player_id,
                        new_player_id,
                        reason
                    FROM mapping_audit
                    """
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(
                audit_rows,
                [("automatic_insert", None, 210506, "automatic_name")],
            )

    def test_same_slug_isolated_by_gender_and_get_many_omits_missing(
        self,
    ) -> None:
        """La clave compuesta impide mezclar M/F y el lote conserva su orden."""

        instants = (
            datetime(2026, 7, 30, 10, tzinfo=UTC),
            datetime(2026, 7, 30, 11, tzinfo=UTC),
        )
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=_SequenceClock(*instants),
            ) as store:
                male = store.put_automatic(
                    gender="M",
                    slug="shared-slug",
                    player_id=42,
                    visible_name="Ejemplo M.",
                    sackmann_player_name="Masculino Ejemplo",
                    sackmann_ioc=None,
                    first_resolved_date=date(2026, 7, 30),
                )
                female = store.put_automatic(
                    gender="F",
                    slug="shared-slug",
                    player_id=42,
                    visible_name="Ejemplo F.",
                    sackmann_player_name="Femenina Ejemplo",
                    sackmann_ioc="ESP",
                    first_resolved_date=date(2026, 7, 30),
                )
                result = store.get_many(
                    (
                        ("F", "shared-slug"),
                        ("M", "missing"),
                        ("M", "shared-slug"),
                        ("F", "shared-slug"),
                    )
                )

                self.assertEqual(
                    list(result),
                    [("F", "shared-slug"), ("M", "shared-slug")],
                )
                self.assertEqual(result[("F", "shared-slug")], female)
                self.assertEqual(result[("M", "shared-slug")], male)
                self.assertEqual(
                    store.list_mappings(),
                    (female, male),
                )


class PlayerMappingStoreManualOverrideTests(unittest.TestCase):
    """Comprueba inserciones/correcciones manuales y su trazabilidad."""

    def test_manual_correction_preserves_first_date_and_audits_old_new(
        self,
    ) -> None:
        """Corrige un falso positivo sin reescribir la fecha de primera resolución."""

        first_time = datetime(2026, 7, 30, 10, tzinfo=UTC)
        correction_time = datetime(2026, 8, 1, 9, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=_SequenceClock(first_time, correction_time),
            ) as store:
                store.put_automatic(
                    gender="F",
                    slug="jones-cff84",
                    player_id=200992,
                    visible_name="Jones E.",
                    sackmann_player_name="Elizabeth Jones",
                    sackmann_ioc="GBR",
                    first_resolved_date=date(2026, 7, 30),
                )
                corrected = store.apply_manual_override(
                    gender="F",
                    slug="jones-cff84",
                    player_id=263644,
                    visible_name="Jones E.",
                    sackmann_player_name="Emerson Jones",
                    sackmann_ioc="AUS",
                    first_resolved_date=date(2026, 8, 1),
                    reason="Ficha australiana verificada manualmente.",
                )

                self.assertEqual(corrected.player_id, 263644)
                self.assertEqual(
                    corrected.resolution_method,
                    "manual_override",
                )
                self.assertEqual(
                    corrected.first_resolved_date,
                    date(2026, 7, 30),
                )
                self.assertEqual(corrected.updated_at_utc, correction_time)

            connection = sqlite3.connect(db_path)
            try:
                correction = connection.execute(
                    """
                    SELECT
                        event_type,
                        old_player_id,
                        new_player_id,
                        old_sackmann_ioc,
                        new_sackmann_ioc,
                        old_first_resolved_date,
                        new_first_resolved_date,
                        reason
                    FROM mapping_audit
                    WHERE event_type = 'manual_correction'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                correction,
                (
                    "manual_correction",
                    200992,
                    263644,
                    "GBR",
                    "AUS",
                    "2026-07-30",
                    "2026-07-30",
                    "Ficha australiana verificada manualmente.",
                ),
            )

    def test_manual_intervention_is_audited_even_when_id_matches(self) -> None:
        """Registra la decisión humana aunque conserve el mismo player_id."""

        first_time = datetime(2026, 7, 30, 10, tzinfo=UTC)
        override_time = datetime(2026, 7, 30, 12, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=_SequenceClock(first_time, override_time),
            ) as store:
                _automatic_michelsen(store)
                overridden = store.apply_manual_override(
                    gender="M",
                    slug="michelsen-a98bb",
                    player_id=210506,
                    visible_name="Michelsen A.",
                    sackmann_player_name="Alex Michelsen",
                    sackmann_ioc="USA",
                    first_resolved_date=date(2026, 7, 30),
                    reason="Identidad confirmada por revisión humana.",
                )

                self.assertEqual(overridden.player_id, 210506)
                self.assertEqual(
                    overridden.resolution_method,
                    "manual_override",
                )
                self.assertEqual(overridden.updated_at_utc, override_time)

            connection = sqlite3.connect(db_path)
            try:
                events = connection.execute(
                    """
                    SELECT event_type, old_player_id, new_player_id
                    FROM mapping_audit
                    ORDER BY audit_id
                    """
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(
                events,
                [
                    ("automatic_insert", None, 210506),
                    ("manual_correction", 210506, 210506),
                ],
            )

    def test_manual_insert_and_blank_reason_validation(self) -> None:
        """Permite un alta manual nueva, pero rechaza motivos vacíos."""

        first_time = datetime(2026, 7, 30, 10, tzinfo=UTC)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=lambda: first_time,
            ) as store:
                with self.assertRaisesRegex(
                    PlayerMappingStoreError,
                    "reason",
                ):
                    store.apply_manual_override(
                        gender="M",
                        slug="nakashima-68876",
                        player_id=206909,
                        visible_name="Nakashima B.",
                        sackmann_player_name="Brandon Nakashima",
                        sackmann_ioc="USA",
                        first_resolved_date=date(2026, 7, 30),
                        reason="   ",
                    )

                inserted = store.apply_manual_override(
                    gender="M",
                    slug="nakashima-68876",
                    player_id=206909,
                    visible_name="Nakashima B.",
                    sackmann_player_name="Brandon Nakashima",
                    sackmann_ioc="USA",
                    first_resolved_date=date(2026, 7, 30),
                    reason="Colisión homónima resuelta por revisión.",
                )

                self.assertEqual(
                    inserted.resolution_method,
                    "manual_override",
                )

            connection = sqlite3.connect(db_path)
            try:
                event = connection.execute(
                    """
                    SELECT event_type, old_player_id, new_player_id
                    FROM mapping_audit
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(event, ("manual_insert", None, 206909))


class PlayerMappingStoreValidationTests(unittest.TestCase):
    """Ejercita validaciones estrictas antes de cualquier escritura."""

    def test_invalid_arguments_and_naive_clock_leave_database_empty(self) -> None:
        """No normaliza género/IOC ni acepta timestamps sin zona."""

        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "mapping.sqlite3"
            with PlayerMappingStore(
                db_path,
                clock=lambda: datetime(2026, 7, 30, 10),
            ) as store:
                with self.assertRaisesRegex(
                    PlayerMappingStoreError,
                    "gender",
                ):
                    store.get("m", "michelsen-a98bb")
                with self.assertRaisesRegex(
                    PlayerMappingStoreError,
                    "sackmann_ioc",
                ):
                    store.put_automatic(
                        gender="M",
                        slug="michelsen-a98bb",
                        player_id=210506,
                        visible_name="Michelsen A.",
                        sackmann_player_name="Alex Michelsen",
                        sackmann_ioc="usa",
                        first_resolved_date=date(2026, 7, 30),
                    )
                with self.assertRaisesRegex(
                    PlayerMappingStoreError,
                    "clock",
                ):
                    store.put_automatic(
                        gender="M",
                        slug="michelsen-a98bb",
                        player_id=210506,
                        visible_name="Michelsen A.",
                        sackmann_player_name="Alex Michelsen",
                        sackmann_ioc="USA",
                        first_resolved_date=date(2026, 7, 30),
                    )
                self.assertEqual(store.list_mappings(), ())


if __name__ == "__main__":
    unittest.main()

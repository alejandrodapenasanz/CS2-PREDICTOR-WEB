"""Tests offline del almacén SQLite y la API temporal Elo."""

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

from src.elo.service import EloQuery, get_elo, get_elos  # noqa: E402
from src.elo.store import (  # noqa: E402
    EloRunNotAvailableError,
    EloStore,
    EloValidationError,
    RatingHistoryRow,
)


SOURCE_COMMIT = "a" * 40
TESTS_ROOT = Path(__file__).resolve().parent


def _temporary_test_directory() -> TemporaryDirectory[str]:
    """Crea un directorio temporal aislado dentro de ``TENNIS/tests``."""

    return TemporaryDirectory(prefix="elo-store-", dir=TESTS_ROOT)


def _create_run(
    store: EloStore,
    *,
    run_id: str,
    gender: str,
    initial_rating: float = 1500.0,
    surface_weight: float = 0.5,
) -> None:
    """Crea un run reproducible en estado building para un test."""

    store.create_run(
        run_id=run_id,
        gender=gender,
        source_commit=SOURCE_COMMIT,
        algorithm_version="test-v1",
        input_fingerprint=f"fingerprint-{run_id}",
        initial_rating=initial_rating,
        surface_weight=surface_weight,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def _rating(
    *,
    gender: str,
    player_id: int,
    state_date: date,
    general_elo: float,
    hard_elo: float,
    general_matches: int = 1,
    hard_matches: int = 1,
) -> RatingHistoryRow:
    """Construye un estado ancho completo con valores de prueba explícitos."""

    return RatingHistoryRow(
        gender=gender,
        player_id=player_id,
        state_date=state_date,
        general_elo=general_elo,
        hard_elo=hard_elo,
        clay_elo=1490.0,
        grass_elo=1480.0,
        carpet_elo=1470.0,
        general_matches=general_matches,
        hard_matches=hard_matches,
        clay_matches=0,
        grass_matches=0,
        carpet_matches=0,
    )


class EloStoreSchemaTest(unittest.TestCase):
    """Comprueba esquema, journal tradicional y defensas de activación."""

    def test_schema_uses_delete_journal_indices_and_activation_trigger(
        self,
    ) -> None:
        """Crea las cuatro tablas e impide activar manualmente un building run."""

        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()
            _create_run(store, run_id="building-m", gender="M")

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
                indices = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'index'
                        """
                    )
                }
                self.assertEqual(journal_mode.casefold(), "delete")
                self.assertTrue(
                    {
                        "runs",
                        "date_blocks",
                        "rating_history",
                        "active_runs",
                    }.issubset(tables)
                )
                self.assertIn("idx_rating_history_lookup", indices)
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO active_runs (
                            gender,
                            run_id,
                            activated_at_utc
                        )
                        VALUES ('M', 'building-m', '2026-07-30T00:00:00+00:00')
                        """
                    )
            finally:
                connection.rollback()
                connection.close()

            with self.assertRaises(EloRunNotAvailableError):
                get_elo(
                    gender="M",
                    player_id=1,
                    surface="Hard",
                    as_of_date=date(2024, 1, 1),
                    db_path=db_path,
                    run_id="building-m",
                )


class EloTemporalServiceTest(unittest.TestCase):
    """Comprueba cold start, mezcla de superficie y corte temporal estricto."""

    def test_get_elo_excludes_equal_date_and_combines_later_state(
        self,
    ) -> None:
        """El bloque D no es visible en D, pero sí a partir de una fecha posterior."""

        block_date = date(2024, 1, 10)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()
            _create_run(store, run_id="men-v1", gender="M")
            store.write_date_block(
                run_id="men-v1",
                block_date=block_date,
                input_hash="block-hash-1",
                match_count=1,
                ratings=(
                    _rating(
                        gender="M",
                        player_id=101,
                        state_date=block_date,
                        general_elo=1520.0,
                        hard_elo=1540.0,
                    ),
                ),
            )
            store.activate_run(
                run_id="men-v1",
                completed_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
            )

            equal_date = get_elo(
                gender="M",
                player_id=101,
                surface="Hard",
                as_of_date=block_date,
                db_path=db_path,
            )
            self.assertTrue(equal_date.is_cold_start)
            self.assertIsNone(equal_date.state_date)
            self.assertEqual(equal_date.general_elo, 1500.0)
            self.assertEqual(equal_date.surface_elo_raw, 1500.0)
            self.assertEqual(equal_date.combined_elo, 1500.0)

            later = get_elo(
                gender="M",
                player_id=101,
                surface="hArD",
                as_of_date=date(2024, 1, 11),
                db_path=db_path,
            )
            self.assertFalse(later.is_cold_start)
            self.assertEqual(later.surface, "Hard")
            self.assertEqual(later.state_date, block_date)
            self.assertEqual(later.general_elo, 1520.0)
            self.assertEqual(later.surface_elo_raw, 1540.0)
            self.assertEqual(later.combined_elo, 1530.0)
            self.assertEqual(later.general_matches, 1)
            self.assertEqual(later.surface_matches, 1)
            self.assertEqual(later.source_commit, SOURCE_COMMIT)

            general = get_elo(
                gender="M",
                player_id=101,
                surface=None,
                as_of_date=date(2024, 1, 11),
                db_path=db_path,
            )
            self.assertIsNone(general.surface)
            self.assertIsNone(general.surface_elo_raw)
            self.assertEqual(general.combined_elo, 1520.0)
            self.assertEqual(general.surface_matches, 0)

    def test_cold_start_uses_run_metadata_and_validation_is_strict(
        self,
    ) -> None:
        """Usa parámetros persistidos y rechaza fecha o superficie ambiguas."""

        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()
            _create_run(
                store,
                run_id="women-custom",
                gender="F",
                initial_rating=1400.0,
                surface_weight=0.25,
            )
            store.activate_run(run_id="women-custom")

            cold_start = get_elo(
                gender="F",
                player_id=999,
                surface="Clay",
                as_of_date=date(2024, 2, 1),
                db_path=db_path,
            )
            self.assertEqual(cold_start.general_elo, 1400.0)
            self.assertEqual(cold_start.surface_elo_raw, 1400.0)
            self.assertEqual(cold_start.combined_elo, 1400.0)
            self.assertTrue(cold_start.is_cold_start)

            with self.assertRaises(EloValidationError):
                get_elo(
                    gender="F",
                    player_id=999,
                    surface="Indoor",
                    as_of_date=date(2024, 2, 1),
                    db_path=db_path,
                )
            with self.assertRaises(EloValidationError):
                get_elo(
                    gender="F",
                    player_id=999,
                    surface="Clay",
                    as_of_date=datetime(2024, 2, 1, tzinfo=UTC),
                    db_path=db_path,
                )


class EloActivationAndBatchTest(unittest.TestCase):
    """Comprueba activación atómica, aislamiento de género y consulta por lotes."""

    def test_building_run_cannot_replace_active_until_activation(
        self,
    ) -> None:
        """Mantiene el run anterior visible mientras el siguiente está building."""

        block_date = date(2024, 3, 1)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()

            _create_run(store, run_id="men-old", gender="M")
            store.write_date_block(
                run_id="men-old",
                block_date=block_date,
                input_hash="old-block",
                match_count=1,
                ratings=(
                    _rating(
                        gender="M",
                        player_id=7,
                        state_date=block_date,
                        general_elo=1510.0,
                        hard_elo=1510.0,
                    ),
                ),
            )
            store.activate_run(run_id="men-old")

            _create_run(store, run_id="men-new", gender="M")
            store.write_date_block(
                run_id="men-new",
                block_date=block_date,
                input_hash="new-block",
                match_count=1,
                ratings=(
                    _rating(
                        gender="M",
                        player_id=7,
                        state_date=block_date,
                        general_elo=1600.0,
                        hard_elo=1600.0,
                    ),
                ),
            )

            while_building = get_elo(
                gender="M",
                player_id=7,
                surface="Hard",
                as_of_date=date(2024, 3, 2),
                db_path=db_path,
            )
            self.assertEqual(while_building.run_id, "men-old")
            self.assertEqual(while_building.general_elo, 1510.0)

            with self.assertRaises(EloRunNotAvailableError):
                get_elo(
                    gender="M",
                    player_id=7,
                    surface="Hard",
                    as_of_date=date(2024, 3, 2),
                    db_path=db_path,
                    run_id="men-new",
                )

            store.activate_run(run_id="men-new")
            after_activation = get_elo(
                gender="M",
                player_id=7,
                surface="Hard",
                as_of_date=date(2024, 3, 2),
                db_path=db_path,
            )
            self.assertEqual(after_activation.run_id, "men-new")
            self.assertEqual(after_activation.general_elo, 1600.0)

    def test_vector_query_preserves_order_and_separates_gender(
        self,
    ) -> None:
        """Resuelve el mismo player_id en dos universos sin mezclarlos."""

        block_date = date(2024, 4, 1)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()
            for gender, run_id, rating in (
                ("M", "men", 1550.0),
                ("F", "women", 1450.0),
            ):
                _create_run(store, run_id=run_id, gender=gender)
                store.write_date_block(
                    run_id=run_id,
                    block_date=block_date,
                    input_hash=f"block-{gender}",
                    match_count=1,
                    ratings=(
                        _rating(
                            gender=gender,
                            player_id=42,
                            state_date=block_date,
                            general_elo=rating,
                            hard_elo=rating,
                        ),
                    ),
                )
                store.activate_run(run_id=run_id)

            snapshots = get_elos(
                queries=(
                    EloQuery("F", 42, "grass", date(2024, 4, 2)),
                    EloQuery("M", 42, "hard", date(2024, 4, 2)),
                    EloQuery("M", 404, None, date(2024, 4, 2)),
                ),
                db_path=db_path,
            )
            self.assertEqual(
                [snapshot.gender for snapshot in snapshots],
                ["F", "M", "M"],
            )
            self.assertEqual(snapshots[0].general_elo, 1450.0)
            self.assertEqual(snapshots[0].surface, "Grass")
            self.assertEqual(snapshots[1].general_elo, 1550.0)
            self.assertTrue(snapshots[2].is_cold_start)

    def test_non_increasing_block_is_rolled_back_as_a_unit(self) -> None:
        """Fechas anteriores o iguales no dejan estados ni checkpoints parciales."""

        block_date = date(2024, 5, 2)
        with _temporary_test_directory() as temporary_directory:
            db_path = Path(temporary_directory) / "elo.sqlite3"
            store = EloStore(db_path)
            store.initialize()
            _create_run(store, run_id="transaction", gender="M")
            store.write_date_block(
                run_id="transaction",
                block_date=block_date,
                input_hash="first",
                match_count=1,
                ratings=(
                    _rating(
                        gender="M",
                        player_id=1,
                        state_date=block_date,
                        general_elo=1510.0,
                        hard_elo=1510.0,
                    ),
                ),
            )

            for rejected_date, player_id in (
                (date(2024, 5, 1), 2),
                (block_date, 3),
            ):
                with self.subTest(rejected_date=rejected_date):
                    with self.assertRaises(EloValidationError):
                        store.write_date_block(
                            run_id="transaction",
                            block_date=rejected_date,
                            input_hash=f"rejected-{rejected_date}",
                            match_count=1,
                            ratings=(
                                _rating(
                                    gender="M",
                                    player_id=player_id,
                                    state_date=rejected_date,
                                    general_elo=1520.0,
                                    hard_elo=1520.0,
                                ),
                            ),
                        ),

            connection = sqlite3.connect(db_path)
            try:
                block_count = connection.execute(
                    "SELECT COUNT(*) FROM date_blocks"
                ).fetchone()[0]
                history_count = connection.execute(
                    "SELECT COUNT(*) FROM rating_history"
                ).fetchone()[0]
                player_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT player_id FROM rating_history"
                    )
                ]
            finally:
                connection.close()
            self.assertEqual(block_count, 1)
            self.assertEqual(history_count, 1)
            self.assertEqual(player_ids, [1])


if __name__ == "__main__":
    unittest.main()

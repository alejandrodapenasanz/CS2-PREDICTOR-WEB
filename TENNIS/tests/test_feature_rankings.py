"""Tests offline del índice causal de rankings de la fase 6."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.rankings import (  # noqa: E402
    RankingIndex,
    RankingSchemaError,
    RankingSnapshot,
)


TESTS_ROOT = Path(__file__).resolve().parent
ATP_COLUMNS = ("ranking_date", "rank", "player", "points")
WTA_COLUMNS = ("ranking_date", "rank", "player", "points", "tours")


def _temporary_test_directory() -> TemporaryDirectory[str]:
    """Crea un directorio temporal aislado dentro de ``TENNIS/tests``."""

    return TemporaryDirectory(prefix="feature-rankings-", dir=TESTS_ROOT)


def _write_ranking_csv(
    raw_dir: Path,
    *,
    gender: str,
    rows: tuple[tuple[object, ...], ...],
    columns: tuple[str, ...] | None = None,
    suffix: str = "fixture",
) -> Path:
    """Escribe un CSV mínimo con el mismo contrato que la fuente real."""

    if gender == "M":
        directory_name = "atp"
        prefix = "atp"
        selected_columns = columns or ATP_COLUMNS
    else:
        directory_name = "wta"
        prefix = "wta"
        selected_columns = columns or WTA_COLUMNS
    directory = raw_dir / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_rankings_{suffix}.csv"
    lines = [",".join(selected_columns)]
    lines.extend(",".join("" if value is None else str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class RankingCausalityTest(unittest.TestCase):
    """Comprueba el corte temporal estricto y la estabilidad ante el futuro."""

    def test_equal_date_is_excluded_and_get_many_preserves_order(self) -> None:
        """Un ranking de D solo es visible después de D, nunca durante D."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="M",
                rows=(
                    (20240101, 10, 101, 1000),
                    (20240110, 5, 101, 1500),
                    (20240101, 20, 202, None),
                ),
            )
            index = RankingIndex.from_raw("M", raw_dir=raw_dir)

            on_first_publication = index.get(101, date(2024, 1, 1))
            self.assertTrue(on_first_publication.is_missing)
            self.assertIsNone(on_first_publication.ranking_date)

            on_second_publication = index.get(101, date(2024, 1, 10))
            self.assertEqual(on_second_publication.ranking_date, date(2024, 1, 1))
            self.assertEqual(on_second_publication.rank, 10)
            self.assertEqual(on_second_publication.points, 1000)
            self.assertEqual(on_second_publication.ranking_age_days, 9)
            self.assertEqual(on_second_publication.conflict_dates_skipped, 0)

            next_day = index.get(101, "2024-01-11")
            self.assertEqual(next_day.ranking_date, date(2024, 1, 10))
            self.assertEqual(next_day.rank, 5)

            batch = index.get_many((202, 101, 202), date(2024, 1, 2))
            self.assertEqual([item.player_id for item in batch], [202, 101, 202])
            self.assertEqual([item.rank for item in batch], [20, 10, 20])
            self.assertIsNone(batch[0].points)

    def test_appending_future_rows_does_not_change_past_snapshot(self) -> None:
        """Añadir rankings posteriores conserva una consulta histórica idéntica."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            path = _write_ranking_csv(
                raw_dir,
                gender="M",
                rows=((20240101, 10, 101, 1000),),
            )
            before = RankingIndex.from_raw("M", raw_dir=raw_dir).get(
                101,
                date(2024, 2, 1),
            )
            with path.open("a", encoding="utf-8", newline="") as stream:
                stream.write("20240301,2,101,2500\n")
            after = RankingIndex.from_raw("M", raw_dir=raw_dir).get(
                101,
                date(2024, 2, 1),
            )
            self.assertEqual(before, after)


class RankingIsolationAndMissingTest(unittest.TestCase):
    """Comprueba separación de género, nulos honestos e inmutabilidad."""

    def test_same_player_id_is_isolated_between_gender_indices(self) -> None:
        """El mismo entero puede tener snapshots distintos en ATP y WTA."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="M",
                rows=((20240101, 7, 42, 1700),),
            )
            _write_ranking_csv(
                raw_dir,
                gender="F",
                rows=((20240101, 70, 42, 700, 12),),
            )

            men = RankingIndex.from_raw("m", raw_dir=raw_dir)
            women = RankingIndex.from_raw("F", raw_dir=raw_dir)
            self.assertEqual(men.get(42, date(2024, 1, 2)).rank, 7)
            self.assertEqual(women.get(42, date(2024, 1, 2)).rank, 70)
            self.assertEqual(men.gender, "M")
            self.assertEqual(women.gender, "F")

    def test_unknown_player_returns_frozen_null_snapshot(self) -> None:
        """Un jugador ausente produce nulos explícitos sin inventar ranking."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="F",
                rows=((20240101, 1, 100, 9000, 20),),
            )
            missing = RankingIndex.from_raw("F", raw_dir=raw_dir).get(
                999,
                date(2024, 1, 2),
            )
            self.assertEqual(
                missing,
                RankingSnapshot(
                    gender="F",
                    player_id=999,
                    as_of_date=date(2024, 1, 2),
                    ranking_date=None,
                    rank=None,
                    points=None,
                    is_missing=True,
                    ranking_age_days=None,
                    conflict_dates_skipped=0,
                ),
            )
            with self.assertRaises(FrozenInstanceError):
                missing.rank = 1  # type: ignore[misc]


class RankingSourceValidationTest(unittest.TestCase):
    """Comprueba cabeceras, tipos, duplicados exactos y conflictos."""

    def test_exact_duplicates_are_collapsed_and_counted(self) -> None:
        """Dos filas idénticas generan un único snapshot auditable."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="F",
                rows=(
                    (20240101, 1, 100, 9000, 20),
                    (20240101, 1, 100, 9000, 20),
                ),
            )
            index = RankingIndex.from_raw("F", raw_dir=raw_dir)
            self.assertEqual(index.source_row_count, 2)
            self.assertEqual(index.row_count, 1)
            self.assertEqual(index.exact_duplicate_count, 1)
            self.assertEqual(index.player_count, 1)
            self.assertEqual(index.min_ranking_date, date(2024, 1, 1))
            self.assertEqual(index.max_ranking_date, date(2024, 1, 1))

    def test_conflict_is_quarantined_and_query_falls_back_causally(self) -> None:
        """Una clave ambigua se salta completa y señala el ranking envejecido."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="M",
                rows=(
                    (20240101, 10, 101, 1000),
                    (20240110, 5, 101, 1500),
                    (20240110, 6, 101, 1400),
                    (20240120, 3, 101, 2000),
                ),
            )
            index = RankingIndex.from_raw("M", raw_dir=raw_dir)

            equal_conflict_date = index.get(101, date(2024, 1, 10))
            self.assertEqual(equal_conflict_date.rank, 10)
            self.assertEqual(equal_conflict_date.conflict_dates_skipped, 0)

            after_conflict = index.get(101, date(2024, 1, 11))
            self.assertEqual(after_conflict.rank, 10)
            self.assertEqual(after_conflict.ranking_age_days, 10)
            self.assertEqual(after_conflict.conflict_dates_skipped, 1)

            after_recovery = index.get(101, date(2024, 1, 21))
            self.assertEqual(after_recovery.rank, 3)
            self.assertEqual(after_recovery.conflict_dates_skipped, 0)
            self.assertEqual(index.conflict_key_count, 1)
            self.assertEqual(index.conflict_observation_count, 2)

    def test_wta_tours_conflict_is_missing_and_inventory_is_atomic(self) -> None:
        """La cuarentena cubre las 226 claves WTA auditadas y puede exportarse."""

        with _temporary_test_directory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            _write_ranking_csv(
                raw_dir,
                gender="F",
                rows=(
                    (20240101, 10, 101, 1000, 8),
                    (20240101, 10, 101, 1000, 9),
                ),
            )
            index = RankingIndex.from_raw("F", raw_dir=raw_dir)
            missing = index.get(101, date(2024, 1, 2))
            self.assertTrue(missing.is_missing)
            self.assertEqual(missing.conflict_dates_skipped, 1)

            inventory_path = raw_dir / "audit" / "conflicts.csv"
            returned_path = index.write_conflict_inventory(inventory_path)
            self.assertEqual(returned_path, inventory_path)
            with inventory_path.open(
                encoding="utf-8",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                set(rows[0]),
                {
                    "gender",
                    "player_id",
                    "ranking_date",
                    "source_file",
                    "source_row",
                    "rank",
                    "points",
                    "tours",
                },
            )
            self.assertEqual({row["tours"] for row in rows}, {"8", "9"})
            self.assertEqual(
                list(inventory_path.parent.glob("*.tmp")),
                [],
            )

    def test_header_dates_and_integer_types_are_strict(self) -> None:
        """Rechaza tanto una cabecera distinta como fecha o entero inválidos."""

        cases = (
            (
                "header",
                ("ranking_date", "rank", "player", "ranking_points"),
                ((20240101, 1, 100, 9000),),
            ),
            (
                "date",
                ATP_COLUMNS,
                ((20240230, 1, 100, 9000),),
            ),
            (
                "integer",
                ATP_COLUMNS,
                ((20240101, "1.5", 100, 9000),),
            ),
        )
        for suffix, columns, rows in cases:
            with self.subTest(suffix=suffix):
                with _temporary_test_directory() as temporary_directory:
                    raw_dir = Path(temporary_directory)
                    _write_ranking_csv(
                        raw_dir,
                        gender="M",
                        rows=rows,
                        columns=columns,
                        suffix=suffix,
                    )
                    with self.assertRaises(RankingSchemaError):
                        RankingIndex.from_raw("M", raw_dir=raw_dir)


if __name__ == "__main__":
    unittest.main()

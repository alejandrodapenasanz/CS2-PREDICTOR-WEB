"""Tests offline de overrides y cola de revisión de identidades.

Cada prueba usa un directorio temporal creado dentro de ``TENNIS/tests``. No
se accede a la red ni al archivo real ``TENNIS/data/overrides.csv``.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest

from src.player_mapping.review import (
    OVERRIDE_COLUMNS,
    UNRESOLVED_COLUMNS,
    PlayerMappingReviewLockError,
    UnresolvedObservation,
    load_overrides,
    update_unresolved_queue,
)
from src.player_mapping.types import (
    PlayerMappingSchemaError,
    PlayerMappingSourceError,
    PlayerMappingValidationError,
)


class OverrideLoaderTests(unittest.TestCase):
    """Comprueba el contrato estricto de la tabla manual."""

    def setUp(self) -> None:
        """Crea un directorio aislado dentro del proyecto."""

        self._temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix=".mapping-review-",
        )
        self.directory = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        """Retira los artefactos temporales de la prueba."""

        self._temporary_directory.cleanup()

    def test_loads_valid_overrides_indexed_by_gender_and_slug(self) -> None:
        """Carga IDs iguales o distintos sin mezclar universos de género."""

        path = self.directory / "overrides.csv"
        path.write_text(
            "gender,slug,player_id,reason\n"
            "M,nakashima-68876,206909,Perfil verificado\n"
            "F,uchijima,220416,Perfil verificado\n",
            encoding="utf-8",
        )

        overrides = load_overrides(path)

        self.assertEqual(
            set(overrides),
            {("M", "nakashima-68876"), ("F", "uchijima")},
        )
        self.assertEqual(
            overrides[("M", "nakashima-68876")].player_id,
            206909,
        )
        self.assertEqual(
            overrides[("F", "uchijima")].reason,
            "Perfil verificado",
        )

    def test_accepts_header_only_override_file(self) -> None:
        """Interpreta una tabla manual aún vacía como cero overrides."""

        path = self.directory / "overrides.csv"
        path.write_text(
            ",".join(OVERRIDE_COLUMNS) + "\n",
            encoding="utf-8",
        )

        self.assertEqual(load_overrides(path), {})

    def test_rejects_missing_or_non_file_override_path(self) -> None:
        """Distingue una fuente ausente de un CSV con contenido inválido."""

        with self.assertRaises(PlayerMappingSourceError):
            load_overrides(self.directory / "missing.csv")
        with self.assertRaises(PlayerMappingSourceError):
            load_overrides(self.directory)

    def test_rejects_wrong_header_and_extra_columns(self) -> None:
        """No tolera columnas omitidas, renombradas ni adicionales."""

        wrong_header = self.directory / "wrong.csv"
        wrong_header.write_text(
            "gender,slug,player_id\nM,test,1\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            PlayerMappingSchemaError,
            "Cabecera inválida",
        ):
            load_overrides(wrong_header)

        extra_value = self.directory / "extra.csv"
        extra_value.write_text(
            "gender,slug,player_id,reason\n"
            "M,test,1,motivo,sobrante\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            PlayerMappingSchemaError,
            "columnas adicionales",
        ):
            load_overrides(extra_value)

    def test_rejects_duplicate_or_invalid_override_rows(self) -> None:
        """Impide decisiones ambiguas y valores corregidos silenciosamente."""

        duplicate = self.directory / "duplicate.csv"
        duplicate.write_text(
            "gender,slug,player_id,reason\n"
            "M,test,1,primero\n"
            "M,test,2,segundo\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            PlayerMappingSchemaError,
            "duplicado",
        ):
            load_overrides(duplicate)

        invalid = self.directory / "invalid.csv"
        invalid.write_text(
            "gender,slug,player_id,reason\n"
            "X,test,0,motivo\n",
            encoding="utf-8",
        )
        with self.assertRaises(PlayerMappingSchemaError):
            load_overrides(invalid)


class UnresolvedQueueTests(unittest.TestCase):
    """Comprueba agregación, deduplicación, retirada y escritura atómica."""

    def setUp(self) -> None:
        """Crea un destino temporal aislado dentro de ``TENNIS/tests``."""

        self._temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix=".mapping-review-",
        )
        self.directory = Path(self._temporary_directory.name)
        self.path = self.directory / "unresolved.csv"

    def tearDown(self) -> None:
        """Elimina CSV y locks temporales al terminar cada prueba."""

        self._temporary_directory.cleanup()

    def _observation(
        self,
        *,
        slug: str | None = "nakashima-68876",
        observed_date: date = date(2026, 7, 30),
        reason: str = "ambiguous_candidates",
        candidate_player_ids: tuple[int, ...] = (206909, 210416),
        tour_level: str = "ATP",
        tournament: str = "Washington",
    ) -> UnresolvedObservation:
        """Construye una observación válida reutilizable por las pruebas."""

        return UnresolvedObservation(
            gender="M",
            slug=slug,
            visible_name="Nakashima B.",
            normalized_last_name="nakashima",
            first_initial="b",
            reason=reason,
            candidate_player_ids=candidate_player_ids,
            observed_date=observed_date,
            tour_level=tour_level,
            tournament=tournament,
        )

    def _read_rows(self) -> list[dict[str, str]]:
        """Lee la cola temporal conservando los nombres de columnas."""

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_observation_derives_candidate_count_and_fallback_key(self) -> None:
        """Deriva el recuento y usa nombre+inicial cuando falta el slug."""

        observation = self._observation(slug=None)

        self.assertEqual(observation.candidate_count, 2)
        self.assertEqual(observation.review_key, ("M", "nakashima b"))

    def test_rejects_invalid_observation_without_writing(self) -> None:
        """Falla antes de adquirir el lock ante datos incompletos."""

        with self.assertRaises(PlayerMappingValidationError):
            UnresolvedObservation(
                gender="M",
                slug="slug with spaces",
                visible_name="Jugador J.",
                normalized_last_name="jugador",
                first_initial="j",
                reason="unresolved",
                candidate_player_ids=(),
                observed_date=date(2026, 7, 30),
                tour_level="ITF",
                tournament="Futures 2026",
            )
        self.assertFalse(self.path.exists())

    def test_creates_canonical_atomic_queue(self) -> None:
        """Escribe todos los campos auditables y no deja restos temporales."""

        pending_count = update_unresolved_queue(
            self.path,
            [self._observation()],
            resolved_keys=(),
        )

        self.assertEqual(pending_count, 1)
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), UNRESOLVED_COLUMNS)
            rows = list(reader)
        self.assertEqual(
            rows,
            [
                {
                    "gender": "M",
                    "slug": "nakashima-68876",
                    "visible_name": "Nakashima B.",
                    "normalized_last_name": "nakashima",
                    "first_initial": "b",
                    "reason": "ambiguous_candidates",
                    "candidate_count": "2",
                    "candidate_player_ids": "206909|210416",
                    "first_seen_date": "2026-07-30",
                    "last_seen_date": "2026-07-30",
                    "observed_dates": "2026-07-30",
                    "occurrences": "1",
                    "last_tour_level": "ATP",
                    "last_tournament": "Washington",
                }
            ],
        )
        leftovers = [
            entry.name
            for entry in self.directory.iterdir()
            if entry.suffix in {".tmp", ".lock"}
        ]
        self.assertEqual(leftovers, [])

    def test_same_date_is_idempotent_and_new_date_is_aggregated(self) -> None:
        """Cuenta fechas únicas y actualiza el diagnóstico más reciente."""

        first = self._observation()
        update_unresolved_queue(self.path, [first, first], resolved_keys=())
        update_unresolved_queue(self.path, [first], resolved_keys=())

        same_date_row = self._read_rows()[0]
        self.assertEqual(same_date_row["occurrences"], "1")
        self.assertEqual(same_date_row["observed_dates"], "2026-07-30")

        later = self._observation(
            observed_date=date(2026, 7, 31),
            reason="no_active_candidate",
            candidate_player_ids=(),
            tour_level="ITF",
            tournament="Futures 2026",
        )
        update_unresolved_queue(self.path, [later], resolved_keys=())

        row = self._read_rows()[0]
        self.assertEqual(row["first_seen_date"], "2026-07-30")
        self.assertEqual(row["last_seen_date"], "2026-07-31")
        self.assertEqual(
            row["observed_dates"],
            "2026-07-30|2026-07-31",
        )
        self.assertEqual(row["occurrences"], "2")
        self.assertEqual(row["reason"], "no_active_candidate")
        self.assertEqual(row["candidate_count"], "0")
        self.assertEqual(row["candidate_player_ids"], "")
        self.assertEqual(row["last_tour_level"], "ITF")
        self.assertEqual(row["last_tournament"], "Futures 2026")

    def test_preserves_old_pending_and_removes_resolved_keys(self) -> None:
        """Conserva ausentes del lote actual y retira ambas clases de clave."""

        slug_observation = self._observation()
        no_slug_observation = self._observation(slug=None)
        other = UnresolvedObservation(
            gender="F",
            slug="uchijima",
            visible_name="Uchijima M.",
            normalized_last_name="uchijima",
            first_initial="m",
            reason="ambiguous_candidates",
            candidate_player_ids=(220416, 264134),
            observed_date=date(2026, 7, 30),
            tour_level="WTA",
            tournament="Vancouver WTA",
        )
        update_unresolved_queue(
            self.path,
            [slug_observation, no_slug_observation, other],
            resolved_keys=(),
        )

        remaining = update_unresolved_queue(
            self.path,
            observations=(),
            resolved_keys={
                ("M", "nakashima-68876"),
                ("M", "nakashima b"),
            },
        )

        self.assertEqual(remaining, 1)
        rows = self._read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gender"], "F")
        self.assertEqual(rows[0]["slug"], "uchijima")

    def test_resolved_key_wins_over_same_batch_observation(self) -> None:
        """No vuelve a insertar una identidad resuelta en el mismo ciclo."""

        remaining = update_unresolved_queue(
            self.path,
            observations=[self._observation()],
            resolved_keys={("M", "nakashima-68876")},
        )

        self.assertEqual(remaining, 0)
        self.assertEqual(self._read_rows(), [])

    def test_rejects_corrupt_existing_queue_without_overwriting_it(self) -> None:
        """Conserva bytes corruptos para revisión en vez de tapar el fallo."""

        original = b"gender,slug\nM,test\n"
        self.path.write_bytes(original)

        with self.assertRaisesRegex(
            PlayerMappingSchemaError,
            "Cabecera inválida",
        ):
            update_unresolved_queue(
                self.path,
                observations=[self._observation()],
                resolved_keys=(),
            )

        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(
            self.path.with_name(f"{self.path.name}.lock").exists()
        )

    def test_exclusive_lock_fails_clearly_and_preserves_queue(self) -> None:
        """No fuerza ni borra un lock que puede pertenecer a otro proceso."""

        update_unresolved_queue(
            self.path,
            observations=[self._observation()],
            resolved_keys=(),
        )
        original = self.path.read_bytes()
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        lock_path.write_text("pid=999999\n", encoding="ascii")

        with self.assertRaisesRegex(
            PlayerMappingReviewLockError,
            "bloqueada",
        ):
            update_unresolved_queue(
                self.path,
                observations=(),
                resolved_keys=(),
            )

        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(lock_path.exists())

    def test_rejects_malformed_resolved_keys_before_writing(self) -> None:
        """Exige pares válidos y no crea una cola ante una clave ambigua."""

        with self.assertRaises(PlayerMappingValidationError):
            update_unresolved_queue(
                self.path,
                observations=(),
                resolved_keys=[("M",)],  # type: ignore[list-item]
            )
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()

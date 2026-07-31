"""Tests del inventario auditado y la normalización de niveles Sackmann."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.levels import (  # noqa: E402
    TourLevelContext,
    UnknownTourLevelError,
    normalize_tour_level,
)


AUDITED_EXPECTATIONS = {
    # ATP principal.
    ("atp_main", "A"): "ATP Tour",
    ("atp_main", "D"): "Team",
    ("atp_main", "F"): "ATP Tour",
    ("atp_main", "G"): "Grand Slam",
    ("atp_main", "M"): "ATP Tour",
    ("atp_main", "O"): "Other",
    # ATP qualifying/Challenger.
    ("atp_qual_chall", "A"): "ATP Tour",
    ("atp_qual_chall", "C"): "Challenger",
    ("atp_qual_chall", "G"): "Grand Slam",
    ("atp_qual_chall", "M"): "ATP Tour",
    # ATP Futures.
    ("atp_futures", "S"): "ITF",
    ("atp_futures", "15"): "ITF",
    ("atp_futures", "25"): "ITF",
    # WTA principal.
    ("wta_main", "35+H"): "ITF",
    ("wta_main", "50+H"): "ITF",
    ("wta_main", "CC"): "ITF",
    ("wta_main", "D"): "Team",
    ("wta_main", "E"): "Other",
    ("wta_main", "F"): "WTA Tour",
    ("wta_main", "G"): "Grand Slam",
    ("wta_main", "I"): "WTA Tour",
    ("wta_main", "J"): "Other",
    ("wta_main", "O"): "Other",
    ("wta_main", "P"): "WTA Tour",
    ("wta_main", "PM"): "WTA Tour",
    ("wta_main", "T1"): "WTA Tour",
    ("wta_main", "T2"): "WTA Tour",
    ("wta_main", "T3"): "WTA Tour",
    ("wta_main", "T4"): "WTA Tour",
    ("wta_main", "T5"): "WTA Tour",
    ("wta_main", "W"): "WTA Tour",
    # WTA qualifying/ITF.
    ("wta_qual_itf", "10"): "ITF",
    ("wta_qual_itf", "100"): "ITF",
    ("wta_qual_itf", "15"): "ITF",
    ("wta_qual_itf", "20"): "ITF",
    ("wta_qual_itf", "25"): "ITF",
    ("wta_qual_itf", "35"): "ITF",
    ("wta_qual_itf", "40"): "ITF",
    ("wta_qual_itf", "50"): "ITF",
    ("wta_qual_itf", "60"): "ITF",
    ("wta_qual_itf", "75"): "ITF",
    ("wta_qual_itf", "80"): "ITF",
    ("wta_qual_itf", "C"): "Challenger",
    ("wta_qual_itf", "E"): "Other",
    ("wta_qual_itf", "G"): "Grand Slam",
    ("wta_qual_itf", "I"): "WTA Tour",
    ("wta_qual_itf", "P"): "WTA Tour",
    ("wta_qual_itf", "PM"): "WTA Tour",
    ("wta_qual_itf", "T1"): "WTA Tour",
    ("wta_qual_itf", "T2"): "WTA Tour",
    ("wta_qual_itf", "T3"): "WTA Tour",
    ("wta_qual_itf", "T4"): "WTA Tour",
    ("wta_qual_itf", "T5"): "WTA Tour",
    ("wta_qual_itf", "W"): "WTA Tour",
}


class TourLevelNormalizationTest(unittest.TestCase):
    """Comprueba el contrato cerrado de niveles observado en el histórico."""

    def test_every_audited_combination_has_expected_category(self) -> None:
        """Cubre todas las combinaciones reales del manifiesto activo."""

        self.assertEqual(len(AUDITED_EXPECTATIONS), 54)
        for (source_family, raw_level), expected in AUDITED_EXPECTATIONS.items():
            with self.subTest(
                source_family=source_family,
                raw_level=raw_level,
            ):
                result = normalize_tour_level(raw_level, source_family)
                self.assertEqual(
                    result,
                    TourLevelContext(
                        raw_level=raw_level,
                        canonical_level=expected,
                        source_family=source_family,
                    ),
                )

    def test_raw_token_and_source_family_are_preserved(self) -> None:
        """Devuelve literalmente ambos identificadores de la fuente."""

        result = normalize_tour_level("35+H", "wta_main")

        self.assertEqual(result.raw_level, "35+H")
        self.assertEqual(result.source_family, "wta_main")
        self.assertEqual(result.canonical_level, "ITF")

    def test_daily_web_labels_have_an_explicit_separate_mapping(self) -> None:
        """Mapea etiquetas web sin presentarlas como códigos Sackmann."""

        expected = {
            "ATP": "ATP Tour",
            "WTA": "WTA Tour",
            "Challenger": "Challenger",
            "ITF": "ITF",
        }
        for raw_level, canonical in expected.items():
            with self.subTest(raw_level=raw_level):
                context = normalize_tour_level(
                    raw_level,
                    "tennis_explorer",
                )
                self.assertEqual(context.raw_level, raw_level)
                self.assertEqual(context.canonical_level, canonical)
                self.assertEqual(
                    context.source_family,
                    "tennis_explorer",
                )

    def test_context_is_immutable(self) -> None:
        """Impide modificar la categoría después de validarla."""

        result = normalize_tour_level("G", "atp_main")

        with self.assertRaises(FrozenInstanceError):
            result.raw_level = "A"  # type: ignore[misc]

    def test_unknown_level_fails_instead_of_becoming_other(self) -> None:
        """Rechaza un código nuevo incluso dentro de una familia conocida."""

        with self.assertRaisesRegex(
            UnknownTourLevelError,
            "no auditada",
        ):
            normalize_tour_level("NEW", "atp_main")

    def test_known_level_in_unknown_family_fails(self) -> None:
        """Rechaza una familia nueva aunque el código ya exista."""

        with self.assertRaisesRegex(
            UnknownTourLevelError,
            "future_family",
        ):
            normalize_tour_level("G", "future_family")

    def test_known_token_in_unobserved_family_pair_fails(self) -> None:
        """No extrapola significados entre familias sin auditoría."""

        with self.assertRaises(UnknownTourLevelError):
            normalize_tour_level("C", "atp_main")

    def test_tokens_with_peripheral_spaces_are_rejected(self) -> None:
        """Evita corregir silenciosamente una mutación del dato fuente."""

        with self.assertRaisesRegex(ValueError, "espacios periféricos"):
            normalize_tour_level(" G", "atp_main")
        with self.assertRaisesRegex(ValueError, "espacios periféricos"):
            normalize_tour_level("G", "atp_main ")

    def test_empty_tokens_are_rejected(self) -> None:
        """Exige que nivel y familia estén explícitamente poblados."""

        with self.assertRaisesRegex(ValueError, "no puede estar vacío"):
            normalize_tour_level("", "atp_main")
        with self.assertRaisesRegex(ValueError, "no puede estar vacío"):
            normalize_tour_level("G", "")

    def test_non_string_tokens_are_rejected(self) -> None:
        """Distingue un tipo inválido de un código desconocido."""

        with self.assertRaisesRegex(TypeError, "raw_level debe ser str"):
            normalize_tour_level(15, "atp_futures")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "source_family debe ser str"):
            normalize_tour_level("G", None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

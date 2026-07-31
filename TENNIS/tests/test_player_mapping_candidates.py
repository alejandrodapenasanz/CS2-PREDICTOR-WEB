"""Tests offline de normalización y candidatos activos sin fugas."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.player_mapping.candidates import (  # noqa: E402
    active_window_start,
    build_active_candidate_index,
)
from src.player_mapping.normalization import (  # noqa: E402
    build_sackmann_name_key,
    normalize_name_text,
    parse_visible_name,
)
from src.player_mapping.types import (  # noqa: E402
    NameKey,
    NameParsingError,
    PlayerMappingValidationError,
)


def _real_case_players() -> pd.DataFrame:
    """Crea un maestro pequeño con identidades reales y controles negativos."""

    return pd.DataFrame(
        {
            "gender": [
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "F",
            ],
            "player_id": [
                105138,
                200000,
                122298,
                202103,
                206909,
                210416,
                299998,
                202103,
            ],
            "player_name": [
                "Roberto Bautista Agut",
                "Felix Auger Aliassime",
                "Botic Van De Zandschulp",
                "Francisco Cerundolo",
                "Brandon Nakashima",
                "Bryce Nakashima",
                "Future Player",
                "Female Identity",
            ],
            "name_first": [
                "Roberto",
                "Felix",
                "Botic",
                "Francisco",
                "Brandon",
                "Bryce",
                "Future",
                "Female",
            ],
            "name_last": [
                "Bautista Agut",
                "Auger Aliassime",
                "Van De Zandschulp",
                "Cerundolo",
                "Nakashima",
                "Nakashima",
                "Player",
                "Identity",
            ],
            "ioc": [
                "ESP",
                "CAN",
                "NED",
                "ARG",
                "USA",
                "USA",
                "USA",
                "ARG",
            ],
        }
    )


def _causal_matches() -> pd.DataFrame:
    """Crea partidos en ambos géneros alrededor de los límites temporales."""

    return pd.DataFrame(
        {
            "gender": ["M", "M", "M", "M", "F"],
            "tourney_date": pd.to_datetime(
                [
                    "2023-07-30",
                    "2024-02-12",
                    "2026-07-29",
                    "2026-07-30",
                    "2025-01-06",
                ]
            ),
            "winner_id": [
                105138,
                122298,
                206909,
                299998,
                202103,
            ],
            "loser_id": [
                200000,
                202103,
                210416,
                105138,
                202103,
            ],
        }
    )


class NameNormalizationTest(unittest.TestCase):
    """Comprueba equivalencias tipográficas sin introducir fuzzy matching."""

    def test_normalize_name_text_removes_accents_hyphens_and_punctuation(
        self,
    ) -> None:
        """Convierte variantes legítimas a una representación ASCII estable."""

        self.assertEqual(
            normalize_name_text("  Auger-Aliassime,  F. "),
            "auger aliassime f",
        )
        self.assertEqual(normalize_name_text("Cerúndolo"), "cerundolo")

    def test_compound_real_names_match_sackmann_keys(self) -> None:
        """Resuelve los tres apellidos compuestos reales exigidos."""

        cases = (
            ("Bautista Agut R.", "Roberto", "Bautista Agut"),
            ("Auger-Aliassime F.", "Felix", "Auger Aliassime"),
            ("Van De Zandschulp B.", "Botic", "Van De Zandschulp"),
        )
        for visible_name, name_first, name_last in cases:
            with self.subTest(visible_name=visible_name):
                self.assertEqual(
                    parse_visible_name(visible_name),
                    build_sackmann_name_key(name_first, name_last),
                )

    def test_accented_visible_surname_matches_unaccented_sackmann_name(
        self,
    ) -> None:
        """Hace equivalente ``Cerúndolo`` a la grafía Sackmann sin acento."""

        self.assertEqual(
            parse_visible_name("Cerúndolo F."),
            NameKey("cerundolo", "f"),
        )
        self.assertEqual(
            parse_visible_name("Cerúndolo F."),
            build_sackmann_name_key("Francisco", "Cerundolo"),
        )

    def test_visible_name_without_one_terminal_initial_is_rejected(self) -> None:
        """Rechaza un nombre completo en lugar de interpretarlo por aproximación."""

        with self.assertRaises(NameParsingError):
            parse_visible_name("Roberto Bautista Agut")


class ActiveCandidateIndexTest(unittest.TestCase):
    """Verifica género, ventana causal, colisiones y campos de candidatos."""

    def setUp(self) -> None:
        """Prepara un corte diario y DataFrames pequeños reproducibles."""

        self.as_of_date = date(2026, 7, 30)
        self.players = _real_case_players()
        self.matches = _causal_matches()

    def test_active_window_is_three_calendar_years_and_handles_leap_day(
        self,
    ) -> None:
        """Calcula años naturales, incluido el 29 de febrero."""

        self.assertEqual(
            active_window_start(self.as_of_date),
            date(2023, 7, 30),
        )
        self.assertEqual(
            active_window_start(date(2024, 2, 29)),
            date(2021, 2, 28),
        )

    def test_index_contains_real_cases_with_last_pre_match_date(self) -> None:
        """Incluye casos reales activos y conserva su última fecha anterior."""

        index = build_active_candidate_index(
            self.players,
            self.matches,
            "M",
            self.as_of_date,
        )

        expected = {
            NameKey("bautista agut", "r"): 105138,
            NameKey("auger aliassime", "f"): 200000,
            NameKey("van de zandschulp", "b"): 122298,
            NameKey("cerundolo", "f"): 202103,
        }
        for name_key, player_id in expected.items():
            with self.subTest(name_key=name_key):
                self.assertEqual(
                    tuple(candidate.player_id for candidate in index[name_key]),
                    (player_id,),
                )
                self.assertLess(
                    index[name_key][0].last_match_date,
                    self.as_of_date,
                )

    def test_same_surname_initial_collision_is_preserved_not_guessed(
        self,
    ) -> None:
        """Conserva Brandon y Bryce Nakashima aunque compartan país."""

        index = build_active_candidate_index(
            self.players,
            self.matches,
            "M",
            self.as_of_date,
        )

        collision = index[NameKey("nakashima", "b")]
        self.assertEqual(
            tuple(candidate.player_id for candidate in collision),
            (206909, 210416),
        )
        self.assertEqual(
            {candidate.ioc for candidate in collision},
            {"USA"},
        )

    def test_match_on_or_after_cutoff_never_activates_a_player(self) -> None:
        """Excluye D y demuestra que añadir futuro no cambia el índice as-of."""

        baseline = build_active_candidate_index(
            self.players,
            self.matches.iloc[:3].copy(),
            "M",
            self.as_of_date,
        )
        with_future = build_active_candidate_index(
            self.players,
            self.matches.iloc[:4].copy(),
            "M",
            self.as_of_date,
        )

        self.assertEqual(baseline, with_future)
        self.assertNotIn(NameKey("player", "f"), with_future)

    def test_other_gender_with_same_player_id_cannot_enter_index(self) -> None:
        """Evita mezclar universos aunque Sackmann reutilice un player_id."""

        index = build_active_candidate_index(
            self.players,
            self.matches,
            "M",
            self.as_of_date,
        )

        cerundolo = index[NameKey("cerundolo", "f")]
        self.assertEqual(len(cerundolo), 1)
        self.assertEqual(cerundolo[0].gender, "M")
        self.assertEqual(cerundolo[0].player_name, "Francisco Cerundolo")

    def test_invalid_gender_is_a_controlled_error(self) -> None:
        """Rechaza valores de género corregibles solo mediante un supuesto."""

        with self.assertRaises(PlayerMappingValidationError):
            build_active_candidate_index(
                self.players,
                self.matches,
                "m",
                self.as_of_date,
            )


if __name__ == "__main__":
    unittest.main()

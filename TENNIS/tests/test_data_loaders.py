"""Tests de los loaders con muestras que conservan el esquema real Sackmann."""

from pathlib import Path
import sys
import unittest

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders import (  # noqa: E402
    KNOWN_DRAW_SIZE_ANOMALY_FLAG,
    SchemaMismatchError,
    _apply_known_match_anomalies,
    load_matches,
    load_players,
)


FIXTURE_RAW_DIR = Path(__file__).resolve().parent / "fixtures" / "raw"


class DataLoadersTest(unittest.TestCase):
    """Comprueba carga, columnas obligatorias, géneros, niveles y fechas."""

    def test_matches_load_all_families_with_typed_dates(self) -> None:
        """Carga las cinco familias y verifica claves y columnas derivadas."""

        matches = load_matches(raw_dir=FIXTURE_RAW_DIR)

        required_columns = {
            "winner_id",
            "loser_id",
            "surface",
            "tourney_date",
            "tourney_level",
            "round",
            "best_of",
            "winner_rank",
            "winner_rank_points",
            "loser_rank",
            "loser_rank_points",
            "gender",
            "tour_level",
            "source_family",
            "source_file",
        }
        self.assertTrue(required_columns.issubset(matches.columns))
        self.assertEqual(len(matches), 5)
        self.assertEqual(set(matches["gender"]), {"M", "F"})
        self.assertFalse(matches["gender"].isna().any())
        self.assertFalse(matches["tour_level"].isna().any())
        self.assertTrue(
            matches["tour_level"].equals(matches["tourney_level"])
        )
        self.assertTrue(is_datetime64_any_dtype(matches["tourney_date"]))
        self.assertEqual(str(matches["winner_id"].dtype), "Int64")

    def test_matches_can_be_loaded_for_one_gender(self) -> None:
        """Limita la carga a hombres sin mezclar filas del universo femenino."""

        matches = load_matches("M", raw_dir=FIXTURE_RAW_DIR)

        self.assertEqual(len(matches), 3)
        self.assertEqual(set(matches["gender"]), {"M"})
        self.assertEqual(
            set(matches["source_family"]),
            {"atp_main", "atp_qual_chall", "atp_futures"},
        )

    def test_players_load_per_gender_with_typed_birth_dates(self) -> None:
        """Carga jugadores por género y convierte DOB inválidos en NaT."""

        players = load_players("F", raw_dir=FIXTURE_RAW_DIR)

        required_columns = {
            "player_id",
            "player_name",
            "hand",
            "ioc",
            "dob",
            "gender",
        }
        self.assertTrue(required_columns.issubset(players.columns))
        self.assertEqual(set(players["gender"]), {"F"})
        self.assertTrue(is_datetime64_any_dtype(players["dob"]))
        self.assertEqual(str(players["player_id"].dtype), "Int64")
        self.assertEqual(players.loc[0, "player_name"], "Martina Hingis")
        self.assertTrue(pd.isna(players.loc[1, "dob"]))

    def test_only_approved_exhibition_draw_size_becomes_missing(self) -> None:
        """Normaliza el único ``exho`` aprobado y conserva una marca explícita."""

        frame = pd.DataFrame(
            {
                "tourney_id": ["1970-1093"],
                "tourney_name": ["Rondebosch Exho"],
                "tourney_level": ["E"],
                "tourney_date": ["19700406"],
                "match_num": ["1"],
                "draw_size": ["exho"],
            },
            dtype="string",
        )

        result = _apply_known_match_anomalies(
            frame,
            Path("wta_matches_1970.csv"),
        )

        self.assertTrue(pd.isna(result.loc[0, "draw_size"]))
        self.assertEqual(
            result.loc[0, "source_anomaly"],
            KNOWN_DRAW_SIZE_ANOMALY_FLAG,
        )

    def test_unapproved_exhibition_draw_size_still_fails(self) -> None:
        """Rechaza cualquier ``exho`` que no coincida con la fila autorizada."""

        frame = pd.DataFrame(
            {
                "tourney_id": ["2099-9999"],
                "tourney_name": ["Unknown"],
                "tourney_level": ["E"],
                "tourney_date": ["20990101"],
                "match_num": ["1"],
                "draw_size": ["exho"],
            },
            dtype="string",
        )

        with self.assertRaises(SchemaMismatchError):
            _apply_known_match_anomalies(
                frame,
                Path("wta_matches_2099.csv"),
            )

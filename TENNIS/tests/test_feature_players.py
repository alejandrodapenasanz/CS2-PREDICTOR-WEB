"""Pruebas offline del índice causal de edad de jugadores."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.players import (  # noqa: E402
    PlayerAgeError,
    PlayerAgeIndex,
)


class PlayerAgeIndexTest(unittest.TestCase):
    """Comprueba orientación temporal, ausencias y aislamiento por género."""

    def test_age_and_distance_are_computed_at_match_date(self) -> None:
        """Calcula edad decimal y distancia a 30 sin usar una edad actual."""

        index = PlayerAgeIndex(
            {("M", 1): date(1994, 1, 1)},
            reference_years=30.0,
            days_per_year=365.2425,
        )

        snapshot = index.get(
            "M",
            1,
            as_of_date=date(2024, 1, 1),
        )

        self.assertAlmostEqual(snapshot.age_years or 0.0, 29.9992, places=3)
        self.assertAlmostEqual(
            snapshot.distance_to_reference or 0.0,
            abs((date(2024, 1, 1) - date(1994, 1, 1)).days / 365.2425 - 30),
        )
        self.assertFalse(snapshot.is_missing)

    def test_missing_and_future_birth_dates_are_not_fabricated(self) -> None:
        """Distingue maestro ausente de una fecha incompatible con el partido."""

        index = PlayerAgeIndex(
            {
                ("F", 1): None,
                ("F", 2): date(2025, 1, 1),
            }
        )

        missing = index.get("F", 1, as_of_date=date(2024, 1, 1))
        invalid = index.get("F", 2, as_of_date=date(2024, 1, 1))
        unknown = index.get("F", 3, as_of_date=date(2024, 1, 1))

        self.assertTrue(missing.is_missing)
        self.assertFalse(missing.invalid_for_date)
        self.assertTrue(invalid.is_missing)
        self.assertTrue(invalid.invalid_for_date)
        self.assertIsNone(invalid.age_years)
        self.assertTrue(unknown.is_missing)

    def test_equal_ids_in_different_genders_do_not_share_birth_date(self) -> None:
        """Mantiene universos biográficos separados por género."""

        index = PlayerAgeIndex(
            {
                ("M", 7): date(1990, 1, 1),
                ("F", 7): date(2000, 1, 1),
            }
        )
        cutoff = date(2024, 1, 1)

        male = index.get("M", 7, as_of_date=cutoff)
        female = index.get("F", 7, as_of_date=cutoff)

        self.assertGreater(male.age_years or 0.0, female.age_years or 0.0)

    def test_dataframe_contract_rejects_missing_or_conflicting_columns(
        self,
    ) -> None:
        """No inventa columnas y falla ante DOB distintas para la misma clave."""

        with self.assertRaises(PlayerAgeError):
            PlayerAgeIndex.from_dataframe(
                pd.DataFrame({"gender": ["M"], "player_id": [1]})
            )
        with self.assertRaises(PlayerAgeError):
            PlayerAgeIndex.from_dataframe(
                pd.DataFrame(
                    {
                        "gender": ["M", "M"],
                        "player_id": [1, 1],
                        "dob": pd.to_datetime(
                            ["1990-01-01", "1991-01-01"]
                        ),
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()

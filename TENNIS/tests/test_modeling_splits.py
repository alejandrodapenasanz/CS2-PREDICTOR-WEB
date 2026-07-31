"""Pruebas sintéticas de las particiones temporales expansivas."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.splits import (  # noqa: E402
    TemporalFold,
    TemporalSplitError,
    build_expanding_season_folds,
    validate_temporal_fold,
)


class ExpandingSeasonSplitsTest(unittest.TestCase):
    """Verifica aislamiento y expansión causal temporada a temporada."""

    def setUp(self) -> None:
        """Crea dos filas por temporada con un índice no posicional."""

        self.frame = pd.DataFrame(
            {
                "record_id": [f"match-{value}" for value in range(12)],
                "match_date": pd.to_datetime(
                    [
                        f"{year}-{month:02d}-15"
                        for year in range(2017, 2023)
                        for month in (2, 9)
                    ]
                ),
            },
            index=range(100, 112),
        )

    def test_fold_uses_past_calibration_previous_year_and_future_test(
        self,
    ) -> None:
        """Fija train <=Y-2, calibración Y-1 y test Y sin solapamientos."""

        folds = build_expanding_season_folds(
            self.frame,
            test_seasons=(2021, 2022),
            record_id_column="record_id",
        )
        self.assertEqual([fold.test_season for fold in folds], [2021, 2022])

        first_train, first_calibration, first_test = folds[0].take(self.frame)
        self.assertLessEqual(first_train["match_date"].dt.year.max(), 2019)
        self.assertEqual(
            set(first_calibration["match_date"].dt.year), {2020}
        )
        self.assertEqual(set(first_test["match_date"].dt.year), {2021})
        self.assertTrue(
            set(folds[0].train_positions).isdisjoint(
                folds[0].calibration_positions
            )
        )
        self.assertTrue(
            set(folds[0].train_positions).isdisjoint(
                folds[0].test_positions
            )
        )
        self.assertGreater(
            len(folds[1].train_positions),
            len(folds[0].train_positions),
        )

    def test_default_discovers_only_complete_folds(self) -> None:
        """Descubre temporadas viables sin crear bloques vacíos."""

        folds = build_expanding_season_folds(self.frame)
        self.assertEqual(
            [fold.test_season for fold in folds],
            [2019, 2020, 2021, 2022],
        )

    def test_missing_previous_season_and_duplicate_identity_fail(self) -> None:
        """Rechaza calibración ausente e identidades repetidas."""

        without_2020 = self.frame.loc[
            self.frame["match_date"].dt.year != 2020
        ]
        with self.assertRaisesRegex(TemporalSplitError, "calibración"):
            build_expanding_season_folds(
                without_2020,
                test_seasons=(2021,),
            )

        duplicated = self.frame.copy()
        duplicated.iloc[-1, duplicated.columns.get_loc("record_id")] = (
            duplicated.iloc[0]["record_id"]
        )
        with self.assertRaisesRegex(TemporalSplitError, "repetidas"):
            build_expanding_season_folds(
                duplicated,
                test_seasons=(2021,),
                record_id_column="record_id",
            )

    def test_manual_leaking_fold_is_rejected(self) -> None:
        """Detecta una fila futura introducida manualmente en train."""

        leaking = TemporalFold(
            test_season=2021,
            train_positions=(0, 1, 10),
            calibration_positions=(6, 7),
            test_positions=(8, 9),
        )
        with self.assertRaisesRegex(
            TemporalSplitError,
            "posteriores",
        ):
            validate_temporal_fold(self.frame["match_date"], leaking)


if __name__ == "__main__":
    unittest.main()

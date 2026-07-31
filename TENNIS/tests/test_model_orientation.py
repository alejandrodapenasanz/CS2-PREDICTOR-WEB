"""Pruebas de antisimetría del contrato de inferencia de fase 9."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np
import pandas as pd

from src.features import MODEL_FEATURE_COLUMNS
from src.modeling.calibration import PlattCalibrator
from src.modeling.orientation import (
    predict_symmetric_calibrated,
    predict_symmetric_raw,
    reverse_model_orientation,
)


def _complete_model_row() -> pd.DataFrame:
    """Crea una fila completa y válida para invertir las dos orientaciones."""

    values: dict[str, object] = {}
    categorical = {
        "surface": "Hard",
        "tour_level": "ATP Tour",
        "tour_level_raw": "ATP",
        "round": "R32",
    }
    booleans = {
        "elo_cold_start_a",
        "elo_cold_start_b",
        "ranking_missing_a",
        "ranking_missing_b",
        "age_missing_a",
        "age_missing_b",
    }
    for position, column in enumerate(MODEL_FEATURE_COLUMNS, start=1):
        if column in categorical:
            values[column] = categorical[column]
        elif column == "best_of":
            values[column] = 3
        elif column in booleans:
            values[column] = position % 2 == 0
        else:
            values[column] = float(position) / 10.0
    values["gender"] = "M"
    return pd.DataFrame([values], index=["match"])


@dataclass
class _FakeEstimator:
    """Estimador deliberadamente no simétrico para probar el promedio."""

    feature_columns: tuple[str, ...] = tuple(MODEL_FEATURE_COLUMNS)

    def predict_probability(self, frame: pd.DataFrame) -> pd.Series:
        """Combina una diferencia con un intercepto que rompe simetría."""

        difference = pd.to_numeric(
            frame["elo_general_diff"], errors="raise"
        ).to_numpy(dtype=float)
        probability = 1.0 / (1.0 + np.exp(-(0.7 + difference / 30.0)))
        return pd.Series(probability, index=frame.index, dtype=float)


class ModelOrientationTests(unittest.TestCase):
    """Asegura que el orden de la web no altera el pronóstico deportivo."""

    def test_reversing_twice_recovers_every_model_feature(self) -> None:
        """La transformación A/B es completa e involutiva."""

        original = _complete_model_row()
        reversed_once = reverse_model_orientation(
            original,
            MODEL_FEATURE_COLUMNS,
        )
        recovered = reverse_model_orientation(
            reversed_once,
            MODEL_FEATURE_COLUMNS,
        )

        pd.testing.assert_frame_equal(
            recovered.loc[:, list(MODEL_FEATURE_COLUMNS)],
            original.loc[:, list(MODEL_FEATURE_COLUMNS)],
            check_dtype=False,
        )

    def test_raw_and_calibrated_probabilities_are_exact_complements(self) -> None:
        """Invertir jugadores produce exactamente uno menos la probabilidad."""

        original = _complete_model_row()
        reversed_frame = reverse_model_orientation(
            original,
            MODEL_FEATURE_COLUMNS,
        )
        estimator = _FakeEstimator()
        raw = predict_symmetric_raw(estimator, original).iloc[0]
        reversed_raw = predict_symmetric_raw(
            estimator,
            reversed_frame,
        ).iloc[0]
        calibrator = PlattCalibrator().fit(
            np.array([0.1, 0.3, 0.7, 0.9]),
            np.array([0, 0, 1, 1]),
        )
        calibrated = predict_symmetric_calibrated(
            calibrator,
            np.array([raw]),
        )[0]
        reversed_calibrated = predict_symmetric_calibrated(
            calibrator,
            np.array([reversed_raw]),
        )[0]

        self.assertAlmostEqual(raw + reversed_raw, 1.0, places=15)
        self.assertAlmostEqual(
            calibrated + reversed_calibrated,
            1.0,
            places=15,
        )


if __name__ == "__main__":
    unittest.main()

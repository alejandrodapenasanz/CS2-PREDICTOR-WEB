"""Pruebas sintéticas de calibración Platt causal."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.calibration import (  # noqa: E402
    CalibrationError,
    PlattCalibrator,
)


class PlattCalibratorTest(unittest.TestCase):
    """Comprueba ajuste separado, validación y probabilidades válidas."""

    def test_fit_uses_only_supplied_calibration_block(self) -> None:
        """Cambiar etiquetas de test no puede alterar el calibrador ajustado."""

        calibration_probability = np.linspace(0.05, 0.95, 200)
        calibration_target = (
            calibration_probability
            > np.linspace(0.2, 0.8, 200)
        ).astype(int)
        test_probability = np.array([0.2, 0.5, 0.8])
        test_target_a = np.array([0, 0, 1])
        test_target_b = 1 - test_target_a

        first = PlattCalibrator().fit(
            calibration_probability,
            calibration_target,
        )
        prediction_a = first.predict_proba(test_probability)
        second = PlattCalibrator().fit(
            calibration_probability,
            calibration_target,
        )
        prediction_b = second.predict_proba(test_probability)

        self.assertFalse(np.array_equal(test_target_a, test_target_b))
        np.testing.assert_allclose(prediction_a, prediction_b)
        self.assertEqual(first.n_calibration_samples_, 200)
        self.assertTrue(((prediction_a >= 0) & (prediction_a <= 1)).all())

    def test_extreme_raw_probabilities_are_safely_clipped(self) -> None:
        """Permite 0 y 1 sin producir logits infinitos."""

        calibrator = PlattCalibrator().fit(
            np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]),
            np.array([0, 0, 0, 1, 1, 1]),
        )
        prediction = calibrator.predict_proba(np.array([0.0, 1.0]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(prediction[0], prediction[1])

    def test_unfitted_single_class_and_invalid_probability_fail(self) -> None:
        """Rechaza estados o bloques para los que Platt no es identificable."""

        with self.assertRaisesRegex(CalibrationError, "ajustarse"):
            PlattCalibrator().predict_proba([0.5])
        with self.assertRaisesRegex(CalibrationError, "ambas clases"):
            PlattCalibrator().fit([0.2, 0.3], [0, 0])
        with self.assertRaisesRegex(CalibrationError, r"\[0, 1\]"):
            PlattCalibrator().fit([0.2, 1.1], [0, 1])


if __name__ == "__main__":
    unittest.main()

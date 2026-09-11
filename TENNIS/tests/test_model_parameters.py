"""Pruebas de configuración versionada de estimadores."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.parameters import (  # noqa: E402
    MODEL_ARTIFACT_VERSION,
    LightGBMParameters,
    LogisticParameters,
)


class ModelingParametersTest(unittest.TestCase):
    """Fija validación y serialización de hiperparámetros."""

    def test_parameters_are_versioned_and_serializable(self) -> None:
        """Expone una versión estable y mappings reproducibles."""

        self.assertEqual(MODEL_ARTIFACT_VERSION, "tennis-model-v3")
        self.assertEqual(LogisticParameters().as_dict()["c"], 1.0)
        self.assertEqual(
            LightGBMParameters().as_dict()["n_estimators"], 500
        )
        self.assertEqual(
            LightGBMParameters().as_dict()["subsample_freq"], 1
        )

    def test_invalid_parameters_fail_without_clamping(self) -> None:
        """Rechaza límites inválidos en vez de corregirlos."""

        with self.assertRaises(ValueError):
            LogisticParameters(c=0.0)
        with self.assertRaises(ValueError):
            LogisticParameters(max_iter=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LightGBMParameters(subsample=1.1)
        with self.assertRaises(ValueError):
            LightGBMParameters(max_depth=0)
        with self.assertRaises(ValueError):
            LightGBMParameters(n_jobs=0)


if __name__ == "__main__":
    unittest.main()

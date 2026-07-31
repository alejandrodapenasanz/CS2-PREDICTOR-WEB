"""Pruebas offline de la configuración versionada de features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.parameters import (  # noqa: E402
    DEFAULT_FEATURE_PARAMETERS,
    FEATURE_SCHEMA_VERSION,
    FeatureParameters,
)


class FeatureParametersTest(unittest.TestCase):
    """Fija defaults aprobados y rechaza configuraciones ambiguas."""

    def test_defaults_match_the_approved_contract(self) -> None:
        """Conserva ventanas, edad de referencia y semilla documentadas."""

        parameters = DEFAULT_FEATURE_PARAMETERS

        self.assertEqual(FEATURE_SCHEMA_VERSION, "tennis-features-v1")
        self.assertEqual(parameters.recent_matches, 10)
        self.assertEqual(parameters.recent_months, 3)
        self.assertEqual(parameters.age_reference_years, 30.0)
        self.assertEqual(parameters.days_per_year, 365.2425)
        self.assertEqual(parameters.orientation_seed, 42)
        self.assertEqual(parameters.as_dict()["recent_matches"], 10)

    def test_invalid_values_fail_without_silent_correction(self) -> None:
        """Rechaza ventanas no positivas, booleanos y referencias inválidas."""

        invalid_builders = (
            lambda: FeatureParameters(recent_matches=0),
            lambda: FeatureParameters(recent_months=True),
            lambda: FeatureParameters(age_reference_years=0.0),
            lambda: FeatureParameters(days_per_year=-1.0),
            lambda: FeatureParameters(orientation_seed=True),
        )
        for builder in invalid_builders:
            with self.subTest(builder=builder):
                with self.assertRaises(ValueError):
                    builder()


if __name__ == "__main__":
    unittest.main()

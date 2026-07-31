"""Pruebas offline del contrato y preprocesamiento de modelos."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
)
from src.modeling.preprocessing import (  # noqa: E402
    MARKET_FEATURE_COLUMNS,
    FeatureContractError,
    build_preprocessor,
    load_feature_contract,
    select_model_frame,
)


def _manifest(*, historical_odds: bool = False) -> dict[str, object]:
    """Construye un manifiesto mínimo fiel al contrato de fase 6."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "fingerprint": "a" * 64,
        "historical_odds_available": historical_odds,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "training_columns": [
            "gender",
            "match_date",
            *MODEL_FEATURE_COLUMNS,
            "y",
        ],
    }


class FeatureContractTest(unittest.TestCase):
    """Comprueba allowlist, perfiles y ausencia de selección silenciosa."""

    def test_auto_profile_excludes_market_without_historical_odds(
        self,
    ) -> None:
        """Resuelve auto como sports_only con el manifiesto real actual."""

        contract = load_feature_contract(_manifest())

        self.assertEqual(contract.resolve_profile(), "sports_only")
        self.assertTrue(
            set(contract.columns_for()).isdisjoint(MARKET_FEATURE_COLUMNS)
        )
        with self.assertRaises(FeatureContractError):
            contract.columns_for("market_enhanced")

    def test_market_profile_requires_and_keeps_declared_features(
        self,
    ) -> None:
        """Mantiene todas las features cuando el histórico de mercado existe."""

        contract = load_feature_contract(_manifest(historical_odds=True))

        self.assertEqual(contract.resolve_profile(), "market_enhanced")
        self.assertEqual(
            contract.columns_for(), tuple(MODEL_FEATURE_COLUMNS)
        )

    def test_manifest_allowlist_must_match_exactly(self) -> None:
        """Rechaza ausencias, extras y cambios de orden en la allowlist."""

        missing = _manifest()
        missing["model_feature_columns"] = list(MODEL_FEATURE_COLUMNS[:-1])
        with self.assertRaises(FeatureContractError):
            load_feature_contract(missing)

        reordered = _manifest()
        columns = list(MODEL_FEATURE_COLUMNS)
        columns[0], columns[1] = columns[1], columns[0]
        reordered["model_feature_columns"] = columns
        with self.assertRaises(FeatureContractError):
            load_feature_contract(reordered)

    def test_select_model_frame_drops_audit_target_and_extra(self) -> None:
        """Devuelve solo sports_only en orden y falla si falta una feature."""

        contract = load_feature_contract(_manifest())
        frame = pd.DataFrame(
            {
                column: [0]
                for column in MODEL_FEATURE_COLUMNS
            }
        )
        frame["gender"] = "M"
        frame["match_date"] = pd.Timestamp("2020-01-01")
        frame["y"] = 1
        frame["future_result_proxy"] = 1

        selected = select_model_frame(frame, contract)

        self.assertEqual(
            tuple(selected.columns), contract.columns_for("sports_only")
        )
        self.assertNotIn("y", selected)
        self.assertNotIn("gender", selected)
        self.assertNotIn("match_date", selected)
        self.assertNotIn("future_result_proxy", selected)
        with self.assertRaises(FeatureContractError):
            select_model_frame(
                frame.drop(columns=[contract.columns_for()[0]]),
                contract,
            )


class PreprocessorTest(unittest.TestCase):
    """Comprueba que estadísticas y categorías proceden solo de train."""

    def test_numeric_median_is_fitted_only_on_supplied_train(self) -> None:
        """Fija la mediana del train y no consulta un futuro extremo."""

        preprocessor = build_preprocessor(
            ("elo_general_diff",),
            scale_numeric=True,
        )
        train = pd.DataFrame({"elo_general_diff": [0.0, 2.0, np.nan]})
        future = pd.DataFrame({"elo_general_diff": [10_000.0, np.nan]})

        preprocessor.fit(train)
        transformed = preprocessor.transform(future)
        imputer = preprocessor.named_transformers_[
            "numeric"
        ].named_steps["imputer"]

        self.assertEqual(float(imputer.statistics_[0]), 1.0)
        self.assertEqual(transformed.shape[0], 2)

    def test_empty_numeric_feature_is_not_silently_dropped(self) -> None:
        """Conserva una feature totalmente nula y añade su indicador."""

        preprocessor = build_preprocessor(
            ("market_probability_a",),
            scale_numeric=False,
        )
        transformed = preprocessor.fit_transform(
            pd.DataFrame({"market_probability_a": [np.nan, np.nan]})
        )

        self.assertEqual(transformed.shape, (2, 2))

    def test_unseen_category_is_supported_without_refit(self) -> None:
        """Codifica una superficie futura desconocida sin alterar categorías."""

        preprocessor = build_preprocessor(
            ("surface",),
            scale_numeric=False,
        )
        preprocessor.fit(pd.DataFrame({"surface": ["Hard", "Clay"]}))
        transformed = preprocessor.transform(
            pd.DataFrame({"surface": ["Grass"]})
        )

        self.assertEqual(transformed.shape, (1, 2))
        self.assertEqual(float(transformed.sum()), 0.0)

    def test_preprocessor_rejects_feature_outside_allowlist(self) -> None:
        """Impide usar el objetivo incluso a través de la API de bajo nivel."""

        with self.assertRaises(FeatureContractError):
            build_preprocessor(("y",), scale_numeric=False)


if __name__ == "__main__":
    unittest.main()

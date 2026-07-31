"""Pruebas sintéticas de pipelines por género de la fase 7."""

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
from src.modeling.estimators import (  # noqa: E402
    EstimatorContractError,
    fit_gender_estimator,
)
from src.modeling.parameters import LightGBMParameters  # noqa: E402
from src.modeling.preprocessing import (  # noqa: E402
    FeatureContractError,
    load_feature_contract,
)


_CATEGORICAL_VALUES = {
    "surface": ("Hard", "Clay"),
    "tour_level": ("ATP Tour", "Grand Slam"),
    "tour_level_raw": ("A", "G"),
    "round": ("R32", "F"),
}
_BOOLEAN_COLUMNS = {
    "elo_cold_start_a",
    "elo_cold_start_b",
    "ranking_missing_a",
    "ranking_missing_b",
    "age_missing_a",
    "age_missing_b",
}


def _manifest(*, historical_odds: bool = False) -> dict[str, object]:
    """Devuelve un manifiesto sintético válido y completo."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "fingerprint": "b" * 64,
        "historical_odds_available": historical_odds,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "training_columns": [
            "gender",
            "match_date",
            *MODEL_FEATURE_COLUMNS,
            "y",
        ],
    }


def _training_frame(
    rows: int = 40,
    *,
    gender: str = "M",
    with_market: bool = False,
) -> pd.DataFrame:
    """Crea un train pequeño con todas las columnas del contrato."""

    data: dict[str, object] = {}
    for position, column in enumerate(MODEL_FEATURE_COLUMNS):
        if column in _CATEGORICAL_VALUES:
            values = _CATEGORICAL_VALUES[column]
            data[column] = [values[index % 2] for index in range(rows)]
        elif column in _BOOLEAN_COLUMNS:
            data[column] = [
                bool((index + position) % 2) for index in range(rows)
            ]
        else:
            data[column] = (
                np.arange(rows, dtype=float) * (position + 1)
            )
    if with_market:
        data["odds_a"] = [1.7 if index % 2 else 2.2 for index in range(rows)]
        data["odds_b"] = [2.2 if index % 2 else 1.7 for index in range(rows)]
        data["market_probability_a"] = [
            0.56 if index % 2 else 0.44 for index in range(rows)
        ]
        data["market_probability_b"] = [
            0.44 if index % 2 else 0.56 for index in range(rows)
        ]
        data["market_overround"] = 1.05
        data["market_margin"] = 0.05
    else:
        for column in (
            "odds_a",
            "odds_b",
            "market_probability_a",
            "market_probability_b",
            "market_overround",
            "market_margin",
        ):
            data[column] = np.nan
    frame = pd.DataFrame(data)
    frame["gender"] = gender
    frame["match_date"] = pd.date_range("2010-01-01", periods=rows)
    frame["y"] = np.arange(rows) % 2
    frame["record_id"] = [f"row-{index}" for index in range(rows)]
    return frame


class GenderEstimatorTest(unittest.TestCase):
    """Valida aislamiento por género, perfiles y predicción probabilística."""

    def test_logistic_uses_sports_profile_and_predicts_probabilities(
        self,
    ) -> None:
        """Entrena sin mercado cuando el manifiesto declara cobertura cero."""

        train = _training_frame()
        contract = load_feature_contract(_manifest())

        fitted = fit_gender_estimator(
            train,
            gender="M",
            kind="logistic",
            contract=contract,
        )
        probabilities = fitted.predict_probability(train.iloc[:5])

        self.assertEqual(fitted.profile, "sports_only")
        self.assertNotIn(
            "market_probability_a", fitted.feature_columns
        )
        self.assertEqual(list(probabilities.index), list(range(5)))
        self.assertTrue(probabilities.between(0.0, 1.0).all())

    def test_lightgbm_pipeline_trains_with_small_explicit_parameters(
        self,
    ) -> None:
        """Ejercita el estimador principal sin lanzar un histórico completo."""

        train = _training_frame()
        fitted = fit_gender_estimator(
            train,
            gender="M",
            kind="lightgbm",
            contract=load_feature_contract(_manifest()),
            lightgbm_parameters=LightGBMParameters(
                n_estimators=8,
                learning_rate=0.1,
                num_leaves=7,
                min_child_samples=2,
                n_jobs=1,
            ),
        )

        probabilities = fitted.predict_probability(train.iloc[:4])

        self.assertEqual(fitted.kind, "lightgbm")
        self.assertTrue(probabilities.between(0.0, 1.0).all())

    def test_gender_mixing_is_rejected_in_train_and_inference(self) -> None:
        """Impide combinar los universos masculino y femenino."""

        train = _training_frame()
        mixed = train.copy()
        mixed.loc[mixed.index[-1], "gender"] = "F"
        contract = load_feature_contract(_manifest())

        with self.assertRaises(EstimatorContractError):
            fit_gender_estimator(
                mixed,
                gender="M",
                kind="logistic",
                contract=contract,
            )
        fitted = fit_gender_estimator(
            train,
            gender="M",
            kind="logistic",
            contract=contract,
        )
        female = train.iloc[:2].copy()
        female["gender"] = "F"
        with self.assertRaises(EstimatorContractError):
            fitted.predict_probability(female)

    def test_market_profile_needs_contract_and_training_coverage(
        self,
    ) -> None:
        """No activa mercado por petición si faltan cuotas históricas."""

        without_market = _training_frame()
        no_odds_contract = load_feature_contract(_manifest())
        with self.assertRaises(FeatureContractError):
            fit_gender_estimator(
                without_market,
                gender="M",
                kind="logistic",
                contract=no_odds_contract,
                profile="market_enhanced",
            )

        odds_contract = load_feature_contract(
            _manifest(historical_odds=True)
        )
        with self.assertRaises(FeatureContractError):
            fit_gender_estimator(
                without_market,
                gender="M",
                kind="logistic",
                contract=odds_contract,
                profile="market_enhanced",
            )

        with_market = _training_frame(with_market=True)
        fitted = fit_gender_estimator(
            with_market,
            gender="M",
            kind="logistic",
            contract=odds_contract,
            profile="market_enhanced",
        )
        self.assertEqual(fitted.profile, "market_enhanced")

    def test_invalid_target_or_estimator_is_rejected(self) -> None:
        """Falla claramente ante etiqueta o nombre de modelo inválido."""

        train = _training_frame()
        contract = load_feature_contract(_manifest())
        invalid_y = train.copy()
        invalid_y.loc[0, "y"] = 2
        with self.assertRaises(EstimatorContractError):
            fit_gender_estimator(
                invalid_y,
                gender="M",
                kind="logistic",
                contract=contract,
            )
        with self.assertRaises(EstimatorContractError):
            fit_gender_estimator(
                train,
                gender="M",
                kind="tree",  # type: ignore[arg-type]
                contract=contract,
            )


if __name__ == "__main__":
    unittest.main()

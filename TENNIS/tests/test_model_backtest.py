"""Pruebas integradas y sintéticas del backtest temporal de fase 7."""

from __future__ import annotations

from dataclasses import replace
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
from src.modeling.backtest import (  # noqa: E402
    BacktestError,
    fit_final_gender_models,
    run_gender_backtest,
)
from src.modeling.evaluation import evaluate_gender_backtest  # noqa: E402
from src.modeling.parameters import (  # noqa: E402
    LightGBMParameters,
    LogisticParameters,
    TemporalEvaluationParameters,
)
from src.modeling.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    load_feature_contract,
)
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402


FEATURE_CONTRACT_FIXTURE = {
    "fingerprint": "f" * 64,
    "schema_version": FEATURE_SCHEMA_VERSION,
    "historical_odds_available": False,
    "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
    "training_columns": [
        "record_id",
        "gender",
        "match_date",
        "result_available_date",
        "rank_a",
        "rank_b",
        *MODEL_FEATURE_COLUMNS,
        "y",
    ],
}


def _synthetic_history(last_year: int) -> pd.DataFrame:
    """Crea temporadas completas pequeñas con todas las features deportivas."""

    contract = load_feature_contract(FEATURE_CONTRACT_FIXTURE)
    feature_columns = contract.columns_for("sports_only")
    rows: list[dict[str, object]] = []
    for year in range(2017, last_year + 1):
        for match_index in range(24):
            y = match_index % 2
            rank_a = 20 + match_index if y else 80 + match_index
            rank_b = 80 + match_index if y else 20 + match_index
            row: dict[str, object] = {
                "record_id": f"{year}-{match_index}",
                "gender": "M",
                "match_date": pd.Timestamp(
                    year=year,
                    month=1 + match_index // 20,
                    day=1 + match_index % 20,
                ),
                "rank_a": rank_a,
                "rank_b": rank_b,
                "market_probability_a": np.nan,
                "market_probability_b": np.nan,
                "y": y,
            }
            match_date = row["match_date"]
            assert isinstance(match_date, pd.Timestamp)
            row["result_available_date"] = (
                DEFAULT_SOURCE_DATE_POLICY.availability_date(
                    match_date.date()
                )
            )
            for column in feature_columns:
                if column == "surface":
                    value: object = "Hard" if match_index % 2 else "Clay"
                elif column == "tour_level":
                    value = "ATP Tour"
                elif column == "tour_level_raw":
                    value = "A"
                elif column == "round":
                    value = "R32" if match_index % 2 else "QF"
                elif column == "rank_diff":
                    value = rank_a - rank_b
                elif column in CATEGORICAL_FEATURE_COLUMNS:
                    value = "known"
                elif column.startswith(("elo_cold_start", "ranking_missing", "age_missing")):
                    value = False
                else:
                    value = float((2 * y - 1) * (1 + match_index % 5))
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


class ModelBacktestTest(unittest.TestCase):
    """Comprueba OOF, estabilidad futura y mercado ausente."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga una única vez el contrato real de columnas."""

        cls.contract = load_feature_contract(FEATURE_CONTRACT_FIXTURE)
        cls.logistic = LogisticParameters(max_iter=300)
        cls.lightgbm = LightGBMParameters(
            n_estimators=8,
            min_child_samples=5,
            n_jobs=1,
        )
        cls.evaluation = TemporalEvaluationParameters(
            first_test_season=2020,
            last_test_season=2020,
            small_segment_threshold=20,
        )

    def _run(self, last_year: int):
        """Ejecuta un único fold sintético con parámetros rápidos."""

        return run_gender_backtest(
            _synthetic_history(last_year),
            gender="M",
            contract=self.contract,
            profile="sports_only",
            evaluation_parameters=self.evaluation,
            logistic_parameters=self.logistic,
            lightgbm_parameters=self.lightgbm,
        )

    def test_future_append_does_not_change_existing_predictions(self) -> None:
        """Añadir 2021–2022 no altera el fold cuyo test es 2020."""

        original = self._run(2020).predictions
        extended = self._run(2022).predictions

        pd.testing.assert_frame_equal(original, extended, check_exact=True)

    def test_fold_predictions_are_unique_balanced_and_temporal(self) -> None:
        """Cada test aparece una vez y respeta train<calibración<test."""

        result = self._run(2020)

        self.assertEqual(len(result.predictions), 24)
        self.assertFalse(result.predictions["record_id"].duplicated().any())
        self.assertAlmostEqual(result.predictions["y"].mean(), 0.5)
        self.assertEqual(len(result.folds), 1)
        fold = result.folds[0]
        self.assertLess(fold.train_max_date, fold.calibration_min_date)
        self.assertLess(fold.calibration_max_date, fold.test_min_date)

    def test_evaluation_declares_market_not_evaluable(self) -> None:
        """Cobertura cero permanece visible y no crea soporte común falso."""

        result = self._run(2020)
        evaluation = evaluate_gender_backtest(
            result, parameters=self.evaluation
        )

        self.assertEqual(evaluation.market_audit.market_rows, 0)
        self.assertEqual(evaluation.market_audit.coverage, 0.0)
        self.assertIn("NO EVALUABLE", evaluation.market_audit.status)
        self.assertNotIn(
            "market_common_support",
            set(evaluation.metrics["population"]),
        )
        market_global = evaluation.metrics.loc[
            (evaluation.metrics["population"] == "native")
            & (evaluation.metrics["scope"] == "global")
            & (evaluation.metrics["model"] == "market_devig")
        ].iloc[0]
        self.assertEqual(market_global["n_evaluated"], 0)
        self.assertTrue(np.isnan(market_global["brier"]))

    def test_high_accuracy_audit_includes_season_segment_cells(self) -> None:
        """No oculta extremos que solo aparecen en una temporada pequeña."""

        result = self._run(2020)
        evaluation = evaluate_gender_backtest(
            result,
            parameters=self.evaluation,
        )
        temporal = evaluation.suspicious_segments.loc[
            evaluation.suspicious_segments["population"].eq(
                "native_temporal_cell"
            )
        ]

        self.assertFalse(temporal.empty)
        self.assertEqual(set(temporal["test_season"]), {2020})
        self.assertTrue(temporal["small_sample"].all())

    def test_final_calibrator_rejects_tampered_oof_labels(self) -> None:
        """El Platt final no acepta labels OOF ajenos al histórico causal."""

        history = _synthetic_history(2020)
        backtest = self._run(2020)
        changed = backtest.predictions.copy()
        changed.loc[0, "y"] = 1 - int(changed.loc[0, "y"])
        tampered = replace(backtest, predictions=changed)

        with self.assertRaisesRegex(BacktestError, "etiquetas OOF"):
            fit_final_gender_models(
                history,
                gender="M",
                contract=self.contract,
                backtest=tampered,
                profile="sports_only",
                logistic_parameters=self.logistic,
                lightgbm_parameters=self.lightgbm,
                training_as_of_date=pd.Timestamp("2021-03-01").date(),
            )

    def test_final_calibrator_rejects_missing_interior_oof_row(self) -> None:
        """No basta conservar extremos: cada ID del test debe reconciliar."""

        history = _synthetic_history(2020)
        backtest = self._run(2020)
        changed = backtest.predictions.drop(
            index=backtest.predictions.index[len(backtest.predictions) // 2]
        ).reset_index(drop=True)
        tampered = replace(backtest, predictions=changed)

        with self.assertRaisesRegex(BacktestError, "inventario OOF"):
            fit_final_gender_models(
                history,
                gender="M",
                contract=self.contract,
                backtest=tampered,
                profile="sports_only",
                logistic_parameters=self.logistic,
                lightgbm_parameters=self.lightgbm,
                training_as_of_date=pd.Timestamp("2021-03-01").date(),
            )


if __name__ == "__main__":
    unittest.main()

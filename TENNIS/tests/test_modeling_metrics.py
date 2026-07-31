"""Pruebas sintéticas de métricas, segmentos y fiabilidad."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.metrics import (  # noqa: E402
    MetricsError,
    audit_high_accuracy_segments,
    binary_classification_metrics,
    evaluate_prediction_columns,
    evaluate_predictions,
    reliability_curve_quantile,
)


class ProbabilisticMetricsTest(unittest.TestCase):
    """Comprueba fórmulas, cobertura y degradación para una sola clase."""

    def test_exact_metrics_and_missing_prediction_coverage(self) -> None:
        """Calcula métricas solo donde existe probabilidad."""

        metrics = binary_classification_metrics(
            [0, 1, 1, 0],
            [0.1, 0.8, np.nan, 0.6],
        )
        self.assertEqual(metrics["n_total"], 4)
        self.assertEqual(metrics["n_evaluated"], 3)
        self.assertAlmostEqual(metrics["coverage"], 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        expected_brier = (0.1**2 + (1 - 0.8) ** 2 + 0.6**2) / 3
        self.assertAlmostEqual(metrics["brier"], expected_brier)
        self.assertTrue(metrics["auc_defined"])

    def test_single_class_has_metrics_but_auc_is_explicitly_undefined(
        self,
    ) -> None:
        """No rompe ni inventa AUC cuando el segmento contiene una clase."""

        metrics = binary_classification_metrics(
            [1, 1, 1],
            [0.6, 0.8, 0.9],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertTrue(np.isfinite(metrics["log_loss"]))
        self.assertTrue(np.isfinite(metrics["brier"]))
        self.assertTrue(np.isnan(metrics["auc"]))
        self.assertFalse(metrics["auc_defined"])

    def test_invalid_targets_and_probabilities_fail(self) -> None:
        """Rechaza etiquetas no binarias y probabilidades fuera del rango."""

        with self.assertRaisesRegex(MetricsError, "0 o 1"):
            binary_classification_metrics([0, 2], [0.2, 0.8])
        with self.assertRaisesRegex(MetricsError, r"\[0, 1\]"):
            binary_classification_metrics([0, 1], [-0.1, 0.8])


class SegmentedMetricsTest(unittest.TestCase):
    """Verifica informes homogéneos y auditoría de alta accuracy."""

    def setUp(self) -> None:
        """Crea dos segmentos, uno deliberadamente perfecto y pequeño."""

        self.frame = pd.DataFrame(
            {
                "tour_level": [
                    "Grand Slam",
                    "Grand Slam",
                    "ITF",
                    "ITF",
                    "ITF",
                    "ITF",
                ],
                "surface": ["Grass", "Grass", "Clay", "Clay", "Clay", "Clay"],
                "y": [0, 1, 0, 1, 0, 1],
                "model_probability": [0.1, 0.9, 0.6, 0.4, 0.7, 0.3],
                "market_probability": [np.nan, np.nan, 0.4, 0.6, 0.3, 0.7],
            }
        )

    def test_global_and_each_level_surface_segment_are_reported(self) -> None:
        """Incluye global y todas las combinaciones nivel por superficie."""

        report = evaluate_predictions(
            self.frame,
            probability_column="model_probability",
            model_name="lightgbm",
        )
        self.assertEqual(len(report), 3)
        self.assertEqual(report.iloc[0]["scope"], "global")
        segments = report.loc[report["scope"] == "segment"]
        self.assertEqual(
            set(zip(segments["tour_level"], segments["surface"])),
            {("Grand Slam", "Grass"), ("ITF", "Clay")},
        )

    def test_multiple_models_preserve_market_coverage(self) -> None:
        """Evalúa modelos juntos sin ocultar la cobertura parcial del mercado."""

        report = evaluate_prediction_columns(
            self.frame,
            {
                "lightgbm": "model_probability",
                "market": "market_probability",
            },
        )
        market_global = report.loc[
            (report["model"] == "market")
            & (report["scope"] == "global")
        ].iloc[0]
        self.assertEqual(market_global["n_total"], 6)
        self.assertEqual(market_global["n_evaluated"], 4)
        self.assertAlmostEqual(market_global["coverage"], 4 / 6)

    def test_high_accuracy_audit_keeps_support_and_small_sample_flag(
        self,
    ) -> None:
        """Señala >85 % incluso cuando puede deberse a poco soporte."""

        report = evaluate_predictions(
            self.frame,
            probability_column="model_probability",
        )
        audit = audit_high_accuracy_segments(
            report,
            small_sample_threshold=10,
        )
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0]["tour_level"], "Grand Slam")
        self.assertEqual(audit.iloc[0]["n_evaluated"], 2)
        self.assertTrue(bool(audit.iloc[0]["small_sample"]))
        self.assertIn("precedencia temporal", audit.iloc[0]["audit_reason"])


class ReliabilityCurveTest(unittest.TestCase):
    """Comprueba bins por cuantiles y cobertura de curvas."""

    def test_quantile_bins_cover_every_available_prediction(self) -> None:
        """La suma de bins coincide con observaciones evaluadas."""

        curve = reliability_curve_quantile(
            [0, 0, 0, 1, 1, 1, 1],
            [0.05, 0.1, np.nan, 0.4, 0.6, 0.8, 0.95],
            n_bins=3,
        )
        self.assertEqual(curve["count"].sum(), 6)
        self.assertTrue((curve["n_total"] == 7).all())
        self.assertTrue((curve["n_evaluated"] == 6).all())
        self.assertTrue(
            (curve["mean_predicted"].diff().dropna() >= 0).all()
        )

    def test_constant_predictions_form_one_bin(self) -> None:
        """Evita bins artificiales cuando todas las probabilidades coinciden."""

        curve = reliability_curve_quantile(
            [0, 1, 0, 1],
            [0.5, 0.5, 0.5, 0.5],
            n_bins=10,
        )
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve.iloc[0]["count"], 4)
        self.assertAlmostEqual(curve.iloc[0]["observed_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

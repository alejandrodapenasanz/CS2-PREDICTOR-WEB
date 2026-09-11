"""Prueba end-to-end pequeña del reentreno, publicación e idempotencia."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
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
from src.artifact_integrity import build_code_inventory  # noqa: E402
from src.features.dataset import FEATURE_CODE_PATHS  # noqa: E402
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402
from src.modeling.parameters import (  # noqa: E402
    LightGBMParameters,
    LogisticParameters,
    TemporalEvaluationParameters,
)
from src.modeling.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
)
from src.modeling.service import (  # noqa: E402
    ModelServiceError,
    load_active_deployment_model,
)
from src.modeling.training import retrain_models  # noqa: E402
from src.modeling.artifacts import MODEL_CODE_PATHS  # noqa: E402


def _sha256(path: Path) -> str:
    """Calcula el hash de un Parquet sintético."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gender_frame(gender: str) -> pd.DataFrame:
    """Crea cuatro temporadas pequeñas con el esquema predictivo completo."""

    rows: list[dict[str, object]] = []
    for year in range(2017, 2021):
        for index in range(16):
            target = index % 2
            rank_a = 10 + index if target else 70 + index
            rank_b = 70 + index if target else 10 + index
            row: dict[str, object] = {
                "record_id": f"{gender}-{year}-{index}",
                "gender": gender,
                "match_date": date(year, 1, index + 1),
                "result_available_date": (
                    DEFAULT_SOURCE_DATE_POLICY.availability_date(
                        date(year, 1, index + 1)
                    )
                ),
                "rank_a": rank_a,
                "rank_b": rank_b,
                "y": target,
            }
            for column in MODEL_FEATURE_COLUMNS:
                if column == "surface":
                    value: object = "Hard" if index % 2 else "Clay"
                elif column == "tour_level":
                    value = "ATP Tour" if gender == "M" else "WTA Tour"
                elif column == "tour_level_raw":
                    value = "A" if gender == "M" else "W"
                elif column == "round":
                    value = "R32"
                elif column == "rank_diff":
                    value = rank_a - rank_b
                elif column in MARKET_FEATURE_COLUMNS:
                    value = np.nan
                elif column in CATEGORICAL_FEATURE_COLUMNS:
                    value = "known"
                elif column.startswith(
                    ("elo_cold_start", "ranking_missing", "age_missing")
                ):
                    value = False
                else:
                    value = float((2 * target - 1) * (1 + index % 4))
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _write_feature_snapshot(root: Path) -> Path:
    """Publica dos Parquet y un manifiesto mínimo íntegro."""

    datasets: list[dict[str, object]] = []
    for gender in ("M", "F"):
        path = root / f"training_{gender}.parquet"
        frame = _gender_frame(gender)
        frame.to_parquet(path, index=False)
        datasets.append(
            {
                "gender": gender,
                "output_path": path.name,
                "output_size": path.stat().st_size,
                "output_sha256": _sha256(path),
                "training_rows": len(frame),
                "min_date": "2017-01-01",
                "max_date": "2020-01-16",
            }
        )
    manifest = {
        "fingerprint": "d" * 64,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source_commit": "c" * 40,
        "historical_odds_available": False,
        "source_date_policy": DEFAULT_SOURCE_DATE_POLICY.as_dict(),
        "code_inventory": list(
            build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS)
        ),
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
        "datasets": datasets,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest_path


class ModelTrainingTest(unittest.TestCase):
    """Comprueba el flujo canónico sin tocar los artefactos reales."""

    def test_small_run_is_complete_versioned_and_idempotent(self) -> None:
        """Publica ambos géneros y reutiliza exactamente el mismo run."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            root = Path(temporary)
            manifest = _write_feature_snapshot(root)
            output = root / "models"
            kwargs = {
                "manifest_path": manifest,
                "output_dir": output,
                "profile": "sports_only",
                "logistic_parameters": LogisticParameters(max_iter=300),
                "lightgbm_parameters": LightGBMParameters(
                    n_estimators=5,
                    min_child_samples=4,
                    n_jobs=1,
                ),
                "evaluation_parameters": TemporalEvaluationParameters(
                    first_test_season=2020,
                    last_test_season=2020,
                    small_segment_threshold=20,
                ),
                "training_as_of_date": date(2021, 3, 1),
            }

            first = retrain_models(**kwargs)
            second = retrain_models(**kwargs)

            self.assertFalse(first.skipped)
            self.assertTrue(second.skipped)
            self.assertEqual(first.promotion.action, "bootstrapped")
            self.assertEqual(second.promotion.action, "active_reused")
            self.assertFalse(first.retention.applied)
            self.assertEqual(
                [
                    item["path"]
                    for item in first.published.manifest["code_inventory"]
                ],
                sorted(MODEL_CODE_PATHS),
            )
            self.assertEqual(
                first.published.fingerprint,
                second.published.fingerprint,
            )
            self.assertEqual(
                {item["gender"] for item in first.summaries}, {"M", "F"}
            )
            self.assertTrue(
                (
                    first.published.run_dir
                    / "models"
                    / "M"
                    / "deployment_bundle.joblib"
                ).is_file()
            )
            self.assertTrue(
                (
                    first.published.run_dir
                    / "models"
                    / "F"
                    / "deployment_bundle.joblib"
                ).is_file()
            )
            self.assertTrue(
                (
                    first.published.run_dir
                    / "report"
                    / "model_report.md"
                ).is_file()
            )
            self.assertEqual(len(first.market_audits), 2)
            self.assertTrue(
                (first.market_audits["market_rows"] == 0).all()
            )
            active = load_active_deployment_model(
                "M", manifest_path=output / "manifest.json"
            )
            prediction = active.predict(
                _gender_frame("M").tail(2),
                as_of_date=date(2025, 1, 1),
            )
            self.assertTrue(
                prediction["model_probability_a"].between(0.0, 1.0).all()
            )
            self.assertTrue(prediction["edge"].isna().all())
            with self.assertRaisesRegex(ModelServiceError, "no es causal"):
                active.predict(
                    _gender_frame("M").tail(1),
                    as_of_date=date.fromisoformat(active.training_max_date),
                )
            unverified_market = _gender_frame("M").tail(1).copy()
            unverified_market["market_probability_a"] = 0.55
            unverified_market["market_probability_b"] = 0.45
            with self.assertRaisesRegex(
                ModelServiceError, "market_retrieved_at_utc"
            ):
                active.predict(
                    unverified_market,
                    as_of_date=date(2025, 1, 1),
                )

            published_manifest = first.published.run_dir / "manifest.json"
            payload = json.loads(
                published_manifest.read_text(encoding="utf-8")
            )
            inference_entry = next(
                item
                for item in payload["code_inventory"]
                if item["path"] == "src/modeling/calibration.py"
            )
            inference_entry["sha256"] = "0" * 64
            published_manifest.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ModelServiceError, "código distinto"):
                load_active_deployment_model(
                    "M", manifest_path=output / "manifest.json"
                )


if __name__ == "__main__":
    unittest.main()

"""Pruebas offline de la CLI reproducible de reentreno."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrain_models import (  # noqa: E402
    build_parser,
    infer_last_complete_season,
)
from src.features import MODEL_FEATURE_COLUMNS  # noqa: E402
from src.artifact_integrity import build_code_inventory  # noqa: E402
from src.features.dataset import FEATURE_CODE_PATHS  # noqa: E402
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_DATASET_MANIFEST_PATH,
    PHASE7_MODELS_DIR,
)


class RetrainModelsScriptTest(unittest.TestCase):
    """Comprueba defaults y resolución de temporadas sin entrenar."""

    def test_defaults_use_canonical_paths_and_auto_profile(self) -> None:
        """La ejecución normal construye ambos modelos en rutas centrales."""

        args = build_parser().parse_args([])

        self.assertEqual(args.manifest, FEATURE_DATASET_MANIFEST_PATH)
        self.assertEqual(args.output_dir, PHASE7_MODELS_DIR)
        self.assertEqual(args.feature_profile, "auto")
        self.assertEqual(args.first_test_season, 2016)
        self.assertIsNone(args.last_test_season)
        self.assertEqual(args.n_jobs, -1)

    def test_explicit_temporal_protocol_and_threads_are_preserved(self) -> None:
        """No adapta silenciosamente temporadas ni paralelismo declarados."""

        args = build_parser().parse_args(
            [
                "--feature-profile",
                "sports_only",
                "--first-test-season",
                "2018",
                "--last-test-season",
                "2024",
                "--n-jobs",
                "3",
            ]
        )

        self.assertEqual(args.feature_profile, "sports_only")
        self.assertEqual(args.first_test_season, 2018)
        self.assertEqual(args.last_test_season, 2024)
        self.assertEqual(args.n_jobs, 3)

    def test_inferred_end_never_uses_current_incomplete_season(self) -> None:
        """El default se limita a una temporada cerrada y disponible."""

        payload = {
            "fingerprint": "f" * 64,
            "schema_version": "tennis-features-v2",
            "source_commit": "a" * 40,
            "historical_odds_available": False,
            "source_date_policy": DEFAULT_SOURCE_DATE_POLICY.as_dict(),
            "code_inventory": list(
                build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS)
            ),
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "datasets": [
                {"gender": "M", "max_date": "2026-06-30"},
                {"gender": "F", "max_date": "2025-12-31"},
            ],
        }
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            training_as_of_date = date(2026, 8, 9)
            inferred = infer_last_complete_season(
                manifest_path,
                training_as_of_date,
            )
        latest_common = min(
            date.fromisoformat(item["max_date"]).year
            for item in payload["datasets"]
            if item["gender"] in {"M", "F"}
        )

        self.assertLessEqual(inferred, training_as_of_date.year - 1)
        self.assertEqual(
            inferred, min(latest_common, training_as_of_date.year - 1)
        )


if __name__ == "__main__":
    unittest.main()

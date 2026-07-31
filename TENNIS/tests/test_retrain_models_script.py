"""Pruebas offline de la CLI reproducible de reentreno."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrain_models import (  # noqa: E402
    build_parser,
    infer_last_complete_season,
)
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

        inferred = infer_last_complete_season(
            FEATURE_DATASET_MANIFEST_PATH
        )
        payload = json.loads(
            FEATURE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        latest_common = min(
            date.fromisoformat(item["max_date"]).year
            for item in payload["datasets"]
            if item["gender"] in {"M", "F"}
        )

        self.assertLessEqual(inferred, date.today().year - 1)
        self.assertEqual(
            inferred, min(latest_common, date.today().year - 1)
        )


if __name__ == "__main__":
    unittest.main()

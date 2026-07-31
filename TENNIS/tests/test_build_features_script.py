"""Pruebas offline de la CLI de construcción de features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_features import build_parser  # noqa: E402
from src.config import FEATURES_PROCESSED_DIR  # noqa: E402


class BuildFeaturesScriptTest(unittest.TestCase):
    """Comprueba defaults y argumentos sin ejecutar el histórico."""

    def test_defaults_build_both_genders_in_centralized_directory(self) -> None:
        """Usa all, rutas configuradas y tamaños positivos documentados."""

        args = build_parser().parse_args([])

        self.assertEqual(args.gender, "all")
        self.assertEqual(args.output_dir, FEATURES_PROCESSED_DIR)
        self.assertGreater(args.chunksize, 0)
        self.assertGreater(args.parquet_buffer_rows, 0)
        self.assertFalse(args.force)

    def test_gender_force_and_buffers_are_explicit(self) -> None:
        """Acepta una reconstrucción diagnóstica de un único universo."""

        args = build_parser().parse_args(
            [
                "--gender",
                "F",
                "--chunksize",
                "123",
                "--parquet-buffer-rows",
                "45",
                "--force",
            ]
        )

        self.assertEqual(args.gender, "F")
        self.assertEqual(args.chunksize, 123)
        self.assertEqual(args.parquet_buffer_rows, 45)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main()

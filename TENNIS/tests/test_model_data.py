"""Pruebas offline de la verificación de datasets para modelado."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import MODEL_FEATURE_COLUMNS  # noqa: E402
from src.modeling.data import (  # noqa: E402
    ModelDataError,
    load_training_dataset,
)


def _sha256(path: Path) -> str:
    """Calcula SHA-256 de un fixture pequeño."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path, *, corrupt_hash: bool = False) -> Path:
    """Publica un manifiesto y Parquet mínimos coherentes dentro de tests."""

    dataset_path = root / "training_M.parquet"
    frame = pd.DataFrame(
        {
            "record_id": ["later", "earlier"],
            "gender": ["M", "M"],
            "match_date": [
                date(2020, 1, 2),
                date(2020, 1, 1),
            ],
            "tour_level": ["ATP Tour", "ATP Tour"],
            "surface": ["Hard", "Clay"],
            "rank_a": [1, 2],
            "rank_b": [2, 1],
            "market_probability_a": [None, None],
            "market_probability_b": [None, None],
            "elo_general_diff": [10.0, -5.0],
            "y": [1, 0],
        }
    )
    frame.to_parquet(dataset_path, index=False)
    digest = "0" * 64 if corrupt_hash else _sha256(dataset_path)
    manifest = {
        "fingerprint": "f" * 64,
        "schema_version": "tennis-features-v1",
        "source_commit": "a" * 40,
        "historical_odds_available": False,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "datasets": [
            {
                "gender": "M",
                "output_path": dataset_path.name,
                "output_size": dataset_path.stat().st_size,
                "output_sha256": digest,
                "training_rows": 2,
                "min_date": "2020-01-01",
                "max_date": "2020-01-02",
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest_path


class ModelDataTest(unittest.TestCase):
    """Comprueba integridad, allowlist y orden cronológico."""

    def test_loads_verified_columns_and_sorts_by_date(self) -> None:
        """Entrega referencias y features en orden temporal estable."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = _write_fixture(Path(temporary))

            loaded = load_training_dataset(
                "M",
                feature_columns=("elo_general_diff",),
                manifest_path=manifest_path,
            )

        self.assertEqual(loaded.metadata.rows, 2)
        self.assertEqual(
            loaded.frame["record_id"].tolist(), ["earlier", "later"]
        )
        self.assertEqual(
            loaded.frame["elo_general_diff"].tolist(), [-5.0, 10.0]
        )

    def test_rejects_artifact_whose_hash_changed(self) -> None:
        """No entrena si el Parquet diverge del manifiesto de fase 6."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = _write_fixture(
                Path(temporary), corrupt_hash=True
            )

            with self.assertRaisesRegex(ModelDataError, "SHA-256"):
                load_training_dataset(
                    "M",
                    feature_columns=("elo_general_diff",),
                    manifest_path=manifest_path,
                )

    def test_rejects_feature_outside_published_allowlist(self) -> None:
        """Impide que un identificador entre en la matriz predictora."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = _write_fixture(Path(temporary))

            with self.assertRaisesRegex(ModelDataError, "allowlist"):
                load_training_dataset(
                    "M",
                    feature_columns=("player_a_id",),
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()

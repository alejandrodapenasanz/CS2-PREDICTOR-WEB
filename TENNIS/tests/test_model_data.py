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
from src.artifact_integrity import build_code_inventory  # noqa: E402
from src.features.dataset import FEATURE_CODE_PATHS  # noqa: E402
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402
from src.features.artifacts import (  # noqa: E402
    preflight_feature_publication,
    publish_feature_run,
)
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
            "result_available_date": [
                date(2020, 1, 23),
                date(2020, 1, 22),
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
    conflicts_path = root / "ranking_conflicts.csv"
    conflicts_path.write_text("gender,player_id\n", encoding="utf-8")
    manifest = {
        "created_at_utc": "2026-08-01T00:00:00+00:00",
        "fingerprint": "f" * 64,
        "schema_version": "tennis-features-v1",
        "source_commit": "a" * 40,
        "historical_odds_available": False,
        "source_date_policy": DEFAULT_SOURCE_DATE_POLICY.as_dict(),
        "code_inventory": list(
            build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS)
        ),
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
        "conflict_inventory": {
            "path": conflicts_path.name,
            "size": conflicts_path.stat().st_size,
            "sha256": _sha256(conflicts_path),
            "rows": 0,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest_path


def _write_generational_fixture(root: Path) -> Path:
    """Publica el fixture mediante un run inmutable y devuelve su puntero."""

    output = root / "features_active"
    preflight_feature_publication(output_dir=output)
    publish = output / ".feature-build-model-data" / "publish"
    publish.mkdir(parents=True)
    _write_fixture(publish)
    publish_feature_run(
        publish,
        fingerprint="f" * 64,
        output_dir=output,
    )
    return output / "manifest.json"


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

    def test_loads_through_verified_active_generation_pointer(self) -> None:
        """Sigue el puntero y resuelve el Parquet relativo al run inmutable."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = _write_generational_fixture(Path(temporary))

            loaded = load_training_dataset(
                "M",
                feature_columns=("elo_general_diff",),
                manifest_path=manifest_path,
            )

        self.assertEqual(loaded.metadata.rows, 2)
        self.assertEqual(loaded.source_manifest.path.parent.name, "f" * 64)

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

    def test_rejects_manifest_built_by_different_code_inventory(self) -> None:
        """Falla cerrado si cambia un hash de código del productor."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            manifest_path = _write_fixture(Path(temporary))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["code_inventory"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ModelDataError,
                "código distinto",
            ):
                load_training_dataset(
                    "M",
                    feature_columns=("elo_general_diff",),
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()

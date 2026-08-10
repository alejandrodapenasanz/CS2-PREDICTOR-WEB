"""Pruebas del contrato entre SQLite operativa y la pestana web de tenis."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest

import pandas as pd

from src.config import BBDD_DIR
from src.operations import OperationsStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEB_BUILDER_PATH = REPOSITORY_ROOT / "WEB" / "build_web.py"
FIXED_NOW = datetime(2026, 8, 2, 8, 0, tzinfo=UTC)


def _load_web_builder() -> ModuleType:
    """Carga el constructor web sin ejecutar su main."""

    spec = importlib.util.spec_from_file_location(
        "tennis_web_builder_test", WEB_BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    """Calcula el hash para detectar cualquier escritura del lector web."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_exact_schema_database(path: Path) -> None:
    """Persiste una prediccion oficial y su settlement mediante la API real."""

    prediction = pd.DataFrame(
        [
            {
                "source_match_id": "3281237",
                "prediction_date": "2026-08-02",
                "prediction_as_of_utc": "2026-08-02T08:00:00Z",
                "source_retrieved_at_utc": "2026-08-02T07:58:00Z",
                "source_snapshot_sha256": "a" * 64,
                "tournament": "ATP Toronto",
                "tour_level": "M",
                "gender": "M",
                "surface": "Hard",
                "scheduled_time": "10:30",
                "status": "scheduled",
                "player_a_name": "Carlos Alcaraz",
                "player_b_name": "Jannik Sinner",
                "player_a_slug": "alcaraz",
                "player_b_slug": "sinner",
                "mapping_status": "mapped",
                "model_probability_raw_a": 0.70,
                "model_probability_a": 0.72,
                "model_probability_b": 0.28,
                "confidence": "LOW",
                "confidence_flags": (
                    "source_snapshot_stale|market_prestart_unverified"
                ),
                "prediction_status": "predicted",
                "model_profile": "sports_only",
                "model_fingerprint": "model-v2",
                "model_training_max_date": "2026-06-30",
                "model_training_available_max_date": "2026-07-21",
                "feature_fingerprint": "features-v2",
                "predicted_winner_name": "Carlos Alcaraz",
                "predicted_winner_probability": 0.72,
            }
        ]
    )
    result = pd.DataFrame(
        [
            {
                "source_match_id": "3281237",
                "status": "finished",
                "player_1_sets_won": 1,
                "player_2_sets_won": 2,
                "sets_score": "6-4 3-6 6-3",
                "winner_side": "player_2",
                "winner_slug": "alcaraz",
                "result_evidence": "terminal_sets_and_slug",
                "observed_at_utc": "2026-08-02T14:00:00Z",
                "source_snapshot_sha256": "b" * 64,
            }
        ]
    )
    with OperationsStore(path, clock=lambda: FIXED_NOW) as store:
        store.register_prediction_run(
            "prediction:2026-08-02T080000Z",
            prediction,
            target_date="2026-08-02",
        )
        store.reconcile_observations(
            "observation:2026-08-02T140000Z",
            result,
            observed_at_utc="2026-08-02T14:00:00Z",
        )


class WebTennisIntegrationTests(unittest.TestCase):
    """Verifica la publicacion web contra la base operativa versionada."""

    def test_reads_schema_without_mutating_database(self) -> None:
        """Publica calibracion, fiabilidad y ganador real por separado."""

        builder = _load_web_builder()
        BBDD_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=BBDD_DIR) as directory:
            database_path = Path(directory) / "web-fixture.sqlite3"
            _build_exact_schema_database(database_path)
            before = _file_sha256(database_path)
            builder.TENNIS_DB_PATH = database_path

            payload = builder.tennis_payload()

            self.assertEqual(_file_sha256(database_path), before)
            self.assertEqual(payload["health"], "ok")
            self.assertEqual(
                payload["latest_run"]["run_id"],
                "prediction:2026-08-02T080000Z",
            )
            match = payload["matches"][0]
            self.assertEqual(
                match["predicted"]["winner"], "Carlos Alcaraz"
            )
            self.assertEqual(match["predicted"]["probability"], 0.72)
            self.assertEqual(match["reliability"]["label"], "LOW")
            self.assertEqual(
                match["reliability"]["flags"],
                ["source_snapshot_stale", "market_prestart_unverified"],
            )
            self.assertEqual(match["actual"]["winner"], "Carlos Alcaraz")
            self.assertEqual(match["actual"]["winner_side"], "A")

    def test_tolerates_absent_database(self) -> None:
        """Una base ausente produce un payload vacio sin romper la web."""

        builder = _load_web_builder()
        builder.TENNIS_DB_PATH = (
            BBDD_DIR / "database-that-does-not-exist.sqlite3"
        )

        payload = builder.tennis_payload()

        self.assertEqual(payload["health"], "missing")
        self.assertEqual(payload["matches"], [])
        self.assertFalse(payload["database"]["exists"])

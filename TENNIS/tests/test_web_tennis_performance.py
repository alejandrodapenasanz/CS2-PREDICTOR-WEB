"""Contrato web para rendimiento real y confianza de inputs de tenis.

Las pruebas construyen SQLite temporales mediante la API operativa real. No
leen ni modifican ``TENNIS/BBDD/tennis.sqlite3`` y verifican además que el
lector web mantenga inmutable cada fixture durante la consulta.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest

import pandas as pd

from src.config import BBDD_DIR
from src.operations import OperationsStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEB_BUILDER_PATH = REPOSITORY_ROOT / "WEB" / "build_web.py"
WEB_INDEX_PATH = REPOSITORY_ROOT / "WEB" / "index.html"
FIXED_NOW = datetime(2026, 8, 3, 8, 0, tzinfo=UTC)
SETTLED_COUNT = 72
CORRECT_COUNT = 51
MODEL_PROBABILITY_A = 0.70


def _load_web_builder() -> ModuleType:
    """Carga ``WEB/build_web.py`` sin ejecutar su punto de entrada."""

    spec = importlib.util.spec_from_file_location(
        "tennis_performance_web_builder_test",
        WEB_BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    """Calcula el hash binario de una SQLite para detectar escrituras."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_row(
    *,
    source_match_id: str,
    prediction_date: str,
    player_number: int,
) -> dict[str, object]:
    """Construye una predicción oficial determinista para el fixture."""

    return {
        "source_match_id": source_match_id,
        "prediction_date": prediction_date,
        "prediction_as_of_utc": f"{prediction_date}T08:00:00Z",
        "source_retrieved_at_utc": f"{prediction_date}T07:58:00Z",
        "source_snapshot_sha256": "c" * 64,
        "tournament": "Fixture tournament",
        "tour_level": "A",
        "gender": "M",
        "surface": "Hard",
        "scheduled_time": "10:30",
        "scheduled_start_utc": f"{prediction_date}T10:30:00Z",
        "status": "scheduled",
        "player_a_name": f"Player A {player_number}",
        "player_b_name": f"Player B {player_number}",
        "player_a_slug": f"player-a-{player_number}",
        "player_b_slug": f"player-b-{player_number}",
        "mapping_status": "mapped",
        "model_probability_raw_a": MODEL_PROBABILITY_A,
        "model_probability_a": MODEL_PROBABILITY_A,
        "model_probability_b": 1.0 - MODEL_PROBABILITY_A,
        "confidence": "LOW",
        "confidence_flags": "history_stale|round_missing",
        "prediction_status": "predicted",
        "model_profile": "sports_only",
        "model_fingerprint": "model-performance-fixture",
        "model_training_max_date": "2026-06-30",
        "model_training_available_max_date": "2026-07-21",
        "feature_fingerprint": "features-performance-fixture",
        "predicted_winner_name": f"Player A {player_number}",
        "predicted_winner_probability": MODEL_PROBABILITY_A,
    }


def _result_row(
    *,
    source_match_id: str,
    player_number: int,
    prediction_was_correct: bool,
) -> dict[str, object]:
    """Construye un resultado terminal con ganador probado por slug."""

    winner_slug = (
        f"player-a-{player_number}" if prediction_was_correct else f"player-b-{player_number}"
    )
    return {
        "source_match_id": source_match_id,
        "status": "finished",
        "player_1_sets_won": 2 if prediction_was_correct else 0,
        "player_2_sets_won": 0 if prediction_was_correct else 2,
        "sets_score": "2-0" if prediction_was_correct else "0-2",
        "winner_side": ("player_1" if prediction_was_correct else "player_2"),
        "winner_slug": winner_slug,
        "result_evidence": "terminal_sets_and_slug",
        "observed_at_utc": "2026-08-01T14:00:00Z",
        "source_snapshot_sha256": "d" * 64,
    }


def _register_prediction_run(
    store: OperationsStore,
    *,
    run_id: str,
    prediction_date: str,
    first_player_number: int,
    count: int,
) -> None:
    """Registra un run de predicciones oficiales aún sin resultados."""

    predictions = pd.DataFrame(
        [
            _prediction_row(
                source_match_id=(f"fixture-{prediction_date}-{player_number:03d}"),
                prediction_date=prediction_date,
                player_number=player_number,
            )
            for player_number in range(
                first_player_number,
                first_player_number + count,
            )
        ]
    )
    store.register_prediction_run(
        run_id,
        predictions,
        target_date=prediction_date,
    )


def _build_historical_performance_database(path: Path) -> None:
    """Crea 72 settlements/51 aciertos y un último día no liquidado."""

    historical_results = pd.DataFrame(
        [
            _result_row(
                source_match_id=f"fixture-2026-08-01-{number:03d}",
                player_number=number,
                prediction_was_correct=number < CORRECT_COUNT,
            )
            for number in range(SETTLED_COUNT)
        ]
    )
    with OperationsStore(path, clock=lambda: FIXED_NOW) as store:
        _register_prediction_run(
            store,
            run_id="prediction:historical",
            prediction_date="2026-08-01",
            first_player_number=0,
            count=SETTLED_COUNT,
        )
        store.reconcile_observations(
            "observation:historical",
            historical_results,
            observed_at_utc="2026-08-01T14:00:00Z",
            target_date="2026-08-01",
        )
        _register_prediction_run(
            store,
            run_id="prediction:latest-unsettled",
            prediction_date="2026-08-02",
            first_player_number=999,
            count=1,
        )


def _build_unsettled_database(path: Path) -> None:
    """Crea una predicción oficial sin ninguna liquidación posterior."""

    with OperationsStore(path, clock=lambda: FIXED_NOW) as store:
        _register_prediction_run(
            store,
            run_id="prediction:no-sample",
            prediction_date="2026-08-02",
            first_player_number=999,
            count=1,
        )


class WebTennisPerformanceTests(unittest.TestCase):
    """Fija el significado de rendimiento real y confianza operativa."""

    def test_uses_all_historical_official_settlements_only(self) -> None:
        """El último día abierto no oculta ni entra en las 72 liquidadas."""

        builder = _load_web_builder()
        BBDD_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=BBDD_DIR) as directory:
            database_path = Path(directory) / "performance.sqlite3"
            _build_historical_performance_database(database_path)
            before = _file_sha256(database_path)
            builder.TENNIS_DB_PATH = database_path

            payload = builder.tennis_payload()

            self.assertEqual(_file_sha256(database_path), before)
            self.assertEqual(payload["latest_run"]["date"], "2026-08-02")
            self.assertEqual(len(payload["matches"]), 1)
            self.assertIsNone(payload["matches"][0]["actual"])
            performance = payload["settled_performance"]
            self.assertEqual(
                performance["scope"],
                "all_official_settled_predictions",
            )
            self.assertEqual(performance["status"], "computed")
            self.assertEqual(performance["settled_count"], SETTLED_COUNT)
            self.assertEqual(performance["evaluated_count"], SETTLED_COUNT)
            self.assertEqual(performance["correct"], CORRECT_COUNT)
            self.assertEqual(
                datetime.fromisoformat(performance["latest_settled_at_utc"]),
                FIXED_NOW,
            )
            self.assertAlmostEqual(
                performance["accuracy"],
                CORRECT_COUNT / SETTLED_COUNT,
            )
            expected_brier = (
                CORRECT_COUNT * (1.0 - MODEL_PROBABILITY_A) ** 2
                + (SETTLED_COUNT - CORRECT_COUNT) * MODEL_PROBABILITY_A**2
            ) / SETTLED_COUNT
            expected_log_loss = (
                -(
                    CORRECT_COUNT * math.log(MODEL_PROBABILITY_A)
                    + (SETTLED_COUNT - CORRECT_COUNT) * math.log(1.0 - MODEL_PROBABILITY_A)
                )
                / SETTLED_COUNT
            )
            self.assertAlmostEqual(performance["brier"], expected_brier)
            self.assertAlmostEqual(performance["log_loss"], expected_log_loss)
            self.assertTrue(performance["provisional"])
            self.assertGreater(
                performance["provisional_threshold"],
                performance["evaluated_count"],
            )
            calibration = performance["calibration"]
            self.assertIsInstance(calibration["method"], str)
            self.assertGreater(calibration["requested_bins"], 0)
            self.assertTrue(calibration["bins"])
            self.assertEqual(
                [option["value"] for option in performance["filter_options"]["probabilities"]],
                [
                    "all",
                    "50_60",
                    "60_70",
                    "70_80",
                    "80_90",
                    "90_100",
                ],
            )
            male_band = performance["segments"]["M|70_80"]
            self.assertEqual(male_band["settled_count"], SETTLED_COUNT)
            self.assertEqual(male_band["evaluated_count"], SETTLED_COUNT)
            self.assertEqual(male_band["correct"], CORRECT_COUNT)
            self.assertAlmostEqual(
                male_band["accuracy"],
                CORRECT_COUNT / SETTLED_COUNT,
            )
            self.assertEqual(
                performance["segments"]["M|60_70"]["settled_count"],
                0,
            )
            self.assertEqual(
                performance["segments"]["F|70_80"]["status"],
                "not_computed",
            )

            match = payload["matches"][0]
            self.assertEqual(
                match["input_confidence"],
                {
                    "level": "LOW",
                    "flags": ["history_stale", "round_missing"],
                },
            )
            self.assertNotIn("reliability", match)

    def test_empty_settlement_sample_is_null_not_zero_accuracy(self) -> None:
        """Sin settlements, las métricas son nulas y los bins quedan vacíos."""

        builder = _load_web_builder()
        BBDD_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=BBDD_DIR) as directory:
            database_path = Path(directory) / "no-sample.sqlite3"
            _build_unsettled_database(database_path)
            before = _file_sha256(database_path)
            builder.TENNIS_DB_PATH = database_path

            payload = builder.tennis_payload()

            self.assertEqual(_file_sha256(database_path), before)
            performance = payload["settled_performance"]
            self.assertEqual(performance["status"], "not_computed")
            self.assertEqual(
                performance["scope"],
                "all_official_settled_predictions",
            )
            self.assertEqual(performance["settled_count"], 0)
            self.assertEqual(performance["evaluated_count"], 0)
            self.assertEqual(performance["correct"], 0)
            self.assertIsNone(performance["latest_settled_at_utc"])
            self.assertIsNone(performance["accuracy"])
            self.assertIsNone(performance["brier"])
            self.assertIsNone(performance["log_loss"])
            self.assertEqual(performance["calibration"]["bins"], [])
            self.assertEqual(len(performance["segments"]), 18)
            self.assertTrue(
                all(
                    segment["status"] == "not_computed"
                    for segment in performance["segments"].values()
                )
            )
            match = payload["matches"][0]
            self.assertEqual(match["input_confidence"]["level"], "LOW")
            self.assertNotIn("score", match["input_confidence"])
            self.assertNotIn("reliability", match)


class WebTennisStaticContractTests(unittest.TestCase):
    """Impide regresiones semánticas en el frontend estático de tenis."""

    @classmethod
    def setUpClass(cls) -> None:
        """Lee una sola vez el HTML real que se publica en producción."""

        cls.html = WEB_INDEX_PATH.read_text(encoding="utf-8")

    def test_input_confidence_is_qualitative_and_never_a_fake_score(self) -> None:
        """La tabla usa categorías y no convierte un nulo en cero por ciento."""

        self.assertIn("Confianza del input", self.html)
        for label in ("ALTA", "MEDIA", "BAJA", "NO DISPONIBLE"):
            with self.subTest(label=label):
                self.assertIn(label, self.html)
        self.assertIn("row.input_confidence", self.html)
        self.assertNotIn("Number(reliabilityData.score)", self.html)
        self.assertNotIn("row.reliability", self.html)
        self.assertNotIn("row?.reliability", self.html)

    def test_settled_performance_has_explicit_empty_state_and_curve(self) -> None:
        """La vista consume todo el contrato real y dibuja su calibración."""

        self.assertIn("Acierto real (partidos liquidados)", self.html)
        self.assertIn("SIN MUESTRA / NO CALCULADO", self.html)
        self.assertIn("TENNIS.settled_performance", self.html)
        self.assertIn('id="tennisCalibration"', self.html)
        self.assertIn("bin.mean_predicted", self.html)
        self.assertIn("bin.observed_rate", self.html)

    def test_probability_header_sorts_tennis_rows_with_missing_values_last(self) -> None:
        """La cabecera de probabilidad controla un orden estable y accesible."""

        self.assertIn('class="tennis-sortable"', self.html)
        self.assertIn('data-tennis-sort="probability"', self.html)
        self.assertIn("function setTennisSort(key)", self.html)
        self.assertIn("function sortedTennisRows()", self.html)
        self.assertIn("if(a === null) return 1;", self.html)
        self.assertIn('state.tennisSortDir = "desc"', self.html)
        self.assertIn('event.key === "Enter"', self.html)

    def test_dashboard_displays_latest_incorporated_settlement(self) -> None:
        """El panel deja visible cuándo cambió por última vez su muestra."""

        self.assertIn("source.latest_settled_at_utc", self.html)
        self.assertIn("Última liquidación incorporada", self.html)

    def test_combined_performance_filters_refresh_every_metric_and_curve(self) -> None:
        """Género y franja comparten un único segmento agregado."""

        self.assertIn('id="tennisPerformanceGender"', self.html)
        self.assertIn('id="tennisPerformanceProbability"', self.html)
        self.assertIn('value="70_80"', self.html)
        self.assertIn('value="90_100"', self.html)
        self.assertIn(
            "`${state.tennisPerformanceGender}|${state.tennisPerformanceProbability}`",
            self.html,
        )
        self.assertIn("root.segments?.[segmentKey]", self.html)
        self.assertIn("renderTennisPerformance();", self.html)
        self.assertIn('id="tennisPerformanceReset"', self.html)


class WebModelBadgeTests(unittest.TestCase):
    """El badge compartido consume metadatos, no el pickle de producción."""

    def test_reads_live_registry_metadata_without_deserializing_model(self) -> None:
        """Un pickle ilegible no oculta un puntero live con metadata válida."""

        builder = _load_web_builder()
        BBDD_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=BBDD_DIR) as directory:
            model_root = Path(directory) / "MODEL"
            registry = model_root / "artifacts" / "registry"
            version = "20260810_064827Z"
            version_dir = registry / version
            version_dir.mkdir(parents=True)
            metadata_path = version_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "production_model": "super_learner_cal",
                            "model": "Super Learner temporal",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (version_dir / "model.pkl").write_bytes(b"this-is-not-a-valid-pickle")
            (registry / "latest.json").write_text(
                json.dumps(
                    {
                        "latest": version,
                        "metadata": str(metadata_path),
                    }
                ),
                encoding="utf-8",
            )
            builder.MODEL_ROOT = model_root

            self.assertEqual(
                builder._production_model(),
                "super_learner_cal · Super Learner temporal",
            )


if __name__ == "__main__":
    unittest.main()

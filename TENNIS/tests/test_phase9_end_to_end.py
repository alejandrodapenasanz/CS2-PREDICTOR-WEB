"""Audita offline el pipeline diario y la base operativa de la fase 9.

La entrada es la captura real de Tennis Explorer del 30-07-2026. Solo red,
mapping, contexto historico y modelo se sustituyen por dobles deterministas;
parser, orquestador, flags y persistencia/conciliacion SQLite son reales.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.config import TESTS_DIR
from src.daily_pipeline import (
    PREDICTION_OUTPUT_COLUMNS,
    run_daily_prediction_pipeline,
)
from src.daily_pipeline.pipeline import _attach_feature_agenda
from src.operations import OperationsStore
from src.temporal import DEFAULT_SOURCE_DATE_POLICY
from src.tennis_explorer import (
    TennisExplorerSchemaError,
    empty_matches_dataframe,
    parse_daily_matches_html,
)


FIXTURE_DATE = date(2026, 7, 30)
FIXTURE_CAPTURED_AT = pd.Timestamp("2026-07-30T09:48:28Z")
PREDICTION_AT = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
FIXTURE_SHA256 = (
    "2910901e884a468c0f0ae3ef55d0032bb44748bc6026752c878f63075f9758ec"
)
FIXTURE_PATH = (
    TESTS_DIR
    / "fixtures"
    / "tennis_explorer"
    / "matches_2026-07-30_all.html"
)


class _FixtureVector:
    """Representa un vector causal minimo con la API de produccion."""

    def __init__(self, values: dict[str, object]) -> None:
        """Conserva una copia para impedir mutaciones laterales."""

        self._values = dict(values)

    def to_dict(self) -> dict[str, object]:
        """Devuelve el vector como mapping independiente."""

        return dict(self._values)


class _FixtureFeatureBuilder:
    """Genera historial suficiente salvo para los IDs 901/902."""

    def build(self, request: object) -> _FixtureVector:
        """Construye estadisticas prepartido deterministas por identidad."""

        no_history = request.player_a_id in {901, 902} or (
            request.player_b_id in {901, 902}
        )
        general_matches = 0 if no_history else 80
        surface_matches = 0 if no_history else 35
        return _FixtureVector(
            {
                "gender": request.gender,
                "surface": request.surface,
                "tour_level": "ATP Tour",
                "tour_level_raw": request.tour_level_raw,
                "best_of": request.best_of,
                "round": request.round,
                "elo_general_a": 1500.0 if no_history else 1780.0,
                "elo_general_b": 1500.0 if no_history else 1690.0,
                "elo_surface_a": 1500.0 if no_history else 1760.0,
                "elo_surface_b": 1500.0 if no_history else 1675.0,
                "elo_general_matches_a": general_matches,
                "elo_general_matches_b": general_matches,
                "elo_surface_matches_a": surface_matches,
                "elo_surface_matches_b": surface_matches,
                "recent_n_win_rate_a": None if no_history else 0.7,
                "recent_n_win_rate_b": None if no_history else 0.5,
                "recent_n_matches_a": general_matches,
                "recent_n_matches_b": general_matches,
                "recent_months_win_rate_a": None if no_history else 0.65,
                "recent_months_win_rate_b": None if no_history else 0.52,
                "recent_months_matches_a": general_matches,
                "recent_months_matches_b": general_matches,
                "h2h_global_balance": 0.0,
                "h2h_global_matches": 0,
                "h2h_surface_balance": 0.0,
                "h2h_surface_matches": 0,
                "rest_days_a": None if no_history else 4,
                "rest_days_b": None if no_history else 3,
                "ranking_date_a": None if no_history else date(2026, 7, 20),
                "ranking_date_b": None if no_history else date(2026, 7, 20),
                "rank_a": None if no_history else 20,
                "rank_b": None if no_history else 35,
                "rank_points_a": None if no_history else 2100.0,
                "rank_points_b": None if no_history else 1450.0,
                "birth_date_a": date(1998, 1, 1),
                "birth_date_b": date(1997, 1, 1),
                "age_a": 28.5,
                "age_b": 29.5,
                "ranking_missing_a": no_history,
                "ranking_missing_b": no_history,
                "ranking_age_days_a": None if no_history else 10,
                "ranking_age_days_b": None if no_history else 10,
                "age_missing_a": False,
                "age_missing_b": False,
                "market_probability_a": None,
                "market_probability_b": None,
            }
        )


class _FixtureModel:
    """Modelo deportivo determinista con la API desplegada."""

    gender = "M"
    training_max_date = "2026-06-20"
    training_available_max_date = "2026-07-11"
    run_fingerprint = "f" * 64
    estimator = SimpleNamespace(profile="sports_only")

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Devuelve probabilidades preservando el indice."""

        if date.fromisoformat(self.training_available_max_date) >= as_of_date:
            raise AssertionError("El doble recibió un corte no causal.")

        return pd.DataFrame(
            {
                "model_probability_raw_a": 0.69,
                "model_probability_a": 0.64,
                "market_probability_a": frame["market_probability_a"],
                "edge": pd.NA,
            },
            index=frame.index,
        )


def _parse_real_fixture() -> pd.DataFrame:
    """Parsea la captura real y anade su evidencia de adquisicion."""

    parsed = parse_daily_matches_html(FIXTURE_PATH.read_bytes(), FIXTURE_DATE)
    scheduled = parsed.loc[
        parsed["status"].eq("scheduled") & parsed["gender"].eq("M")
    ].head(4).reset_index(drop=True)
    assert len(scheduled) == 4, "La fixture debe contener cuatro ATP futuros."
    scheduled["retrieved_at_utc"] = FIXTURE_CAPTURED_AT
    scheduled["snapshot_sha256"] = FIXTURE_SHA256
    scheduled["scheduled_start_utc"] = pd.date_range(
        "2026-07-30T18:00:00Z",
        periods=len(scheduled),
        freq="1h",
    )
    return scheduled


def _mapped_fixture(scraped: pd.DataFrame) -> pd.DataFrame:
    """Asigna casos mapeado, no mapeado, sin historial y sin cuotas."""

    mapped = scraped.copy()
    mapped["player_1_id"] = pd.Series([101, pd.NA, 901, 301], dtype="Int64")
    mapped["player_2_id"] = pd.Series([102, 202, 902, 302], dtype="Int64")
    mapped["mapping_status"] = pd.Series(
        ["mapped", "unmapped", "mapped", "mapped"], dtype="string"
    )
    mapped.loc[3, ["player_1_odds", "player_2_odds"]] = pd.NA
    return mapped


def _empty_mapped_fixture() -> pd.DataFrame:
    """Devuelve el contrato mapeado vacio sin inventar partidos."""

    return _mapped_fixture(_parse_real_fixture()).iloc[0:0].copy()


def _feature_context() -> object:
    """Crea un contexto causal minimo vinculado al modelo fixture."""

    gender_context = SimpleNamespace(
        builder=_FixtureFeatureBuilder(),
        training_metadata=SimpleNamespace(max_date=date(2026, 6, 20)),
        ranking_max_date=date(2026, 7, 20),
    )
    return SimpleNamespace(
        feature_fingerprint="d" * 64,
        source_date_policy=DEFAULT_SOURCE_DATE_POLICY,
        by_gender={"M": gender_context},
    )


def _run_offline_pipeline(
    scraped: pd.DataFrame,
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Ejecuta el orquestador con las fronteras externas inyectadas."""

    clock_values = iter((PREDICTION_AT, PREDICTION_AT))
    causal = scraped.copy()
    if not causal.empty:
        causal["retrieved_at_utc"] = pd.Timestamp("2026-07-29T09:48:28Z")
        causal["snapshot_sha256"] = "c" * 64
    loaded = _attach_feature_agenda(scraped, causal)
    mapped_with_features = mapped.copy()
    for column in loaded.columns:
        if column.startswith("feature_"):
            mapped_with_features[column] = loaded[column]
    with (
        patch(
            "src.daily_pipeline.pipeline._load_daily_agenda",
            return_value=loaded,
        ) as scraper,
        patch(
            "src.daily_pipeline.pipeline.resolve_scraped_matches",
            return_value=mapped_with_features,
        ) as mapper,
        patch(
            "src.daily_pipeline.pipeline._load_active_quarantine_keys",
            return_value=frozenset(),
        ),
        patch(
            "src.daily_pipeline.pipeline.load_active_deployment_model",
            return_value=_FixtureModel(),
        ),
        patch(
            "src.daily_pipeline.pipeline.build_daily_feature_context",
            return_value=_feature_context(),
        ),
    ):
        run = run_daily_prediction_pipeline(
            FIXTURE_DATE,
            clock=lambda: next(clock_values),
            publish=False,
        )
    scraper.assert_called_once_with(
        FIXTURE_DATE,
        observed_local_date=FIXTURE_DATE,
    )
    mapper.assert_called_once()
    return run.predictions


def _reordered_terminal_result(prediction: pd.Series) -> pd.DataFrame:
    """Crea un final donde A reaparece como player_2 y aun asi gana."""

    return pd.DataFrame(
        [
            {
                "source_match_id": prediction["source_match_id"],
                "status": "finished",
                "player_1_sets_won": 0,
                "player_2_sets_won": 2,
                "sets_score": "0-2",
                "winner_side": "player_2",
                "winner_slug": prediction["player_a_slug"],
                "result_evidence": "terminal_sets_and_slug_fixture",
                "observed_at_utc": "2026-07-30T23:00:00+00:00",
                "source_snapshot_sha256": "b" * 64,
            }
        ]
    )


class Phase9EndToEndTests(unittest.TestCase):
    """Fuerza los casos limite de la auditoria sin acceso de red."""

    def test_fixture_pipeline_flags_and_slug_settlement(self) -> None:
        """Recorre fixture-prediccion-SQLite-final ignorando el orden final."""

        scraped = _parse_real_fixture()
        predictions = _run_offline_pipeline(scraped, _mapped_fixture(scraped))

        self.assertEqual(tuple(predictions.columns), PREDICTION_OUTPUT_COLUMNS)
        self.assertTrue(predictions["source_match_id"].notna().all())
        self.assertTrue(predictions["source_match_id"].is_unique)

        predicted = predictions.iloc[0]
        self.assertEqual(predicted["prediction_status"], "predicted")
        self.assertAlmostEqual(predicted["model_probability_a"], 0.64)
        self.assertEqual(
            predicted["predicted_winner_name"], predicted["player_a_name"]
        )
        self.assertAlmostEqual(
            predicted["predicted_winner_probability"], 0.64
        )
        self.assertIsInstance(predicted["ranking_date_a"], pd.Timestamp)
        self.assertIsInstance(predicted["birth_date_a"], pd.Timestamp)
        self.assertIn(
            "market_prestart_unverified", predicted["confidence_flags"]
        )

        unmapped = predictions.iloc[1]
        self.assertEqual(unmapped["confidence"], "UNAVAILABLE")
        self.assertIn("player_unmapped", unmapped["confidence_flags"])
        self.assertTrue(pd.isna(unmapped["model_probability_a"]))

        no_history = predictions.iloc[2]
        self.assertEqual(no_history["prediction_status"], "predicted")
        self.assertEqual(no_history["confidence"], "LOW")
        self.assertIn(
            "limited_general_history_a", no_history["confidence_flags"]
        )
        self.assertIn(
            "limited_general_history_b", no_history["confidence_flags"]
        )

        missing_odds = predictions.iloc[3]
        self.assertEqual(missing_odds["prediction_status"], "predicted")
        self.assertEqual(missing_odds["market_comparison_status"], "missing")
        self.assertIn("market_missing", missing_odds["confidence_flags"])
        self.assertTrue(pd.isna(missing_odds["market_probability_a"]))
        self.assertTrue(pd.isna(missing_odds["market_probability_b"]))
        self.assertTrue(pd.isna(missing_odds["edge_a"]))
        self.assertTrue(pd.isna(missing_odds["edge_b"]))

        with TemporaryDirectory(dir=TESTS_DIR) as temporary:
            database_path = Path(temporary) / "phase9.sqlite3"
            with OperationsStore(
                database_path,
                clock=lambda: datetime(2026, 7, 31, 8, 0, tzinfo=UTC),
            ) as store:
                registration = store.register_prediction_run(
                    "phase9-fixture-predictions",
                    predictions,
                    target_date=FIXTURE_DATE,
                )
                self.assertEqual(registration.input_rows, 4)
                self.assertEqual(registration.valid_predictions, 3)
                self.assertEqual(registration.official_predictions_selected, 3)

                reconciliation = store.reconcile_observations(
                    "phase9-fixture-results",
                    _reordered_terminal_result(predicted),
                )
                settlement = store.load_settlements().iloc[0]

                self.assertEqual(reconciliation.settlements_inserted, 1)
                self.assertEqual(
                    settlement["source_match_id"], predicted["source_match_id"]
                )
                self.assertEqual(
                    settlement["winner_slug"], predicted["player_a_slug"]
                )
                self.assertEqual(settlement["actual_outcome_a"], 1)

    def test_empty_day_completes_pipeline_and_operations(self) -> None:
        """Una jornada vacia conserva esquema y registra un run sin filas."""

        predictions = _run_offline_pipeline(
            empty_matches_dataframe(), _empty_mapped_fixture()
        )
        self.assertTrue(predictions.empty)
        self.assertEqual(tuple(predictions.columns), PREDICTION_OUTPUT_COLUMNS)
        with TemporaryDirectory(dir=TESTS_DIR) as temporary:
            database_path = Path(temporary) / "phase9-empty.sqlite3"
            with OperationsStore(
                database_path, clock=lambda: PREDICTION_AT
            ) as store:
                registration = store.register_prediction_run(
                    "phase9-empty-day",
                    predictions,
                    target_date=FIXTURE_DATE,
                )
                self.assertEqual(registration.input_rows, 0)
                self.assertEqual(registration.predictions_inserted, 0)
                self.assertTrue(store.load_official_predictions().empty)

    def test_unexpected_html_fails_without_inventing_rows(self) -> None:
        """Un cambio estructural produce error, nunca filas plausibles."""

        changed_html = "<html><body><div>formato inesperado</div></body></html>"
        with self.assertRaisesRegex(TennisExplorerSchemaError, "tabla diaria"):
            parse_daily_matches_html(changed_html, FIXTURE_DATE)


if __name__ == "__main__":
    unittest.main()

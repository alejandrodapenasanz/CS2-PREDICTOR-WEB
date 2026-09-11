"""Tests causales y de degradación del pipeline diario de la fase 8."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

from src.daily_pipeline import (
    PREDICTION_OUTPUT_COLUMNS,
    DailyPredictionError,
    assess_vector_confidence,
    predict_mapped_matches,
    publish_predictions_csv,
    rebuild_history_state,
)
from src.daily_pipeline.context import _read_target_history
from src.daily_pipeline.pipeline import (
    _attach_feature_agenda,
    _attach_surface_catalog,
    _prediction_timestamp_after_snapshot,
    _retrieved_strictly_before_date,
    _resolve_daily_identities,
)
from src.features import FeatureParameters
from src.surface_catalog import SurfaceCatalog, SurfaceEvidence
from src.temporal import DEFAULT_SOURCE_DATE_POLICY


def _history_frame(include_future: bool = True) -> pd.DataFrame:
    """Construye resultados orientados A/B con una fila futura opcional."""

    rows = [
        {
            "record_id": "r1",
            "gender": "M",
            "match_date": date(2024, 1, 1),
            "result_available_date": date(2024, 1, 22),
            "player_a_id": 1,
            "player_b_id": 2,
            "surface": "Hard",
            "y": 1,
        },
        {
            "record_id": "r2",
            "gender": "M",
            "match_date": date(2024, 1, 2),
            "result_available_date": date(2024, 1, 23),
            "player_a_id": 3,
            "player_b_id": 1,
            "surface": "Clay",
            "y": 1,
        },
    ]
    if include_future:
        rows.append(
            {
                "record_id": "future",
                "gender": "M",
                "match_date": date(2024, 2, 1),
                "result_available_date": date(2024, 2, 22),
                "player_a_id": 1,
                "player_b_id": 2,
                "surface": "Hard",
                "y": 0,
            }
        )
    return pd.DataFrame(rows)


def _mapped_frame() -> pd.DataFrame:
    """Crea una cartelera mínima con predicho, no mapeado y terminado."""

    rows = []
    for status, mapping, first_id, second_id in (
        ("scheduled", "mapped", 1, 2),
        ("scheduled", "unmapped", pd.NA, 4),
        ("finished", "mapped", 5, 6),
    ):
        rows.append(
            {
                "match_date": pd.Timestamp("2024-02-01"),
                "tournament": "Test Open",
                "tour_level": "ATP",
                "gender": "M",
                "surface": "Hard",
                "scheduled_time": "10:00",
                "scheduled_start_utc": pd.Timestamp("2024-02-01T10:00:00Z"),
                "status": status,
                "status_evidence": f"fixture_{status}",
                "player_1_name": f"A{len(rows)}",
                "player_2_name": f"B{len(rows)}",
                "player_1_slug": f"a-{len(rows)}",
                "player_2_slug": f"b-{len(rows)}",
                "player_1_id": first_id,
                "player_2_id": second_id,
                "player_1_odds": 1.5,
                "player_2_odds": 3.0,
                "mapping_status": mapping,
                "retrieved_at_utc": pd.Timestamp("2024-02-01T08:00:00Z"),
                "snapshot_sha256": "a" * 64,
                "feature_agenda_retrieved_at_utc": pd.Timestamp("2024-01-31T08:00:00Z"),
                "feature_agenda_snapshot_sha256": "b" * 64,
                "feature_surface": "Hard",
                "feature_tour_level": "ATP",
                "feature_player_1_odds": 1.5,
                "feature_player_2_odds": 3.0,
                "source_match_id": str(1000 + len(rows)),
                "match_detail_href": (f"/match-detail/?id={1000 + len(rows)}"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["player_1_id"] = frame["player_1_id"].astype("Int64")
    frame["player_2_id"] = frame["player_2_id"].astype("Int64")
    return frame


class _FakeVector:
    """Vector mínimo con la API usada por el orquestador."""

    def __init__(self, values: dict[str, object]) -> None:
        """Conserva valores de prueba por copia."""

        self._values = dict(values)

    def to_dict(self) -> dict[str, object]:
        """Devuelve una copia como hace MatchFeatureVector."""

        return dict(self._values)


class _FakeBuilder:
    """Builder determinista que respeta identidad y mercado del request."""

    def build(self, request: object) -> _FakeVector:
        """Produce las columnas que consumen modelo, output y confianza."""

        return _FakeVector(
            {
                "gender": request.gender,
                "surface": request.surface,
                "tour_level": "ATP Tour",
                "tour_level_raw": request.tour_level_raw,
                "best_of": request.best_of,
                "round": request.round,
                "elo_general_matches_a": 100,
                "elo_general_matches_b": 90,
                "elo_surface_matches_a": 40,
                "elo_surface_matches_b": 35,
                "ranking_missing_a": False,
                "ranking_missing_b": False,
                "ranking_age_days_a": 5,
                "ranking_age_days_b": 5,
                "age_missing_a": False,
                "age_missing_b": False,
                "market_probability_a": (0.5 if request.odds_a is not None else None),
                "market_probability_b": (0.5 if request.odds_b is not None else None),
            }
        )


class _FakeModel:
    """Modelo calibrado falso con el contrato mínimo del servicio real."""

    def __init__(self, training_max_date: str = "2024-01-01") -> None:
        """Configura un corte causal o no causal para cada prueba."""

        self.gender = "M"
        self.training_max_date = training_max_date
        self.training_available_max_date = DEFAULT_SOURCE_DATE_POLICY.availability_date(
            date.fromisoformat(training_max_date)
        ).isoformat()
        self.run_fingerprint = "f" * 64
        self.estimator = SimpleNamespace(profile="sports_only")

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Devuelve probabilidades deterministas preservando el índice."""

        if date.fromisoformat(self.training_max_date) >= as_of_date:
            raise AssertionError("El doble recibió un corte no causal.")

        return pd.DataFrame(
            {
                "model_probability_raw_a": 0.7,
                "model_probability_a": 0.65,
                "market_probability_a": frame["market_probability_a"],
                "edge": 0.15,
            },
            index=frame.index,
        )


def _feature_context() -> object:
    """Crea metadata causal mínima para el género masculino."""

    gender_context = SimpleNamespace(
        builder=_FakeBuilder(),
        training_metadata=SimpleNamespace(max_date=date(2024, 1, 1)),
        ranking_max_date=date(2024, 1, 22),
    )
    return SimpleNamespace(
        feature_fingerprint="d" * 64,
        source_date_policy=DEFAULT_SOURCE_DATE_POLICY,
        by_gender={"M": gender_context},
    )


class DailyTimestampTests(unittest.TestCase):
    """Verifica el orden lógico captura-predicción en relojes de baja resolución."""

    def test_equal_wall_clock_ticks_receive_one_microsecond_order(self) -> None:
        """Impide que una captura anterior quede inválida por timestamp igual."""

        captured_at = pd.Timestamp("2026-08-09T08:00:00Z")
        scraped = pd.DataFrame({"retrieved_at_utc": [captured_at]})
        prediction_at = _prediction_timestamp_after_snapshot(
            captured_at.to_pydatetime(),
            scraped,
        )

        self.assertEqual(
            prediction_at,
            (captured_at + pd.Timedelta(microseconds=1)).to_pydatetime(),
        )

    def test_pandas3_second_resolution_keeps_strict_daily_cutoff(self) -> None:
        """`datetime64[s]` accepts only observations before UTC midnight D."""

        retrieved = pd.Series(
            pd.to_datetime(
                [
                    "2026-08-23T23:59:59Z",
                    "2026-08-24T00:00:00Z",
                    "2026-08-25T00:00:00Z",
                    None,
                ],
                utc=True,
            ),
            dtype="datetime64[s, UTC]",
        )

        eligible = _retrieved_strictly_before_date(
            retrieved,
            date(2026, 8, 24),
        )

        self.assertEqual(eligible.tolist(), [True, False, False, False])


class DailySurfaceCatalogTests(unittest.TestCase):
    """Comprueba que la edición exacta completa el input diario auditable."""

    def test_prior_same_edition_surface_fills_daily_feature(self) -> None:
        """Una captura anterior a D rellena superficie y procedencia."""

        presentation = _mapped_frame().iloc[[0]].copy()
        presentation["source_match_id"] = "tennisratio:M:today"
        presentation["tournament"] = "Exact Open"
        presentation["tournament_href"] = pd.NA
        causal = presentation.iloc[0:0]
        frozen = _attach_feature_agenda(presentation, causal)
        evidence = SurfaceEvidence(
            evidence_id="surface-evidence",
            source_family="tennisratio",
            evidence_kind="direct_agenda",
            source_match_id="tennisratio:M:yesterday",
            gender="M",
            edition_year=2024,
            tournament="Exact Open",
            tournament_href=None,
            tour_level="ATP",
            surface="Clay",
            effective_date=date(2024, 1, 31),
            captured_at_utc=datetime(2024, 1, 30, 8, tzinfo=UTC),
            source_url="https://example.test/agenda",
            source_sha256="e" * 64,
        )
        catalog = SurfaceCatalog(
            as_of_date=date(2024, 2, 1),
            evidence=(evidence,),
            fingerprint="f" * 64,
            source_counts={"tennisratio": 1},
        )

        enriched = _attach_surface_catalog(
            frozen,
            catalog=catalog,
            match_date=date(2024, 2, 1),
        )

        self.assertEqual(enriched.at[0, "feature_surface"], "Clay")
        self.assertEqual(
            enriched.at[0, "feature_surface_resolution"],
            "same_edition_propagation",
        )
        self.assertEqual(
            enriched.at[0, "feature_surface_evidence_id"],
            "surface-evidence",
        )
        self.assertEqual(enriched.at[0, "feature_tour_level"], "ATP")
        predicted = predict_mapped_matches(
            enriched,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        self.assertEqual(predicted.at[0, "confidence"], "MEDIUM")
        self.assertNotIn("surface_missing", predicted.at[0, "confidence_flags"])

    def test_match_day_surface_does_not_enter_feature(self) -> None:
        """La superficie visible de D sigue fuera del vector causal."""

        presentation = _mapped_frame().iloc[[0]].copy()
        presentation["source_match_id"] = "tennisratio:M:today"
        presentation["tournament"] = "No Prior Evidence"
        presentation["tournament_href"] = pd.NA
        frozen = _attach_feature_agenda(presentation, presentation)
        catalog = SurfaceCatalog(
            as_of_date=date(2024, 2, 1),
            evidence=(),
            fingerprint="f" * 64,
            source_counts={},
        )

        enriched = _attach_surface_catalog(
            frozen,
            catalog=catalog,
            match_date=date(2024, 2, 1),
        )

        self.assertTrue(pd.isna(enriched.at[0, "feature_surface"]))
        self.assertEqual(
            enriched.at[0, "feature_surface_resolution_reason"],
            "no_exact_pre_cutoff_surface_evidence",
        )


class TennisRatioIdentityRoutingTests(unittest.TestCase):
    """Comprueba el contrato aislado de identidad de la nueva cartelera."""

    @staticmethod
    def _agenda(*, gender: str = "M") -> pd.DataFrame:
        """Crea una fila TennisRatio sin IDs Sackmann preasignados."""

        return pd.DataFrame(
            [
                {
                    "source_match_id": f"tennisratio:{gender}:fixture",
                    "gender": gender,
                    "player_1_slug": "tennisratio:Shared",
                    "player_2_slug": "tennisratio:Opponent",
                }
            ]
        )

    def test_exact_sidecar_identities_bypass_legacy_mapper(self) -> None:
        """Mantiene ATP/WTA separados aunque compartan la clave fuente."""

        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "tennisratio.sqlite3"
            database.touch()
            identities = (
                SimpleNamespace(
                    gender="M",
                    source_player_key="tennisratio:Shared",
                    sackmann_player_id=101,
                ),
                SimpleNamespace(
                    gender="M",
                    source_player_key="tennisratio:Opponent",
                    sackmann_player_id=102,
                ),
                SimpleNamespace(
                    gender="F",
                    source_player_key="tennisratio:Shared",
                    sackmann_player_id=201,
                ),
                SimpleNamespace(
                    gender="F",
                    source_player_key="tennisratio:Opponent",
                    sackmann_player_id=202,
                ),
            )
            agenda = pd.concat(
                [self._agenda(gender="M"), self._agenda(gender="F")],
                ignore_index=True,
            )
            with (
                mock.patch(
                    "src.daily_pipeline.pipeline.TENNISRATIO_DATABASE_PATH",
                    database,
                ),
                mock.patch(
                    "src.daily_pipeline.pipeline.load_identity_resolutions",
                    return_value=identities,
                ) as identity_loader,
                mock.patch("src.daily_pipeline.pipeline.resolve_scraped_matches") as legacy_mapper,
            ):
                mapped = _resolve_daily_identities(
                    agenda,
                    as_of_date=date(2026, 8, 22),
                )

        legacy_mapper.assert_not_called()
        identity_loader.assert_called_once_with(
            as_of_date=date(2026, 8, 22),
            database_path=database,
        )
        self.assertEqual(mapped["player_1_id"].tolist(), [101, 201])
        self.assertEqual(mapped["player_2_id"].tolist(), [102, 202])
        self.assertEqual(mapped["mapping_status"].tolist(), ["mapped", "mapped"])
        self.assertEqual(
            mapped["player_1_mapping_method"].tolist(),
            ["tennisratio_exact_identity", "tennisratio_exact_identity"],
        )

    def test_conflicting_sidecar_identity_degrades_the_whole_match(self) -> None:
        """Nunca elige un ID según el orden ante evidencia contradictoria."""

        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "tennisratio.sqlite3"
            database.touch()
            identities = (
                SimpleNamespace(
                    gender="M",
                    source_player_key="tennisratio:Shared",
                    sackmann_player_id=101,
                ),
                SimpleNamespace(
                    gender="M",
                    source_player_key="tennisratio:Shared",
                    sackmann_player_id=999,
                ),
                SimpleNamespace(
                    gender="M",
                    source_player_key="tennisratio:Opponent",
                    sackmann_player_id=102,
                ),
            )
            with (
                mock.patch(
                    "src.daily_pipeline.pipeline.TENNISRATIO_DATABASE_PATH",
                    database,
                ),
                mock.patch(
                    "src.daily_pipeline.pipeline.load_identity_resolutions",
                    return_value=identities,
                ),
            ):
                mapped = _resolve_daily_identities(
                    self._agenda(),
                    as_of_date=date(2026, 8, 22),
                )

        self.assertTrue(pd.isna(mapped.at[0, "player_1_id"]))
        self.assertTrue(pd.isna(mapped.at[0, "player_2_id"]))
        self.assertEqual(mapped.at[0, "mapping_status"], "unmapped")
        self.assertEqual(
            mapped.at[0, "player_1_mapping_method"],
            "tennisratio_identity_conflict",
        )

    def test_current_status_and_causal_model_metadata_stay_separate(self) -> None:
        """No etiqueta cuotas/superficie de D con el timestamp de D-1."""

        feature_columns = [
            column for column in _mapped_frame().columns if column.startswith("feature_")
        ]
        presentation = _mapped_frame().iloc[[0]].drop(columns=feature_columns)
        presentation.loc[0, "status"] = "finished"
        causal = presentation.copy()
        causal.loc[0, "status"] = "scheduled"
        causal.loc[0, "retrieved_at_utc"] = pd.Timestamp("2024-01-31T08:00:00Z")
        causal.loc[0, "snapshot_sha256"] = "c" * 64
        causal.loc[0, "surface"] = "Clay"
        causal.loc[0, "player_1_odds"] = 1.8
        causal.loc[0, "scheduled_start_utc"] = pd.Timestamp("2024-02-01T07:00:00Z")

        merged = _attach_feature_agenda(presentation, causal)

        self.assertEqual(merged.at[0, "status"], "finished")
        self.assertEqual(
            merged.at[0, "retrieved_at_utc"],
            pd.Timestamp("2024-02-01T08:00:00Z"),
        )
        self.assertEqual(
            merged.at[0, "feature_agenda_retrieved_at_utc"],
            pd.Timestamp("2024-01-31T08:00:00Z"),
        )
        self.assertEqual(merged.at[0, "feature_surface"], "Clay")
        self.assertEqual(merged.at[0, "feature_player_1_odds"], 1.8)
        self.assertEqual(
            merged.at[0, "scheduled_start_utc"],
            pd.Timestamp("2024-02-01T10:00:00Z"),
        )


class DailyTargetHistoryReadTests(unittest.TestCase):
    """Verifica la lectura Parquet causal sin el módulo opcional dataset."""

    def test_batch_reader_filters_cutoff_and_players_without_duplicates(self) -> None:
        """La igualdad con D se excluye y un partido bilateral aparece una vez."""

        class FakeBatch:
            """Lote mínimo compatible con ``RecordBatch.to_pandas``."""

            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame

            def to_pandas(self) -> pd.DataFrame:
                """Devuelve una copia para simular la conversión Arrow."""

                return self.frame.copy()

        rows = pd.DataFrame(
            [
                {
                    "record_id": "eligible-a",
                    "gender": "M",
                    "match_date": date(2026, 8, 1),
                    "result_available_date": date(2026, 8, 23),
                    "player_a_id": 10,
                    "player_b_id": 20,
                    "surface": "Hard",
                    "y": 1,
                },
                {
                    "record_id": "same-day",
                    "gender": "M",
                    "match_date": date(2026, 8, 2),
                    "result_available_date": date(2026, 8, 24),
                    "player_a_id": 10,
                    "player_b_id": 30,
                    "surface": "Clay",
                    "y": 0,
                },
                {
                    "record_id": "unrelated",
                    "gender": "M",
                    "match_date": date(2026, 8, 3),
                    "result_available_date": date(2026, 8, 23),
                    "player_a_id": 40,
                    "player_b_id": 50,
                    "surface": "Grass",
                    "y": 1,
                },
                {
                    "record_id": "eligible-bilateral",
                    "gender": "M",
                    "match_date": date(2026, 8, 4),
                    "result_available_date": date(2026, 8, 23),
                    "player_a_id": 10,
                    "player_b_id": 20,
                    "surface": "Hard",
                    "y": 0,
                },
            ]
        )
        parquet = mock.Mock()
        parquet.iter_batches.return_value = (
            FakeBatch(rows.iloc[:2]),
            FakeBatch(rows.iloc[2:]),
        )
        metadata = SimpleNamespace(path=Path("history.parquet"), gender="M")

        with mock.patch(
            "src.daily_pipeline.context.pq.ParquetFile",
            return_value=parquet,
        ):
            selected = _read_target_history(
                metadata,
                as_of_date=date(2026, 8, 24),
                player_ids=(10, 20),
            )

        self.assertEqual(
            selected["record_id"].tolist(),
            ["eligible-a", "eligible-bilateral"],
        )
        parquet.iter_batches.assert_called_once_with(
            batch_size=65_536,
            columns=list(selected.columns),
        )


class DailyHistoryRebuildTests(unittest.TestCase):
    """Comprueba reconstrucción dirigida y corte estricto por fecha."""

    def test_future_rows_do_not_change_target_history_snapshot(self) -> None:
        """Añadir D/futuro no modifica forma, H2H ni descanso as-of D."""

        parameters = FeatureParameters()
        snapshots = []
        for include_future in (False, True):
            state = rebuild_history_state(
                _history_frame(include_future),
                gender="M",
                as_of_date=date(2024, 2, 1),
                target_player_ids=frozenset({1, 2}),
                feature_parameters=parameters,
                source_date_policy=DEFAULT_SOURCE_DATE_POLICY,
            )
            snapshots.append(
                state.snapshot(
                    "M",
                    1,
                    2,
                    surface="Hard",
                    as_of_date=date(2024, 2, 1),
                )
            )
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[0].recent_n_matches_a, 2)
        self.assertAlmostEqual(snapshots[0].recent_n_win_rate_a, 0.5)
        self.assertEqual(snapshots[0].h2h_global_matches, 1)
        self.assertEqual(snapshots[0].rest_days_a, 30)


class DailyConfidenceTests(unittest.TestCase):
    """Fija los umbrales y prueba que no dependen de la probabilidad."""

    def test_stale_or_poor_history_is_low_confidence(self) -> None:
        """Fuentes obsoletas y poca muestra generan flags explícitos."""

        values = {
            "surface": "Hard",
            "best_of": None,
            "round": None,
            "elo_general_matches_a": 2,
            "elo_general_matches_b": 50,
            "elo_surface_matches_a": 1,
            "elo_surface_matches_b": 20,
            "ranking_missing_a": True,
            "ranking_missing_b": False,
            "ranking_age_days_a": None,
            "ranking_age_days_b": 40,
            "age_missing_a": False,
            "age_missing_b": True,
            "market_probability_a": 0.5,
        }
        assessment = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_available_max_date=date(2024, 1, 1),
            ranking_max_date=date(2024, 1, 1),
        )
        self.assertEqual(assessment.level, "LOW")
        self.assertIn("limited_general_history_a", assessment.flags)
        self.assertIn("ranking_missing_a", assessment.flags)
        self.assertIn("history_available_stale_31d", assessment.flags)

    def test_causal_availability_cutoff_avoids_false_stale_flag(self) -> None:
        """El embargo de 21 días no cuenta como retraso de publicación."""

        values = {
            "surface": "Hard",
            "best_of": 3,
            "round": "R32",
            "elo_general_matches_a": 30,
            "elo_general_matches_b": 30,
            "elo_surface_matches_a": 12,
            "elo_surface_matches_b": 12,
            "ranking_missing_a": False,
            "ranking_missing_b": False,
            "ranking_age_days_a": 1,
            "ranking_age_days_b": 1,
            "age_missing_a": False,
            "age_missing_b": False,
            "market_probability_a": 0.5,
        }
        source_match_date = date(2024, 1, 10)
        available_date = DEFAULT_SOURCE_DATE_POLICY.availability_date(source_match_date)
        self.assertEqual(available_date, date(2024, 1, 31))

        assessment = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_available_max_date=available_date,
            ranking_max_date=date(2024, 1, 31),
        )

        self.assertEqual(assessment.level, "MEDIUM")
        self.assertFalse(
            any(flag.startswith("history_available_stale_") for flag in assessment.flags)
        )

    def test_history_availability_lag_has_configurable_boundary(self) -> None:
        """Catorce días pasan; quince fallan sin mover el corte causal."""

        values = {
            "surface": "Hard",
            "best_of": 3,
            "round": "R32",
            "elo_general_matches_a": 30,
            "elo_general_matches_b": 30,
            "elo_surface_matches_a": 12,
            "elo_surface_matches_b": 12,
            "ranking_missing_a": False,
            "ranking_missing_b": False,
            "ranking_age_days_a": 1,
            "ranking_age_days_b": 1,
            "age_missing_a": False,
            "age_missing_b": False,
            "market_probability_a": 0.5,
        }
        fresh = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_available_max_date=date(2024, 1, 18),
            ranking_max_date=date(2024, 1, 31),
        )
        stale = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_available_max_date=date(2024, 1, 17),
            ranking_max_date=date(2024, 1, 31),
        )
        stricter = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_available_max_date=date(2024, 1, 28),
            ranking_max_date=date(2024, 1, 31),
            max_history_availability_lag_days=3,
        )

        self.assertNotIn("history_available_stale_14d", fresh.flags)
        self.assertIn("history_available_stale_15d", stale.flags)
        self.assertIn("history_available_stale_4d", stricter.flags)

    def test_history_availability_cutoff_must_precede_match(self) -> None:
        """Un corte disponible en D o futuro se rechaza como fuga."""

        values: dict[str, object] = {}
        for invalid_cutoff in (date(2024, 2, 1), date(2024, 2, 2)):
            with self.subTest(invalid_cutoff=invalid_cutoff):
                with self.assertRaisesRegex(
                    ValueError,
                    "history_available_max_date debe ser estrictamente anterior",
                ):
                    assess_vector_confidence(
                        values,
                        match_date=date(2024, 2, 1),
                        history_available_max_date=invalid_cutoff,
                        ranking_max_date=date(2024, 1, 31),
                    )


class DailyPredictionTableTests(unittest.TestCase):
    """Valida orientación A/B, calibración y degradación por fila."""

    def test_only_scheduled_mapped_match_receives_probabilities(self) -> None:
        """No mapeados y terminados se conservan sin inventar probabilidad."""

        output = predict_mapped_matches(
            _mapped_frame(),
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        self.assertEqual(tuple(output.columns), PREDICTION_OUTPUT_COLUMNS)
        self.assertAlmostEqual(output.loc[0, "model_probability_a"], 0.65)
        self.assertAlmostEqual(output.loc[0, "model_probability_b"], 0.35)
        self.assertTrue(pd.isna(output.loc[0, "edge_a"]))
        self.assertTrue(pd.isna(output.loc[0, "edge_b"]))
        self.assertEqual(
            output.loc[0, "market_comparison_status"],
            "prestart_unverified",
        )
        self.assertAlmostEqual(output.loc[0, "market_probability_a"], 2 / 3)
        self.assertAlmostEqual(output.loc[0, "market_probability_b"], 1 / 3)
        self.assertIn(
            "market_prestart_unverified",
            output.loc[0, "confidence_flags"],
        )
        self.assertNotIn(
            "history_available_stale_",
            output.loc[0, "confidence_flags"],
        )
        self.assertEqual(
            output.loc[0, "feature_history_max_date"],
            pd.Timestamp("2024-01-01"),
        )
        self.assertEqual(
            output.loc[0, "feature_history_available_max_date"],
            pd.Timestamp("2024-01-22"),
        )
        self.assertEqual(
            output.loc[0, "model_training_available_max_date"],
            pd.Timestamp("2024-01-22"),
        )
        self.assertEqual(output.loc[0, "predicted_winner_name"], "A0")
        self.assertAlmostEqual(
            output.loc[0, "predicted_winner_probability"],
            0.65,
        )
        self.assertEqual(output.loc[0, "player_a_name"], "A0")
        self.assertEqual(output.loc[1, "confidence"], "UNAVAILABLE")
        self.assertIn("player_unmapped", output.loc[1, "confidence_flags"])
        self.assertTrue(pd.isna(output.loc[1, "model_probability_a"]))
        self.assertIn("status_finished", output.loc[2, "confidence_flags"])
        self.assertTrue(pd.isna(output.loc[2, "model_probability_a"]))

    def test_stale_agenda_and_fallback_are_explicit_and_lower_confidence(self) -> None:
        """Una cartelera vieja o secundaria conserva p, pero nunca confianza alta."""

        frame = _mapped_frame().iloc[[0]].copy()
        frame.loc[0, "retrieved_at_utc"] = pd.Timestamp("2024-01-30T08:00:00Z")
        frame.loc[0, "source_match_id"] = "tennisratio:old-match"
        stale = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        self.assertAlmostEqual(stale.loc[0, "model_probability_a"], 0.65)
        self.assertEqual(stale.loc[0, "data_freshness_status"], "stale")
        self.assertEqual(stale.loc[0, "confidence"], "LOW")
        self.assertIn("data_stale_tennisratio_49h", stale.loc[0, "confidence_flags"])

        fallback_frame = _mapped_frame().iloc[[0]].copy()
        fallback = predict_mapped_matches(
            fallback_frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        self.assertTrue(bool(fallback.loc[0, "fallback_source_used"]))
        self.assertEqual(fallback.loc[0, "confidence"], "LOW")
        self.assertIn(
            "agenda_fallback_tennis_explorer",
            fallback.loc[0, "confidence_flags"],
        )

    def test_same_day_agenda_metadata_is_never_offered_to_model(self) -> None:
        """Una captura de D queda visible pero sin predicción retrospectiva."""

        frame = _mapped_frame().iloc[[0]].copy()
        frame.loc[0, "feature_agenda_retrieved_at_utc"] = pd.Timestamp("2024-02-01T07:00:00Z")
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={"M": _FakeModel()},
            feature_context=None,
        )

        self.assertEqual(output.loc[0, "confidence"], "UNAVAILABLE")
        self.assertIn(
            "feature_agenda_not_strictly_before_match_date",
            output.loc[0, "confidence_flags"],
        )
        self.assertTrue(pd.isna(output.loc[0, "model_probability_a"]))

    def test_missing_pre_date_agenda_uses_causal_history_with_low_confidence(self) -> None:
        """Una alta nueva conserva predicción sin usar contexto capturado en D."""

        frame = _mapped_frame().iloc[[0]].copy()
        frame.loc[0, "source_match_id"] = "tennisratio:M:new-on-d"
        frame.loc[0, "feature_agenda_retrieved_at_utc"] = pd.NaT
        frame.loc[0, "feature_agenda_snapshot_sha256"] = pd.NA
        frame.loc[0, "feature_surface"] = pd.NA
        frame.loc[0, "feature_tour_level"] = pd.NA
        frame.loc[0, "feature_player_1_odds"] = pd.NA
        frame.loc[0, "feature_player_2_odds"] = pd.NA

        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )

        self.assertEqual(output.loc[0, "prediction_status"], "predicted")
        self.assertAlmostEqual(output.loc[0, "model_probability_a"], 0.65)
        self.assertTrue(pd.isna(output.loc[0, "canonical_tour_level"]))
        self.assertEqual(output.loc[0, "confidence"], "LOW")
        self.assertIn(
            "feature_agenda_availability_missing",
            output.loc[0, "confidence_flags"],
        )
        self.assertIn("feature_tour_level_missing", output.loc[0, "confidence_flags"])

    def test_prediction_must_be_strictly_before_published_utc_start(self) -> None:
        """Antes predice; igualdad, después y TBD quedan visibles sin probabilidad."""

        cases = (
            (
                "before",
                pd.Timestamp("2024-02-01T10:00:00Z"),
                datetime(2024, 2, 1, 9, tzinfo=UTC),
                "predicted",
                None,
            ),
            (
                "equal",
                pd.Timestamp("2024-02-01T10:00:00Z"),
                datetime(2024, 2, 1, 10, tzinfo=UTC),
                "not_predicted",
                "prediction_not_strictly_before_scheduled_start",
            ),
            (
                "after",
                pd.Timestamp("2024-02-01T10:00:00Z"),
                datetime(2024, 2, 1, 11, tzinfo=UTC),
                "not_predicted",
                "prediction_not_strictly_before_scheduled_start",
            ),
            (
                "tbd",
                pd.NaT,
                datetime(2024, 2, 1, 9, tzinfo=UTC),
                "not_predicted",
                "scheduled_start_utc_missing",
            ),
            (
                "timezone_missing",
                "2024-02-01T10:00:00",
                datetime(2024, 2, 1, 9, tzinfo=UTC),
                "not_predicted",
                "scheduled_start_utc_invalid",
            ),
        )
        for name, scheduled_start, prediction_at, expected, flag in cases:
            with self.subTest(case=name):
                frame = _mapped_frame().iloc[[0]].copy()
                frame["scheduled_start_utc"] = frame["scheduled_start_utc"].astype("object")
                frame.at[0, "scheduled_start_utc"] = scheduled_start
                output = predict_mapped_matches(
                    frame,
                    match_date=date(2024, 2, 1),
                    prediction_as_of_utc=prediction_at,
                    models={"M": _FakeModel()},
                    feature_context=(_feature_context() if expected == "predicted" else None),
                )

                self.assertEqual(output.loc[0, "prediction_status"], expected)
                if flag is None:
                    self.assertAlmostEqual(
                        output.loc[0, "model_probability_a"],
                        0.65,
                    )
                else:
                    self.assertIn(flag, output.loc[0, "confidence_flags"])
                    self.assertTrue(pd.isna(output.loc[0, "model_probability_a"]))

    def test_tennisratio_tour_level_must_match_gender(self) -> None:
        """No cruza una etiqueta ATP con el universo femenino por error fuente."""

        frame = _mapped_frame().iloc[[0]].copy()
        frame.loc[0, "source_match_id"] = "tennisratio:F:fixture"
        frame.loc[0, "gender"] = "F"
        frame.loc[0, "feature_tour_level"] = "ATP"
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(2024, 2, 1, 9, tzinfo=UTC),
            models={},
            feature_context=None,
        )

        self.assertEqual(output.loc[0, "confidence"], "UNAVAILABLE")
        self.assertIn(
            "feature_tour_gender_conflict",
            output.loc[0, "confidence_flags"],
        )

    def test_model_trained_on_date_is_blocked_as_future_leakage(self) -> None:
        """Un modelo cuyo corte no precede D nunca recibe la fila."""

        frame = _mapped_frame().iloc[[0]].copy()
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel("2024-02-01")},
            feature_context=None,
        )
        self.assertTrue(pd.isna(output.iloc[0]["model_probability_a"]))
        self.assertEqual(output.iloc[0]["confidence"], "UNAVAILABLE")
        self.assertEqual(
            output.iloc[0]["confidence_flags"],
            "model_not_causal_for_date",
        )

    def test_feature_availability_cutoff_must_match_model(self) -> None:
        """Serving falla cerrado si features y bundle declaran cortes distintos."""

        context = _feature_context()
        context.by_gender["M"].training_metadata.max_date = date(2024, 1, 2)

        with self.assertRaisesRegex(
            DailyPredictionError,
            "corte disponible del histórico no coincide",
        ):
            predict_mapped_matches(
                _mapped_frame().iloc[[0]].copy(),
                match_date=date(2024, 2, 1),
                prediction_as_of_utc=datetime(
                    2024,
                    2,
                    1,
                    9,
                    tzinfo=UTC,
                ),
                models={"M": _FakeModel()},
                feature_context=context,
            )

    def test_quarantined_identity_degrades_only_its_match(self) -> None:
        """Una identidad ambigua queda sin probabilidad y la jornada continúa."""

        frame = _mapped_frame().iloc[[0]].copy()
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=None,
            excluded_player_keys={("M", 1)},
        )

        self.assertTrue(output.iloc[0]["identity_quarantined"])
        self.assertTrue(pd.isna(output.iloc[0]["model_probability_a"]))
        self.assertEqual(output.iloc[0]["confidence"], "UNAVAILABLE")
        self.assertEqual(
            output.iloc[0]["confidence_flags"],
            "identity_quarantined",
        )

    def test_csv_is_timestamped_and_round_trips_inside_tennis(self) -> None:
        """La publicación es única, legible y permanece dentro del proyecto."""

        output = predict_mapped_matches(
            _mapped_frame().iloc[[0]].copy(),
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        with TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            with mock.patch(
                "src.daily_pipeline.pipeline.PROJECT_ROOT",
                temporary_root,
            ):
                path = publish_predictions_csv(
                    output,
                    match_date=date(2024, 2, 1),
                    prediction_as_of_utc=datetime(
                        2024,
                        2,
                        1,
                        9,
                        tzinfo=UTC,
                    ),
                    output_dir=temporary_root,
                )
            self.assertTrue(path.is_file())
            self.assertIn("predictions_2024-02-01_", path.name)
            loaded = pd.read_csv(path)
            self.assertEqual(tuple(loaded.columns), PREDICTION_OUTPUT_COLUMNS)
            self.assertAlmostEqual(loaded.loc[0, "model_probability_a"], 0.65)


if __name__ == "__main__":
    unittest.main()

"""Tests offline de la orquestación diaria, SQLite y conciliación."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.config import PROJECT_ROOT, TESTS_DIR
from src.daily_pipeline import DailyPredictionRun
from src.operations import (
    OperationsStore,
    reconcile_stored_tennis_explorer_results,
    run_operational_daily_pipeline,
)
from src.operations.result_sources import (
    TennisExplorerMappedResultSnapshot,
    TennisRatioResultSnapshot,
)
from src.player_mapping import PlayerMappingStore
from src.tennis_explorer import (
    TennisExplorerHttpError,
    TennisExplorerResultSnapshot,
)
from src.tennisratio import MappedResult
from src.temporal import DEFAULT_SOURCE_DATE_POLICY


NOW = datetime(2026, 7, 30, 8, 0, tzinfo=UTC)


def _prediction_frame(match_date: date, source_match_id: str) -> pd.DataFrame:
    """Construye una predicción operativamente válida con stats as-of."""

    prediction_at = datetime.combine(
        match_date,
        datetime.min.time(),
        tzinfo=UTC,
    ).replace(hour=7, minute=5)
    source_at = prediction_at.replace(minute=0)
    row: dict[str, object] = {
        "source_match_id": source_match_id,
        "prediction_date": match_date.isoformat(),
        "prediction_as_of_utc": prediction_at,
        "source_retrieved_at_utc": source_at,
        "source_snapshot_sha256": "a" * 64,
        "match_detail_href": "/match-detail/?id=123",
        "tournament": "Example Open",
        "tournament_href": "/example-open/",
        "tour_level": "ATP",
        "gender": "M",
        "surface": "Hard",
        "scheduled_start_utc": datetime.combine(
            match_date,
            datetime.min.time(),
            tzinfo=UTC,
        ).replace(hour=10),
        "status": "scheduled",
        "player_a_name": "Alpha A.",
        "player_b_name": "Beta B.",
        "player_a_slug": "alpha-player",
        "player_b_slug": "beta-player",
        "player_a_id": 101,
        "player_b_id": 202,
        "mapping_status": "mapped",
        "model_probability_raw_a": 0.61,
        "model_probability_a": 0.62,
        "model_probability_b": 0.38,
        "market_probability_a": 0.55,
        "market_probability_b": 0.45,
        "edge_a": 0.07,
        "edge_b": -0.07,
        "confidence": "medium",
        "confidence_flags": "ranking_stale",
        "prediction_status": "predicted",
        "model_profile": "sports_only",
        "model_fingerprint": "model-m-v2",
        "model_training_max_date": match_date.replace(year=match_date.year - 1),
        "model_training_available_max_date": (
            DEFAULT_SOURCE_DATE_POLICY.availability_date(
                match_date.replace(year=match_date.year - 1)
            )
        ),
        "feature_fingerprint": "features-v2",
    }
    for statistic in (
        "elo_general",
        "elo_surface",
        "elo_general_matches",
        "elo_surface_matches",
        "recent_n_win_rate",
        "recent_n_matches",
        "recent_months_win_rate",
        "recent_months_matches",
        "rest_days",
        "rank",
        "rank_points",
        "age",
    ):
        row[f"{statistic}_a"] = 10.0
        row[f"{statistic}_b"] = 20.0
    return pd.DataFrame([row])


def _daily_runner(frame: pd.DataFrame, match_date: date):
    """Devuelve un stub con la misma firma que el pipeline causal."""

    def run(
        requested_date: date | None,
        *,
        clock=None,
        output_dir=None,
        publish=True,
    ) -> DailyPredictionRun:
        """Construye un resultado diario sin red ni artefactos."""

        selected = requested_date or match_date
        prediction_at = datetime.combine(
            selected,
            datetime.min.time(),
            tzinfo=UTC,
        ).replace(hour=7, minute=5)
        return DailyPredictionRun(
            match_date=selected,
            prediction_as_of_utc=prediction_at,
            predictions=frame.copy(),
            output_path=None,
        )

    return run


def _result_snapshot(match_date: date) -> TennisExplorerResultSnapshot:
    """Construye evidencia terminal con el ganador reordenado por la web."""

    observed = datetime.combine(
        match_date,
        datetime.min.time(),
        tzinfo=UTC,
    ).replace(hour=18)
    frame = pd.DataFrame(
        [
            {
                "source_match_id": "te:prior",
                "match_date": pd.Timestamp(match_date),
                "status": "finished",
                "player_1_sets_won": 0,
                "player_2_sets_won": 2,
                "sets_score": "4-6 3-6",
                "winner_side": "player_2",
                "winner_slug": "alpha-player",
                "result_evidence": "terminal_sets_and_score",
                "source_snapshot_sha256": "b" * 64,
                "retrieved_at_utc": observed,
            }
        ]
    )
    return TennisExplorerResultSnapshot(
        match_date=match_date,
        retrieved_at_utc=observed,
        source_url="https://www.tennisexplorer.com/matches/",
        snapshot_sha256="b" * 64,
        matches=frame,
        html_path=TESTS_DIR / "fixture-result.html",
        metadata_path=TESTS_DIR / "fixture-result.metadata.json",
    )


def _cross_source_result_snapshot(
    match_date: date,
    *,
    duplicate: bool = False,
) -> TennisExplorerResultSnapshot:
    """Construye resultados Explorer cuyo ID no coincide con TennisRatio."""

    observed = datetime.combine(
        match_date,
        datetime.min.time(),
        tzinfo=UTC,
    ).replace(hour=18)
    rows: list[dict[str, object]] = [
        {
            "source_match_id": "3307289",
            "match_date": pd.Timestamp(match_date),
            "tournament": "Example Open",
            "gender": "M",
            "status": "finished",
            "player_1_slug": "beta-explorer",
            "player_2_slug": "alpha-explorer",
            "player_1_sets_won": 2,
            "player_2_sets_won": 1,
            "sets_score": "6-4 3-6 6-3",
            "winner_side": "player_1",
            "winner_slug": "beta-explorer",
            "result_evidence": "winner_from_terminal_sets_and_slug",
            "source_snapshot_sha256": "d" * 64,
            "retrieved_at_utc": observed,
        }
    ]
    if duplicate:
        duplicate_row = dict(rows[0])
        duplicate_row["source_match_id"] = "3307290"
        rows.append(duplicate_row)
    return TennisExplorerResultSnapshot(
        match_date=match_date,
        retrieved_at_utc=observed,
        source_url="https://www.tennisexplorer.com/matches/",
        snapshot_sha256="d" * 64,
        matches=pd.DataFrame(rows),
        html_path=TESTS_DIR / "fixture-cross-source-result.html",
        metadata_path=TESTS_DIR / "fixture-cross-source-result.metadata.json",
    )


def _populate_explorer_mapping(
    database: Path,
    first_seen: date,
    *,
    include_alpha: bool = True,
) -> None:
    """Registra las dos identidades Explorer usadas por el fixture."""

    with PlayerMappingStore(database, clock=lambda: NOW) as store:
        if include_alpha:
            store.put_automatic(
                gender="M",
                slug="alpha-explorer",
                player_id=101,
                visible_name="Alpha A.",
                sackmann_player_name="Alpha A",
                sackmann_ioc=None,
                first_resolved_date=first_seen,
            )
        store.put_automatic(
            gender="M",
            slug="beta-explorer",
            player_id=202,
            visible_name="Beta B.",
            sackmann_player_name="Beta B",
            sackmann_ioc=None,
            first_resolved_date=first_seen,
        )


def _mapped_result(
    match_date: date,
    *,
    available_date: date,
    canonical_match_id: str = "tr:canonical-prior",
    winner_id: int = 101,
    loser_id: int = 202,
) -> MappedResult:
    """Construye un resultado lateral terminal y mapeado sin red."""

    first_seen = datetime.combine(
        available_date,
        datetime.min.time(),
        tzinfo=UTC,
    ).replace(hour=20)
    return MappedResult(
        canonical_match_id=canonical_match_id,
        gender="M",
        effective_date=match_date,
        available_date=available_date,
        first_seen_at_utc=first_seen,
        tournament="Example Open",
        tour_level="ATP",
        surface="Hard",
        round="R32",
        winner_sackmann_id=winner_id,
        loser_sackmann_id=loser_id,
        winner_source_key="tennisratio:Alpha-Player",
        loser_source_key="tennisratio:Beta-Player",
        winner_name="Alpha A.",
        loser_name="Beta B.",
        score="6-4 6-3",
        winner_sets_won=2,
        loser_sets_won=0,
        source_url="https://www.tennisratio.com/players/Alpha-Player.html",
        source_sha256="c" * 64,
    )


class OperationalDailyTests(unittest.TestCase):
    """Comprueba persistencia, scheduler responsable y degradación."""

    def test_two_mornings_register_stats_and_settle_by_slug(self) -> None:
        """La segunda mañana concilia ayer sin confiar en el lado visual."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        calls: list[date] = []
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            first = run_operational_daily_pipeline(
                first_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    _prediction_frame(first_date, "te:prior"),
                    first_date,
                ),
                result_refresher=lambda *_args, **_kwargs: self.fail(
                    "No debía consultar la fecha recién predicha."
                ),
            )
            self.assertIsNone(first.result_date)
            self.assertEqual(first.statistics_registration.input_rows, 24)

            def refresh(selected: date, **_kwargs):
                """Registra la fecha pedida y devuelve evidencia fixture."""

                calls.append(selected)
                return _result_snapshot(selected)

            second = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=refresh,
                tennisratio_result_loader=lambda *_args, **_kwargs: (),
            )
            self.assertEqual(calls, [first_date])
            self.assertEqual(second.result_date, first_date)
            self.assertIsNone(second.reconciliation_warning)
            self.assertEqual(
                second.observation_reconciliation.settlements_inserted,
                1,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                settlement = store.load_settlements().iloc[0]
                self.assertEqual(settlement["winner_slug"], "alpha-player")
                self.assertEqual(settlement["actual_outcome_a"], 1)

    def test_empty_refresh_rotates_to_another_unobserved_date(self) -> None:
        """Una página vacía queda auditada y no atasca el scheduler."""

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-old",
                    _prediction_frame(date(2026, 7, 28), "te:old"),
                    target_date=date(2026, 7, 28),
                )
                store.register_prediction_run(
                    "prediction-new",
                    _prediction_frame(date(2026, 7, 29), "te:new"),
                    target_date=date(2026, 7, 29),
                )
                self.assertEqual(
                    store.select_pending_result_date(before_date=date(2026, 7, 30)),
                    date(2026, 7, 29),
                )
                store.reconcile_observations(
                    "observation-empty-new",
                    pd.DataFrame(),
                    observed_at_utc=NOW,
                    target_date=date(2026, 7, 29),
                )
                self.assertEqual(
                    store.select_pending_result_date(before_date=date(2026, 7, 30)),
                    date(2026, 7, 28),
                )

    def test_scheduler_uses_latest_run_or_observation_timestamp(self) -> None:
        """La prioridad usa el maximo real, no COALESCE sesgado al run."""

        mutable_now = {"value": datetime(2026, 7, 30, 8, tzinfo=UTC)}
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(
                database,
                clock=lambda: mutable_now["value"],
            ) as store:
                for selected, match_id in (
                    (date(2026, 7, 27), "te:older-run"),
                    (date(2026, 7, 28), "te:newer-run"),
                ):
                    store.register_prediction_run(
                        f"prediction-{selected}",
                        _prediction_frame(selected, match_id),
                        target_date=selected,
                    )

                observation_columns = {
                    "status": "scheduled",
                    "player_1_sets_won": None,
                    "player_2_sets_won": None,
                    "sets_score": None,
                    "winner_side": None,
                    "winner_slug": None,
                    "result_evidence": None,
                }
                mutable_now["value"] = datetime(2026, 7, 30, 10, tzinfo=UTC)
                store.reconcile_observations(
                    "observation-first",
                    pd.DataFrame(
                        [
                            {
                                "source_match_id": "te:older-run",
                                **observation_columns,
                            }
                        ]
                    ),
                    observed_at_utc=datetime(2026, 7, 30, 12, tzinfo=UTC),
                    target_date=date(2026, 7, 27),
                )
                mutable_now["value"] = datetime(2026, 7, 30, 11, tzinfo=UTC)
                store.reconcile_observations(
                    "observation-second",
                    pd.DataFrame(
                        [
                            {
                                "source_match_id": "te:newer-run",
                                **observation_columns,
                            }
                        ]
                    ),
                    observed_at_utc=datetime(2026, 7, 30, 11, 30, tzinfo=UTC),
                    target_date=date(2026, 7, 28),
                )
                self.assertEqual(
                    store.select_pending_result_date(before_date=date(2026, 7, 30)),
                    date(2026, 7, 28),
                )

    def test_result_http_failure_keeps_the_daily_prediction(self) -> None:
        """Un fallo de conciliación se informa sin borrar el trabajo diario."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    _prediction_frame(first_date, "te:prior"),
                    target_date=first_date,
                )

            def fail_refresh(_selected: date, **_kwargs):
                """Simula un único fallo HTTP controlado."""

                raise TennisExplorerHttpError("sin conexión")

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    _prediction_frame(second_date, "te:today"),
                    second_date,
                ),
                result_refresher=fail_refresh,
                tennisratio_result_loader=lambda *_args, **_kwargs: (),
            )
            self.assertEqual(result.result_date, first_date)
            self.assertIn("TennisExplorerHttpError", result.reconciliation_warning)
            with OperationsStore(database, clock=lambda: NOW) as store:
                count = store.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                self.assertEqual(count, 2)

    def test_tennisratio_complete_result_avoids_explorer_network(self) -> None:
        """La fuente lateral completa se concilia sin tocar el fallback web."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        loader_calls: list[tuple[date, dict[str, date]]] = []
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    _prediction_frame(first_date, "te:prior"),
                    target_date=first_date,
                )

            def load_results(selected: date, **kwargs):
                """Registra el corte causal solicitado y devuelve evidencia."""

                loader_calls.append((selected, dict(kwargs["base_cutoff_by_gender"])))
                return (
                    _mapped_result(
                        first_date,
                        available_date=first_date,
                    ),
                )

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=lambda *_args, **_kwargs: self.fail(
                    "Tennis Explorer no debe consultarse con lote completo."
                ),
                tennisratio_result_loader=load_results,
            )

            self.assertEqual(loader_calls[0][0], second_date)
            self.assertEqual(
                loader_calls[0][1],
                {"M": date(2026, 7, 28), "F": date(2026, 7, 28)},
            )
            self.assertIsNone(result.reconciliation_warning)
            self.assertEqual(
                result.observation_reconciliation.settlements_inserted,
                1,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                observation = store.connection.execute(
                    """
                    SELECT source_system, winner_slug, payload_json
                    FROM observations
                    """
                ).fetchone()
                self.assertEqual(observation["source_system"], "tennisratio")
                self.assertEqual(observation["winner_slug"], "alpha-player")
                self.assertIn(
                    "tennisratio:Alpha-Player",
                    observation["payload_json"],
                )

    def test_tennisratio_owned_agenda_settles_safe_partial_coverage(self) -> None:
        """Un resultado inequívoco no se pierde porque falte otro del lote."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        second_prediction = _prediction_frame(
            first_date,
            "tennisratio:M:other",
        )
        replacements = {
            "player_a_name": "Gamma G.",
            "player_b_name": "Delta D.",
            "player_a_slug": "tennisratio:Gamma",
            "player_b_slug": "tennisratio:Delta",
            "player_a_id": 303,
            "player_b_id": 404,
        }
        for column, value in replacements.items():
            second_prediction.loc[0, column] = value
        predictions = pd.concat(
            [
                _prediction_frame(first_date, "tennisratio:M:prior"),
                second_prediction,
            ],
            ignore_index=True,
        )

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    predictions,
                    target_date=first_date,
                )

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=lambda *_args, **_kwargs: self.fail(
                    "Una coincidencia TennisRatio segura no debe descartarse."
                ),
                tennisratio_result_loader=lambda *_args, **_kwargs: (
                    _mapped_result(first_date, available_date=first_date),
                ),
            )

            self.assertIsInstance(
                result.result_snapshot,
                TennisRatioResultSnapshot,
            )
            snapshot = result.result_snapshot
            assert isinstance(snapshot, TennisRatioResultSnapshot)
            self.assertEqual(snapshot.official_pending_count, 2)
            self.assertEqual(snapshot.matched_count, 1)
            self.assertEqual(snapshot.unmatched_count, 1)
            self.assertEqual(
                result.observation_reconciliation.settlements_inserted,
                1,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM settlements").fetchone()[0],
                    1,
                )
                metadata = store.connection.execute(
                    "SELECT metadata_json FROM runs WHERE run_type = 'observation'"
                ).fetchone()["metadata_json"]
                self.assertIn('"matched_count":1', metadata)
                self.assertIn('"unmatched_count":1', metadata)

    def test_incomplete_tennisratio_batch_uses_explorer_fallback(self) -> None:
        """Una cobertura lateral parcial no escribe un lote a medias."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        second_prediction = _prediction_frame(first_date, "te:other")
        replacements = {
            "player_a_name": "Gamma G.",
            "player_b_name": "Delta D.",
            "player_a_slug": "gamma-player",
            "player_b_slug": "delta-player",
            "player_a_id": 303,
            "player_b_id": 404,
        }
        for column, value in replacements.items():
            second_prediction.loc[0, column] = value
        predictions = pd.concat(
            [_prediction_frame(first_date, "te:prior"), second_prediction],
            ignore_index=True,
        )
        fallback_calls: list[date] = []
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    predictions,
                    target_date=first_date,
                )

            def refresh(selected: date, **_kwargs):
                """Devuelve el snapshot de respaldo y registra la llamada."""

                fallback_calls.append(selected)
                return _result_snapshot(selected)

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=refresh,
                tennisratio_result_loader=lambda *_args, **_kwargs: (
                    _mapped_result(first_date, available_date=first_date),
                ),
            )

            self.assertEqual(fallback_calls, [first_date])
            self.assertIsInstance(
                result.result_snapshot,
                TennisExplorerResultSnapshot,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                sources = store.connection.execute(
                    "SELECT DISTINCT source_system FROM observations"
                ).fetchall()
                self.assertEqual(
                    [row["source_system"] for row in sources],
                    ["tennis_explorer"],
                )

    def test_explorer_fallback_maps_cross_source_identity_and_settles(self) -> None:
        """Explorer se une por IDs Sackmann y conserva toda la procedencia."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        prediction = _prediction_frame(first_date, "tennisratio:M:259356")
        prediction.loc[0, "player_a_slug"] = "tennisratio:Alpha"
        prediction.loc[0, "player_b_slug"] = "tennisratio:Beta"

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            root = Path(directory)
            database = root / "tennis.sqlite3"
            mapping_database = root / "player_mapping.sqlite3"
            _populate_explorer_mapping(mapping_database, first_date)
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    prediction,
                    target_date=first_date,
                )

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=lambda selected, **_kwargs: _cross_source_result_snapshot(
                    selected
                ),
                tennisratio_result_loader=lambda *_args, **_kwargs: (),
                player_mapping_database_path=mapping_database,
            )

            self.assertIsInstance(
                result.result_snapshot,
                TennisExplorerMappedResultSnapshot,
            )
            snapshot = result.result_snapshot
            assert isinstance(snapshot, TennisExplorerMappedResultSnapshot)
            self.assertEqual(snapshot.exact_id_count, 0)
            self.assertEqual(snapshot.identity_mapped_count, 1)
            self.assertEqual(snapshot.unmatched_official_count, 0)
            self.assertEqual(
                result.observation_reconciliation.settlements_inserted,
                1,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                observation = store.connection.execute(
                    "SELECT source_match_id, source_system, winner_slug, payload_json "
                    "FROM observations"
                ).fetchone()
                self.assertEqual(
                    observation["source_match_id"],
                    "tennisratio:M:259356",
                )
                self.assertEqual(observation["source_system"], "tennis_explorer")
                self.assertEqual(observation["winner_slug"], "tennisratio:Beta")
                self.assertIn(
                    '"tennis_explorer_source_match_id":"3307289"', observation["payload_json"]
                )
                settlement = store.connection.execute(
                    "SELECT winner_slug, actual_outcome_a FROM settlements"
                ).fetchone()
                self.assertEqual(settlement["winner_slug"], "tennisratio:Beta")
                self.assertEqual(settlement["actual_outcome_a"], 0)

    def test_explorer_cross_source_duplicate_pair_is_not_settled(self) -> None:
        """Dos resultados para la misma pareja y fecha quedan ambiguos."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        prediction = _prediction_frame(first_date, "tennisratio:M:259356")
        prediction.loc[0, "player_a_slug"] = "tennisratio:Alpha"
        prediction.loc[0, "player_b_slug"] = "tennisratio:Beta"

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            root = Path(directory)
            database = root / "tennis.sqlite3"
            mapping_database = root / "player_mapping.sqlite3"
            _populate_explorer_mapping(mapping_database, first_date)
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    prediction,
                    target_date=first_date,
                )

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=lambda selected, **_kwargs: _cross_source_result_snapshot(
                    selected, duplicate=True
                ),
                tennisratio_result_loader=lambda *_args, **_kwargs: (),
                player_mapping_database_path=mapping_database,
            )

            snapshot = result.result_snapshot
            assert isinstance(snapshot, TennisExplorerMappedResultSnapshot)
            self.assertEqual(snapshot.identity_mapped_count, 0)
            self.assertEqual(snapshot.ambiguous_pair_rows, 2)
            self.assertEqual(
                result.observation_reconciliation.settlements_inserted,
                0,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM settlements").fetchone()[0],
                    0,
                )

    def test_explorer_single_identity_anchors_unique_official_pair(self) -> None:
        """Una identidad exacta permite completar solo una pareja oficial Ãºnica."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        prediction = _prediction_frame(first_date, "tennisratio:M:259356")
        prediction.loc[0, "player_a_slug"] = "tennisratio:Alpha"
        prediction.loc[0, "player_b_slug"] = "tennisratio:Beta"

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            root = Path(directory)
            database = root / "tennis.sqlite3"
            mapping_database = root / "player_mapping.sqlite3"
            _populate_explorer_mapping(
                mapping_database,
                first_date,
                include_alpha=False,
            )
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    prediction,
                    target_date=first_date,
                )

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=lambda selected, **_kwargs: _cross_source_result_snapshot(
                    selected
                ),
                tennisratio_result_loader=lambda *_args, **_kwargs: (),
                player_mapping_database_path=mapping_database,
            )

            snapshot = result.result_snapshot
            assert isinstance(snapshot, TennisExplorerMappedResultSnapshot)
            self.assertEqual(snapshot.identity_mapped_count, 1)
            self.assertEqual(snapshot.paired_inferred_count, 1)
            self.assertEqual(snapshot.unmatched_official_count, 0)
            with OperationsStore(database, clock=lambda: NOW) as store:
                observation = store.connection.execute(
                    "SELECT payload_json FROM observations"
                ).fetchone()
                self.assertIn(
                    "single_sackmann_id_official_pair_complement",
                    observation["payload_json"],
                )
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM settlements").fetchone()[0],
                    1,
                )

    def test_stored_explorer_result_replay_is_append_only_and_idempotent(self) -> None:
        """El replay conserva la fila cruda y aÃ±ade una liquidaciÃ³n una vez."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        prediction = _prediction_frame(first_date, "tennisratio:M:259356")
        prediction.loc[0, "player_a_slug"] = "tennisratio:Alpha"
        prediction.loc[0, "player_b_slug"] = "tennisratio:Beta"
        raw_snapshot = _cross_source_result_snapshot(first_date)

        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            root = Path(directory)
            database = root / "tennis.sqlite3"
            mapping_database = root / "player_mapping.sqlite3"
            _populate_explorer_mapping(mapping_database, first_date)
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    prediction,
                    target_date=first_date,
                )
                raw_reconciliation = store.reconcile_observations(
                    "raw-explorer-run",
                    raw_snapshot.matches,
                    source_system="tennis_explorer",
                    metadata={
                        "pipeline": "daily_result_reconciliation_v1",
                        "html_path": "tests/fixture-cross-source-result.html",
                        "metadata_path": ("tests/fixture-cross-source-result.metadata.json"),
                        "snapshot_sha256": raw_snapshot.snapshot_sha256,
                        "source_url": raw_snapshot.source_url,
                    },
                    observed_at_utc=raw_snapshot.retrieved_at_utc,
                    target_date=first_date,
                )
                self.assertEqual(raw_reconciliation.settlements_inserted, 0)

            first = reconcile_stored_tennis_explorer_results(
                second_date,
                database_path=database,
                mapping_database_path=mapping_database,
                clock=lambda: NOW,
            )
            second = reconcile_stored_tennis_explorer_results(
                second_date,
                database_path=database,
                mapping_database_path=mapping_database,
                clock=lambda: NOW,
            )

            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].reconciliation.settlements_inserted, 1)
            self.assertEqual(second, ())
            with OperationsStore(database, clock=lambda: NOW) as store:
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
                    2,
                )
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM settlements").fetchone()[0],
                    1,
                )
                trigger_count = store.connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM sqlite_master
                    WHERE type = 'trigger'
                      AND name IN (
                        'predictions_no_update',
                        'predictions_no_delete',
                        'observations_no_update',
                        'observations_no_delete',
                        'settlements_no_update',
                        'settlements_no_delete'
                      )
                    """
                ).fetchone()[0]
                self.assertEqual(trigger_count, 6)

    def test_same_day_available_tennisratio_result_is_not_visible(self) -> None:
        """Una observación disponible en D no es elegible hasta D+1."""

        first_date = date(2026, 7, 29)
        second_date = date(2026, 7, 30)
        fallback_calls: list[date] = []
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / "tennis.sqlite3"
            with OperationsStore(database, clock=lambda: NOW) as store:
                store.register_prediction_run(
                    "prediction-prior",
                    _prediction_frame(first_date, "te:prior"),
                    target_date=first_date,
                )

            def refresh(selected: date, **_kwargs):
                """Confirma que la evidencia D se descartó antes del fallback."""

                fallback_calls.append(selected)
                return _result_snapshot(selected)

            result = run_operational_daily_pipeline(
                second_date,
                clock=lambda: NOW,
                database_path=database,
                publish=False,
                daily_runner=_daily_runner(
                    pd.DataFrame(columns=["model_probability_a"]),
                    second_date,
                ),
                result_refresher=refresh,
                tennisratio_result_loader=lambda *_args, **_kwargs: (
                    _mapped_result(first_date, available_date=second_date),
                ),
            )

            self.assertEqual(fallback_calls, [first_date])
            self.assertIsInstance(
                result.result_snapshot,
                TennisExplorerResultSnapshot,
            )

    def test_launcher_trains_by_default_and_keeps_required_order(self) -> None:
        """El lanzador actualiza, reconstruye y entrena antes del diario."""

        launcher = (PROJECT_ROOT / "run_tennis.ps1").read_text(encoding="utf-8")
        self.assertIn("[switch]$Retrain", launcher)
        self.assertIn("[string]$Date", launcher)
        retrain_order = (
            "scripts\\update_sources.py",
            "scripts\\audit_identities.py",
            "scripts\\build_elo.py",
            "scripts\\build_features.py",
            "scripts\\retrain_models.py",
        )
        positions = [launcher.index(item) for item in retrain_order]
        positions.append(launcher.index("& $PythonExecutable $DailyScript"))
        self.assertEqual(positions, sorted(positions))
        self.assertIn("((-not [bool]$Date) -or $Retrain)", launcher)
        training_block = launcher.split("$TrainingScripts = @(", 1)[1].split(
            "$DailyArguments = @()", 1
        )[0]
        self.assertNotIn("if ($Retrain)", training_block)
        self.assertIn("$DailyArguments += '--retrained'", launcher)
        self.assertIn("'WEB\\build_web.py'", launcher)
        self.assertGreater(
            launcher.index("& $PythonExecutable $WebBuildScript"),
            launcher.index("& $PythonExecutable $DailyScript"),
        )
        self.assertIn("if ($DailyExitCode -ne 0)", launcher)
        self.assertNotIn("start.ps1", launcher)


if __name__ == "__main__":
    unittest.main()

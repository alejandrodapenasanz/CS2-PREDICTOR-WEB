"""Tests offline de la orquestación diaria, SQLite y conciliación."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.config import PROJECT_ROOT, TESTS_DIR
from src.daily_pipeline import DailyPredictionRun
from src.operations import OperationsStore, run_operational_daily_pipeline
from src.tennis_explorer import (
    TennisExplorerHttpError,
    TennisExplorerResultSnapshot,
)
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
        "model_training_max_date": match_date.replace(
            year=match_date.year - 1
        ),
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
                    store.select_pending_result_date(
                        before_date=date(2026, 7, 30)
                    ),
                    date(2026, 7, 29),
                )
                store.reconcile_observations(
                    "observation-empty-new",
                    pd.DataFrame(),
                    observed_at_utc=NOW,
                    target_date=date(2026, 7, 29),
                )
                self.assertEqual(
                    store.select_pending_result_date(
                        before_date=date(2026, 7, 30)
                    ),
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
                mutable_now["value"] = datetime(
                    2026, 7, 30, 10, tzinfo=UTC
                )
                store.reconcile_observations(
                    "observation-first",
                    pd.DataFrame([
                        {
                            "source_match_id": "te:older-run",
                            **observation_columns,
                        }
                    ]),
                    observed_at_utc=datetime(
                        2026, 7, 30, 12, tzinfo=UTC
                    ),
                    target_date=date(2026, 7, 27),
                )
                mutable_now["value"] = datetime(
                    2026, 7, 30, 11, tzinfo=UTC
                )
                store.reconcile_observations(
                    "observation-second",
                    pd.DataFrame([
                        {
                            "source_match_id": "te:newer-run",
                            **observation_columns,
                        }
                    ]),
                    observed_at_utc=datetime(
                        2026, 7, 30, 11, 30, tzinfo=UTC
                    ),
                    target_date=date(2026, 7, 28),
                )
                self.assertEqual(
                    store.select_pending_result_date(
                        before_date=date(2026, 7, 30)
                    ),
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
            )
            self.assertEqual(result.result_date, first_date)
            self.assertIn("TennisExplorerHttpError", result.reconciliation_warning)
            with OperationsStore(database, clock=lambda: NOW) as store:
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM predictions"
                ).fetchone()[0]
                self.assertEqual(count, 2)

    def test_launcher_exposes_retrain_and_keeps_required_order(self) -> None:
        """El lanzador actualiza, reconstruye y entrena antes del diario."""

        launcher = (PROJECT_ROOT / "run_tennis.ps1").read_text(
            encoding="utf-8"
        )
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
        self.assertIn("'WEB\\build_web.py'", launcher)
        self.assertGreater(
            launcher.index("& $PythonExecutable $WebBuildScript"),
            launcher.index("& $PythonExecutable $DailyScript"),
        )
        self.assertIn("if ($DailyExitCode -ne 0)", launcher)
        self.assertNotIn("start.ps1", launcher)


if __name__ == "__main__":
    unittest.main()

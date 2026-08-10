"""Tests de persistencia, idempotencia y conciliación por slug.

Los tests usan bases temporales dentro de ``TENNIS/tests`` para verificar
también la barrera que impide escribir la SQLite fuera del proyecto.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
import math
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import Iterator
import unittest

import pandas as pd

from src.config import OPERATIONS_DATABASE_PATH, PROJECT_ROOT, TESTS_DIR
from src.operations import (
    OperationsConflictError,
    OperationsStore,
    OperationsValidationError,
    SCHEMA_VERSION,
    derive_source_match_id,
)
from src.operations.schema import SCHEMA_SQL


FIXED_NOW = datetime(2026, 7, 30, 8, 0, tzinfo=UTC)


@contextmanager
def assert_raises(
    exception_type: type[BaseException],
    *,
    match: str | None = None,
) -> Iterator[None]:
    """Comprueba una excepción y, opcionalmente, su mensaje con stdlib."""

    try:
        yield
    except exception_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"El mensaje {str(exc)!r} no coincide con {match!r}."
            ) from exc
    else:
        raise AssertionError(
            f"No se lanzó la excepción {exception_type.__name__}."
        )


@contextmanager
def temporary_store() -> Iterator[OperationsStore]:
    """Abre y elimina una base aislada situada dentro de TENNIS."""

    with TemporaryDirectory(dir=TESTS_DIR) as directory:
        database_path = Path(directory) / "operations.sqlite3"
        with OperationsStore(
            database_path,
            clock=lambda: FIXED_NOW,
        ) as store:
            yield store


def prediction_row(
    *,
    source_match_id: str = "te:match-1",
    player_a_slug: object = "alpha-player",
    player_b_slug: object = "beta-player",
    probability_a: object = 0.62,
    probability_b: object = 0.38,
    prediction_as_of_utc: str = "2026-07-30T07:05:00+00:00",
    source_retrieved_at_utc: str = "2026-07-30T07:00:00+00:00",
    status: str = "scheduled",
    mapping_status: str = "mapped",
    prediction_status: str = "predicted",
    model_fingerprint: object = "model-m-v1",
) -> dict[str, object]:
    """Construye una predicción válida salvo los overrides indicados."""

    return {
        "source_match_id": source_match_id,
        "prediction_date": "2026-07-30",
        "prediction_as_of_utc": prediction_as_of_utc,
        "source_retrieved_at_utc": source_retrieved_at_utc,
        "source_snapshot_sha256": "a" * 64,
        "tournament": "Example Open",
        "tournament_href": "/example-open/",
        "tour_level": "ATP",
        "gender": "M",
        "status": status,
        "player_a_name": "Alpha A.",
        "player_b_name": "Beta B.",
        "player_a_slug": player_a_slug,
        "player_b_slug": player_b_slug,
        "player_a_id": 101,
        "player_b_id": 202,
        "mapping_status": mapping_status,
        "model_probability_raw_a": probability_a,
        "model_probability_a": probability_a,
        "model_probability_b": probability_b,
        "market_probability_a": 0.55,
        "market_probability_b": 0.45,
        "edge_a": 0.07,
        "edge_b": -0.07,
        "confidence": "medium",
        "confidence_flags": "",
        "prediction_status": prediction_status,
        "model_profile": "M",
        "model_fingerprint": model_fingerprint,
        "model_training_max_date": "2026-07-01",
        "model_training_available_max_date": "2026-07-22",
        "feature_fingerprint": "features-v1",
    }


def result_row(
    *,
    source_match_id: str = "te:match-1",
    winner_slug: str = "alpha-player",
    winner_side: str = "player_1",
    first_sets: int = 2,
    second_sets: int = 0,
    observed_at_utc: str = "2026-07-30T12:00:00+00:00",
) -> dict[str, object]:
    """Construye evidencia terminal explícita para una observación."""

    return {
        "source_match_id": source_match_id,
        "status": "finished",
        "player_1_sets_won": first_sets,
        "player_2_sets_won": second_sets,
        "sets_score": "6-4 6-3",
        "winner_side": winner_side,
        "winner_slug": winner_slug,
        "result_evidence": "terminal_sets_and_score",
        "observed_at_utc": observed_at_utc,
        "source_snapshot_sha256": "b" * 64,
    }


def test_schema_is_versioned_and_default_path_is_inside_tennis() -> None:
    """El esquema y la ruta productiva quedan versionados dentro del proyecto."""

    assert OPERATIONS_DATABASE_PATH == (
        PROJECT_ROOT / "BBDD" / "tennis.sqlite3"
    )
    with temporary_store() as store:
        version = store.connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        recorded = store.connection.execute(
            "SELECT version FROM schema_versions"
        ).fetchone()[0]
        assert version == SCHEMA_VERSION == recorded == 2


def test_schema_migrates_v1_and_unversioned_databases() -> None:
    """Bases v1 o completas sin PRAGMA se elevan a v2 sin perder filas."""

    legacy_sql = SCHEMA_SQL.replace(
        "    model_training_available_max_date TEXT,\n",
        "",
    )
    for mode in ("v1", "unversioned"):
        with TemporaryDirectory(dir=TESTS_DIR) as directory:
            database = Path(directory) / f"{mode}.sqlite3"
            connection = sqlite3.connect(database)
            connection.executescript(
                legacy_sql if mode == "v1" else SCHEMA_SQL
            )
            if mode == "v1":
                connection.execute(
                    "INSERT INTO schema_versions VALUES (1, ?, ?)",
                    (FIXED_NOW.isoformat(), "legacy schema"),
                )
                connection.execute("PRAGMA user_version = 1")
            connection.commit()
            connection.close()

            with OperationsStore(database, clock=lambda: FIXED_NOW) as store:
                columns = {
                    str(row[1])
                    for row in store.connection.execute(
                        "PRAGMA table_info(predictions)"
                    )
                }
                versions = {
                    int(row[0])
                    for row in store.connection.execute(
                        "SELECT version FROM schema_versions"
                    )
                }
                assert "model_training_available_max_date" in columns
                assert store.connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0] == 2
                assert 2 in versions


def test_empty_runs_are_supported_and_idempotent() -> None:
    """Una mañana sin partidos genera una ejecución completa reutilizable."""

    with temporary_store() as store:
        first = store.register_prediction_run(
            "prediction-empty",
            pd.DataFrame(),
        )
        second = store.register_prediction_run(
            "prediction-empty",
            pd.DataFrame(),
        )
        observation = store.reconcile_observations(
            "observation-empty",
            pd.DataFrame(),
        )

        assert first.input_rows == 0
        assert first.predictions_inserted == 0
        assert not first.reused
        assert second.reused
        assert observation.input_rows == 0
        assert observation.observations_inserted == 0


def test_first_valid_prediction_is_official_and_immutable() -> None:
    """Una fila inválida no reclama la oficial y una válida posterior no la pisa."""

    invalid = prediction_row(
        player_a_slug=None,
        player_b_slug=None,
        mapping_status="unmapped",
        probability_a=None,
        probability_b=None,
        model_fingerprint=None,
    )
    first_valid = prediction_row(probability_a=0.62, probability_b=0.38)
    later_valid = prediction_row(probability_a=0.91, probability_b=0.09)

    with temporary_store() as store:
        invalid_summary = store.register_prediction_run(
            "prediction-invalid",
            pd.DataFrame([invalid]),
        )
        first_summary = store.register_prediction_run(
            "prediction-first-valid",
            pd.DataFrame([first_valid]),
        )
        later_summary = store.register_prediction_run(
            "prediction-later-valid",
            pd.DataFrame([later_valid]),
        )
        official = store.load_official_predictions().iloc[0]

        assert invalid_summary.valid_predictions == 0
        assert invalid_summary.official_predictions_selected == 0
        assert first_summary.official_predictions_selected == 1
        assert later_summary.official_predictions_selected == 0
        assert official["run_id"] == "prediction-first-valid"
        assert math.isclose(
            float(official["model_probability_a"]),
            0.62,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        with assert_raises(sqlite3.IntegrityError, match="inmutables"):
            store.connection.execute(
                """
                UPDATE predictions
                SET model_probability_a = 0.1
                WHERE prediction_id = ?
                """,
                (official["prediction_id"],),
            )
        with assert_raises(sqlite3.IntegrityError, match="inmutables"):
            store.connection.execute(
                """
                DELETE FROM official_predictions
                WHERE source_match_id = 'te:match-1'
                """
            )


def test_settlement_uses_winner_slug_not_source_position() -> None:
    """Un resultado invertido respecto a A/B conserva el label correcto."""

    prediction = prediction_row(
        player_a_slug="alpha-player",
        player_b_slug="beta-player",
    )
    reversed_source_result = result_row(
        winner_slug="alpha-player",
        winner_side="player_2",
        first_sets=0,
        second_sets=2,
    )

    with temporary_store() as store:
        store.register_prediction_run(
            "prediction-1",
            pd.DataFrame([prediction]),
        )
        summary = store.reconcile_observations(
            "observation-reversed",
            pd.DataFrame([reversed_source_result]),
        )
        settlement = store.load_settlements().iloc[0]

        assert summary.settlements_inserted == 1
        assert settlement["winner_slug"] == "alpha-player"
        assert settlement["actual_outcome_a"] == 1


def test_observation_before_registration_blocks_official_prediction() -> None:
    """Una observación ya conocida impide crear un backtest postdicto."""

    with temporary_store() as store:
        observation = store.reconcile_observations(
            "observation-before-prediction",
            pd.DataFrame([result_row()]),
        )
        assert observation.queued_rows == 1
        assert store.load_settlements().empty

        prediction = store.register_prediction_run(
            "prediction-after-observation",
            pd.DataFrame([prediction_row()]),
        )
        queues = store.load_review_queue()
        stored = store.connection.execute(
            """
            SELECT is_valid, invalid_reason FROM predictions
            WHERE run_id = 'prediction-after-observation'
            """
        ).fetchone()

        assert prediction.valid_predictions == 0
        assert prediction.official_predictions_selected == 0
        assert prediction.settlements_inserted == 0
        assert store.load_official_predictions().empty
        assert store.load_settlements().empty
        assert stored["is_valid"] == 0
        assert "observation_already_exists" in stored["invalid_reason"]
        assert set(queues["status"]) == {"open"}


def test_historical_replay_is_stored_but_never_official() -> None:
    """Captura o predicción posterior al día objetivo queda no oficial."""

    replay = prediction_row(
        prediction_as_of_utc="2026-07-31T07:05:00+00:00",
        source_retrieved_at_utc="2026-07-31T07:00:00+00:00",
    )
    with temporary_store() as store:
        summary = store.register_prediction_run(
            "historical-replay",
            pd.DataFrame([replay]),
        )
        stored = store.connection.execute(
            "SELECT is_valid, invalid_reason FROM predictions"
        ).fetchone()

        assert summary.predictions_inserted == 1
        assert summary.valid_predictions == 0
        assert summary.official_predictions_selected == 0
        assert stored["is_valid"] == 0
        assert "prediction_created_after_match_date" in stored["invalid_reason"]
        assert "source_captured_after_match_date" in stored["invalid_reason"]


def test_model_training_cutoff_is_required_and_strictly_past() -> None:
    """Sin corte acreditado o con corte en D no hay predicción oficial."""

    missing = prediction_row(source_match_id="te:missing-cutoff")
    missing["model_training_max_date"] = None
    same_day = prediction_row(source_match_id="te:same-day-cutoff")
    same_day["model_training_max_date"] = "2026-07-09"
    same_day["model_training_available_max_date"] = "2026-07-30"
    inconsistent = prediction_row(source_match_id="te:inconsistent-cutoff")
    inconsistent["model_training_available_max_date"] = "2026-07-21"
    with temporary_store() as store:
        summary = store.register_prediction_run(
            "invalid-model-cutoffs",
            pd.DataFrame([missing, same_day, inconsistent]),
        )
        reasons = {
            row["source_match_id"]: row["invalid_reason"]
            for row in store.connection.execute(
                """
                SELECT source_match_id, invalid_reason
                FROM predictions ORDER BY source_match_id
                """
            )
        }

        assert summary.valid_predictions == 0
        assert summary.official_predictions_selected == 0
        assert "model_training_max_date_missing" in reasons[
            "te:missing-cutoff"
        ]
        assert "model_results_not_available_before_match" in reasons[
            "te:same-day-cutoff"
        ]
        assert "model_training_dates_inconsistent" in reasons[
            "te:inconsistent-cutoff"
        ]

    absent = prediction_row()
    absent.pop("model_training_max_date")
    with temporary_store() as store:
        with assert_raises(
            OperationsValidationError,
            match="model_training_max_date",
        ):
            store.register_prediction_run(
                "missing-required-column",
                pd.DataFrame([absent]),
            )


def test_observation_must_be_strictly_after_prediction_to_settle() -> None:
    """Un resultado con timestamp anterior o igual nunca crea un label."""

    with temporary_store() as store:
        store.register_prediction_run(
            "prediction-before-result",
            pd.DataFrame([prediction_row()]),
        )
        summary = store.reconcile_observations(
            "observation-with-old-clock",
            pd.DataFrame(
                [
                    result_row(
                        observed_at_utc="2026-07-30T07:05:00+00:00"
                    )
                ]
            ),
        )
        queue = store.load_review_queue(status="open")

        assert summary.settlements_inserted == 0
        assert summary.queued_rows == 1
        assert store.load_settlements().empty
        assert set(queue["issue_type"]) == {
            "observation_not_after_prediction"
        }


def test_winner_outside_official_slugs_is_conflict_not_label() -> None:
    """Un slug ajeno nunca se convierte en label por lado o nombre."""

    with temporary_store() as store:
        store.register_prediction_run(
            "prediction-1",
            pd.DataFrame([prediction_row()]),
        )
        summary = store.reconcile_observations(
            "observation-wrong-slug",
            pd.DataFrame(
                [
                    result_row(
                        winner_slug="different-player",
                        winner_side="player_1",
                    )
                ]
            ),
        )

        assert summary.settlements_inserted == 0
        assert summary.conflicts_inserted == 1
        assert store.load_settlements().empty
        queue = store.load_review_queue(status="open").iloc[0]
        assert (
            queue["issue_type"]
            == "winner_slug_not_in_official_prediction"
        )


def test_existing_settlement_never_changes_on_later_conflict() -> None:
    """Un resultado posterior contradictorio se audita sin sobrescribir."""

    with temporary_store() as store:
        store.register_prediction_run(
            "prediction-1",
            pd.DataFrame([prediction_row()]),
        )
        store.reconcile_observations(
            "observation-alpha",
            pd.DataFrame([result_row(winner_slug="alpha-player")]),
        )
        conflicting = store.reconcile_observations(
            "observation-beta",
            pd.DataFrame(
                [
                    result_row(
                        winner_slug="beta-player",
                        winner_side="player_2",
                        first_sets=0,
                        second_sets=2,
                    )
                ]
            ),
        )
        settlement = store.load_settlements().iloc[0]

        assert conflicting.conflicts_inserted == 1
        assert settlement["winner_slug"] == "alpha-player"
        assert settlement["actual_outcome_a"] == 1


def test_source_match_fallback_is_stable_under_orientation_and_noise() -> None:
    """La identidad fallback ignora A/B, nombres, cuotas y timestamps."""

    first = prediction_row()
    first.pop("source_match_id")
    second = dict(first)
    second["player_a_slug"] = first["player_b_slug"]
    second["player_b_slug"] = first["player_a_slug"]
    second["player_a_name"] = "Texto cambiado"
    second["player_b_name"] = "Otro texto"
    second["model_probability_a"] = 0.2
    second["model_probability_b"] = 0.8
    second["prediction_as_of_utc"] = "2026-07-30T07:30:00+00:00"

    assert derive_source_match_id(first) == derive_source_match_id(second)


def test_divergent_run_id_is_audited_and_rejected() -> None:
    """El mismo run_id no puede ocultar un payload distinto."""

    with temporary_store() as store:
        store.register_prediction_run(
            "same-run",
            pd.DataFrame([prediction_row()]),
        )
        with assert_raises(OperationsConflictError):
            store.register_prediction_run(
                "same-run",
                pd.DataFrame(
                    [
                        prediction_row(
                            probability_a=0.7,
                            probability_b=0.3,
                        )
                    ]
                ),
            )
        conflict = store.connection.execute(
            """
            SELECT conflict_type FROM conflicts
            WHERE run_id = 'same-run'
            """
        ).fetchone()
        assert conflict["conflict_type"] == "divergent_run_payload"


def test_unmapped_row_with_explicit_identity_is_stored_but_not_official() -> None:
    """Un jugador no mapeado no rompe el run ni recibe probabilidad inventada."""

    row = prediction_row(
        player_a_slug=None,
        player_b_slug=None,
        probability_a=None,
        probability_b=None,
        mapping_status="unmapped",
        prediction_status="not_predicted",
        model_fingerprint=None,
    )
    with temporary_store() as store:
        summary = store.register_prediction_run(
            "unmapped-run",
            pd.DataFrame([row]),
        )
        stored = store.connection.execute(
            "SELECT is_valid, invalid_reason FROM predictions"
        ).fetchone()

        assert summary.predictions_inserted == 1
        assert summary.official_predictions_selected == 0
        assert stored["is_valid"] == 0
        assert "mapping_not_complete" in stored["invalid_reason"]


def test_statistics_are_append_only_and_reject_prediction_signals() -> None:
    """La tabla estadística no admite probabilidades, labels ni resultados."""

    valid_statistic = {
        "source_match_id": "te:match-1",
        "player_slug": "alpha-player",
        "player_id": 101,
        "gender": "M",
        "as_of_date": "2026-07-30",
        "statistic_name": "elo_general",
        "statistic_value": 1875.0,
        "source_kind": "historical_matches",
        "source_snapshot_id": None,
    }
    forbidden_statistic = dict(valid_statistic)
    forbidden_statistic["statistic_name"] = "model_probability_a"

    with temporary_store() as store:
        store.register_prediction_run(
            "prediction-with-stats",
            pd.DataFrame([prediction_row()]),
        )
        first = store.append_player_statistics(
            "prediction-with-stats",
            pd.DataFrame([valid_statistic]),
        )
        second = store.append_player_statistics(
            "prediction-with-stats",
            pd.DataFrame([valid_statistic]),
        )

        assert first.statistics_inserted == 1
        assert second.statistics_inserted == 0
        with assert_raises(
            OperationsValidationError,
            match="predicción",
        ):
            store.append_player_statistics(
                "prediction-with-stats",
                pd.DataFrame([forbidden_statistic]),
            )
        statistic_id = store.connection.execute(
            "SELECT statistic_id FROM player_statistics"
        ).fetchone()[0]
        with assert_raises(sqlite3.IntegrityError, match="inmutables"):
            store.connection.execute(
                """
                DELETE FROM player_statistics WHERE statistic_id = ?
                """,
                (statistic_id,),
            )


def test_structural_failure_rolls_back_entire_run() -> None:
    """Un tipo inválido no deja cabecera ni filas parciales."""

    valid = prediction_row(source_match_id="te:valid")
    invalid = prediction_row(source_match_id="te:invalid")
    invalid["player_a_id"] = 1.5

    with temporary_store() as store:
        with assert_raises(OperationsValidationError):
            store.register_prediction_run(
                "rollback-run",
                pd.DataFrame([valid, invalid]),
            )
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM runs WHERE run_id = 'rollback-run'"
            ).fetchone()[0]
            == 0
        )
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0]
            == 0
        )


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Registra las funciones históricas como casos ``unittest``."""

    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(
                unittest.FunctionTestCase(
                    value,
                    description=getattr(value, "__doc__", None),
                )
            )
    return suite


if __name__ == "__main__":
    unittest.main()

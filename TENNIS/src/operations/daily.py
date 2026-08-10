"""Orquesta la predicción diaria, su registro y la conciliación operativa.

El módulo mantiene separadas tres responsabilidades: el pipeline causal de
predicción, la persistencia append-only en SQLite y una sola observación de
resultados de una jornada anterior pendiente. No limita cuántas veces puede
ejecutarse el lanzador en un día y nunca convierte la predicción en etiqueta o
estadística de entrenamiento.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
import pandas as pd

from ..config import (
    OPERATIONS_DATABASE_PATH,
    PREDICTIONS_PROCESSED_DIR,
    PROJECT_ROOT,
)
from ..daily_pipeline import (
    DailyPredictionRun,
    run_daily_prediction_pipeline,
)
from ..tennis_explorer import (
    TennisExplorerError,
    TennisExplorerResultSnapshot,
    refresh_daily_results,
)
from .store import OperationsStore
from .types import (
    ObservationReconciliation,
    PredictionRegistration,
    StatisticsRegistration,
)


DailyRunner = Callable[..., DailyPredictionRun]
ResultRefresher = Callable[..., TennisExplorerResultSnapshot]

_PLAYER_STATISTICS: Mapping[str, str] = {
    "elo_general": "historical_matches",
    "elo_surface": "historical_matches",
    "elo_general_matches": "historical_matches",
    "elo_surface_matches": "historical_matches",
    "recent_n_win_rate": "historical_matches",
    "recent_n_matches": "historical_matches",
    "recent_months_win_rate": "historical_matches",
    "recent_months_matches": "historical_matches",
    "rest_days": "historical_matches",
    "rank": "sackmann_rankings",
    "rank_points": "sackmann_rankings",
    "age": "sackmann_players",
}


@dataclass(frozen=True, slots=True)
class OperationalDailyRun:
    """Resume una ejecución diaria ya persistida y su conciliación opcional."""

    daily_run: DailyPredictionRun
    prediction_registration: PredictionRegistration
    statistics_registration: StatisticsRegistration
    result_date: date | None
    result_snapshot: TennisExplorerResultSnapshot | None
    observation_reconciliation: ObservationReconciliation | None
    reconciliation_warning: str | None

def _optional_text(value: object) -> str | None:
    """Normaliza un texto escalar y conserva los nulos de Pandas."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _utc_token(value: datetime) -> str:
    """Serializa un instante consciente como token UTC de microsegundos."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("El timestamp operativo debe incluir zona horaria.")
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _prediction_run_id(run: DailyPredictionRun) -> str:
    """Crea una identidad de ejecución reproducible y legible."""

    return (
        f"prediction:{run.match_date.isoformat()}:"
        f"{_utc_token(run.prediction_as_of_utc)}"
    )


def _observation_run_id(snapshot: TennisExplorerResultSnapshot) -> str:
    """Crea una identidad append-only para un snapshot de resultados."""

    return (
        f"observation:{snapshot.match_date.isoformat()}:"
        f"{_utc_token(snapshot.retrieved_at_utc)}:"
        f"{snapshot.snapshot_sha256[:12]}"
    )


def _project_relative(path: Path | None) -> str | None:
    """Convierte una ruta del proyecto en texto relativo auditable."""

    if path is None:
        return None
    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("El artefacto operativo debe vivir en TENNIS/.") from exc


def _prediction_metadata(
    run: DailyPredictionRun,
    *,
    retrained: bool,
) -> dict[str, object]:
    """Construye metadata pequeña sin duplicar el payload de predicciones."""

    return {
        "pipeline": "daily_prediction_v1",
        "csv_path": _project_relative(run.output_path),
        "predicted_matches": run.predicted_matches,
        "retrained_before_run": bool(retrained),
    }


def build_player_statistics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convierte features individuales prepartido a formato largo auditable.

    Solo se materializan jugadores mapeados con ``source_match_id`` estable.
    H2H y contexto de partido permanecen en el payload íntegro de la
    predicción; no se atribuyen artificialmente a un solo jugador. Los valores
    guardados son inputs as-of, nunca probabilidad, edge, ganador ni label.
    """

    columns = (
        "source_match_id",
        "player_slug",
        "player_id",
        "gender",
        "as_of_date",
        "statistic_name",
        "statistic_value",
        "source_kind",
        "source_snapshot_id",
    )
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in predictions.to_dict(orient="records"):
        source_match_id = _optional_text(row.get("source_match_id"))
        gender = _optional_text(row.get("gender"))
        as_of_date = row.get("prediction_date")
        if (
            source_match_id is None
            or gender not in {"M", "F"}
            or _optional_text(row.get("mapping_status")) != "mapped"
        ):
            continue
        for side in ("a", "b"):
            player_slug = _optional_text(row.get(f"player_{side}_slug"))
            if player_slug is None:
                continue
            for statistic_name, source_kind in _PLAYER_STATISTICS.items():
                column = f"{statistic_name}_{side}"
                if column not in predictions.columns:
                    continue
                rows.append(
                    {
                        "source_match_id": source_match_id,
                        "player_slug": player_slug,
                        "player_id": row.get(f"player_{side}_id"),
                        "gender": gender,
                        "as_of_date": as_of_date,
                        "statistic_name": statistic_name,
                        "statistic_value": row.get(column),
                        "source_kind": source_kind,
                        "source_snapshot_id": None,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _filter_registered_statistics(
    store: OperationsStore,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Retira stats de identidades que el almacén envió a revisión."""

    if statistics.empty:
        return statistics
    matches = pd.read_sql_query(
        """
        SELECT source_match_id, player_1_slug, player_2_slug
        FROM matches
        """,
        store.connection,
    )
    allowed = {
        (str(row.source_match_id), str(slug))
        for row in matches.itertuples(index=False)
        for slug in (row.player_1_slug, row.player_2_slug)
        if slug is not None
    }
    keep = [
        (str(row.source_match_id), str(row.player_slug)) in allowed
        for row in statistics.itertuples(index=False)
    ]
    return statistics.loc[keep].reset_index(drop=True)


def _result_metadata(
    snapshot: TennisExplorerResultSnapshot,
) -> dict[str, object]:
    """Describe un snapshot de resultado sin rutas absolutas de máquina."""

    return {
        "pipeline": "daily_result_reconciliation_v1",
        "html_path": _project_relative(snapshot.html_path),
        "metadata_path": _project_relative(snapshot.metadata_path),
        "snapshot_sha256": snapshot.snapshot_sha256,
        "source_url": snapshot.source_url,
    }


def run_operational_daily_pipeline(
    match_date: date | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
    output_dir: Path = PREDICTIONS_PROCESSED_DIR,
    database_path: Path = OPERATIONS_DATABASE_PATH,
    publish: bool = True,
    retrained: bool = False,
    daily_runner: DailyRunner = run_daily_prediction_pipeline,
    result_refresher: ResultRefresher = refresh_daily_results,
) -> OperationalDailyRun:
    """Predice, persiste y observa una única fecha anterior pendiente.

    Un fallo HTTP, WAF, caché o HTML durante la conciliación se devuelve como
    advertencia explícita después de haber conservado la predicción del día.
    Los fallos de integridad de SQLite sí se propagan: no deben maquillarse.
    Cada invocación puede refrescar una fecha, pero no existe contador ni tope
    diario en código.
    """

    daily = daily_runner(
        match_date,
        clock=clock,
        output_dir=output_dir,
        publish=publish,
    )
    prediction_run_id = _prediction_run_id(daily)
    with OperationsStore(database_path, clock=clock) as store:
        prediction_registration = store.register_prediction_run(
            prediction_run_id,
            daily.predictions,
            metadata=_prediction_metadata(daily, retrained=retrained),
            target_date=daily.match_date,
        )
        statistics = build_player_statistics_frame(daily.predictions)
        statistics = _filter_registered_statistics(store, statistics)
        statistics_registration = store.append_player_statistics(
            prediction_run_id,
            statistics,
        )
        pending_date = store.select_pending_result_date(
            before_date=daily.match_date
        )

    if pending_date is None:
        return OperationalDailyRun(
            daily_run=daily,
            prediction_registration=prediction_registration,
            statistics_registration=statistics_registration,
            result_date=None,
            result_snapshot=None,
            observation_reconciliation=None,
            reconciliation_warning=None,
        )

    try:
        result_snapshot = result_refresher(pending_date, clock=clock)
    except TennisExplorerError as exc:
        return OperationalDailyRun(
            daily_run=daily,
            prediction_registration=prediction_registration,
            statistics_registration=statistics_registration,
            result_date=pending_date,
            result_snapshot=None,
            observation_reconciliation=None,
            reconciliation_warning=f"{type(exc).__name__}: {exc}",
        )

    with OperationsStore(database_path, clock=clock) as store:
        observation_reconciliation = store.reconcile_observations(
            _observation_run_id(result_snapshot),
            result_snapshot.matches,
            metadata=_result_metadata(result_snapshot),
            observed_at_utc=result_snapshot.retrieved_at_utc,
            target_date=pending_date,
        )
    return OperationalDailyRun(
        daily_run=daily,
        prediction_registration=prediction_registration,
        statistics_registration=statistics_registration,
        result_date=pending_date,
        result_snapshot=result_snapshot,
        observation_reconciliation=observation_reconciliation,
        reconciliation_warning=None,
    )

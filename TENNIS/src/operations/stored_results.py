"""Reprocesa evidencia Tennis Explorer ya guardada mediante la puerta operativa.

El mÃ³dulo solo lee observaciones append-only anteriores y crea nuevas
observaciones canÃ³nicas/settlements a travÃ©s de :class:`OperationsStore`. No
edita ni elimina predicciones, observaciones o settlements existentes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path

import pandas as pd

from ..config import (
    OPERATIONS_DATABASE_PATH,
    PLAYER_MAPPING_DATABASE_PATH,
    PROJECT_ROOT,
)
from ..tennis_explorer import TennisExplorerResultSnapshot
from .result_sources import (
    TennisExplorerMappedResultSnapshot,
    build_tennis_explorer_mapped_result_snapshot,
)
from .store import OperationsStore
from .types import ObservationReconciliation, OperationsSchemaError


@dataclass(frozen=True, slots=True)
class StoredResultReconciliation:
    """Resume una conciliaciÃ³n derivada de evidencia cruda ya persistida."""

    result_date: date
    raw_run_id: str
    snapshot: TennisExplorerMappedResultSnapshot
    reconciliation: ObservationReconciliation


def _aware_utc(value: object, field: str) -> datetime:
    """Convierte un timestamp persistido y exige zona horaria."""

    try:
        parsed = pd.Timestamp(value).to_pydatetime()
    except (TypeError, ValueError) as exc:
        raise OperationsSchemaError(f"{field} no contiene un timestamp vÃ¡lido.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationsSchemaError(f"{field} debe incluir zona horaria.")
    return parsed.astimezone(UTC)


def _project_path(value: object, field: str) -> Path:
    """Resuelve una ruta de metadata sin permitir escapar de TENNIS/."""

    if not isinstance(value, str) or not value.strip():
        raise OperationsSchemaError(f"{field} falta en la metadata del run crudo.")
    path = (PROJECT_ROOT / value).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise OperationsSchemaError(f"{field} escapa de TENNIS/.") from exc
    return path


def _pending_dates(store: OperationsStore, before_date: date) -> tuple[date, ...]:
    """Lista fechas pendientes con alguna predicciÃ³n oficial no liquidada."""

    rows = store.connection.execute(
        """
        SELECT DISTINCT m.match_date
        FROM matches AS m
        JOIN official_predictions AS op
          ON op.source_match_id = m.source_match_id
        LEFT JOIN settlements AS s
          ON s.source_match_id = m.source_match_id
        WHERE m.match_date < ?
          AND s.source_match_id IS NULL
        ORDER BY m.match_date
        """,
        (before_date.isoformat(),),
    ).fetchall()
    dates: list[date] = []
    for row in rows:
        try:
            dates.append(date.fromisoformat(str(row["match_date"])))
        except ValueError as exc:
            raise OperationsSchemaError("matches contiene un match_date no canÃ³nico.") from exc
    return tuple(dates)


def _load_latest_raw_snapshot(
    store: OperationsStore,
    result_date: date,
) -> tuple[str, TennisExplorerResultSnapshot] | None:
    """Reconstruye el Ãºltimo snapshot Explorer v1 de una fecha pendiente."""

    runs = store.connection.execute(
        """
        SELECT run_id, metadata_json
        FROM runs
        WHERE run_type = 'observation'
          AND source_system = 'tennis_explorer'
          AND target_date = ?
          AND status = 'complete'
        ORDER BY started_at_utc DESC, run_id DESC
        """,
        (result_date.isoformat(),),
    ).fetchall()
    selected_run_id: str | None = None
    selected_metadata: dict[str, object] | None = None
    for run in runs:
        try:
            metadata = json.loads(str(run["metadata_json"]))
        except json.JSONDecodeError as exc:
            raise OperationsSchemaError("Un run contiene metadata_json invÃ¡lido.") from exc
        if not isinstance(metadata, dict):
            raise OperationsSchemaError("metadata_json debe ser un objeto JSON.")
        if metadata.get("pipeline") == "daily_result_reconciliation_v1":
            selected_run_id = str(run["run_id"])
            selected_metadata = metadata
            break
    if selected_run_id is None or selected_metadata is None:
        return None

    rows = store.connection.execute(
        """
        SELECT o.payload_json, o.observed_at_utc
        FROM observations AS o
        JOIN run_observations AS ro
          ON ro.observation_id = o.observation_id
        WHERE ro.run_id = ?
        ORDER BY o.observation_id
        """,
        (selected_run_id,),
    ).fetchall()
    if not rows:
        return None
    payloads: list[dict[str, object]] = []
    observed_values: list[datetime] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as exc:
            raise OperationsSchemaError(
                "Una observaciÃ³n cruda contiene payload_json invÃ¡lido."
            ) from exc
        if not isinstance(payload, dict):
            raise OperationsSchemaError("payload_json debe ser un objeto JSON.")
        payloads.append(payload)
        observed_values.append(_aware_utc(row["observed_at_utc"], "observed_at_utc"))

    raw_sha256 = selected_metadata.get("snapshot_sha256")
    source_url = selected_metadata.get("source_url")
    if not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
        raise OperationsSchemaError("El run crudo carece de snapshot_sha256 vÃ¡lido.")
    if not isinstance(source_url, str) or not source_url:
        raise OperationsSchemaError("El run crudo carece de source_url vÃ¡lida.")
    snapshot = TennisExplorerResultSnapshot(
        match_date=result_date,
        retrieved_at_utc=max(observed_values),
        source_url=source_url,
        snapshot_sha256=raw_sha256,
        matches=pd.DataFrame(payloads),
        html_path=_project_path(selected_metadata.get("html_path"), "html_path"),
        metadata_path=_project_path(
            selected_metadata.get("metadata_path"),
            "metadata_path",
        ),
    )
    return selected_run_id, snapshot


def _mapped_metadata(
    raw_run_id: str,
    snapshot: TennisExplorerMappedResultSnapshot,
) -> dict[str, object]:
    """Construye trazabilidad compacta para el run de remapeo."""

    return {
        "pipeline": "stored_result_cross_source_reconciliation_v1",
        "raw_run_id": raw_run_id,
        "raw_snapshot_sha256": snapshot.raw_snapshot_sha256,
        "mapped_snapshot_sha256": snapshot.snapshot_sha256,
        "identity_join": "exact_id_or_date_gender_sackmann_identity",
        "official_pending_count": snapshot.official_pending_count,
        "terminal_result_count": snapshot.terminal_result_count,
        "exact_id_count": snapshot.exact_id_count,
        "identity_mapped_count": snapshot.identity_mapped_count,
        "paired_inferred_count": snapshot.paired_inferred_count,
        "unmatched_official_count": snapshot.unmatched_official_count,
        "unmapped_player_rows": snapshot.unmapped_player_rows,
        "ambiguous_pair_rows": snapshot.ambiguous_pair_rows,
    }


def reconcile_stored_tennis_explorer_results(
    before_date: date,
    *,
    database_path: Path = OPERATIONS_DATABASE_PATH,
    mapping_database_path: Path = PLAYER_MAPPING_DATABASE_PATH,
    clock: Callable[[], datetime] | None = None,
) -> tuple[StoredResultReconciliation, ...]:
    """Liquida fechas pendientes usando snapshots Explorer ya almacenados.

    El corte ``before_date`` es exclusivo. Solo se reproducen runs crudos v1 y
    cada traducciÃ³n genera un run determinista distinto. Repetir la funciÃ³n es
    idempotente; un mapping ampliado puede aÃ±adir nuevas observaciones sin
    alterar las anteriores.
    """

    if not isinstance(before_date, date) or isinstance(before_date, datetime):
        raise TypeError("before_date debe ser datetime.date estricto.")
    reports: list[StoredResultReconciliation] = []
    with OperationsStore(database_path, clock=clock) as store:
        for result_date in _pending_dates(store, before_date):
            loaded = _load_latest_raw_snapshot(store, result_date)
            if loaded is None:
                continue
            raw_run_id, raw_snapshot = loaded
            mapped_snapshot = build_tennis_explorer_mapped_result_snapshot(
                store,
                raw_snapshot,
                mapping_database_path=mapping_database_path,
            )
            run_id = (
                f"observation-remap:{result_date.isoformat()}:"
                f"{mapped_snapshot.snapshot_sha256[:12]}"
            )
            reconciliation = store.reconcile_observations(
                run_id,
                mapped_snapshot.matches,
                source_system="tennis_explorer",
                metadata=_mapped_metadata(raw_run_id, mapped_snapshot),
                observed_at_utc=mapped_snapshot.retrieved_at_utc,
                target_date=result_date,
            )
            reports.append(
                StoredResultReconciliation(
                    result_date=result_date,
                    raw_run_id=raw_run_id,
                    snapshot=mapped_snapshot,
                    reconciliation=reconciliation,
                )
            )
    return tuple(reports)

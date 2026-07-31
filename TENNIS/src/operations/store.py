"""Almacén SQLite transaccional para predicciones y resultados operativos.

La API acepta ``DataFrame`` de las fases anteriores, conserva cada payload
como evidencia inmutable y selecciona como predicción oficial la primera fila
válida registrada para un ``source_match_id``. Los resultados se liquidan
exclusivamente comparando ``winner_slug`` con los slugs A/B guardados en esa
predicción oficial; ni el nombre visible ni la posición de la fuente deciden
el label observado.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from ..config import OPERATIONS_DATABASE_PATH, PROJECT_ROOT

from .identifiers import (
    canonical_date,
    canonical_json,
    canonical_slug,
    canonical_utc_datetime,
    derive_source_match_id,
    optional_scalar,
    optional_text,
    payload_sha256,
    required_text,
)
from .schema import SCHEMA_SQL, SCHEMA_VERSION
from .types import (
    ObservationReconciliation,
    OperationsConflictError,
    OperationsSchemaError,
    OperationsValidationError,
    PredictionRegistration,
    StatisticsRegistration,
)
from .validation import (
    OBSERVATION_REQUIRED_COLUMNS,
    PREDICTION_REQUIRED_COLUMNS,
    STATISTICS_REQUIRED_COLUMNS,
    optional_float,
    optional_integer,
    require_frame_columns,
    validate_observation_row,
    validate_prediction_row,
    validate_source_kind,
    validate_statistic_name,
)


Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    """Devuelve el instante UTC actual para sellos operativos."""

    return datetime.now(UTC)


def _row_dicts(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Materializa filas Pandas como mappings ordinarios."""

    return [
        {str(column): value for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _canonical_frame_payload(
    rows: list[dict[str, object]],
    *,
    run_type: str,
    source_system: str,
    target_date: str | None,
    metadata: Mapping[str, object],
    observed_at_utc: object | None = None,
) -> str:
    """Hashea una ejecución sin hacer depender la identidad del orden de filas."""

    canonical_rows = sorted(canonical_json(row) for row in rows)
    return payload_sha256(
        {
            "run_type": run_type,
            "source_system": source_system,
            "target_date": target_date,
            "metadata": dict(metadata),
            "observed_at_utc_argument": observed_at_utc,
            "rows": canonical_rows,
        }
    )


def _identifier(prefix: str, payload: Mapping[str, object]) -> str:
    """Crea una clave compacta y reproducible con prefijo legible."""

    return f"{prefix}:{payload_sha256(payload)}"


class OperationsStore(AbstractContextManager["OperationsStore"]):
    """Gestiona SQLite con migración, transacciones e invariantes operativos.

    Parameters
    ----------
    database_path:
        Ruta SQLite situada obligatoriamente dentro de ``TENNIS/``.
    clock:
        Reloj inyectable para tests reproducibles. Debe devolver un
        ``datetime`` consciente de zona horaria.
    """

    def __init__(
        self,
        database_path: str | Path = OPERATIONS_DATABASE_PATH,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Abre la base y aplica la versión de esquema compatible."""

        self.database_path = self._validate_database_path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or _default_clock
        self.connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            timeout=30.0,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._initialize_schema()

    @staticmethod
    def _validate_database_path(database_path: str | Path) -> Path:
        """Impide que una configuración accidental escriba fuera de TENNIS."""

        path = Path(database_path).expanduser().resolve(strict=False)
        project_root = PROJECT_ROOT.resolve(strict=False)
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise OperationsValidationError(
                "La base operativa debe vivir dentro de TENNIS/."
            ) from exc
        if path.exists() and path.is_dir():
            raise OperationsValidationError(
                "database_path debe apuntar a un archivo SQLite."
            )
        return path

    def __enter__(self) -> "OperationsStore":
        """Devuelve el almacén abierto."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Cierra la conexión y revierte cualquier transacción incompleta."""

        if self.connection.in_transaction:
            self.connection.rollback()
        self.connection.close()

    def close(self) -> None:
        """Cierra explícitamente la conexión SQLite."""

        if self.connection.in_transaction:
            self.connection.rollback()
        self.connection.close()

    def _now(self) -> str:
        """Normaliza el reloj inyectado a ISO UTC."""

        return required_text(
            canonical_utc_datetime(self._clock(), "clock"),
            "clock",
        )

    def _initialize_schema(self) -> None:
        """Aplica de forma atómica el esquema v1 o rechaza una versión futura."""

        current_version = int(
            self.connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if current_version > SCHEMA_VERSION:
            self.connection.close()
            raise OperationsSchemaError(
                "La base usa una versión más nueva que este código: "
                f"{current_version} > {SCHEMA_VERSION}."
            )
        escaped_now = self._now().replace("'", "''")
        migration = (
            "BEGIN IMMEDIATE;\n"
            f"{SCHEMA_SQL}\n"
            "INSERT OR IGNORE INTO schema_versions("
            "version, applied_at_utc, description"
            f") VALUES ({SCHEMA_VERSION}, '{escaped_now}', "
            "'Esquema operativo inicial');\n"
            f"PRAGMA user_version = {SCHEMA_VERSION};\n"
            "COMMIT;"
        )
        try:
            self.connection.executescript(migration)
        except sqlite3.DatabaseError as exc:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise OperationsSchemaError(
                "No se pudo inicializar el esquema operativo."
            ) from exc

    def _begin(self) -> None:
        """Inicia una transacción de escritura que serializa la selección oficial."""

        self.connection.execute("BEGIN IMMEDIATE")

    def _existing_run(
        self,
        run_id: str,
        *,
        run_type: str,
        payload_hash: str,
    ) -> sqlite3.Row | None:
        """Comprueba idempotencia y registra un conflicto de run divergente."""

        existing = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is None:
            return None
        if (
            existing["run_type"] == run_type
            and existing["payload_sha256"] == payload_hash
            and existing["status"] == "complete"
        ):
            return existing
        self._record_divergent_run_conflict(
            run_id=run_id,
            existing_payload=str(existing["payload_sha256"]),
            incoming_payload=payload_hash,
            existing_run_type=str(existing["run_type"]),
            incoming_run_type=run_type,
        )
        raise OperationsConflictError(
            f"run_id '{run_id}' ya existe con un payload distinto."
        )

    def _record_divergent_run_conflict(
        self,
        *,
        run_id: str,
        existing_payload: str,
        incoming_payload: str,
        existing_run_type: str,
        incoming_run_type: str,
    ) -> None:
        """Conserva de forma append-only un intento de reutilización divergente."""

        now = self._now()
        details = {
            "existing_run_type": existing_run_type,
            "incoming_run_type": incoming_run_type,
        }
        conflict_id = _identifier(
            "conflict",
            {
                "type": "divergent_run_payload",
                "run_id": run_id,
                "incoming_payload": incoming_payload,
            },
        )
        queue_id = _identifier(
            "queue",
            {
                "issue_type": "divergent_run_payload",
                "run_id": run_id,
                "incoming_payload": incoming_payload,
            },
        )
        self._begin()
        try:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO conflicts(
                    conflict_id, conflict_type, run_id,
                    existing_reference, incoming_reference,
                    details_json, created_at_utc
                ) VALUES (?, 'divergent_run_payload', ?, ?, ?, ?, ?)
                """,
                (
                    conflict_id,
                    run_id,
                    existing_payload,
                    incoming_payload,
                    canonical_json(details),
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO review_queue(
                    queue_id, issue_type, run_id, conflict_id,
                    payload_json, created_at_utc
                ) VALUES (?, 'divergent_run_payload', ?, ?, ?, ?)
                """,
                (
                    queue_id,
                    run_id,
                    conflict_id,
                    canonical_json(details),
                    now,
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _insert_run(
        self,
        *,
        run_id: str,
        run_type: str,
        source_system: str,
        target_date: str | None,
        input_rows: int,
        payload_hash: str,
        metadata: Mapping[str, object],
        now: str,
    ) -> None:
        """Inserta la cabecera provisional de una ejecución."""

        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, run_type, source_system, target_date, status,
                input_rows, payload_sha256, metadata_json,
                started_at_utc, created_at_utc
            ) VALUES (?, ?, ?, ?, 'building', ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_type,
                source_system,
                target_date,
                input_rows,
                payload_hash,
                canonical_json(dict(metadata)),
                now,
                now,
            ),
        )

    def _complete_run(
        self,
        run_id: str,
        *,
        accepted_rows: int,
        queued_rows: int,
        now: str,
    ) -> None:
        """Marca completa una cabecera dentro de su misma transacción."""

        self.connection.execute(
            """
            UPDATE runs
            SET status = 'complete',
                accepted_rows = ?,
                queued_rows = ?,
                completed_at_utc = ?
            WHERE run_id = ? AND status = 'building'
            """,
            (accepted_rows, queued_rows, now, run_id),
        )

    def _register_snapshot(
        self,
        row: Mapping[str, object],
        *,
        run_id: str,
        source_system: str,
        retrieved_at_utc: str | None,
        now: str,
    ) -> str | None:
        """Registra y vincula un snapshot cuando existe hash de contenido."""

        snapshot_sha = optional_text(
            row.get("source_snapshot_sha256"),
            "source_snapshot_sha256",
        )
        if snapshot_sha is None or retrieved_at_utc is None:
            return None
        snapshot_id = _identifier(
            "snapshot",
            {
                "source_system": source_system,
                "snapshot_sha256": snapshot_sha,
                "retrieved_at_utc": retrieved_at_utc,
            },
        )
        snapshot_date_value = optional_scalar(
            row.get("prediction_date", row.get("match_date"))
        )
        snapshot_date = (
            canonical_date(snapshot_date_value, "snapshot_date")
            if snapshot_date_value is not None
            else None
        )
        payload = {
            "source_snapshot_sha256": snapshot_sha,
            "source_retrieved_at_utc": retrieved_at_utc,
            "snapshot_date": snapshot_date,
        }
        self.connection.execute(
            """
            INSERT OR IGNORE INTO snapshots(
                snapshot_id, source_system, snapshot_sha256,
                retrieved_at_utc, snapshot_date, payload_json,
                created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                source_system,
                snapshot_sha,
                retrieved_at_utc,
                snapshot_date,
                canonical_json(payload),
                now,
            ),
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO run_snapshots(run_id, snapshot_id)
            VALUES (?, ?)
            """,
            (run_id, snapshot_id),
        )
        return snapshot_id

    @staticmethod
    def _prediction_match_identity(
        row: Mapping[str, object],
        *,
        source_match_id: str,
        source_system: str,
        match_date: str,
        player_a_slug: str | None,
        player_b_slug: str | None,
    ) -> dict[str, object]:
        """Construye los campos comparables de la identidad de un partido."""

        tournament_key = optional_text(
            row.get("tournament_href"),
            "tournament_href",
        )
        gender = required_text(row.get("gender"), "gender")
        slug_pair = (
            sorted((player_a_slug, player_b_slug))
            if player_a_slug is not None and player_b_slug is not None
            else None
        )
        return {
            "source_match_id": source_match_id,
            "source_system": source_system,
            "match_date": match_date,
            "gender": gender,
            "tournament_key": tournament_key,
            "player_slugs": slug_pair,
        }

    def _match_conflict_details(
        self,
        existing: sqlite3.Row,
        incoming: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Devuelve diferencias inequívocas sin exigir completar campos nulos."""

        differences: dict[str, object] = {}
        for field in ("source_system", "match_date", "gender"):
            old = existing[field]
            new = incoming[field]
            if old is not None and new is not None and old != new:
                differences[field] = {"existing": old, "incoming": new}
        old_tournament = existing["tournament_key"]
        new_tournament = incoming["tournament_key"]
        if (
            old_tournament is not None
            and new_tournament is not None
            and old_tournament != new_tournament
        ):
            differences["tournament_key"] = {
                "existing": old_tournament,
                "incoming": new_tournament,
            }
        old_first = existing["player_1_slug"]
        old_second = existing["player_2_slug"]
        new_pair = incoming["player_slugs"]
        if (
            old_first is not None
            and old_second is not None
            and new_pair is not None
            and sorted((old_first, old_second)) != list(new_pair)
        ):
            differences["player_slugs"] = {
                "existing": sorted((old_first, old_second)),
                "incoming": list(new_pair),
            }
        return differences or None

    def _record_conflict(
        self,
        *,
        conflict_type: str,
        run_id: str,
        source_match_id: str | None,
        existing_reference: str | None,
        incoming_reference: str | None,
        details: Mapping[str, object],
        observation_id: str | None = None,
        now: str,
    ) -> tuple[str, bool]:
        """Añade conflicto y cola deterministas sin duplicar el mismo hallazgo."""

        conflict_id = _identifier(
            "conflict",
            {
                "conflict_type": conflict_type,
                "run_id": run_id,
                "source_match_id": source_match_id,
                "existing_reference": existing_reference,
                "incoming_reference": incoming_reference,
                "details": dict(details),
            },
        )
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO conflicts(
                conflict_id, conflict_type, source_match_id, run_id,
                existing_reference, incoming_reference,
                details_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict_id,
                conflict_type,
                source_match_id,
                run_id,
                existing_reference,
                incoming_reference,
                canonical_json(dict(details)),
                now,
            ),
        )
        queue_id = _identifier(
            "queue",
            {
                "issue_type": conflict_type,
                "run_id": run_id,
                "source_match_id": source_match_id,
                "observation_id": observation_id,
                "conflict_id": conflict_id,
            },
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO review_queue(
                queue_id, issue_type, source_match_id, run_id,
                observation_id, conflict_id, payload_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                conflict_type,
                source_match_id,
                run_id,
                observation_id,
                conflict_id,
                canonical_json(dict(details)),
                now,
            ),
        )
        return conflict_id, cursor.rowcount == 1

    def _queue_issue(
        self,
        *,
        issue_type: str,
        run_id: str,
        source_match_id: str | None,
        observation_id: str | None,
        payload: Mapping[str, object],
        now: str,
    ) -> bool:
        """Añade una incidencia no conflictiva a revisión humana."""

        queue_id = _identifier(
            "queue",
            {
                "issue_type": issue_type,
                "run_id": run_id,
                "source_match_id": source_match_id,
                "observation_id": observation_id,
                "payload": dict(payload),
            },
        )
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO review_queue(
                queue_id, issue_type, source_match_id, run_id,
                observation_id, payload_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                issue_type,
                source_match_id,
                run_id,
                observation_id,
                canonical_json(dict(payload)),
                now,
            ),
        )
        return cursor.rowcount == 1

    def _upsert_match(
        self,
        row: Mapping[str, object],
        *,
        run_id: str,
        source_match_id: str,
        source_system: str,
        match_date: str,
        player_a_slug: str | None,
        player_b_slug: str | None,
        snapshot_id: str | None,
        now: str,
    ) -> tuple[bool, bool]:
        """Inserta un partido o encola una contradicción de identidad."""

        identity = self._prediction_match_identity(
            row,
            source_match_id=source_match_id,
            source_system=source_system,
            match_date=match_date,
            player_a_slug=player_a_slug,
            player_b_slug=player_b_slug,
        )
        existing = self.connection.execute(
            "SELECT * FROM matches WHERE source_match_id = ?",
            (source_match_id,),
        ).fetchone()
        if existing is not None:
            differences = self._match_conflict_details(existing, identity)
            if differences is None:
                return False, False
            self._record_conflict(
                conflict_type="match_identity_conflict",
                run_id=run_id,
                source_match_id=source_match_id,
                existing_reference=str(existing["identity_payload_sha256"]),
                incoming_reference=payload_sha256(identity),
                details=differences,
                now=now,
            )
            return False, True
        self.connection.execute(
            """
            INSERT INTO matches(
                source_match_id, source_system, match_date, tournament,
                tournament_key, tour_level, gender, player_1_slug,
                player_2_slug, identity_payload_sha256,
                first_snapshot_id, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_match_id,
                source_system,
                match_date,
                optional_text(row.get("tournament"), "tournament"),
                identity["tournament_key"],
                optional_text(row.get("tour_level"), "tour_level"),
                identity["gender"],
                player_a_slug,
                player_b_slug,
                payload_sha256(identity),
                snapshot_id,
                now,
            ),
        )
        return True, False

    def register_prediction_run(
        self,
        run_id: str,
        predictions: pd.DataFrame,
        *,
        source_system: str = "tennis_explorer",
        metadata: Mapping[str, object] | None = None,
        target_date: object | None = None,
    ) -> PredictionRegistration:
        """Registra una ejecución y congela la primera predicción válida.

        Las filas sin identidad estable o con conflicto de identidad se
        conservan en ``review_queue`` y no rompen las demás filas. Una
        infracción estructural o de tipos revierte la ejecución completa.
        """

        run_key = required_text(run_id, "run_id")
        source = required_text(source_system, "source_system")
        require_frame_columns(
            predictions,
            PREDICTION_REQUIRED_COLUMNS,
            frame_name="predictions",
        )
        rows = _row_dicts(predictions)
        metadata_payload = dict(metadata or {})
        target = (
            canonical_date(target_date, "target_date")
            if optional_scalar(target_date) is not None
            else None
        )
        payload_hash = _canonical_frame_payload(
            rows,
            run_type="prediction",
            source_system=source,
            target_date=target,
            metadata=metadata_payload,
        )
        existing = self._existing_run(
            run_key,
            run_type="prediction",
            payload_hash=payload_hash,
        )
        if existing is not None:
            return self._prediction_reuse_summary(run_key, len(rows))

        now = self._now()
        matches_inserted = 0
        predictions_inserted = 0
        valid_predictions = 0
        official_selected = 0
        settlements_inserted = 0
        queued_rows = 0
        self._begin()
        try:
            self._insert_run(
                run_id=run_key,
                run_type="prediction",
                source_system=source,
                target_date=target,
                input_rows=len(rows),
                payload_hash=payload_hash,
                metadata=metadata_payload,
                now=now,
            )
            seen_match_ids: dict[str, str] = {}
            for row in rows:
                row_payload_json = canonical_json(row)
                row_payload_hash = payload_sha256(row)
                try:
                    source_match_id = derive_source_match_id(
                        row,
                        source_system=source,
                    )
                except OperationsValidationError as exc:
                    queued_rows += int(
                        self._queue_issue(
                            issue_type="missing_stable_match_identity",
                            run_id=run_key,
                            source_match_id=None,
                            observation_id=None,
                            payload={
                                "error": str(exc),
                                "row_payload_sha256": row_payload_hash,
                            },
                            now=now,
                        )
                    )
                    continue
                prior_hash = seen_match_ids.get(source_match_id)
                if prior_hash is not None:
                    if prior_hash != row_payload_hash:
                        self._record_conflict(
                            conflict_type="duplicate_match_in_run",
                            run_id=run_key,
                            source_match_id=source_match_id,
                            existing_reference=prior_hash,
                            incoming_reference=row_payload_hash,
                            details={
                                "message": (
                                    "Dos filas distintas comparten "
                                    "source_match_id en la ejecución."
                                )
                            },
                            now=now,
                        )
                    queued_rows += 1
                    continue
                seen_match_ids[source_match_id] = row_payload_hash
                validity = validate_prediction_row(row)
                snapshot_id = self._register_snapshot(
                    row,
                    run_id=run_key,
                    source_system=source,
                    retrieved_at_utc=validity.source_retrieved_at_utc,
                    now=now,
                )
                match_inserted, match_conflict = self._upsert_match(
                    row,
                    run_id=run_key,
                    source_match_id=source_match_id,
                    source_system=source,
                    match_date=validity.match_date,
                    player_a_slug=validity.player_a_slug,
                    player_b_slug=validity.player_b_slug,
                    snapshot_id=snapshot_id,
                    now=now,
                )
                matches_inserted += int(match_inserted)
                if match_conflict:
                    queued_rows += 1
                    continue
                prediction_id = _identifier(
                    "prediction",
                    {
                        "run_id": run_key,
                        "source_match_id": source_match_id,
                    },
                )
                self.connection.execute(
                    """
                    INSERT INTO predictions(
                        prediction_id, run_id, source_match_id, snapshot_id,
                        prediction_as_of_utc, source_retrieved_at_utc,
                        source_status, player_a_name, player_b_name,
                        player_a_slug, player_b_slug, player_a_id, player_b_id,
                        mapping_status, model_probability_raw_a,
                        model_probability_a, model_probability_b,
                        market_probability_a, market_probability_b,
                        edge_a, edge_b, confidence, confidence_flags,
                        prediction_status, model_profile, model_fingerprint,
                        model_training_max_date, feature_fingerprint,
                        is_valid, invalid_reason, payload_sha256,
                        payload_json, created_at_utc
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        prediction_id,
                        run_key,
                        source_match_id,
                        snapshot_id,
                        validity.prediction_as_of_utc,
                        validity.source_retrieved_at_utc,
                        optional_text(row.get("status"), "status"),
                        optional_text(row.get("player_a_name"), "player_a_name"),
                        optional_text(row.get("player_b_name"), "player_b_name"),
                        validity.player_a_slug,
                        validity.player_b_slug,
                        optional_integer(row.get("player_a_id"), "player_a_id"),
                        optional_integer(row.get("player_b_id"), "player_b_id"),
                        optional_text(
                            row.get("mapping_status"),
                            "mapping_status",
                        ),
                        validity.model_probability_raw_a,
                        validity.model_probability_a,
                        validity.model_probability_b,
                        optional_float(
                            row.get("market_probability_a"),
                            "market_probability_a",
                            probability=True,
                        ),
                        optional_float(
                            row.get("market_probability_b"),
                            "market_probability_b",
                            probability=True,
                        ),
                        optional_float(row.get("edge_a"), "edge_a"),
                        optional_float(row.get("edge_b"), "edge_b"),
                        optional_text(row.get("confidence"), "confidence"),
                        optional_text(
                            row.get("confidence_flags"),
                            "confidence_flags",
                        ),
                        optional_text(
                            row.get("prediction_status"),
                            "prediction_status",
                        ),
                        optional_text(
                            row.get("model_profile"),
                            "model_profile",
                        ),
                        optional_text(
                            row.get("model_fingerprint"),
                            "model_fingerprint",
                        ),
                        (
                            canonical_date(
                                row.get("model_training_max_date"),
                                "model_training_max_date",
                            )
                            if optional_scalar(
                                row.get("model_training_max_date")
                            )
                            is not None
                            else None
                        ),
                        optional_text(
                            row.get("feature_fingerprint"),
                            "feature_fingerprint",
                        ),
                        int(validity.is_valid),
                        validity.invalid_reason,
                        row_payload_hash,
                        row_payload_json,
                        now,
                    ),
                )
                predictions_inserted += 1
                valid_predictions += int(validity.is_valid)
                if validity.is_valid:
                    official_cursor = self.connection.execute(
                        """
                        INSERT OR IGNORE INTO official_predictions(
                            source_match_id, prediction_id, selection_rule,
                            selected_at_utc
                        ) VALUES (?, ?, 'first_registered_valid', ?)
                        """,
                        (source_match_id, prediction_id, now),
                    )
                    if official_cursor.rowcount == 1:
                        official_selected += 1
                        settlements_inserted += (
                            self._settle_pending_observations(
                                source_match_id,
                                fallback_run_id=run_key,
                                now=now,
                            )
                        )
            self._complete_run(
                run_key,
                accepted_rows=predictions_inserted,
                queued_rows=queued_rows,
                now=now,
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return PredictionRegistration(
            run_id=run_key,
            input_rows=len(rows),
            matches_inserted=matches_inserted,
            predictions_inserted=predictions_inserted,
            valid_predictions=valid_predictions,
            official_predictions_selected=official_selected,
            settlements_inserted=settlements_inserted,
            queued_rows=queued_rows,
            reused=False,
        )

    def _prediction_reuse_summary(
        self,
        run_id: str,
        input_rows: int,
    ) -> PredictionRegistration:
        """Reconstruye contadores de una ejecución idempotente ya completada."""

        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS predictions_inserted,
                COALESCE(SUM(p.is_valid), 0) AS valid_predictions,
                COALESCE(SUM(
                    CASE WHEN op.prediction_id IS NOT NULL THEN 1 ELSE 0 END
                ), 0) AS official_predictions_selected,
                COALESCE(SUM(
                    CASE WHEN s.settlement_id IS NOT NULL THEN 1 ELSE 0 END
                ), 0) AS settlements_inserted
            FROM predictions AS p
            LEFT JOIN official_predictions AS op
                ON op.prediction_id = p.prediction_id
            LEFT JOIN settlements AS s
                ON s.official_prediction_id = p.prediction_id
            WHERE p.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        run = self.connection.execute(
            "SELECT queued_rows FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        matches = self.connection.execute(
            """
            SELECT COUNT(DISTINCT m.source_match_id)
            FROM matches AS m
            JOIN predictions AS p
                ON p.source_match_id = m.source_match_id
            WHERE p.run_id = ? AND m.created_at_utc = p.created_at_utc
            """,
            (run_id,),
        ).fetchone()[0]
        return PredictionRegistration(
            run_id=run_id,
            input_rows=input_rows,
            matches_inserted=int(matches),
            predictions_inserted=int(row["predictions_inserted"]),
            valid_predictions=int(row["valid_predictions"]),
            official_predictions_selected=int(
                row["official_predictions_selected"]
            ),
            settlements_inserted=int(row["settlements_inserted"]),
            queued_rows=int(run["queued_rows"]),
            reused=True,
        )

    def reconcile_observations(
        self,
        run_id: str,
        observations: pd.DataFrame,
        *,
        source_system: str = "tennis_explorer",
        metadata: Mapping[str, object] | None = None,
        observed_at_utc: object | None = None,
    ) -> ObservationReconciliation:
        """Añade observaciones y liquida únicamente mediante ``winner_slug``."""

        run_key = required_text(run_id, "run_id")
        source = required_text(source_system, "source_system")
        require_frame_columns(
            observations,
            OBSERVATION_REQUIRED_COLUMNS,
            frame_name="observations",
        )
        rows = _row_dicts(observations)
        metadata_payload = dict(metadata or {})
        observed_argument = (
            canonical_utc_datetime(
                observed_at_utc,
                "observed_at_utc",
            )
            if optional_scalar(observed_at_utc) is not None
            else None
        )
        payload_hash = _canonical_frame_payload(
            rows,
            run_type="observation",
            source_system=source,
            target_date=None,
            metadata=metadata_payload,
            observed_at_utc=observed_argument,
        )
        existing = self._existing_run(
            run_key,
            run_type="observation",
            payload_hash=payload_hash,
        )
        if existing is not None:
            return self._observation_reuse_summary(run_key, len(rows))

        now = self._now()
        observations_inserted = 0
        settlements_inserted = 0
        conflicts_inserted = 0
        queued_rows = 0
        self._begin()
        try:
            self._insert_run(
                run_id=run_key,
                run_type="observation",
                source_system=source,
                target_date=None,
                input_rows=len(rows),
                payload_hash=payload_hash,
                metadata=metadata_payload,
                now=now,
            )
            for row in rows:
                row_payload_json = canonical_json(row)
                row_payload_hash = payload_sha256(row)
                try:
                    source_match_id = derive_source_match_id(
                        row,
                        source_system=source,
                    )
                except OperationsValidationError as exc:
                    queued_rows += int(
                        self._queue_issue(
                            issue_type="missing_stable_match_identity",
                            run_id=run_key,
                            source_match_id=None,
                            observation_id=None,
                            payload={
                                "error": str(exc),
                                "row_payload_sha256": row_payload_hash,
                            },
                            now=now,
                        )
                    )
                    continue
                validity = validate_observation_row(row)
                observed_value = optional_scalar(
                    row.get("observed_at_utc")
                )
                observed_at = (
                    canonical_utc_datetime(
                        observed_value,
                        "observed_at_utc",
                    )
                    if observed_value is not None
                    else observed_argument or now
                )
                snapshot_id = self._register_snapshot(
                    row,
                    run_id=run_key,
                    source_system=source,
                    retrieved_at_utc=observed_at,
                    now=now,
                )
                observation_id = _identifier(
                    "observation",
                    {
                        "source_system": source,
                        "source_match_id": source_match_id,
                        "payload_sha256": row_payload_hash,
                    },
                )
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO observations(
                        observation_id, source_match_id, source_system,
                        snapshot_id, observed_at_utc, status,
                        player_1_sets_won, player_2_sets_won, sets_score,
                        winner_side, winner_slug, result_evidence,
                        is_valid, invalid_reason, payload_sha256,
                        payload_json, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        source_match_id,
                        source,
                        snapshot_id,
                        observed_at,
                        validity.status,
                        validity.player_1_sets_won,
                        validity.player_2_sets_won,
                        validity.sets_score,
                        validity.winner_side,
                        validity.winner_slug,
                        validity.result_evidence,
                        int(validity.is_valid),
                        validity.invalid_reason,
                        row_payload_hash,
                        row_payload_json,
                        now,
                    ),
                )
                inserted = cursor.rowcount == 1
                observations_inserted += int(inserted)
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO run_observations(
                        run_id, observation_id
                    ) VALUES (?, ?)
                    """,
                    (run_key, observation_id),
                )
                match_exists = (
                    self.connection.execute(
                        """
                        SELECT 1 FROM matches WHERE source_match_id = ?
                        """,
                        (source_match_id,),
                    ).fetchone()
                    is not None
                )
                if not match_exists:
                    queued_rows += int(
                        self._queue_issue(
                            issue_type="unknown_match",
                            run_id=run_key,
                            source_match_id=source_match_id,
                            observation_id=observation_id,
                            payload={
                                "row_payload_sha256": row_payload_hash,
                            },
                            now=now,
                        )
                    )
                    continue
                if not validity.is_valid:
                    if validity.status == "finished" or (
                        validity.invalid_reason or ""
                    ).startswith("winner_present"):
                        queued_rows += int(
                            self._queue_issue(
                                issue_type="invalid_terminal_observation",
                                run_id=run_key,
                                source_match_id=source_match_id,
                                observation_id=observation_id,
                                payload={
                                    "reason": validity.invalid_reason,
                                },
                                now=now,
                            )
                        )
                    continue
                outcome = self._settle_observation(
                    observation_id,
                    run_id=run_key,
                    now=now,
                )
                settlements_inserted += int(outcome == "inserted")
                conflicts_inserted += int(outcome == "conflict_inserted")
                if outcome in {"queued", "conflict_inserted"}:
                    queued_rows += 1
            self._complete_run(
                run_key,
                accepted_rows=observations_inserted,
                queued_rows=queued_rows,
                now=now,
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return ObservationReconciliation(
            run_id=run_key,
            input_rows=len(rows),
            observations_inserted=observations_inserted,
            settlements_inserted=settlements_inserted,
            conflicts_inserted=conflicts_inserted,
            queued_rows=queued_rows,
            reused=False,
        )

    def _settle_observation(
        self,
        observation_id: str,
        *,
        run_id: str,
        now: str,
    ) -> str:
        """Intenta liquidar una observación sin usar lado ni nombre visible."""

        row = self.connection.execute(
            """
            SELECT
                o.observation_id,
                o.source_match_id,
                o.winner_slug,
                o.is_valid,
                op.prediction_id AS official_prediction_id,
                p.player_a_slug,
                p.player_b_slug
            FROM observations AS o
            LEFT JOIN official_predictions AS op
                ON op.source_match_id = o.source_match_id
            LEFT JOIN predictions AS p
                ON p.prediction_id = op.prediction_id
            WHERE o.observation_id = ?
            """,
            (observation_id,),
        ).fetchone()
        if row is None or int(row["is_valid"]) != 1:
            return "ignored"
        source_match_id = str(row["source_match_id"])
        if row["official_prediction_id"] is None:
            inserted = self._queue_issue(
                issue_type="missing_official_prediction",
                run_id=run_id,
                source_match_id=source_match_id,
                observation_id=observation_id,
                payload={"message": "No existe predicción oficial válida."},
                now=now,
            )
            return "queued" if inserted else "already_queued"

        winner_slug = str(row["winner_slug"])
        player_a_slug = str(row["player_a_slug"])
        player_b_slug = str(row["player_b_slug"])
        if winner_slug == player_a_slug:
            actual_outcome_a = 1
        elif winner_slug == player_b_slug:
            actual_outcome_a = 0
        else:
            _, inserted = self._record_conflict(
                conflict_type="winner_slug_not_in_official_prediction",
                run_id=run_id,
                source_match_id=source_match_id,
                existing_reference=str(row["official_prediction_id"]),
                incoming_reference=observation_id,
                details={
                    "winner_slug": winner_slug,
                    "official_player_a_slug": player_a_slug,
                    "official_player_b_slug": player_b_slug,
                    "winner_side_ignored": True,
                },
                observation_id=observation_id,
                now=now,
            )
            return "conflict_inserted" if inserted else "already_conflict"

        existing = self.connection.execute(
            """
            SELECT settlement_id, winner_slug
            FROM settlements
            WHERE source_match_id = ?
            """,
            (source_match_id,),
        ).fetchone()
        if existing is not None:
            if existing["winner_slug"] == winner_slug:
                return "already_settled"
            _, inserted = self._record_conflict(
                conflict_type="settlement_winner_conflict",
                run_id=run_id,
                source_match_id=source_match_id,
                existing_reference=str(existing["settlement_id"]),
                incoming_reference=observation_id,
                details={
                    "settled_winner_slug": existing["winner_slug"],
                    "incoming_winner_slug": winner_slug,
                },
                observation_id=observation_id,
                now=now,
            )
            return "conflict_inserted" if inserted else "already_conflict"

        settlement_id = _identifier(
            "settlement",
            {
                "source_match_id": source_match_id,
                "official_prediction_id": row["official_prediction_id"],
            },
        )
        self.connection.execute(
            """
            INSERT INTO settlements(
                settlement_id, source_match_id, official_prediction_id,
                observation_id, winner_slug, actual_outcome_a,
                settled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                settlement_id,
                source_match_id,
                row["official_prediction_id"],
                observation_id,
                winner_slug,
                actual_outcome_a,
                now,
            ),
        )
        self.connection.execute(
            """
            UPDATE review_queue
            SET status = 'resolved', resolved_at_utc = ?
            WHERE source_match_id = ?
              AND status = 'open'
              AND issue_type IN (
                  'unknown_match', 'missing_official_prediction'
              )
            """,
            (now, source_match_id),
        )
        return "inserted"

    def _settle_pending_observations(
        self,
        source_match_id: str,
        *,
        fallback_run_id: str,
        now: str,
    ) -> int:
        """Reintenta observaciones válidas recibidas antes de la predicción."""

        rows = self.connection.execute(
            """
            SELECT
                o.observation_id,
                (
                    SELECT ro.run_id
                    FROM run_observations AS ro
                    WHERE ro.observation_id = o.observation_id
                    ORDER BY ro.run_id
                    LIMIT 1
                ) AS observation_run_id
            FROM observations AS o
            WHERE o.source_match_id = ? AND o.is_valid = 1
            ORDER BY o.observed_at_utc, o.created_at_utc, o.observation_id
            """,
            (source_match_id,),
        ).fetchall()
        inserted = 0
        for row in rows:
            outcome = self._settle_observation(
                str(row["observation_id"]),
                run_id=str(row["observation_run_id"] or fallback_run_id),
                now=now,
            )
            inserted += int(outcome == "inserted")
        return inserted

    def _observation_reuse_summary(
        self,
        run_id: str,
        input_rows: int,
    ) -> ObservationReconciliation:
        """Reconstruye contadores de una conciliación idempotente."""

        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS observations_inserted,
                COALESCE(SUM(
                    CASE WHEN s.observation_id = o.observation_id
                    THEN 1 ELSE 0 END
                ), 0) AS settlements_inserted
            FROM run_observations AS ro
            JOIN observations AS o
                ON o.observation_id = ro.observation_id
            LEFT JOIN settlements AS s
                ON s.observation_id = o.observation_id
            WHERE ro.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        run = self.connection.execute(
            "SELECT queued_rows FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        conflicts = self.connection.execute(
            "SELECT COUNT(*) FROM conflicts WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        return ObservationReconciliation(
            run_id=run_id,
            input_rows=input_rows,
            observations_inserted=int(row["observations_inserted"]),
            settlements_inserted=int(row["settlements_inserted"]),
            conflicts_inserted=int(conflicts),
            queued_rows=int(run["queued_rows"]),
            reused=True,
        )

    def append_player_statistics(
        self,
        run_id: str,
        statistics: pd.DataFrame,
    ) -> StatisticsRegistration:
        """Añade estadísticas prepartido en formato largo y append-only.

        La ejecución indicada debe existir. El método rechaza nombres o
        ``source_kind`` asociados a predicciones, labels o resultados para
        impedir que esas señales entren accidentalmente como features.
        """

        run_key = required_text(run_id, "run_id")
        require_frame_columns(
            statistics,
            STATISTICS_REQUIRED_COLUMNS,
            frame_name="statistics",
        )
        run = self.connection.execute(
            """
            SELECT status FROM runs WHERE run_id = ?
            """,
            (run_key,),
        ).fetchone()
        if run is None or run["status"] != "complete":
            raise OperationsValidationError(
                "Las estadísticas requieren una ejecución completa existente."
            )
        rows = _row_dicts(statistics)
        now = self._now()
        inserted = 0
        self._begin()
        try:
            for row in rows:
                source_match_id = optional_text(
                    row.get("source_match_id"),
                    "source_match_id",
                )
                match = None
                if source_match_id is not None:
                    match = self.connection.execute(
                        """
                        SELECT player_1_slug, player_2_slug
                        FROM matches WHERE source_match_id = ?
                        """,
                        (source_match_id,),
                    ).fetchone()
                    if match is None:
                        raise OperationsValidationError(
                            "source_match_id de estadística no existe: "
                            f"{source_match_id}."
                        )
                player_slug = canonical_slug(
                    row.get("player_slug"),
                    "player_slug",
                )
                if player_slug is None:
                    raise OperationsValidationError(
                        "player_slug de estadística no puede ser nulo."
                    )
                if (
                    match is not None
                    and match["player_1_slug"] is not None
                    and match["player_2_slug"] is not None
                    and player_slug
                    not in {match["player_1_slug"], match["player_2_slug"]}
                ):
                    raise OperationsValidationError(
                        "player_slug no pertenece al partido indicado."
                    )
                gender = required_text(row.get("gender"), "gender")
                if gender not in {"M", "F"}:
                    raise OperationsValidationError(
                        "gender debe ser 'M' o 'F'."
                    )
                as_of_date = canonical_date(
                    row.get("as_of_date"),
                    "as_of_date",
                )
                statistic_name = validate_statistic_name(
                    row.get("statistic_name")
                )
                source_kind = validate_source_kind(row.get("source_kind"))
                statistic_value = optional_float(
                    row.get("statistic_value"),
                    "statistic_value",
                )
                source_snapshot_id = optional_text(
                    row.get("source_snapshot_id"),
                    "source_snapshot_id",
                )
                if source_snapshot_id is not None:
                    snapshot_exists = self.connection.execute(
                        """
                        SELECT 1 FROM snapshots WHERE snapshot_id = ?
                        """,
                        (source_snapshot_id,),
                    ).fetchone()
                    if snapshot_exists is None:
                        raise OperationsValidationError(
                            "source_snapshot_id de estadística no existe."
                        )
                payload = {
                    "source_match_id": source_match_id,
                    "player_slug": player_slug,
                    "player_id": optional_integer(
                        row.get("player_id"),
                        "player_id",
                    ),
                    "gender": gender,
                    "as_of_date": as_of_date,
                    "statistic_name": statistic_name,
                    "statistic_value": statistic_value,
                    "source_kind": source_kind,
                    "source_snapshot_id": source_snapshot_id,
                }
                statistic_id = _identifier(
                    "statistic",
                    {"run_id": run_key, "payload": payload},
                )
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO player_statistics(
                        statistic_id, run_id, source_match_id, player_slug,
                        player_id, gender, as_of_date, statistic_name,
                        statistic_value, source_kind, source_snapshot_id,
                        payload_sha256, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        statistic_id,
                        run_key,
                        source_match_id,
                        player_slug,
                        payload["player_id"],
                        gender,
                        as_of_date,
                        statistic_name,
                        statistic_value,
                        source_kind,
                        source_snapshot_id,
                        payload_sha256(payload),
                        now,
                    ),
                )
                inserted += int(cursor.rowcount == 1)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return StatisticsRegistration(
            run_id=run_key,
            input_rows=len(rows),
            statistics_inserted=inserted,
        )

    def load_official_predictions(self) -> pd.DataFrame:
        """Devuelve las predicciones oficiales con su orientación original."""

        return pd.read_sql_query(
            """
            SELECT
                op.source_match_id,
                op.prediction_id,
                op.selection_rule,
                op.selected_at_utc,
                p.run_id,
                p.player_a_slug,
                p.player_b_slug,
                p.model_probability_a,
                p.model_probability_b,
                p.market_probability_a,
                p.market_probability_b,
                p.model_fingerprint
            FROM official_predictions AS op
            JOIN predictions AS p
                ON p.prediction_id = op.prediction_id
            ORDER BY op.selected_at_utc, op.source_match_id
            """,
            self.connection,
        )

    def load_settlements(self) -> pd.DataFrame:
        """Devuelve labels observados ya conciliados con la oficial."""

        return pd.read_sql_query(
            """
            SELECT *
            FROM settlements
            ORDER BY settled_at_utc, source_match_id
            """,
            self.connection,
        )

    def load_review_queue(
        self,
        *,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Consulta la cola completa o solo un estado ``open``/``resolved``."""

        if status is None:
            return pd.read_sql_query(
                """
                SELECT * FROM review_queue
                ORDER BY created_at_utc, queue_id
                """,
                self.connection,
            )
        status_value = required_text(status, "status")
        if status_value not in {"open", "resolved"}:
            raise OperationsValidationError(
                "status de cola debe ser 'open' o 'resolved'."
            )
        return pd.read_sql_query(
            """
            SELECT * FROM review_queue
            WHERE status = ?
            ORDER BY created_at_utc, queue_id
            """,
            self.connection,
            params=(status_value,),
        )


def register_prediction_dataframe(
    run_id: str,
    predictions: pd.DataFrame,
    *,
    database_path: str | Path = OPERATIONS_DATABASE_PATH,
    source_system: str = "tennis_explorer",
    metadata: Mapping[str, object] | None = None,
    target_date: object | None = None,
) -> PredictionRegistration:
    """Atajo autocontenido para registrar un DataFrame de predicciones."""

    with OperationsStore(database_path) as store:
        return store.register_prediction_run(
            run_id,
            predictions,
            source_system=source_system,
            metadata=metadata,
            target_date=target_date,
        )


def reconcile_observation_dataframe(
    run_id: str,
    observations: pd.DataFrame,
    *,
    database_path: str | Path = OPERATIONS_DATABASE_PATH,
    source_system: str = "tennis_explorer",
    metadata: Mapping[str, object] | None = None,
    observed_at_utc: object | None = None,
) -> ObservationReconciliation:
    """Atajo autocontenido para registrar y conciliar observaciones."""

    with OperationsStore(database_path) as store:
        return store.reconcile_observations(
            run_id,
            observations,
            source_system=source_system,
            metadata=metadata,
            observed_at_utc=observed_at_utc,
        )

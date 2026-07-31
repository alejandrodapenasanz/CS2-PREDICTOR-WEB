"""Persistencia SQLite transaccional para ejecuciones y estados Elo.

El almacén conserva ejecuciones separadas por género, checkpoints por fecha y
un historial ancho de ratings. Cada fila de ``rating_history`` representa el
estado posterior a procesar ``state_date``; las consultas temporales deben usar
siempre ``state_date < as_of_date``.

SQLite se configura deliberadamente con journal tradicional ``DELETE`` y un
único escritor. No se activa WAL, para evitar archivos laterales persistentes
en el directorio sincronizado del proyecto.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence, cast

from .types import Gender, PlayerEloState


RatingHistoryRow = PlayerEloState

DEFAULT_INITIAL_RATING: Final[float] = 1500.0
DEFAULT_SURFACE_WEIGHT: Final[float] = 0.5
WRITE_TIMEOUT_SECONDS: Final[float] = 30.0
READ_TIMEOUT_SECONDS: Final[float] = 10.0
MAX_BATCH_QUERIES: Final[int] = 300

_SHA1_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")

_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    source_commit TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building', 'complete', 'failed')),
    created_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    min_event_date TEXT,
    max_event_date TEXT,
    failure_reason TEXT,
    CHECK (
        (status = 'complete' AND completed_at_utc IS NOT NULL)
        OR status != 'complete'
    )
);

CREATE TABLE IF NOT EXISTS date_blocks (
    run_id TEXT NOT NULL,
    block_date TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    match_count INTEGER NOT NULL CHECK (match_count >= 0),
    excluded_count INTEGER NOT NULL CHECK (excluded_count >= 0),
    PRIMARY KEY (run_id, block_date),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS rating_history (
    run_id TEXT NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    player_id INTEGER NOT NULL,
    state_date TEXT NOT NULL,
    general_elo REAL NOT NULL,
    hard_elo REAL NOT NULL,
    clay_elo REAL NOT NULL,
    grass_elo REAL NOT NULL,
    carpet_elo REAL NOT NULL,
    general_matches INTEGER NOT NULL CHECK (general_matches >= 0),
    hard_matches INTEGER NOT NULL CHECK (hard_matches >= 0),
    clay_matches INTEGER NOT NULL CHECK (clay_matches >= 0),
    grass_matches INTEGER NOT NULL CHECK (grass_matches >= 0),
    carpet_matches INTEGER NOT NULL CHECK (carpet_matches >= 0),
    PRIMARY KEY (run_id, gender, player_id, state_date),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS active_runs (
    gender TEXT PRIMARY KEY CHECK (gender IN ('M', 'F')),
    run_id TEXT NOT NULL UNIQUE,
    activated_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_runs_gender_status
    ON runs(gender, status);

CREATE INDEX IF NOT EXISTS idx_date_blocks_run_date
    ON date_blocks(run_id, block_date);

CREATE INDEX IF NOT EXISTS idx_rating_history_lookup
    ON rating_history(run_id, gender, player_id, state_date DESC);

CREATE TRIGGER IF NOT EXISTS trg_date_blocks_building_run
BEFORE INSERT ON date_blocks
WHEN NOT EXISTS (
    SELECT 1
    FROM runs
    WHERE run_id = NEW.run_id
      AND status = 'building'
)
BEGIN
    SELECT RAISE(ABORT, 'date_blocks require a building run');
END;

CREATE TRIGGER IF NOT EXISTS trg_rating_history_building_run
BEFORE INSERT ON rating_history
WHEN NOT EXISTS (
    SELECT 1
    FROM runs
    WHERE run_id = NEW.run_id
      AND gender = NEW.gender
      AND status = 'building'
)
BEGIN
    SELECT RAISE(ABORT, 'rating_history requires a matching building run');
END;

CREATE TRIGGER IF NOT EXISTS trg_active_runs_complete_insert
BEFORE INSERT ON active_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM runs
    WHERE run_id = NEW.run_id
      AND gender = NEW.gender
      AND status = 'complete'
)
BEGIN
    SELECT RAISE(ABORT, 'active_runs require a complete run');
END;

CREATE TRIGGER IF NOT EXISTS trg_active_runs_complete_update
BEFORE UPDATE ON active_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM runs
    WHERE run_id = NEW.run_id
      AND gender = NEW.gender
      AND status = 'complete'
)
BEGIN
    SELECT RAISE(ABORT, 'active_runs require a complete run');
END;
"""


class EloStoreError(RuntimeError):
    """Indica un fallo de persistencia o integridad del almacén Elo."""


class EloValidationError(EloStoreError):
    """Indica que un argumento no cumple el contrato persistente."""


class EloRunNotFoundError(EloStoreError):
    """Indica que no existe la ejecución Elo solicitada."""


class EloRunNotAvailableError(EloStoreError):
    """Indica que una ejecución existe pero todavía no es consultable."""


@dataclass(frozen=True)
class RunRecord:
    """Metadatos inmutables de una ejecución Elo persistida."""

    run_id: str
    gender: Gender
    source_commit: str
    algorithm_version: str
    input_fingerprint: str
    parameters: Mapping[str, Any]
    status: str
    created_at_utc: str
    completed_at_utc: str | None
    min_event_date: date | None
    max_event_date: date | None
    failure_reason: str | None


def _validate_gender(gender: object) -> Gender:
    """Valida un género sin inferir ni normalizar universos."""

    if gender not in {"M", "F"}:
        raise EloValidationError("gender debe ser exactamente 'M' o 'F'.")
    return cast(Gender, gender)


def _validate_date(value: object, name: str) -> date:
    """Acepta exclusivamente ``date`` y rechaza ``datetime`` y strings."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise EloValidationError(f"{name} debe ser datetime.date estricto.")
    return value


def _validate_non_empty_text(value: object, name: str) -> str:
    """Valida un texto obligatorio sin modificar su contenido."""

    if not isinstance(value, str) or not value:
        raise EloValidationError(f"{name} debe ser un texto no vacío.")
    return value


def _validate_player_id(player_id: object) -> int:
    """Valida un identificador entero y rechaza booleanos."""

    if isinstance(player_id, bool) or not isinstance(player_id, int):
        raise EloValidationError("player_id debe ser un entero.")
    return player_id


def _validate_rating(value: object, name: str) -> float:
    """Valida que un rating sea numérico, finito y no booleano."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EloValidationError(f"{name} debe ser numérico.")
    result = float(value)
    if not math.isfinite(result):
        raise EloValidationError(f"{name} debe ser finito.")
    return result


def _validate_count(value: object, name: str) -> int:
    """Valida un contador entero no negativo."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EloValidationError(f"{name} debe ser un entero no negativo.")
    return value


def _utc_iso(value: datetime | None = None) -> str:
    """Convierte un instante consciente a la representación ISO UTC."""

    active_value = value or datetime.now(UTC)
    if active_value.tzinfo is None or active_value.utcoffset() is None:
        raise EloValidationError("El instante debe incluir zona horaria.")
    return active_value.astimezone(UTC).isoformat()


def _optional_iso_date(value: object) -> date | None:
    """Convierte una fecha ISO opcional leída desde SQLite."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise EloStoreError("SQLite devolvió una fecha con tipo inesperado.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise EloStoreError(f"Fecha ISO inválida en SQLite: {value!r}.") from exc


def _normalise_parameters(
    parameters: Mapping[str, Any] | None,
    *,
    initial_rating: float | None,
    surface_weight: float | None,
) -> dict[str, Any]:
    """Combina parámetros adicionales con los dos valores usados por consultas."""

    payload = dict(parameters or {})
    configured_initial = payload.get(
        "initial_rating",
        DEFAULT_INITIAL_RATING,
    )
    configured_weight = payload.get(
        "surface_weight",
        DEFAULT_SURFACE_WEIGHT,
    )
    validated_configured_initial = _validate_rating(
        configured_initial,
        "parameters.initial_rating",
    )
    validated_configured_weight = _validate_rating(
        configured_weight,
        "parameters.surface_weight",
    )
    resolved_initial = (
        validated_configured_initial
        if initial_rating is None
        else initial_rating
    )
    resolved_weight = (
        validated_configured_weight
        if surface_weight is None
        else surface_weight
    )
    if (
        initial_rating is not None
        and "initial_rating" in payload
        and validated_configured_initial
        != _validate_rating(initial_rating, "initial_rating")
    ):
        raise EloValidationError(
            "initial_rating contradice el valor incluido en parameters."
        )
    if (
        surface_weight is not None
        and "surface_weight" in payload
        and validated_configured_weight
        != _validate_rating(surface_weight, "surface_weight")
    ):
        raise EloValidationError(
            "surface_weight contradice el valor incluido en parameters."
        )

    payload["initial_rating"] = _validate_rating(
        resolved_initial,
        "initial_rating",
    )
    weight = _validate_rating(resolved_weight, "surface_weight")
    if not 0.0 <= weight <= 1.0:
        raise EloValidationError("surface_weight debe estar entre 0 y 1.")
    payload["surface_weight"] = weight
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EloValidationError(
            "parameters debe poder serializarse como JSON finito."
        ) from exc
    return payload


def _row_to_run_record(row: sqlite3.Row) -> RunRecord:
    """Convierte una fila SQLite validada en metadatos de ejecución."""

    try:
        parameters = json.loads(row["parameters_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise EloStoreError(
            f"parameters_json inválido para el run {row['run_id']!r}."
        ) from exc
    if not isinstance(parameters, Mapping):
        raise EloStoreError(
            f"parameters_json no es un objeto para el run {row['run_id']!r}."
        )
    return RunRecord(
        run_id=row["run_id"],
        gender=cast(Gender, row["gender"]),
        source_commit=row["source_commit"],
        algorithm_version=row["algorithm_version"],
        input_fingerprint=row["input_fingerprint"],
        parameters=parameters,
        status=row["status"],
        created_at_utc=row["created_at_utc"],
        completed_at_utc=row["completed_at_utc"],
        min_event_date=_optional_iso_date(row["min_event_date"]),
        max_event_date=_optional_iso_date(row["max_event_date"]),
        failure_reason=row["failure_reason"],
    )


def _row_to_rating_record(row: sqlite3.Row) -> RatingHistoryRow:
    """Convierte una fila SQLite en un estado ancho tipado."""

    return PlayerEloState(
        gender=cast(Gender, row["gender"]),
        player_id=row["player_id"],
        state_date=_validate_date(
            _optional_iso_date(row["state_date"]),
            "state_date",
        ),
        general_elo=row["general_elo"],
        hard_elo=row["hard_elo"],
        clay_elo=row["clay_elo"],
        grass_elo=row["grass_elo"],
        carpet_elo=row["carpet_elo"],
        general_matches=row["general_matches"],
        hard_matches=row["hard_matches"],
        clay_matches=row["clay_matches"],
        grass_matches=row["grass_matches"],
        carpet_matches=row["carpet_matches"],
    )


@contextmanager
def _immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Abre una transacción de escritura inmediata con rollback seguro."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


class EloStore:
    """Gestiona un archivo SQLite Elo mediante conexiones de vida corta."""

    def __init__(self, db_path: Path) -> None:
        """Conserva la ruta explícita de la base sin abrirla todavía."""

        self.db_path = Path(db_path)

    def _open_writer(self) -> sqlite3.Connection:
        """Abre una conexión escritora con journal DELETE e integridad activa."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.db_path,
                timeout=WRITE_TIMEOUT_SECONDS,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            journal_mode = connection.execute(
                "PRAGMA journal_mode = DELETE"
            ).fetchone()[0]
            if str(journal_mode).casefold() != "delete":
                connection.close()
                raise EloStoreError(
                    "SQLite no aceptó journal_mode=DELETE."
                )
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise EloStoreError(
                f"No se pudo abrir la base Elo para escritura: {self.db_path}."
            ) from exc

    def _open_reader(self) -> sqlite3.Connection:
        """Abre la base existente en modo de solo lectura y consulta."""

        if not self.db_path.is_file():
            raise EloStoreError(f"No existe la base Elo: {self.db_path}.")
        database_uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        try:
            connection = sqlite3.connect(
                database_uri,
                uri=True,
                timeout=READ_TIMEOUT_SECONDS,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo abrir la base Elo para lectura: {self.db_path}."
            ) from exc

    def initialize(self) -> None:
        """Crea de forma idempotente tablas, índices y triggers."""

        connection = self._open_writer()
        try:
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{_SCHEMA_SQL}\nCOMMIT;"
            )
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise EloStoreError(
                f"No se pudo inicializar el esquema Elo: {self.db_path}."
            ) from exc
        finally:
            connection.close()

    def create_run(
        self,
        *,
        run_id: str,
        gender: Gender,
        source_commit: str,
        algorithm_version: str,
        input_fingerprint: str,
        parameters: Mapping[str, Any] | None = None,
        initial_rating: float | None = None,
        surface_weight: float | None = None,
        created_at: datetime | None = None,
    ) -> RunRecord:
        """Crea una ejecución ``building`` con parámetros reproducibles."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        validated_gender = _validate_gender(gender)
        validated_commit = _validate_non_empty_text(
            source_commit,
            "source_commit",
        )
        if _SHA1_PATTERN.fullmatch(validated_commit) is None:
            raise EloValidationError(
                "source_commit debe ser un SHA-1 Git hexadecimal de 40 caracteres."
            )
        validated_version = _validate_non_empty_text(
            algorithm_version,
            "algorithm_version",
        )
        validated_fingerprint = _validate_non_empty_text(
            input_fingerprint,
            "input_fingerprint",
        )
        parameter_payload = _normalise_parameters(
            parameters,
            initial_rating=initial_rating,
            surface_weight=surface_weight,
        )
        created_at_utc = _utc_iso(created_at)

        connection = self._open_writer()
        try:
            with _immediate_transaction(connection):
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id,
                        gender,
                        source_commit,
                        algorithm_version,
                        input_fingerprint,
                        parameters_json,
                        status,
                        created_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'building', ?)
                    """,
                    (
                        validated_run_id,
                        validated_gender,
                        validated_commit,
                        validated_version,
                        validated_fingerprint,
                        json.dumps(
                            parameter_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        created_at_utc,
                    ),
                )
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo crear el run Elo {validated_run_id!r}."
            ) from exc
        finally:
            connection.close()
        return self.get_run(validated_run_id)

    def write_date_block(
        self,
        *,
        run_id: str,
        block_date: date,
        input_hash: str,
        match_count: int,
        ratings: Iterable[RatingHistoryRow],
        excluded_count: int = 0,
    ) -> None:
        """Persiste atómicamente un checkpoint y todos sus estados posteriores."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        validated_date = _validate_date(block_date, "block_date")
        validated_hash = _validate_non_empty_text(input_hash, "input_hash")
        validated_match_count = _validate_count(match_count, "match_count")
        validated_excluded_count = _validate_count(
            excluded_count,
            "excluded_count",
        )
        materialised_ratings = tuple(ratings)
        run = self.get_run(validated_run_id)
        if run.status != "building":
            raise EloRunNotAvailableError(
                f"El run {validated_run_id!r} no está en estado building."
            )

        seen_keys: set[tuple[Gender, int, date]] = set()
        rows: list[tuple[object, ...]] = []
        for rating in materialised_ratings:
            if not isinstance(rating, RatingHistoryRow):
                raise EloValidationError(
                    "ratings debe contener exclusivamente RatingHistoryRow."
                )
            rating_gender = _validate_gender(rating.gender)
            if rating_gender != run.gender:
                raise EloValidationError(
                    "Un estado no puede cruzar el género de su run."
                )
            player_id = _validate_player_id(rating.player_id)
            state_date = _validate_date(rating.state_date, "state_date")
            if state_date != validated_date:
                raise EloValidationError(
                    "state_date debe coincidir con block_date."
                )
            key = (rating_gender, player_id, state_date)
            if key in seen_keys:
                raise EloValidationError(
                    f"Estado duplicado dentro del bloque: {key!r}."
                )
            seen_keys.add(key)
            rows.append(
                (
                    validated_run_id,
                    rating_gender,
                    player_id,
                    state_date.isoformat(),
                    _validate_rating(rating.general_elo, "general_elo"),
                    _validate_rating(rating.hard_elo, "hard_elo"),
                    _validate_rating(rating.clay_elo, "clay_elo"),
                    _validate_rating(rating.grass_elo, "grass_elo"),
                    _validate_rating(rating.carpet_elo, "carpet_elo"),
                    _validate_count(
                        rating.general_matches,
                        "general_matches",
                    ),
                    _validate_count(rating.hard_matches, "hard_matches"),
                    _validate_count(rating.clay_matches, "clay_matches"),
                    _validate_count(rating.grass_matches, "grass_matches"),
                    _validate_count(rating.carpet_matches, "carpet_matches"),
                )
            )

        connection = self._open_writer()
        try:
            with _immediate_transaction(connection):
                latest_block_row = connection.execute(
                    """
                    SELECT MAX(block_date) AS latest_block_date
                    FROM date_blocks
                    WHERE run_id = ?
                    """,
                    (validated_run_id,),
                ).fetchone()
                latest_block_date = latest_block_row["latest_block_date"]
                if (
                    latest_block_date is not None
                    and validated_date.isoformat() <= latest_block_date
                ):
                    raise EloValidationError(
                        f"block_date debe ser posterior al último bloque "
                        f"persistido ({latest_block_date}) del run "
                        f"{validated_run_id!r}."
                    )
                connection.execute(
                    """
                    INSERT INTO date_blocks (
                        run_id,
                        block_date,
                        input_hash,
                        match_count,
                        excluded_count
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        validated_run_id,
                        validated_date.isoformat(),
                        validated_hash,
                        validated_match_count,
                        validated_excluded_count,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO rating_history (
                        run_id,
                        gender,
                        player_id,
                        state_date,
                        general_elo,
                        hard_elo,
                        clay_elo,
                        grass_elo,
                        carpet_elo,
                        general_matches,
                        hard_matches,
                        clay_matches,
                        grass_matches,
                        carpet_matches
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.execute(
                    """
                    UPDATE runs
                    SET min_event_date = CASE
                            WHEN min_event_date IS NULL
                              OR min_event_date > ?
                            THEN ?
                            ELSE min_event_date
                        END,
                        max_event_date = CASE
                            WHEN max_event_date IS NULL
                              OR max_event_date < ?
                            THEN ?
                            ELSE max_event_date
                        END
                    WHERE run_id = ?
                    """,
                    (
                        validated_date.isoformat(),
                        validated_date.isoformat(),
                        validated_date.isoformat(),
                        validated_date.isoformat(),
                        validated_run_id,
                    ),
                )
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo persistir el bloque {validated_date} "
                f"del run {validated_run_id!r}."
            ) from exc
        finally:
            connection.close()

    def activate_run(
        self,
        *,
        run_id: str,
        completed_at: datetime | None = None,
    ) -> RunRecord:
        """Completa el run y cambia el puntero activo en una sola transacción."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        completed_at_utc = _utc_iso(completed_at)
        connection = self._open_writer()
        try:
            with _immediate_transaction(connection):
                row = connection.execute(
                    "SELECT gender, status FROM runs WHERE run_id = ?",
                    (validated_run_id,),
                ).fetchone()
                if row is None:
                    raise EloRunNotFoundError(
                        f"No existe el run Elo {validated_run_id!r}."
                    )
                if row["status"] == "complete":
                    active = connection.execute(
                        """
                        SELECT run_id
                        FROM active_runs
                        WHERE gender = ?
                        """,
                        (row["gender"],),
                    ).fetchone()
                    if active is not None and active["run_id"] == validated_run_id:
                        continue_activation = False
                    else:
                        raise EloRunNotAvailableError(
                            "Un run completo histórico no se reactiva "
                            "implícitamente."
                        )
                else:
                    continue_activation = True
                if row["status"] not in {"building", "complete"}:
                    raise EloRunNotAvailableError(
                        f"El run {validated_run_id!r} no está en building."
                    )
                if continue_activation:
                    connection.execute(
                        """
                        UPDATE runs
                        SET status = 'complete',
                            completed_at_utc = ?,
                            failure_reason = NULL
                        WHERE run_id = ?
                        """,
                        (completed_at_utc, validated_run_id),
                    )
                    connection.execute(
                        """
                        INSERT INTO active_runs (
                            gender,
                            run_id,
                            activated_at_utc
                        )
                        VALUES (?, ?, ?)
                        ON CONFLICT(gender) DO UPDATE SET
                            run_id = excluded.run_id,
                            activated_at_utc = excluded.activated_at_utc
                        """,
                        (row["gender"], validated_run_id, completed_at_utc),
                    )
        except (EloRunNotFoundError, EloRunNotAvailableError):
            raise
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo activar el run Elo {validated_run_id!r}."
            ) from exc
        finally:
            connection.close()
        return self.get_run(validated_run_id)

    def mark_run_failed(
        self,
        *,
        run_id: str,
        reason: str,
    ) -> RunRecord:
        """Marca un run en construcción como fallido sin alterar el activo."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        validated_reason = _validate_non_empty_text(reason, "reason")
        connection = self._open_writer()
        try:
            with _immediate_transaction(connection):
                cursor = connection.execute(
                    """
                    UPDATE runs
                    SET status = 'failed',
                        failure_reason = ?
                    WHERE run_id = ?
                      AND status = 'building'
                    """,
                    (validated_reason, validated_run_id),
                )
                if cursor.rowcount != 1:
                    raise EloRunNotAvailableError(
                        f"El run {validated_run_id!r} no está en building."
                    )
        except EloRunNotAvailableError:
            raise
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo marcar como fallido {validated_run_id!r}."
            ) from exc
        finally:
            connection.close()
        return self.get_run(validated_run_id)

    def get_run(self, run_id: str) -> RunRecord:
        """Devuelve los metadatos de un run, sea o no consultable."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        connection = self._open_reader()
        try:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (validated_run_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise EloStoreError(
                f"No se pudo consultar el run {validated_run_id!r}."
            ) from exc
        finally:
            connection.close()
        if row is None:
            raise EloRunNotFoundError(
                f"No existe el run Elo {validated_run_id!r}."
            )
        return _row_to_run_record(row)

    def resolve_complete_run(
        self,
        *,
        gender: Gender,
        run_id: str | None = None,
    ) -> RunRecord:
        """Resuelve un run explícito o el activo, exigiendo estado ``complete``."""

        validated_gender = _validate_gender(gender)
        connection = self._open_reader()
        try:
            if run_id is None:
                row = connection.execute(
                    """
                    SELECT runs.*
                    FROM active_runs
                    JOIN runs ON runs.run_id = active_runs.run_id
                    WHERE active_runs.gender = ?
                      AND runs.gender = ?
                      AND runs.status = 'complete'
                    """,
                    (validated_gender, validated_gender),
                ).fetchone()
                if row is None:
                    raise EloRunNotAvailableError(
                        f"No hay un run Elo activo para {validated_gender}."
                    )
            else:
                validated_run_id = _validate_non_empty_text(run_id, "run_id")
                row = connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?",
                    (validated_run_id,),
                ).fetchone()
                if row is None:
                    raise EloRunNotFoundError(
                        f"No existe el run Elo {validated_run_id!r}."
                    )
                if row["gender"] != validated_gender:
                    raise EloRunNotAvailableError(
                        "El run solicitado pertenece a otro género."
                    )
                if row["status"] != "complete":
                    raise EloRunNotAvailableError(
                        f"El run {validated_run_id!r} no está completo."
                    )
        except (
            EloRunNotAvailableError,
            EloRunNotFoundError,
        ):
            raise
        except sqlite3.Error as exc:
            raise EloStoreError("No se pudo resolver el run Elo.") from exc
        finally:
            connection.close()
        return _row_to_run_record(row)

    def fetch_rating_before(
        self,
        *,
        run_id: str,
        gender: Gender,
        player_id: int,
        as_of_date: date,
    ) -> PlayerEloState | None:
        """Obtiene el último estado con fecha estrictamente anterior al corte."""

        results = self.fetch_ratings_before(
            run_id=run_id,
            gender=gender,
            queries=((player_id, as_of_date),),
        )
        return results[0]

    def fetch_ratings_before(
        self,
        *,
        run_id: str,
        gender: Gender,
        queries: Sequence[tuple[int, date]],
    ) -> tuple[PlayerEloState | None, ...]:
        """Resuelve por lotes estados ``< as_of_date`` preservando el orden."""

        validated_run_id = _validate_non_empty_text(run_id, "run_id")
        validated_gender = _validate_gender(gender)
        validated_queries = tuple(
            (
                _validate_player_id(player_id),
                _validate_date(as_of_date, "as_of_date"),
            )
            for player_id, as_of_date in queries
        )
        self.resolve_complete_run(
            gender=validated_gender,
            run_id=validated_run_id,
        )
        if not validated_queries:
            return ()

        connection = self._open_reader()
        collected: list[PlayerEloState | None] = []
        try:
            for offset in range(0, len(validated_queries), MAX_BATCH_QUERIES):
                chunk = validated_queries[
                    offset : offset + MAX_BATCH_QUERIES
                ]
                value_clause = ", ".join("(?, ?, ?)" for _ in chunk)
                parameters: list[object] = []
                for local_index, (player_id, cutoff) in enumerate(chunk):
                    parameters.extend(
                        (local_index, player_id, cutoff.isoformat())
                    )
                parameters.extend((validated_run_id, validated_gender))
                rows = connection.execute(
                    f"""
                    WITH requested(query_index, player_id, as_of_date) AS (
                        VALUES {value_clause}
                    )
                    SELECT
                        requested.query_index,
                        history.gender,
                        history.player_id,
                        history.state_date,
                        history.general_elo,
                        history.hard_elo,
                        history.clay_elo,
                        history.grass_elo,
                        history.carpet_elo,
                        history.general_matches,
                        history.hard_matches,
                        history.clay_matches,
                        history.grass_matches,
                        history.carpet_matches
                    FROM requested
                    LEFT JOIN rating_history AS history
                      ON history.rowid = (
                          SELECT candidate.rowid
                          FROM rating_history AS candidate
                          JOIN runs
                            ON runs.run_id = candidate.run_id
                          WHERE candidate.run_id = ?
                            AND candidate.gender = ?
                            AND candidate.player_id = requested.player_id
                            AND candidate.state_date < requested.as_of_date
                            AND runs.status = 'complete'
                          ORDER BY candidate.state_date DESC
                          LIMIT 1
                      )
                    ORDER BY requested.query_index
                    """,
                    parameters,
                ).fetchall()
                if len(rows) != len(chunk):
                    raise EloStoreError(
                        "SQLite no devolvió una fila por consulta Elo."
                    )
                collected.extend(
                    None
                    if row["state_date"] is None
                    else _row_to_rating_record(row)
                    for row in rows
                )
        except sqlite3.Error as exc:
            raise EloStoreError(
                "No se pudieron resolver las consultas Elo por lotes."
            ) from exc
        finally:
            connection.close()
        return tuple(collected)

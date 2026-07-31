"""Persistencia SQLite auditable para identidades de Tennis Explorer.

El almacén usa ``(gender, slug)`` como clave primaria, porque los universos
masculino y femenino son independientes. Una resolución automática solo puede
insertar una clave nueva: nunca actualiza una asociación ya persistida. Las
correcciones requieren un override manual con motivo no vacío y dejan una fila
append-only con los valores anteriores y posteriores.

SQLite se configura con journal ``DELETE`` y sincronización ``FULL`` para
evitar artefactos WAL persistentes en el directorio sincronizado del proyecto.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
import re
import sqlite3
from typing import Final, Literal, cast

from .types import Gender, PlayerMappingError

ResolutionMethod = Literal["automatic_name", "manual_override"]
MappingKey = tuple[Gender, str]

WRITE_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_KEYS_PER_QUERY: Final[int] = 400
SCHEMA_VERSION: Final[int] = 1

_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[^/?#\s]+$")
_IOC_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")

_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS player_mappings (
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    slug TEXT NOT NULL CHECK (
        length(slug) > 0
        AND instr(slug, '/') = 0
        AND instr(slug, '?') = 0
        AND instr(slug, '#') = 0
    ),
    player_id INTEGER NOT NULL CHECK (player_id > 0),
    visible_name TEXT NOT NULL CHECK (length(visible_name) > 0),
    sackmann_player_name TEXT NOT NULL CHECK (
        length(sackmann_player_name) > 0
    ),
    sackmann_ioc TEXT CHECK (
        sackmann_ioc IS NULL
        OR (
            length(sackmann_ioc) = 3
            AND sackmann_ioc = upper(sackmann_ioc)
        )
    ),
    resolution_method TEXT NOT NULL CHECK (
        resolution_method IN ('automatic_name', 'manual_override')
    ),
    first_resolved_date TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (gender, slug)
);

CREATE TABLE IF NOT EXISTS mapping_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'automatic_insert',
            'manual_insert',
            'manual_correction'
        )
    ),
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    slug TEXT NOT NULL,
    old_player_id INTEGER,
    new_player_id INTEGER NOT NULL CHECK (new_player_id > 0),
    old_visible_name TEXT,
    new_visible_name TEXT NOT NULL,
    old_sackmann_player_name TEXT,
    new_sackmann_player_name TEXT NOT NULL,
    old_sackmann_ioc TEXT,
    new_sackmann_ioc TEXT,
    old_resolution_method TEXT,
    new_resolution_method TEXT NOT NULL CHECK (
        new_resolution_method IN ('automatic_name', 'manual_override')
    ),
    old_first_resolved_date TEXT,
    new_first_resolved_date TEXT NOT NULL,
    old_updated_at_utc TEXT,
    new_updated_at_utc TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    changed_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mapping_audit_identity
    ON mapping_audit(gender, slug, audit_id);

CREATE TRIGGER IF NOT EXISTS trg_mapping_audit_no_update
BEFORE UPDATE ON mapping_audit
BEGIN
    SELECT RAISE(ABORT, 'mapping_audit is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_mapping_audit_no_delete
BEFORE DELETE ON mapping_audit
BEGIN
    SELECT RAISE(ABORT, 'mapping_audit is append-only');
END;
"""


class PlayerMappingStoreError(PlayerMappingError):
    """Indica un argumento inválido o un fallo de integridad/persistencia."""


@dataclass(frozen=True)
class MappingRecord:
    """Representa una asociación persistida entre slug e identidad Sackmann."""

    gender: Gender
    slug: str
    player_id: int
    visible_name: str
    sackmann_player_name: str
    sackmann_ioc: str | None
    resolution_method: ResolutionMethod
    first_resolved_date: date
    updated_at_utc: datetime


def _validate_gender(value: object) -> Gender:
    """Acepta exclusivamente los dos universos de género declarados."""

    if value not in {"M", "F"}:
        raise PlayerMappingStoreError(
            "gender debe ser exactamente 'M' o 'F'."
        )
    return cast(Gender, value)


def _validate_slug(value: object) -> str:
    """Valida un segmento de URL sin modificar ni inferir su contenido."""

    if not isinstance(value, str) or _SLUG_PATTERN.fullmatch(value) is None:
        raise PlayerMappingStoreError(
            "slug debe ser texto no vacío sin espacios, '/', '?' ni '#'."
        )
    return value


def _validate_player_id(value: object) -> int:
    """Valida un identificador Sackmann entero positivo y no booleano."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlayerMappingStoreError(
            "player_id debe ser un entero positivo."
        )
    return value


def _validate_required_text(value: object, field: str) -> str:
    """Valida texto obligatorio y rechaza valores compuestos solo por espacios."""

    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
    ):
        raise PlayerMappingStoreError(
            f"{field} debe ser un texto no vacío."
        )
    return value


def _validate_ioc(value: object) -> str | None:
    """Valida un IOC Sackmann opcional en su forma canónica de tres letras."""

    if value is None:
        return None
    if not isinstance(value, str) or _IOC_PATTERN.fullmatch(value) is None:
        raise PlayerMappingStoreError(
            "sackmann_ioc debe ser None o tres letras ASCII mayúsculas."
        )
    return value


def _validate_date(value: object, field: str) -> date:
    """Acepta una fecha pura y rechaza datetimes o conversiones implícitas."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise PlayerMappingStoreError(
            f"{field} debe ser datetime.date estricto."
        )
    return value


def _validate_clock_value(value: object) -> datetime:
    """Convierte un instante consciente a UTC sin perder precisión."""

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PlayerMappingStoreError(
            "clock debe devolver datetime consciente de zona horaria."
        )
    return value.astimezone(UTC)


def _parse_iso_date(value: object, field: str) -> date:
    """Convierte una fecha ISO leída de SQLite y detecta corrupción."""

    if not isinstance(value, str):
        raise PlayerMappingStoreError(
            f"SQLite devolvió {field} con un tipo inesperado."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PlayerMappingStoreError(
            f"SQLite devolvió {field} con una fecha ISO inválida."
        ) from exc
    return parsed


def _parse_utc_datetime(value: object, field: str) -> datetime:
    """Convierte un instante SQLite y exige que represente UTC."""

    if not isinstance(value, str):
        raise PlayerMappingStoreError(
            f"SQLite devolvió {field} con un tipo inesperado."
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PlayerMappingStoreError(
            f"SQLite devolvió {field} con un timestamp ISO inválido."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PlayerMappingStoreError(
            f"SQLite devolvió {field} sin una zona UTC inequívoca."
        )
    return parsed.astimezone(UTC)


def _row_to_record(row: sqlite3.Row) -> MappingRecord:
    """Convierte y valida una fila SQLite antes de exponerla al llamador."""

    method = row["resolution_method"]
    if method not in {"automatic_name", "manual_override"}:
        raise PlayerMappingStoreError(
            "SQLite devolvió un resolution_method desconocido."
        )
    return MappingRecord(
        gender=_validate_gender(row["gender"]),
        slug=_validate_slug(row["slug"]),
        player_id=_validate_player_id(row["player_id"]),
        visible_name=_validate_required_text(
            row["visible_name"],
            "visible_name",
        ),
        sackmann_player_name=_validate_required_text(
            row["sackmann_player_name"],
            "sackmann_player_name",
        ),
        sackmann_ioc=_validate_ioc(row["sackmann_ioc"]),
        resolution_method=cast(ResolutionMethod, method),
        first_resolved_date=_parse_iso_date(
            row["first_resolved_date"],
            "first_resolved_date",
        ),
        updated_at_utc=_parse_utc_datetime(
            row["updated_at_utc"],
            "updated_at_utc",
        ),
    )


@contextmanager
def _immediate_transaction(
    connection: sqlite3.Connection,
) -> Iterator[None]:
    """Ejecuta una escritura atómica con bloqueo inmediato y rollback seguro."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


class PlayerMappingStore:
    """Gestiona el mapa persistente durante un bloque ``with`` explícito."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Conserva ruta y reloj inyectable sin abrir todavía la base."""

        self.db_path = Path(db_path)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> PlayerMappingStore:
        """Abre, configura e inicializa el archivo SQLite."""

        if self._connection is not None:
            raise PlayerMappingStoreError(
                "PlayerMappingStore no puede abrirse dos veces a la vez."
            )
        self._connection = self._open_connection()
        try:
            self._initialize_schema()
        except BaseException:
            self._connection.close()
            self._connection = None
            raise
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Cierra la conexión y nunca suprime la excepción del bloque."""

        if self._connection is not None:
            if self._connection.in_transaction:
                self._connection.rollback()
            self._connection.close()
            self._connection = None

    def _open_connection(self) -> sqlite3.Connection:
        """Abre una conexión escritora con durabilidad y FK activas."""

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
                raise PlayerMappingStoreError(
                    "SQLite no aceptó journal_mode=DELETE."
                )
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise PlayerMappingStoreError(
                f"No se pudo abrir la base de mapeo: {self.db_path}."
            ) from exc
        except BaseException:
            if connection is not None:
                connection.close()
            raise

    def _initialize_schema(self) -> None:
        """Crea el esquema idempotente y valida su versión declarada."""

        connection = self._require_connection()
        try:
            observed_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if observed_version not in {0, SCHEMA_VERSION}:
                raise PlayerMappingStoreError(
                    "Versión de esquema de mapeo no compatible: "
                    f"{observed_version}."
                )
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{_SCHEMA_SQL}\n"
                f"PRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;"
            )
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise PlayerMappingStoreError(
                f"No se pudo inicializar el esquema de mapeo: {self.db_path}."
            ) from exc

    def _require_connection(self) -> sqlite3.Connection:
        """Devuelve la conexión activa o exige usar el context manager."""

        if self._connection is None:
            raise PlayerMappingStoreError(
                "PlayerMappingStore debe usarse dentro de un bloque 'with'."
            )
        return self._connection

    def _now(self) -> datetime:
        """Lee una sola vez el reloj inyectado y devuelve el instante UTC."""

        try:
            value = self._clock()
        except Exception as exc:
            raise PlayerMappingStoreError(
                "clock falló al producir el timestamp de auditoría."
            ) from exc
        return _validate_clock_value(value)

    def get(self, gender: Gender, slug: str) -> MappingRecord | None:
        """Busca una clave exacta sin mezclar los universos de género."""

        validated_gender = _validate_gender(gender)
        validated_slug = _validate_slug(slug)
        connection = self._require_connection()
        try:
            row = connection.execute(
                """
                SELECT *
                FROM player_mappings
                WHERE gender = ? AND slug = ?
                """,
                (validated_gender, validated_slug),
            ).fetchone()
        except sqlite3.Error as exc:
            raise PlayerMappingStoreError(
                "No se pudo consultar el mapa de jugadores."
            ) from exc
        return None if row is None else _row_to_record(row)

    def get_many(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], MappingRecord]:
        """Consulta claves por lotes y devuelve solo las asociaciones presentes."""

        try:
            materialised = tuple(keys)
        except TypeError as exc:
            raise PlayerMappingStoreError(
                "keys debe ser un iterable de pares (gender, slug)."
            ) from exc

        validated: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for key in materialised:
            if not isinstance(key, tuple) or len(key) != 2:
                raise PlayerMappingStoreError(
                    "Cada elemento de keys debe ser una tupla "
                    "(gender, slug)."
                )
            validated_key = (
                _validate_gender(key[0]),
                _validate_slug(key[1]),
            )
            if validated_key not in seen:
                validated.append(validated_key)
                seen.add(validated_key)

        connection = self._require_connection()
        found: dict[tuple[str, str], MappingRecord] = {}
        try:
            for offset in range(0, len(validated), MAX_KEYS_PER_QUERY):
                chunk = validated[offset : offset + MAX_KEYS_PER_QUERY]
                if not chunk:
                    continue
                values_sql = ", ".join("(?, ?)" for _ in chunk)
                parameters = [
                    value
                    for key in chunk
                    for value in key
                ]
                rows = connection.execute(
                    f"""
                    WITH requested(gender, slug) AS (
                        VALUES {values_sql}
                    )
                    SELECT mappings.*
                    FROM requested
                    JOIN player_mappings AS mappings
                      ON mappings.gender = requested.gender
                     AND mappings.slug = requested.slug
                    """,
                    parameters,
                ).fetchall()
                for row in rows:
                    record = _row_to_record(row)
                    found[(record.gender, record.slug)] = record
        except sqlite3.Error as exc:
            raise PlayerMappingStoreError(
                "No se pudo consultar el mapa de jugadores por lotes."
            ) from exc
        return {
            key: found[key]
            for key in validated
            if key in found
        }

    def put_automatic(
        self,
        *,
        gender: Gender,
        slug: str,
        player_id: int,
        visible_name: str,
        sackmann_player_name: str,
        sackmann_ioc: str | None,
        first_resolved_date: date,
    ) -> MappingRecord:
        """Inserta una resolución automática o devuelve la existente intacta."""

        values = self._validated_mapping_values(
            gender=gender,
            slug=slug,
            player_id=player_id,
            visible_name=visible_name,
            sackmann_player_name=sackmann_player_name,
            sackmann_ioc=sackmann_ioc,
            first_resolved_date=first_resolved_date,
        )
        now = self._now()
        connection = self._require_connection()
        try:
            with _immediate_transaction(connection):
                existing = self._select_mapping(
                    connection,
                    values["gender"],
                    values["slug"],
                )
                if existing is None:
                    self._insert_mapping(
                        connection,
                        values=values,
                        resolution_method="automatic_name",
                        updated_at=now,
                    )
                    self._insert_audit(
                        connection,
                        event_type="automatic_insert",
                        old=None,
                        values=values,
                        resolution_method="automatic_name",
                        updated_at=now,
                        reason="automatic_name",
                    )
        except sqlite3.Error as exc:
            raise PlayerMappingStoreError(
                "No se pudo persistir la resolución automática."
            ) from exc

        result = self.get(
            cast(Gender, values["gender"]),
            cast(str, values["slug"]),
        )
        if result is None:
            raise PlayerMappingStoreError(
                "La resolución automática no quedó persistida."
            )
        return result

    def apply_manual_override(
        self,
        *,
        gender: Gender,
        slug: str,
        player_id: int,
        visible_name: str,
        sackmann_player_name: str,
        sackmann_ioc: str | None,
        first_resolved_date: date,
        reason: str,
    ) -> MappingRecord:
        """Inserta o corrige una asociación con motivo manual obligatorio."""

        values = self._validated_mapping_values(
            gender=gender,
            slug=slug,
            player_id=player_id,
            visible_name=visible_name,
            sackmann_player_name=sackmann_player_name,
            sackmann_ioc=sackmann_ioc,
            first_resolved_date=first_resolved_date,
        )
        validated_reason = _validate_required_text(reason, "reason")
        now = self._now()
        connection = self._require_connection()
        try:
            with _immediate_transaction(connection):
                existing_row = self._select_mapping(
                    connection,
                    values["gender"],
                    values["slug"],
                )
                old = (
                    None
                    if existing_row is None
                    else _row_to_record(existing_row)
                )
                if old is None:
                    event_type = "manual_insert"
                    first_date = cast(date, values["first_resolved_date"])
                    self._insert_mapping(
                        connection,
                        values=values,
                        resolution_method="manual_override",
                        updated_at=now,
                    )
                else:
                    event_type = "manual_correction"
                    first_date = old.first_resolved_date
                    values["first_resolved_date"] = first_date
                    connection.execute(
                        """
                        UPDATE player_mappings
                        SET
                            player_id = ?,
                            visible_name = ?,
                            sackmann_player_name = ?,
                            sackmann_ioc = ?,
                            resolution_method = 'manual_override',
                            updated_at_utc = ?
                        WHERE gender = ? AND slug = ?
                        """,
                        (
                            values["player_id"],
                            values["visible_name"],
                            values["sackmann_player_name"],
                            values["sackmann_ioc"],
                            now.isoformat(),
                            values["gender"],
                            values["slug"],
                        ),
                    )
                self._insert_audit(
                    connection,
                    event_type=event_type,
                    old=old,
                    values=values,
                    resolution_method="manual_override",
                    updated_at=now,
                    reason=validated_reason,
                )
        except sqlite3.Error as exc:
            raise PlayerMappingStoreError(
                "No se pudo aplicar el override manual."
            ) from exc

        result = self.get(
            cast(Gender, values["gender"]),
            cast(str, values["slug"]),
        )
        if result is None:
            raise PlayerMappingStoreError(
                "El override manual no quedó persistido."
            )
        return result

    def list_mappings(self) -> tuple[MappingRecord, ...]:
        """Lista todas las asociaciones ordenadas por género y slug."""

        connection = self._require_connection()
        try:
            rows = connection.execute(
                """
                SELECT *
                FROM player_mappings
                ORDER BY gender, slug
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise PlayerMappingStoreError(
                "No se pudo listar el mapa de jugadores."
            ) from exc
        return tuple(_row_to_record(row) for row in rows)

    def _validated_mapping_values(
        self,
        *,
        gender: object,
        slug: object,
        player_id: object,
        visible_name: object,
        sackmann_player_name: object,
        sackmann_ioc: object,
        first_resolved_date: object,
    ) -> dict[str, object]:
        """Valida todos los campos de una asociación antes de una transacción."""

        return {
            "gender": _validate_gender(gender),
            "slug": _validate_slug(slug),
            "player_id": _validate_player_id(player_id),
            "visible_name": _validate_required_text(
                visible_name,
                "visible_name",
            ),
            "sackmann_player_name": _validate_required_text(
                sackmann_player_name,
                "sackmann_player_name",
            ),
            "sackmann_ioc": _validate_ioc(sackmann_ioc),
            "first_resolved_date": _validate_date(
                first_resolved_date,
                "first_resolved_date",
            ),
        }

    def _select_mapping(
        self,
        connection: sqlite3.Connection,
        gender: object,
        slug: object,
    ) -> sqlite3.Row | None:
        """Lee una asociación dentro de la transacción activa."""

        return connection.execute(
            """
            SELECT *
            FROM player_mappings
            WHERE gender = ? AND slug = ?
            """,
            (gender, slug),
        ).fetchone()

    def _insert_mapping(
        self,
        connection: sqlite3.Connection,
        *,
        values: dict[str, object],
        resolution_method: ResolutionMethod,
        updated_at: datetime,
    ) -> None:
        """Inserta la fila principal ya validada."""

        first_date = cast(date, values["first_resolved_date"])
        connection.execute(
            """
            INSERT INTO player_mappings (
                gender,
                slug,
                player_id,
                visible_name,
                sackmann_player_name,
                sackmann_ioc,
                resolution_method,
                first_resolved_date,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["gender"],
                values["slug"],
                values["player_id"],
                values["visible_name"],
                values["sackmann_player_name"],
                values["sackmann_ioc"],
                resolution_method,
                first_date.isoformat(),
                updated_at.isoformat(),
            ),
        )

    def _insert_audit(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        old: MappingRecord | None,
        values: dict[str, object],
        resolution_method: ResolutionMethod,
        updated_at: datetime,
        reason: str,
    ) -> None:
        """Añade un evento con el estado completo anterior y posterior."""

        first_date = cast(date, values["first_resolved_date"])
        connection.execute(
            """
            INSERT INTO mapping_audit (
                event_type,
                gender,
                slug,
                old_player_id,
                new_player_id,
                old_visible_name,
                new_visible_name,
                old_sackmann_player_name,
                new_sackmann_player_name,
                old_sackmann_ioc,
                new_sackmann_ioc,
                old_resolution_method,
                new_resolution_method,
                old_first_resolved_date,
                new_first_resolved_date,
                old_updated_at_utc,
                new_updated_at_utc,
                reason,
                changed_at_utc
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                event_type,
                values["gender"],
                values["slug"],
                None if old is None else old.player_id,
                values["player_id"],
                None if old is None else old.visible_name,
                values["visible_name"],
                None if old is None else old.sackmann_player_name,
                values["sackmann_player_name"],
                None if old is None else old.sackmann_ioc,
                values["sackmann_ioc"],
                None if old is None else old.resolution_method,
                resolution_method,
                (
                    None
                    if old is None
                    else old.first_resolved_date.isoformat()
                ),
                first_date.isoformat(),
                (
                    None
                    if old is None
                    else old.updated_at_utc.isoformat()
                ),
                updated_at.isoformat(),
                reason,
                updated_at.isoformat(),
            ),
        )

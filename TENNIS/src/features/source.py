"""Entrada causal y acotada en memoria para construir features históricas.

El módulo vuelca las filas verificadas del manifiesto Sackmann a un SQLite
temporal. Reutiliza la frontera pública de la fase 3 para que fechas,
identificadores y ``source_record_hash`` tengan exactamente la misma semántica
que el sistema Elo. Después permite recorrer bloques de fecha completos en
orden estable sin cargar el histórico entero en memoria.

El archivo SQLite es un artefacto de *staging*: este módulo nunca borra ni
reemplaza una ruta existente. El llamador es responsable de crear una ruta
nueva y de gestionar su ciclo de vida.
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Final, Iterator, Literal

import pandas as pd

from src.elo import (
    iter_manifest_match_frames,
    prepare_manifest_event_frame,
)
from src.elo.build import VerifiedManifest


Gender = Literal["M", "F"]

DEFAULT_FEATURE_SOURCE_CHUNKSIZE: Final[int] = 50_000
FEATURE_SPOOL_TABLE: Final[str] = "_feature_input_rows"
FEATURE_SPOOL_INDEX: Final[str] = "idx_feature_spool_order"
FEATURE_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "tourney_date",
    "winner_id",
    "loser_id",
    "surface",
    "tourney_level",
    "score",
    "source_family",
    "source_commit",
    "source_path",
    "source_row_number",
    "source_record_hash",
    "tourney_id",
    "tourney_name",
    "match_num",
    "round",
    "best_of",
)
_RAW_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "surface",
    "tourney_level",
    "score",
    "source_family",
    "source_commit",
    "source_path",
    "source_record_hash",
    "tourney_id",
    "tourney_name",
    "match_num",
    "round",
)
_POSITIVE_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+$")
_SQLITE_SIGNED_INTEGER_MAX: Final[int] = (1 << 63) - 1


class FeatureSourceError(RuntimeError):
    """Indica que no se pudo construir o consultar el spool de features."""


class FeatureSourceValidationError(FeatureSourceError):
    """Indica que una entrada no cumple el contrato explícito del spool."""


class ExistingFeatureSpoolError(FeatureSourceError):
    """Indica que la ruta de staging ya existe y no debe sobrescribirse."""


def _validate_gender(gender: object) -> Gender:
    """Valida un universo de género sin aceptar conversiones implícitas."""

    if gender not in ("M", "F"):
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    return gender


def _validate_chunksize(chunksize: object) -> int:
    """Valida el tamaño de lectura como entero estrictamente positivo."""

    if (
        isinstance(chunksize, bool)
        or not isinstance(chunksize, int)
        or chunksize <= 0
    ):
        raise ValueError("chunksize debe ser un entero positivo.")
    return chunksize


def _normalise_spool_path(spool_path: Path) -> Path:
    """Normaliza una ruta Path sin crear directorios ni archivos."""

    if not isinstance(spool_path, Path):
        raise TypeError("spool_path debe ser pathlib.Path.")
    return spool_path.resolve()


def _parse_best_of_column(frame: pd.DataFrame) -> tuple[int | None, ...]:
    """Convierte ``best_of`` vacío a nulo y exige enteros positivos.

    Args:
        frame: Chunk ya tipado por :func:`prepare_manifest_event_frame`.

    Returns:
        Valores compatibles con ``INTEGER NULL`` de SQLite.

    Raises:
        FeatureSourceValidationError: Si un valor presente no es un entero
            positivo representable por SQLite.
    """

    parsed: list[int | None] = []
    for position, raw_value in enumerate(frame["best_of"].tolist()):
        source_path = str(frame["source_path"].iloc[position])
        source_row = int(frame["source_row_number"].iloc[position])
        if not isinstance(raw_value, str):
            raise FeatureSourceValidationError(
                f"{source_path}, fila {source_row}: best_of no es texto raw."
            )
        stripped = raw_value.strip()
        if not stripped:
            parsed.append(None)
            continue
        if _POSITIVE_INTEGER_PATTERN.fullmatch(stripped) is None:
            raise FeatureSourceValidationError(
                f"{source_path}, fila {source_row}: best_of={raw_value!r} "
                "no es un entero positivo."
            )
        value = int(stripped)
        if value <= 0 or value > _SQLITE_SIGNED_INTEGER_MAX:
            raise FeatureSourceValidationError(
                f"{source_path}, fila {source_row}: best_of={raw_value!r} "
                "no es un entero positivo representable por SQLite."
            )
        parsed.append(value)
    return tuple(parsed)


def _validate_prepared_frame(
    frame: pd.DataFrame,
    *,
    gender: Gender,
) -> None:
    """Comprueba columnas y género tras aplicar el contrato común de fase 3."""

    missing = sorted(set(FEATURE_SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise FeatureSourceValidationError(
            f"El chunk preparado carece de columnas requeridas: {missing}."
        )
    observed_genders = set(frame["gender"].astype("string").tolist())
    if observed_genders != {gender}:
        raise FeatureSourceValidationError(
            "El chunk preparado mezcla universos de género o no coincide con "
            f"la selección: esperado={gender!r}, observado="
            f"{sorted(observed_genders)!r}."
        )


def _create_feature_spool(spool_path: Path) -> sqlite3.Connection:
    """Crea una base nueva con journal DELETE y un esquema explícito."""

    if spool_path.exists():
        raise ExistingFeatureSpoolError(
            f"El spool ya existe y no se sobrescribirá: {spool_path}."
        )
    if not spool_path.parent.is_dir():
        raise FeatureSourceError(
            f"El directorio del spool no existe: {spool_path.parent}."
        )

    try:
        connection = sqlite3.connect(spool_path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            f"""
            CREATE TABLE {FEATURE_SPOOL_TABLE} (
                gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
                tourney_date TEXT NOT NULL,
                winner_id INTEGER NOT NULL,
                loser_id INTEGER NOT NULL,
                surface TEXT NOT NULL,
                tourney_level TEXT NOT NULL,
                score TEXT NOT NULL,
                source_family TEXT NOT NULL,
                source_commit TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_row_number INTEGER NOT NULL,
                source_record_hash TEXT NOT NULL,
                tourney_id TEXT NOT NULL,
                tourney_name TEXT NOT NULL,
                match_num TEXT NOT NULL,
                round_name TEXT NOT NULL,
                best_of INTEGER,
                UNIQUE (source_path, source_row_number)
            )
            """
        )
        connection.commit()
    except sqlite3.Error as exc:
        try:
            connection.close()
        except UnboundLocalError:
            pass
        raise FeatureSourceError(
            f"No se pudo crear el spool de features {spool_path}."
        ) from exc
    return connection


def _prepared_records(
    frame: pd.DataFrame,
    *,
    best_of_values: tuple[int | None, ...],
) -> Iterator[tuple[object, ...]]:
    """Transforma un chunk validado en registros SQLite deterministas."""

    selected_columns = (
        "gender",
        "tourney_date",
        "winner_id",
        "loser_id",
        "surface",
        "tourney_level",
        "score",
        "source_family",
        "source_commit",
        "source_path",
        "source_row_number",
        "source_record_hash",
        "tourney_id",
        "tourney_name",
        "match_num",
        "round",
    )
    for position, values in enumerate(
        frame.loc[:, list(selected_columns)].itertuples(
            index=False,
            name=None,
        )
    ):
        (
            gender,
            event_date,
            winner_id,
            loser_id,
            surface,
            tourney_level,
            score,
            source_family,
            source_commit,
            source_path,
            source_row_number,
            source_record_hash,
            tourney_id,
            tourney_name,
            match_num,
            round_name,
        ) = values
        yield (
            str(gender),
            pd.Timestamp(event_date).date().isoformat(),
            int(winner_id),
            int(loser_id),
            str(surface),
            str(tourney_level),
            str(score),
            str(source_family),
            str(source_commit),
            str(source_path),
            int(source_row_number),
            str(source_record_hash),
            str(tourney_id),
            str(tourney_name),
            str(match_num),
            str(round_name),
            best_of_values[position],
        )


def spool_manifest_feature_rows(
    manifest: VerifiedManifest,
    *,
    gender: Gender,
    spool_path: Path,
    chunksize: int = DEFAULT_FEATURE_SOURCE_CHUNKSIZE,
) -> int:
    """Vuelca un género verificado a un SQLite temporal ordenable.

    Las filas provienen exclusivamente de
    :func:`src.elo.iter_manifest_match_frames` y se preparan con
    :func:`src.elo.prepare_manifest_event_frame`. Así se comparte con Elo la
    interpretación estricta de ``tourney_date``, IDs y duplicado exacto.

    Args:
        manifest: Manifiesto ya validado mediante
            :func:`src.elo.load_verified_manifest`.
        gender: Universo aislado, exactamente ``"M"`` o ``"F"``.
        spool_path: Ruta nueva del SQLite de staging.
        chunksize: Máximo de filas fuente leídas en cada chunk.

    Returns:
        Número total de filas insertadas.

    Raises:
        ExistingFeatureSpoolError: Si ``spool_path`` ya existe.
        FeatureSourceValidationError: Si ``best_of`` u otro contrato de fila
            es inválido.
        FeatureSourceError: Si SQLite no puede completar la operación.
    """

    selected_gender = _validate_gender(gender)
    selected_chunksize = _validate_chunksize(chunksize)
    resolved_spool = _normalise_spool_path(spool_path)
    if not isinstance(manifest, VerifiedManifest):
        raise TypeError(
            "manifest debe ser el resultado de load_verified_manifest()."
        )

    connection = _create_feature_spool(resolved_spool)
    insert_sql = f"""
        INSERT INTO {FEATURE_SPOOL_TABLE} (
            gender,
            tourney_date,
            winner_id,
            loser_id,
            surface,
            tourney_level,
            score,
            source_family,
            source_commit,
            source_path,
            source_row_number,
            source_record_hash,
            tourney_id,
            tourney_name,
            match_num,
            round_name,
            best_of
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    try:
        for raw_frame in iter_manifest_match_frames(
            manifest,
            gender=selected_gender,
            chunksize=selected_chunksize,
        ):
            if raw_frame.empty:
                continue
            prepared = prepare_manifest_event_frame(raw_frame)
            _validate_prepared_frame(
                prepared,
                gender=selected_gender,
            )
            best_of_values = _parse_best_of_column(prepared)
            connection.executemany(
                insert_sql,
                _prepared_records(
                    prepared,
                    best_of_values=best_of_values,
                ),
            )
            inserted += len(prepared)
            connection.commit()
        connection.execute(
            f"""
            CREATE INDEX {FEATURE_SPOOL_INDEX}
            ON {FEATURE_SPOOL_TABLE} (
                gender,
                tourney_date,
                source_path,
                source_row_number
            )
            """
        )
        connection.commit()
    except (FeatureSourceValidationError, TypeError, ValueError):
        connection.rollback()
        raise
    except sqlite3.Error as exc:
        connection.rollback()
        raise FeatureSourceError(
            f"No se pudo construir el spool de features {resolved_spool}."
        ) from exc
    finally:
        connection.close()
    return inserted


def _validate_spool_schema(connection: sqlite3.Connection) -> None:
    """Comprueba que la tabla consultada tiene el esquema creado aquí."""

    expected = (
        "gender",
        "tourney_date",
        "winner_id",
        "loser_id",
        "surface",
        "tourney_level",
        "score",
        "source_family",
        "source_commit",
        "source_path",
        "source_row_number",
        "source_record_hash",
        "tourney_id",
        "tourney_name",
        "match_num",
        "round_name",
        "best_of",
    )
    observed = tuple(
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({FEATURE_SPOOL_TABLE})"
        ).fetchall()
    )
    if observed != expected:
        raise FeatureSourceValidationError(
            "El SQLite no contiene el esquema de spool esperado: "
            f"observado={observed!r}."
        )


def _typed_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Restaura dtypes públicos tras leer un bloque de fecha desde SQLite."""

    result = frame.reset_index(drop=True).copy()
    result["tourney_date"] = pd.to_datetime(
        result["tourney_date"],
        format="%Y-%m-%d",
        errors="raise",
    )
    for column in ("winner_id", "loser_id", "source_row_number"):
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("int64")
    result["best_of"] = pd.to_numeric(
        result["best_of"],
        errors="raise",
    ).astype("Int64")
    for column in ("gender", *_RAW_TEXT_COLUMNS):
        result[column] = result[column].astype("string")
    return result.loc[:, list(FEATURE_SOURCE_COLUMNS)]


def iter_spooled_feature_date_frames(
    spool_path: Path,
    *,
    gender: Gender,
    chunksize: int = DEFAULT_FEATURE_SOURCE_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Emite bloques de fecha completos en orden cronológico y estable.

    El límite ``chunksize`` controla las lecturas desde SQLite, no fragmenta
    una fecha: si una jornada cruza uno o varios chunks, sus filas se retienen
    hasta poder emitirla completa. Dentro de la fecha el orden es
    ``source_path`` y ``source_row_number``.

    Args:
        spool_path: SQLite creado por
            :func:`spool_manifest_feature_rows`.
        gender: Universo aislado, exactamente ``"M"`` o ``"F"``.
        chunksize: Máximo de filas por lectura SQLite.

    Yields:
        Un DataFrame tipado por fecha, nunca una fecha parcial.

    Raises:
        FeatureSourceError: Si la ruta no existe o SQLite falla.
        FeatureSourceValidationError: Si el esquema o los valores persistidos
            no cumplen el contrato.
    """

    selected_gender = _validate_gender(gender)
    selected_chunksize = _validate_chunksize(chunksize)
    resolved_spool = _normalise_spool_path(spool_path)
    if not resolved_spool.is_file():
        raise FeatureSourceError(
            f"No existe el spool de features: {resolved_spool}."
        )

    query = f"""
        SELECT
            gender,
            tourney_date,
            winner_id,
            loser_id,
            surface,
            tourney_level,
            score,
            source_family,
            source_commit,
            source_path,
            source_row_number,
            source_record_hash,
            tourney_id,
            tourney_name,
            match_num,
            round_name AS round,
            best_of
        FROM {FEATURE_SPOOL_TABLE}
        WHERE gender = ?
        ORDER BY
            tourney_date,
            source_path,
            source_row_number
    """
    connection: sqlite3.Connection | None = None
    pending: pd.DataFrame | None = None
    try:
        connection = sqlite3.connect(resolved_spool)
        _validate_spool_schema(connection)
        chunks = pd.read_sql_query(
            query,
            connection,
            params=(selected_gender,),
            chunksize=selected_chunksize,
        )
        for chunk in chunks:
            combined = (
                chunk
                if pending is None
                else pd.concat((pending, chunk), ignore_index=True)
            )
            if combined.empty:
                continue
            trailing_date = combined["tourney_date"].iloc[-1]
            ready = combined.loc[
                combined["tourney_date"] != trailing_date
            ]
            pending = combined.loc[
                combined["tourney_date"] == trailing_date
            ].copy()
            for _, date_frame in ready.groupby(
                "tourney_date",
                sort=False,
            ):
                yield _typed_output_frame(date_frame)
        if pending is not None and not pending.empty:
            yield _typed_output_frame(pending)
    except FeatureSourceValidationError:
        raise
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise FeatureSourceError(
            f"No se pudo leer causalmente el spool {resolved_spool}."
        ) from exc
    finally:
        if connection is not None:
            connection.close()

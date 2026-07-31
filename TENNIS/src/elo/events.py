"""Conversión validada de DataFrames canónicos a eventos Elo auditables.

La función pública no filtra resultados. Conserva una fila por evento para que
el motor pueda producir una decisión de auditoría por cada registro recibido.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .types import EventProvenance, Gender, MatchEvent, Surface


class EventValidationError(ValueError):
    """Indica que una fila no se puede convertir sin inventar información."""


class UnknownSurfaceError(EventValidationError):
    """Indica que una superficie poblada no pertenece al vocabulario aprobado."""


@dataclass(frozen=True)
class EventColumns:
    """Mapea el esquema del DataFrame a los campos requeridos por Elo."""

    date: str = "tourney_date"
    gender: str = "gender"
    winner_id: str = "winner_id"
    loser_id: str = "loser_id"
    surface: str = "surface"
    tour_level: str = "tour_level"
    score: str = "score"
    source_path: str = "source_file"
    source_row: str | None = None
    source_record_hash: str | None = None
    tourney_id: str | None = "tourney_id"
    match_num: str | None = "match_num"
    round: str | None = "round"


_DERIVED_COLUMNS = frozenset(
    {
        "gender",
        "tour_level",
        "source_family",
        "source_file",
        "source_path",
        "source_row",
        "source_row_number",
        "source_anomaly",
        "source_commit",
        "source_record_hash",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_missing(value: object) -> bool:
    """Detecta escalares nulos de pandas sin confundir colecciones."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _normalise_date(value: object) -> date:
    """Convierte una fecha tipada de pandas/Python sin aceptar texto ambiguo."""

    if _is_missing(value):
        raise EventValidationError("La fecha del evento no puede ser nula.")
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise EventValidationError(
                "Las fechas Elo no pueden incluir zona horaria."
            )
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            raise EventValidationError(
                "Las fechas Elo no pueden incluir zona horaria."
            )
        return value.date()
    raise EventValidationError(
        "La fecha debe estar previamente tipada como date/datetime."
    )


def _normalise_gender(value: object) -> Gender:
    """Normaliza ``M``/``F`` sin crear universos adicionales."""

    if not isinstance(value, str):
        raise EventValidationError("gender debe ser texto M o F.")
    normalised = value.strip().upper()
    if normalised == "M":
        return "M"
    if normalised == "F":
        return "F"
    raise EventValidationError(f"gender desconocido: {value!r}.")


def normalise_surface(value: object) -> Surface | None:
    """Normaliza superficie sin distinguir mayúsculas y admite nulo."""

    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise UnknownSurfaceError(
            f"La superficie poblada debe ser texto: {value!r}."
        )
    stripped = value.strip()
    if not stripped:
        return None
    surfaces: dict[str, Surface] = {
        "hard": "Hard",
        "clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
    }
    try:
        return surfaces[stripped.casefold()]
    except KeyError as exc:
        raise UnknownSurfaceError(
            f"Superficie desconocida: {value!r}."
        ) from exc


def _normalise_player_id(value: object, column: str) -> int:
    """Convierte un identificador entero y rechaza nulos, bool y decimales."""

    if _is_missing(value) or isinstance(value, bool):
        raise EventValidationError(f"{column} debe ser un entero no nulo.")
    if isinstance(value, Integral):
        player_id = int(value)
    elif isinstance(value, float) and value.is_integer():
        player_id = int(value)
    else:
        raise EventValidationError(f"{column} debe ser un entero.")
    if player_id < 0:
        raise EventValidationError(f"{column} no puede ser negativo.")
    return player_id


def _normalise_source_row(value: object, column: str) -> int:
    """Convierte un ordinal de procedencia y exige que sea uno-basado."""

    source_row = _normalise_player_id(value, column)
    if source_row < 1:
        raise EventValidationError(
            f"{column} debe ser un entero uno-basado."
        )
    return source_row


def _normalise_required_text(value: object, column: str) -> str:
    """Normaliza texto obligatorio y rechaza valores vacíos o nulos."""

    if _is_missing(value) or not isinstance(value, str):
        raise EventValidationError(f"{column} debe ser texto no nulo.")
    normalised = value.strip()
    if not normalised:
        raise EventValidationError(f"{column} no puede estar vacío.")
    return normalised


def _normalise_optional_text(value: object) -> str | None:
    """Convierte un texto opcional y conserva como nulos los vacíos."""

    if _is_missing(value):
        return None
    if not isinstance(value, str):
        return str(value)
    normalised = value.strip()
    return normalised or None


def _required_columns(
    columns: EventColumns,
    *,
    source_path: str | Path | None,
) -> set[str]:
    """Calcula las columnas obligatorias para una configuración concreta."""

    required = {
        columns.date,
        columns.gender,
        columns.winner_id,
        columns.loser_id,
        columns.surface,
        columns.tour_level,
        columns.score,
    }
    if source_path is None:
        required.add(columns.source_path)
    if columns.source_row is not None:
        required.add(columns.source_row)
    if columns.source_record_hash is not None:
        required.add(columns.source_record_hash)
    return required


def _optional_value(
    row: dict[str, Any],
    column: str | None,
) -> object:
    """Obtiene una columna contextual opcional si está disponible."""

    if column is None or column not in row:
        return None
    return row[column]


def _canonical_hash_value(value: object) -> tuple[str, object]:
    """Serializa un escalar con tipo explícito para un hash estable."""

    if _is_missing(value):
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Integral):
        return ("int", str(int(value)))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EventValidationError(
                "Una columna fuente contiene un float no finito."
            )
        return ("float", value.hex())
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            raise EventValidationError(
                "Una columna fuente contiene timestamp con zona horaria."
            )
        return ("datetime", value.isoformat())
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise EventValidationError(
                "Una columna fuente contiene datetime con zona horaria."
            )
        return ("datetime", value.isoformat())
    if isinstance(value, date):
        return ("date", value.isoformat())
    if isinstance(value, str):
        return ("str", value)
    return (
        f"object:{type(value).__module__}.{type(value).__qualname__}",
        str(value),
    )


def _default_hash_columns(
    frame: pd.DataFrame,
    columns: EventColumns,
) -> tuple[str, ...]:
    """Elige columnas fuente y excluye procedencia o campos derivados."""

    configured_provenance = {
        columns.source_path,
        columns.source_row,
        columns.source_record_hash,
    }
    excluded = _DERIVED_COLUMNS.union(
        name for name in configured_provenance if name is not None
    )
    selected = tuple(
        sorted(column for column in frame.columns if column not in excluded)
    )
    if not selected:
        raise EventValidationError(
            "No quedan columnas fuente para calcular source_record_hash."
        )
    return selected


def _calculate_source_record_hash(
    row: dict[str, Any],
    hash_columns: Sequence[str],
) -> str:
    """Calcula SHA-256 sobre todas las columnas fuente seleccionadas."""

    payload = [
        (column, _canonical_hash_value(row[column]))
        for column in hash_columns
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_source_record_hash(value: object) -> str:
    """Valida un SHA-256 proporcionado previamente por el build."""

    if not isinstance(value, str):
        raise EventValidationError(
            "source_record_hash debe ser texto SHA-256."
        )
    normalised = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalised) is None:
        raise EventValidationError(
            "source_record_hash debe tener 64 dígitos hexadecimales."
        )
    return normalised


def events_from_dataframe(
    frame: pd.DataFrame,
    *,
    source_commit: str,
    columns: EventColumns = EventColumns(),
    source_path: str | Path | None = None,
    hash_columns: Sequence[str] | None = None,
) -> tuple[MatchEvent, ...]:
    """Construye eventos con procedencia estable desde un DataFrame.

    Si ``columns.source_row`` es ``None``, ``source_row`` es el ordinal
    uno-basado de la fila dentro de su ``source_path`` en el DataFrame
    recibido. Para conservar la procedencia después de filtrar o reordenar, el
    llamador debe proporcionar una columna explícita mediante
    ``EventColumns(source_row=...)``.

    La deduplicación usa un SHA-256 de contenido. Puede proporcionarse mediante
    ``EventColumns(source_record_hash=...)``. En caso contrario se calcula con
    ``hash_columns`` o, si tampoco se facilita, con todas las columnas que no
    sean procedencia o derivadas conocidas.

    Args:
        frame: Partidos tipados con el esquema indicado por ``columns``.
        source_commit: Commit inmutable que identifica el snapshot de entrada.
        columns: Mapeo explícito de nombres de columnas.
        source_path: Ruta común opcional; si falta, se lee por fila.
        hash_columns: Columnas fuente exactas para el hash, opcionales.

    Returns:
        Eventos en el mismo orden que las filas, todavía sin filtrar.

    Raises:
        EventValidationError: Si faltan columnas o una fila no es convertible.
        UnknownSurfaceError: Si aparece una superficie poblada desconocida.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame debe ser un pandas.DataFrame.")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise EventValidationError("source_commit debe ser texto no vacío.")
    required = _required_columns(columns, source_path=source_path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EventValidationError(
            f"Faltan columnas obligatorias para construir eventos: {missing}."
        )
    if columns.source_record_hash is not None:
        selected_hash_columns: tuple[str, ...] = ()
    elif hash_columns is None:
        selected_hash_columns = _default_hash_columns(frame, columns)
    else:
        selected_hash_columns = tuple(hash_columns)
        missing_hash_columns = sorted(
            set(selected_hash_columns).difference(frame.columns)
        )
        if missing_hash_columns:
            raise EventValidationError(
                "Faltan columnas solicitadas para source_record_hash: "
                f"{missing_hash_columns}."
            )
        if not selected_hash_columns:
            raise EventValidationError("hash_columns no puede estar vacío.")

    records = frame.to_dict(orient="records")
    path_ordinals: dict[str, int] = {}
    events: list[MatchEvent] = []
    for position, row in enumerate(records):
        raw_path = source_path if source_path is not None else row[
            columns.source_path
        ]
        if isinstance(raw_path, Path):
            raw_path = raw_path.as_posix()
        path_text = _normalise_required_text(raw_path, "source_path")
        if columns.source_row is None:
            row_number = path_ordinals.get(path_text, 0) + 1
            path_ordinals[path_text] = row_number
        else:
            row_number = _normalise_source_row(
                row[columns.source_row],
                columns.source_row,
            )
        if columns.source_record_hash is None:
            source_record_hash = _calculate_source_record_hash(
                row,
                selected_hash_columns,
            )
        else:
            source_record_hash = _normalise_source_record_hash(
                row[columns.source_record_hash]
            )

        try:
            event = MatchEvent(
                date=_normalise_date(row[columns.date]),
                gender=_normalise_gender(row[columns.gender]),
                winner_id=_normalise_player_id(
                    row[columns.winner_id],
                    columns.winner_id,
                ),
                loser_id=_normalise_player_id(
                    row[columns.loser_id],
                    columns.loser_id,
                ),
                surface=normalise_surface(row[columns.surface]),
                tour_level=_normalise_required_text(
                    row[columns.tour_level],
                    columns.tour_level,
                ),
                score=_normalise_optional_text(row[columns.score]),
                provenance=EventProvenance(
                    source_commit=source_commit.strip(),
                    source_path=path_text,
                    source_row=row_number,
                ),
                source_record_hash=source_record_hash,
                tourney_id=_normalise_optional_text(
                    _optional_value(row, columns.tourney_id)
                ),
                match_num=_normalise_optional_text(
                    _optional_value(row, columns.match_num)
                ),
                round=_normalise_optional_text(
                    _optional_value(row, columns.round)
                ),
            )
        except EventValidationError as exc:
            raise type(exc)(
                f"Fila {position} de {path_text}: {exc}"
            ) from exc
        events.append(event)
    return tuple(events)

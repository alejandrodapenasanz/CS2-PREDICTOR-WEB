"""Identidades y serialización canónica para la base operativa.

``source_match_id`` prioriza siempre la identidad explícita de la fuente. Como
compatibilidad con la salida actual de la fase 8, puede derivarse de un enlace
de detalle o, en último término, de fecha, género, torneo y el par no ordenado
de slugs. Nombres, cuotas, posiciones, estado y probabilidades nunca forman
parte de esa identidad.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .types import OperationsValidationError


def optional_scalar(value: object) -> object | None:
    """Convierte un escalar ausente de Pandas/NumPy en ``None``."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def optional_text(value: object, field_name: str) -> str | None:
    """Devuelve texto recortado o ``None`` sin inventar contenido."""

    scalar = optional_scalar(value)
    if scalar is None:
        return None
    if not isinstance(scalar, str):
        raise OperationsValidationError(
            f"{field_name} debe ser texto o nulo."
        )
    text = scalar.strip()
    return text or None


def required_text(value: object, field_name: str) -> str:
    """Exige un texto no vacío y sin caracteres de control."""

    text = optional_text(value, field_name)
    if text is None:
        raise OperationsValidationError(
            f"{field_name} debe ser texto no vacío."
        )
    if any(ord(character) < 32 for character in text):
        raise OperationsValidationError(
            f"{field_name} contiene caracteres de control."
        )
    return text


def canonical_slug(value: object, field_name: str) -> str | None:
    """Valida un slug exacto sin cambiar mayúsculas, guiones ni transliterar."""

    slug = optional_text(value, field_name)
    if slug is None:
        return None
    if any(character in slug for character in ("/", "?", "#")):
        raise OperationsValidationError(
            f"{field_name} debe ser un segmento, no una ruta."
        )
    if any(character.isspace() for character in slug):
        raise OperationsValidationError(
            f"{field_name} no puede contener espacios."
        )
    return slug


def canonical_date(value: object, field_name: str) -> str:
    """Convierte una fecha civil a ISO y rechaza timestamps ambiguos."""

    scalar = optional_scalar(value)
    if scalar is None:
        raise OperationsValidationError(f"{field_name} no puede ser nulo.")
    if isinstance(scalar, datetime):
        return scalar.date().isoformat()
    if isinstance(scalar, date):
        return scalar.isoformat()
    try:
        parsed = pd.Timestamp(scalar)
    except (TypeError, ValueError) as exc:
        raise OperationsValidationError(
            f"{field_name} no es una fecha válida."
        ) from exc
    if pd.isna(parsed):
        raise OperationsValidationError(f"{field_name} no puede ser nulo.")
    return parsed.date().isoformat()


def canonical_utc_datetime(
    value: object,
    field_name: str,
    *,
    required: bool = True,
) -> str | None:
    """Normaliza un timestamp consciente de zona a ISO UTC."""

    scalar = optional_scalar(value)
    if scalar is None:
        if required:
            raise OperationsValidationError(
                f"{field_name} no puede ser nulo."
            )
        return None
    try:
        parsed = pd.Timestamp(scalar)
    except (TypeError, ValueError) as exc:
        raise OperationsValidationError(
            f"{field_name} no es un timestamp válido."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationsValidationError(
            f"{field_name} debe incluir zona horaria."
        )
    return parsed.to_pydatetime().astimezone(UTC).isoformat(
        timespec="microseconds"
    )


def canonical_json_value(value: object) -> Any:
    """Convierte escalares comunes en un valor JSON estricto y estable."""

    scalar = optional_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, (np.bool_, bool)):
        return bool(scalar)
    if isinstance(scalar, (np.integer, int)) and not isinstance(
        scalar, bool
    ):
        return int(scalar)
    if isinstance(scalar, (np.floating, float)):
        number = float(scalar)
        if not math.isfinite(number):
            raise OperationsValidationError(
                "El payload no admite números infinitos."
            )
        return number
    if isinstance(scalar, pd.Timestamp):
        if scalar.tzinfo is not None:
            return scalar.to_pydatetime().astimezone(UTC).isoformat(
                timespec="microseconds"
            )
        return scalar.isoformat()
    if isinstance(scalar, datetime):
        if scalar.tzinfo is not None and scalar.utcoffset() is not None:
            return scalar.astimezone(UTC).isoformat(timespec="microseconds")
        return scalar.isoformat()
    if isinstance(scalar, date):
        return scalar.isoformat()
    if isinstance(scalar, str):
        return scalar
    if isinstance(scalar, Mapping):
        return {
            str(key): canonical_json_value(item)
            for key, item in scalar.items()
        }
    if isinstance(scalar, (list, tuple)):
        return [canonical_json_value(item) for item in scalar]
    raise OperationsValidationError(
        f"Tipo no serializable en payload operativo: {type(scalar).__name__}."
    )


def canonical_json(payload: Mapping[str, object]) -> str:
    """Serializa un mapping con orden determinista y sin NaN."""

    normalized = {
        str(key): canonical_json_value(value)
        for key, value in payload.items()
    }
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OperationsValidationError(
            "El payload operativo no es JSON canónico."
        ) from exc


def sha256_text(text: str) -> str:
    """Calcula SHA-256 hexadecimal sobre texto UTF-8."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_sha256(payload: Mapping[str, object]) -> str:
    """Hashea un mapping mediante su JSON canónico."""

    return sha256_text(canonical_json(payload))


def _row_value(
    row: Mapping[str, object],
    *columns: str,
) -> object | None:
    """Obtiene el primer alias presente de una fila tabular."""

    for column in columns:
        if column in row:
            return row[column]
    return None


def derive_source_match_id(
    row: Mapping[str, object],
    *,
    source_system: str = "tennis_explorer",
) -> str:
    """Devuelve o deriva una identidad estable y ajena al orden A/B.

    La prioridad es:

    1. ``source_match_id`` explícito;
    2. ``match_detail_href`` de la fuente;
    3. fecha, género, clave de torneo y par ordenado de slugs.

    Una fila sin ninguna de estas identidades se rechaza para que el llamador
    pueda enviarla a revisión sin crear una clave basada en nombre o posición.
    """

    source = required_text(source_system, "source_system")
    explicit = optional_text(
        _row_value(row, "source_match_id"),
        "source_match_id",
    )
    if explicit is not None:
        if len(explicit) > 512:
            raise OperationsValidationError(
                "source_match_id supera 512 caracteres."
            )
        return explicit

    detail_href = optional_text(
        _row_value(row, "match_detail_href"),
        "match_detail_href",
    )
    if detail_href is not None:
        digest = payload_sha256(
            {
                "source_system": source,
                "match_detail_href": detail_href,
            }
        )
        return f"{source}:{digest}"

    match_date = canonical_date(
        _row_value(row, "prediction_date", "match_date"),
        "match_date",
    )
    gender = required_text(_row_value(row, "gender"), "gender")
    if gender not in {"M", "F"}:
        raise OperationsValidationError("gender debe ser 'M' o 'F'.")
    tournament_key = optional_text(
        _row_value(row, "tournament_href"),
        "tournament_href",
    )
    if tournament_key is None:
        tournament_key = required_text(
            _row_value(row, "tournament"),
            "tournament",
        )
    slug_a = canonical_slug(
        _row_value(row, "player_a_slug", "player_1_slug"),
        "player_a_slug",
    )
    slug_b = canonical_slug(
        _row_value(row, "player_b_slug", "player_2_slug"),
        "player_b_slug",
    )
    if slug_a is None or slug_b is None or slug_a == slug_b:
        raise OperationsValidationError(
            "No se puede derivar source_match_id sin dos slugs distintos."
        )
    digest = payload_sha256(
        {
            "source_system": source,
            "match_date": match_date,
            "gender": gender,
            "tournament_key": tournament_key,
            "player_slugs": sorted((slug_a, slug_b)),
        }
    )
    return f"{source}:{digest}"


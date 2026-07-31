"""Normalización exacta de nombres Tennis Explorer y Sackmann.

La transformación elimina diferencias tipográficas previsibles —mayúsculas,
acentos, puntuación y guiones—, pero no aplica distancia difusa ni completa
nombres. Una clave no unívoca debe permanecer ambigua.
"""

from __future__ import annotations

import re
from typing import Final

from unidecode import unidecode

from .types import NameKey, NameParsingError


_NON_ALPHANUMERIC_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def normalize_name_text(value: str) -> str:
    """Devuelve texto comparable sin acentos, puntuación ni espacios repetidos.

    Args:
        value: Fragmento de nombre procedente de una fuente inspeccionada.

    Returns:
        Texto ASCII en minúsculas, con puntuación y guiones convertidos en un
        único espacio.

    Raises:
        NameParsingError: Si ``value`` no es texto o queda vacío al normalizar.
    """

    if not isinstance(value, str):
        raise NameParsingError("El nombre que se normaliza debe ser texto.")
    transliterated = unidecode(value).casefold()
    normalized = _NON_ALPHANUMERIC_PATTERN.sub(" ", transliterated)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise NameParsingError("El nombre queda vacío después de normalizarlo.")
    return normalized


def parse_visible_name(visible_name: str) -> NameKey:
    """Interpreta el formato visible ``Apellido compuesto I.`` sin adivinar.

    El último componente debe ser exactamente una inicial. Todos los
    componentes anteriores forman el apellido, por lo que se conservan
    explícitamente apellidos compuestos como ``Bautista Agut`` o
    ``Van De Zandschulp``.

    Args:
        visible_name: Nombre abreviado mostrado por Tennis Explorer.

    Returns:
        Clave exacta de apellido normalizado e inicial.

    Raises:
        NameParsingError: Si el texto no termina en una única inicial.
    """

    normalized = normalize_name_text(visible_name)
    components = normalized.split()
    if len(components) < 2 or len(components[-1]) != 1:
        raise NameParsingError(
            "El nombre visible debe tener el formato 'Apellido I.'."
        )
    normalized_last_name = " ".join(components[:-1])
    return NameKey(
        normalized_last_name=normalized_last_name,
        first_initial=components[-1],
    )


def build_sackmann_name_key(name_first: str, name_last: str) -> NameKey:
    """Construye la misma clave a partir del maestro de jugadores Sackmann.

    Args:
        name_first: Nombre o nombres de pila de Sackmann.
        name_last: Apellido o apellidos de Sackmann.

    Returns:
        Clave exacta compatible con :func:`parse_visible_name`.

    Raises:
        NameParsingError: Si alguno de los dos componentes no es utilizable.
    """

    normalized_first = normalize_name_text(name_first)
    normalized_last = normalize_name_text(name_last)
    return NameKey(
        normalized_last_name=normalized_last,
        first_initial=normalized_first[0],
    )

"""Tipos públicos y errores controlados del mapeo de jugadores.

Los identificadores de Sackmann solo son únicos dentro de su universo de
género. Por ello, todos los candidatos conservan el género junto al
``player_id`` y el índice nunca mezcla hombres y mujeres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, TypeAlias


Gender = Literal["M", "F"]


class PlayerMappingError(RuntimeError):
    """Error base controlado de la resolución de identidades."""


class PlayerMappingValidationError(PlayerMappingError):
    """Indica que un argumento no cumple el contrato público."""


class PlayerMappingSchemaError(PlayerMappingError):
    """Indica que un DataFrame o CSV no tiene el esquema Sackmann esperado."""


class PlayerMappingSourceError(PlayerMappingError):
    """Indica que faltan archivos o años requeridos de la fuente local."""


class NameParsingError(PlayerMappingValidationError):
    """Indica que un nombre no puede convertirse honestamente en una clave."""


@dataclass(frozen=True, slots=True)
class NameKey:
    """Clave exacta formada por apellido normalizado e inicial del nombre."""

    normalized_last_name: str
    first_initial: str

    def __post_init__(self) -> None:
        """Valida que la clave ya esté normalizada y no sea ambigua."""

        if (
            not isinstance(self.normalized_last_name, str)
            or not self.normalized_last_name
            or self.normalized_last_name != self.normalized_last_name.strip()
        ):
            raise PlayerMappingValidationError(
                "normalized_last_name debe ser texto normalizado no vacío."
            )
        if (
            not isinstance(self.first_initial, str)
            or len(self.first_initial) != 1
            or not self.first_initial.isascii()
            or not self.first_initial.isalnum()
            or self.first_initial != self.first_initial.casefold()
        ):
            raise PlayerMappingValidationError(
                "first_initial debe ser un único carácter ASCII normalizado."
            )


@dataclass(frozen=True, slots=True)
class ActiveCandidate:
    """Jugador Sackmann activo que puede corresponder a un nombre visible."""

    gender: Gender
    player_id: int
    player_name: str
    name_first: str
    name_last: str
    ioc: str | None
    last_match_date: date
    name_key: NameKey

    def __post_init__(self) -> None:
        """Impide construir candidatos incompletos o de género inválido."""

        if self.gender not in {"M", "F"}:
            raise PlayerMappingValidationError(
                "gender debe ser exactamente 'M' o 'F'."
            )
        if isinstance(self.player_id, bool) or not isinstance(self.player_id, int):
            raise PlayerMappingValidationError(
                "player_id debe ser un entero no booleano."
            )
        for field_name in ("player_name", "name_first", "name_last"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise PlayerMappingValidationError(
                    f"{field_name} debe ser texto no vacío."
                )
        if self.ioc is not None and (
            not isinstance(self.ioc, str) or not self.ioc.strip()
        ):
            raise PlayerMappingValidationError(
                "ioc debe ser texto no vacío o None."
            )
        if not isinstance(self.last_match_date, date) or isinstance(
            self.last_match_date,
            datetime,
        ):
            raise PlayerMappingValidationError(
                "last_match_date debe ser datetime.date."
            )
        if not isinstance(self.name_key, NameKey):
            raise PlayerMappingValidationError("name_key debe ser NameKey.")


CandidateIndex: TypeAlias = dict[NameKey, tuple[ActiveCandidate, ...]]

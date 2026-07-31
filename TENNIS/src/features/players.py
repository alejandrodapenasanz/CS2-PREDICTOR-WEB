"""Índice biográfico para calcular edad estrictamente en la fecha del partido.

La fecha de nacimiento es un atributo fijo, pero la edad se deriva siempre
contra ``as_of_date``. Una fecha ausente o incompatible con el corte no se
rellena: se devuelve un snapshot explícitamente incompleto.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from typing import Mapping, cast

import pandas as pd

from ..config import RAW_DATA_DIR
from ..data_loaders import load_players
from ..elo.types import Gender
from .parameters import DEFAULT_FEATURE_PARAMETERS


class PlayerAgeError(ValueError):
    """Indica que el maestro biográfico no cumple el contrato esperado."""


@dataclass(frozen=True, slots=True)
class PlayerAgeSnapshot:
    """Edad de un jugador en un corte temporal concreto."""

    gender: Gender
    player_id: int
    as_of_date: date
    birth_date: date | None
    age_years: float | None
    distance_to_reference: float | None
    is_missing: bool
    invalid_for_date: bool


def _validated_gender(value: object) -> Gender:
    """Valida el universo de género sin inferirlo desde un identificador."""

    if value not in {"M", "F"}:
        raise PlayerAgeError("gender debe ser exactamente 'M' o 'F'.")
    return cast(Gender, value)


def _validated_player_id(value: object) -> int:
    """Valida un player_id entero no negativo y rechaza booleanos."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlayerAgeError("player_id debe ser un entero no negativo.")
    return value


def _validated_date(value: object, field_name: str) -> date:
    """Exige ``datetime.date`` estricto, sin componente horaria."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise PlayerAgeError(f"{field_name} debe ser datetime.date estricto.")
    return value


def _optional_birth_date(value: object) -> date | None:
    """Convierte fechas pandas/Python y conserva ausencias como ``None``."""

    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise PlayerAgeError(
        "dob debe contener fechas tipadas o valores nulos; "
        f"se recibió {value!r}."
    )


class PlayerAgeIndex:
    """Mantiene fechas de nacimiento aisladas por género y player_id."""

    def __init__(
        self,
        birth_dates: Mapping[tuple[Gender, int], date | None],
        *,
        reference_years: float = (
            DEFAULT_FEATURE_PARAMETERS.age_reference_years
        ),
        days_per_year: float = DEFAULT_FEATURE_PARAMETERS.days_per_year,
    ) -> None:
        """Construye un índice validado desde un mapping inmutable por copia."""

        for field_name, value in (
            ("reference_years", reference_years),
            ("days_per_year", days_per_year),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise PlayerAgeError(
                    f"{field_name} debe ser numérico, finito y positivo."
                )
        copied: dict[tuple[Gender, int], date | None] = {}
        for (gender, player_id), birth_date in birth_dates.items():
            checked_key = (
                _validated_gender(gender),
                _validated_player_id(player_id),
            )
            checked_birth = (
                None
                if birth_date is None
                else _validated_date(birth_date, "birth_date")
            )
            copied[checked_key] = checked_birth
        self._birth_dates = copied
        self.reference_years = float(reference_years)
        self.days_per_year = float(days_per_year)

    @classmethod
    def from_dataframe(
        cls,
        players: pd.DataFrame,
        *,
        reference_years: float = (
            DEFAULT_FEATURE_PARAMETERS.age_reference_years
        ),
        days_per_year: float = DEFAULT_FEATURE_PARAMETERS.days_per_year,
    ) -> "PlayerAgeIndex":
        """Construye el índice desde las columnas reales ``gender/id/dob``."""

        required = {"gender", "player_id", "dob"}
        missing = sorted(required.difference(players.columns))
        if missing:
            raise PlayerAgeError(
                f"Faltan columnas biográficas obligatorias: {missing}."
            )
        birth_dates: dict[tuple[Gender, int], date | None] = {}
        for row in players.loc[
            :,
            ["gender", "player_id", "dob"],
        ].itertuples(index=False, name=None):
            gender = _validated_gender(row[0])
            raw_player_id = row[1]
            if pd.isna(raw_player_id):
                raise PlayerAgeError("player_id contiene un valor nulo.")
            player_id = _validated_player_id(int(raw_player_id))
            birth_date = _optional_birth_date(row[2])
            key = (gender, player_id)
            if key in birth_dates and birth_dates[key] != birth_date:
                raise PlayerAgeError(
                    "El maestro contiene fechas de nacimiento conflictivas "
                    f"para {key}: {birth_dates[key]!r} y {birth_date!r}."
                )
            birth_dates[key] = birth_date
        return cls(
            birth_dates,
            reference_years=reference_years,
            days_per_year=days_per_year,
        )

    @classmethod
    def from_raw(
        cls,
        *,
        raw_dir: Path = RAW_DATA_DIR,
        reference_years: float = (
            DEFAULT_FEATURE_PARAMETERS.age_reference_years
        ),
        days_per_year: float = DEFAULT_FEATURE_PARAMETERS.days_per_year,
    ) -> "PlayerAgeIndex":
        """Carga los maestros Sackmann ya validados y construye el índice."""

        return cls.from_dataframe(
            load_players(raw_dir=Path(raw_dir)),
            reference_years=reference_years,
            days_per_year=days_per_year,
        )

    def get(
        self,
        gender: Gender,
        player_id: int,
        *,
        as_of_date: date,
    ) -> PlayerAgeSnapshot:
        """Calcula edad decimal y distancia absoluta al valor de referencia."""

        checked_gender = _validated_gender(gender)
        checked_player = _validated_player_id(player_id)
        cutoff = _validated_date(as_of_date, "as_of_date")
        birth_date = self._birth_dates.get(
            (checked_gender, checked_player)
        )
        if birth_date is None:
            return PlayerAgeSnapshot(
                gender=checked_gender,
                player_id=checked_player,
                as_of_date=cutoff,
                birth_date=None,
                age_years=None,
                distance_to_reference=None,
                is_missing=True,
                invalid_for_date=False,
            )
        if birth_date >= cutoff:
            return PlayerAgeSnapshot(
                gender=checked_gender,
                player_id=checked_player,
                as_of_date=cutoff,
                birth_date=birth_date,
                age_years=None,
                distance_to_reference=None,
                is_missing=True,
                invalid_for_date=True,
            )
        age_years = (cutoff - birth_date).days / self.days_per_year
        return PlayerAgeSnapshot(
            gender=checked_gender,
            player_id=checked_player,
            as_of_date=cutoff,
            birth_date=birth_date,
            age_years=age_years,
            distance_to_reference=abs(
                age_years - self.reference_years
            ),
            is_missing=False,
            invalid_for_date=False,
        )

    def get_pair(
        self,
        gender: Gender,
        player_a_id: int,
        player_b_id: int,
        *,
        as_of_date: date,
    ) -> tuple[PlayerAgeSnapshot, PlayerAgeSnapshot]:
        """Devuelve los snapshots de A y B bajo el mismo corte temporal."""

        return (
            self.get(
                gender,
                player_a_id,
                as_of_date=as_of_date,
            ),
            self.get(
                gender,
                player_b_id,
                as_of_date=as_of_date,
            ),
        )

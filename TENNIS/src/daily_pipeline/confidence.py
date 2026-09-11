"""Política explícita y configurable de confianza para predicciones diarias.

La confianza describe calidad y frescura de inputs; nunca aumenta porque una
probabilidad sea extrema. Los umbrales son operativos y conservadores: al
menos 20 partidos generales, 10 en la superficie, y un lag máximo de 14 días
desde el último resultado ya disponible y desde el ranking. No alteran la
probabilidad, solo su etiqueta auditable.

La frescura del histórico se mide contra ``training_available_max_date``, no
contra la fecha deportiva del último partido. Esto es esencial porque el
contrato causal retrasa 21 días cada resultado: comparar directamente con la
fecha deportiva marcaría como obsoleta incluso una fuente perfectamente al
día.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, Mapping


ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"]

MIN_GENERAL_MATCHES: Final[int] = 20
MIN_SURFACE_MATCHES: Final[int] = 10
MAX_HISTORY_AVAILABILITY_LAG_DAYS: Final[int] = 14
# Alias público conservado para consumidores anteriores. Su semántica es el
# lag desde el corte disponible, nunca la edad de la fecha deportiva.
MAX_HISTORY_AGE_DAYS: Final[int] = MAX_HISTORY_AVAILABILITY_LAG_DAYS
MAX_RANKING_AGE_DAYS: Final[int] = 14


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """Etiqueta y lista ordenada de razones que la justifican."""

    level: ConfidenceLevel
    flags: tuple[str, ...]


def unavailable_confidence(*flags: str) -> ConfidenceAssessment:
    """Construye una evaluación no disponible con razones no vacías."""

    cleaned = tuple(dict.fromkeys(flag for flag in flags if flag))
    if not cleaned:
        raise ValueError("UNAVAILABLE requiere al menos un flag.")
    return ConfidenceAssessment(level="UNAVAILABLE", flags=cleaned)


def _optional_integer(values: Mapping[str, object], column: str) -> int | None:
    """Obtiene un entero opcional ya producido por el vector causal."""

    value = values.get(column)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{column} debe ser un entero o None.")
    return value


def assess_vector_confidence(
    values: Mapping[str, object],
    *,
    match_date: date,
    history_available_max_date: date,
    ranking_max_date: date,
    max_history_availability_lag_days: int = (
        MAX_HISTORY_AVAILABILITY_LAG_DAYS
    ),
) -> ConfidenceAssessment:
    """Clasifica cobertura y frescura de un vector que sí puede predecirse.

    ``history_available_max_date`` es el corte causal después de aplicar la
    política de disponibilidad a la última fecha deportiva. Debe preceder
    estrictamente a ``match_date``. El umbral se inyecta para que una política
    operativa pueda endurecerlo sin cambiar la semántica temporal.
    """

    if (
        isinstance(max_history_availability_lag_days, bool)
        or not isinstance(max_history_availability_lag_days, int)
        or max_history_availability_lag_days < 1
    ):
        raise ValueError(
            "max_history_availability_lag_days debe ser un entero positivo."
        )

    low_flags: list[str] = []
    warning_flags: list[str] = []
    history_availability_lag = (
        match_date - history_available_max_date
    ).days
    ranking_source_age = (match_date - ranking_max_date).days
    if history_availability_lag < 1:
        raise ValueError(
            "history_available_max_date debe ser estrictamente anterior a D."
        )
    if ranking_source_age < 1:
        raise ValueError("ranking_max_date debe ser estrictamente anterior a D.")
    if history_availability_lag > max_history_availability_lag_days:
        low_flags.append(
            f"history_available_stale_{history_availability_lag}d"
        )
    if ranking_source_age > MAX_RANKING_AGE_DAYS:
        low_flags.append(f"ranking_source_stale_{ranking_source_age}d")

    if values.get("surface") is None:
        low_flags.append("surface_missing")
    if values.get("best_of") is None:
        warning_flags.append("best_of_missing")
    if values.get("round") is None:
        warning_flags.append("round_missing")
    warning_flags.append("daily_raw_level_not_sackmann_code")

    for side in ("a", "b"):
        general_matches = _optional_integer(
            values,
            f"elo_general_matches_{side}",
        )
        surface_matches = _optional_integer(
            values,
            f"elo_surface_matches_{side}",
        )
        if general_matches is None or general_matches < MIN_GENERAL_MATCHES:
            low_flags.append(f"limited_general_history_{side}")
        if values.get("surface") is not None and (
            surface_matches is None
            or surface_matches < MIN_SURFACE_MATCHES
        ):
            warning_flags.append(f"limited_surface_history_{side}")
        if bool(values.get(f"ranking_missing_{side}")):
            low_flags.append(f"ranking_missing_{side}")
        ranking_age = _optional_integer(
            values,
            f"ranking_age_days_{side}",
        )
        if (
            ranking_age is not None
            and ranking_age > MAX_RANKING_AGE_DAYS
        ):
            low_flags.append(f"ranking_stale_{side}_{ranking_age}d")
        if bool(values.get(f"age_missing_{side}")):
            warning_flags.append(f"age_missing_{side}")

    if values.get("market_probability_a") is None:
        warning_flags.append("market_missing")
    flags = tuple(dict.fromkeys((*low_flags, *warning_flags)))
    if low_flags:
        level: ConfidenceLevel = "LOW"
    elif warning_flags:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return ConfidenceAssessment(level=level, flags=flags)

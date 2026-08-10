"""Política causal para fechas históricas aproximadas de Jeff Sackmann.

Los CSV de partidos usan ``tourney_date`` como fecha aproximada de inicio del
torneo, no como día real de cada partido. Por ello, un resultado no se admite
en Elo, forma, H2H, descanso o entrenamiento inmediatamente después de esa
fecha fuente. Este módulo centraliza un embargo conservador, versionado y
configurable para que todas las capas apliquen exactamente la misma regla.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Mapping


SOURCE_DATE_POLICY_VERSION: Final[str] = (
    "sackmann-tourney-start-embargo-v1"
)
DEFAULT_RESULT_EMBARGO_DAYS: Final[int] = 21


class SourceDatePolicyError(ValueError):
    """Indica una fecha o configuración incompatible con el embargo causal."""


def _strict_date(value: object, field_name: str) -> date:
    """Exige una fecha civil sin hora ni conversión implícita."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise SourceDatePolicyError(
            f"{field_name} debe ser datetime.date estricto."
        )
    return value


@dataclass(frozen=True, slots=True)
class SourceDatePolicy:
    """Define cuándo puede usarse un resultado fechado al inicio del torneo.

    ``availability_date`` es el final civil del embargo. La observación solo
    se admite cuando ``availability_date < as_of_date``; la igualdad permanece
    excluida. Con el valor por defecto de 21 días, un torneo fechado en ``D``
    puede afectar por primera vez a una predicción de ``D + 22``.
    """

    result_embargo_days: int = DEFAULT_RESULT_EMBARGO_DAYS

    def __post_init__(self) -> None:
        """Rechaza retrasos nulos, negativos, booleanos o no enteros."""

        value = self.result_embargo_days
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SourceDatePolicyError(
                "result_embargo_days debe ser un entero positivo."
            )

    def availability_date(self, tourney_date: date) -> date:
        """Devuelve la fecha fuente más los días de embargo configurados."""

        source_date = _strict_date(tourney_date, "tourney_date")
        try:
            return source_date + timedelta(days=self.result_embargo_days)
        except OverflowError as exc:
            raise SourceDatePolicyError(
                "tourney_date no admite sumar el embargo configurado."
            ) from exc

    def is_available(self, tourney_date: date, as_of_date: date) -> bool:
        """Indica si el resultado está disponible estrictamente antes del corte."""

        cutoff = _strict_date(as_of_date, "as_of_date")
        return self.availability_date(tourney_date) < cutoff

    def as_dict(self) -> dict[str, object]:
        """Serializa la política completa para fingerprints y manifiestos."""

        return {
            "version": SOURCE_DATE_POLICY_VERSION,
            "source_field": "tourney_date",
            "source_semantics": "approximate_tournament_start_date",
            "result_embargo_days": self.result_embargo_days,
            "availability_rule": "availability_date < as_of_date",
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SourceDatePolicy":
        """Reconstruye una política y valida toda su semántica publicada."""

        if not isinstance(payload, Mapping):
            raise SourceDatePolicyError(
                "source_date_policy debe ser un mapping."
            )
        expected_static = {
            "version": SOURCE_DATE_POLICY_VERSION,
            "source_field": "tourney_date",
            "source_semantics": "approximate_tournament_start_date",
            "availability_rule": "availability_date < as_of_date",
        }
        for field_name, expected in expected_static.items():
            if payload.get(field_name) != expected:
                raise SourceDatePolicyError(
                    f"source_date_policy.{field_name} no coincide con el "
                    "contrato temporal soportado."
                )
        days = payload.get("result_embargo_days")
        if isinstance(days, bool) or not isinstance(days, int):
            raise SourceDatePolicyError(
                "source_date_policy.result_embargo_days debe ser entero."
            )
        return cls(result_embargo_days=days)


DEFAULT_SOURCE_DATE_POLICY: Final[SourceDatePolicy] = SourceDatePolicy()


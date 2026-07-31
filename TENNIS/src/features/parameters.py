"""Parámetros versionados de la construcción causal de features.

Los valores por defecto fueron aprobados para la fase 6 y se mantienen en una
estructura inmutable para que el manifiesto de cada dataset pueda registrar la
configuración exacta usada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Mapping


FEATURE_SCHEMA_VERSION: Final[str] = "tennis-features-v1"


@dataclass(frozen=True, slots=True)
class FeatureParameters:
    """Configura ventanas, orientación y transformación de edad."""

    recent_matches: int = 10
    recent_months: int = 3
    age_reference_years: float = 30.0
    days_per_year: float = 365.2425
    orientation_seed: int = 42

    def __post_init__(self) -> None:
        """Valida los parámetros sin corregir valores silenciosamente."""

        for field_name, value in (
            ("recent_matches", self.recent_matches),
            ("recent_months", self.recent_months),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{field_name} debe ser un entero positivo.")
        for field_name, value in (
            ("age_reference_years", self.age_reference_years),
            ("days_per_year", self.days_per_year),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field_name} debe ser numérico y positivo.")
        if (
            isinstance(self.orientation_seed, bool)
            or not isinstance(self.orientation_seed, int)
        ):
            raise ValueError("orientation_seed debe ser un entero.")

    def as_dict(self) -> Mapping[str, int | float]:
        """Devuelve una copia serializable para fingerprints y manifiestos."""

        return asdict(self)


DEFAULT_FEATURE_PARAMETERS: Final[FeatureParameters] = FeatureParameters()

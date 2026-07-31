"""Parámetros matemáticos configurables del sistema Elo de tenis.

Los valores por defecto reproducen el núcleo publicado por FiveThirtyEight:
rating inicial 1500, escala logística 400 y
``K(n) = 250 / (n + 5) ** 0.4``. La combinación por superficie usa el reparto
50/50 documentado por Tennis Abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final


ALGORITHM_VERSION: Final[str] = "sackmann-elo-v3"


@dataclass(frozen=True)
class EloParameters:
    """Agrupa y valida todos los parámetros numéricos del motor Elo."""

    initial_rating: float = 1500.0
    scale: float = 400.0
    k_numerator: float = 250.0
    k_offset: float = 5.0
    k_exponent: float = 0.4
    surface_weight: float = 0.5

    def __post_init__(self) -> None:
        """Rechaza parámetros no finitos o incompatibles con las fórmulas."""

        numeric_values = {
            "initial_rating": self.initial_rating,
            "scale": self.scale,
            "k_numerator": self.k_numerator,
            "k_offset": self.k_offset,
            "k_exponent": self.k_exponent,
            "surface_weight": self.surface_weight,
        }
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} debe ser numérico.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} debe ser finito.")
        if self.scale <= 0:
            raise ValueError("scale debe ser mayor que cero.")
        if self.k_numerator <= 0:
            raise ValueError("k_numerator debe ser mayor que cero.")
        if self.k_offset <= 0:
            raise ValueError("k_offset debe ser mayor que cero.")
        if self.k_exponent < 0:
            raise ValueError("k_exponent no puede ser negativo.")
        if not 0.0 <= self.surface_weight <= 1.0:
            raise ValueError("surface_weight debe estar entre cero y uno.")

    def expected_score(
        self,
        rating: float,
        opponent_rating: float,
    ) -> float:
        """Calcula la expectativa logística de un jugador antes del partido."""

        difference = (float(rating) - float(opponent_rating)) / self.scale
        if difference >= 0:
            inverse_power = 10.0 ** (-difference)
            return 1.0 / (1.0 + inverse_power)
        power = 10.0 ** difference
        return power / (1.0 + power)

    def k_factor(self, prior_matches: int) -> float:
        """Devuelve el K dinámico usando solo el contador previo al partido."""

        if (
            isinstance(prior_matches, bool)
            or not isinstance(prior_matches, int)
        ):
            raise TypeError("prior_matches debe ser un entero.")
        if prior_matches < 0:
            raise ValueError("prior_matches no puede ser negativo.")
        return self.k_numerator / (
            (prior_matches + self.k_offset) ** self.k_exponent
        )

    def combine(
        self,
        general_rating: float,
        surface_rating: float | None,
    ) -> float:
        """Combina Elo general y de superficie, o conserva solo el general."""

        if surface_rating is None:
            return float(general_rating)
        return (
            (1.0 - self.surface_weight) * float(general_rating)
            + self.surface_weight * float(surface_rating)
        )

    def as_dict(self) -> dict[str, float]:
        """Devuelve una representación serializable y estable."""

        return {
            "initial_rating": float(self.initial_rating),
            "scale": float(self.scale),
            "k_numerator": float(self.k_numerator),
            "k_offset": float(self.k_offset),
            "k_exponent": float(self.k_exponent),
            "surface_weight": float(self.surface_weight),
        }


DEFAULT_ELO_PARAMETERS: Final[EloParameters] = EloParameters()

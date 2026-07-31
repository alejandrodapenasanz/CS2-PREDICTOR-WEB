"""Utilidades causales para transformar cuotas decimales de un mercado a dos vías.

El módulo conserva tanto las probabilidades implícitas brutas como las
probabilidades normalizadas sin margen. La marca temporal del mercado se valida
por separado porque una cuota solo es admisible si se observó estrictamente
antes del instante de predicción.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Number
from typing import Any

import pandas as pd


class InvalidOddsError(ValueError):
    """Indica que una cuota presente no cumple el contrato decimal."""


class MarketTimestampError(ValueError):
    """Indica que una observación de mercado no es causal o no tiene zona."""


@dataclass(frozen=True)
class MarketProbabilities:
    """Resultado completo de quitar el margen de dos cuotas decimales.

    Todos los campos son ``None`` cuando falta al menos una de las dos cuotas.
    En un resultado disponible, ``overround`` es la suma de las probabilidades
    implícitas brutas y ``margin`` es ``overround - 1``.
    """

    raw_implied_probability_a: float | None
    raw_implied_probability_b: float | None
    market_probability_a_devig: float | None
    market_probability_b_devig: float | None
    overround: float | None
    margin: float | None

    @property
    def is_available(self) -> bool:
        """Devuelve si el mercado contiene las dos cuotas necesarias."""

        return self.raw_implied_probability_a is not None


def _is_missing(value: Any) -> bool:
    """Reconoce ausencias escalares habituales sin aceptar colecciones."""

    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing
    if hasattr(missing, "item") and getattr(missing, "ndim", 1) == 0:
        return bool(missing.item())
    return False


def _validated_decimal_odds(value: Any, *, field_name: str) -> float:
    """Convierte una cuota numérica presente y exige que sea finita y mayor que uno."""

    if isinstance(value, bool) or not isinstance(value, Number):
        raise InvalidOddsError(
            f"{field_name} debe ser una cuota decimal numérica; "
            f"se recibió {value!r}."
        )
    try:
        odds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidOddsError(
            f"{field_name} debe poder representarse como número real finito."
        ) from exc
    if not math.isfinite(odds):
        raise InvalidOddsError(f"{field_name} debe ser finita.")
    if odds <= 1.0:
        raise InvalidOddsError(
            f"{field_name} debe ser una cuota decimal estrictamente mayor que 1."
        )
    return odds


def calculate_two_way_market_probabilities(
    odds_a: Any,
    odds_b: Any,
) -> MarketProbabilities:
    """Calcula probabilidades brutas y sin margen para un mercado de dos lados.

    Se usa ``q_i = 1 / odds_i`` y ``p_i = q_i / (q_a + q_b)``. Si falta
    cualquiera de las cuotas, ningún valor parcial se conserva: todos los
    campos del resultado son ``None``.
    """

    missing_a = _is_missing(odds_a)
    missing_b = _is_missing(odds_b)
    decimal_a = (
        None
        if missing_a
        else _validated_decimal_odds(odds_a, field_name="odds_a")
    )
    decimal_b = (
        None
        if missing_b
        else _validated_decimal_odds(odds_b, field_name="odds_b")
    )
    if missing_a or missing_b:
        return MarketProbabilities(
            raw_implied_probability_a=None,
            raw_implied_probability_b=None,
            market_probability_a_devig=None,
            market_probability_b_devig=None,
            overround=None,
            margin=None,
        )

    assert decimal_a is not None
    assert decimal_b is not None
    raw_a = 1.0 / decimal_a
    raw_b = 1.0 / decimal_b
    overround = raw_a + raw_b

    return MarketProbabilities(
        raw_implied_probability_a=raw_a,
        raw_implied_probability_b=raw_b,
        market_probability_a_devig=raw_a / overround,
        market_probability_b_devig=raw_b / overround,
        overround=overround,
        margin=overround - 1.0,
    )


def _require_timezone_aware(value: datetime, *, field_name: str) -> None:
    """Exige un ``datetime`` escalar cuyo desplazamiento UTC esté definido."""

    if not isinstance(value, datetime):
        raise MarketTimestampError(
            f"{field_name} debe ser datetime con zona horaria."
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise MarketTimestampError(
            f"{field_name} debe incluir una zona horaria explícita."
        )


def validate_market_timestamp(
    retrieved_at_utc: datetime,
    prediction_as_of_utc: datetime,
) -> None:
    """Valida que las cuotas se recuperaron antes del instante de predicción.

    Los dos argumentos deben ser conscientes de zona horaria. Se permite que
    representen UTC mediante zonas equivalentes, pero se rechazan igualdad y
    observaciones futuras para mantener la desigualdad causal estricta
    ``retrieved_at_utc < prediction_as_of_utc``.
    """

    _require_timezone_aware(
        retrieved_at_utc,
        field_name="retrieved_at_utc",
    )
    _require_timezone_aware(
        prediction_as_of_utc,
        field_name="prediction_as_of_utc",
    )
    if retrieved_at_utc >= prediction_as_of_utc:
        raise MarketTimestampError(
            "retrieved_at_utc debe ser estrictamente anterior a "
            "prediction_as_of_utc."
        )

"""Parámetros versionados y configurables de los estimadores de la fase 7.

Los valores se concentran en dataclasses inmutables para que el proceso de
reentreno pueda serializarlos junto al modelo. Ningún parámetro se adapta
silenciosamente al conjunto de evaluación.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Final, Mapping


MODEL_ARTIFACT_VERSION: Final[str] = "tennis-model-v3"
DEFAULT_RANDOM_SEED: Final[int] = 42


def _require_positive_finite(name: str, value: object) -> None:
    """Exige un número finito estrictamente positivo."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} debe ser numérico, finito y positivo.")


def _require_non_negative_finite(name: str, value: object) -> None:
    """Exige un número finito mayor o igual que cero."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{name} debe ser numérico, finito y no negativo."
        )


def _require_positive_integer(name: str, value: object) -> None:
    """Exige un entero estrictamente positivo y rechaza booleanos."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} debe ser un entero positivo.")


@dataclass(frozen=True, slots=True)
class LogisticParameters:
    """Configura la regresión logística probabilística de referencia."""

    c: float = 1.0
    max_iter: int = 2_000
    solver: str = "lbfgs"
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Valida la configuración sin corregir valores inválidos."""

        _require_positive_finite("c", self.c)
        _require_positive_integer("max_iter", self.max_iter)
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError("solver debe ser un texto no vacío.")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValueError("random_seed debe ser un entero.")

    def as_dict(self) -> Mapping[str, object]:
        """Devuelve los parámetros en una copia serializable."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LightGBMParameters:
    """Configura el LightGBM principal con regularización conservadora."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.90
    subsample_freq: int = 1
    colsample_bytree: float = 0.90
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_seed: int = DEFAULT_RANDOM_SEED
    n_jobs: int = -1

    def __post_init__(self) -> None:
        """Valida límites que LightGBM no debe ajustar implícitamente."""

        _require_positive_integer("n_estimators", self.n_estimators)
        _require_positive_finite("learning_rate", self.learning_rate)
        _require_positive_integer("num_leaves", self.num_leaves)
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth == 0
            or self.max_depth < -1
        ):
            raise ValueError("max_depth debe ser -1 o un entero positivo.")
        _require_positive_integer(
            "min_child_samples", self.min_child_samples
        )
        _require_positive_integer("subsample_freq", self.subsample_freq)
        for name, value in (
            ("subsample", self.subsample),
            ("colsample_bytree", self.colsample_bytree),
        ):
            _require_positive_finite(name, value)
            if float(value) > 1.0:
                raise ValueError(f"{name} no puede superar 1.")
        _require_non_negative_finite("reg_alpha", self.reg_alpha)
        _require_non_negative_finite("reg_lambda", self.reg_lambda)
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValueError("random_seed debe ser un entero.")
        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValueError("n_jobs debe ser -1 o un entero positivo.")

    def as_dict(self) -> Mapping[str, object]:
        """Devuelve los parámetros en una copia serializable."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalEvaluationParameters:
    """Configura el backtest temporal, la calibración y sus auditorías."""

    first_test_season: int = 2016
    last_test_season: int = 2025
    reliability_bins: int = 10
    suspicious_accuracy_threshold: float = 0.85
    small_segment_threshold: int = 200

    def __post_init__(self) -> None:
        """Valida temporadas y umbrales sin adaptar valores al test."""

        for name, value in (
            ("first_test_season", self.first_test_season),
            ("last_test_season", self.last_test_season),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1900
            ):
                raise ValueError(f"{name} debe ser un año entero válido.")
        if self.last_test_season < self.first_test_season:
            raise ValueError(
                "last_test_season no puede preceder first_test_season."
            )
        _require_positive_integer("reliability_bins", self.reliability_bins)
        threshold = self.suspicious_accuracy_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                "suspicious_accuracy_threshold debe estar en [0, 1]."
            )
        _require_positive_integer(
            "small_segment_threshold", self.small_segment_threshold
        )

    def as_dict(self) -> Mapping[str, object]:
        """Devuelve el protocolo temporal en una copia serializable."""

        return asdict(self)


DEFAULT_LOGISTIC_PARAMETERS: Final[LogisticParameters] = (
    LogisticParameters()
)
DEFAULT_LIGHTGBM_PARAMETERS: Final[LightGBMParameters] = (
    LightGBMParameters()
)
DEFAULT_TEMPORAL_EVALUATION_PARAMETERS: Final[
    TemporalEvaluationParameters
] = TemporalEvaluationParameters()

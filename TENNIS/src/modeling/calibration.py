"""Calibración Platt de probabilidades usando exclusivamente datos pasados.

El calibrador transforma cada probabilidad cruda ``p`` en ``logit(p)`` y
ajusta una regresión logística unidimensional. El código no recibe el bloque
de test durante ``fit``: el llamador debe pasar únicamente las predicciones y
etiquetas del bloque temporal de calibración.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibrationError(ValueError):
    """Indica entradas inválidas o un calibrador aún no ajustado."""


@dataclass(frozen=True, slots=True)
class PlattParameters:
    """Parámetros explícitos y serializables de la calibración Platt."""

    clip_epsilon: float = 1e-6
    regularization_c: float = 1_000_000.0
    max_iter: int = 1_000
    random_state: int = 42

    def validate(self) -> None:
        """Comprueba que los parámetros definen un ajuste válido."""

        if not 0.0 < self.clip_epsilon < 0.5:
            raise CalibrationError("clip_epsilon debe estar entre 0 y 0.5.")
        if self.regularization_c <= 0.0:
            raise CalibrationError("regularization_c debe ser positivo.")
        if self.max_iter <= 0:
            raise CalibrationError("max_iter debe ser positivo.")


def _as_probability_vector(
    values: object,
    *,
    name: str,
) -> np.ndarray:
    """Convierte probabilidades a un vector finito dentro de ``[0, 1]``."""

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} no es numérico.") from exc
    if vector.ndim != 1:
        raise CalibrationError(f"{name} debe ser un vector unidimensional.")
    if vector.size == 0:
        raise CalibrationError(f"{name} no puede estar vacío.")
    if not np.isfinite(vector).all():
        raise CalibrationError(f"{name} debe contener valores finitos.")
    if ((vector < 0.0) | (vector > 1.0)).any():
        raise CalibrationError(f"{name} debe estar dentro de [0, 1].")
    return vector


def _as_binary_targets(values: object, expected_size: int) -> np.ndarray:
    """Convierte etiquetas a un vector binario del tamaño esperado."""

    try:
        vector = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("Las etiquetas no son válidas.") from exc
    if vector.ndim != 1 or vector.size != expected_size:
        raise CalibrationError(
            "Las etiquetas deben ser un vector del mismo tamaño."
        )
    try:
        numeric = vector.astype(np.int8)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("Las etiquetas deben ser 0 o 1.") from exc
    if not np.array_equal(vector, numeric) or not np.isin(
        numeric, (0, 1)
    ).all():
        raise CalibrationError("Las etiquetas deben ser exactamente 0 o 1.")
    return numeric


class PlattCalibrator:
    """Calibrador Platt serializable basado en regresión logística."""

    def __init__(
        self,
        parameters: PlattParameters | None = None,
    ) -> None:
        """Inicializa un calibrador no ajustado con parámetros explícitos."""

        self.parameters = parameters or PlattParameters()
        self.parameters.validate()
        self.estimator_: LogisticRegression | None = None
        self.n_calibration_samples_: int | None = None
        self.positive_rate_: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Indica si el calibrador ya fue ajustado."""

        return self.estimator_ is not None

    def _logit_feature(self, probabilities: np.ndarray) -> np.ndarray:
        """Transforma probabilidades recortadas en una feature logit."""

        epsilon = self.parameters.clip_epsilon
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        logits = np.log(clipped / (1.0 - clipped))
        return logits.reshape(-1, 1)

    def fit(
        self,
        calibration_probabilities: object,
        calibration_targets: object,
    ) -> "PlattCalibrator":
        """Ajusta Platt únicamente sobre el bloque temporal de calibración.

        Args:
            calibration_probabilities: Probabilidades crudas generadas para el
                bloque ``Y-1`` por un modelo entrenado hasta ``Y-2``.
            calibration_targets: Resultados observados en ese mismo bloque.

        Returns:
            La propia instancia ajustada.

        Raises:
            CalibrationError: Si las entradas son inválidas o solo contienen
                una clase, caso en el que Platt no es identificable.
        """

        probabilities = _as_probability_vector(
            calibration_probabilities,
            name="calibration_probabilities",
        )
        targets = _as_binary_targets(
            calibration_targets,
            probabilities.size,
        )
        if np.unique(targets).size != 2:
            raise CalibrationError(
                "Platt necesita ambas clases en el bloque de calibración."
            )
        estimator = LogisticRegression(
            C=self.parameters.regularization_c,
            solver="lbfgs",
            max_iter=self.parameters.max_iter,
            random_state=self.parameters.random_state,
        )
        estimator.fit(self._logit_feature(probabilities), targets)
        self.estimator_ = estimator
        self.n_calibration_samples_ = int(probabilities.size)
        self.positive_rate_ = float(targets.mean())
        return self

    def predict_proba(self, probabilities: object) -> np.ndarray:
        """Devuelve la probabilidad calibrada de la clase positiva."""

        if self.estimator_ is None:
            raise CalibrationError(
                "El calibrador debe ajustarse antes de predecir."
            )
        raw = _as_probability_vector(
            probabilities,
            name="probabilities",
        )
        return self.estimator_.predict_proba(
            self._logit_feature(raw)
        )[:, 1]

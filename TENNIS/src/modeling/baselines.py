"""Baselines causales de ranking y mercado para la evaluación temporal.

El favorito determinista de ranking sirve para accuracy. El baseline
probabilístico de ranking se ajusta exclusivamente en el train de cada fold y
permite comparar log-loss, Brier y AUC. El mercado solo produce predicción
cuando ya existen probabilidades de-vigadas válidas; nunca fabrica cuotas.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .parameters import (
    DEFAULT_LOGISTIC_PARAMETERS,
    LogisticParameters,
)


class BaselineDataError(ValueError):
    """Indica datos presentes pero inválidos para un baseline."""


@dataclass(frozen=True, slots=True)
class BaselinePredictions:
    """Contiene probabilidades y decisiones con cobertura explícita."""

    name: str
    probability_a: pd.Series
    predicted_y: pd.Series

    @property
    def probability_coverage(self) -> float:
        """Calcula la fracción con probabilidad evaluable."""

        if len(self.probability_a) == 0:
            return 0.0
        return float(self.probability_a.notna().mean())

    @property
    def decision_coverage(self) -> float:
        """Calcula la fracción con favorito evaluable."""

        if len(self.predicted_y) == 0:
            return 0.0
        return float(self.predicted_y.notna().mean())


def _empty_probability(index: pd.Index) -> pd.Series:
    """Crea una serie nullable de probabilidades sin inventar valores."""

    return pd.Series(
        pd.array([pd.NA] * len(index), dtype="Float64"),
        index=index,
        name="probability_a",
    )


def _empty_decision(index: pd.Index) -> pd.Series:
    """Crea una serie nullable de decisiones binarias."""

    return pd.Series(
        pd.array([pd.NA] * len(index), dtype="Int8"),
        index=index,
        name="predicted_y",
    )


def ranking_favorite_predictions(
    frame: pd.DataFrame,
) -> BaselinePredictions:
    """Predice el ganador con ``rank_diff``; empate/ausencia no se evalúa."""

    if "rank_diff" not in frame.columns:
        raise BaselineDataError("Falta rank_diff para el favorito.")
    rank_diff = pd.to_numeric(frame["rank_diff"], errors="coerce")
    finite = pd.Series(
        np.isfinite(rank_diff.to_numpy(dtype=float, na_value=np.nan)),
        index=frame.index,
    )
    decisions = _empty_decision(frame.index)
    decisions.loc[finite & rank_diff.lt(0)] = 1
    decisions.loc[finite & rank_diff.gt(0)] = 0
    return BaselinePredictions(
        name="ranking_favorite",
        probability_a=_empty_probability(frame.index),
        predicted_y=decisions,
    )


@dataclass(slots=True)
class RankProbabilityBaseline:
    """Logística univariable causal ajustada únicamente con ``rank_diff``."""

    parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS
    pipeline: Pipeline | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series | np.ndarray,
    ) -> "RankProbabilityBaseline":
        """Ajusta el mapeo rank→probabilidad con casos completos del train."""

        if "rank_diff" not in frame.columns:
            raise BaselineDataError("Falta rank_diff para entrenar.")
        rank_diff = pd.to_numeric(frame["rank_diff"], errors="coerce")
        target_array = np.asarray(target)
        if target_array.ndim != 1 or target_array.size != len(frame):
            raise BaselineDataError(
                "target debe ser un vector del mismo tamaño que frame."
            )
        target_series = pd.Series(target_array, index=frame.index)
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        valid = (
            rank_diff.notna()
            & np.isfinite(rank_diff)
            & target_numeric.isin([0, 1])
        )
        if not bool(valid.any()):
            raise BaselineDataError(
                "No hay observaciones completas para el baseline de ranking."
            )
        y = target_numeric.loc[valid].astype(int)
        if y.nunique() != 2:
            raise BaselineDataError(
                "El baseline probabilístico requiere ambas clases en train."
            )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "estimator",
                    LogisticRegression(
                        C=float(self.parameters.c),
                        max_iter=self.parameters.max_iter,
                        solver=self.parameters.solver,
                        random_state=self.parameters.random_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(
            rank_diff.loc[valid].to_numpy(dtype=float).reshape(-1, 1),
            y.to_numpy(dtype=np.int8),
        )
        self.pipeline = pipeline
        return self

    def predict(self, frame: pd.DataFrame) -> BaselinePredictions:
        """Devuelve probabilidades solo para filas con ranking disponible."""

        if self.pipeline is None:
            raise BaselineDataError(
                "RankProbabilityBaseline debe ajustarse antes de predecir."
            )
        if "rank_diff" not in frame.columns:
            raise BaselineDataError("Falta rank_diff para predecir.")
        rank_diff = pd.to_numeric(frame["rank_diff"], errors="coerce")
        valid = rank_diff.notna() & np.isfinite(rank_diff)
        probabilities = _empty_probability(frame.index)
        if bool(valid.any()):
            values = self.pipeline.predict_proba(
                rank_diff.loc[valid].to_numpy(dtype=float).reshape(-1, 1)
            )[:, 1]
            probabilities.loc[valid] = values
        decisions = _empty_decision(frame.index)
        decisions.loc[probabilities.notna()] = (
            probabilities.loc[probabilities.notna()].astype(float) >= 0.5
        ).astype("int8")
        return BaselinePredictions(
            name="ranking_probability",
            probability_a=probabilities,
            predicted_y=decisions,
        )


def market_baseline_predictions(
    frame: pd.DataFrame,
    *,
    sum_tolerance: float = 1e-6,
) -> BaselinePredictions:
    """Usa la probabilidad de-vigada y deja ausente lo no observable."""

    if (
        isinstance(sum_tolerance, bool)
        or not isinstance(sum_tolerance, (int, float))
        or not math.isfinite(float(sum_tolerance))
        or float(sum_tolerance) < 0.0
    ):
        raise ValueError("sum_tolerance debe ser finita y no negativa.")
    columns = ("market_probability_a", "market_probability_b")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BaselineDataError(
            f"Faltan columnas de mercado: {missing}."
        )
    raw = frame.loc[:, list(columns)]
    numeric = raw.apply(
        pd.to_numeric, errors="coerce"
    )
    non_numeric = raw.notna() & numeric.isna()
    if bool(non_numeric.any(axis=1).any()):
        invalid_indices = list(
            non_numeric.any(axis=1)[lambda values: values].index[:5]
        )
        raise BaselineDataError(
            "Probabilidades de mercado no numéricas en índices "
            f"{invalid_indices}."
        )
    both_missing = numeric.isna().all(axis=1)
    partial_missing = numeric.isna().any(axis=1) & ~both_missing
    if bool(partial_missing.any()):
        # Una cuota unilateral no permite quitar el vig; se declara no
        # evaluable, igual que una pareja completamente ausente.
        numeric.loc[partial_missing, :] = np.nan
    complete = numeric.notna().all(axis=1)
    if bool(complete.any()):
        complete_values = numeric.loc[complete]
        in_range = (
            complete_values.ge(0.0).all(axis=1)
            & complete_values.le(1.0).all(axis=1)
        )
        sums_valid = complete_values.sum(axis=1).sub(1.0).abs().le(
            float(sum_tolerance)
        )
        invalid = ~(in_range & sums_valid)
        if bool(invalid.any()):
            invalid_indices = list(complete_values.index[invalid][:5])
            raise BaselineDataError(
                "Probabilidades de mercado presentes pero inválidas en "
                f"índices {invalid_indices}."
            )
    probabilities = _empty_probability(frame.index)
    probabilities.loc[complete] = numeric.loc[
        complete, "market_probability_a"
    ].to_numpy(dtype=float)
    decisions = _empty_decision(frame.index)
    strictly_a = probabilities.notna() & probabilities.gt(0.5)
    strictly_b = probabilities.notna() & probabilities.lt(0.5)
    decisions.loc[strictly_a] = 1
    decisions.loc[strictly_b] = 0
    return BaselinePredictions(
        name="market_devig",
        probability_a=probabilities,
        predicted_y=decisions,
    )

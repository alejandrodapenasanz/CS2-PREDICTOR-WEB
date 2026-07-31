"""Pipelines probabilísticos por género para regresión logística y LightGBM.

Las funciones reciben únicamente el fold de entrenamiento. El preprocesador
forma parte del pipeline, por lo que medianas, escalas y categorías nunca se
ajustan con validación o test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import TARGET_COLUMN

from .parameters import (
    DEFAULT_LIGHTGBM_PARAMETERS,
    DEFAULT_LOGISTIC_PARAMETERS,
    LightGBMParameters,
    LogisticParameters,
)
from .preprocessing import (
    FeatureContract,
    FeatureContractError,
    FeatureProfile,
    ResolvedFeatureProfile,
    build_preprocessor,
    select_model_frame,
    validate_market_training_coverage,
)


EstimatorKind = Literal["logistic", "lightgbm"]


class EstimatorContractError(ValueError):
    """Indica que género, etiqueta o estimador incumple el contrato."""


class ModelingDependencyError(RuntimeError):
    """Indica que falta una dependencia declarada para entrenar."""


def _validate_gender(gender: str) -> str:
    """Exige uno de los dos universos de modelado independientes."""

    if gender not in {"M", "F"}:
        raise EstimatorContractError("gender debe ser 'M' o 'F'.")
    return gender


def _extract_target(frame: pd.DataFrame) -> np.ndarray:
    """Extrae una etiqueta binaria completa como vector entero."""

    if TARGET_COLUMN not in frame.columns:
        raise EstimatorContractError(
            f"Falta la etiqueta {TARGET_COLUMN!r}."
        )
    numeric = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise EstimatorContractError("y debe contener únicamente 0 y 1.")
    target = numeric.to_numpy(dtype=np.int8)
    if np.unique(target).size != 2:
        raise EstimatorContractError(
            "El fold de entrenamiento debe contener ambas clases."
        )
    return target


def _validate_training_gender(
    frame: pd.DataFrame,
    gender: str,
) -> None:
    """Impide mezclar hombres y mujeres dentro de un estimador."""

    _validate_gender(gender)
    if "gender" not in frame.columns:
        raise EstimatorContractError("Falta la columna gender.")
    observed = set(frame["gender"].dropna().astype(str).unique())
    if observed != {gender} or frame["gender"].isna().any():
        raise EstimatorContractError(
            f"El train de {gender} contiene géneros incompatibles: "
            f"{sorted(observed)}."
        )


def build_logistic_pipeline(
    feature_columns: tuple[str, ...],
    parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS,
) -> Pipeline:
    """Construye la regresión logística con imputación y escala internas."""

    preprocessor = build_preprocessor(
        feature_columns,
        scale_numeric=True,
    )
    estimator = LogisticRegression(
        C=float(parameters.c),
        max_iter=parameters.max_iter,
        solver=parameters.solver,
        random_state=parameters.random_seed,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("estimator", estimator)]
    )


def build_lightgbm_pipeline(
    feature_columns: tuple[str, ...],
    parameters: LightGBMParameters = DEFAULT_LIGHTGBM_PARAMETERS,
) -> Pipeline:
    """Construye LightGBM con preprocesamiento ajustable en cada fold."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ModelingDependencyError(
            "Falta lightgbm; instale TENNIS/requirements.txt."
        ) from exc
    preprocessor = build_preprocessor(
        feature_columns,
        scale_numeric=False,
    )
    estimator = LGBMClassifier(
        objective="binary",
        n_estimators=parameters.n_estimators,
        learning_rate=float(parameters.learning_rate),
        num_leaves=parameters.num_leaves,
        max_depth=parameters.max_depth,
        min_child_samples=parameters.min_child_samples,
        subsample=float(parameters.subsample),
        subsample_freq=parameters.subsample_freq,
        colsample_bytree=float(parameters.colsample_bytree),
        reg_alpha=float(parameters.reg_alpha),
        reg_lambda=float(parameters.reg_lambda),
        random_state=parameters.random_seed,
        n_jobs=parameters.n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("estimator", estimator)]
    )


@dataclass(slots=True)
class FittedGenderEstimator:
    """Encapsula un pipeline y su contrato de género/features."""

    gender: str
    kind: EstimatorKind
    profile: ResolvedFeatureProfile
    feature_columns: tuple[str, ...]
    dataset_fingerprint: str
    pipeline: Pipeline

    def predict_probability(self, frame: pd.DataFrame) -> pd.Series:
        """Predice ``P(A gana)`` sin aceptar otro género ni otro esquema."""

        _validate_gender(self.gender)
        if "gender" not in frame.columns:
            raise EstimatorContractError(
                "Falta gender para verificar el universo de inferencia."
            )
        observed = set(frame["gender"].dropna().astype(str).unique())
        if (
            observed != {self.gender}
            or frame["gender"].isna().any()
        ):
            raise EstimatorContractError(
                "El frame de inferencia mezcla universos de género."
            )
        missing = [
            column
            for column in self.feature_columns
            if column not in frame.columns
        ]
        if missing:
            raise FeatureContractError(
                f"Faltan features en inferencia: {missing}."
            )
        selected = frame.loc[:, list(self.feature_columns)]
        if self.kind == "lightgbm":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "X does not have valid feature names, but "
                        "LGBMClassifier was fitted with feature names"
                    ),
                    category=UserWarning,
                )
                probabilities = self.pipeline.predict_proba(selected)[:, 1]
        else:
            probabilities = self.pipeline.predict_proba(selected)[:, 1]
        return pd.Series(
            probabilities,
            index=frame.index,
            name="model_probability_a",
            dtype=float,
        )


def fit_gender_estimator(
    train: pd.DataFrame,
    *,
    gender: str,
    kind: EstimatorKind,
    contract: FeatureContract,
    profile: FeatureProfile = "auto",
    logistic_parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS,
    lightgbm_parameters: LightGBMParameters = (
        DEFAULT_LIGHTGBM_PARAMETERS
    ),
) -> FittedGenderEstimator:
    """Ajusta un modelo de un género usando solo el frame recibido."""

    if not isinstance(train, pd.DataFrame) or train.empty:
        raise EstimatorContractError(
            "El fold de entrenamiento debe ser un DataFrame no vacío."
        )
    _validate_training_gender(train, gender)
    target = _extract_target(train)
    resolved_profile = contract.resolve_profile(profile)
    selected = select_model_frame(train, contract, resolved_profile)
    if resolved_profile == "market_enhanced":
        validate_market_training_coverage(selected)
    feature_columns = contract.columns_for(resolved_profile)
    if kind == "logistic":
        pipeline = build_logistic_pipeline(
            feature_columns, logistic_parameters
        )
    elif kind == "lightgbm":
        pipeline = build_lightgbm_pipeline(
            feature_columns, lightgbm_parameters
        )
    else:
        raise EstimatorContractError(
            "kind debe ser 'logistic' o 'lightgbm'."
        )
    pipeline.fit(selected, target)
    return FittedGenderEstimator(
        gender=gender,
        kind=kind,
        profile=resolved_profile,
        feature_columns=feature_columns,
        dataset_fingerprint=contract.dataset_fingerprint,
        pipeline=pipeline,
    )

"""Contrato de features y preprocesamiento entrenado solo sobre el pasado.

El manifiesto de la fase 6 es la única fuente autorizada de nombres de
features. El código exige que su allowlist coincida exactamente con el
contrato local versionado: una columna nueva no entra al modelo por accidente.
Los perfiles distinguen expresamente modelos deportivos de modelos que usan
mercado.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


FeatureProfile = Literal["auto", "sports_only", "market_enhanced"]
ResolvedFeatureProfile = Literal["sports_only", "market_enhanced"]

CATEGORICAL_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"surface", "tour_level", "tour_level_raw", "round"}
)
MARKET_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "odds_a",
    "odds_b",
    "market_probability_a",
    "market_probability_b",
    "market_overround",
    "market_margin",
)
_FORBIDDEN_PREDICTOR_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        TARGET_COLUMN,
        "gender",
        "match_date",
        "record_id",
        "player_a_id",
        "player_b_id",
        "model_probability_a",
        "edge",
    }
)


class FeatureContractError(ValueError):
    """Indica un manifiesto o una matriz incompatible con la fase 6."""


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """Representa la allowlist inmutable declarada por un manifiesto."""

    schema_version: str
    dataset_fingerprint: str
    feature_columns: tuple[str, ...]
    training_columns: tuple[str, ...]
    historical_odds_available: bool

    def resolve_profile(
        self,
        profile: FeatureProfile = "auto",
    ) -> ResolvedFeatureProfile:
        """Resuelve el perfil sin activar mercado cuando no hay histórico."""

        if profile not in {"auto", "sports_only", "market_enhanced"}:
            raise FeatureContractError(
                "profile debe ser auto, sports_only o market_enhanced."
            )
        if profile == "auto":
            return (
                "market_enhanced"
                if self.historical_odds_available
                else "sports_only"
            )
        if (
            profile == "market_enhanced"
            and not self.historical_odds_available
        ):
            raise FeatureContractError(
                "El manifiesto declara historical_odds_available=false; "
                "market_enhanced no es evaluable."
            )
        return profile

    def columns_for(
        self,
        profile: FeatureProfile = "auto",
    ) -> tuple[str, ...]:
        """Devuelve las columnas activas, conservando el orden contractual."""

        resolved = self.resolve_profile(profile)
        if resolved == "market_enhanced":
            return self.feature_columns
        return tuple(
            column
            for column in self.feature_columns
            if column not in MARKET_FEATURE_COLUMNS
        )


def _load_manifest_mapping(
    manifest: str | Path | Mapping[str, object],
) -> Mapping[str, object]:
    """Lee un manifiesto JSON o copia la referencia a un mapping."""

    if isinstance(manifest, Mapping):
        return manifest
    path = Path(manifest)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureContractError(
            f"No se pudo leer el manifiesto de features {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FeatureContractError(
            "La raíz del manifiesto debe ser un objeto JSON."
        )
    return payload


def _required_string(
    payload: Mapping[str, object],
    key: str,
) -> str:
    """Extrae una cadena obligatoria no vacía del manifiesto."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise FeatureContractError(
            f"El manifiesto requiere {key!r} como texto no vacío."
        )
    return value


def _required_string_tuple(
    payload: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    """Extrae una lista obligatoria de nombres únicos y no vacíos."""

    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item
            for item in value
        )
    ):
        raise FeatureContractError(
            f"El manifiesto requiere {key!r} como lista de textos."
        )
    result = tuple(value)
    if len(result) != len(set(result)):
        raise FeatureContractError(
            f"El manifiesto contiene duplicados en {key!r}."
        )
    return result


def load_feature_contract(
    manifest: str | Path | Mapping[str, object],
) -> FeatureContract:
    """Valida el manifiesto de fase 6 y construye su contrato de modelado."""

    payload = _load_manifest_mapping(manifest)
    schema_version = _required_string(payload, "schema_version")
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise FeatureContractError(
            "Versión de features incompatible: "
            f"{schema_version!r} != {FEATURE_SCHEMA_VERSION!r}."
        )
    feature_columns = _required_string_tuple(
        payload, "model_feature_columns"
    )
    if feature_columns != tuple(MODEL_FEATURE_COLUMNS):
        missing = sorted(set(MODEL_FEATURE_COLUMNS) - set(feature_columns))
        extra = sorted(set(feature_columns) - set(MODEL_FEATURE_COLUMNS))
        raise FeatureContractError(
            "La allowlist del manifiesto no coincide exactamente con "
            "MODEL_FEATURE_COLUMNS: "
            f"missing={missing}, extra={extra}, order_equal="
            f"{feature_columns == tuple(MODEL_FEATURE_COLUMNS)}."
        )
    if set(feature_columns) & _FORBIDDEN_PREDICTOR_COLUMNS:
        raise FeatureContractError(
            "La allowlist contiene identidad, tiempo, objetivo o postdicción."
        )
    training_columns = _required_string_tuple(payload, "training_columns")
    required_training = {
        *feature_columns,
        TARGET_COLUMN,
        "gender",
        "match_date",
    }
    if not required_training.issubset(training_columns):
        missing = sorted(required_training - set(training_columns))
        raise FeatureContractError(
            f"Faltan columnas contractuales de entrenamiento: {missing}."
        )
    historical_odds = payload.get("historical_odds_available")
    if not isinstance(historical_odds, bool):
        raise FeatureContractError(
            "historical_odds_available debe ser booleano."
        )
    return FeatureContract(
        schema_version=schema_version,
        dataset_fingerprint=_required_string(payload, "fingerprint"),
        feature_columns=feature_columns,
        training_columns=training_columns,
        historical_odds_available=historical_odds,
    )


def select_model_frame(
    frame: pd.DataFrame,
    contract: FeatureContract,
    profile: FeatureProfile = "auto",
) -> pd.DataFrame:
    """Selecciona solo la allowlist activa y falla ante columnas ausentes."""

    if not isinstance(frame, pd.DataFrame):
        raise FeatureContractError("La matriz predictora debe ser DataFrame.")
    columns = contract.columns_for(profile)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureContractError(
            f"Faltan features exigidas por el manifiesto: {missing}."
        )
    return frame.loc[:, list(columns)].copy()


def validate_market_training_coverage(frame: pd.DataFrame) -> None:
    """Exige alguna pareja de probabilidades de mercado usable en train.

    Las ausencias por partido siguen admitidas. Esta comprobación únicamente
    impide entrenar un perfil de mercado cuando su señal completa es nula.
    """

    required = ("market_probability_a", "market_probability_b")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FeatureContractError(
            f"Faltan probabilidades de mercado: {missing}."
        )
    probabilities = frame.loc[:, list(required)].apply(
        pd.to_numeric, errors="coerce"
    )
    usable = (
        probabilities.notna().all(axis=1)
        & probabilities.ge(0.0).all(axis=1)
        & probabilities.le(1.0).all(axis=1)
    )
    if not bool(usable.any()):
        raise FeatureContractError(
            "market_enhanced requiere al menos una pareja de probabilidades "
            "válida en el conjunto de entrenamiento."
        )


def build_preprocessor(
    feature_columns: Sequence[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Construye transformaciones que se ajustarán dentro del pipeline.

    La imputación mediana, el escalado y las categorías se aprenden cuando el
    pipeline recibe ``fit(train)``. ``keep_empty_features=True`` evita que una
    columna contractual desaparezca silenciosamente en un fold.
    """

    columns = tuple(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise FeatureContractError(
            "feature_columns debe contener nombres únicos."
        )
    if any(not isinstance(column, str) or not column for column in columns):
        raise FeatureContractError(
            "Cada feature debe ser un nombre de columna no vacío."
        )
    unknown = [
        column
        for column in columns
        if column not in MODEL_FEATURE_COLUMNS
    ]
    if unknown:
        raise FeatureContractError(
            f"Features fuera de la allowlist de fase 6: {unknown}."
        )
    canonical_positions = {
        column: position
        for position, column in enumerate(MODEL_FEATURE_COLUMNS)
    }
    if list(columns) != sorted(
        columns, key=canonical_positions.__getitem__
    ):
        raise FeatureContractError(
            "Las features deben conservar el orden de la allowlist."
        )
    categorical = [
        column
        for column in columns
        if column in CATEGORICAL_FEATURE_COLUMNS
    ]
    numeric = [
        column
        for column in columns
        if column not in CATEGORICAL_FEATURE_COLUMNS
    ]

    numeric_steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

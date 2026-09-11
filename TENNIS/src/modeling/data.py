"""Carga y verifica los datasets causales usados por la fase 7.

El módulo nunca infiere columnas ni rutas a partir de convenciones externas:
lee el manifiesto publicado por la fase 6, verifica tamaño y SHA-256 del
Parquet seleccionado y solo entonces entrega un ``DataFrame`` ordenado. Las
columnas de auditoría necesarias para los baselines permanecen separadas de la
allowlist que recibirá el estimador.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence, cast

import pandas as pd

from ..artifact_integrity import CodeInventoryError, verify_code_inventory
from ..config import FEATURE_DATASET_MANIFEST_PATH, PROJECT_ROOT
from ..features import MODEL_FEATURE_COLUMNS
from ..features.dataset import FEATURE_CODE_PATHS
from ..features.artifacts import (
    FeatureArtifactError,
    resolve_feature_manifest_path,
)
from ..temporal import SourceDatePolicy, SourceDatePolicyError


Gender = Literal["M", "F"]

REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "gender",
    "match_date",
    "result_available_date",
    "tour_level",
    "surface",
    "rank_a",
    "rank_b",
    "market_probability_a",
    "market_probability_b",
    "y",
)


class ModelDataError(RuntimeError):
    """Indica que las features publicadas no son aptas para entrenar."""


@dataclass(frozen=True, slots=True)
class FeatureSourceManifest:
    """Identidad verificada de los datasets publicados por la fase 6."""

    path: Path
    fingerprint: str
    schema_version: str
    source_commit: str
    historical_odds_available: bool
    model_feature_columns: tuple[str, ...]
    source_date_policy: SourceDatePolicy
    raw_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TrainingDatasetMetadata:
    """Metadatos verificados de un Parquet de un único género."""

    gender: Gender
    path: Path
    rows: int
    min_date: date
    max_date: date
    sha256: str
    size: int
    feature_fingerprint: str


@dataclass(frozen=True, slots=True)
class LoadedTrainingDataset:
    """Combina las filas cargadas con su identidad fuente inmutable."""

    frame: pd.DataFrame
    metadata: TrainingDatasetMetadata
    source_manifest: FeatureSourceManifest


def _ensure_project_path(path: Path, field_name: str) -> Path:
    """Resuelve una ruta y exige que permanezca dentro de ``TENNIS/``."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ModelDataError(
            f"{field_name} debe permanecer dentro de TENNIS/: {resolved}."
        )
    return resolved


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques para un artefacto existente."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelDataError(f"No se pudo leer {path} para verificarlo.") from exc
    return digest.hexdigest()


def _required_string(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    """Obtiene una cadena no vacía o falla con contexto reproducible."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ModelDataError(f"{context}.{key} debe ser una cadena no vacía.")
    return value


def _required_nonnegative_integer(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    """Obtiene un entero no negativo rechazando booleanos."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelDataError(f"{context}.{key} debe ser un entero no negativo.")
    return value


def load_feature_source_manifest(
    manifest_path: Path = FEATURE_DATASET_MANIFEST_PATH,
) -> FeatureSourceManifest:
    """Carga y valida el contrato mínimo del manifiesto de fase 6.

    Args:
        manifest_path: Ruta al manifiesto publicado junto a los Parquet.

    Returns:
        Identidad tipada del snapshot de features.

    Raises:
        ModelDataError: Si el JSON, el esquema o la allowlist divergen.
    """

    requested = _ensure_project_path(manifest_path, "manifest_path")
    try:
        resolved = resolve_feature_manifest_path(requested)
    except FeatureArtifactError as exc:
        raise ModelDataError(
            f"No se pudo resolver el manifiesto activo {requested}."
        ) from exc
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelDataError(
            f"No se pudo cargar el manifiesto de features {resolved}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelDataError("El manifiesto de features debe ser un objeto JSON.")
    try:
        verify_code_inventory(
            PROJECT_ROOT,
            FEATURE_CODE_PATHS,
            payload.get("code_inventory"),
        )
    except CodeInventoryError as exc:
        raise ModelDataError(
            "El dataset de features fue generado por código distinto del "
            "runtime actual; se exige reconstruirlo."
        ) from exc

    columns = payload.get("model_feature_columns")
    if (
        not isinstance(columns, list)
        or not all(isinstance(column, str) for column in columns)
    ):
        raise ModelDataError(
            "model_feature_columns debe ser una lista íntegra de cadenas."
        )
    typed_columns = tuple(cast(list[str], columns))
    if typed_columns != tuple(MODEL_FEATURE_COLUMNS):
        raise ModelDataError(
            "La allowlist del manifiesto no coincide exactamente con "
            "src.features.MODEL_FEATURE_COLUMNS."
        )

    historical_odds = payload.get("historical_odds_available")
    if not isinstance(historical_odds, bool):
        raise ModelDataError(
            "historical_odds_available debe ser booleano en el manifiesto."
        )
    raw_date_policy = payload.get("source_date_policy")
    if not isinstance(raw_date_policy, Mapping):
        raise ModelDataError(
            "El manifiesto no declara source_date_policy; reconstruya features."
        )
    try:
        source_date_policy = SourceDatePolicy.from_mapping(raw_date_policy)
    except SourceDatePolicyError as exc:
        raise ModelDataError(
            "source_date_policy del manifiesto no es compatible."
        ) from exc

    return FeatureSourceManifest(
        path=resolved,
        fingerprint=_required_string(
            payload, "fingerprint", context="manifest"
        ),
        schema_version=_required_string(
            payload, "schema_version", context="manifest"
        ),
        source_commit=_required_string(
            payload, "source_commit", context="manifest"
        ),
        historical_odds_available=historical_odds,
        model_feature_columns=typed_columns,
        source_date_policy=source_date_policy,
        raw_payload=cast(Mapping[str, object], payload),
    )


def _dataset_entry(
    source_manifest: FeatureSourceManifest,
    gender: Gender,
) -> Mapping[str, object]:
    """Selecciona exactamente una entrada de dataset para el género."""

    datasets = source_manifest.raw_payload.get("datasets")
    if not isinstance(datasets, list):
        raise ModelDataError("manifest.datasets debe ser una lista.")
    matches = [
        item
        for item in datasets
        if isinstance(item, Mapping) and item.get("gender") == gender
    ]
    if len(matches) != 1:
        raise ModelDataError(
            f"Se esperaba un único dataset para {gender}; encontrados "
            f"{len(matches)}."
        )
    return cast(Mapping[str, object], matches[0])


def verify_training_dataset(
    source_manifest: FeatureSourceManifest,
    gender: Gender,
) -> TrainingDatasetMetadata:
    """Verifica ruta, tamaño, hash, filas y rango declarado de un Parquet."""

    if gender not in {"M", "F"}:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    entry = _dataset_entry(source_manifest, gender)
    relative_path = _required_string(
        entry, "output_path", context=f"datasets[{gender}]"
    )
    candidate = (source_manifest.path.parent / relative_path).resolve()
    candidate = _ensure_project_path(candidate, "dataset_path")
    if not candidate.is_relative_to(source_manifest.path.parent.resolve()):
        raise ModelDataError(
            f"El Parquet {relative_path!r} sale del directorio del manifiesto."
        )
    expected_size = _required_nonnegative_integer(
        entry, "output_size", context=f"datasets[{gender}]"
    )
    expected_rows = _required_nonnegative_integer(
        entry, "training_rows", context=f"datasets[{gender}]"
    )
    expected_sha = _required_string(
        entry, "output_sha256", context=f"datasets[{gender}]"
    )
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ModelDataError(
            f"datasets[{gender}].output_sha256 no es SHA-256 hexadecimal."
        )
    if not candidate.is_file():
        raise ModelDataError(f"No existe el dataset declarado: {candidate}.")
    actual_size = candidate.stat().st_size
    if actual_size != expected_size:
        raise ModelDataError(
            f"Tamaño divergente para {candidate}: "
            f"{actual_size} != {expected_size}."
        )
    actual_sha = sha256_file(candidate)
    if actual_sha != expected_sha:
        raise ModelDataError(
            f"SHA-256 divergente para {candidate}: "
            f"{actual_sha} != {expected_sha}."
        )
    try:
        min_date = date.fromisoformat(
            _required_string(entry, "min_date", context=f"datasets[{gender}]")
        )
        max_date = date.fromisoformat(
            _required_string(entry, "max_date", context=f"datasets[{gender}]")
        )
    except ValueError as exc:
        raise ModelDataError(
            f"Rango de fechas inválido para el dataset {gender}."
        ) from exc

    return TrainingDatasetMetadata(
        gender=gender,
        path=candidate,
        rows=expected_rows,
        min_date=min_date,
        max_date=max_date,
        sha256=actual_sha,
        size=actual_size,
        feature_fingerprint=source_manifest.fingerprint,
    )


def load_training_dataset(
    gender: Gender,
    *,
    feature_columns: Sequence[str],
    manifest_path: Path = FEATURE_DATASET_MANIFEST_PATH,
) -> LoadedTrainingDataset:
    """Carga las columnas solicitadas tras verificar el artefacto completo.

    Args:
        gender: Universo masculino (``M``) o femenino (``F``).
        feature_columns: Allowlist predictora ya resuelta por el perfil.
        manifest_path: Manifiesto de la fase 6.

    Returns:
        DataFrame cronológico con referencias, objetivo y features.

    Raises:
        ModelDataError: Si faltan columnas o cualquier invariante diverge.
    """

    source_manifest = load_feature_source_manifest(manifest_path)
    metadata = verify_training_dataset(source_manifest, gender)
    requested_features = tuple(feature_columns)
    if len(set(requested_features)) != len(requested_features):
        raise ModelDataError("feature_columns contiene nombres duplicados.")
    unsupported = sorted(
        set(requested_features).difference(
            source_manifest.model_feature_columns
        )
    )
    if unsupported:
        raise ModelDataError(
            f"Features fuera de la allowlist publicada: {unsupported}."
        )
    columns = tuple(dict.fromkeys((*REFERENCE_COLUMNS, *requested_features)))
    try:
        frame = pd.read_parquet(metadata.path, columns=list(columns))
    except Exception as exc:
        raise ModelDataError(
            f"No se pudo leer el Parquet verificado {metadata.path}."
        ) from exc
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ModelDataError(f"El Parquet carece de columnas: {missing}.")
    if len(frame) != metadata.rows:
        raise ModelDataError(
            f"Filas divergentes para {gender}: {len(frame)} != {metadata.rows}."
        )

    frame["match_date"] = pd.to_datetime(
        frame["match_date"], errors="raise"
    ).dt.normalize()
    if frame["match_date"].isna().any():
        raise ModelDataError("match_date contiene valores nulos.")
    frame["result_available_date"] = pd.to_datetime(
        frame["result_available_date"], errors="raise"
    ).dt.normalize()
    if frame["result_available_date"].isna().any():
        raise ModelDataError("result_available_date contiene valores nulos.")
    expected_available = frame["match_date"].map(
        lambda value: pd.Timestamp(
            source_manifest.source_date_policy.availability_date(value.date())
        )
    )
    if not frame["result_available_date"].equals(expected_available):
        raise ModelDataError(
            "result_available_date no coincide con source_date_policy."
        )
    observed_genders = set(frame["gender"].dropna().unique())
    if observed_genders != {gender}:
        raise ModelDataError(
            f"El dataset {gender} contiene géneros {observed_genders}."
        )
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ModelDataError("record_id debe ser no nulo y único.")
    observed_targets = set(frame["y"].dropna().unique())
    if frame["y"].isna().any() or not observed_targets.issubset({0, 1}):
        raise ModelDataError(f"Objetivos inválidos: {observed_targets}.")
    if frame.empty:
        raise ModelDataError(f"El dataset {gender} está vacío.")

    observed_min = cast(pd.Timestamp, frame["match_date"].min()).date()
    observed_max = cast(pd.Timestamp, frame["match_date"].max()).date()
    if (observed_min, observed_max) != (metadata.min_date, metadata.max_date):
        raise ModelDataError(
            f"Rango divergente para {gender}: "
            f"{observed_min}..{observed_max} != "
            f"{metadata.min_date}..{metadata.max_date}."
        )
    frame = frame.sort_values(
        ["match_date", "record_id"], kind="stable"
    ).reset_index(drop=True)
    return LoadedTrainingDataset(
        frame=frame,
        metadata=metadata,
        source_manifest=source_manifest,
    )

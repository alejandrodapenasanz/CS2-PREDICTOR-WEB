"""Carga verificada e inferencia del modelo principal activo por género.

Antes de deserializar Joblib se comprueban todos los hashes del run. La API
devuelve ``P(A gana)`` calibrada y calcula ``edge`` únicamente si la fila
incluye una probabilidad de mercado de-vigada válida.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Literal, Mapping
import warnings

import joblib
import numpy as np
import pandas as pd

from ..config import (
    PHASE7_ACTIVE_MANIFEST_PATH,
    PROJECT_ROOT,
)
from .artifacts import (
    ArtifactError,
    verify_model_code_inventory,
    verify_published_run,
)
from .calibration import PlattCalibrator
from .estimators import FittedGenderEstimator
from .parameters import MODEL_ARTIFACT_VERSION
from .orientation import (
    predict_symmetric_calibrated,
    predict_symmetric_raw,
)
from ..temporal import SourceDatePolicy, SourceDatePolicyError


Gender = Literal["M", "F"]


class ModelServiceError(RuntimeError):
    """Indica que el modelo activo no es íntegro o compatible."""


@dataclass(frozen=True, slots=True)
class LoadedDeploymentModel:
    """Bundle principal verificado listo para inferencia."""

    gender: Gender
    estimator: FittedGenderEstimator
    calibrator: PlattCalibrator
    training_rows: int
    training_source_rows: int
    training_excluded_unavailable: int
    training_max_date: str
    training_available_max_date: str
    training_as_of_date: str
    source_date_policy: SourceDatePolicy
    calibration_rows: int
    run_fingerprint: str
    run_dir: Path

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Predice solo si el modelo termina estrictamente antes del corte."""

        if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
            raise ModelServiceError("as_of_date debe ser datetime.date estricto.")
        try:
            training_cutoff = date.fromisoformat(
                self.training_available_max_date
            )
        except ValueError as exc:
            raise ModelServiceError(
                "training_available_max_date del bundle no es YYYY-MM-DD."
            ) from exc
        if training_cutoff >= as_of_date:
            raise ModelServiceError(
                "El modelo no es causal para as_of_date: se exige "
                "training_available_max_date < as_of_date."
            )

        raw = predict_symmetric_raw(self.estimator, frame)
        calibrated = predict_symmetric_calibrated(
            self.calibrator,
            raw.to_numpy(dtype=float)
        )
        output = pd.DataFrame(
            {
                "model_probability_raw_a": raw.to_numpy(dtype=float),
                "model_probability_a": calibrated,
            },
            index=frame.index,
        )
        market = pd.Series(np.nan, index=frame.index, dtype=float)
        market_columns = {
            "market_probability_a",
            "market_probability_b",
        }
        present_columns = market_columns.intersection(frame.columns)
        if present_columns and present_columns != market_columns:
            raise ModelServiceError(
                "El mercado requiere probabilidades A y B conjuntamente."
            )
        if present_columns:
            numeric = pd.to_numeric(
                frame["market_probability_a"], errors="coerce"
            )
            numeric_b = pd.to_numeric(
                frame["market_probability_b"], errors="coerce"
            )
            invalid = (
                numeric.notna() & ~numeric.between(0.0, 1.0)
            ) | (
                numeric_b.notna() & ~numeric_b.between(0.0, 1.0)
            )
            if invalid.any():
                raise ModelServiceError(
                    "Las probabilidades de mercado salen de [0, 1]."
                )
            mismatched_missing = numeric.isna() ^ numeric_b.isna()
            if mismatched_missing.any():
                raise ModelServiceError(
                    "Las probabilidades de mercado A/B deben faltar juntas."
                )
            present = numeric.notna() & numeric_b.notna()
            non_complementary = present & ~np.isclose(
                numeric + numeric_b,
                1.0,
                rtol=0.0,
                atol=1e-9,
            )
            if non_complementary.any():
                raise ModelServiceError(
                    "Las probabilidades de-vigadas A/B deben sumar uno."
                )
            if present.any():
                if "market_retrieved_at_utc" not in frame.columns:
                    raise ModelServiceError(
                        "El mercado requiere market_retrieved_at_utc."
                    )
                try:
                    retrieved = pd.to_datetime(
                        frame["market_retrieved_at_utc"],
                        errors="raise",
                        utc=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise ModelServiceError(
                        "market_retrieved_at_utc no es un timestamp UTC válido."
                    ) from exc
                invalid_time = retrieved.isna() | retrieved.dt.date.map(
                    lambda value: value >= as_of_date
                )
                if (present & invalid_time).any():
                    raise ModelServiceError(
                        "El mercado no cumple retrieved_date < as_of_date."
                    )
                market.loc[present] = numeric.loc[present]
        output["market_probability_a"] = market
        output["edge"] = output["model_probability_a"] - market
        return output


def _ensure_project_path(path: Path, field_name: str) -> Path:
    """Resuelve una ruta y exige que permanezca dentro de ``TENNIS/``."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ModelServiceError(
            f"{field_name} debe permanecer dentro de TENNIS/: {resolved}."
        )
    return resolved


def _load_active_payload(path: Path) -> Mapping[str, object]:
    """Carga el manifiesto activo como mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelServiceError(
            f"No se pudo leer el manifiesto activo {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelServiceError(
            "El manifiesto activo debe ser un objeto JSON."
        )
    return payload


def _model_entry(
    payload: Mapping[str, object],
    gender: Gender,
) -> Mapping[str, object]:
    """Selecciona exactamente un bundle del género solicitado."""

    models = payload.get("models")
    if not isinstance(models, list):
        raise ModelServiceError("El manifiesto activo carece de models.")
    entries = [
        item
        for item in models
        if isinstance(item, Mapping) and item.get("gender") == gender
    ]
    if len(entries) != 1:
        raise ModelServiceError(
            f"Se esperaba un modelo activo para {gender}; "
            f"encontrados {len(entries)}."
        )
    return entries[0]


def load_active_deployment_model(
    gender: Gender,
    *,
    manifest_path: Path = PHASE7_ACTIVE_MANIFEST_PATH,
) -> LoadedDeploymentModel:
    """Verifica y carga el LightGBM+Platt activo de un género."""

    if gender not in {"M", "F"}:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    active_path = _ensure_project_path(manifest_path, "manifest_path")
    active = _load_active_payload(active_path)
    active_run = active.get("active_run")
    if not isinstance(active_run, str) or not active_run:
        raise ModelServiceError("El manifiesto no declara active_run.")
    run_dir = (active_path.parent / active_run).resolve()
    if not run_dir.is_relative_to(active_path.parent.resolve()):
        raise ModelServiceError("active_run sale del directorio de modelos.")
    verified = verify_published_run(run_dir)
    if verified.get("fingerprint") != active.get("fingerprint"):
        raise ModelServiceError(
            "El fingerprint activo no coincide con el run verificado."
        )
    try:
        verify_model_code_inventory(verified.get("code_inventory"))
    except ArtifactError as exc:
        raise ModelServiceError(
            "El modelo activo fue producido por código distinto; "
            "reentrene antes de deserializarlo."
        ) from exc
    entry = _model_entry(verified, gender)
    paths = entry.get("paths")
    if not isinstance(paths, Mapping):
        raise ModelServiceError("La entrada de modelo carece de paths.")
    relative_bundle = paths.get("deployment_bundle")
    if not isinstance(relative_bundle, str):
        raise ModelServiceError("Falta la ruta deployment_bundle.")
    bundle_path = (run_dir / relative_bundle).resolve()
    if not bundle_path.is_relative_to(run_dir):
        raise ModelServiceError("deployment_bundle sale del run.")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting the shape on a NumPy array has been deprecated",
                category=DeprecationWarning,
                module="joblib.numpy_pickle",
            )
            bundle = joblib.load(bundle_path)
    except Exception as exc:
        raise ModelServiceError(
            f"No se pudo cargar el bundle verificado {bundle_path}."
        ) from exc
    if not isinstance(bundle, Mapping):
        raise ModelServiceError("El bundle no es un mapping compatible.")
    if (
        bundle.get("artifact_version") != MODEL_ARTIFACT_VERSION
        or bundle.get("gender") != gender
        or not isinstance(bundle.get("estimator"), FittedGenderEstimator)
        or not isinstance(bundle.get("calibrator"), PlattCalibrator)
    ):
        raise ModelServiceError(
            "El bundle activo no cumple versión, género o tipos esperados."
        )
    training_rows = bundle.get("training_rows")
    training_source_rows = bundle.get("training_source_rows")
    training_excluded = bundle.get("training_excluded_unavailable")
    calibration_rows = bundle.get("calibration_rows")
    training_max_date = bundle.get("training_max_date")
    training_available_max_date = bundle.get(
        "training_available_max_date"
    )
    training_as_of_date = bundle.get("training_as_of_date")
    raw_date_policy = bundle.get("source_date_policy")
    if (
        isinstance(training_rows, bool)
        or not isinstance(training_rows, int)
        or training_rows <= 0
        or isinstance(training_source_rows, bool)
        or not isinstance(training_source_rows, int)
        or training_source_rows < training_rows
        or isinstance(training_excluded, bool)
        or not isinstance(training_excluded, int)
        or training_excluded != training_source_rows - training_rows
        or isinstance(calibration_rows, bool)
        or not isinstance(calibration_rows, int)
        or calibration_rows <= 0
        or not isinstance(training_max_date, str)
        or not isinstance(training_available_max_date, str)
        or not isinstance(training_as_of_date, str)
        or not isinstance(raw_date_policy, Mapping)
    ):
        raise ModelServiceError("Metadatos de entrenamiento inválidos.")
    try:
        source_max = date.fromisoformat(training_max_date)
        available_max = date.fromisoformat(training_available_max_date)
        training_cutoff = date.fromisoformat(training_as_of_date)
        policy = SourceDatePolicy.from_mapping(raw_date_policy)
    except (ValueError, SourceDatePolicyError) as exc:
        raise ModelServiceError(
            "Fechas o source_date_policy del bundle no son válidas."
        ) from exc
    if policy.availability_date(source_max) != available_max:
        raise ModelServiceError(
            "training_available_max_date no deriva de training_max_date."
        )
    if available_max >= training_cutoff:
        raise ModelServiceError(
            "El bundle incorporó un resultado no disponible en su corte."
        )
    fingerprint = verified.get("fingerprint")
    assert isinstance(fingerprint, str)
    return LoadedDeploymentModel(
        gender=gender,
        estimator=bundle["estimator"],
        calibrator=bundle["calibrator"],
        training_rows=training_rows,
        training_source_rows=training_source_rows,
        training_excluded_unavailable=training_excluded,
        training_max_date=training_max_date,
        training_available_max_date=training_available_max_date,
        training_as_of_date=training_as_of_date,
        source_date_policy=policy,
        calibration_rows=calibration_rows,
        run_fingerprint=fingerprint,
        run_dir=run_dir,
    )

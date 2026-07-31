"""Carga verificada e inferencia del modelo principal activo por género.

Antes de deserializar Joblib se comprueban todos los hashes del run. La API
devuelve ``P(A gana)`` calibrada y calcula ``edge`` únicamente si la fila
incluye una probabilidad de mercado de-vigada válida.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .artifacts import verify_published_run
from .calibration import PlattCalibrator
from .estimators import FittedGenderEstimator
from .parameters import MODEL_ARTIFACT_VERSION
from .orientation import (
    predict_symmetric_calibrated,
    predict_symmetric_raw,
)


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
    training_max_date: str
    calibration_rows: int
    run_fingerprint: str
    run_dir: Path

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predice raw, calibrada, mercado y edge conservando el índice."""

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
        if "market_probability_a" in frame.columns:
            numeric = pd.to_numeric(
                frame["market_probability_a"], errors="coerce"
            )
            invalid = numeric.notna() & ~numeric.between(0.0, 1.0)
            if invalid.any():
                raise ModelServiceError(
                    "market_probability_a presente fuera de [0, 1]."
                )
            market.loc[numeric.notna()] = numeric.loc[numeric.notna()]
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
    calibration_rows = bundle.get("calibration_rows")
    training_max_date = bundle.get("training_max_date")
    if (
        isinstance(training_rows, bool)
        or not isinstance(training_rows, int)
        or training_rows <= 0
        or isinstance(calibration_rows, bool)
        or not isinstance(calibration_rows, int)
        or calibration_rows <= 0
        or not isinstance(training_max_date, str)
    ):
        raise ModelServiceError("Metadatos de entrenamiento inválidos.")
    fingerprint = verified.get("fingerprint")
    assert isinstance(fingerprint, str)
    return LoadedDeploymentModel(
        gender=gender,
        estimator=bundle["estimator"],
        calibrator=bundle["calibrator"],
        training_rows=training_rows,
        training_max_date=training_max_date,
        calibration_rows=calibration_rows,
        run_fingerprint=fingerprint,
        run_dir=run_dir,
    )

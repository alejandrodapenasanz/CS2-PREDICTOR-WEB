"""Orquestación reproducible del backtest y reentreno de fase 7.

La función pública verifica primero los Parquet de fase 6, calcula una
identidad que incluye fuentes, código, parámetros y librerías, y reutiliza un
run íntegro si ya existe. Un run nuevo procesa cada género por separado,
publica modelos y calibradores inmutables y actualiza el informe activo solo
después de verificar todos los artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import (
    DOCS_DIR,
    FEATURE_DATASET_MANIFEST_PATH,
    PHASE7_MODELS_DIR,
    PROJECT_ROOT,
)
from .artifacts import (
    PublishedRun,
    activate_published_run,
    create_staging_directory,
    dump_joblib_artifact,
    find_verified_run,
    fingerprint_payload,
    modeling_source_inventory,
    publish_staged_run,
    runtime_versions,
    write_json_artifact,
)
from .backtest import (
    GenderFinalModels,
    fit_final_gender_models,
    run_gender_backtest,
)
from .data import (
    FeatureSourceManifest,
    TrainingDatasetMetadata,
    load_feature_source_manifest,
    load_training_dataset,
    sha256_file,
    verify_training_dataset,
)
from .evaluation import (
    GenderEvaluationResult,
    evaluate_gender_backtest,
)
from .parameters import (
    DEFAULT_LIGHTGBM_PARAMETERS,
    DEFAULT_LOGISTIC_PARAMETERS,
    DEFAULT_TEMPORAL_EVALUATION_PARAMETERS,
    MODEL_ARTIFACT_VERSION,
    LightGBMParameters,
    LogisticParameters,
    TemporalEvaluationParameters,
)
from .plots import save_reliability_plot
from .preprocessing import (
    FeatureContract,
    FeatureProfile,
    load_feature_contract,
)
from .reporting import render_model_report, write_model_report


class TrainingError(RuntimeError):
    """Indica que no se pudo completar un run canónico de modelos."""


@dataclass(frozen=True, slots=True)
class ModelTrainingRun:
    """Resultado consumible por la CLI tras publicar o reutilizar."""

    published: PublishedRun
    summaries: tuple[Mapping[str, object], ...]
    metrics: pd.DataFrame
    market_audits: pd.DataFrame
    suspicious_segments: pd.DataFrame

    @property
    def skipped(self) -> bool:
        """Indica si los artefactos ya existían y se verificaron."""

        return self.published.skipped


def _parameter_payload(
    *,
    feature_source: FeatureSourceManifest,
    metadata: Sequence[TrainingDatasetMetadata],
    contract: FeatureContract,
    profile: str,
    logistic: LogisticParameters,
    lightgbm: LightGBMParameters,
    evaluation: TemporalEvaluationParameters,
    script_path: Path | None,
    training_as_of_date: date,
) -> Mapping[str, object]:
    """Construye la identidad completa del run antes de entrenar."""

    extras = (
        (script_path,)
        if script_path is not None and script_path.is_file()
        else ()
    )
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "feature_source": {
            "manifest_path": feature_source.path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "manifest_sha256": sha256_file(feature_source.path),
            "fingerprint": feature_source.fingerprint,
            "schema_version": feature_source.schema_version,
            "source_commit": feature_source.source_commit,
            "historical_odds_available": (
                feature_source.historical_odds_available
            ),
            "source_date_policy": feature_source.source_date_policy.as_dict(),
            "datasets": [
                {
                    "gender": item.gender,
                    "rows": item.rows,
                    "min_date": item.min_date.isoformat(),
                    "max_date": item.max_date.isoformat(),
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in metadata
            ],
        },
        "feature_contract": {
            "profile": profile,
            "columns": list(contract.columns_for(profile)),  # type: ignore[arg-type]
        },
        "parameters": {
            "training_as_of_date": training_as_of_date.isoformat(),
            "logistic": dict(logistic.as_dict()),
            "lightgbm": dict(lightgbm.as_dict()),
            "temporal_evaluation": dict(evaluation.as_dict()),
            "calibration": {
                "method": "Platt",
                "fold_fit_block": "Y-1",
                "fold_test_block": "Y",
                "final_fit_source": "temporal OOF test predictions",
            },
        },
        "runtime_versions": dict(runtime_versions()),
        "code": list(modeling_source_inventory(extra_paths=extras)),
    }


def _safe_rmtree_staging(staging: Path, output_dir: Path) -> None:
    """Elimina únicamente un staging exacto que no llegó a publicarse."""

    resolved = staging.resolve()
    output = output_dir.resolve()
    if (
        resolved.exists()
        and resolved.parent == output
        and resolved.name.startswith(".staging-")
    ):
        shutil.rmtree(resolved)


def _write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    """Escribe una tabla CSV UTF-8 dentro del staging."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _json_value(value: object) -> object:
    """Normaliza escalares NumPy/Pandas y ausencias para JSON estricto."""

    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_records(frame: pd.DataFrame) -> list[Mapping[str, object]]:
    """Convierte un DataFrame pequeño a registros JSON sin NaN."""

    return [
        {
            str(column): _json_value(value)
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _serialize_gender_models(
    models: GenderFinalModels,
    *,
    staging: Path,
    feature_source: FeatureSourceManifest,
) -> Mapping[str, object]:
    """Guarda principal, baseline, calibradores y bundle de inferencia."""

    gender_dir = staging / "models" / models.gender
    paths = {
        "lightgbm": gender_dir / "lightgbm.joblib",
        "lightgbm_calibrator": gender_dir / "lightgbm_platt.joblib",
        "logistic": gender_dir / "logistic.joblib",
        "logistic_calibrator": gender_dir / "logistic_platt.joblib",
        "ranking_probability": (
            gender_dir / "ranking_probability.joblib"
        ),
        "deployment_bundle": gender_dir / "deployment_bundle.joblib",
    }
    dump_joblib_artifact(models.lightgbm, paths["lightgbm"])
    dump_joblib_artifact(
        models.lightgbm_calibrator, paths["lightgbm_calibrator"]
    )
    dump_joblib_artifact(models.logistic, paths["logistic"])
    dump_joblib_artifact(
        models.logistic_calibrator, paths["logistic_calibrator"]
    )
    dump_joblib_artifact(
        models.ranking_probability, paths["ranking_probability"]
    )
    deployment_bundle = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "gender": models.gender,
        "estimator": models.lightgbm,
        "calibrator": models.lightgbm_calibrator,
        "training_rows": models.training_rows,
        "training_source_rows": models.training_source_rows,
        "training_excluded_unavailable": (
            models.training_excluded_unavailable
        ),
        "training_max_date": models.training_max_date.isoformat(),
        "training_available_max_date": (
            models.training_available_max_date.isoformat()
        ),
        "training_as_of_date": models.training_as_of_date.isoformat(),
        "source_date_policy": feature_source.source_date_policy.as_dict(),
        "calibration_source": "temporal OOF predictions",
        "calibration_rows": models.calibration_rows,
    }
    dump_joblib_artifact(
        deployment_bundle, paths["deployment_bundle"]
    )
    return {
        "gender": models.gender,
        "training_rows": models.training_rows,
        "training_source_rows": models.training_source_rows,
        "training_excluded_unavailable": (
            models.training_excluded_unavailable
        ),
        "training_max_date": models.training_max_date.isoformat(),
        "training_available_max_date": (
            models.training_available_max_date.isoformat()
        ),
        "training_as_of_date": models.training_as_of_date.isoformat(),
        "calibration_rows": models.calibration_rows,
        "primary_model": "lightgbm",
        "calibration": "Platt sobre predicciones OOF temporales",
        "paths": {
            key: path.relative_to(staging).as_posix()
            for key, path in paths.items()
        },
    }


def _gender_summary(
    *,
    metadata: TrainingDatasetMetadata,
    evaluation: GenderEvaluationResult,
    fold_count: int,
    models: GenderFinalModels,
) -> Mapping[str, object]:
    """Resume volumen final, OOF y cobertura de mercado."""

    return {
        "gender": metadata.gender,
        "training_rows": models.training_rows,
        "training_source_rows": models.training_source_rows,
        "training_excluded_unavailable": (
            models.training_excluded_unavailable
        ),
        "training_min_date": metadata.min_date.isoformat(),
        "training_max_date": models.training_max_date.isoformat(),
        "training_available_max_date": (
            models.training_available_max_date.isoformat()
        ),
        "training_as_of_date": models.training_as_of_date.isoformat(),
        "evaluation_rows": int(
            evaluation.market_audit.total_rows
        ),
        "folds": fold_count,
        "market_rows": evaluation.market_audit.market_rows,
        "market_coverage": evaluation.market_audit.coverage,
        "market_status": evaluation.market_audit.status,
        "suspicious_segments": len(evaluation.suspicious_segments),
    }


def _copy_atomic(source: Path, target: Path) -> None:
    """Copia un artefacto a una ruta activa mediante reemplazo atómico."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _sync_active_documentation(published: PublishedRun) -> None:
    """Actualiza informe y figuras de docs desde un run ya verificado."""

    run_dir = published.run_dir
    report_source = run_dir / "report" / "model_report.md"
    if not report_source.is_file():
        raise TrainingError(
            f"El run no contiene su informe: {report_source}."
        )
    _copy_atomic(report_source, DOCS_DIR / "model_report.md")
    for gender in ("M", "F"):
        plot_source = (
            run_dir / "evaluation" / f"calibration_{gender}.png"
        )
        if not plot_source.is_file():
            raise TrainingError(
                f"El run no contiene la curva {plot_source}."
            )
        _copy_atomic(
            plot_source,
            DOCS_DIR / "assets" / f"model_calibration_{gender}.png",
        )


def _load_run_tables(
    published: PublishedRun,
) -> tuple[
    tuple[Mapping[str, object], ...],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Recupera resúmenes y tablas pequeñas de un run verificado."""

    summaries = published.manifest.get("gender_summaries")
    if not isinstance(summaries, list) or not all(
        isinstance(item, Mapping) for item in summaries
    ):
        raise TrainingError("El run no contiene gender_summaries válidos.")
    evaluation_dir = published.run_dir / "evaluation"
    return (
        tuple(summaries),
        pd.read_csv(evaluation_dir / "metrics.csv"),
        pd.read_csv(evaluation_dir / "market_coverage.csv"),
        pd.read_csv(evaluation_dir / "suspicious_segments.csv"),
    )


def retrain_models(
    *,
    manifest_path: Path = FEATURE_DATASET_MANIFEST_PATH,
    output_dir: Path = PHASE7_MODELS_DIR,
    profile: FeatureProfile = "auto",
    logistic_parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS,
    lightgbm_parameters: LightGBMParameters = (
        DEFAULT_LIGHTGBM_PARAMETERS
    ),
    evaluation_parameters: TemporalEvaluationParameters = (
        DEFAULT_TEMPORAL_EVALUATION_PARAMETERS
    ),
    script_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
    training_as_of_date: date | None = None,
) -> ModelTrainingRun:
    """Ejecuta o reutiliza el reentreno canónico de ambos géneros.

    Args:
        manifest_path: Manifiesto verificado de features.
        output_dir: Raíz versionada de artefactos dentro de ``TENNIS/``.
        profile: Perfil de features explícito o ``auto`` contractual.
        logistic_parameters: Hiperparámetros de la baseline logística.
        lightgbm_parameters: Hiperparámetros del modelo principal.
        evaluation_parameters: Temporadas y umbrales del backtest.
        script_path: Script que debe entrar en el fingerprint del código.
        progress: Callback opcional de progreso.
        training_as_of_date: Corte civil del reentreno final; por defecto hoy
            UTC. Solo se ajusta con ``result_available_date <`` ese corte.

    Returns:
        Run publicado/reutilizado y sus tablas principales.
    """

    resolved_training_cutoff = (
        datetime.now(UTC).date()
        if training_as_of_date is None
        else training_as_of_date
    )
    if (
        isinstance(resolved_training_cutoff, datetime)
        or not isinstance(resolved_training_cutoff, date)
    ):
        raise TrainingError("training_as_of_date debe ser date estricto.")
    feature_source = load_feature_source_manifest(manifest_path)
    contract = load_feature_contract(feature_source.raw_payload)
    resolved_profile = contract.resolve_profile(profile)
    metadata = tuple(
        verify_training_dataset(feature_source, gender)
        for gender in ("M", "F")
    )
    identity_payload = _parameter_payload(
        feature_source=feature_source,
        metadata=metadata,
        contract=contract,
        profile=resolved_profile,
        logistic=logistic_parameters,
        lightgbm=lightgbm_parameters,
        evaluation=evaluation_parameters,
        script_path=script_path,
        training_as_of_date=resolved_training_cutoff,
    )
    fingerprint = fingerprint_payload(identity_payload)
    canonical_output = (
        Path(output_dir).resolve() == PHASE7_MODELS_DIR.resolve()
    )
    existing = find_verified_run(fingerprint, output_dir=output_dir)
    if existing is not None:
        if canonical_output:
            existing = activate_published_run(
                existing,
                output_dir=output_dir,
            )
            _sync_active_documentation(existing)
        summaries, metrics, market, suspicious = _load_run_tables(existing)
        return ModelTrainingRun(
            published=existing,
            summaries=summaries,
            metrics=metrics,
            market_audits=market,
            suspicious_segments=suspicious,
        )

    staging = create_staging_directory(output_dir)
    all_metrics: list[pd.DataFrame] = []
    all_reliability: list[pd.DataFrame] = []
    all_suspicious: list[pd.DataFrame] = []
    market_rows: list[Mapping[str, object]] = []
    fold_rows: list[Mapping[str, object]] = []
    summaries: list[Mapping[str, object]] = []
    model_summaries: list[Mapping[str, object]] = []
    try:
        for dataset_metadata in metadata:
            gender = dataset_metadata.gender
            if progress is not None:
                progress(f"{gender}: carga y verificación del dataset")
            loaded = load_training_dataset(
                gender,
                feature_columns=contract.columns_for(resolved_profile),
                manifest_path=manifest_path,
            )
            available_dates = pd.to_datetime(
                loaded.frame["result_available_date"],
                errors="raise",
            ).dt.date
            causal_mask = available_dates.map(
                lambda value: value < resolved_training_cutoff
            )
            causal_frame = loaded.frame.loc[causal_mask].copy()
            if causal_frame.empty:
                raise TrainingError(
                    f"{gender}: el corte no deja resultados disponibles."
                )
            if not (
                pd.to_datetime(
                    causal_frame["result_available_date"], errors="raise"
                ).dt.date
                < resolved_training_cutoff
            ).all():
                raise TrainingError(
                    f"{gender}: el subset causal no reconcilia con el corte."
                )
            backtest = run_gender_backtest(
                causal_frame,
                gender=gender,
                contract=contract,
                profile=resolved_profile,
                evaluation_parameters=evaluation_parameters,
                logistic_parameters=logistic_parameters,
                lightgbm_parameters=lightgbm_parameters,
                progress=progress,
            )
            evaluation = evaluate_gender_backtest(
                backtest, parameters=evaluation_parameters
            )
            final_models = fit_final_gender_models(
                causal_frame,
                gender=gender,
                contract=contract,
                backtest=backtest,
                profile=resolved_profile,
                logistic_parameters=logistic_parameters,
                lightgbm_parameters=lightgbm_parameters,
                progress=progress,
                training_as_of_date=resolved_training_cutoff,
                source_training_rows=len(loaded.frame),
            )
            model_summaries.append(
                _serialize_gender_models(
                    final_models,
                    staging=staging,
                    feature_source=feature_source,
                )
            )
            evaluation_dir = staging / "evaluation"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            backtest.predictions.to_parquet(
                evaluation_dir / f"oof_{gender}.parquet",
                index=False,
                compression="zstd",
            )
            save_reliability_plot(
                evaluation.reliability,
                gender=gender,
                output_path=(
                    evaluation_dir / f"calibration_{gender}.png"
                ),
            )
            all_metrics.append(evaluation.metrics)
            all_reliability.append(evaluation.reliability)
            all_suspicious.append(evaluation.suspicious_segments)
            market_rows.append(evaluation.market_audit.as_dict())
            fold_rows.extend(fold.as_dict() for fold in backtest.folds)
            summaries.append(
                _gender_summary(
                    metadata=dataset_metadata,
                    evaluation=evaluation,
                    fold_count=len(backtest.folds),
                    models=final_models,
                )
            )
            del loaded, causal_frame, backtest, evaluation, final_models

        metrics = pd.concat(all_metrics, ignore_index=True)
        reliability = pd.concat(all_reliability, ignore_index=True)
        nonempty_suspicious = [
            frame for frame in all_suspicious if not frame.empty
        ]
        suspicious = (
            pd.concat(nonempty_suspicious, ignore_index=True)
            if nonempty_suspicious
            else pd.DataFrame(
                columns=all_suspicious[0].columns
                if all_suspicious
                else ()
            )
        )
        market = pd.DataFrame(market_rows)
        folds = pd.DataFrame(fold_rows)
        evaluation_dir = staging / "evaluation"
        _write_dataframe_csv(metrics, evaluation_dir / "metrics.csv")
        _write_dataframe_csv(
            reliability, evaluation_dir / "reliability_bins.csv"
        )
        _write_dataframe_csv(
            suspicious, evaluation_dir / "suspicious_segments.csv"
        )
        _write_dataframe_csv(
            market, evaluation_dir / "market_coverage.csv"
        )
        _write_dataframe_csv(folds, evaluation_dir / "folds.csv")

        run_relative_from_docs = (
            Path("..")
            / "models"
            / "phase7"
            / "runs"
            / fingerprint
        ).as_posix()
        report = render_model_report(
            fingerprint=fingerprint,
            feature_fingerprint=feature_source.fingerprint,
            source_commit=feature_source.source_commit,
            profile=resolved_profile,
            evaluation_first_year=(
                evaluation_parameters.first_test_season
            ),
            evaluation_last_year=(
                evaluation_parameters.last_test_season
            ),
            gender_summaries=summaries,
            metrics=metrics,
            suspicious=suspicious,
            market_audits=market,
            run_relative_path=run_relative_from_docs,
        )
        write_model_report(
            report, staging / "report" / "model_report.md"
        )

        global_metrics = metrics.loc[
            (metrics["population"] == "native")
            & (metrics["scope"] == "global")
        ]
        manifest_payload: Mapping[str, object] = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "identity": identity_payload,
            "code_inventory": identity_payload["code"],
            "feature_fingerprint": feature_source.fingerprint,
            "source_commit": feature_source.source_commit,
            "feature_profile": resolved_profile,
            "historical_odds_available": (
                feature_source.historical_odds_available
            ),
            "source_date_policy": feature_source.source_date_policy.as_dict(),
            "training_as_of_date": resolved_training_cutoff.isoformat(),
            "gender_summaries": list(summaries),
            "models": model_summaries,
            "global_metrics": _json_records(global_metrics),
            "market_coverage": _json_records(market),
            "suspicious_segment_count": len(suspicious),
        }
        published = publish_staged_run(
            staging,
            fingerprint=fingerprint,
            manifest_payload=manifest_payload,
            output_dir=output_dir,
        )
        if canonical_output:
            _sync_active_documentation(published)
        return ModelTrainingRun(
            published=published,
            summaries=tuple(summaries),
            metrics=metrics,
            market_audits=market,
            suspicious_segments=suspicious,
        )
    except Exception:
        _safe_rmtree_staging(staging, output_dir)
        raise

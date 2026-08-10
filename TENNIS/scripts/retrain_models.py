"""Reentrena y valida temporalmente los modelos de tenis de la fase 7.

Qué hace:
    Verifica los Parquet causales de fase 6, ejecuta folds expansivos por
    temporada, calibra con Platt usando únicamente el bloque anterior, calcula
    métricas globales/segmentadas y publica modelos versionados por género.

Qué recibe:
    Rutas dentro de ``TENNIS/``, perfil de features, temporadas de evaluación
    y número de hilos de LightGBM. Si se omite la última temporada, usa la
    última temporada cerrada disponible para ambos géneros.

Cómo se ejecuta, desde ``TENNIS/``:
    ``python scripts/retrain_models.py``
    ``python scripts/retrain_models.py --first-test-season 2016
    --last-test-season 2025 --n-jobs 4``
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FEATURE_DATASET_MANIFEST_PATH,
    PHASE7_MODELS_DIR,
)
from src.modeling.parameters import (  # noqa: E402
    DEFAULT_LIGHTGBM_PARAMETERS,
    TemporalEvaluationParameters,
)
from src.modeling.data import (  # noqa: E402
    ModelDataError,
    load_feature_source_manifest,
)
from src.modeling.training import (  # noqa: E402
    ModelTrainingRun,
    retrain_models,
)


def build_parser() -> argparse.ArgumentParser:
    """Construye una CLI explícita para el reentreno canónico."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena, calibra y valida temporalmente un modelo por género."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FEATURE_DATASET_MANIFEST_PATH,
        help="Manifiesto de features publicado por la fase 6.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHASE7_MODELS_DIR,
        help="Directorio versionado de modelos dentro de TENNIS/.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=("auto", "sports_only", "market_enhanced"),
        default="auto",
        help=(
            "auto respeta historical_odds_available; el histórico actual "
            "resuelve a sports_only."
        ),
    )
    parser.add_argument(
        "--first-test-season",
        type=int,
        default=2016,
        help="Primera temporada de test del backtest expansivo.",
    )
    parser.add_argument(
        "--last-test-season",
        type=int,
        default=None,
        help=(
            "Última temporada de test; por defecto, la última cerrada y "
            "disponible para ambos géneros."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Hilos de LightGBM; -1 usa los disponibles.",
    )
    parser.add_argument(
        "--training-as-of-date",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Corte causal del reentreno; por defecto hoy UTC. Solo usa "
            "resultados con result_available_date estrictamente anterior."
        ),
    )
    return parser


def infer_last_complete_season(
    manifest_path: Path,
    training_as_of_date: date,
) -> int:
    """Obtiene la última temporada cerrada común sin usar filas futuras."""

    try:
        source = load_feature_source_manifest(manifest_path)
        payload = source.raw_payload
    except ModelDataError as exc:
        raise ValueError(
            f"No se pudo leer el manifiesto {manifest_path}."
        ) from exc
    datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    if not isinstance(datasets, list):
        raise ValueError("El manifiesto no contiene datasets.")
    max_years: list[int] = []
    for gender in ("M", "F"):
        rows = [
            item
            for item in datasets
            if isinstance(item, Mapping) and item.get("gender") == gender
        ]
        if len(rows) != 1:
            raise ValueError(f"No hay un único dataset para {gender}.")
        value = rows[0].get("max_date")
        if not isinstance(value, str):
            raise ValueError(f"max_date inválida para {gender}.")
        try:
            max_years.append(date.fromisoformat(value).year)
        except ValueError as exc:
            raise ValueError(f"max_date inválida para {gender}.") from exc
    latest_admissible_source_date = training_as_of_date - timedelta(
        days=source.source_date_policy.result_embargo_days + 1
    )
    last_fully_available_year = (
        latest_admissible_source_date.year
        if (
            latest_admissible_source_date.month,
            latest_admissible_source_date.day,
        )
        == (12, 31)
        else latest_admissible_source_date.year - 1
    )
    return min(min(max_years), last_fully_available_year)


def _print_report(run: ModelTrainingRun) -> None:
    """Muestra identidad, tamaños y métricas principales."""

    action = "reutilizado" if run.skipped else "entrenado"
    print(
        f"Run {action}: fingerprint={run.published.fingerprint}"
    )
    print(f"Ruta: {run.published.run_dir}")
    print("gender\ttraining_rows\tevaluation_rows\tfolds\tmarket_status")
    for summary in run.summaries:
        print(
            f"{summary['gender']}\t{summary['training_rows']}\t"
            f"{summary['evaluation_rows']}\t{summary['folds']}\t"
            f"{summary['market_status']}"
        )
    global_principal = run.metrics.loc[
        (run.metrics["population"] == "native")
        & (run.metrics["scope"] == "global")
        & (run.metrics["model"] == "lightgbm_platt")
    ]
    print("gender\taccuracy\tlog_loss\tbrier\tauc")
    for _, row in global_principal.iterrows():
        print(
            f"{row['gender']}\t{row['accuracy']:.6f}\t"
            f"{row['log_loss']:.6f}\t{row['brier']:.6f}\t"
            f"{row['auc']:.6f}"
        )
    print(
        "Segmentos con accuracy > 85 %: "
        f"{len(run.suspicious_segments)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el reentreno completo y devuelve cero tras verificarlo."""

    args = build_parser().parse_args(argv)
    training_cutoff = (
        datetime.now(UTC).date()
        if args.training_as_of_date is None
        else args.training_as_of_date
    )
    last_season = (
        args.last_test_season
        if args.last_test_season is not None
        else infer_last_complete_season(args.manifest, training_cutoff)
    )
    evaluation = TemporalEvaluationParameters(
        first_test_season=args.first_test_season,
        last_test_season=last_season,
    )
    lightgbm = replace(
        DEFAULT_LIGHTGBM_PARAMETERS, n_jobs=args.n_jobs
    )
    run = retrain_models(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        profile=args.feature_profile,
        lightgbm_parameters=lightgbm,
        evaluation_parameters=evaluation,
        script_path=Path(__file__).resolve(),
        progress=lambda message: print(message, flush=True),
        training_as_of_date=training_cutoff,
    )
    _print_report(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

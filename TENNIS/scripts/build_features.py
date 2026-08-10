"""Construye los datasets históricos causales de la fase 6.

Qué hace:
    Verifica el manifiesto Sackmann, carga rankings y jugadores, recorre todos
    los partidos por bloques completos de fecha y publica un Parquet separado
    por género junto con su manifiesto y la cuarentena de rankings.

Qué recibe:
    Opcionalmente género, rutas dentro de ``TENNIS/``, tamaños de chunk/buffer
    y ``--force`` para recomprobar un fingerprint ya publicado sin
    sobrescribir su run inmutable.

Cómo se ejecuta, desde ``TENNIS/``:
    ``python scripts/build_features.py``
    ``python scripts/build_features.py --gender M --output-dir
    data/processed/features_active/diagnostic_M --force``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FEATURES_PROCESSED_DIR,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from src.features import (  # noqa: E402
    FeatureDatasetBuildReport,
    build_training_datasets,
)
from src.elo.build import load_verified_manifest  # noqa: E402
from src.identity_integrity import load_identity_quarantine  # noqa: E402
from src.temporal import (  # noqa: E402
    DEFAULT_RESULT_EMBARGO_DAYS,
    SourceDatePolicy,
)


def build_parser() -> argparse.ArgumentParser:
    """Define una CLI explícita sin modificar fuentes automáticamente."""

    parser = argparse.ArgumentParser(
        description=(
            "Construye datasets causales por género para la fase 6."
        )
    )
    parser.add_argument(
        "--gender",
        choices=("M", "F", "all"),
        default="all",
        help=(
            "Universo a construir; una selección parcial exige un "
            "--output-dir distinto del canónico."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SACKMANN_MANIFEST_PATH,
        help="Manifiesto Sackmann activo.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raíz de los datos crudos verificados.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FEATURES_PROCESSED_DIR,
        help="Directorio procesado donde publicar Parquet y manifiesto.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Filas máximas por lectura de CSV/SQLite.",
    )
    parser.add_argument(
        "--parquet-buffer-rows",
        type=int,
        default=10_000,
        help="Filas por buffer antes de escribir un row group.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruye aunque el fingerprint activo coincida.",
    )
    parser.add_argument(
        "--result-embargo-days",
        type=int,
        default=DEFAULT_RESULT_EMBARGO_DAYS,
        help="Días causales desde tourney_date; igualdad aún se excluye.",
    )
    return parser


def _print_report(report: FeatureDatasetBuildReport) -> None:
    """Muestra rutas, tamaños, balance y auditoría de rankings."""

    action = "reutilizado" if report.skipped else "construido"
    print(
        f"Dataset {action}: schema={report.schema_version}, "
        f"commit={report.source_commit}, "
        f"fingerprint={report.fingerprint}"
    )
    print(
        "gender\trows\ty_rate\tmin_date\tmax_date\texcluded\t"
        "size_bytes\tpath"
    )
    for audit in report.datasets:
        print(
            f"{audit.gender}\t{audit.training_rows}\t"
            f"{audit.y_rate:.6f}\t{audit.min_date}\t{audit.max_date}\t"
            f"{audit.excluded_rows}\t{audit.output_size}\t"
            f"{audit.output_path}"
        )
    print(
        "gender\tranking_rows\tindexed\tduplicates\t"
        "conflict_keys\tconflict_observations"
    )
    for ranking in report.rankings:
        print(
            f"{ranking.gender}\t{ranking.source_rows}\t"
            f"{ranking.indexed_rows}\t{ranking.exact_duplicates}\t"
            f"{ranking.conflict_keys}\t"
            f"{ranking.conflict_observations}"
        )
    print(
        "Conflictos de ranking: "
        f"{report.conflict_inventory_rows} filas en "
        f"{report.conflict_inventory_path}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la construcción y devuelve cero tras publicación válida."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    verified = load_verified_manifest(args.manifest, args.raw_dir)
    quarantine = load_identity_quarantine(
        expected_source_commit=verified.source_commit
    )
    report = build_training_datasets(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        gender=args.gender,
        chunksize=args.chunksize,
        parquet_buffer_rows=args.parquet_buffer_rows,
        force=args.force,
        identity_exclusion_after_dates=quarantine.exclusion_after_dates,
        source_date_policy=SourceDatePolicy(args.result_embargo_days),
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Descarga y valida el histórico de singles de Jeff Sackmann.

Qué hace:
    Descubre el snapshot autorizado, descarga a ``data/raw`` los CSV de ATP y
    WTA, valida jugadores y partidos con los loaders y muestra su cobertura.

Qué recibe:
    ``--force`` repara archivos locales que divergen del manifiesto.
    ``--download-only`` omite la carga completa y su resumen en memoria.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/ingest_sackmann.py``
    ``python scripts/ingest_sackmann.py --force``
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders import (  # noqa: E402
    audit_match_file_coverage,
    load_matches,
    load_players,
    summarize_match_coverage,
)
from src.sackmann_download import download_sackmann_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos del punto de entrada."""

    parser = argparse.ArgumentParser(
        description=(
            "Descarga idempotente y validación del histórico ATP/WTA de Sackmann."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Autoriza reparar archivos locales divergentes; los blobs "
            "correctos siguen sin descargarse."
        ),
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Descarga los CSV sin construir los DataFrames completos.",
    )
    return parser


def _print_dataframe(title: str, frame: object) -> None:
    """Imprime un título y la representación tabular de un DataFrame."""

    print(f"\n{title}")
    print(frame.to_string(index=False))  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la descarga y, salvo indicación, valida loaders y cobertura."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    report = download_sackmann_data(force=args.force)
    print(f"\nCommit fuente: {report.source_commit}")
    print(f"Archivos descargados: {report.downloaded_count}")
    print(f"Archivos conservados: {report.skipped_count}")
    print(f"Manifiesto activo: {report.manifest_path}")
    print(f"Manifiesto versionado: {report.versioned_manifest_path}")

    file_coverage = audit_match_file_coverage()
    _print_dataframe("Cobertura de archivos anuales", file_coverage)

    if args.download_only:
        return 0

    players = load_players()
    player_counts = (
        players.groupby("gender", dropna=False)
        .size()
        .rename("players")
        .reset_index()
    )
    _print_dataframe("Jugadores por género", player_counts)

    summary_frames: list[pd.DataFrame] = []
    total_matches = 0
    total_source_anomalies = 0
    for gender in ("M", "F"):
        gender_matches = load_matches(gender)
        total_matches += len(gender_matches)
        total_source_anomalies += int(
            gender_matches["source_anomaly"].notna().sum()
        )
        summary_frames.append(summarize_match_coverage(gender_matches))
        del gender_matches
        gc.collect()

    summary = pd.concat(summary_frames, ignore_index=True)
    summary = summary.sort_values(
        ["gender", "tour_level"],
        kind="stable",
    ).reset_index(drop=True)
    printable_summary = summary.copy()
    printable_summary["min_date"] = printable_summary["min_date"].dt.strftime(
        "%Y-%m-%d"
    )
    printable_summary["max_date"] = printable_summary["max_date"].dt.strftime(
        "%Y-%m-%d"
    )
    _print_dataframe("Partidos por género y tour_level", printable_summary)
    print(f"\nPartidos totales: {total_matches}")
    print(f"Anomalías fuente normalizadas: {total_source_anomalies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

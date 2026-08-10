"""Actualiza las tres fuentes crudas aprobadas para el predictor.

Qué hace:
    Sincroniza el mirror histórico Sackmann y Match Charting Project por commit,
    y consulta secuencialmente los dos informes Elo públicos de Tennis
    Abstract. La frecuencia la decide el operador; el cliente conserva
    peticiones condicionales y un cortacircuitos de 24 horas tras un fallo.

Qué recibe:
    ``--force`` autoriza reparar divergencias Sackmann y revalida snapshots Elo
    locales; no desactiva las defensas ante fallos. Los tres ``--skip-*``
    permiten omitir una fuente concreta durante una ejecución de diagnóstico.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/update_sources.py``
    ``python scripts/update_sources.py --force``
    ``python scripts/update_sources.py --skip-tennis-abstract``
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

from src.match_charting_download import (  # noqa: E402
    DownloadReport as MatchChartingDownloadReport,
    MatchChartingDownloadError,
    download_match_charting_data,
)
from src.sackmann_download import (  # noqa: E402
    DownloadReport as SackmannDownloadReport,
    SackmannDownloadError,
    download_sackmann_data,
)
from src.tennis_abstract_elo import (  # noqa: E402
    EloFetchReport,
    TennisAbstractEloError,
    download_tennis_abstract_elo,
    load_latest_elo,
)


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construye los argumentos del actualizador coordinado."""

    parser = argparse.ArgumentParser(
        description=(
            "Actualiza las fuentes históricas, auxiliares y Elo respetando "
            "versionado, integridad y límites HTTP."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Repara divergencias Sackmann y revalida Elo local; no desactiva "
            "el cortacircuitos posterior a fallos."
        ),
    )
    parser.add_argument(
        "--skip-sackmann",
        action="store_true",
        help="Omite el mirror histórico Sackmann.",
    )
    parser.add_argument(
        "--skip-match-charting",
        action="store_true",
        help="Omite Match Charting Project.",
    )
    parser.add_argument(
        "--skip-tennis-abstract",
        action="store_true",
        help="Omite los dos informes Elo de Tennis Abstract.",
    )
    return parser


def _print_sackmann_report(report: SackmannDownloadReport) -> None:
    """Muestra el commit y los contadores del mirror histórico."""

    print("\nSackmann histórico")
    print(f"Commit: {report.source_commit}")
    print(f"Descargados/actualizados: {report.downloaded_count}")
    print(f"Conservados: {report.skipped_count}")
    print(f"Manifiesto: {report.manifest_path}")
    print(f"Snapshot de manifiesto: {report.versioned_manifest_path}")


def _print_match_charting_report(
    report: MatchChartingDownloadReport,
) -> None:
    """Muestra el commit y cambios de Match Charting Project."""

    print("\nMatch Charting Project")
    print(f"Commit: {report.source_commit}")
    print(f"Nuevos: {report.downloaded_count}")
    print(f"Actualizados: {report.updated_count}")
    print(f"Conservados: {report.skipped_count}")
    print(f"Manifiesto: {report.active_manifest_path}")


def _print_elo_report(report: EloFetchReport) -> None:
    """Muestra solicitudes, resultados y cobertura de snapshots Elo válidos."""

    print("\nTennis Abstract Elo")
    print(f"GET consumidos: {report.get_count}")
    for result in report.results:
        retry_after = (
            result.retry_after_utc.isoformat()
            if result.retry_after_utc is not None
            else "-"
        )
        print(
            f"{result.gender}: {result.outcome}; "
            f"HTTP={result.status_code}; próximo acceso={retry_after}"
        )
        if result.detail:
            print(f"  Detalle: {result.detail}")

        if result.outcome not in {"downloaded", "not_modified"}:
            continue
        frame = load_latest_elo(result.gender)
        rating_date = frame["rating_date"].iloc[0].strftime("%Y-%m-%d")
        print(f"  Filas válidas: {len(frame)}; fecha Elo: {rating_date}")


def main(argv: Sequence[str] | None = None) -> int:
    """Actualiza las fuentes seleccionadas y devuelve cero solo si son válidas."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        if not args.skip_sackmann:
            _print_sackmann_report(
                download_sackmann_data(force=args.force)
            )

        if not args.skip_match_charting:
            _print_match_charting_report(download_match_charting_data())

        if not args.skip_tennis_abstract:
            elo_report = download_tennis_abstract_elo(force=args.force)
            _print_elo_report(elo_report)
            if elo_report.failed_count:
                LOGGER.error(
                    "La actualización Elo conservó el fallo y no activó "
                    "el snapshot afectado."
                )
                return 1
    except (
        SackmannDownloadError,
        MatchChartingDownloadError,
        TennisAbstractEloError,
    ) as exc:
        LOGGER.error("Actualización detenida: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

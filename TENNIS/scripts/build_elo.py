"""Construye y valida el histórico Elo causal de ATP/WTA.

Qué hace:
    Verifica el manifiesto Sackmann y cada blob CSV, calcula dos universos Elo
    independientes por bloques de ``tourney_date`` y publica una base SQLite
    únicamente después de completar todas las validaciones.

Qué recibe:
    ``--gender`` selecciona ``M``, ``F`` o ambos; ``--force`` obliga a
    reconstruir un fingerprint ya activo. ``--result-embargo-days`` configura
    el retraso causal desde ``tourney_date``. Las rutas y el tamaño de chunk
    se pueden sobrescribir para tests o diagnósticos.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/build_elo.py``
    ``python scripts/build_elo.py --force``
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import logging
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ELO_HANDOFF_CONFIG_PATH,
    ELO_DATABASE_PATH,
    ELO_OPERATIONAL_OMISSIONS_PATH,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
    SURFACE_CATALOG_PATH,
)
from src.data_loaders import load_players  # noqa: E402
from src.elo.build import (  # noqa: E402
    DEFAULT_CHUNKSIZE,
    EloBuildReport,
    build_elo_database,
    load_verified_manifest,
)
from src.elo.operational import (  # noqa: E402
    build_operational_elo_overlay_with_fallback,
    load_elo_handoff_config,
    publish_operational_omissions,
)
from src.identity_integrity import load_identity_quarantine  # noqa: E402
from src.surface_catalog import build_surface_catalog  # noqa: E402
from src.temporal import (  # noqa: E402
    DEFAULT_RESULT_EMBARGO_DAYS,
    SourceDatePolicy,
)


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser documentado del punto de entrada Elo."""

    parser = argparse.ArgumentParser(
        description=("Construye el Elo general y por superficie sin fugas temporales.")
    )
    parser.add_argument(
        "--gender",
        choices=("M", "F", "all"),
        default="all",
        help="Universo que se construirá; por defecto, ambos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruye aunque el fingerprint activo ya coincida.",
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
        help="Raíz local contra la que se resuelven los CSV.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ELO_DATABASE_PATH,
        help="SQLite Elo que se publicará atómicamente.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="Filas máximas leídas por chunk CSV/SQLite.",
    )
    parser.add_argument(
        "--result-embargo-days",
        type=int,
        default=DEFAULT_RESULT_EMBARGO_DAYS,
        help=(
            "Días desde tourney_date hasta availability_date; la igualdad "
            "con el corte sigue excluida."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Corte causal del overlay; por defecto hoy UTC. Un resultado "
            "solo entra si fecha efectiva y disponibilidad son anteriores."
        ),
    )
    parser.add_argument(
        "--handoff-config",
        type=Path,
        default=ELO_HANDOFF_CONFIG_PATH,
        help="Configuración versionada de commit base, corte y precedencia.",
    )
    parser.add_argument(
        "--disable-operational-overlay",
        action="store_true",
        help="Diagnóstico explícito Sackmann-only; nunca es el default.",
    )
    parser.add_argument(
        "--surface-catalog",
        type=Path,
        default=SURFACE_CATALOG_PATH,
        help="Caché fija del catálogo causal torneo+edición -> superficie.",
    )
    return parser


def _load_player_names(raw_dir: Path) -> Mapping[tuple[str, int], str]:
    """Carga nombres maestros para que el top final sea validable por humanos."""

    players = load_players(raw_dir=raw_dir)
    names: dict[tuple[str, int], str] = {}
    for row in players.itertuples(index=False):
        name = row.player_name
        if isinstance(name, str) and name.strip():
            names[(str(row.gender), int(row.player_id))] = name.strip()
    return names


def _print_report(
    report: EloBuildReport,
    player_names: Mapping[tuple[str, int], str],
) -> None:
    """Imprime metadatos, auditoría y top general con nombres comprobables."""

    print(f"\nBase Elo: {report.database_path}")
    print(f"Commit fuente: {report.source_commit}")
    print(f"Versión del algoritmo: {report.algorithm_version}")
    print(f"Sin cambios (idempotente): {'sí' if report.skipped else 'no'}")
    for gender, run_id in report.run_ids.items():
        print(f"Run {gender}: {run_id}")
        print(f"Fingerprint {gender}: {report.input_fingerprints[gender]}")

    for audit in report.audits:
        print(f"\nAuditoría {audit.gender}")
        print(f"Filas fuente: {audit.source_rows}")
        print(f"  Sackmann base: {audit.base_source_rows}")
        print(f"  Overlay operativo: {audit.operational_source_rows}")
        print(f"Partidos incluidos: {audit.included_events}")
        print(f"Partidos excluidos: {audit.excluded_events}")
        print(f"Duplicados exactos: {audit.exact_duplicates}")
        print(f"Bloques de fecha: {audit.date_blocks}")
        print(f"Estados persistidos: {audit.rating_rows}")
        for reason, count in audit.exclusions_by_reason.items():
            print(f"  {reason}: {count}")

    if report.operational_overlay is not None:
        overlay = report.operational_overlay
        print("\nOverlay operativo")
        print(f"Estado: {overlay['status']}")
        print(f"Corte as-of: {overlay['as_of_date']}")
        print(f"Handoff: > {overlay['cutoff_date']}")
        print(f"Eventos seleccionados: {overlay['selected_events']}")
        print(f"Omitidos: {overlay['omitted_records']}")
        print(f"Solapamientos resueltos: {overlay['overlaps_resolved']}")
        print(
            "Superficie fiable: "
            f"{overlay.get('surface_resolved_events', 0)}; "
            f"sin superficie: {overlay.get('surface_missing_events', 0)}"
        )
        print(f"Fingerprint catálogo de superficie: {overlay.get('surface_catalog_fingerprint')}")
        sources = overlay.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                print(
                    f"  {source['source']}: elegibles={source['eligible_records']} "
                    f"mapeados={source['mapped_records']} "
                    f"incorporados={source['selected_events']} "
                    f"omitidos={source['omitted_records']} "
                    f"aliases={source['collapsed_within_source']} "
                    f"precedencia={source['suppressed_by_precedence']}"
                )
        if overlay.get("failure"):
            print(f"AVISO fallback: {overlay['failure']}")

    for gender, rows in report.top_ratings.items():
        print(f"\nTop 10 Elo general {gender}")
        print("player_id\tplayer_name\tgeneral_elo\tmatches\tstate_date")
        for row in rows:
            raw_player_id = row["player_id"]
            raw_general_elo = row["general_elo"]
            if not isinstance(raw_player_id, int):
                raise RuntimeError("El top Elo contiene un player_id inválido.")
            if not isinstance(raw_general_elo, (int, float)):
                raise RuntimeError("El top Elo contiene un general_elo inválido.")
            player_id = raw_player_id
            player_name = player_names.get((gender, player_id))
            if player_name is None:
                raise RuntimeError(
                    "El top Elo contiene un ID sin nombre maestro: "
                    f"gender={gender}, player_id={player_id}."
                )
            print(
                f"{player_id}\t"
                f"{player_name}\t"
                f"{float(raw_general_elo):.3f}\t"
                f"{row['general_matches']}\t"
                f"{row['state_date']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la construcción Elo y devuelve cero tras publicación válida."""

    args = build_parser().parse_args(argv)
    as_of_date = datetime.now(UTC).date() if args.as_of_date is None else args.as_of_date
    player_names = _load_player_names(args.raw_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    overlay = None
    manifest_path = args.manifest
    if not args.disable_operational_overlay:
        handoff = load_elo_handoff_config(
            args.handoff_config,
            project_root=PROJECT_ROOT,
        )
        if args.manifest.resolve() != handoff.base_manifest_path.resolve():
            raise ValueError(
                "--manifest no coincide con el manifiesto base congelado; "
                "un re-baseline exige actualizar elo_handoff.json."
            )
        manifest_path = args.manifest
        surface_catalog = build_surface_catalog(
            as_of_date=as_of_date,
            tennisratio_database_path=handoff.tennisratio_database_path,
            operations_database_path=handoff.operations_database_path,
            cache_path=args.surface_catalog,
        )
        overlay = build_operational_elo_overlay_with_fallback(
            config=handoff,
            as_of_date=as_of_date,
            surface_catalog=surface_catalog,
        )
        if overlay.status == "fallback_sackmann_only":
            logging.warning(
                "Overlay operativo no disponible; se construirá Sackmann-only: %s",
                overlay.failure,
            )
    verified = load_verified_manifest(manifest_path, args.raw_dir)
    quarantine = load_identity_quarantine(expected_source_commit=verified.source_commit)
    report = build_elo_database(
        manifest_path=manifest_path,
        raw_dir=args.raw_dir,
        database_path=args.database,
        gender=args.gender,
        chunksize=args.chunksize,
        force=args.force,
        identity_exclusion_after_dates={},
        source_date_policy=SourceDatePolicy(args.result_embargo_days),
        operational_overlay=overlay,
    )
    if overlay is not None:
        publish_operational_omissions(
            ELO_OPERATIONAL_OMISSIONS_PATH,
            overlay,
        )
    _print_report(
        report,
        player_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

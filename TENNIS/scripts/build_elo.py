"""Construye y valida el histórico Elo causal de ATP/WTA.

Qué hace:
    Verifica el manifiesto Sackmann y cada blob CSV, calcula dos universos Elo
    independientes por bloques de ``tourney_date`` y publica una base SQLite
    únicamente después de completar todas las validaciones.

Qué recibe:
    ``--gender`` selecciona ``M``, ``F`` o ambos; ``--force`` obliga a
    reconstruir un fingerprint ya activo. Las rutas y el tamaño de chunk se
    pueden sobrescribir para tests o diagnósticos.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/build_elo.py``
    ``python scripts/build_elo.py --force``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ELO_DATABASE_PATH,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from src.data_loaders import load_players  # noqa: E402
from src.elo.build import (  # noqa: E402
    DEFAULT_CHUNKSIZE,
    EloBuildReport,
    build_elo_database,
    load_verified_manifest,
)
from src.identity_integrity import load_identity_quarantine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser documentado del punto de entrada Elo."""

    parser = argparse.ArgumentParser(
        description=(
            "Construye el Elo general y por superficie sin fugas temporales."
        )
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
        print(f"Partidos incluidos: {audit.included_events}")
        print(f"Partidos excluidos: {audit.excluded_events}")
        print(f"Duplicados exactos: {audit.exact_duplicates}")
        print(f"Bloques de fecha: {audit.date_blocks}")
        print(f"Estados persistidos: {audit.rating_rows}")
        for reason, count in audit.exclusions_by_reason.items():
            print(f"  {reason}: {count}")

    for gender, rows in report.top_ratings.items():
        print(f"\nTop 10 Elo general {gender}")
        print("player_id\tplayer_name\tgeneral_elo\tmatches\tstate_date")
        for row in rows:
            player_id = int(row["player_id"])
            player_name = player_names.get((gender, player_id))
            if player_name is None:
                raise RuntimeError(
                    "El top Elo contiene un ID sin nombre maestro: "
                    f"gender={gender}, player_id={player_id}."
                )
            print(
                f"{player_id}\t"
                f"{player_name}\t"
                f"{float(row['general_elo']):.3f}\t"
                f"{row['general_matches']}\t"
                f"{row['state_date']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la construcción Elo y devuelve cero tras publicación válida."""

    args = build_parser().parse_args(argv)
    player_names = _load_player_names(args.raw_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    verified = load_verified_manifest(args.manifest, args.raw_dir)
    quarantine = load_identity_quarantine(
        expected_source_commit=verified.source_commit
    )
    report = build_elo_database(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        database_path=args.database,
        gender=args.gender,
        chunksize=args.chunksize,
        force=args.force,
        excluded_player_keys=quarantine.keys,
    )
    _print_report(
        report,
        player_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

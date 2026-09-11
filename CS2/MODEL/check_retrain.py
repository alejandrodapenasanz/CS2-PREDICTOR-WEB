"""Report whether CS2 should retrain after enough newly labelled matches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from cs2model import dataio
from cs2model.artifacts import ARTIFACT_PATH, load_artifact
from cs2model.config import (
    DEFAULT_CONFIG_PATH,
    get_runtime_config,
    load_config,
    set_runtime_config,
)
from cs2model.retrain_policy import DEFAULT_NEW_LABELED_THRESHOLD, decide_retrain


DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_MASTER = ROOT / "PIPELINE" / "master" / "matches.json"
DEFAULT_REGISTRY = ROOT / "MODEL" / "artifacts" / "registry"
DEFAULT_RAW = (
    ROOT
    / "SCRAPER"
    / "hltv-scraper-api"
    / "hltv_scraper"
    / "data"
    / "raw"
    / "history_10000_2026-06-28"
    / "results_all.json"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Configuracion YAML versionada.")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH), help="Artefacto vivo que aporta metadata.date_max.")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Registry read-only usado para no repetir un challenger ya intentado.",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="BBDD viva usada como fuente por defecto.")
    parser.add_argument(
        "--master",
        default=str(DEFAULT_MASTER),
        help="Master diario opcional combinado con --raw, igual que en train.py.",
    )
    parser.add_argument("--raw", default="", help="Compatibilidad: results_all.json; por defecto usa --db.")
    parser.add_argument("--no-cs2-filter", action="store_true")
    parser.add_argument(
        "--threshold",
        type=positive_int,
        default=None,
        help=(
            "Override del numero de nuevos partidos etiquetados; por defecto usa "
            f"promotion.auto_retrain_new_labeled del config ({DEFAULT_NEW_LABELED_THRESHOLD} en produccion)."
        ),
    )
    parser.add_argument("--output", help="Ruta JSON opcional; stdout siempre recibe la misma decision.")
    return parser


def _load_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    master_path = Path(args.master)
    raw_source = args.raw or None
    rows = dataio.load_training_rows(
        raw_source,
        master_path if master_path.exists() else None,
        cs2_only=not args.no_cs2_filter,
        db_path=args.db,
    )
    if not rows and not raw_source and DEFAULT_RAW.exists():
        rows = dataio.load_training_rows(
            DEFAULT_RAW,
            master_path if master_path.exists() else None,
            cs2_only=not args.no_cs2_filter,
            db_path=None,
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_runtime_config(load_config(args.config))
    threshold = (
        args.threshold if args.threshold is not None else get_runtime_config().promotion.auto_retrain_new_labeled
    )
    rows = _load_rows(args)
    artifact = load_artifact(Path(args.artifact))
    decision = decide_retrain(rows, artifact, threshold=threshold, registry=Path(args.registry))
    payload = json.dumps(decision.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

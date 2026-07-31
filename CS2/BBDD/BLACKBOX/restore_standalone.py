#!/usr/bin/env python3
"""Restaurador AUTONOMO de la caja negra CS2 (BLACKBOX).

Este script NO necesita el proyecto CS2-Predictor: solo Python 3 (biblioteca
estandar). Viaja dentro de la propia carpeta BLACKBOX para que el "futuro-yo"
pueda reconstruir la base de datos aunque pierda el codigo del proyecto.

Que hace:
  1. Lee `manifest.json` de esta carpeta.
  2. Verifica el SHA-256 de cada fichero (aborta si algo no cuadra).
  3. Descomprime el SQLite canonico (`data/cs2_blackbox.sqlite.gz`) en la ruta
     de destino que indiques y comprueba su integridad.

Uso (situate en esta carpeta o pasa --blackbox):
    python restore_standalone.py --target ./cs2.db
    python restore_standalone.py --blackbox /ruta/BLACKBOX --target /ruta/cs2.db
    python restore_standalone.py --verify-only

Restauracion 100% manual sin este script (solo con sqlite3):
    gzip -dk data/cs2_blackbox.sql.gz
    sqlite3 cs2_restaurada.db < data/cs2_blackbox.sql
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(blackbox: Path) -> tuple[bool, dict]:
    manifest_path = blackbox / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no existe {manifest_path}", file=sys.stderr)
        return False, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    for rel, meta in manifest.get("files", {}).items():
        p = blackbox / rel
        if not p.exists():
            print(f"ERROR: falta {rel}", file=sys.stderr)
            ok = False
            continue
        actual = sha256_file(p)
        if actual != meta["sha256"]:
            print(f"ERROR: checksum NO coincide en {rel}", file=sys.stderr)
            print(f"  esperado={meta['sha256']}", file=sys.stderr)
            print(f"  actual  ={actual}", file=sys.stderr)
            ok = False
    schema = blackbox / "schema.sql"
    if schema.exists() and sha256_file(schema) != manifest.get("schema_sha256"):
        print("ERROR: schema.sql no coincide con el manifest", file=sys.stderr)
        ok = False
    if ok:
        print(f"OK: verificacion correcta ({len(manifest.get('files', {}))} ficheros, "
              f"{manifest.get('total_source_rows', '?')} filas fuente, "
              f"creado {manifest.get('created_at_utc', '?')}).")
    return ok, manifest


def restore(blackbox: Path, target: Path) -> int:
    ok, manifest = verify(blackbox)
    if not ok:
        print("Abortado: la verificacion fallo.", file=sys.stderr)
        return 1
    canonical = blackbox / manifest["canonical_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".restoring")
    with gzip.open(canonical, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    conn = sqlite3.connect(tmp)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finally:
        conn.close()
    if integrity != "ok":
        tmp.unlink(missing_ok=True)
        print(f"ERROR: la BBDD restaurada no pasa integrity_check ({integrity})", file=sys.stderr)
        return 2
    if target.exists():
        backup = target.with_suffix(target.suffix + ".pre_restore")
        shutil.copy2(target, backup)
        print(f"Aviso: {target} existia; copia previa en {backup}")
    tmp.replace(target)
    print(f"OK: restaurada {target} con {matches} partidos.")
    print("NOTA: las tablas derivadas (ratings_history, match_features) se")
    print("      regeneran al entrenar el modelo; no forman parte de la caja negra.")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Restaurador autonomo de la caja negra CS2.")
    p.add_argument("--blackbox", type=Path, default=here, help="Carpeta BLACKBOX (por defecto: la de este script).")
    p.add_argument("--target", type=Path, default=here / "cs2.db", help="Ruta de salida de la BBDD.")
    p.add_argument("--verify-only", action="store_true", help="Solo verifica; no restaura.")
    args = p.parse_args(argv)
    if args.verify_only:
        ok, _ = verify(args.blackbox)
        return 0 if ok else 1
    return restore(args.blackbox, args.target)


if __name__ == "__main__":
    raise SystemExit(main())

"""BLACKBOX — caja negra portatil de la fuente de verdad de CS2.

Objetivo: poder reconstruir `BBDD/cs2.db` COMPLETA a partir de una unica carpeta
portatil (`BBDD/BLACKBOX/`) que se guarda en USB/nube. En BLACKBOX se guarda SOLO
la fuente de verdad NO recomputable (hechos crudos/canonicos). Lo derivado
(ratings, features, cachEs, logs) se REGENERA; no se guarda.

Cuatro mecanismos (subcomandos):
  * export    lee cs2.db y escribe BLACKBOX (sqlite podado + volcado .sql + manifest).
  * verify    comprueba integridad contra los SHA-256 del manifest.
  * restore   reconstruye cs2.db desde BLACKBOX (idempotente, con guardian).
  * autoheal  si cs2.db falta/vacia/corrupta, restaura desde BLACKBOX antes de seguir.

Sin dependencias externas: solo biblioteca estandar (sqlite3, gzip, hashlib, json).

Formato duradero y no propietario:
  - `data/cs2_blackbox.sqlite.gz` : SQLite (formato de archivo recomendado por LoC)
     con SOLO las tablas fuente-de-verdad; preserva tipos, NULLs y claves.
  - `data/cs2_blackbox.sql.gz`    : volcado SQL de texto (fallback restaurable con
     cualquier herramienta: `sqlite3 cs2.db < dump.sql`).
  - `manifest.json`               : version, timestamp UTC, SHA-256 por fichero,
     filas y hash de contenido por tabla, hash del esquema.
  - `schema.sql`                  : copia del esquema canonico.
  - `README.md` / `restore_standalone.py` : restauracion A MANO sin este proyecto.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # .../CS2/BBDD
CS2_ROOT = ROOT.parent                             # .../CS2
DEFAULT_DB = ROOT / "cs2.db"
DEFAULT_BLACKBOX = ROOT / "BLACKBOX"
DEFAULT_BACKUP_DIR = ROOT / "backups"
SCHEMA_SQL = ROOT / "cs2_prediction_schema.sql"
BUILD_DB = ROOT / "build_db.py"

BLACKBOX_FORMAT_VERSION = 1
TOOL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Clasificacion FUENTE DE VERDAD vs DERIVADO (ver docs/AUDIT.md y el esquema).
# El orden respeta las claves foraneas: padres antes que hijos.
# ---------------------------------------------------------------------------
SOURCE_OF_TRUTH_TABLES: tuple[str, ...] = (
    # Identidad estable
    "teams",
    "players",
    "events",
    # Pertenencia temporal
    "team_rosters",
    # Hechos inmutables
    "matches",
    "maps",
    "veto",
    "match_lineups",
    "prematch_lineup_snapshots",
    "map_player_stats",
    "map_player_side_stats",
    # Mercado / integridad
    "odds",
    "sanctions",
    # Staging raw (JSON capturado; no re-scrapeable identico)
    "raw_results",
    "raw_snapshots",
    "player_stat_snapshots",
    "team_ranking_snapshots",
    "match_analytics_snapshots",
    "match_analytics_map_stats",
    "match_analytics_map_handicap",
    # Auditoria del modelo (congelada, no recomputable)
    "predictions",
)

# Excluidas: se REGENERAN. No entran en BLACKBOX.
DERIVED_EXCLUDED_TABLES: tuple[str, ...] = (
    "ratings_history",   # recomputable: Glicko-2 cronologico desde los hechos
    "match_features",    # recomputable: features point-in-time
    "fetch_state",       # cache operativa del scraper (se rehace sola)
    "ingest_runs",       # log de auditoria de ingest (empieza limpio)
)

DATA_SUBDIR = "data"
SQLITE_NAME = "cs2_blackbox.sqlite"
SQLITE_GZ = SQLITE_NAME + ".gz"
SQLDUMP_NAME = "cs2_blackbox.sql"
SQLDUMP_GZ = SQLDUMP_NAME + ".gz"
MANIFEST_NAME = "manifest.json"
SCHEMA_COPY_NAME = "schema.sql"


# ===========================================================================
# Utilidades
# ===========================================================================
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log(msg: str, *, level: str = "INFO") -> None:
    print(f"[blackbox] {level}: {msg}", flush=True)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _table_row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] or 0)
    except sqlite3.DatabaseError:
        return 0


def table_content_hash(conn: sqlite3.Connection, table: str) -> str:
    """SHA-256 estable del contenido de una tabla (independiente del orden fisico).

    Serializa cada fila como JSON de una lista y ordena por todas las columnas,
    de modo que dos bases con las mismas filas producen el mismo hash aunque el
    rowid o el orden de insercion difieran.
    """
    cols = _column_names(conn, table)
    if not cols:
        return sha256_bytes(b"")
    col_list = ", ".join(f'"{c}"' for c in cols)
    h = hashlib.sha256()
    h.update((",".join(cols) + "\n").encode("utf-8"))  # cabecera: nombres de columna
    for row in conn.execute(f'SELECT {col_list} FROM "{table}" ORDER BY {col_list}'):
        line = json.dumps(list(row), ensure_ascii=False, default=str, sort_keys=True)
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _snapshot_db(src_db: Path, dest_file: Path) -> None:
    """Copia consistente de la BBDD viva usando la API backup de SQLite.

    Lee correctamente el estado confirmado aunque la BBDD este en modo WAL, y
    no modifica el origen. El destino queda como un unico fichero estable.
    """
    if dest_file.exists():
        dest_file.unlink()
    src = sqlite3.connect(src_db)
    dst = sqlite3.connect(dest_file)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")   # un solo fichero, sin -wal/-shm
        dst.commit()
    finally:
        dst.close()
        src.close()


def _gzip_file(src: Path, dest_gz: Path) -> None:
    with src.open("rb") as fin, gzip.open(dest_gz, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout)


def _gunzip_file(src_gz: Path, dest: Path) -> None:
    with gzip.open(src_gz, "rb") as fin, dest.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


# ===========================================================================
# EXPORT
# ===========================================================================
def _build_blackbox_sqlite(snapshot_db: Path, dest_sqlite: Path) -> dict[str, dict]:
    """Crea un SQLite nuevo con SOLO las tablas fuente-de-verdad.

    Copia desde `snapshot_db` (una copia estable, single-file, de cs2.db).
    Devuelve estadisticas por tabla: {tabla: {rows, content_sha256}}.
    """
    if dest_sqlite.exists():
        dest_sqlite.unlink()
    schema_text = SCHEMA_SQL.read_text(encoding="utf-8")

    dest = sqlite3.connect(dest_sqlite)
    try:
        dest.execute("PRAGMA foreign_keys = OFF;")   # copia masiva sin importar el orden
        dest.executescript(schema_text)              # crea TODAS las tablas del esquema canonico
        dest.commit()
        dest.execute("ATTACH DATABASE ? AS src", (str(snapshot_db),))

        src_tables = {
            row[0]
            for row in dest.execute("SELECT name FROM src.sqlite_master WHERE type='table'")
        }
        dest_tables = _table_names(dest)
        stats: dict[str, dict] = {}
        for table in SOURCE_OF_TRUTH_TABLES:
            if table not in dest_tables:
                log(f"tabla {table} no existe en el esquema canonico; se omite", level="WARN")
                continue
            if table not in src_tables:
                log(f"tabla {table} ausente en la BBDD de origen; se copia vacia", level="WARN")
                stats[table] = {"rows": 0, "content_sha256": table_content_hash(dest, table)}
                continue
            dest_cols = _column_names(dest, table)
            src_cols = {row[1] for row in dest.execute(f'PRAGMA src.table_info("{table}")')}
            cols = [c for c in dest_cols if c in src_cols]  # interseccion, tolera drift de esquema
            col_list = ", ".join(f'"{c}"' for c in cols)
            dest.execute(
                f'INSERT INTO "{table}" ({col_list}) SELECT {col_list} FROM src."{table}"'
            )
            stats[table] = {
                "rows": _table_row_count(dest, table),
                "content_sha256": table_content_hash(dest, table),
            }
        dest.commit()
        dest.execute("DETACH DATABASE src")

        # Colapsa cualquier WAL en el fichero principal y valida.
        dest.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dest.execute("PRAGMA journal_mode=DELETE")
        integrity = dest.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"integrity_check del blackbox sqlite fallo: {integrity}")
        dest.commit()
    finally:
        dest.close()
    return stats


def _dump_sql(sqlite_path: Path, dest_sql: Path) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        with dest_sql.open("w", encoding="utf-8") as fh:
            for line in conn.iterdump():
                fh.write(line + "\n")
    finally:
        conn.close()


def export(db_path: Path, blackbox_dir: Path) -> int:
    if not db_path.exists():
        log(f"no existe la BBDD de origen: {db_path}", level="ERROR")
        return 2
    if not SCHEMA_SQL.exists():
        log(f"no existe el esquema canonico: {SCHEMA_SQL}", level="ERROR")
        return 2

    # Comprobacion rapida: origen sano y con datos.
    src = sqlite3.connect(db_path)
    try:
        integrity = src.execute("PRAGMA integrity_check").fetchone()[0]
        matches = _table_row_count(src, "matches")
    finally:
        src.close()
    if integrity != "ok":
        log(f"la BBDD de origen no pasa integrity_check ({integrity}); abortando export", level="ERROR")
        return 2
    if matches == 0:
        log("la BBDD de origen tiene 0 partidos; abortando export para no crear un blackbox vacio", level="ERROR")
        return 2

    blackbox_dir.mkdir(parents=True, exist_ok=True)
    staging = blackbox_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / DATA_SUBDIR).mkdir(parents=True, exist_ok=True)

    log(f"exportando fuente de verdad desde {db_path} -> {blackbox_dir}")
    snapshot = staging / "_snapshot.sqlite"
    _snapshot_db(db_path, snapshot)

    tmp_sqlite = staging / DATA_SUBDIR / SQLITE_NAME
    tmp_sql = staging / DATA_SUBDIR / SQLDUMP_NAME
    stats = _build_blackbox_sqlite(snapshot, tmp_sqlite)
    _dump_sql(tmp_sqlite, tmp_sql)

    # Comprimir y borrar los intermedios sin comprimir.
    _gzip_file(tmp_sqlite, staging / DATA_SUBDIR / SQLITE_GZ)
    _gzip_file(tmp_sql, staging / DATA_SUBDIR / SQLDUMP_GZ)
    tmp_sqlite.unlink()
    tmp_sql.unlink()
    snapshot.unlink(missing_ok=True)

    # Copia del esquema canonico.
    schema_copy = staging / SCHEMA_COPY_NAME
    shutil.copyfile(SCHEMA_SQL, schema_copy)

    # Manifest con checksums.
    files_meta: dict[str, dict] = {}
    for rel in [f"{DATA_SUBDIR}/{SQLITE_GZ}", f"{DATA_SUBDIR}/{SQLDUMP_GZ}", SCHEMA_COPY_NAME]:
        p = staging / rel
        files_meta[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    total_rows = sum(v["rows"] for v in stats.values())
    manifest = {
        "blackbox_format_version": BLACKBOX_FORMAT_VERSION,
        "tool_version": TOOL_VERSION,
        "created_at_utc": utcnow_iso(),
        "source_db": str(db_path),
        "schema_sha256": sha256_file(SCHEMA_SQL),
        "source_of_truth_tables": list(SOURCE_OF_TRUTH_TABLES),
        "derived_excluded_tables": list(DERIVED_EXCLUDED_TABLES),
        "total_source_rows": total_rows,
        "tables": stats,
        "files": files_meta,
        "canonical_file": f"{DATA_SUBDIR}/{SQLITE_GZ}",
        "text_dump_file": f"{DATA_SUBDIR}/{SQLDUMP_GZ}",
    }
    (staging / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Auto-verificacion del staging ANTES de publicar.
    rc = _verify_dir(staging, deep=True)
    if rc != 0:
        log("la auto-verificacion del export fallo; NO se publica (se conserva el BLACKBOX anterior)", level="ERROR")
        shutil.rmtree(staging, ignore_errors=True)
        return 3

    # Swap: conserva una generacion anterior en .prev.
    _publish_staging(blackbox_dir, staging)

    log(f"export OK: {total_rows} filas fuente en {len(stats)} tablas -> {blackbox_dir}")
    log(f"canonico: {DATA_SUBDIR}/{SQLITE_GZ} ({files_meta[f'{DATA_SUBDIR}/{SQLITE_GZ}']['bytes']} bytes)")
    return 0


def _publish_staging(blackbox_dir: Path, staging: Path) -> None:
    prev = blackbox_dir / ".prev"
    if prev.exists():
        shutil.rmtree(prev)
    prev.mkdir(parents=True, exist_ok=True)
    for name in [DATA_SUBDIR, MANIFEST_NAME, SCHEMA_COPY_NAME]:
        current = blackbox_dir / name
        if current.exists():
            shutil.move(str(current), str(prev / name))
    for name in [DATA_SUBDIR, MANIFEST_NAME, SCHEMA_COPY_NAME]:
        shutil.move(str(staging / name), str(blackbox_dir / name))
    shutil.rmtree(staging, ignore_errors=True)


# ===========================================================================
# VERIFY
# ===========================================================================
def _verify_dir(blackbox_dir: Path, *, deep: bool) -> int:
    manifest_path = blackbox_dir / MANIFEST_NAME
    if not manifest_path.exists():
        log(f"no existe manifest: {manifest_path}", level="ERROR")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok = True
    # 1) Checksums de cada fichero.
    for rel, meta in manifest.get("files", {}).items():
        p = blackbox_dir / rel
        if not p.exists():
            log(f"falta fichero listado en el manifest: {rel}", level="ERROR")
            ok = False
            continue
        actual = sha256_file(p)
        if actual != meta["sha256"]:
            log(f"checksum NO coincide en {rel}", level="ERROR")
            log(f"  esperado={meta['sha256']}", level="ERROR")
            log(f"  actual  ={actual}", level="ERROR")
            ok = False
        elif p.stat().st_size != meta.get("bytes", p.stat().st_size):
            log(f"tamano NO coincide en {rel}", level="ERROR")
            ok = False

    # 2) Hash del esquema.
    schema_copy = blackbox_dir / SCHEMA_COPY_NAME
    if schema_copy.exists():
        if sha256_file(schema_copy) != manifest.get("schema_sha256"):
            log("el hash de schema.sql no coincide con el manifest", level="ERROR")
            ok = False

    if not ok:
        return 1

    # 3) Deep: abre el sqlite canonico, integrity + cuadre de filas/hashes por tabla.
    if deep:
        rc = _verify_deep(blackbox_dir, manifest)
        if rc != 0:
            return rc

    log(f"verify OK: {len(manifest.get('files', {}))} ficheros, "
        f"{manifest.get('total_source_rows', '?')} filas fuente")
    return 0


def _verify_deep(blackbox_dir: Path, manifest: dict) -> int:
    canonical = blackbox_dir / manifest["canonical_file"]
    tmp = blackbox_dir / ".verify_tmp.sqlite"
    try:
        _gunzip_file(canonical, tmp)
        conn = sqlite3.connect(tmp)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                log(f"integrity_check del canonico fallo: {integrity}", level="ERROR")
                return 1
            ok = True
            for table, meta in manifest.get("tables", {}).items():
                rows = _table_row_count(conn, table)
                if rows != meta["rows"]:
                    log(f"tabla {table}: filas {rows} != manifest {meta['rows']}", level="ERROR")
                    ok = False
                content = table_content_hash(conn, table)
                if content != meta["content_sha256"]:
                    log(f"tabla {table}: hash de contenido no coincide", level="ERROR")
                    ok = False
            return 0 if ok else 1
        finally:
            conn.close()
    finally:
        if tmp.exists():
            tmp.unlink()


def verify(blackbox_dir: Path, *, deep: bool = True) -> int:
    return _verify_dir(blackbox_dir, deep=deep)


# ===========================================================================
# RESTORE
# ===========================================================================
def _db_health(db_path: Path) -> str:
    """Devuelve 'missing' | 'corrupt' | 'fk_violation' | 'empty' | 'healthy'."""
    if not db_path.exists():
        return "missing"
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError:
        return "corrupt"
    try:
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError:
            return "corrupt"
        if integrity != "ok":
            return "corrupt"
        try:
            fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.DatabaseError:
            return "corrupt"
        if fk:
            return "fk_violation"
        if "matches" not in _table_names(conn):
            return "empty"
        if _table_row_count(conn, "matches") == 0:
            return "empty"
        return "healthy"
    finally:
        conn.close()


def _backup_existing(db_path: Path, backup_dir: Path, reason: str) -> Path | None:
    if not db_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    dest = backup_dir / f"pre_restore_{reason}_{ts}.db"
    shutil.copy2(db_path, dest)
    log(f"guardian: BBDD existente respaldada en {dest}")
    return dest


def restore(
    db_path: Path,
    blackbox_dir: Path,
    *,
    force: bool = False,
    run_build_db: bool = True,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    reason: str = "manual",
) -> int:
    # 1) Verificar el blackbox antes de tocar nada.
    rc = _verify_dir(blackbox_dir, deep=True)
    if rc != 0:
        log("el BLACKBOX no pasa verify; NO se restaura", level="ERROR")
        return rc

    # 2) Guardian: no pisar una BBDD sana y con datos sin --force.
    health = _db_health(db_path)
    if health == "healthy" and not force:
        log(f"la BBDD destino esta SANA y con datos ({db_path}). No se sobrescribe.", level="ERROR")
        log("usa --force si de verdad quieres reemplazarla (se respaldara antes).", level="ERROR")
        return 4
    if db_path.exists():
        _backup_existing(db_path, backup_dir, reason=health)

    # 3) Descomprimir el sqlite canonico directamente como la nueva cs2.db.
    manifest = json.loads((blackbox_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    canonical = blackbox_dir / manifest["canonical_file"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = db_path.with_suffix(db_path.suffix + ".restoring")
    for suffix in ("-wal", "-shm"):    # limpia WAL/SHM colgantes de una BBDD anterior
        stale = Path(str(db_path) + suffix)
        if stale.exists():
            stale.unlink()
    if tmp_target.exists():
        tmp_target.unlink()
    _gunzip_file(canonical, tmp_target)

    # 4) Validar el restaurado y publicarlo.
    conn = sqlite3.connect(tmp_target)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        restored_matches = _table_row_count(conn, "matches")
    finally:
        conn.close()
    if integrity != "ok":
        tmp_target.unlink(missing_ok=True)
        log(f"la BBDD restaurada no pasa integrity_check ({integrity}); abortando", level="ERROR")
        return 3
    os.replace(tmp_target, db_path)
    log(f"restore: cs2.db reconstruida con {restored_matches} partidos desde {blackbox_dir}")

    # 5) Regenerar lo derivado / migraciones idempotentes.
    if run_build_db:
        rc = _run_build_db()
        if rc != 0:
            log("build_db.py devolvio error tras el restore (migracion/derivado)", level="WARN")
    else:
        log("build_db.py omitido (--no-build-db): las tablas derivadas se regeneran al entrenar")

    log("restore OK (idempotente).")
    return 0


def _run_build_db() -> int:
    if not BUILD_DB.exists():
        log(f"no existe {BUILD_DB}; se omite build_db", level="WARN")
        return 0
    log("ejecutando build_db.py (esquema vivo + derivado)...")
    proc = subprocess.run([sys.executable, str(BUILD_DB)], cwd=str(CS2_ROOT))
    return proc.returncode


# ===========================================================================
# AUTO-HEAL
# ===========================================================================
def autoheal(
    db_path: Path,
    blackbox_dir: Path,
    *,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    run_build_db: bool = True,
) -> int:
    """Si la BBDD esta mala y hay un BLACKBOX valido, restaura antes de seguir.

    Nunca sobrescribe una BBDD sana (guardian). Nunca aborta la pipeline: si no
    puede curar, avisa y devuelve 0 para que el resto del arranque siga.
    """
    health = _db_health(db_path)
    if health == "healthy":
        log(f"auto-heal: BBDD sana ({db_path}); no se hace nada.")
        return 0

    log(f"auto-heal: BBDD en estado '{health}' ({db_path}).", level="WARN")

    if not (blackbox_dir / MANIFEST_NAME).exists():
        log("auto-heal: no hay BLACKBOX disponible para restaurar.", level="WARN")
        log("auto-heal: se continua; build_db.py inicializara/sembrara lo que pueda.", level="WARN")
        return 0

    rc = _verify_dir(blackbox_dir, deep=True)
    if rc != 0:
        log("auto-heal: el BLACKBOX no pasa verify; NO se restaura para no empeorar.", level="ERROR")
        log("auto-heal: se continua sin restaurar. Revisa el BLACKBOX.", level="WARN")
        return 0

    log("auto-heal: BLACKBOX valido; restaurando automaticamente...", level="WARN")
    # force=True porque el estado no es 'healthy'; el guardian ya respalda lo que haya.
    rc = restore(
        db_path,
        blackbox_dir,
        force=True,
        run_build_db=run_build_db,
        backup_dir=backup_dir,
        reason=f"autoheal_{health}",
    )
    if rc == 0:
        log("auto-heal: restauracion completada. La pipeline puede continuar.")
    else:
        log(f"auto-heal: la restauracion fallo (rc={rc}). Se continua igualmente.", level="ERROR")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLACKBOX: caja negra portatil de la fuente de verdad de CS2.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--db", type=Path, default=DEFAULT_DB, help="Ruta de cs2.db.")
        sp.add_argument("--blackbox", type=Path, default=DEFAULT_BLACKBOX, help="Carpeta BLACKBOX.")

    sp_export = sub.add_parser("export", help="Escribe BLACKBOX desde cs2.db.")
    add_common(sp_export)

    sp_verify = sub.add_parser("verify", help="Comprueba integridad contra el manifest.")
    add_common(sp_verify)
    sp_verify.add_argument("--shallow", action="store_true", help="Solo checksums de fichero (sin abrir el sqlite).")

    sp_restore = sub.add_parser("restore", help="Reconstruye cs2.db desde BLACKBOX.")
    add_common(sp_restore)
    sp_restore.add_argument("--force", action="store_true", help="Sobrescribe aunque la BBDD este sana (se respalda antes).")
    sp_restore.add_argument("--no-build-db", action="store_true", help="No ejecutar build_db.py tras restaurar.")
    sp_restore.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)

    sp_heal = sub.add_parser("autoheal", help="Restaura solo si la BBDD falta/vacia/corrupta.")
    add_common(sp_heal)
    sp_heal.add_argument("--no-build-db", action="store_true")
    sp_heal.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        return export(args.db, args.blackbox)
    if args.command == "verify":
        return verify(args.blackbox, deep=not args.shallow)
    if args.command == "restore":
        return restore(
            args.db,
            args.blackbox,
            force=args.force,
            run_build_db=not args.no_build_db,
            backup_dir=args.backup_dir,
        )
    if args.command == "autoheal":
        return autoheal(
            args.db,
            args.blackbox,
            backup_dir=args.backup_dir,
            run_build_db=not args.no_build_db,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

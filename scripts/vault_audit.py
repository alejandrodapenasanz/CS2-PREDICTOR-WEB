"""Read-only integrity audit for VAULT migration; never opens SQLite writable.

Run: python scripts/vault_audit.py [--output VAULT/verification/before.json].
During migration it locates each DB in VAULT or its still-unmoved legacy path.
Records file hashes, integrity, sacred-table counts/digests and trigger hashes.
"""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from vault_migrate import ROOT, digest, save_json

DATABASES = {
    "CS2/BBDD/cs2.db": ("prediction_ledger",),
    "TENNIS/BBDD/tennis.sqlite3": ("predictions", "observations", "settlements"),
    "TELEGRAM/data/telegram_publish_state.sqlite3": (),
}


def audit(root: Path) -> dict[str, object]:
    """Check physical and logical content without changing database or triggers."""
    report: dict[str, object] = {}
    for relative, tables in DATABASES.items():
        path = root / "VAULT" / relative
        if not path.is_file():
            path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Required database missing: {relative}")
        wal = Path(str(path) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise RuntimeError(f"Active WAL: stop/checkpoint via normal component before copying {relative}")
        before = digest(path)
        table_info: dict[str, object] = {}
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise RuntimeError(f"Integrity check failed: {relative}")
            for table in tables:
                checksum = hashlib.sha256()
                count = 0
                for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
                    checksum.update(json.dumps(row, default=str, ensure_ascii=False).encode("utf-8") + b"\n")
                    count += 1
                table_info[table] = {"rows": count, "sha256": checksum.hexdigest()}
            triggers = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
        if before != digest(path):
            raise RuntimeError(f"Database changed during read-only audit: {relative}")
        report[relative] = {
            "sha256": before,
            "integrity": "ok",
            "tables": table_info,
            "triggers_sha256": hashlib.sha256(json.dumps(triggers).encode()).hexdigest(),
        }
    return report


def main() -> int:
    """Print only metadata, never database records or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

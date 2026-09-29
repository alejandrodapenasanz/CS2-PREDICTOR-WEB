"""Inventory/migrate ignored state into VAULT without rewriting database rows.

Run with CPython 3.13: python scripts/vault_migrate.py [--apply]. Stop operational
processes first. A same-volume rename preserves bytes and timestamps; each data
file is hashed before and after. Conflicts and unreadable paths are never forced.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGENERABLE = {".venv", ".venv.previous", ".venv.build", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SEEDS = (
    "TENNIS/data/processed/identity_quarantine.csv",
    "TENNIS/data/processed/identity_quarantine.manifest.json",
)
CRITICAL = (
    "CS2/BBDD/cs2.db",
    "CS2/MODEL/artifacts/model.pkl",
    "CS2/MODEL/artifacts/registry/latest.json",
    "CS2/MODEL/artifacts/registry/last_good.json",
    "CS2/PIPELINE/master/matches.json",
    "TENNIS/BBDD/tennis.sqlite3",
    "TENNIS/models/phase7/manifest.json",
    "TENNIS/data/processed/elo/elo.sqlite3",
    "TENNIS/data/processed/player_mapping.sqlite3",
    "TENNIS/data/processed/tennisratio.sqlite3",
    "TELEGRAM/.env",
    "TELEGRAM/data/telegram_publish_state.sqlite3",
)


def assert_quiescent(root: Path) -> None:
    """Refuse to relocate state while a component Python process is alive."""
    if os.name != "nt":
        return
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
        "Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    processes = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    for process in processes:
        executable = process.get("ExecutablePath")
        if executable and process["ProcessId"] != os.getpid():
            if Path(executable).resolve().is_relative_to(root):
                raise RuntimeError(f"Stop component Python PID {process['ProcessId']} before migration")


def validate_ready(root: Path) -> dict[str, Any]:
    """Fail closed on a partial/missing vault, never initialize an empty replacement DB."""
    vault = root.resolve() / "VAULT"
    report = json.loads((vault / "migration.json").read_text(encoding="utf-8"))
    if report["status"] not in {"migrated", "migrated_with_pending"}:
        raise RuntimeError("VAULT migration is incomplete")
    for name in report.get("required", []):
        if not confined(vault, name).is_file():
            raise RuntimeError(f"Required private state is missing: VAULT/{name}")
    return report


def digest(path: Path) -> str:
    """Hash bytes without loading large SQLite/model files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def confined(root: Path, relative: str) -> Path:
    """Validate the resolved absolute target before any move or file write."""
    if not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive or ".." in Path(relative).parts:
        raise ValueError("Invalid relative migration path")
    base = root.resolve()
    path = base / relative
    resolved = path.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError(f"Path escapes migration root: {relative}")
    return path


def ignored_entries(root: Path) -> list[str]:
    """Use Git's real ignore contract, never treat arbitrary untracked code as data."""
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"],
        capture_output=True,
        check=True,
    )
    names = sorted(
        {name.rstrip("/") for name in completed.stdout.decode("utf-8").split("\0") if name},
        key=lambda name: (len(Path(name).parts), name),
    )
    selected: list[str] = []
    for name in names:
        if name.split("/")[0] in {"VAULT", ".git", ".agents", ".codex"}:
            continue
        if not any(name == prior or name.startswith(prior + "/") for prior in selected):
            selected.append(name)
    return selected


def inventory(path: Path) -> tuple[dict[str, str], list[str]]:
    """Hash readable data; retain inaccessible descendants as explicit findings."""
    hashes: dict[str, str] = {}
    inaccessible: list[str] = []
    if path.name in REGENERABLE or path.name.startswith((".pytest-", ".tennis-quality-")):
        return hashes, inaccessible
    pending = [path]
    while pending:
        item = pending.pop()
        try:
            mode = item.lstat()
            if stat.S_ISLNK(mode.st_mode) or getattr(mode, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Reparse points are not migrated: {item}")
            if item.is_dir():
                pending.extend(item.iterdir())
            else:
                key = "." if item == path else item.relative_to(path).as_posix()
                hashes[key] = digest(item)
        except OSError:
            inaccessible.append("." if item == path else item.relative_to(path).as_posix())
    return hashes, sorted(inaccessible)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the migration journal; never truncate its last good copy."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def remove_empty_residue_directories(root: Path, path: Path) -> None:
    """Remove only empty ignored directories left by verified file moves.

    No recursive deletion or symlink following: rmdir itself refuses to remove
    anything containing a file, including tracked .gitkeep placeholders.
    """
    try:
        # Windows junctions are not always reported as symlinks by os.walk.
        # Inspect each node BEFORE descent, not just before removing its parent.
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            return
        candidate = confined(root, path.relative_to(root).as_posix())
        children = list(candidate.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if child.is_dir():
                remove_empty_residue_directories(root, child)
        except OSError:
            pass
    try:
        candidate.rmdir()
    except OSError:
        pass  # Nonempty or unreadable directories remain explicit residues.


def migrate(root: Path, *, apply: bool = False) -> dict[str, Any]:
    """Move only reviewed ignored paths, preserve data and journal every completed step."""
    root = root.resolve()
    vault = root / "VAULT"
    names = ignored_entries(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "origin_root": str(root),
        "status": "planned",
        "entries": [],
        "pending": [],
    }
    if not apply:
        report["entries"] = names
        return report
    assert_quiescent(root)
    if vault.is_symlink() or (vault.exists() and getattr(vault.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("VAULT must be a real directory")
    vault.mkdir(exist_ok=True)
    lock = vault / ".migration.lock"
    if lock.exists() and os.name == "nt":
        pid = int(lock.read_text(encoding="ascii"))
        check = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 1 }}",
            ],
            check=False,
        )
        if check.returncode != 0:
            raise RuntimeError("Another migration is running")
        lock.unlink()  # Only our exact, validated stale lock, never a component lock.
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    journal = vault / "migration.json"
    try:
        if journal.exists():
            report = json.loads(journal.read_text(encoding="utf-8"))
        else:
            report["required"] = [name for name in CRITICAL if confined(root, name).is_file()]
            if not report["required"]:
                raise RuntimeError("No private state found: copy your complete VAULT beside the code first")
        report["status"] = "migrating"
        report["pending"] = []
        # Complete a rename interrupted between the operation and its journal commit.
        for entry in report["entries"]:
            if entry["status"] != "moving":
                continue
            source = confined(root, entry["source"])
            destination = confined(vault, entry["destination"])
            if destination.exists() and not source.exists():
                hashes, unreadable = inventory(destination)
                if hashes != entry["sha256"] or unreadable != entry["unreadable_descendants"]:
                    raise RuntimeError("Interrupted migration does not match its saved inventory")
                entry["status"] = "moved_verified"
            elif source.exists() and not destination.exists():
                entry["status"] = "not_moved"
            else:
                raise RuntimeError("Ambiguous interrupted migration; no files overwritten")
        save_json(journal, report)
        for name in names:
            source = confined(root, name)
            try:
                if source.is_dir():
                    remove_empty_residue_directories(root, source)
                if not source.exists():
                    continue
            except OSError as exc:
                report["pending"].append({"source": name, "reason": type(exc).__name__})
                continue
            namespace = name.split("/")[0]
            destination_name = name if namespace in {"CS2", "TENNIS", "TELEGRAM", "WEB"} else "archive/" + name
            destination = confined(vault, destination_name)
            try:
                destination_exists = destination.exists()
            except OSError as exc:
                report["pending"].append({"source": name, "reason": "destination_" + type(exc).__name__})
                continue
            if destination_exists:
                if source.is_dir() and destination.is_dir() and not source.is_symlink():
                    try:
                        names.extend(child.relative_to(root).as_posix() for child in sorted(source.iterdir()))
                    except OSError as exc:
                        report["pending"].append({"source": name, "reason": type(exc).__name__})
                    continue
                if any(Path(required).is_relative_to(Path(name)) for required in report["required"]):
                    report["pending"].append({"source": name, "reason": "destination_exists"})
                    continue
                # Preserve both versions. Never replace live state with a leftover
                # legacy file, even if only its filename matches.
                destination_name = "archive/legacy_collisions/" + name
                destination = confined(vault, destination_name)
                if destination.exists():
                    report["pending"].append({"source": name, "reason": "archive_destination_exists"})
                    continue
            try:
                before, inaccessible = inventory(source)
                if not before and inaccessible:
                    report["pending"].append({"source": name, "reason": "unreadable"})
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Revalidate both targets immediately before the same-volume move.
                confined(root, name)
                confined(vault, destination_name)
                entry = {
                    "source": name,
                    "destination": destination_name,
                    "sha256": before,
                    "unreadable_descendants": inaccessible,
                    "status": "moving",
                }
                report["entries"].append(entry)
                save_json(journal, report)
                os.rename(source, destination)
                try:
                    after, after_inaccessible = inventory(destination)
                    if before != after or inaccessible != after_inaccessible:
                        raise ValueError("Post-move verification differs")
                except Exception:
                    if source.exists():
                        raise RuntimeError("Cannot roll back: original path was recreated") from None
                    os.rename(destination, source)
                    entry["status"] = "rolled_back"
                    save_json(journal, report)
                    raise
                entry["status"] = "moved_verified"
                print(f"[VAULT] moved {name} ({len(before)} verified files)", flush=True)
                save_json(journal, report)
            except OSError as exc:
                if report["entries"] and report["entries"][-1].get("status") == "moving":
                    report["entries"][-1]["status"] = "not_moved"
                report["pending"].append({"source": name, "reason": type(exc).__name__})
                save_json(journal, report)
        for name in SEEDS:
            source, destination = confined(root, name), confined(vault, name)
            if source.is_file() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if digest(source) != digest(destination):
                    raise ValueError("Seed verification failed")
        missing = [name for name in report["required"] if not confined(vault, name).is_file()]
        conflicts = [
            item["source"]
            for item in report["pending"]
            if any(Path(name).is_relative_to(Path(item["source"])) for name in report["required"])
        ]
        report["status"] = (
            "blocked" if missing or conflicts else "migrated_with_pending" if report["pending"] else "migrated"
        )
        save_json(journal, report)
        if missing or conflicts:
            raise RuntimeError("Required state did not migrate: " + ", ".join(missing))
    finally:
        lock.unlink()
    return report


def main() -> int:
    """Print a preview by default; --apply explicitly performs the migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        report = validate_ready(args.root)
        print(json.dumps({"status": report["status"], "required_files": len(report.get("required", []))}))
        return 0
    report = migrate(args.root, apply=args.apply)
    if args.apply:
        print(json.dumps({key: report[key] for key in ("status", "pending")}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["pending"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

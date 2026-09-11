"""Persistent, non-destructive registry of retention cleanup residues.

The file managed here is operational metadata only: it never authorizes a
deletion.  Retention still validates its own plan, pointers and quarantine
before touching an artifact.  This sidecar merely makes failed cleanup visible
across later runs until the exact path disappears.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

PENDING_CLEANUP_SCHEMA_VERSION = 1
CleanupKind = Literal["registry", "runs"]


class PendingCleanupStateError(RuntimeError):
    """The cleanup sidecar is unsafe or malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": PENDING_CLEANUP_SCHEMA_VERSION,
        "updated_at_utc": None,
        "items": [],
    }


def _is_safe_file(path: Path) -> bool:
    return not path.is_symlink() and (not os.path.lexists(path) or path.is_file())


def _validate_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PendingCleanupStateError("pending cleanup item must be an object")
    required_text = (
        "kind",
        "name",
        "root",
        "canonical_path",
        "actual_path",
        "phase",
        "error_type",
        "error",
        "first_failed_at_utc",
        "last_failed_at_utc",
    )
    if any(not isinstance(value.get(field), str) for field in required_text):
        raise PendingCleanupStateError("pending cleanup item has invalid text fields")
    if value["kind"] not in {"registry", "runs"}:
        raise PendingCleanupStateError("pending cleanup item has invalid kind")
    name = value["name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise PendingCleanupStateError("pending cleanup item has unsafe name")
    attempts = value.get("attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise PendingCleanupStateError("pending cleanup item has invalid attempts")
    return dict(value)


def read_pending_cleanup(path: str | Path) -> dict[str, Any]:
    """Read and validate a pending-cleanup sidecar without changing it."""

    source = Path(path).absolute()
    if not os.path.lexists(source):
        return _empty_state()
    if not _is_safe_file(source):
        raise PendingCleanupStateError(f"pending cleanup path is unsafe: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PendingCleanupStateError(f"pending cleanup state is invalid: {source}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PENDING_CLEANUP_SCHEMA_VERSION
        or not isinstance(payload.get("items"), list)
    ):
        raise PendingCleanupStateError(f"pending cleanup state has an invalid schema: {source}")
    items = [_validate_item(item) for item in payload["items"]]
    return {
        "schema_version": PENDING_CLEANUP_SCHEMA_VERSION,
        "updated_at_utc": payload.get("updated_at_utc"),
        "items": items,
    }


def _write_pending_cleanup(path: Path, payload: dict[str, Any]) -> None:
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir() or not _is_safe_file(target):
        raise PendingCleanupStateError(f"pending cleanup destination is unsafe: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _lexically_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def record_cleanup_failure(
    state_path: str | Path,
    *,
    kind: CleanupKind,
    root: str | Path,
    names: Iterable[str],
    phase: str,
    error: BaseException,
    quarantine: str | Path | None = None,
    protected_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Record exact failed targets without modifying any artifact directory."""

    root_path = Path(root).absolute()
    protected = set(protected_names)
    normalized = tuple(dict.fromkeys(str(name) for name in names))
    if not normalized:
        raise PendingCleanupStateError("cannot record an empty cleanup failure")
    if any(Path(name).name != name or not name for name in normalized):
        raise PendingCleanupStateError("cleanup failure contains an unsafe target name")
    conflicts = sorted(protected.intersection(normalized))
    if conflicts:
        raise PendingCleanupStateError("refusing to record protected cleanup targets: " + ", ".join(conflicts))

    state_file = Path(state_path).absolute()
    state = read_pending_cleanup(state_file)
    now = _utc_now()
    by_key = {
        (str(item["kind"]), str(Path(item["root"]).absolute()), str(item["name"])): item for item in state["items"]
    }
    quarantine_path = Path(quarantine).absolute() if quarantine is not None else None
    for name in normalized:
        canonical = root_path / name
        actual = quarantine_path / name if quarantine_path is not None else canonical
        if not _lexically_within(canonical, root_path) or not _lexically_within(actual, root_path):
            raise PendingCleanupStateError(f"cleanup target escapes its root: {actual}")
        key = (kind, str(root_path), name)
        previous = by_key.get(key)
        by_key[key] = {
            "kind": kind,
            "name": name,
            "root": str(root_path),
            "canonical_path": str(canonical),
            "actual_path": str(actual),
            "phase": str(phase),
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "attempts": int(previous.get("attempts", 0)) + 1 if previous else 1,
            "first_failed_at_utc": previous.get("first_failed_at_utc", now) if previous else now,
            "last_failed_at_utc": now,
        }

    items = sorted(by_key.values(), key=lambda item: (item["kind"], item["name"]))
    updated = {
        "schema_version": PENDING_CLEANUP_SCHEMA_VERSION,
        "updated_at_utc": now,
        "items": items,
    }
    _write_pending_cleanup(state_file, updated)
    return cleanup_summary(updated, state_path=state_file)


def reconcile_pending_cleanup(state_path: str | Path) -> dict[str, Any]:
    """Drop records whose canonical and quarantined paths no longer exist."""

    state_file = Path(state_path).absolute()
    state = read_pending_cleanup(state_file)
    existing = [
        item
        for item in state["items"]
        if os.path.lexists(Path(item["actual_path"])) or os.path.lexists(Path(item["canonical_path"]))
    ]
    if existing != state["items"]:
        state = {
            "schema_version": PENDING_CLEANUP_SCHEMA_VERSION,
            "updated_at_utc": _utc_now(),
            "items": existing,
        }
        _write_pending_cleanup(state_file, state)
    return cleanup_summary(state, state_path=state_file)


def cleanup_summary(payload: dict[str, Any], *, state_path: str | Path) -> dict[str, Any]:
    """Return the stable shape published in CLI logs and the dashboard."""

    items = list(payload.get("items", []))
    return {
        "schema_version": PENDING_CLEANUP_SCHEMA_VERSION,
        "status": "warning" if items else "ok",
        "count": len(items),
        "state_path": str(Path(state_path).absolute()),
        "updated_at_utc": payload.get("updated_at_utc"),
        "items": items,
    }

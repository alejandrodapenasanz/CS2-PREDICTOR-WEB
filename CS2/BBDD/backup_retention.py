"""Preview-first, fail-closed retention for CS2 SQLite backup files.

This module never opens the live SQLite database.  It inventories filesystem
metadata, validates the selected retained backup, and reads sacred-ledger
counts from candidates in immutable read-only mode.  It can unlink only an
exact, confirmed set of regular files.  Database backups and the two explicitly
allowlisted legacy extras deliberately use separate plans.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

POLICY_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
APPROVAL_MARKER_VERSION = 1
DEFAULT_BBDD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DEFAULT_BBDD_DIR / "backup_retention.json"
LIVE_PROTECTED_NAMES = ("cs2.db", "cs2.db-wal", "cs2.db-shm", "cs2.db-journal")
HARD_CODED_CLEANUP_TARGETS = ("cs2_dump.sql", "cs2_integrity_test.db")
RETENTION_LOCK_NAME = ".backup-retention.lock"
APPROVAL_MARKER_NAME = ".backup-retention-approved.json"
_STAMPED_BACKUP_RE = re.compile(r"^(?:cs2|pre_restore)_[A-Za-z0-9_-]*?(?P<stamp>\d{8}_\d{6}(?:_\d{6})?Z)\.db$")
_CONFIRM_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")

PlanKind = Literal["prune", "cleanup"]


class BackupRetentionError(RuntimeError):
    """A retention request was rejected before an unsafe mutation."""


class StaleBackupPlanError(BackupRetentionError):
    """The confirmation token no longer represents the filesystem."""


class UnsafeBackupPathError(BackupRetentionError):
    """A root or deletion target failed containment/type checks."""


class BackupRetentionConfigError(BackupRetentionError):
    """The BBDD-owned retention policy file is missing or invalid."""


class BackupRetentionApprovalError(BackupRetentionError):
    """Automatic retention lacks a safe marker for the exact current policy."""


class BackupDeletionError(BackupRetentionError):
    """An OS failure interrupted an already-confirmed deletion batch."""

    def __init__(self, message: str, *, deleted: tuple[str, ...], deleted_bytes: int) -> None:
        super().__init__(message)
        self.deleted = deleted
        self.deleted_bytes = deleted_bytes


@dataclass(frozen=True)
class BackupRetentionConfig:
    """Validated BBDD-owned policy configuration."""

    path: Path
    sha256: str
    default_keep: int
    cleanup_targets: tuple[str, ...]


@dataclass(frozen=True)
class FileState:
    """Non-content filesystem identity used by an exact confirmation token."""

    name: str
    path: Path
    kind: str
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    reparse_or_symlink: bool

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    def token_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "device": self.device,
            "inode": self.inode,
            "reparse_or_symlink": self.reparse_or_symlink,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.token_dict(),
            "path": str(self.path),
        }


@dataclass(frozen=True)
class PlanEntry:
    """One path and the reasons for keeping or deleting it."""

    state: FileState
    reasons: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.state.name

    def token_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.token_dict(),
            "reasons": list(self.reasons),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.state.as_dict(),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BackupValidation:
    """Read-only SQLite validation of the selected retained backup."""

    name: str
    mode: str
    ok: bool
    result: tuple[str, ...]
    matches: int | None
    prediction_ledger_rows: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "ok": self.ok,
            "result": list(self.result),
            "matches": self.matches,
            "prediction_ledger_rows": self.prediction_ledger_rows,
        }


@dataclass(frozen=True)
class BackupLedgerCount:
    """Read-only observation used to prevent sacred-ledger regression."""

    name: str
    prediction_ledger_rows: int | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prediction_ledger_rows": self.prediction_ledger_rows,
            "error": self.error,
        }


@dataclass(frozen=True)
class BackupRetentionPlan:
    """Immutable preview which must be rebuilt immediately before deletion."""

    kind: PlanKind
    bbdd_dir: Path
    backups_dir: Path
    config: BackupRetentionConfig
    keep_latest: int | None
    required_keep: PlanEntry | None
    cleanup_targets: tuple[str, ...]
    keep: tuple[PlanEntry, ...]
    delete: tuple[PlanEntry, ...]
    protected: tuple[PlanEntry, ...]
    retained_backup_validation: BackupValidation | None
    ledger_counts: tuple[BackupLedgerCount, ...]
    blocked: tuple[str, ...]
    token: str

    @property
    def keep_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.keep)

    @property
    def delete_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.delete)

    @property
    def delete_bytes(self) -> int:
        return sum(entry.state.size for entry in self.delete)

    @property
    def newest_backup_validation(self) -> BackupValidation | None:
        """Backward-compatible alias for the formerly name-selected backup."""

        return self.retained_backup_validation

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "kind": self.kind,
            "bbdd_dir": str(self.bbdd_dir),
            "backups_dir": str(self.backups_dir),
            "config": {
                "path": str(self.config.path),
                "sha256": self.config.sha256,
                "schema_version": CONFIG_SCHEMA_VERSION,
            },
            "keep_latest": self.keep_latest,
            "required_keep": self.required_keep.as_dict() if self.required_keep else None,
            "cleanup_targets": list(self.cleanup_targets),
            "keep": [entry.as_dict() for entry in self.keep],
            "delete": [entry.as_dict() for entry in self.delete],
            "protected": [entry.as_dict() for entry in self.protected],
            "retained_backup_validation": (
                self.retained_backup_validation.as_dict() if self.retained_backup_validation else None
            ),
            "newest_backup_validation": (
                self.retained_backup_validation.as_dict() if self.retained_backup_validation else None
            ),
            "ledger_counts": [observation.as_dict() for observation in self.ledger_counts],
            "blocked": list(self.blocked),
            "keep_count": len(self.keep),
            "delete_count": len(self.delete),
            "delete_bytes": self.delete_bytes,
            "token": self.token,
        }


@dataclass(frozen=True)
class BackupApplyResult:
    """Exact files removed after confirmation."""

    kind: PlanKind
    token: str
    deleted: tuple[str, ...]
    deleted_bytes: int
    approval_marker: Path | None = None
    automatic: bool = False


@dataclass(frozen=True)
class BackupAutomaticResult:
    """Automatic decision: preview/no-op until an exact policy is approved."""

    plan: BackupRetentionPlan
    applied: bool
    reason: str | None
    result: BackupApplyResult | None
    approval_marker: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path, status: os.stat_result | None = None) -> bool:
    try:
        current = status if status is not None else path.lstat()
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = int(getattr(current, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & reparse_flag)


def _state(path: Path) -> FileState:
    status = path.lstat()
    mode = status.st_mode
    if stat.S_ISREG(mode):
        kind = "file"
    elif stat.S_ISDIR(mode):
        kind = "directory"
    else:
        kind = "other"
    return FileState(
        name=path.name,
        path=path,
        kind=kind,
        size=int(status.st_size),
        mtime_ns=int(status.st_mtime_ns),
        ctime_ns=int(status.st_ctime_ns),
        device=int(status.st_dev),
        inode=int(status.st_ino),
        reparse_or_symlink=_is_reparse_or_symlink(path, status),
    )


def _require_real_directory(path_value: str | Path, *, label: str) -> tuple[Path, Path]:
    lexical = Path(path_value).absolute()
    if not os.path.lexists(lexical):
        raise FileNotFoundError(f"{label} does not exist: {lexical}")
    status = lexical.lstat()
    if _is_reparse_or_symlink(lexical, status):
        raise UnsafeBackupPathError(f"{label} must not be a symlink or reparse point: {lexical}")
    if not stat.S_ISDIR(status.st_mode):
        raise UnsafeBackupPathError(f"{label} is not a directory: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeBackupPathError(f"cannot resolve {label}: {lexical}") from exc
    return lexical, resolved


def _prepare_layout(bbdd_dir: str | Path) -> tuple[Path, Path, Path]:
    bbdd, resolved_bbdd = _require_real_directory(bbdd_dir, label="BBDD directory")
    backups = bbdd / "backups"
    backups_lexical, resolved_backups = _require_real_directory(backups, label="backup directory")
    if backups_lexical.parent != bbdd or backups_lexical.name != "backups":
        raise UnsafeBackupPathError("backup directory must be the exact BBDD/backups child")
    if resolved_backups.parent != resolved_bbdd:
        raise UnsafeBackupPathError("resolved backup directory escapes BBDD")
    return bbdd, backups_lexical, resolved_backups


def _validate_positive_keep(value: int) -> int:
    if isinstance(value, bool) or value < 1:
        raise BackupRetentionConfigError("keep must be an integer >= 1")
    return int(value)


def load_retention_config(path_value: str | Path = DEFAULT_CONFIG) -> BackupRetentionConfig:
    """Load and validate the small BBDD-owned JSON policy."""

    path = Path(path_value).absolute()
    if not os.path.lexists(path):
        raise BackupRetentionConfigError(f"retention config does not exist: {path}")
    status = path.lstat()
    if _is_reparse_or_symlink(path, status) or not stat.S_ISREG(status.st_mode):
        raise BackupRetentionConfigError(f"retention config must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupRetentionConfigError(f"invalid retention config: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise BackupRetentionConfigError("unsupported backup retention config schema")
    default_keep = payload.get("default_keep")
    if not isinstance(default_keep, int):
        raise BackupRetentionConfigError("default_keep must be an integer")
    default_keep = _validate_positive_keep(default_keep)
    targets_raw = payload.get("cleanup_targets")
    if not isinstance(targets_raw, list) or not all(isinstance(item, str) for item in targets_raw):
        raise BackupRetentionConfigError("cleanup_targets must be a list of exact filenames")
    if len(targets_raw) != len(set(targets_raw)):
        raise BackupRetentionConfigError("cleanup_targets must not contain duplicates")
    targets = tuple(sorted(targets_raw))
    if not targets or not set(targets).issubset(HARD_CODED_CLEANUP_TARGETS):
        raise BackupRetentionConfigError("cleanup_targets contains a non-allowlisted filename")
    return BackupRetentionConfig(
        path=path,
        sha256=_sha256_file(path),
        default_keep=default_keep,
        cleanup_targets=targets,
    )


def _backup_timestamp(name: str) -> str | None:
    match = _STAMPED_BACKUP_RE.fullmatch(name)
    if match is None:
        return None
    stamp = match.group("stamp")
    for date_format in ("%Y%m%d_%H%M%S_%fZ", "%Y%m%d_%H%M%SZ"):
        try:
            parsed = datetime.strptime(stamp, date_format)
        except ValueError:
            continue
        if parsed.strftime(date_format) != stamp:
            return None
        return parsed.strftime("%Y%m%d_%H%M%S_%fZ")
    return None


def _protected_states(bbdd: Path) -> tuple[PlanEntry, ...]:
    entries: list[PlanEntry] = []
    for name in LIVE_PROTECTED_NAMES:
        path = bbdd / name
        if os.path.lexists(path):
            entries.append(PlanEntry(_state(path), ("live_database_or_sidecar", "never_delete")))
    blackbox = bbdd / "BLACKBOX"
    if os.path.lexists(blackbox):
        entries.append(PlanEntry(_state(blackbox), ("blackbox", "never_delete")))
    return tuple(sorted(entries, key=lambda entry: entry.name))


def _live_safety_blocks(bbdd: Path, protected: Sequence[PlanEntry]) -> list[str]:
    """Require a real live DB and real sidecars when those sidecars exist."""

    by_name = {entry.name: entry for entry in protected}
    blocked: list[str] = []
    live = by_name.get("cs2.db")
    if live is None:
        blocked.append("live_database_missing")
    elif live.state.reparse_or_symlink or live.state.kind != "file":
        blocked.append("live_database_unsafe")
    for name in LIVE_PROTECTED_NAMES[1:]:
        entry = by_name.get(name)
        if entry is not None and (entry.state.reparse_or_symlink or entry.state.kind != "file"):
            blocked.append(f"live_sidecar_unsafe:{name}")
    blackbox = bbdd / "BLACKBOX"
    if os.path.lexists(blackbox):
        state = by_name.get("BLACKBOX")
        if state is None or state.state.reparse_or_symlink or state.state.kind != "directory":
            blocked.append("blackbox_unsafe")
    return blocked


def _live_identities(protected: Sequence[PlanEntry]) -> set[tuple[int, int]]:
    return {
        entry.state.identity
        for entry in protected
        if "live_database_or_sidecar" in entry.reasons and entry.state.kind == "file"
    }


def _unsafe_adjacent_transaction_file(state: FileState) -> str | None:
    """Reject uncheckpointed WAL/rollback-journal state beside one snapshot."""

    for suffix, label in (("-wal", "WAL"), ("-journal", "rollback journal")):
        adjacent = Path(str(state.path) + suffix)
        if not os.path.lexists(adjacent):
            continue
        adjacent_state = _state(adjacent)
        if adjacent_state.reparse_or_symlink or adjacent_state.kind != "file" or adjacent_state.size != 0:
            return f"adjacent {label} is unsafe or non-empty"
    return None


def _validate_sqlite_backup(state: FileState) -> BackupValidation:
    """Run SQLite quick_check in immutable read-only mode on one backup."""

    connection: sqlite3.Connection | None = None
    try:
        transaction_error = _unsafe_adjacent_transaction_file(state)
        if transaction_error is not None:
            return BackupValidation(
                name=state.name,
                mode="sqlite_mode_ro_immutable_quick_check",
                ok=False,
                result=(transaction_error,),
                matches=None,
                prediction_ledger_rows=None,
            )
        uri = state.path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.execute("PRAGMA query_only=ON")
        rows = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check").fetchall())
        matches_table = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'matches'"
        ).fetchone()
        matches = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]) if matches_table else None
        ledger_table = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'prediction_ledger'"
        ).fetchone()
        ledger_rows = (
            int(connection.execute("SELECT COUNT(*) FROM prediction_ledger").fetchone()[0]) if ledger_table else None
        )
    except (OSError, sqlite3.Error) as exc:
        return BackupValidation(
            name=state.name,
            mode="sqlite_mode_ro_immutable_quick_check",
            ok=False,
            result=(f"{type(exc).__name__}: {exc}",),
            matches=None,
            prediction_ledger_rows=None,
        )
    finally:
        if connection is not None:
            connection.close()
    return BackupValidation(
        name=state.name,
        mode="sqlite_mode_ro_immutable_quick_check",
        ok=rows == ("ok",) and matches is not None and matches > 0 and ledger_rows is not None,
        result=rows,
        matches=matches,
        prediction_ledger_rows=ledger_rows,
    )


def _read_prediction_ledger_count(state: FileState) -> BackupLedgerCount:
    """Read only the sacred-ledger count from one immutable backup."""

    connection: sqlite3.Connection | None = None
    try:
        transaction_error = _unsafe_adjacent_transaction_file(state)
        if transaction_error is not None:
            return BackupLedgerCount(
                name=state.name,
                prediction_ledger_rows=None,
                error=transaction_error,
            )
        uri = state.path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.execute("PRAGMA query_only=ON")
        table = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'prediction_ledger'"
        ).fetchone()
        if table is None:
            return BackupLedgerCount(
                name=state.name,
                prediction_ledger_rows=None,
                error="prediction_ledger table is missing",
            )
        count = int(connection.execute("SELECT COUNT(*) FROM prediction_ledger").fetchone()[0])
        return BackupLedgerCount(
            name=state.name,
            prediction_ledger_rows=count,
            error=None,
        )
    except (OSError, sqlite3.Error) as exc:
        return BackupLedgerCount(
            name=state.name,
            prediction_ledger_rows=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if connection is not None:
            connection.close()


def validate_sqlite_backup(path_value: str | Path) -> BackupValidation:
    """Validate one real backup without opening the live database."""

    path = Path(path_value).absolute()
    if not os.path.lexists(path):
        return BackupValidation(
            name=path.name,
            mode="sqlite_mode_ro_immutable_quick_check",
            ok=False,
            result=("backup does not exist",),
            matches=None,
            prediction_ledger_rows=None,
        )
    state = _state(path)
    if state.reparse_or_symlink or state.kind != "file":
        return BackupValidation(
            name=state.name,
            mode="sqlite_mode_ro_immutable_quick_check",
            ok=False,
            result=("backup is not a real regular file",),
            matches=None,
            prediction_ledger_rows=None,
        )
    return _validate_sqlite_backup(state)


def approval_marker_path(plan: BackupRetentionPlan) -> Path:
    """Return the fixed marker path owned by this backup directory."""

    return plan.backups_dir / APPROVAL_MARKER_NAME


def _approval_policy(plan: BackupRetentionPlan) -> dict[str, Any]:
    if plan.kind != "prune" or plan.keep_latest is None:
        raise BackupRetentionApprovalError("only a prune plan can approve automatic retention")
    return {
        "policy_version": POLICY_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config_sha256": plan.config.sha256,
        "kind": plan.kind,
        "keep_latest": plan.keep_latest,
    }


def _empty_approval_marker() -> dict[str, Any]:
    return {
        "marker_version": APPROVAL_MARKER_VERSION,
        "approval": None,
    }


def _read_approval_marker(marker: Path) -> dict[str, Any]:
    if not os.path.lexists(marker):
        return _empty_approval_marker()
    state = _state(marker)
    if state.reparse_or_symlink or state.kind != "file":
        raise BackupRetentionApprovalError(f"approval marker is unsafe: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupRetentionApprovalError(f"approval marker is invalid: {marker}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("marker_version") != APPROVAL_MARKER_VERSION
        or (payload.get("approval") is not None and not isinstance(payload.get("approval"), dict))
    ):
        raise BackupRetentionApprovalError(f"approval marker is invalid: {marker}")
    return payload


def is_backup_retention_auto_approved(
    plan: BackupRetentionPlan,
    *,
    approval_marker: str | Path | None = None,
) -> bool:
    """Return whether the marker approves this policy, config hash and N."""

    marker = Path(approval_marker) if approval_marker is not None else approval_marker_path(plan)
    return _read_approval_marker(marker).get("approval") == _approval_policy(plan)


def _write_approval_marker(plan: BackupRetentionPlan, marker: Path) -> None:
    expected = approval_marker_path(plan)
    if marker.absolute() != expected.absolute():
        raise BackupRetentionApprovalError(f"approval marker must be the fixed BBDD-owned path: {expected}")
    _, backups, resolved_backups = _prepare_layout(plan.bbdd_dir)
    if marker.parent != backups or marker.parent.resolve(strict=True) != resolved_backups:
        raise BackupRetentionApprovalError(f"approval marker escapes the backup directory: {marker}")
    if os.path.lexists(marker):
        current = _state(marker)
        if current.reparse_or_symlink or current.kind != "file":
            raise BackupRetentionApprovalError(f"approval marker is unsafe: {marker}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{APPROVAL_MARKER_NAME}.",
        suffix=".tmp",
        dir=backups,
    )
    temporary = Path(temporary_name)
    try:
        temporary_state = _state(temporary)
        if (
            temporary.parent.resolve(strict=True) != resolved_backups
            or temporary_state.reparse_or_symlink
            or temporary_state.kind != "file"
        ):
            raise BackupRetentionApprovalError(f"temporary approval marker is unsafe: {temporary}")
        payload = {
            "marker_version": APPROVAL_MARKER_VERSION,
            "approval": _approval_policy(plan),
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        if _read_approval_marker(marker).get("approval") != _approval_policy(plan):
            raise BackupRetentionApprovalError(f"approval marker verification failed: {marker}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary_state = _state(temporary)
            if (
                temporary.parent == backups
                and not temporary_state.reparse_or_symlink
                and temporary_state.kind == "file"
            ):
                temporary.unlink()


def _plan_token_payload(plan: BackupRetentionPlan) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "kind": plan.kind,
        "bbdd_dir": str(plan.bbdd_dir),
        "backups_dir": str(plan.backups_dir),
        "config_sha256": plan.config.sha256,
        "keep_latest": plan.keep_latest,
        "required_keep": plan.required_keep.token_dict() if plan.required_keep else None,
        "cleanup_targets": list(plan.cleanup_targets),
        "keep": [entry.token_dict() for entry in plan.keep],
        "delete": [entry.token_dict() for entry in plan.delete],
        "protected": [entry.token_dict() for entry in plan.protected],
        "retained_backup_validation": (
            plan.retained_backup_validation.as_dict() if plan.retained_backup_validation else None
        ),
        "ledger_counts": [observation.as_dict() for observation in plan.ledger_counts],
        "blocked": list(plan.blocked),
    }


def _with_token(plan: BackupRetentionPlan) -> BackupRetentionPlan:
    encoded = json.dumps(
        _plan_token_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return replace(plan, token=hashlib.sha256(encoded).hexdigest())


def _resolve_required_keep(
    required_keep: str | Path | None,
    backups: Path,
    managed: Sequence[tuple[str, FileState]],
) -> FileState | None:
    if required_keep is None:
        return None
    supplied = Path(required_keep)
    target = (supplied if supplied.is_absolute() else backups / supplied).absolute()
    if target.parent != backups or Path(target.name).name != target.name:
        raise UnsafeBackupPathError(f"required keep must be an exact BBDD/backups child: {target}")
    by_name = {state.name: state for _, state in managed}
    state = by_name.get(target.name)
    if state is None or state.path.absolute() != target:
        raise UnsafeBackupPathError(f"required keep is not a canonical managed backup: {target}")
    return state


def plan_backup_retention(
    bbdd_dir: str | Path = DEFAULT_BBDD_DIR,
    *,
    keep_latest: int | None = None,
    required_keep: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    _owned_lock: FileState | None = None,
) -> BackupRetentionPlan:
    """Return a pure preview for timestamped ``BBDD/backups/*.db`` files."""

    config = load_retention_config(config_path)
    keep_count = _validate_positive_keep(config.default_keep if keep_latest is None else keep_latest)
    bbdd, backups, _ = _prepare_layout(bbdd_dir)
    protected = _protected_states(bbdd)
    live_identities = _live_identities(protected)
    managed: list[tuple[str, FileState]] = []
    kept: list[PlanEntry] = []
    blocked = _live_safety_blocks(bbdd, protected)

    for child in sorted(backups.iterdir(), key=lambda path: path.name):
        state = _state(child)
        if state.name == RETENTION_LOCK_NAME:
            if _owned_lock is not None and state.token_dict() == _owned_lock.token_dict():
                continue
            kept.append(PlanEntry(state, ("active_or_stale_retention_lock",)))
            blocked.append("retention_lock_present")
            continue
        stamp = _backup_timestamp(state.name)
        reasons: list[str] = []
        if state.reparse_or_symlink:
            reasons.append("symlink_or_reparse")
            blocked.append(f"unsafe_backup_entry:{state.name}")
        if state.kind != "file":
            reasons.append("non_regular_file")
        if state.identity in live_identities:
            reasons.append("hardlink_to_live_database")
            blocked.append(f"hardlink_to_live_database:{state.name}")
        if not state.name.lower().endswith(".db"):
            reasons.append("non_db_file")
        elif stamp is None:
            reasons.append("unmanaged_name")
        if reasons:
            kept.append(PlanEntry(state, tuple(sorted(set(reasons)))))
        else:
            assert stamp is not None
            managed.append((stamp, state))

    managed.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    required_state = _resolve_required_keep(required_keep, backups, managed)
    selected: list[FileState] = []
    if required_state is not None:
        selected.append(required_state)
    for _, state in managed:
        if len(selected) >= keep_count:
            break
        if required_state is not None and state.name == required_state.name:
            continue
        selected.append(state)
    recent_names = {state.name for state in selected}
    retained_state = required_state or (managed[0][1] if managed else None)
    retained_validation = _validate_sqlite_backup(retained_state) if retained_state is not None else None
    if retained_validation is not None and not retained_validation.ok:
        blocked.append(f"retained_backup_integrity_failed:{retained_validation.name}")

    ledger_counts = tuple(_read_prediction_ledger_count(state) for _, state in managed)
    for observation in ledger_counts:
        if observation.prediction_ledger_rows is None:
            blocked.append(f"prediction_ledger_unreadable:{observation.name}")
    retained_ledger_rows = retained_validation.prediction_ledger_rows if retained_validation is not None else None
    if retained_ledger_rows is not None:
        assert retained_validation is not None
        for observation in ledger_counts:
            candidate_rows = observation.prediction_ledger_rows
            if candidate_rows is not None and candidate_rows > retained_ledger_rows:
                blocked.append(
                    "prediction_ledger_regression:"
                    f"retained={retained_validation.name}:{retained_ledger_rows}:"
                    f"candidate={observation.name}:{candidate_rows}"
                )

    deleted: list[PlanEntry] = []
    required_entry: PlanEntry | None = None
    for _, state in sorted(managed, key=lambda item: item[1].name):
        if state.name in recent_names:
            keep_reasons = ("required_current_snapshot",) if required_state is state else ("recent",)
            entry = PlanEntry(state, keep_reasons)
            kept.append(entry)
            if required_state is state:
                required_entry = entry
        else:
            deleted.append(PlanEntry(state, ("expired",)))

    return _with_token(
        BackupRetentionPlan(
            kind="prune",
            bbdd_dir=bbdd,
            backups_dir=backups,
            config=config,
            keep_latest=keep_count,
            required_keep=required_entry,
            cleanup_targets=(),
            keep=tuple(sorted(kept, key=lambda entry: entry.name)),
            delete=tuple(deleted),
            protected=protected,
            retained_backup_validation=retained_validation,
            ledger_counts=ledger_counts,
            blocked=tuple(sorted(set(blocked))),
            token="",
        )
    )


def plan_extra_cleanup(
    targets: Sequence[str],
    bbdd_dir: str | Path = DEFAULT_BBDD_DIR,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    _owned_lock: FileState | None = None,
) -> BackupRetentionPlan:
    """Preview explicitly selected, allowlisted legacy extras in ``BBDD``."""

    config = load_retention_config(config_path)
    selected = tuple(sorted(targets))
    if not selected:
        raise BackupRetentionError("cleanup requires at least one explicit --target")
    if len(selected) != len(set(selected)):
        raise BackupRetentionError("cleanup targets must not contain duplicates")
    for name in selected:
        if Path(name).name != name or name not in config.cleanup_targets or name not in HARD_CODED_CLEANUP_TARGETS:
            raise BackupRetentionError(f"cleanup target is not allowlisted: {name}")

    bbdd, backups, _ = _prepare_layout(bbdd_dir)
    protected = list(_protected_states(bbdd))
    live_identities = _live_identities(protected)
    kept: list[PlanEntry] = []
    deleted: list[PlanEntry] = []
    blocked = _live_safety_blocks(bbdd, protected)
    if os.path.lexists(backups / RETENTION_LOCK_NAME):
        lock_state = _state(backups / RETENTION_LOCK_NAME)
        if _owned_lock is None or lock_state.token_dict() != _owned_lock.token_dict():
            blocked.append("retention_lock_present")
    selected_set = set(selected)
    for name in config.cleanup_targets:
        path = bbdd / name
        if not os.path.lexists(path):
            continue
        state = _state(path)
        if name not in selected_set:
            kept.append(PlanEntry(state, ("not_selected",)))
            continue
        reasons: list[str] = []
        if state.reparse_or_symlink:
            reasons.append("symlink_or_reparse")
            blocked.append(f"unsafe_cleanup_entry:{name}")
        if state.kind != "file":
            reasons.append("non_regular_file")
            blocked.append(f"unsafe_cleanup_entry:{name}")
        if state.identity in live_identities:
            reasons.append("hardlink_to_live_database")
            blocked.append(f"hardlink_to_live_database:{name}")
        if reasons:
            kept.append(PlanEntry(state, tuple(sorted(set(reasons)))))
        else:
            deleted.append(PlanEntry(state, ("explicit_cleanup",)))

    return _with_token(
        BackupRetentionPlan(
            kind="cleanup",
            bbdd_dir=bbdd,
            backups_dir=backups,
            config=config,
            keep_latest=None,
            required_keep=None,
            cleanup_targets=selected,
            keep=tuple(sorted(kept, key=lambda entry: entry.name)),
            delete=tuple(sorted(deleted, key=lambda entry: entry.name)),
            protected=tuple(protected),
            retained_backup_validation=None,
            ledger_counts=(),
            blocked=tuple(sorted(set(blocked))),
            token="",
        )
    )


def _refresh_plan(
    plan: BackupRetentionPlan,
    *,
    owned_lock: FileState | None = None,
) -> BackupRetentionPlan:
    if plan.kind == "prune":
        return plan_backup_retention(
            plan.bbdd_dir,
            keep_latest=plan.keep_latest,
            required_keep=(plan.required_keep.state.path if plan.required_keep else None),
            config_path=plan.config.path,
            _owned_lock=owned_lock,
        )
    return plan_extra_cleanup(
        plan.cleanup_targets,
        plan.bbdd_dir,
        config_path=plan.config.path,
        _owned_lock=owned_lock,
    )


def _validate_delete_target(plan: BackupRetentionPlan, entry: PlanEntry) -> Path:
    state = entry.state
    if Path(state.name).name != state.name or state.name in LIVE_PROTECTED_NAMES or state.name == "BLACKBOX":
        raise UnsafeBackupPathError(f"forbidden deletion target: {state.name}")
    bbdd, backups, resolved_backups = _prepare_layout(plan.bbdd_dir)
    if plan.kind == "prune":
        if _backup_timestamp(state.name) is None or not state.name.lower().endswith(".db"):
            raise UnsafeBackupPathError(f"non-canonical backup target: {state.name}")
        target = backups / state.name
        expected_parent = backups
        resolved_parent = resolved_backups
    else:
        if state.name not in plan.cleanup_targets or state.name not in HARD_CODED_CLEANUP_TARGETS:
            raise UnsafeBackupPathError(f"non-allowlisted cleanup target: {state.name}")
        target = bbdd / state.name
        expected_parent = bbdd
        resolved_parent = bbdd.resolve(strict=True)
    if target.parent != expected_parent or target.resolve(strict=False).parent != resolved_parent:
        raise UnsafeBackupPathError(f"deletion target escapes its exact root: {target}")
    if not os.path.lexists(target):
        raise StaleBackupPlanError(f"deletion target disappeared: {target}")
    current = _state(target)
    if current.reparse_or_symlink or current.kind != "file":
        raise UnsafeBackupPathError(f"deletion target is not a real regular file: {target}")
    if current.token_dict() != state.token_dict():
        raise StaleBackupPlanError(f"deletion target changed after preview: {target}")
    live_identities = _live_identities(_protected_states(bbdd))
    if current.identity in live_identities:
        raise UnsafeBackupPathError(f"deletion target is linked to live database state: {target}")
    return target


def _validate_protected_unchanged(plan: BackupRetentionPlan) -> None:
    fresh = _protected_states(plan.bbdd_dir)
    approved = [entry.token_dict() for entry in plan.protected]
    current = [entry.token_dict() for entry in fresh]
    if current != approved:
        raise StaleBackupPlanError("live database, sidecar, or BLACKBOX state changed after preview")


def _validate_required_keep_unchanged(plan: BackupRetentionPlan) -> None:
    entry = plan.required_keep
    if entry is None:
        return
    target = entry.state.path
    if target.parent != plan.backups_dir or not os.path.lexists(target):
        raise StaleBackupPlanError("required current snapshot disappeared or escaped BBDD/backups")
    current = _state(target)
    if current.reparse_or_symlink or current.kind != "file" or current.token_dict() != entry.state.token_dict():
        raise StaleBackupPlanError("required current snapshot identity changed after planning")


@contextmanager
def _exclusive_backup_lock(backups: Path, token: str) -> Iterator[FileState]:
    """Serialize snapshot publication and every manual/automatic deletion."""

    lock = backups / RETENTION_LOCK_NAME
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    lock_state: FileState | None = None
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise UnsafeBackupPathError(f"retention lock already exists: {lock}") from exc
    except OSError as exc:
        raise UnsafeBackupPathError(f"cannot acquire retention lock: {lock}") from exc
    body_error: BaseException | None = None
    try:
        payload = json.dumps(
            {"pid": os.getpid(), "token": token},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        lock_state = _state(lock)
        if lock_state.reparse_or_symlink or lock_state.kind != "file":
            raise UnsafeBackupPathError(f"retention lock is unsafe: {lock}")
        yield lock_state
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_error: BackupRetentionError | None = None
        try:
            os.close(descriptor)
        except OSError as exc:
            cleanup_error = UnsafeBackupPathError(f"could not close retention lock descriptor; lock may remain: {lock}")
            cleanup_error.__cause__ = exc
        if os.path.lexists(lock):
            try:
                current = _state(lock)
                if (
                    lock_state is None
                    or current.identity != lock_state.identity
                    or current.reparse_or_symlink
                    or current.kind != "file"
                ):
                    raise UnsafeBackupPathError(f"retention lock changed while held; left in place: {lock}")
                lock.unlink()
            except (OSError, BackupRetentionError) as exc:
                if cleanup_error is None:
                    cleanup_error = UnsafeBackupPathError(
                        f"could not safely remove retention lock; left in place: {lock}"
                    )
                    cleanup_error.__cause__ = exc
        elif cleanup_error is None:
            cleanup_error = UnsafeBackupPathError(
                f"retention lock disappeared while held; mutual exclusion was lost: {lock}"
            )
        if cleanup_error is not None:
            if body_error is not None:
                body_error.add_note(str(cleanup_error))
            else:
                raise cleanup_error


@contextmanager
def backup_operation_lock(bbdd_dir: str | Path = DEFAULT_BBDD_DIR) -> Iterator[FileState]:
    """Hold the BBDD-owned lock from before snapshot creation through pruning."""

    _, backups, _ = _prepare_layout(bbdd_dir)
    with _exclusive_backup_lock(backups, "backup_database") as lock_state:
        yield lock_state


@contextmanager
def _apply_lock(
    plan: BackupRetentionPlan,
    held_lock: FileState | None,
) -> Iterator[FileState]:
    if held_lock is None:
        with _exclusive_backup_lock(plan.backups_dir, plan.token) as lock_state:
            yield lock_state
        return
    _validate_held_lock(plan.backups_dir, held_lock)
    yield held_lock


def _validate_held_lock(backups: Path, held_lock: FileState) -> None:
    """Prove that the caller still owns the exact lock filesystem object."""

    lock = backups / RETENTION_LOCK_NAME
    if held_lock.path.absolute() != lock.absolute():
        raise UnsafeBackupPathError("caller-owned backup lock belongs to another directory")
    if not os.path.lexists(lock):
        raise UnsafeBackupPathError("caller-owned backup lock disappeared")
    try:
        current = _state(lock)
    except OSError as exc:
        raise UnsafeBackupPathError("caller-owned backup lock cannot be inspected") from exc
    if current.reparse_or_symlink or current.kind != "file" or current.token_dict() != held_lock.token_dict():
        raise UnsafeBackupPathError("caller-owned backup lock identity changed")


def apply_backup_retention(
    plan: BackupRetentionPlan,
    confirmation_token: str,
    *,
    record_approval: bool = False,
    automatic: bool = False,
    _held_lock: FileState | None = None,
) -> BackupApplyResult:
    """Apply an unchanged plan; automatic use requires an exact approval."""

    if not _CONFIRM_TOKEN_RE.fullmatch(confirmation_token):
        raise StaleBackupPlanError("confirmation must be the exact lowercase SHA-256 preview token")
    if record_approval and automatic:
        raise BackupRetentionApprovalError("manual approval and automatic mode are mutually exclusive")
    if (record_approval or automatic) and plan.kind != "prune":
        raise BackupRetentionApprovalError("cleanup cannot approve or use automatic backup retention")
    deleted: list[str] = []
    deleted_bytes = 0
    marker: Path | None = None
    # The exclusive lock covers the authoritative refresh, prevalidation and
    # every unlink.  That refresh ignores exactly our reserved lock filename;
    # a pre-existing/concurrent lock cannot be acquired in the first place.
    with _apply_lock(plan, _held_lock) as owned_lock:
        _validate_held_lock(plan.backups_dir, owned_lock)
        current = _refresh_plan(plan, owned_lock=owned_lock)
        if confirmation_token != plan.token or confirmation_token != current.token:
            raise StaleBackupPlanError("backup plan changed; generate and confirm a fresh preview token")
        if current.blocked:
            raise UnsafeBackupPathError("backup plan is safety-blocked: " + ", ".join(current.blocked))
        _validate_protected_unchanged(current)
        _validate_required_keep_unchanged(current)
        if record_approval or automatic:
            marker = approval_marker_path(current)
            marker_payload = _read_approval_marker(marker)
            if automatic and current.required_keep is None:
                raise BackupRetentionApprovalError("automatic retention requires the current snapshot as required_keep")
            if automatic and marker_payload.get("approval") != _approval_policy(current):
                raise BackupRetentionApprovalError(
                    "automatic retention requires a marker matching policy, config hash and N"
                )
        # Validate every target before the first unlink, then validate each
        # again immediately before its own unlink.  No recursive delete/glob
        # can reach this point.
        for entry in current.delete:
            _validate_held_lock(current.backups_dir, owned_lock)
            _validate_delete_target(current, entry)
        for entry in current.delete:
            target = entry.state.path
            try:
                _validate_held_lock(current.backups_dir, owned_lock)
                _validate_required_keep_unchanged(current)
                target = _validate_delete_target(current, entry)
                target.unlink()
                deleted.append(entry.name)
                deleted_bytes += entry.state.size
                _validate_held_lock(current.backups_dir, owned_lock)
            except BackupRetentionError as exc:
                if deleted:
                    raise BackupDeletionError(
                        "deletion batch stopped after mutual-exclusion or state validation failed; "
                        f"already deleted: {deleted}",
                        deleted=tuple(deleted),
                        deleted_bytes=deleted_bytes,
                    ) from exc
                raise
            except OSError as exc:
                raise BackupDeletionError(
                    f"deletion batch stopped at {target}; already deleted: {deleted}",
                    deleted=tuple(deleted),
                    deleted_bytes=deleted_bytes,
                ) from exc
        if record_approval:
            assert marker is not None
            try:
                _validate_held_lock(current.backups_dir, owned_lock)
                _write_approval_marker(current, marker)
            except (BackupRetentionError, OSError) as exc:
                raise BackupDeletionError(
                    "retention completed but automatic-approval marker finalization failed",
                    deleted=tuple(deleted),
                    deleted_bytes=deleted_bytes,
                ) from exc
    return BackupApplyResult(
        kind=current.kind,
        token=current.token,
        deleted=tuple(deleted),
        deleted_bytes=deleted_bytes,
        approval_marker=marker,
        automatic=automatic,
    )


def run_automatic_backup_retention(
    bbdd_dir: str | Path = DEFAULT_BBDD_DIR,
    *,
    keep_latest: int | None = None,
    required_keep: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    _held_lock: FileState | None = None,
) -> BackupAutomaticResult:
    """Apply an approved policy or return a successful first-run preview/no-op."""

    plan = plan_backup_retention(
        bbdd_dir,
        keep_latest=keep_latest,
        required_keep=required_keep,
        config_path=config_path,
        _owned_lock=_held_lock,
    )
    marker = approval_marker_path(plan)
    if plan.required_keep is None:
        return BackupAutomaticResult(
            plan=plan,
            applied=False,
            reason="current_snapshot_required",
            result=None,
            approval_marker=marker,
        )
    if not is_backup_retention_auto_approved(plan, approval_marker=marker):
        return BackupAutomaticResult(
            plan=plan,
            applied=False,
            reason="approval_required",
            result=None,
            approval_marker=marker,
        )
    result = apply_backup_retention(
        plan,
        plan.token,
        automatic=True,
        _held_lock=_held_lock,
    )
    return BackupAutomaticResult(
        plan=plan,
        applied=True,
        reason=None,
        result=result,
        approval_marker=marker,
    )

"""Safe, preview-first retention plans for CS2 model artifacts and runs.

Planning is deliberately side-effect free.  Applying a plan requires the exact
token produced from the current filesystem state.  The first manual approval
records the policy in a marker; later automatic applications are accepted only
while that marker still matches the policy version, kind and retention count.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from .promotion import PromotionValidationError, deployment_lock, read_model_pointer, sha256_file

POLICY_VERSION = 4
APPROVAL_MARKER_VERSION = 1
QUARANTINE_MANIFEST_VERSION = 2
QUARANTINE_DIRECTORY = ".retention-quarantine"
DEFAULT_REGISTRY_KEEP = 0
DEFAULT_RUNS_KEEP = 2
DEFAULT_APPROVAL_MARKER = Path(__file__).resolve().parents[1] / "artifacts" / "registry" / ".retention-approved.json"

RetentionKind = Literal["registry", "runs"]


class RetentionError(RuntimeError):
    """Base error for a retention operation that was refused safely."""


class StaleRetentionPlanError(RetentionError):
    """The supplied token no longer describes the filesystem."""


class RetentionApprovalError(RetentionError):
    """Automatic deletion was attempted without a matching approval marker."""


class UnsafeRetentionPathError(RetentionError):
    """A path failed the final, immediately-before-delete safety checks."""


class RetentionCleanupBlockedError(UnsafeRetentionPathError):
    """A validated target could not be inspected because cleanup is blocked."""

    def __init__(self, message: str, *, target: Path, blocked_path: Path) -> None:
        super().__init__(message)
        self.target = target
        self.blocked_path = blocked_path


class RetentionApplyError(RetentionError):
    """A staged retention mutation failed with structured recovery details."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        kind: RetentionKind,
        token: str,
        quarantine: Path | None,
        staged: Iterable[str] = (),
        rolled_back: Iterable[str] = (),
        deleted: Iterable[str] = (),
        pending: Iterable[str] = (),
        failed: str | None = None,
        recovery_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.kind = kind
        self.token = token
        self.quarantine = quarantine
        self.staged = tuple(staged)
        self.rolled_back = tuple(rolled_back)
        self.deleted = tuple(deleted)
        self.pending = tuple(pending)
        self.failed = failed
        self.recovery_required = recovery_required

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON details for the CLI and recovery tooling."""

        return {
            "phase": self.phase,
            "kind": self.kind,
            "token": self.token,
            "quarantine": str(self.quarantine) if self.quarantine is not None else None,
            "staged": list(self.staged),
            "rolled_back": list(self.rolled_back),
            "deleted": list(self.deleted),
            "pending": list(self.pending),
            "failed": self.failed,
            "recovery_required": self.recovery_required,
        }


@dataclass(frozen=True)
class RetentionEntry:
    """One immediate child directory and why it is kept or deleted."""

    name: str
    path: Path
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RetentionPlan:
    """Immutable preview of one component's retention decision."""

    kind: RetentionKind
    root: Path
    keep_last: int
    keep: tuple[RetentionEntry, ...]
    delete: tuple[RetentionEntry, ...]
    warnings: tuple[str, ...]
    token: str
    latest_pointer: Path | None = None
    last_good_pointer: Path | None = None
    master_manifest: Path | None = None
    referenced_run_ids: tuple[str, ...] = ()
    deferred_cleanup_names: tuple[str, ...] = ()
    enforce_registry_health: bool = False

    @property
    def keep_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.keep)

    @property
    def delete_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.delete)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "kind": self.kind,
            "root": str(self.root),
            "keep_last": self.keep_last,
            "keep": [entry.as_dict() for entry in self.keep],
            "delete": [entry.as_dict() for entry in self.delete],
            "warnings": list(self.warnings),
            "token": self.token,
            "deferred_cleanup_names": list(self.deferred_cleanup_names),
            "enforce_registry_health": self.enforce_registry_health,
        }


@dataclass(frozen=True)
class RetentionApplyResult:
    """Directories removed after a validated application."""

    kind: RetentionKind
    token: str
    deleted: tuple[str, ...]
    marker: Path
    automatic: bool
    staged: tuple[str, ...] = ()
    rolled_back: tuple[str, ...] = ()
    quarantine: Path | None = None
    resumed: bool = False


def _is_canonical_registry_id(name: str) -> bool:
    try:
        parsed = datetime.strptime(name, "%Y%m%d_%H%M%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y%m%d_%H%M%SZ") == name


def _is_canonical_run_id(name: str) -> bool:
    try:
        parsed = datetime.strptime(name, "%Y-%m-%d_%H%M%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%d_%H%M%SZ") == name


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = int(getattr(status, "st_file_attributes", 0))
    return path.is_symlink() or bool(file_attributes & reparse_flag)


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _inside_root(path: Path, resolved_root: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(resolved_root)
    except (OSError, RuntimeError):
        return False


def _valid_token(token: str) -> bool:
    return len(token) == 64 and all(character in "0123456789abcdef" for character in token)


def _validated_direct_directory(path: Path, parent: Path, *, label: str) -> Path:
    if path.parent != parent or _is_reparse_or_symlink(path) or not _is_real_directory(path):
        raise UnsafeRetentionPathError(f"{label} is unsafe: {path}")
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeRetentionPathError(f"Cannot resolve {label} safely: {path}") from exc
    if resolved_path.parent != resolved_parent or not resolved_path.is_relative_to(resolved_parent):
        raise UnsafeRetentionPathError(f"{label} escapes its parent: {path}")
    return resolved_path


def _validated_quarantine_root(root: Path) -> tuple[Path, Path]:
    quarantine_root = root / QUARANTINE_DIRECTORY
    resolved = _validated_direct_directory(quarantine_root, root, label="Quarantine root")
    return quarantine_root, resolved


def _quarantine_warnings(root: Path) -> tuple[str, ...]:
    quarantine_root = root / QUARANTINE_DIRECTORY
    if not os.path.lexists(quarantine_root):
        return ()
    try:
        _validated_quarantine_root(root)
        entries = tuple(quarantine_root.iterdir())
    except (OSError, RetentionError):
        return ("unsafe_pending_quarantine",)
    warnings: list[str] = []
    for entry in entries:
        if _valid_token(entry.name):
            try:
                _validated_direct_directory(entry, quarantine_root, label="Quarantine operation")
            except RetentionError:
                warnings.append(f"unsafe_pending_quarantine:{entry.name}")
            else:
                warnings.append(f"pending_quarantine:{entry.name}")
        else:
            warnings.append(f"unsafe_pending_quarantine_entry:{entry.name}")
    return tuple(sorted(warnings))


def _validate_registry_health(root: Path, *, deletion_names: Iterable[str] = ()) -> tuple[str, ...]:
    """Validate both pointers, runtime hash and protected target identities."""

    latest_path = root / "latest.json"
    if _is_reparse_or_symlink(latest_path):
        raise PromotionValidationError("latest pointer must not be a symlink/reparse point")
    latest = read_model_pointer(root, "latest")
    protected = {latest.version}

    last_good_path = root / "last_good.json"
    if os.path.lexists(last_good_path):
        if _is_reparse_or_symlink(last_good_path):
            raise PromotionValidationError("last_good pointer must not be a symlink/reparse point")
        protected.add(read_model_pointer(root, "last_good").version)

    runtime = root.parent / "model.pkl"
    if _is_reparse_or_symlink(runtime) or not runtime.is_file():
        raise PromotionValidationError("runtime model.pkl is missing or unsafe")
    if sha256_file(runtime) != latest.sha256:
        raise PromotionValidationError("runtime model hash does not match latest")

    conflicts = sorted(protected.intersection(str(name) for name in deletion_names))
    if conflicts:
        raise PromotionValidationError("retention targets a live pointer: " + ", ".join(conflicts))
    return tuple(sorted(protected))


def _validate_keep_last(value: int) -> int:
    if isinstance(value, bool) or value < 0:
        raise ValueError("keep_last must be an integer >= 0")
    return int(value)


def _prepare_root(root: str | Path) -> tuple[Path, Path, bool]:
    lexical_root = Path(root).absolute()
    if not _is_real_directory(lexical_root) and not _is_reparse_or_symlink(lexical_root):
        raise FileNotFoundError(f"Retention root is not a directory: {lexical_root}")
    root_is_unsafe = _is_reparse_or_symlink(lexical_root)
    try:
        resolved_root = lexical_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeRetentionPathError(f"Cannot resolve retention root safely: {lexical_root}") from exc
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Retention root is not a directory: {lexical_root}")
    return lexical_root, resolved_root, root_is_unsafe


def _read_pointer_version(
    pointer: Path,
    *,
    root: Path,
    canonical: Any,
    legacy_key: str,
) -> tuple[str | None, str | None]:
    """Read only an in-root version ID; artifact paths are never trusted."""

    try:
        if pointer.parent.resolve(strict=False) != root.resolve(strict=True):
            return None, f"outside_pointer:{pointer.name}"
    except (OSError, RuntimeError):
        return None, f"unsafe_pointer:{pointer.name}"
    if not pointer.exists() and not os.path.lexists(pointer):
        return None, None
    if _is_reparse_or_symlink(pointer):
        return None, f"unsafe_pointer:{pointer.name}"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, f"invalid_pointer:{pointer.name}"
    if not isinstance(payload, dict):
        return None, f"invalid_pointer:{pointer.name}"

    preferred = payload.get("version")
    legacy = payload.get(legacy_key)
    if preferred is None:
        preferred = legacy
    elif legacy is not None and legacy != preferred:
        return None, f"conflicting_pointer_versions:{pointer.name}"
    if not isinstance(preferred, str) or not canonical(preferred):
        return None, f"invalid_pointer_version:{pointer.name}"

    target = root / preferred
    if not os.path.lexists(target):
        return None, f"missing_pointer_target:{pointer.name}"
    if _is_reparse_or_symlink(target):
        return preferred, f"unsafe_pointer_target:{pointer.name}"
    if not _is_real_directory(target):
        return None, f"invalid_pointer_target:{pointer.name}"
    return preferred, None


def _read_master_run_id(
    manifest: Path | None,
    *,
    root: Path,
) -> tuple[str | None, str | None]:
    if manifest is None or (not manifest.exists() and not os.path.lexists(manifest)):
        return None, None
    if _is_reparse_or_symlink(manifest):
        return None, f"unsafe_master_manifest:{manifest.name}"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, f"invalid_master_manifest:{manifest.name}"
    run_id = payload.get("last_run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not _is_canonical_run_id(run_id):
        return None, f"invalid_master_run_id:{manifest.name}"
    if not os.path.lexists(root / run_id):
        return None, f"missing_master_run:{run_id}"
    if _is_reparse_or_symlink(root / run_id):
        return run_id, f"unsafe_master_run:{run_id}"
    if not _is_real_directory(root / run_id):
        return None, f"invalid_master_run:{run_id}"
    return run_id, None


def _build_entries(
    *,
    kind: RetentionKind,
    root: Path,
    resolved_root: Path,
    keep_last: int,
    protected: dict[str, set[str]],
    safety_warnings: Iterable[str],
    root_is_unsafe: bool,
) -> tuple[tuple[RetentionEntry, ...], tuple[RetentionEntry, ...], tuple[str, ...]]:
    canonical = _is_canonical_registry_id if kind == "registry" else _is_canonical_run_id
    children: list[tuple[Path, bool, bool, bool]] = []
    for child in root.iterdir():
        unsafe_link = _is_reparse_or_symlink(child)
        real_directory = _is_real_directory(child)
        if not real_directory and not unsafe_link:
            continue
        inside = _inside_root(child, resolved_root)
        children.append((child, unsafe_link, real_directory, inside))

    managed_names = sorted(
        (
            child.name
            for child, unsafe_link, real_directory, inside in children
            if canonical(child.name) and not unsafe_link and real_directory and inside
        ),
        reverse=True,
    )
    recent_names = set(managed_names[:keep_last])
    warnings = sorted(set(str(item) for item in safety_warnings if item))
    if root_is_unsafe:
        warnings.append("unsafe_retention_root")
    safety_blocked = bool(warnings)

    keep_entries: list[RetentionEntry] = []
    delete_entries: list[RetentionEntry] = []
    for child, unsafe_link, _, inside in sorted(children, key=lambda row: row[0].name):
        reasons = set(protected.get(child.name, set()))
        managed = canonical(child.name)
        if child.name in recent_names:
            reasons.add("recent")
        if not managed:
            reasons.add("unmanaged_name")
        if unsafe_link:
            reasons.add("symlink_or_reparse")
        if not inside:
            reasons.add("outside_root")
        if safety_blocked:
            reasons.add("safety_blocked")

        if reasons:
            keep_entries.append(RetentionEntry(child.name, child, tuple(sorted(reasons))))
        else:
            delete_entries.append(RetentionEntry(child.name, child, ("expired",)))
    return tuple(keep_entries), tuple(delete_entries), tuple(sorted(set(warnings)))


def _token_payload(plan: RetentionPlan) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "kind": plan.kind,
        "root": str(plan.root),
        "keep_last": plan.keep_last,
        "keep": [{"name": entry.name, "reasons": list(entry.reasons)} for entry in plan.keep],
        "delete": [{"name": entry.name, "reasons": list(entry.reasons)} for entry in plan.delete],
        "warnings": list(plan.warnings),
        "referenced_run_ids": list(plan.referenced_run_ids),
        "deferred_cleanup_names": list(plan.deferred_cleanup_names),
        "enforce_registry_health": plan.enforce_registry_health,
    }


def _with_token(plan: RetentionPlan) -> RetentionPlan:
    payload = json.dumps(_token_payload(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    token = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return replace(plan, token=token)


def plan_registry_retention(
    registry_root: str | Path,
    *,
    keep_last: int = DEFAULT_REGISTRY_KEEP,
    latest_pointer: str | Path | None = None,
    last_good_pointer: str | Path | None = None,
    deferred_cleanup_names: Iterable[str] = (),
    enforce_registry_health: bool = True,
) -> RetentionPlan:
    """Preview registry retention without changing pointers or directories."""

    keep_last = _validate_keep_last(keep_last)
    root, resolved_root, root_is_unsafe = _prepare_root(registry_root)
    latest_path = Path(latest_pointer) if latest_pointer else root / "latest.json"
    last_good_path = Path(last_good_pointer) if last_good_pointer else root / "last_good.json"
    protected: dict[str, set[str]] = {}
    warnings: list[str] = list(_quarantine_warnings(root))
    if enforce_registry_health:
        try:
            _validate_registry_health(root)
        except (OSError, PromotionValidationError) as exc:
            warnings.append(f"invalid_registry_health:{type(exc).__name__}")
    for pointer, legacy_key, reason in (
        (latest_path, "latest", "latest"),
        (last_good_path, "last_good", "last_good"),
    ):
        version, warning = _read_pointer_version(
            pointer,
            root=root,
            canonical=_is_canonical_registry_id,
            legacy_key=legacy_key,
        )
        if warning:
            warnings.append(warning)
        if version:
            protected.setdefault(version, set()).add(reason)

    deferred = tuple(sorted(set(str(value) for value in deferred_cleanup_names)))
    for name in deferred:
        if not _is_canonical_registry_id(name) or Path(name).name != name:
            warnings.append(f"unsafe_deferred_cleanup_name:{name}")
            continue
        if os.path.lexists(root / name):
            protected.setdefault(name, set()).add("cleanup_pending")

    keep, delete, plan_warnings = _build_entries(
        kind="registry",
        root=root,
        resolved_root=resolved_root,
        keep_last=keep_last,
        protected=protected,
        safety_warnings=warnings,
        root_is_unsafe=root_is_unsafe,
    )
    return _with_token(
        RetentionPlan(
            kind="registry",
            root=root,
            keep_last=keep_last,
            keep=keep,
            delete=delete,
            warnings=plan_warnings,
            token="",
            latest_pointer=latest_path,
            last_good_pointer=last_good_path,
            deferred_cleanup_names=deferred,
            enforce_registry_health=enforce_registry_health,
        )
    )


def plan_runs_retention(
    runs_root: str | Path,
    *,
    keep_last: int = DEFAULT_RUNS_KEEP,
    master_manifest: str | Path | None = None,
    referenced_run_ids: Iterable[str] = (),
    deferred_cleanup_names: Iterable[str] = (),
) -> RetentionPlan:
    """Preview run retention, preserving published and externally referenced runs."""

    keep_last = _validate_keep_last(keep_last)
    root, resolved_root, root_is_unsafe = _prepare_root(runs_root)
    manifest_path = Path(master_manifest) if master_manifest is not None else root.parent / "master" / "manifest.json"
    references = tuple(sorted(set(str(value) for value in referenced_run_ids)))
    protected: dict[str, set[str]] = {}
    warnings: list[str] = list(_quarantine_warnings(root))

    master_run_id, warning = _read_master_run_id(manifest_path, root=root)
    if warning:
        warnings.append(warning)
    if master_run_id:
        protected.setdefault(master_run_id, set()).add("master")

    for run_id in references:
        if Path(run_id).name != run_id:
            warnings.append(f"unsafe_referenced_run_id:{run_id}")
            continue
        if os.path.lexists(root / run_id):
            protected.setdefault(run_id, set()).add("referenced")

    deferred = tuple(sorted(set(str(value) for value in deferred_cleanup_names)))
    for name in deferred:
        if Path(name).name != name:
            warnings.append(f"unsafe_deferred_cleanup_name:{name}")
            continue
        if os.path.lexists(root / name):
            protected.setdefault(name, set()).add("cleanup_pending")

    keep, delete, plan_warnings = _build_entries(
        kind="runs",
        root=root,
        resolved_root=resolved_root,
        keep_last=keep_last,
        protected=protected,
        safety_warnings=warnings,
        root_is_unsafe=root_is_unsafe,
    )
    return _with_token(
        RetentionPlan(
            kind="runs",
            root=root,
            keep_last=keep_last,
            keep=keep,
            delete=delete,
            warnings=plan_warnings,
            token="",
            master_manifest=manifest_path,
            referenced_run_ids=references,
            deferred_cleanup_names=deferred,
        )
    )


def _refresh_plan(plan: RetentionPlan) -> RetentionPlan:
    if plan.kind == "registry":
        return plan_registry_retention(
            plan.root,
            keep_last=plan.keep_last,
            latest_pointer=plan.latest_pointer,
            last_good_pointer=plan.last_good_pointer,
            deferred_cleanup_names=plan.deferred_cleanup_names,
            enforce_registry_health=plan.enforce_registry_health,
        )
    return plan_runs_retention(
        plan.root,
        keep_last=plan.keep_last,
        master_manifest=plan.master_manifest,
        referenced_run_ids=plan.referenced_run_ids,
        deferred_cleanup_names=plan.deferred_cleanup_names,
    )


def _approval_policy(plan: RetentionPlan) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "kind": plan.kind,
        "keep_last": plan.keep_last,
    }


def is_retention_auto_approved(
    plan: RetentionPlan,
    *,
    approval_marker: str | Path = DEFAULT_APPROVAL_MARKER,
) -> bool:
    """Return whether automatic apply is approved for this policy/version/N.

    A missing marker is a normal ``False``.  An existing malformed or unsafe
    marker raises :class:`RetentionApprovalError` so callers do not mistake
    corruption for an ordinary first-run preview.
    """

    marker_payload = _read_marker(Path(approval_marker))
    return marker_payload["approvals"].get(plan.kind) == _approval_policy(plan)


def _read_marker(marker: Path) -> dict[str, Any]:
    if not marker.exists() and not os.path.lexists(marker):
        return {
            "marker_version": APPROVAL_MARKER_VERSION,
            "approvals": {},
        }
    if _is_reparse_or_symlink(marker):
        raise RetentionApprovalError(f"Approval marker is unsafe: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetentionApprovalError(f"Approval marker is invalid: {marker}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("marker_version") != APPROVAL_MARKER_VERSION
        or not isinstance(payload.get("approvals"), dict)
    ):
        raise RetentionApprovalError(f"Approval marker is invalid: {marker}")
    return payload


def _write_marker(marker: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(marker, payload, label="approval marker")


def _atomic_write_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    """Atomically write JSON through an exclusively-created same-dir temp."""

    target = path.absolute()
    parent = target.parent
    if _is_reparse_or_symlink(parent) or not _is_real_directory(parent):
        raise UnsafeRetentionPathError(f"{label.capitalize()} parent is unsafe: {parent}")
    try:
        resolved_parent = parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeRetentionPathError(f"Cannot resolve {label} parent safely: {parent}") from exc
    if os.path.lexists(target) and _is_reparse_or_symlink(target):
        raise UnsafeRetentionPathError(f"{label.capitalize()} is unsafe: {target}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        if temporary.parent.resolve(strict=True) != resolved_parent or _is_reparse_or_symlink(temporary):
            raise UnsafeRetentionPathError(f"Temporary {label} path is unsafe: {temporary}")
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if _is_reparse_or_symlink(target) or not target.is_file():
            raise UnsafeRetentionPathError(f"Published {label} is unsafe: {target}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_delete_path(plan: RetentionPlan, name: str) -> Path:
    canonical = _is_canonical_registry_id if plan.kind == "registry" else _is_canonical_run_id
    if not canonical(name) or Path(name).name != name:
        raise UnsafeRetentionPathError(f"Non-canonical deletion target: {name}")
    root, resolved_root, root_is_unsafe = _prepare_root(plan.root)
    if root_is_unsafe:
        raise UnsafeRetentionPathError(f"Retention root is unsafe: {root}")
    target = root / name
    if _is_reparse_or_symlink(target):
        raise UnsafeRetentionPathError(f"Refusing symlink/reparse target: {target}")
    if not _is_real_directory(target):
        raise UnsafeRetentionPathError(f"Deletion target is not a directory: {target}")
    if target.parent != root or not _inside_root(target, resolved_root):
        raise UnsafeRetentionPathError(f"Deletion target escapes root: {target}")
    return target


def _preflight_delete_tree(target: Path) -> None:
    """Verify an entire deletion tree is inspectable before any mutation.

    This cannot make recursive deletion transactional, but it catches denied
    ACLs, unreadable descendants, unsafe links/reparse points and obvious
    write-permission failures before the first target is removed.
    """

    directory_mode = os.R_OK if os.name == "nt" else os.R_OK | os.W_OK | os.X_OK
    if not os.access(target.parent, os.W_OK):
        raise RetentionCleanupBlockedError(
            f"Deletion parent is not writable: {target.parent}",
            target=target,
            blocked_path=target.parent,
        )

    stack = [target]
    while stack:
        directory = stack.pop()
        if not os.access(directory, directory_mode):
            raise RetentionCleanupBlockedError(
                f"Deletion directory is not accessible: {directory}",
                target=target,
                blocked_path=directory,
            )
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as exc:
            raise RetentionCleanupBlockedError(
                f"Cannot enumerate deletion tree: {directory}",
                target=target,
                blocked_path=directory,
            ) from exc

        for entry in entries:
            child = Path(entry.path)
            try:
                child_status = entry.stat(follow_symlinks=False)
                child_is_link = entry.is_symlink()
            except OSError as exc:
                raise RetentionCleanupBlockedError(
                    f"Cannot inspect deletion entry: {child}",
                    target=target,
                    blocked_path=child,
                ) from exc
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            file_attributes = int(getattr(child_status, "st_file_attributes", 0))
            if child_is_link or bool(file_attributes & reparse_flag):
                raise UnsafeRetentionPathError(f"Refusing nested symlink/reparse target: {child}")
            if stat.S_ISDIR(child_status.st_mode):
                stack.append(child)


def _tree_fingerprint(root: Path) -> str:
    """Hash every relative path and file byte without following links."""

    digest = hashlib.sha256()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise UnsafeRetentionPathError(f"Cannot fingerprint directory: {directory}") from exc
        for entry in entries:
            child = Path(entry.path)
            relative = child.relative_to(root).as_posix()
            try:
                child_status = entry.stat(follow_symlinks=False)
                child_is_link = entry.is_symlink()
            except OSError as exc:
                raise UnsafeRetentionPathError(f"Cannot fingerprint entry: {child}") from exc
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            attributes = int(getattr(child_status, "st_file_attributes", 0))
            if child_is_link or bool(attributes & reparse_flag):
                raise UnsafeRetentionPathError(f"Cannot fingerprint symlink/reparse entry: {child}")
            if stat.S_ISDIR(child_status.st_mode):
                digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
                stack.append(child)
                continue
            if not stat.S_ISREG(child_status.st_mode):
                raise UnsafeRetentionPathError(f"Cannot fingerprint non-regular entry: {child}")
            digest.update(b"F\0" + relative.encode("utf-8") + b"\0")
            try:
                with child.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise UnsafeRetentionPathError(f"Cannot fingerprint file: {child}") from exc
            digest.update(b"\0")
    return digest.hexdigest()


def _clear_windows_readonly_in_quarantine(target: Path) -> None:
    """Clear Windows read-only attributes only after a bundle is quarantined."""

    if os.name != "nt":
        return
    stack = [target]
    while stack:
        path = stack.pop()
        try:
            path_status = path.lstat()
        except OSError as exc:
            raise UnsafeRetentionPathError(f"Cannot inspect quarantined attributes: {path}") from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = int(getattr(path_status, "st_file_attributes", 0))
        if path.is_symlink() or bool(attributes & reparse_flag):
            raise UnsafeRetentionPathError(f"Refusing attribute change on symlink/reparse: {path}")
        try:
            os.chmod(path, path_status.st_mode | stat.S_IWRITE)
        except OSError as exc:
            raise UnsafeRetentionPathError(f"Cannot clear read-only attribute in quarantine: {path}") from exc
        try:
            updated_attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
        except OSError as exc:
            raise UnsafeRetentionPathError(f"Cannot verify quarantined attributes: {path}") from exc
        readonly_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
        if bool(updated_attributes & readonly_flag):
            raise UnsafeRetentionPathError(f"Read-only attribute remains in quarantine: {path}")
        if stat.S_ISDIR(path_status.st_mode):
            try:
                with os.scandir(path) as iterator:
                    stack.extend(Path(entry.path) for entry in iterator)
            except OSError as exc:
                raise UnsafeRetentionPathError(f"Cannot enumerate quarantined attributes: {path}") from exc


def _quarantine_paths(root: Path, token: str) -> tuple[Path, Path, Path]:
    if not _valid_token(token):
        raise RetentionApprovalError("Quarantine token must be a lowercase SHA-256")
    quarantine_root = root / QUARANTINE_DIRECTORY
    operation = quarantine_root / token
    return quarantine_root, operation, operation / "manifest.json"


def _write_quarantine_manifest(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(path, payload, label="quarantine manifest")


def _remove_empty_quarantine(operation: Path, manifest: Path, payload: dict[str, Any]) -> bool:
    """Best-effort cleanup which never recursively removes quarantine data."""

    try:
        entries = tuple(operation.iterdir())
    except OSError:
        return False
    if any(entry != manifest for entry in entries):
        return False
    try:
        manifest.unlink(missing_ok=True)
        try:
            operation.rmdir()
        except OSError:
            if operation.exists() and not os.path.lexists(manifest):
                try:
                    _write_quarantine_manifest(manifest, payload)
                except OSError:
                    pass
            return False
        quarantine_root = operation.parent
        if not any(quarantine_root.iterdir()):
            try:
                quarantine_root.rmdir()
            except OSError:
                pass
    except OSError:
        return False
    return True


def _new_quarantine_operation(
    plan: RetentionPlan,
    *,
    target_names: tuple[str, ...],
    identities: dict[str, str],
    approval_marker: Path,
    automatic: bool,
    approval_policy: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    root = plan.root
    quarantine_root, operation, manifest = _quarantine_paths(root, plan.token)
    if os.path.lexists(quarantine_root):
        if _is_reparse_or_symlink(quarantine_root) or not _is_real_directory(quarantine_root):
            raise UnsafeRetentionPathError(f"Quarantine root is unsafe: {quarantine_root}")
        try:
            existing = tuple(quarantine_root.iterdir())
        except OSError as exc:
            raise UnsafeRetentionPathError(f"Cannot inspect quarantine root: {quarantine_root}") from exc
        if existing:
            raise RetentionApplyError(
                "Pending retention quarantine must be resumed before a new apply",
                phase="recovery_required",
                kind=plan.kind,
                token=plan.token,
                quarantine=quarantine_root,
                pending=(entry.name for entry in existing),
                recovery_required=True,
            )
    else:
        quarantine_root.mkdir()
    _validated_quarantine_root(root)
    operation.mkdir()
    _validated_direct_directory(operation, quarantine_root, label="Quarantine operation")

    payload: dict[str, Any] = {
        "manifest_version": QUARANTINE_MANIFEST_VERSION,
        "policy_version": POLICY_VERSION,
        "kind": plan.kind,
        "token": plan.token,
        "root": str(root.resolve(strict=True)),
        "keep_last": plan.keep_last,
        "approval_marker": str(approval_marker.absolute()),
        "approval_policy": approval_policy,
        "automatic": automatic,
        "state": "staging",
        "targets": list(target_names),
        "identities": identities,
        "staged": [],
        "rolled_back": [],
        "deleted": [],
        "pending": [],
        "failure": None,
    }
    try:
        _write_quarantine_manifest(manifest, payload)
    except Exception as exc:
        payload.update(
            {
                "state": "setup_failed",
                "pending": [],
                "failure": {"target": None, "error": str(exc)},
            }
        )
        cleanup_ok = _remove_empty_quarantine(operation, manifest, payload)
        recovery_required = not cleanup_ok
        if recovery_required and not os.path.lexists(manifest):
            try:
                _write_quarantine_manifest(manifest, payload)
            except Exception:
                pass
        error = RetentionApplyError(
            "Quarantine manifest initialization failed before staging",
            phase="staging_setup",
            kind=plan.kind,
            token=plan.token,
            quarantine=operation if recovery_required else None,
            pending=(),
            recovery_required=recovery_required,
        )
        raise error from exc
    return operation, manifest, payload


def _stage_targets(
    plan: RetentionPlan,
    targets: tuple[tuple[str, Path], ...],
    *,
    approval_marker: Path,
    automatic: bool,
    approval_policy: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    names = tuple(name for name, _ in targets)
    identities = {name: _tree_fingerprint(source) for name, source in targets}
    operation, manifest, payload = _new_quarantine_operation(
        plan,
        target_names=names,
        identities=identities,
        approval_marker=approval_marker,
        automatic=automatic,
        approval_policy=approval_policy,
    )
    staged: list[tuple[str, Path, Path]] = []
    failed_name: str | None = None
    try:
        for name, source in targets:
            failed_name = name
            destination = operation / name
            if os.path.lexists(destination):
                raise UnsafeRetentionPathError(f"Quarantine target already exists: {destination}")
            os.replace(source, destination)
            staged.append((name, source, destination))
            if _tree_fingerprint(destination) != identities[name]:
                raise UnsafeRetentionPathError(f"Staged target identity changed: {name}")
            payload["staged"] = [item[0] for item in staged]
            payload["pending"] = [item[0] for item in staged]
            _write_quarantine_manifest(manifest, payload)
        payload.update(
            {
                "state": "staged",
                "staged": list(names),
                "pending": list(names),
                "failure": None,
            }
        )
        _write_quarantine_manifest(manifest, payload)
    except Exception as exc:
        rolled_back: list[str] = []
        rollback_failures: list[str] = []
        for name, source, destination in reversed(staged):
            try:
                if os.path.lexists(source):
                    raise FileExistsError(f"canonical rollback destination exists: {source}")
                os.replace(destination, source)
                rolled_back.append(name)
            except OSError as rollback_exc:
                rollback_failures.append(f"{name}: {rollback_exc}")

        remaining = [name for name, _, destination in staged if os.path.lexists(destination)]
        payload.update(
            {
                "state": "rollback_failed" if rollback_failures else "rolled_back",
                "staged": remaining,
                "rolled_back": rolled_back,
                "pending": remaining,
                "failure": {
                    "target": failed_name,
                    "error": str(exc),
                    "rollback_errors": rollback_failures,
                },
            }
        )
        cleanup_ok = False
        try:
            _write_quarantine_manifest(manifest, payload)
            if not rollback_failures:
                cleanup_ok = _remove_empty_quarantine(operation, manifest, payload)
        except OSError as manifest_exc:
            rollback_failures.append(f"manifest: {manifest_exc}")
        recovery_required = bool(rollback_failures or not cleanup_ok)
        error = RetentionApplyError(
            "Retention staging failed; canonical names were restored"
            if not recovery_required
            else "Retention staging recovery is incomplete; inspect quarantine before retrying",
            phase="staging",
            kind=plan.kind,
            token=plan.token,
            quarantine=operation if recovery_required else None,
            staged=(name for name, _, _ in staged),
            rolled_back=rolled_back,
            pending=remaining,
            failed=failed_name,
            recovery_required=recovery_required,
        )
        raise error from exc
    return operation, manifest, payload


def _canonical_name(kind: RetentionKind, name: str) -> bool:
    validator = _is_canonical_registry_id if kind == "registry" else _is_canonical_run_id
    return validator(name) and Path(name).name == name


def _validate_staged_target(operation: Path, kind: RetentionKind, name: str) -> Path:
    if not _canonical_name(kind, name):
        raise UnsafeRetentionPathError(f"Invalid quarantined target name: {name}")
    quarantine_root = operation.parent
    root = quarantine_root.parent
    _validated_quarantine_root(root)
    _validated_direct_directory(operation, quarantine_root, label="Quarantine operation")
    target = operation / name
    _validated_direct_directory(target, operation, label="Quarantined target")
    return target


def _retention_failure(
    message: str,
    *,
    phase: str,
    kind: RetentionKind,
    token: str,
    operation: Path,
    names: tuple[str, ...],
    deleted: Iterable[str],
    failed: str | None,
) -> RetentionApplyError:
    deleted_names = tuple(name for name in names if name in set(deleted))
    pending = tuple(name for name in names if os.path.lexists(operation / name))
    return RetentionApplyError(
        message,
        phase=phase,
        kind=kind,
        token=token,
        quarantine=operation,
        staged=names,
        deleted=deleted_names,
        pending=pending,
        failed=failed,
        recovery_required=True,
    )


_PRE_DELETION_STATES = frozenset({"setup_failed", "staging", "rollback_failed", "rolled_back"})
_DELETION_STATES = frozenset(
    {"staged", "deletion_preflight_failed", "deleting", "deletion_failed", "marker_failed", "complete"}
)


def _validated_manifest_state(
    payload: dict[str, Any],
    *,
    kind: RetentionKind,
    manifest: Path,
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    state = payload.get("state")
    targets_value = payload.get("targets")
    identities_value = payload.get("identities")
    if not isinstance(state, str) or state not in _PRE_DELETION_STATES | _DELETION_STATES:
        raise UnsafeRetentionPathError(f"Quarantine manifest has invalid state: {manifest}")
    if (
        not isinstance(targets_value, list)
        or not targets_value
        or not all(isinstance(name, str) for name in targets_value)
    ):
        raise UnsafeRetentionPathError(f"Quarantine manifest has invalid targets: {manifest}")
    targets = tuple(targets_value)
    if len(set(targets)) != len(targets) or any(not _canonical_name(kind, name) for name in targets):
        raise UnsafeRetentionPathError(f"Quarantine manifest has unsafe targets: {manifest}")
    if not isinstance(identities_value, dict) or set(identities_value) != set(targets):
        raise UnsafeRetentionPathError(f"Quarantine manifest has invalid identities: {manifest}")
    identities: dict[str, str] = {}
    for name, identity in identities_value.items():
        if not isinstance(name, str) or not isinstance(identity, str) or not _valid_token(identity):
            raise UnsafeRetentionPathError(f"Quarantine manifest has invalid identity: {manifest}")
        identities[name] = identity

    lists: dict[str, tuple[str, ...]] = {}
    for field in ("staged", "pending", "rolled_back", "deleted"):
        value = payload.get(field, [])
        if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
            raise UnsafeRetentionPathError(f"Quarantine manifest has invalid {field}: {manifest}")
        normalized = tuple(value)
        if len(set(normalized)) != len(normalized) or not set(normalized).issubset(targets):
            raise UnsafeRetentionPathError(f"Quarantine manifest has unsafe {field}: {manifest}")
        lists[field] = normalized
    if not set(lists["pending"]).issubset(lists["staged"]):
        raise UnsafeRetentionPathError(f"Quarantine pending entries were never staged: {manifest}")
    if set(lists["pending"]).intersection(lists["deleted"]):
        raise UnsafeRetentionPathError(f"Quarantine entries are both pending and deleted: {manifest}")
    if state in _PRE_DELETION_STATES and lists["deleted"]:
        raise UnsafeRetentionPathError(f"Pre-deletion quarantine records deleted entries: {manifest}")
    return state, targets, identities


def _validate_canonical_recovery_target(root: Path, kind: RetentionKind, name: str) -> Path:
    if not _canonical_name(kind, name):
        raise UnsafeRetentionPathError(f"Invalid canonical recovery target: {name}")
    return_path = root / name
    _validated_direct_directory(return_path, root, label="Canonical recovery target")
    return return_path


def _rollback_partial_staging(
    *,
    root: Path,
    kind: RetentionKind,
    token: str,
    operation: Path,
    manifest: Path,
    payload: dict[str, Any],
    marker: Path,
) -> RetentionApplyResult:
    _, names, identities = _validated_manifest_state(payload, kind=kind, manifest=manifest)
    staged_targets: list[tuple[str, Path, Path]] = []
    canonical_names: list[str] = []
    for name in names:
        canonical = root / name
        quarantined = operation / name
        canonical_exists = os.path.lexists(canonical)
        quarantined_exists = os.path.lexists(quarantined)
        if canonical_exists == quarantined_exists:
            error = _retention_failure(
                "Staging recovery requires exactly one canonical/quarantined copy",
                phase="rollback_validation",
                kind=kind,
                token=token,
                operation=operation,
                names=names,
                deleted=(),
                failed=name,
            )
            raise error
        if canonical_exists:
            selected = _validate_canonical_recovery_target(root, kind, name)
            canonical_names.append(name)
        else:
            selected = _validate_staged_target(operation, kind, name)
            staged_targets.append((name, selected, canonical))
        if _tree_fingerprint(selected) != identities[name]:
            error = _retention_failure(
                "Staging recovery bundle identity does not match its sealed manifest",
                phase="rollback_validation",
                kind=kind,
                token=token,
                operation=operation,
                names=names,
                deleted=(),
                failed=name,
            )
            raise error

    restored = list(canonical_names)
    try:
        for name, quarantined, canonical in staged_targets:
            os.replace(quarantined, canonical)
            restored.append(name)
            payload.update(
                {
                    "state": "rollback_failed",
                    "staged": [candidate for candidate in names if os.path.lexists(operation / candidate)],
                    "pending": [candidate for candidate in names if os.path.lexists(operation / candidate)],
                    "rolled_back": restored,
                    "deleted": [],
                    "failure": None,
                }
            )
            _write_quarantine_manifest(manifest, payload)
    except Exception as exc:
        remaining = [name for name in names if os.path.lexists(operation / name)]
        restored = [name for name in names if os.path.lexists(root / name)]
        payload.update(
            {
                "state": "rollback_failed",
                "staged": remaining,
                "pending": remaining,
                "rolled_back": restored,
                "deleted": [],
                "failure": {"target": name, "error": str(exc)},
            }
        )
        try:
            _write_quarantine_manifest(manifest, payload)
        except OSError:
            pass
        error = RetentionApplyError(
            "Staging rollback remains incomplete; resume the same quarantine token",
            phase="rollback",
            kind=kind,
            token=token,
            quarantine=operation,
            staged=remaining,
            rolled_back=restored,
            pending=remaining,
            failed=name,
            recovery_required=True,
        )
        raise error from exc

    payload.update(
        {
            "state": "rolled_back",
            "staged": [],
            "pending": [],
            "rolled_back": list(names),
            "deleted": [],
            "failure": None,
        }
    )
    _write_quarantine_manifest(manifest, payload)
    cleanup_ok = _remove_empty_quarantine(operation, manifest, payload)
    return RetentionApplyResult(
        kind=kind,
        token=token,
        deleted=(),
        marker=marker,
        automatic=bool(payload.get("automatic", False)),
        staged=tuple(name for name, _, _ in staged_targets),
        rolled_back=names,
        quarantine=None if cleanup_ok else operation,
        resumed=True,
    )


def _delete_staged_operation(
    *,
    root: Path,
    kind: RetentionKind,
    token: str,
    operation: Path,
    manifest: Path,
    payload: dict[str, Any],
    marker: Path,
    automatic: bool,
    approval_policy: dict[str, Any],
    resumed: bool,
) -> RetentionApplyResult:
    state, names, _ = _validated_manifest_state(payload, kind=kind, manifest=manifest)
    if state not in _DELETION_STATES:
        raise UnsafeRetentionPathError(f"Quarantine is not in a deletion state: {manifest}")

    deleted_value = payload.get("deleted", [])
    deleted = [name for name in deleted_value if isinstance(name, str) and name in names]
    staged_targets: list[tuple[str, Path]] = []
    for name in names:
        canonical = root / name
        staged = operation / name
        if os.path.lexists(canonical):
            error = _retention_failure(
                "Canonical target reappeared while its quarantine is pending",
                phase="recovery_validation",
                kind=kind,
                token=token,
                operation=operation,
                names=names,
                deleted=deleted,
                failed=name,
            )
            raise error
        if os.path.lexists(staged):
            staged_targets.append((name, _validate_staged_target(operation, kind, name)))
        elif name not in deleted:
            deleted.append(name)

    try:
        for _, staged in staged_targets:
            _preflight_delete_tree(staged)
        for _, staged in staged_targets:
            _clear_windows_readonly_in_quarantine(staged)
    except Exception as exc:
        payload.update(
            {
                "state": "deletion_preflight_failed",
                "deleted": deleted,
                "pending": [name for name, _ in staged_targets],
                "failure": {"target": staged.name, "error": str(exc)},
            }
        )
        try:
            _write_quarantine_manifest(manifest, payload)
        except OSError:
            pass
        error = _retention_failure(
            "Quarantined targets failed deletion preflight; no staged target was removed",
            phase="deletion_preflight",
            kind=kind,
            token=token,
            operation=operation,
            names=names,
            deleted=deleted,
            failed=staged.name,
        )
        raise error from exc

    for name, staged in staged_targets:
        try:
            shutil.rmtree(staged)
            deleted.append(name)
            payload.update(
                {
                    "state": "deleting",
                    "deleted": deleted,
                    "pending": [candidate for candidate in names if os.path.lexists(operation / candidate)],
                    "failure": None,
                }
            )
            _write_quarantine_manifest(manifest, payload)
        except Exception as exc:
            if not os.path.lexists(staged) and name not in deleted:
                deleted.append(name)
            payload.update(
                {
                    "state": "deletion_failed",
                    "deleted": deleted,
                    "pending": [candidate for candidate in names if os.path.lexists(operation / candidate)],
                    "failure": {"target": name, "error": str(exc)},
                }
            )
            try:
                _write_quarantine_manifest(manifest, payload)
            except OSError:
                pass
            error = _retention_failure(
                "Deletion from quarantine failed; retry the recorded quarantine token",
                phase="deletion",
                kind=kind,
                token=token,
                operation=operation,
                names=names,
                deleted=deleted,
                failed=name,
            )
            raise error from exc

    try:
        marker_payload = _read_marker(marker)
        if automatic and marker_payload["approvals"].get(kind) != approval_policy:
            raise RetentionApprovalError("Automatic retention approval changed while quarantine was active")
        if not automatic:
            approvals = dict(marker_payload["approvals"])
            approvals[kind] = approval_policy
            _write_marker(
                marker,
                {
                    "marker_version": APPROVAL_MARKER_VERSION,
                    "approvals": approvals,
                },
            )
    except Exception as exc:
        payload.update(
            {
                "state": "marker_failed",
                "deleted": list(names),
                "pending": [],
                "failure": {"target": None, "error": str(exc)},
            }
        )
        try:
            _write_quarantine_manifest(manifest, payload)
        except OSError:
            pass
        error = _retention_failure(
            "Retention data was removed but approval-marker finalization failed",
            phase="marker",
            kind=kind,
            token=token,
            operation=operation,
            names=names,
            deleted=names,
            failed=None,
        )
        raise error from exc

    payload.update({"state": "complete", "deleted": list(names), "pending": [], "failure": None})
    try:
        _write_quarantine_manifest(manifest, payload)
    except OSError:
        pass
    cleanup_ok = _remove_empty_quarantine(operation, manifest, payload)
    return RetentionApplyResult(
        kind=kind,
        token=token,
        deleted=names,
        marker=marker,
        automatic=automatic,
        staged=names,
        quarantine=None if cleanup_ok else operation,
        resumed=resumed,
    )


def _read_quarantine_operation(
    root: Path,
    kind: RetentionKind,
    token: str,
) -> tuple[Path, Path, dict[str, Any]]:
    quarantine_root, operation, manifest = _quarantine_paths(root, token)
    _validated_quarantine_root(root)
    resolved_operation = _validated_direct_directory(operation, quarantine_root, label="Quarantine operation")
    if _is_reparse_or_symlink(manifest) or not manifest.is_file():
        raise UnsafeRetentionPathError(f"Quarantine manifest does not exist safely: {manifest}")
    try:
        resolved_manifest = manifest.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeRetentionPathError(f"Cannot resolve quarantine manifest safely: {manifest}") from exc
    if manifest.parent != operation or resolved_manifest.parent != resolved_operation:
        raise UnsafeRetentionPathError(f"Quarantine manifest escapes its operation: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsafeRetentionPathError(f"Quarantine manifest is invalid: {manifest}") from exc
    if not isinstance(payload, dict):
        raise UnsafeRetentionPathError(f"Quarantine manifest is invalid: {manifest}")
    if (
        payload.get("manifest_version") != QUARANTINE_MANIFEST_VERSION
        or payload.get("policy_version") != POLICY_VERSION
        or payload.get("kind") != kind
        or payload.get("token") != token
        or payload.get("root") != str(root.resolve(strict=True))
    ):
        raise UnsafeRetentionPathError(f"Quarantine manifest does not match this policy/root: {manifest}")
    return operation, manifest, payload


def _resume_retention_quarantine_locked(
    root: Path,
    kind: RetentionKind,
    token: str,
    *,
    approval_marker: Path,
    enforce_registry_health: bool,
) -> RetentionApplyResult:
    operation, manifest, payload = _read_quarantine_operation(root, kind, token)
    state, names, _ = _validated_manifest_state(payload, kind=kind, manifest=manifest)
    approval_policy = payload.get("approval_policy")
    automatic = payload.get("automatic")
    keep_last = payload.get("keep_last")
    sealed_marker = payload.get("approval_marker")
    expected_policy = {
        "policy_version": POLICY_VERSION,
        "kind": kind,
        "keep_last": keep_last,
    }
    if (
        not isinstance(keep_last, int)
        or isinstance(keep_last, bool)
        or keep_last < 0
        or approval_policy != expected_policy
        or not isinstance(automatic, bool)
        or not isinstance(sealed_marker, str)
        or Path(sealed_marker).absolute() != approval_marker.absolute()
    ):
        raise UnsafeRetentionPathError(f"Quarantine manifest has invalid approval metadata: {manifest}")
    if kind == "registry" and enforce_registry_health:
        _validate_registry_health(
            root,
            deletion_names=names if state in _DELETION_STATES else (),
        )
    if state in _PRE_DELETION_STATES:
        return _rollback_partial_staging(
            root=root,
            kind=kind,
            token=token,
            operation=operation,
            manifest=manifest,
            payload=payload,
            marker=approval_marker,
        )
    return _delete_staged_operation(
        root=root,
        kind=kind,
        token=token,
        operation=operation,
        manifest=manifest,
        payload=payload,
        marker=approval_marker,
        automatic=automatic,
        approval_policy=approval_policy,
        resumed=True,
    )


def resume_retention_quarantine(
    root: str | Path,
    token: str,
    *,
    kind: RetentionKind,
    approval_marker: str | Path,
    enforce_registry_health: bool = True,
) -> RetentionApplyResult:
    """Retry deletion/finalization for one validated quarantine operation."""

    lexical_root, _, root_is_unsafe = _prepare_root(root)
    if root_is_unsafe:
        raise UnsafeRetentionPathError(f"Retention root is unsafe: {lexical_root}")
    marker = Path(approval_marker)
    if kind == "registry":
        with deployment_lock(lexical_root):
            return _resume_retention_quarantine_locked(
                lexical_root,
                kind,
                token,
                approval_marker=marker,
                enforce_registry_health=enforce_registry_health,
            )
    return _resume_retention_quarantine_locked(
        lexical_root,
        kind,
        token,
        approval_marker=marker,
        enforce_registry_health=enforce_registry_health,
    )


def _apply_current_retention_plan(
    plan: RetentionPlan,
    token: str,
    *,
    approval_marker: str | Path = DEFAULT_APPROVAL_MARKER,
    auto: bool = False,
) -> RetentionApplyResult:
    current = _refresh_plan(plan)
    if token != plan.token or token != current.token:
        raise StaleRetentionPlanError("Retention plan changed; generate and approve a fresh preview token")
    if current.warnings:
        raise UnsafeRetentionPathError("Retention plan is safety-blocked: " + ", ".join(current.warnings))

    marker = Path(approval_marker)
    marker_payload = _read_marker(marker)
    expected_policy = _approval_policy(current)
    if auto and marker_payload["approvals"].get(current.kind) != expected_policy:
        raise RetentionApprovalError("Automatic retention requires a marker matching policy/version/N")

    # Validate and inspect every target before the first mutation. Only names
    # present in the freshly rebuilt plan can be atomically moved to quarantine.
    validated_targets: list[tuple[str, Path]] = []
    current_delete_names = set(current.delete_names)
    for entry in current.delete:
        if entry.name not in current_delete_names:
            raise UnsafeRetentionPathError(f"Target is absent from current deletion plan: {entry.name}")
        target = _validate_delete_path(current, entry.name)
        _preflight_delete_tree(target)
        validated_targets.append((entry.name, target))

    if current.kind == "registry" and current.enforce_registry_health:
        _validate_registry_health(
            current.root,
            deletion_names=(name for name, _ in validated_targets),
        )

    if not validated_targets:
        if not auto:
            approvals = dict(marker_payload["approvals"])
            approvals[current.kind] = expected_policy
            _write_marker(
                marker,
                {
                    "marker_version": APPROVAL_MARKER_VERSION,
                    "approvals": approvals,
                },
            )
        return RetentionApplyResult(
            kind=current.kind,
            token=current.token,
            deleted=(),
            marker=marker,
            automatic=auto,
        )

    operation, manifest, payload = _stage_targets(
        current,
        tuple(validated_targets),
        approval_marker=marker,
        automatic=auto,
        approval_policy=expected_policy,
    )
    return _delete_staged_operation(
        root=current.root,
        kind=current.kind,
        token=current.token,
        operation=operation,
        manifest=manifest,
        payload=payload,
        marker=marker,
        automatic=auto,
        approval_policy=expected_policy,
        resumed=False,
    )


def apply_retention_plan(
    plan: RetentionPlan,
    token: str,
    *,
    approval_marker: str | Path = DEFAULT_APPROVAL_MARKER,
    auto: bool = False,
) -> RetentionApplyResult:
    """Apply an exact current plan, refusing stale or unapproved deletion.

    Registry plans hold the deployment lock from the final plan refresh through
    atomic staging, quarantine deletion and approval-marker finalization. This
    prevents a concurrent promotion, rollback, or ``last_good`` update from
    making the refreshed keep-list stale during deletion.
    """

    if plan.kind == "registry":
        with deployment_lock(plan.root):
            return _apply_current_retention_plan(
                plan,
                token,
                approval_marker=approval_marker,
                auto=auto,
            )
    return _apply_current_retention_plan(
        plan,
        token,
        approval_marker=approval_marker,
        auto=auto,
    )


# Compact aliases for callers that prefer the domain nouns.
plan_registry = plan_registry_retention
plan_runs = plan_runs_retention
apply_plan = apply_retention_plan

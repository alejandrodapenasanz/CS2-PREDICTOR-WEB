"""Inspect, initialize, roll back, and safely prune CS2 model state."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from .cs2model.artifacts import ModelArtifact, load_artifact
    from .cs2model.cleanup_pending import (
        PendingCleanupStateError,
        reconcile_pending_cleanup,
        record_cleanup_failure,
    )
    from .cs2model.config import load_config
    from .cs2model.promotion import (
        ModelReference,
        PromotionValidationError,
        read_model_pointer,
        reference_for_version,
        rollback_last_good,
        set_last_good_version,
        sha256_file,
        write_model_pointer,
    )
    from .cs2model.retention import (
        QUARANTINE_DIRECTORY,
        RetentionApplyError,
        RetentionApplyResult,
        RetentionCleanupBlockedError,
        RetentionError,
        RetentionKind,
        RetentionPlan,
        apply_retention_plan,
        is_retention_auto_approved,
        plan_registry_retention,
        plan_runs_retention,
        resume_retention_quarantine,
    )
else:
    from cs2model.artifacts import ModelArtifact, load_artifact
    from cs2model.cleanup_pending import (
        PendingCleanupStateError,
        reconcile_pending_cleanup,
        record_cleanup_failure,
    )
    from cs2model.config import load_config
    from cs2model.promotion import (
        ModelReference,
        PromotionValidationError,
        read_model_pointer,
        reference_for_version,
        rollback_last_good,
        set_last_good_version,
        sha256_file,
        write_model_pointer,
    )
    from cs2model.retention import (
        QUARANTINE_DIRECTORY,
        RetentionApplyError,
        RetentionApplyResult,
        RetentionCleanupBlockedError,
        RetentionError,
        RetentionKind,
        RetentionPlan,
        apply_retention_plan,
        is_retention_auto_approved,
        plan_registry_retention,
        plan_runs_retention,
        resume_retention_quarantine,
    )

CS2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CS2_ROOT / "MODEL" / "config.yaml"
DEFAULT_REGISTRY = CS2_ROOT / "MODEL" / "artifacts" / "registry"
DEFAULT_PRODUCTION = CS2_ROOT / "MODEL" / "artifacts" / "model.pkl"
DEFAULT_RUNS = CS2_ROOT / "PIPELINE" / "runs"
DEFAULT_MASTER_MANIFEST = CS2_ROOT / "PIPELINE" / "master" / "manifest.json"
DEFAULT_DB = CS2_ROOT / "BBDD" / "cs2.db"

_RUN_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z_-])(\d{4}-\d{2}-\d{2}_\d{6}Z)(?![0-9A-Za-z_-])")
_CONFIRM_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_COLUMNS = frozenset({"run_id", "source_run_id", "source_file"})


class ModelManagementError(RuntimeError):
    """A management request was rejected before mutating production state."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("retention counts must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("retention count must be >= 0")
    return parsed


def _sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not _CONFIRM_TOKEN_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError("SHA-256 must contain exactly 64 lowercase hexadecimal characters")
    return normalized


def _model_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        "--registry-dir",
        dest="registry",
        default=str(DEFAULT_REGISTRY),
        help="Model registry root.",
    )
    parser.add_argument(
        "--production",
        default=str(DEFAULT_PRODUCTION),
        help="Runtime MODEL/artifacts/model.pkl copy.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Validate latest, runtime hash, and optional last_good.")
    _model_paths(status)

    initialize = commands.add_parser(
        "initialize-last-good",
        help="One-time migration: initialize an absent last_good from validated latest.",
    )
    _model_paths(initialize)

    rollback = commands.add_parser("rollback", help="Restore last_good and rotate the outgoing live model.")
    _model_paths(rollback)

    set_last_good = commands.add_parser(
        "set-last-good",
        help="Pin one validated registry version as last_good without changing latest/runtime.",
    )
    _model_paths(set_last_good)
    set_last_good.add_argument("--version", required=True, help="Canonical registry version YYYYMMDD_HHMMSSZ.")
    set_last_good.add_argument(
        "--expected-sha256",
        required=True,
        type=_sha256,
        help="Previously inspected SHA-256 of the exact rollback artifact.",
    )

    prune = commands.add_parser("prune", help="Preview retention, or apply an exactly confirmed preview.")
    prune.add_argument("--scope", choices=("registry", "runs", "all"), default="all")
    prune.add_argument("--config", default=str(DEFAULT_CONFIG))
    prune.add_argument("--registry", "--registry-dir", dest="registry", default=str(DEFAULT_REGISTRY))
    prune.add_argument("--runs", "--runs-dir", dest="runs", default=str(DEFAULT_RUNS))
    prune.add_argument("--master-manifest", default=str(DEFAULT_MASTER_MANIFEST))
    prune.add_argument("--db", default=str(DEFAULT_DB))
    prune.add_argument("--registry-keep", type=_non_negative_int, default=None)
    prune.add_argument("--runs-keep", type=_positive_int, default=None)
    prune.add_argument(
        "--pending-cleanup",
        default=None,
        help=(
            "Persistent cleanup-residue sidecar. Defaults to pending_cleanup.json "
            "beside the supplied registry artifacts directory."
        ),
    )
    action = prune.add_mutually_exclusive_group()
    action.add_argument(
        "--confirm",
        metavar="TOKEN",
        help=("Apply a manual preview. For --scope all use registry=<sha256>,runs=<sha256>."),
    )
    action.add_argument(
        "--auto",
        action="store_true",
        help="Apply the current plan only when its scope marker approves policy/version/N.",
    )
    action.add_argument(
        "--resume",
        metavar="TOKEN",
        type=_sha256,
        help="Retry one staged quarantine; requires --scope registry or --scope runs.",
    )
    prune.add_argument("--output", help="Also write the emitted JSON to this path.")
    return parser


def _validated_runtime_state(registry_value: str | Path, production_value: str | Path) -> dict[str, Any]:
    supplied_registry = Path(registry_value)
    if supplied_registry.is_symlink():
        raise ModelManagementError("registry root must not be a symlink")
    registry = supplied_registry.resolve(strict=True)
    production = Path(production_value).absolute()
    expected_production = registry.parent / "model.pkl"
    if production != expected_production.absolute():
        raise ModelManagementError(f"production path must be the registry sibling model.pkl: {expected_production}")
    if production.is_symlink():
        raise ModelManagementError("production model.pkl must not be a symlink")
    if not production.exists() or not production.is_file():
        raise ModelManagementError(f"production artifact does not exist: {production}")

    latest_path = registry / "latest.json"
    if latest_path.is_symlink():
        raise ModelManagementError("latest pointer must not be a symlink")
    latest = read_model_pointer(registry, "latest")
    runtime_hash = sha256_file(production)
    if runtime_hash != latest.sha256:
        raise ModelManagementError("runtime model hash does not match latest")

    last_good_path = registry / "last_good.json"
    if os.path.lexists(last_good_path) and last_good_path.is_symlink():
        raise ModelManagementError("last_good pointer must not be a symlink")
    last_good = read_model_pointer(registry, "last_good") if os.path.lexists(last_good_path) else None
    return {
        "ok": True,
        "registry": str(registry),
        "latest": latest.to_dict(),
        "runtime": {
            "path": str(production),
            "sha256": runtime_hash,
            "matches_latest": True,
        },
        "last_good": last_good.to_dict() if last_good else None,
    }


def _status(args: argparse.Namespace) -> dict[str, Any]:
    return {"command": "status", **_validated_runtime_state(args.registry, args.production)}


def _initialize_last_good(args: argparse.Namespace) -> dict[str, Any]:
    state = _validated_runtime_state(args.registry, args.production)
    registry = Path(args.registry).resolve(strict=True)
    latest = read_model_pointer(registry, "latest")
    pointer_path = registry / "last_good.json"
    if os.path.lexists(pointer_path):
        existing = read_model_pointer(registry, "last_good")
        return {
            "command": "initialize-last-good",
            "ok": True,
            "changed": False,
            "reason": "last_good_already_exists",
            "latest": latest.to_dict(),
            "last_good": existing.to_dict(),
            "runtime": state["runtime"],
        }

    write_model_pointer(registry, "last_good", latest)
    initialized = read_model_pointer(registry, "last_good")
    return {
        "command": "initialize-last-good",
        "ok": True,
        "changed": True,
        "reason": "last_good_initialized_from_latest",
        "latest": latest.to_dict(),
        "last_good": initialized.to_dict(),
        "runtime": state["runtime"],
    }


def _rollback(args: argparse.Namespace) -> dict[str, Any]:
    result = rollback_last_good(
        args.registry,
        production_path=args.production,
    )
    return {"command": "rollback", "ok": True, **result.to_dict()}


def _validate_loadable_rollback_target(reference: ModelReference) -> None:
    try:
        artifact = load_artifact(reference.artifact)
    except Exception as exc:
        raise ModelManagementError("last_good target cannot be loaded") from exc
    if not isinstance(artifact, ModelArtifact):
        raise ModelManagementError("last_good target is not a ModelArtifact")
    if not artifact.feature_columns or not artifact.components:
        raise ModelManagementError("last_good target has no model features/components")
    metadata = artifact.metadata if isinstance(artifact.metadata, Mapping) else {}
    reproducibility = metadata.get("reproducibility")
    recorded_seed = metadata.get("random_seed")
    if recorded_seed is None and isinstance(reproducibility, Mapping):
        recorded_seed = reproducibility.get("random_seed")
    if recorded_seed != 42:
        raise ModelManagementError("last_good target does not record the production seed 42")
    neutral_row = {column: 0.0 for column in artifact.feature_columns}
    try:
        probabilities = artifact.predict_proba_team1([neutral_row])
        probability = float(probabilities[0])
    except Exception as exc:
        raise ModelManagementError("last_good target cannot score a deterministic health row") from exc
    if len(probabilities) != 1 or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ModelManagementError("last_good target returned an invalid health probability")


def _set_last_good(args: argparse.Namespace) -> dict[str, Any]:
    _validated_runtime_state(args.registry, args.production)
    registry = Path(args.registry).resolve(strict=True)
    expected_latest = read_model_pointer(registry, "latest")
    last_good_path = registry / "last_good.json"
    expected_last_good = read_model_pointer(registry, "last_good") if last_good_path.exists() else None
    target = reference_for_version(registry, args.version)
    if target.sha256 != args.expected_sha256:
        raise ModelManagementError("expected SHA-256 does not match the selected registry version")
    _validate_loadable_rollback_target(target)
    result = set_last_good_version(
        registry,
        target.version,
        production_path=args.production,
        expected_latest=expected_latest,
        expected_last_good=expected_last_good,
        expected_target_sha256=args.expected_sha256,
    )
    return {"command": "set-last-good", "ok": True, **result.to_dict()}


def _walk_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_json_strings(key)
            yield from _walk_json_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_json_strings(child)


def _canonical_ids_in_text(value: str) -> set[str]:
    return set(_RUN_ID_PATTERN.findall(value))


def _safe_child_id(value: str) -> str | None:
    candidate = value.strip()
    if candidate and Path(candidate).name == candidate and candidate not in {".", ".."}:
        return candidate
    return None


def _source_file_ids(value: str) -> set[str]:
    output = _canonical_ids_in_text(value)
    parts = [part for part in value.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "runs":
            safe = _safe_child_id(parts[index + 1])
            if safe:
                output.add(safe)
    return output


def _manifest_run_references(runs_root: Path, master_manifest: Path) -> set[str]:
    root = runs_root.resolve(strict=True)
    candidates = [path for path in root.rglob("*.json") if "manifest" in path.name.lower()]
    if master_manifest.exists():
        candidates.append(master_manifest)

    references: set[str] = set()
    for path in sorted(set(candidates), key=lambda item: str(item)):
        if path.is_symlink():
            raise ModelManagementError(f"manifest must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        allowed = resolved.is_relative_to(root) or resolved == master_manifest.resolve(strict=True)
        if not allowed or not resolved.is_file():
            raise ModelManagementError(f"unsafe manifest path: {path}")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManagementError(f"manifest JSON is unreadable: {path}") from exc
        for text in _walk_json_strings(payload):
            references.update(_canonical_ids_in_text(text))
    return references


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _database_run_references(db_path: Path) -> set[str]:
    if not db_path.exists() or not db_path.is_file():
        raise ModelManagementError(f"database required for safe run retention: {db_path}")
    uri = db_path.resolve(strict=True).as_uri() + "?mode=ro"
    references: set[str] = set()
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        table_rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        for (table_name_raw,) in table_rows:
            table_name = str(table_name_raw)
            quoted_table = _quote_identifier(table_name)
            columns = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            selected_columns = [str(column[1]) for column in columns if str(column[1]).lower() in _REFERENCE_COLUMNS]
            for column_name in selected_columns:
                quoted_column = _quote_identifier(column_name)
                rows = connection.execute(
                    f"SELECT DISTINCT {quoted_column} FROM {quoted_table} WHERE {quoted_column} IS NOT NULL"
                ).fetchall()
                for (raw_value,) in rows:
                    if raw_value is None:
                        continue
                    value = str(raw_value).strip()
                    references.update(_canonical_ids_in_text(value))
                    if column_name.lower() == "source_file":
                        references.update(_source_file_ids(value))
                    else:
                        safe = _safe_child_id(value)
                        if safe:
                            references.add(safe)
    return references


def discover_run_references(
    runs_root: str | Path,
    master_manifest: str | Path,
    db_path: str | Path,
) -> set[str]:
    """Discover protected run IDs without opening SQLite for writes."""

    runs = Path(runs_root)
    master = Path(master_manifest)
    references = _manifest_run_references(runs, master)
    references.update(_database_run_references(Path(db_path)))
    return references


def _plan_payload(plan: RetentionPlan, marker: Path) -> dict[str, Any]:
    return {
        **plan.as_dict(),
        "approval_marker": str(marker),
    }


def _parse_confirmation(value: str, plan_kinds: tuple[str, ...]) -> dict[str, str]:
    raw = value.strip().lower()
    if len(plan_kinds) == 1 and _CONFIRM_TOKEN_PATTERN.fullmatch(raw):
        return {plan_kinds[0]: raw}

    parsed: dict[str, str] = {}
    for item in raw.split(","):
        key, separator, token = item.partition("=")
        if not separator or key not in {"registry", "runs"} or not _CONFIRM_TOKEN_PATTERN.fullmatch(token):
            raise ModelManagementError(
                "invalid confirmation; use a SHA256 token or registry=<sha256>,runs=<sha256> for scope all"
            )
        if key in parsed:
            raise ModelManagementError(f"duplicate confirmation token for {key}")
        parsed[key] = token
    if set(parsed) != set(plan_kinds):
        raise ModelManagementError("confirmation must contain exactly: " + ", ".join(plan_kinds))
    return parsed


def _pending_cleanup_path(args: argparse.Namespace, registry: Path, runs: Path) -> Path:
    supplied = getattr(args, "pending_cleanup", None)
    if supplied:
        return Path(supplied).absolute()
    if args.scope in {"registry", "all"}:
        return registry.absolute().parent / "pending_cleanup.json"
    return runs.absolute().parent / "pending_cleanup.json"


def _protected_registry_names(plan: RetentionPlan) -> set[str]:
    if plan.kind != "registry":
        return set()
    return {entry.name for entry in plan.keep if {"latest", "last_good"}.intersection(entry.reasons)}


def _deferred_cleanup_names(
    error: RetentionApplyError | RetentionCleanupBlockedError,
    plan: RetentionPlan,
) -> tuple[str, ...]:
    candidates: tuple[str, ...]
    if isinstance(error, RetentionCleanupBlockedError):
        candidates = (error.target.name,)
    else:
        candidates = error.pending
        if not candidates and error.failed is not None:
            candidates = (error.failed,)
    planned = set(plan.delete_names)
    names = tuple(name for name in candidates if name in planned)
    if not names:
        raise error
    return names


def _record_deferred_cleanup(
    state_path: Path,
    *,
    plan: RetentionPlan,
    error: RetentionApplyError | RetentionCleanupBlockedError,
) -> dict[str, Any]:
    names = _deferred_cleanup_names(error, plan)
    quarantine = error.quarantine if isinstance(error, RetentionApplyError) else None
    phase = error.phase if isinstance(error, RetentionApplyError) else "preflight"
    return record_cleanup_failure(
        state_path,
        kind=plan.kind,
        root=plan.root,
        names=names,
        phase=phase,
        error=error,
        quarantine=quarantine,
        protected_names=_protected_registry_names(plan),
    )


def _cleanup_deferred_payload(
    error: RetentionApplyError | RetentionCleanupBlockedError,
    plan: RetentionPlan,
) -> dict[str, Any]:
    names = _deferred_cleanup_names(error, plan)
    payload: dict[str, Any] = {
        "status": "cleanup_pending",
        "deleted": list(error.deleted) if isinstance(error, RetentionApplyError) else [],
        "pending": list(names),
        "failed": error.failed if isinstance(error, RetentionApplyError) else error.target.name,
        "phase": error.phase if isinstance(error, RetentionApplyError) else "preflight",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if isinstance(error, RetentionApplyError):
        payload["quarantine"] = str(error.quarantine) if error.quarantine is not None else None
        payload["recovery_required"] = error.recovery_required
    else:
        payload["blocked_path"] = str(error.blocked_path)
        payload["quarantine"] = None
        payload["recovery_required"] = False
    return payload


_NON_FATAL_CLEANUP_PHASES = frozenset(
    {
        "staging_setup",
        "staging",
        "deletion_preflight",
        "deletion",
        "rollback",
    }
)


def _prune(args: argparse.Namespace) -> dict[str, Any]:
    registry = Path(args.registry)
    runs = Path(args.runs)
    master = Path(args.master_manifest)
    pending_path = _pending_cleanup_path(args, registry, runs)
    pending_cleanup = reconcile_pending_cleanup(pending_path)

    if args.resume is not None:
        if args.scope == "all":
            raise ModelManagementError("--resume requires exactly one scope: registry or runs")
        kind: RetentionKind = "registry" if args.scope == "registry" else "runs"
        root = registry if kind == "registry" else runs
        marker = root.absolute() / ".retention-approved.json"
        result = resume_retention_quarantine(
            root,
            args.resume,
            kind=kind,
            approval_marker=marker,
        )
        pending_cleanup = reconcile_pending_cleanup(pending_path)
        return {
            "command": "prune",
            "ok": True,
            "scope": kind,
            "mode": "resume",
            "applied": {kind: _apply_result_payload(result)},
            "pending_cleanup": pending_cleanup,
        }

    config = load_config(args.config)
    registry_keep = args.registry_keep if args.registry_keep is not None else config.retention.registry_keep
    runs_keep = args.runs_keep if args.runs_keep is not None else config.retention.pipeline_runs_keep
    plans: dict[str, RetentionPlan] = {}
    markers: dict[str, Path] = {}

    pending_by_kind: dict[str, set[str]] = {"registry": set(), "runs": set()}
    for item in pending_cleanup.get("items", []):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        name = item.get("name")
        item_root = item.get("root")
        expected_root = registry if kind == "registry" else runs if kind == "runs" else None
        if (
            expected_root is not None
            and isinstance(name, str)
            and Path(name).name == name
            and isinstance(item_root, str)
            and Path(item_root).absolute() == expected_root.absolute()
        ):
            pending_by_kind[kind].add(name)

    if args.scope in {"registry", "all"}:
        plans["registry"] = plan_registry_retention(
            registry,
            keep_last=registry_keep,
            deferred_cleanup_names=pending_by_kind["registry"],
        )
        markers["registry"] = registry.absolute() / ".retention-approved.json"
    references: set[str] = set()
    if args.scope in {"runs", "all"}:
        references = discover_run_references(runs, master, args.db)
        plans["runs"] = plan_runs_retention(
            runs,
            keep_last=runs_keep,
            master_manifest=master,
            referenced_run_ids=references,
            deferred_cleanup_names=pending_by_kind["runs"],
        )
        markers["runs"] = runs.absolute() / ".retention-approved.json"

    plan_kinds = tuple(plans)
    response: dict[str, Any] = {
        "command": "prune",
        "ok": True,
        "scope": args.scope,
        "mode": "preview",
        "plans": {kind: _plan_payload(plan, markers[kind]) for kind, plan in plans.items()},
        "pending_cleanup": pending_cleanup,
    }
    if references:
        response["referenced_run_ids"] = sorted(references)

    recovery_warnings = {
        kind: [
            warning
            for warning in plan.warnings
            if warning.startswith(("pending_quarantine", "unsafe_pending_quarantine"))
        ]
        for kind, plan in plans.items()
    }
    recovery_warnings = {kind: warnings for kind, warnings in recovery_warnings.items() if warnings}
    if recovery_warnings:
        response["recovery_required"] = recovery_warnings
        if args.confirm is None and not args.auto:
            response["confirmation"] = None
            return response
        if args.auto:
            response.update(
                {
                    "applied": False,
                    "reason": "cleanup_pending",
                    "cleanup_deferred": True,
                }
            )
            return response
        recovery_kind = next(iter(recovery_warnings))
        warning = recovery_warnings[recovery_kind][0]
        _, separator, pending_token = warning.partition(":")
        plan = plans[recovery_kind]
        raise RetentionApplyError(
            "Pending quarantine must be recovered before planning another retention apply",
            phase="recovery_required",
            kind="registry" if recovery_kind == "registry" else "runs",
            token=pending_token if separator and _CONFIRM_TOKEN_PATTERN.fullmatch(pending_token) else plan.token,
            quarantine=plan.root / QUARANTINE_DIRECTORY,
            pending=recovery_warnings[recovery_kind],
            recovery_required=True,
        )

    safety_warnings = {kind: list(plan.warnings) for kind, plan in plans.items() if plan.warnings}
    if safety_warnings:
        response["safety_blocked"] = safety_warnings
        if args.confirm is None and not args.auto:
            response["confirmation"] = None
            return response
        raise ModelManagementError("retention is safety-blocked: " + json.dumps(safety_warnings, sort_keys=True))

    if args.confirm is None and not args.auto:
        if len(plan_kinds) == 1:
            kind = plan_kinds[0]
            response["confirmation"] = f"--confirm {plans[kind].token}"
        else:
            response["confirmation"] = (
                '--confirm "registry=' + plans["registry"].token + ",runs=" + plans["runs"].token + '"'
            )
        return response

    if args.auto:
        approval_required = [
            kind
            for kind, plan in plans.items()
            if not is_retention_auto_approved(
                plan,
                approval_marker=markers[kind],
            )
        ]
        if approval_required:
            response.update(
                {
                    "applied": False,
                    "reason": "approval_required",
                    "approval_required": approval_required,
                }
            )
            return response

    tokens = (
        {kind: plans[kind].token for kind in plan_kinds} if args.auto else _parse_confirmation(args.confirm, plan_kinds)
    )
    # Validate every manual token before the first plan can mutate a directory.
    for kind in plan_kinds:
        if tokens[kind] != plans[kind].token:
            raise ModelManagementError(f"stale or incorrect confirmation token for {kind}")

    applied: dict[str, Any] = {}
    cleanup_deferred = False
    for kind, plan in plans.items():
        try:
            result = apply_retention_plan(
                plan,
                tokens[kind],
                approval_marker=markers[kind],
                auto=bool(args.auto),
            )
        except RetentionCleanupBlockedError as exc:
            pending_cleanup = _record_deferred_cleanup(pending_path, plan=plan, error=exc)
            applied[kind] = _cleanup_deferred_payload(exc, plan)
            cleanup_deferred = True
            continue
        except RetentionApplyError as exc:
            if exc.phase not in _NON_FATAL_CLEANUP_PHASES:
                raise
            pending_cleanup = _record_deferred_cleanup(pending_path, plan=plan, error=exc)
            applied[kind] = _cleanup_deferred_payload(exc, plan)
            cleanup_deferred = True
            continue
        applied[kind] = _apply_result_payload(result)
        pending_cleanup = reconcile_pending_cleanup(pending_path)
    response["mode"] = "auto" if args.auto else "confirmed"
    response["applied"] = applied
    response["pending_cleanup"] = pending_cleanup
    if cleanup_deferred:
        response["cleanup_deferred"] = True
        response["warning"] = "Retention cleanup is pending; production pointers were preserved."
    return response


def _apply_result_payload(result: RetentionApplyResult) -> dict[str, Any]:
    return {
        "token": result.token,
        "deleted": list(result.deleted),
        "staged": list(result.staged),
        "rolled_back": list(result.rolled_back),
        "quarantine": str(result.quarantine) if result.quarantine is not None else None,
        "approval_marker": str(result.marker),
        "automatic": result.automatic,
        "resumed": result.resumed,
    }


def _emit(payload: dict[str, Any], output_path: str | None = None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            payload = _status(args)
        elif args.command == "initialize-last-good":
            payload = _initialize_last_good(args)
        elif args.command == "rollback":
            payload = _rollback(args)
        elif args.command == "set-last-good":
            payload = _set_last_good(args)
        else:
            payload = _prune(args)
        _emit(payload, getattr(args, "output", None))
        return 0
    except (
        ModelManagementError,
        PendingCleanupStateError,
        PromotionValidationError,
        RetentionError,
        FileNotFoundError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        error = {
            "command": getattr(args, "command", None),
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, RetentionApplyError):
            error["details"] = exc.as_dict()
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

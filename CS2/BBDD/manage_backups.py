"""CLI for previewing and exactly confirming CS2 database backup retention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .backup_retention import (
        DEFAULT_BBDD_DIR,
        DEFAULT_CONFIG,
        BackupDeletionError,
        BackupRetentionError,
        approval_marker_path,
        apply_backup_retention,
        is_backup_retention_auto_approved,
        plan_backup_retention,
        plan_extra_cleanup,
    )
else:
    from backup_retention import (
        DEFAULT_BBDD_DIR,
        DEFAULT_CONFIG,
        BackupDeletionError,
        BackupRetentionError,
        approval_marker_path,
        apply_backup_retention,
        is_backup_retention_auto_approved,
        plan_backup_retention,
        plan_extra_cleanup,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("keep must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview-first retention for CS2/BBDD backups; no confirmation means no deletion."
    )
    parser.add_argument("--bbdd-dir", default=str(DEFAULT_BBDD_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)

    prune = commands.add_parser("prune", help="Keep N newest timestamped .db backups and preview/delete older ones.")
    prune.add_argument("--keep", type=_positive_int, default=None)
    prune.add_argument(
        "--required-keep",
        help="Exact current snapshot path/name; mandatory evidence for automatic deletion.",
    )
    prune_mode = prune.add_mutually_exclusive_group()
    prune_mode.add_argument("--confirm", metavar="SHA256")
    prune_mode.add_argument(
        "--auto",
        action="store_true",
        help="Apply only when a prior manual confirmation approved this policy/config/N.",
    )

    cleanup = commands.add_parser(
        "cleanup",
        help="Preview/delete only explicitly selected allowlisted legacy extras.",
    )
    cleanup.add_argument(
        "--target",
        action="append",
        required=True,
        choices=("cs2_integrity_test.db", "cs2_dump.sql"),
        help="Exact BBDD-root extra; repeat to select both.",
    )
    cleanup.add_argument("--confirm", metavar="SHA256")
    return parser


def _response(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prune":
        plan = plan_backup_retention(
            args.bbdd_dir,
            keep_latest=args.keep,
            required_keep=args.required_keep,
            config_path=args.config,
        )
    else:
        plan = plan_extra_cleanup(
            args.target,
            args.bbdd_dir,
            config_path=args.config,
        )
    payload: dict[str, Any] = {
        "ok": True,
        "command": args.command,
        "mode": "preview",
        "plan": plan.as_dict(),
    }
    automatic = bool(getattr(args, "auto", False))
    if args.command == "prune":
        marker = approval_marker_path(plan)
        payload["approval_marker"] = str(marker)
        if automatic and plan.required_keep is None:
            payload.update(
                {
                    "applied": False,
                    "reason": "current_snapshot_required",
                }
            )
            return payload
        if automatic and not is_backup_retention_auto_approved(plan, approval_marker=marker):
            payload.update(
                {
                    "applied": False,
                    "reason": "approval_required",
                }
            )
            return payload
    if args.confirm is None and not automatic:
        payload["confirmation"] = f"--confirm {plan.token}"
        return payload
    applied = apply_backup_retention(
        plan,
        plan.token if automatic else args.confirm,
        record_approval=args.command == "prune" and not automatic,
        automatic=automatic,
    )
    payload.update(
        {
            "mode": "auto" if automatic else "confirmed",
            "applied": True,
            "deleted": list(applied.deleted),
            "deleted_bytes": applied.deleted_bytes,
            "automatic": applied.automatic,
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _response(args)
    except BackupDeletionError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "partial": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "deleted": list(exc.deleted),
                    "deleted_bytes": exc.deleted_bytes,
                    "warnings": list(getattr(exc, "__notes__", ())),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3
    except (BackupRetentionError, FileNotFoundError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "warnings": list(getattr(exc, "__notes__", ())),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

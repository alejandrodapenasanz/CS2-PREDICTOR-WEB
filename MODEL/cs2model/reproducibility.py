"""Deterministic seeds and immutable experiment fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ProjectConfig


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_value(item) for item in value), key=lambda item: repr(item))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _stable_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def dataset_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = _stable_value(row)
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def experiment_manifest(
    rows: list[dict[str, Any]],
    config: ProjectConfig,
    root: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_version": config.version,
        "config_sha256": config.sha256,
        "dataset_sha256": dataset_fingerprint(rows),
        "dataset_rows": len(rows),
        "random_seed": config.random_seed,
        "git": _git_state(root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(
                ("numpy", "scikit-learn", "lightgbm", "catboost", "xgboost", "optuna")
            ),
        },
        "arguments": arguments,
    }

"""Deterministic seeds and immutable experiment fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ProjectConfig


# Predictions from the same immutable artifact are considered equivalent when
# their absolute difference does not exceed this numerical-noise boundary.  The
# runtime still forces supported estimators to one inference thread so normal
# repeated calls are bit-stable; the tolerance covers native-library/CPU changes
# across fresh processes.
INFERENCE_DETERMINISM_ATOL = 1e-12

INFERENCE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_THREAD_PARAMETERS = ("n_jobs", "thread_count", "num_threads", "nthread")
_ESTIMATOR_CHILD_ATTRIBUTES = (
    "components",
    "odds_components",
    "mixed_components",
    "estimator",
    "estimator_",
    "base",
    "base_estimator",
    "base_estimator_",
    "calibrator",
    "calibrators",
    "calibrated_classifiers_",
    "rich_target_estimator",
    "steps",
    "named_steps",
    "named_estimators",
    "named_estimators_",
    "classifier",
    "classifier_",
    "regressor",
    "regressor_",
    "clf",
    "clf_",
    "model",
    "model_",
)


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def force_single_threaded_inference(root: Any) -> int:
    """Force supported estimators in a loaded artifact to one inference thread.

    Training objects on disk are left untouched.  The traversal includes fitted
    calibration wrappers and pipelines because their prediction path may use a
    cloned estimator rather than the constructor-level estimator.

    Returns the number of estimator parameters changed (already-single-threaded
    parameters are not counted).
    """

    for variable in INFERENCE_THREAD_ENVIRONMENT:
        os.environ[variable] = "1"

    visited: set[int] = set()
    changed = 0

    def visit(value: Any) -> None:
        nonlocal changed
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, complex, bool, Path)):
            return
        if isinstance(value, (np.ndarray, np.generic)):
            return

        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, Sequence):
            for child in value:
                visit(child)
            return

        get_params = getattr(value, "get_params", None)
        set_params = getattr(value, "set_params", None)
        if callable(get_params) and callable(set_params):
            try:
                parameters = get_params(deep=False)
            except (AttributeError, TypeError, ValueError):
                parameters = {}
            updates = {name: 1 for name in _THREAD_PARAMETERS if name in parameters and parameters[name] != 1}
            if updates:
                try:
                    set_params(**updates)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"No se pudo fijar inferencia monohilo en {type(value).__module__}.{type(value).__name__}"
                    ) from exc
                changed += len(updates)

        for attribute in _ESTIMATOR_CHILD_ATTRIBUTES:
            try:
                child = getattr(value, attribute)
            except (AttributeError, TypeError, ValueError):
                continue
            visit(child)

    visit(root)
    return changed


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
            "packages": _package_versions(("numpy", "scikit-learn", "lightgbm", "catboost", "xgboost", "optuna")),
        },
        "arguments": arguments,
    }

"""Narrow, audited storage-only compatibility for the pre-VAULT champion.

The approved service diff adds STATE_ROOT to path validation (plus formatting).
Estimator, symmetry, calibration and temporal guards are unchanged. Both ends
are pinned: this is not permission to load models under arbitrary changed code.
The original artifact and its fingerprint are never rewritten.
"""

from collections.abc import Mapping
import hashlib
from pathlib import Path

LEGACY_SERVICE_SHA256 = "af19e922e3ab8c1c1f84407b52df15e352c88b0405f5def352afef40124f2db8"
VAULT_SERVICE_SHA256 = "b32b22698ffdcacb31b5ce37d831f4ca35e5f4920ca6562ac50c9ff8c82b0a44"


def relocated_service_entry(root: Path, entry: Mapping[str, object]) -> dict[str, object]:
    """Accept only the exact reviewed service pair, allowing Git line endings."""
    result = dict(entry)
    if entry.get("path") != "src/modeling/service.py":
        return result
    if entry.get("sha256") != LEGACY_SERVICE_SHA256 or entry.get("size") != 13008:
        return result
    payload = (root / "src/modeling/service.py").read_bytes()
    if hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() != VAULT_SERVICE_SHA256:
        return result
    result.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return result

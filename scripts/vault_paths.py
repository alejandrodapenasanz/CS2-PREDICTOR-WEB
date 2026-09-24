"""Portable, dependency-free contract for private repository state.

Code stays in Git. Mutable state mirrors its former relative path under VAULT.
No filesystem mutation happens at import time. CLI: python scripts/vault_paths.py.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def state_path(relative: str, *, repository: Path = REPOSITORY_ROOT) -> Path:
    """Resolve a repository-relative private path without allowing traversal."""
    normalized = relative.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or PureWindowsPath(relative).drive or ".." in posix.parts or not posix.parts:
        raise ValueError("VAULT requires a confined repository-relative path")
    if posix.parts[0] not in {"CS2", "TENNIS", "TELEGRAM", "WEB", "runtime", "archive"}:
        raise ValueError("Unknown VAULT namespace")
    return Path(repository).resolve() / "VAULT" / Path(*posix.parts)


if __name__ == "__main__":
    print(json.dumps({"schema_version": 1, "vault": str(REPOSITORY_ROOT / "VAULT")}))

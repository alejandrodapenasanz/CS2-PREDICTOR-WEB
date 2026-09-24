"""VAULT migration and legacy pointer portability are part of the CS2 gate."""

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from MODEL.cs2model.promotion import PromotionValidationError, read_model_pointer, sha256_file
from BBDD import build_db


def test_roster_provenance_remains_relative_after_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest accepts VAULT rosters without referencing the old source-code tree."""
    state = tmp_path / "VAULT/CS2"
    roster = state / "PIPELINE/master/roster_history.json"
    roster.parent.mkdir(parents=True)
    roster.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_db, "STATE_ROOT", state)
    monkeypatch.setattr(build_db, "DEFAULT_ROSTER_HISTORY", roster)
    assert build_db.source_name(roster) == str(Path("PIPELINE/master/roster_history.json"))
    with sqlite3.connect(":memory:") as conn:
        assert build_db.insert_roster_history(conn.cursor(), {}) == 0


def test_standalone_storage_gate() -> None:
    """Run shared, stdlib-only migration fixtures without touching live state."""
    repository = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(repository / "scripts/test_vault.py")], check=True)


def test_legacy_pointer_uses_only_migration_proven_canonical_artifact(tmp_path: Path) -> None:
    """An old drive/path cannot redirect loading or bypass an artifact hash."""
    vault = tmp_path / "VAULT"
    registry = vault / "CS2/MODEL/artifacts/registry"
    version = "20260101_000000Z"
    artifact = registry / version / "model.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture, not deserialized")
    origin = tmp_path / "old repository"
    relative = Path("CS2/MODEL/artifacts/registry") / version / "model.pkl"
    (registry / "latest.json").write_text(
        json.dumps(
            {
                "latest": version,
                "artifact": str(origin / relative),
            }
        ),
        encoding="utf-8",
    )
    (vault / "migration.json").write_text(
        json.dumps(
            {
                "origin_root": str(origin),
                "entries": [
                    {
                        "status": "moved_verified",
                        "destination": "CS2/MODEL/artifacts",
                        "sha256": {f"registry/{version}/model.pkl": sha256_file(artifact)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert read_model_pointer(registry, "latest").artifact == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(PromotionValidationError, match="migration hash"):
        read_model_pointer(registry, "latest")

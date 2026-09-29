"""Portable state references retain bytes, temporal evidence and model identity."""

import hashlib
import json
from pathlib import Path

import pytest

from src import config
from src.modeling.vault_compatibility import (
    LEGACY_SERVICE_SHA256,
    relocated_service_entry,
)
from src.responsible_http import HttpCacheError, _validated_cached_object


def test_recorded_path_relocation_never_guesses_by_filename(tmp_path, monkeypatch):
    """An old origin is explicitly recorded; unrelated data cannot masquerade as it."""
    root = tmp_path / "new"
    source = root / "TENNIS"
    state = root / "VAULT/TENNIS"
    state.mkdir(parents=True)
    origin = tmp_path / "old"
    (state.parent / "migration.json").write_text(json.dumps({"origin_root": str(origin)}))
    monkeypatch.setattr(config, "PROJECT_ROOT", source)
    monkeypatch.setattr(config, "STATE_ROOT", state)
    for value in (
        "data/raw/a.bin",
        str(origin / "TENNIS/data/raw/a.bin"),
        str(origin / "VAULT/TENNIS/data/raw/a.bin"),
    ):
        assert config.resolve_state_reference(value) == state / "data/raw/a.bin"
    with pytest.raises(ValueError):
        config.resolve_state_reference(str(tmp_path / "unrelated/data/raw/a.bin"))
    with pytest.raises(ValueError):
        config.resolve_state_reference("../outside")


def test_relocated_http_cache_keeps_containment(tmp_path, monkeypatch):
    """An old absolute cache reference rebases but a sibling file stays forbidden."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "TENNIS")
    monkeypatch.setattr(config, "STATE_ROOT", tmp_path / "VAULT/TENNIS")
    parent = config.STATE_ROOT / "data/raw/_http_cache/objects"
    legacy = config.PROJECT_ROOT / "data/raw/_http_cache/objects/a.bin"
    assert _validated_cached_object(str(legacy), expected_parent=parent) == parent / "a.bin"
    with pytest.raises(HttpCacheError):
        _validated_cached_object(str(legacy), expected_parent=parent / "different")


def test_only_reviewed_model_service_transition_is_accepted(tmp_path):
    """Pin both code versions; never bypass the inference inventory generally."""
    root = Path(__file__).resolve().parents[1]
    old = {"path": "src/modeling/service.py", "size": 13008, "sha256": LEGACY_SERVICE_SHA256}
    actual = (root / old["path"]).read_bytes()
    updated = relocated_service_entry(root, old)
    assert updated["sha256"] == hashlib.sha256(actual).hexdigest()
    altered = tmp_path / old["path"]
    altered.parent.mkdir(parents=True)
    altered.write_bytes(actual + b"\n# unreviewed change\n")
    assert relocated_service_entry(tmp_path, old) == old

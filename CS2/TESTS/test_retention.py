from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.promotion import PromotionValidationError
import cs2model.retention as retention_module
from cs2model.retention import (
    DEFAULT_REGISTRY_KEEP,
    DEFAULT_RUNS_KEEP,
    RetentionApplyError,
    RetentionApprovalError,
    RetentionPlan,
    StaleRetentionPlanError,
    UnsafeRetentionPathError,
    apply_retention_plan,
    plan_registry_retention,
    plan_runs_retention,
    resume_retention_quarantine,
)


def _directories(root: Path, names: list[str]) -> None:
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "sentinel.txt").write_text(name, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _plan_registry(registry: Path, *, keep_last: int) -> RetentionPlan:
    """Build isolated retention plans without production pointer/runtime health."""

    return plan_registry_retention(
        registry,
        keep_last=keep_last,
        enforce_registry_health=False,
    )


def test_defaults_are_documented_in_the_api() -> None:
    assert DEFAULT_REGISTRY_KEEP == 0
    assert DEFAULT_RUNS_KEEP == 2


def test_registry_preview_is_pure_and_token_is_deterministic(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)

    before = sorted(path.name for path in registry.iterdir())
    first = _plan_registry(registry, keep_last=1)
    second = _plan_registry(registry, keep_last=1)

    assert first.token == second.token
    assert first.keep_names == ("20260103_000000Z",)
    assert first.delete_names == (
        "20260101_000000Z",
        "20260102_000000Z",
    )
    assert sorted(path.name for path in registry.iterdir()) == before
    assert all((registry / version / "sentinel.txt").exists() for version in versions)


def test_registry_keeps_latest_last_good_and_unmanaged_names(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
        "20260104_000000Z",
    ]
    _directories(registry, versions + ["manual-import", "20269999_999999Z"])
    _write_json(registry / "latest.json", {"version": versions[0]})
    _write_json(registry / "last_good.json", {"version": versions[1]})

    plan = _plan_registry(registry, keep_last=0)

    assert set(plan.keep_names) == {
        versions[0],
        versions[1],
        "manual-import",
        "20269999_999999Z",
    }
    assert plan.delete_names == (versions[2], versions[3])
    reasons = {entry.name: set(entry.reasons) for entry in plan.keep}
    assert "latest" in reasons[versions[0]]
    assert "last_good" in reasons[versions[1]]
    assert "unmanaged_name" in reasons["manual-import"]


def test_registry_accepts_legacy_latest_pointer(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    _write_json(
        registry / "latest.json",
        {
            "latest": versions[0],
            "artifact": str(tmp_path / "outside" / "model.pkl"),
        },
    )

    plan = _plan_registry(registry, keep_last=1)

    assert set(plan.keep_names) == set(versions)
    assert not plan.warnings


def test_pending_cleanup_does_not_block_other_expired_versions(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)

    plan = plan_registry_retention(
        registry,
        keep_last=0,
        deferred_cleanup_names={versions[0]},
        enforce_registry_health=False,
    )

    assert plan.keep_names == (versions[0],)
    assert plan.delete_names == (versions[1], versions[2])
    assert "cleanup_pending" in set(plan.keep[0].reasons)


def test_runs_keep_recent_master_referenced_and_non_timestamp(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    run_ids = [
        "2026-01-01_000000Z",
        "2026-01-02_000000Z",
        "2026-01-03_000000Z",
        "2026-01-04_000000Z",
        "2026-01-05_000000Z",
    ]
    _directories(runs, run_ids + ["manual_assets_2026", "_tmp_unit"])
    manifest = tmp_path / "master" / "manifest.json"
    _write_json(manifest, {"last_run_id": run_ids[0]})

    plan = plan_runs_retention(
        runs,
        keep_last=1,
        master_manifest=manifest,
        referenced_run_ids={run_ids[1]},
    )

    assert set(plan.keep_names) == {
        run_ids[0],
        run_ids[1],
        run_ids[-1],
        "manual_assets_2026",
        "_tmp_unit",
    }
    assert plan.delete_names == (run_ids[2], run_ids[3])
    reasons = {entry.name: set(entry.reasons) for entry in plan.keep}
    assert "master" in reasons[run_ids[0]]
    assert "referenced" in reasons[run_ids[1]]


def test_stale_token_refuses_every_deletion_and_marker_creation(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _directories(
        registry,
        ["20260101_000000Z", "20260102_000000Z", "20260103_000000Z"],
    )
    marker = tmp_path / "approval.json"
    plan = _plan_registry(registry, keep_last=1)
    (registry / "20260104_000000Z").mkdir()

    with pytest.raises(StaleRetentionPlanError):
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert not marker.exists()
    assert (registry / "20260101_000000Z").exists()
    assert (registry / "20260102_000000Z").exists()


def test_first_confirmation_creates_marker_then_auto_can_apply(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    _directories(
        registry,
        ["20260101_000000Z", "20260102_000000Z", "20260103_000000Z"],
    )

    first = _plan_registry(registry, keep_last=1)
    result = apply_retention_plan(first, first.token, approval_marker=marker)

    assert result.deleted == ("20260101_000000Z", "20260102_000000Z")
    assert marker.exists()
    assert (registry / "20260103_000000Z").exists()

    _directories(registry, ["20260104_000000Z", "20260105_000000Z"])
    automatic = _plan_registry(registry, keep_last=1)
    auto_result = apply_retention_plan(
        automatic,
        automatic.token,
        approval_marker=marker,
        auto=True,
    )

    assert auto_result.automatic
    assert auto_result.deleted == ("20260103_000000Z", "20260104_000000Z")
    assert (registry / "20260105_000000Z").exists()


def test_registry_apply_refuses_active_deployment_before_deletion(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    (registry / ".deployment.lock").write_text("pid=test\n", encoding="utf-8")

    with pytest.raises(PromotionValidationError, match="deployment is active"):
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert all((registry / version).exists() for version in versions)
    assert not marker.exists()


def test_registry_preflights_every_tree_before_first_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)
    denied = registry / versions[1] / "private"
    denied.mkdir()
    plan = _plan_registry(registry, keep_last=1)
    real_scandir = os.scandir

    def denied_scandir(path: str | os.PathLike[str]) -> Any:
        if Path(path) == denied:
            raise PermissionError("simulated denied ACL")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", denied_scandir)

    with pytest.raises(UnsafeRetentionPathError, match="Cannot enumerate deletion tree"):
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert all((registry / version).exists() for version in versions)
    assert not marker.exists()


def test_registry_staging_rename_failure_rolls_back_every_canonical_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_replace = retention_module.os.replace

    def fail_second_rename(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(source) == registry / versions[1] and Path(destination).name == versions[1]:
            raise PermissionError("simulated atomic rename denial")
        real_replace(source, destination)

    monkeypatch.setattr(retention_module.os, "replace", fail_second_rename)

    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert caught.value.phase == "staging"
    assert caught.value.failed == versions[1]
    assert caught.value.rolled_back == (versions[0],)
    assert not caught.value.recovery_required
    assert all((registry / version / "sentinel.txt").exists() for version in versions)
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()
    assert not marker.exists()


def test_registry_rmtree_failure_stays_quarantined_and_can_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_rmtree = retention_module.shutil.rmtree

    def partial_rmtree_failure(path: str | os.PathLike[str]) -> None:
        target = Path(path)
        if target.name == versions[0]:
            (target / "sentinel.txt").unlink()
            raise PermissionError("simulated final rmdir denial")
        real_rmtree(path)

    monkeypatch.setattr(retention_module.shutil, "rmtree", partial_rmtree_failure)

    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    error = caught.value
    assert error.phase == "deletion"
    assert error.failed == versions[0]
    assert error.recovery_required
    assert error.pending == (versions[0], versions[1])
    assert error.quarantine is not None
    assert not (registry / versions[0]).exists()
    assert not (registry / versions[1]).exists()
    assert (registry / versions[2] / "sentinel.txt").exists()
    assert (error.quarantine / versions[0]).is_dir()
    assert (error.quarantine / versions[1] / "sentinel.txt").exists()
    manifest = json.loads((error.quarantine / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "deletion_failed"
    assert not marker.exists()
    assert error.as_dict()["quarantine"] == str(error.quarantine)

    monkeypatch.setattr(retention_module.shutil, "rmtree", real_rmtree)
    resumed = resume_retention_quarantine(
        registry,
        plan.token,
        kind="registry",
        approval_marker=marker,
        enforce_registry_health=False,
    )

    assert resumed.resumed
    assert resumed.deleted == (versions[0], versions[1])
    assert marker.exists()
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


def test_registry_stages_and_removes_an_already_empty_canonical_bundle(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    (registry / versions[0] / "sentinel.txt").unlink()
    plan = _plan_registry(registry, keep_last=1)

    result = apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert result.deleted == (versions[0],)
    assert not (registry / versions[0]).exists()
    assert (registry / versions[1]).exists()
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only directory semantics")
def test_registry_clears_readonly_only_after_windows_staging(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    candidate = registry / versions[0]
    os.chmod(candidate, stat.S_IREAD)
    readonly_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
    if not int(getattr(candidate.lstat(), "st_file_attributes", 0)) & readonly_flag:
        pytest.skip("Filesystem did not preserve the Windows read-only directory attribute")
    plan = _plan_registry(registry, keep_last=1)

    try:
        result = apply_retention_plan(plan, plan.token, approval_marker=marker)
    finally:
        for possible_path in (
            candidate,
            registry / retention_module.QUARANTINE_DIRECTORY / plan.token / versions[0],
        ):
            if possible_path.exists():
                os.chmod(possible_path, stat.S_IWRITE)

    assert result.deleted == (versions[0],)
    assert not candidate.exists()
    assert (registry / versions[1]).exists()
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only directory semantics")
def test_windows_readonly_quarantine_can_resume_after_one_rmtree_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    candidate = registry / versions[0]
    os.chmod(candidate, stat.S_IREAD)
    readonly_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
    if not int(getattr(candidate.lstat(), "st_file_attributes", 0)) & readonly_flag:
        pytest.skip("Filesystem did not preserve the Windows read-only directory attribute")
    plan = _plan_registry(registry, keep_last=1)
    real_rmtree = retention_module.shutil.rmtree
    attempts = 0

    def fail_once_after_attribute_clear(path: str | os.PathLike[str]) -> None:
        nonlocal attempts
        target = Path(path)
        attributes = int(getattr(target.lstat(), "st_file_attributes", 0))
        assert not attributes & readonly_flag
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient Windows rmdir denial")
        real_rmtree(path)

    monkeypatch.setattr(retention_module.shutil, "rmtree", fail_once_after_attribute_clear)
    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert caught.value.phase == "deletion"
    assert not candidate.exists()
    resumed = resume_retention_quarantine(
        registry,
        plan.token,
        kind="registry",
        approval_marker=marker,
        enforce_registry_health=False,
    )

    assert resumed.deleted == (versions[0],)
    assert attempts == 2
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


def test_resume_rolls_back_a_kill_during_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_replace = retention_module.os.replace

    def kill_on_second_rename(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(source) == registry / versions[1] and Path(destination).name == versions[1]:
            raise KeyboardInterrupt("simulated process kill")
        real_replace(source, destination)

    monkeypatch.setattr(retention_module.os, "replace", kill_on_second_rename)
    with pytest.raises(KeyboardInterrupt):
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    operation = registry / retention_module.QUARANTINE_DIRECTORY / plan.token
    assert not (registry / versions[0]).exists()
    assert (operation / versions[0] / "sentinel.txt").exists()
    assert (registry / versions[1] / "sentinel.txt").exists()
    assert json.loads((operation / "manifest.json").read_text(encoding="utf-8"))["state"] == "staging"

    monkeypatch.setattr(retention_module.os, "replace", real_replace)
    recovered = resume_retention_quarantine(
        registry,
        plan.token,
        kind="registry",
        approval_marker=marker,
        enforce_registry_health=False,
    )

    assert recovered.deleted == ()
    assert recovered.rolled_back == (versions[0], versions[1])
    assert all((registry / version / "sentinel.txt").exists() for version in versions)
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()
    assert not marker.exists()


def test_resume_completes_a_previous_partial_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = [
        "20260101_000000Z",
        "20260102_000000Z",
        "20260103_000000Z",
    ]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_replace = retention_module.os.replace

    def fail_stage_and_rollback(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == registry / versions[1] and destination_path.name == versions[1]:
            raise PermissionError("simulated staging denial")
        if source_path.name == versions[0] and source_path.parent.name == plan.token:
            raise PermissionError("simulated rollback denial")
        real_replace(source, destination)

    monkeypatch.setattr(retention_module.os, "replace", fail_stage_and_rollback)
    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    operation = registry / retention_module.QUARANTINE_DIRECTORY / plan.token
    assert caught.value.recovery_required
    assert json.loads((operation / "manifest.json").read_text(encoding="utf-8"))["state"] == "rollback_failed"
    assert (operation / versions[0] / "sentinel.txt").exists()
    assert (registry / versions[1] / "sentinel.txt").exists()

    monkeypatch.setattr(retention_module.os, "replace", real_replace)
    recovered = resume_retention_quarantine(
        registry,
        plan.token,
        kind="registry",
        approval_marker=marker,
        enforce_registry_health=False,
    )

    assert recovered.rolled_back == (versions[0], versions[1])
    assert all((registry / version / "sentinel.txt").exists() for version in versions)
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


def test_staging_resume_rejects_bundle_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_replace = retention_module.os.replace
    renamed = False

    def kill_after_first_manifest(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal renamed
        if Path(source) == registry / versions[0]:
            real_replace(source, destination)
            renamed = True
            return
        if renamed and Path(source).name.startswith(".manifest.json."):
            raise KeyboardInterrupt("simulated manifest publication kill")
        real_replace(source, destination)

    monkeypatch.setattr(retention_module.os, "replace", kill_after_first_manifest)
    with pytest.raises(KeyboardInterrupt):
        apply_retention_plan(plan, plan.token, approval_marker=marker)
    monkeypatch.setattr(retention_module.os, "replace", real_replace)

    operation = registry / retention_module.QUARANTINE_DIRECTORY / plan.token
    staged = operation / versions[0]
    (staged / "sentinel.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RetentionApplyError, match="identity") as caught:
        resume_retention_quarantine(
            registry,
            plan.token,
            kind="registry",
            approval_marker=marker,
            enforce_registry_health=False,
        )

    assert caught.value.phase == "rollback_validation"
    assert not (registry / versions[0]).exists()
    assert staged.exists()


def test_initial_manifest_failure_cleans_empty_quarantine_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)

    def fail_manifest(_path: Path, _payload: dict[str, Any]) -> None:
        raise PermissionError("simulated initial manifest failure")

    monkeypatch.setattr(retention_module, "_write_quarantine_manifest", fail_manifest)
    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)

    assert caught.value.phase == "staging_setup"
    assert not caught.value.recovery_required
    assert all((registry / version).exists() for version in versions)
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


def test_cleanup_keeps_manifest_when_quarantine_has_unexpected_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    versions = ["20260101_000000Z", "20260102_000000Z"]
    _directories(registry, versions)
    plan = _plan_registry(registry, keep_last=1)
    real_rmtree = retention_module.shutil.rmtree

    def fail_rmtree(path: str | os.PathLike[str]) -> None:
        raise PermissionError(f"simulated failure for {path}")

    monkeypatch.setattr(retention_module.shutil, "rmtree", fail_rmtree)
    with pytest.raises(RetentionApplyError) as caught:
        apply_retention_plan(plan, plan.token, approval_marker=marker)
    operation = caught.value.quarantine
    assert operation is not None
    (operation / "operator-note.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(retention_module.shutil, "rmtree", real_rmtree)
    result = resume_retention_quarantine(
        registry,
        plan.token,
        kind="registry",
        approval_marker=marker,
        enforce_registry_health=False,
    )

    assert result.quarantine == operation
    assert (operation / "manifest.json").exists()
    assert (operation / "operator-note.txt").exists()


def test_auto_refuses_marker_when_retention_count_changes(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    marker = tmp_path / "approval.json"
    _directories(
        registry,
        ["20260101_000000Z", "20260102_000000Z", "20260103_000000Z"],
    )
    approved = _plan_registry(registry, keep_last=2)
    apply_retention_plan(approved, approved.token, approval_marker=marker)
    _directories(registry, ["20260104_000000Z", "20260105_000000Z"])

    changed = _plan_registry(registry, keep_last=1)
    with pytest.raises(RetentionApprovalError):
        apply_retention_plan(
            changed,
            changed.token,
            approval_marker=marker,
            auto=True,
        )

    assert (registry / "20260102_000000Z").exists()
    assert (registry / "20260103_000000Z").exists()


def test_directory_symlink_is_always_kept_when_supported(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    outside = tmp_path / "outside"
    registry.mkdir()
    outside.mkdir()
    (outside / "sentinel.txt").write_text("safe", encoding="utf-8")
    link = registry / "20260101_000000Z"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are not available in this environment")

    plan = _plan_registry(registry, keep_last=0)

    assert plan.delete_names == ()
    entry = next(item for item in plan.keep if item.name == link.name)
    assert "symlink_or_reparse" in entry.reasons
    assert "outside_root" in entry.reasons
    assert (outside / "sentinel.txt").exists()


def test_resume_rejects_a_symlinked_quarantine_root_when_supported(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    outside = tmp_path / "outside"
    registry.mkdir()
    outside.mkdir()
    (outside / "sentinel.txt").write_text("safe", encoding="utf-8")
    quarantine = registry / retention_module.QUARANTINE_DIRECTORY
    try:
        os.symlink(outside, quarantine, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are not available in this environment")
    token = "a" * 64

    plan = _plan_registry(registry, keep_last=0)
    assert "unsafe_pending_quarantine" in plan.warnings
    with pytest.raises(UnsafeRetentionPathError, match="Quarantine root"):
        resume_retention_quarantine(
            registry,
            token,
            kind="registry",
            approval_marker=registry / ".retention-approved.json",
            enforce_registry_health=False,
        )

    assert (outside / "sentinel.txt").read_text(encoding="utf-8") == "safe"


def test_start_rotates_logs_to_current_and_previous_run() -> None:
    launcher = (ROOT / "start.ps1").read_text(encoding="utf-8")

    assert "$LogRetentionKeep = 2" in launcher
    assert "function Invoke-PipelineLogRetention" in launcher
    assert launcher.count("Invoke-PipelineLogRetention") == 4
    assert "start|retention|retrain_decision" in launcher

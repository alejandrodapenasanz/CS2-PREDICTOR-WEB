from __future__ import annotations

import io
import json
import os
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from MODEL import manage_models
from MODEL.cs2model import retention as retention_module
from MODEL.cs2model.artifacts import Component, ModelArtifact, ProbabilityColumnEstimator
from MODEL.cs2model.promotion import (
    PromotionValidationError,
    read_model_pointer,
    reference_for_version,
    set_last_good_version,
    sha256_file,
    write_model_pointer,
)

VERSION_1 = "20260101_000000Z"
VERSION_2 = "20260201_000000Z"
VERSION_3 = "20260301_000000Z"
VERSION_4 = "20260401_000000Z"
VERSION_5 = "20260501_000000Z"


def _invoke(arguments: list[str]) -> tuple[int, dict[str, object], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = manage_models.main(arguments)
    stream = stdout.getvalue() if stdout.getvalue() else stderr.getvalue()
    return exit_code, json.loads(stream), stderr.getvalue()


def _registry(tmp_path: Path, versions: tuple[str, ...] = (VERSION_1, VERSION_2)) -> tuple[Path, Path]:
    artifacts = tmp_path / "MODEL" / "artifacts"
    registry = artifacts / "registry"
    for version in versions:
        version_dir = registry / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "model.pkl").write_bytes(f"model-{version}".encode())
    latest = reference_for_version(registry, versions[0])
    write_model_pointer(registry, "latest", latest)
    production = artifacts / "model.pkl"
    production.write_bytes(latest.artifact.read_bytes())
    return registry, production


def _model_arguments(command: str, registry: Path, production: Path) -> list[str]:
    return [
        command,
        "--registry",
        str(registry),
        "--production",
        str(production),
    ]


def _write_loadable_model(path: Path) -> None:
    ModelArtifact(
        feature_columns=["rating_probability"],
        components=[Component("rating", ProbabilityColumnEstimator(column_index=0), None)],
        metadata={"random_seed": 42},
    ).save(path)


def _make_runs(tmp_path: Path, run_ids: list[str]) -> tuple[Path, Path, Path]:
    runs = tmp_path / "PIPELINE" / "runs"
    for run_id in run_ids:
        directory = runs / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sentinel.txt").write_text(run_id, encoding="utf-8")
    master = tmp_path / "PIPELINE" / "master" / "manifest.json"
    master.parent.mkdir(parents=True)
    master.write_text(json.dumps({"last_run_id": run_ids[0]}), encoding="utf-8")
    db = tmp_path / "BBDD" / "cs2.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE run_references(run_id TEXT, source_run_id TEXT, source_file TEXT)")
    return runs, master, db


def test_status_and_initialize_last_good_are_validated_and_idempotent(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path)

    status_code, status, _ = _invoke(_model_arguments("status", registry, production))
    assert status_code == 0
    assert status["latest"]["version"] == VERSION_1  # type: ignore[index]
    assert status["last_good"] is None
    assert status["runtime"]["matches_latest"] is True  # type: ignore[index]

    init_code, initialized, _ = _invoke(_model_arguments("initialize-last-good", registry, production))
    assert init_code == 0
    assert initialized["changed"] is True
    first_bytes = (registry / "last_good.json").read_bytes()

    again_code, again, _ = _invoke(_model_arguments("initialize-last-good", registry, production))
    assert again_code == 0
    assert again["changed"] is False
    assert (registry / "last_good.json").read_bytes() == first_bytes


def test_status_rejects_runtime_hash_mismatch(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path)
    production.write_bytes(b"tampered-runtime")

    exit_code, payload, stderr = _invoke(_model_arguments("status", registry, production))

    assert exit_code == 2
    assert payload["ok"] is False
    assert "does not match latest" in str(payload["error"])
    assert stderr


def test_registry_prune_is_safety_blocked_when_runtime_health_is_invalid(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path, (VERSION_1, VERSION_2, VERSION_3))
    production.write_bytes(b"tampered-runtime")
    arguments = [
        "prune",
        "--scope",
        "registry",
        "--registry",
        str(registry),
        "--registry-keep",
        "1",
    ]

    preview_code, preview, _ = _invoke(arguments)

    assert preview_code == 0
    assert preview["confirmation"] is None
    warnings = preview["safety_blocked"]["registry"]  # type: ignore[index]
    assert any(str(warning).startswith("invalid_registry_health") for warning in warnings)
    auto_code, auto_error, auto_stderr = _invoke([*arguments, "--auto"])
    assert auto_code == 2
    assert auto_stderr
    assert auto_error["error_type"] == "ModelManagementError"
    assert (registry / VERSION_2).exists()


def test_rollback_command_restores_last_good_and_rotates_latest(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path)
    write_model_pointer(registry, "last_good", reference_for_version(registry, VERSION_2))

    exit_code, payload, _ = _invoke(_model_arguments("rollback", registry, production))

    assert exit_code == 0
    assert payload["action"] == "rolled_back"
    assert read_model_pointer(registry, "latest").version == VERSION_2
    assert read_model_pointer(registry, "last_good").version == VERSION_1
    assert sha256_file(production) == reference_for_version(registry, VERSION_2).sha256


def test_set_last_good_validates_target_and_keeps_latest_runtime_unchanged(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path, (VERSION_1, VERSION_2, VERSION_3))
    _write_loadable_model(registry / VERSION_2 / "model.pkl")
    write_model_pointer(registry, "last_good", reference_for_version(registry, VERSION_3))
    target = reference_for_version(registry, VERSION_2)
    latest_before = (registry / "latest.json").read_bytes()
    runtime_before = production.read_bytes()

    arguments = [
        *_model_arguments("set-last-good", registry, production),
        "--version",
        VERSION_2,
        "--expected-sha256",
        target.sha256,
    ]
    exit_code, payload, stderr = _invoke(arguments)

    assert exit_code == 0
    assert not stderr
    assert payload["changed"] is True
    assert payload["previous_last_good"]["version"] == VERSION_3  # type: ignore[index]
    assert payload["last_good"]["version"] == VERSION_2  # type: ignore[index]
    assert (registry / "latest.json").read_bytes() == latest_before
    assert production.read_bytes() == runtime_before
    assert read_model_pointer(registry, "latest").version == VERSION_1
    assert read_model_pointer(registry, "last_good").version == VERSION_2

    again_code, again, _ = _invoke(arguments)
    assert again_code == 0
    assert again["changed"] is False
    assert (registry / "latest.json").read_bytes() == latest_before
    assert production.read_bytes() == runtime_before


def test_set_last_good_rejects_wrong_target_hash_and_existing_lock(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path, (VERSION_1, VERSION_2))
    _write_loadable_model(registry / VERSION_2 / "model.pkl")
    last_good_path = registry / "last_good.json"
    arguments = [
        *_model_arguments("set-last-good", registry, production),
        "--version",
        VERSION_2,
        "--expected-sha256",
        "0" * 64,
    ]

    wrong_code, wrong, _ = _invoke(arguments)

    assert wrong_code == 2
    assert "does not match" in str(wrong["error"])
    assert not last_good_path.exists()

    target = reference_for_version(registry, VERSION_2)
    lock = registry / ".deployment.lock"
    lock.write_text("busy", encoding="utf-8")
    locked_code, locked, _ = _invoke([*arguments[:-1], target.sha256])

    assert locked_code == 2
    assert "deployment is active" in str(locked["error"])
    assert not last_good_path.exists()


def test_set_last_good_compare_and_swap_rejects_stale_latest(tmp_path: Path) -> None:
    registry, production = _registry(tmp_path, (VERSION_1, VERSION_2, VERSION_3))
    expected_latest = read_model_pointer(registry, "latest")
    target = reference_for_version(registry, VERSION_2)
    write_model_pointer(registry, "latest", reference_for_version(registry, VERSION_3))

    with pytest.raises(PromotionValidationError, match="compare-and-swap failed"):
        set_last_good_version(
            registry,
            VERSION_2,
            production_path=production,
            expected_latest=expected_latest,
            expected_last_good=None,
            expected_target_sha256=target.sha256,
        )

    assert not (registry / "last_good.json").exists()


def test_registry_zero_keep_retains_only_latest_and_last_good(tmp_path: Path) -> None:
    registry, _production = _registry(tmp_path, (VERSION_1, VERSION_2, VERSION_3))
    write_model_pointer(registry, "last_good", reference_for_version(registry, VERSION_2))

    exit_code, payload, _ = _invoke(
        [
            "prune",
            "--scope",
            "registry",
            "--registry",
            str(registry),
            "--registry-keep",
            "0",
        ]
    )

    assert exit_code == 0
    plan = payload["plans"]["registry"]  # type: ignore[index]
    assert {entry["name"] for entry in plan["keep"]} == {VERSION_1, VERSION_2}
    assert [entry["name"] for entry in plan["delete"]] == [VERSION_3]
    assert plan["keep_last"] == 0
    assert (registry / VERSION_3).exists()


def test_runs_preview_protects_master_manifest_database_and_unknown_ids(
    tmp_path: Path,
) -> None:
    run_ids = [
        "2026-01-01_000000Z",
        "2026-01-02_000000Z",
        "2026-01-03_000000Z",
        "2026-01-04_000000Z",
        "2026-01-05_000000Z",
    ]
    runs, master, db = _make_runs(tmp_path, run_ids)
    unknown = runs / "legacy-run-name"
    unknown.mkdir()
    manifest = runs / run_ids[-1] / "nested_manifest.json"
    manifest.write_text(json.dumps({"previous": run_ids[1]}), encoding="utf-8")
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO run_references(run_id,source_run_id,source_file) VALUES (?,?,?)",
            (run_ids[2], "legacy-run-name", f"PIPELINE/runs/{run_ids[2]}/manifest.json"),
        )

    output = tmp_path / "preview.json"
    before = sorted(path.name for path in runs.iterdir())
    exit_code, payload, _ = _invoke(
        [
            "prune",
            "--scope",
            "runs",
            "--runs",
            str(runs),
            "--master-manifest",
            str(master),
            "--db",
            str(db),
            "--runs-keep",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert payload["mode"] == "preview"
    plan = payload["plans"]["runs"]  # type: ignore[index]
    keep = {entry["name"] for entry in plan["keep"]}
    assert {run_ids[0], run_ids[1], run_ids[2], run_ids[-1], "legacy-run-name"} <= keep
    assert [entry["name"] for entry in plan["delete"]] == [run_ids[3]]
    assert sorted(path.name for path in runs.iterdir()) == before
    assert not (runs / ".retention-approved.json").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_registry_confirm_then_compatible_auto_apply_only_tmp_plans(tmp_path: Path) -> None:
    registry, _production = _registry(
        tmp_path,
        (VERSION_1, VERSION_2, VERSION_3),
    )
    preview_args = [
        "prune",
        "--scope",
        "registry",
        "--registry",
        str(registry),
        "--registry-keep",
        "1",
    ]
    preview_code, preview, _ = _invoke(preview_args)
    assert preview_code == 0
    assert (registry / VERSION_2).exists()
    token = preview["plans"]["registry"]["token"]  # type: ignore[index]

    confirm_code, confirmed, _ = _invoke([*preview_args, "--confirm", str(token)])
    assert confirm_code == 0
    assert confirmed["mode"] == "confirmed"
    assert confirmed["applied"]["registry"]["deleted"] == [VERSION_2]  # type: ignore[index]
    marker = registry / ".retention-approved.json"
    assert marker.exists()

    for version in (VERSION_4, VERSION_5):
        version_dir = registry / version
        version_dir.mkdir()
        (version_dir / "model.pkl").write_bytes(version.encode())
    auto_code, automatic, _ = _invoke([*preview_args, "--auto"])

    assert auto_code == 0
    assert automatic["mode"] == "auto"
    assert automatic["applied"]["registry"]["deleted"] == [VERSION_3, VERSION_4]  # type: ignore[index]
    assert (registry / VERSION_1).exists()
    assert (registry / VERSION_5).exists()


def test_registry_deletion_failure_reports_quarantine_and_cli_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, production = _registry(
        tmp_path,
        (VERSION_1, VERSION_2, VERSION_3),
    )
    arguments = [
        "prune",
        "--scope",
        "registry",
        "--registry",
        str(registry),
        "--registry-keep",
        "1",
    ]
    preview_code, preview, _ = _invoke(arguments)
    assert preview_code == 0
    token = str(preview["plans"]["registry"]["token"])  # type: ignore[index]
    real_rmtree = retention_module.shutil.rmtree

    def partial_rmtree_failure(path: str | os.PathLike[str]) -> None:
        target = Path(path)
        if target.name == VERSION_2:
            (target / "model.pkl").unlink()
            raise PermissionError("simulated Windows rmdir denial")
        real_rmtree(path)

    monkeypatch.setattr(retention_module.shutil, "rmtree", partial_rmtree_failure)
    failed_code, failed, failed_stderr = _invoke([*arguments, "--confirm", token])

    assert failed_code == 0
    assert not failed_stderr
    assert failed["ok"] is True
    assert failed["cleanup_deferred"] is True
    details = failed["applied"]["registry"]  # type: ignore[index]
    assert details["phase"] == "deletion"  # type: ignore[index]
    assert details["pending"] == [VERSION_2]  # type: ignore[index]
    assert details["recovery_required"] is True  # type: ignore[index]
    pending = failed["pending_cleanup"]  # type: ignore[assignment]
    assert pending["count"] == 1  # type: ignore[index]
    assert pending["items"][0]["name"] == VERSION_2  # type: ignore[index]
    assert not (registry / VERSION_2).exists()

    blocked_preview_code, blocked_preview, _ = _invoke(arguments)
    assert blocked_preview_code == 0
    assert blocked_preview["confirmation"] is None
    assert "registry" in blocked_preview["recovery_required"]  # type: ignore[operator]
    blocked_auto_code, blocked_auto, blocked_auto_stderr = _invoke([*arguments, "--auto"])
    assert blocked_auto_code == 0
    assert not blocked_auto_stderr
    assert blocked_auto["cleanup_deferred"] is True
    assert blocked_auto["reason"] == "cleanup_pending"
    assert blocked_auto["pending_cleanup"]["count"] == 1  # type: ignore[index]

    production.write_bytes(b"tampered-while-quarantined")
    unhealthy_resume_code, unhealthy_resume, unhealthy_stderr = _invoke(
        [
            "prune",
            "--scope",
            "registry",
            "--registry",
            str(registry),
            "--resume",
            token,
        ]
    )
    assert unhealthy_resume_code == 2
    assert unhealthy_stderr
    assert unhealthy_resume["error_type"] == "PromotionValidationError"
    assert (registry / retention_module.QUARANTINE_DIRECTORY / token).exists()
    production.write_bytes((registry / VERSION_1 / "model.pkl").read_bytes())

    monkeypatch.setattr(retention_module.shutil, "rmtree", real_rmtree)
    resume_code, resumed, resume_stderr = _invoke(
        [
            "prune",
            "--scope",
            "registry",
            "--registry",
            str(registry),
            "--config",
            str(tmp_path / "missing-config.yaml"),
            "--resume",
            token,
        ]
    )

    assert resume_code == 0
    assert not resume_stderr
    assert resumed["mode"] == "resume"
    assert resumed["applied"]["registry"]["deleted"] == [VERSION_2]  # type: ignore[index]
    assert resumed["applied"]["registry"]["resumed"] is True  # type: ignore[index]
    assert resumed["pending_cleanup"]["count"] == 0  # type: ignore[index]
    assert not (registry / retention_module.QUARANTINE_DIRECTORY).exists()


def test_acl_preflight_failure_is_nonfatal_and_never_targets_live_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _production = _registry(
        tmp_path,
        (VERSION_1, VERSION_2, VERSION_3),
    )
    write_model_pointer(registry, "last_good", reference_for_version(registry, VERSION_2))
    arguments = [
        "prune",
        "--scope",
        "registry",
        "--registry",
        str(registry),
        "--registry-keep",
        "0",
    ]
    preview_code, preview, _ = _invoke(arguments)
    assert preview_code == 0
    token = str(preview["plans"]["registry"]["token"])  # type: ignore[index]
    blocked = registry / VERSION_3
    real_scandir = retention_module.os.scandir

    def deny_challenger(path: str | os.PathLike[str]):  # type: ignore[no-untyped-def]
        if Path(path) == blocked:
            raise PermissionError("simulated protected ACL")
        return real_scandir(path)

    monkeypatch.setattr(retention_module.os, "scandir", deny_challenger)
    exit_code, result, stderr = _invoke([*arguments, "--confirm", token])

    assert exit_code == 0
    assert not stderr
    assert result["cleanup_deferred"] is True
    pending = result["pending_cleanup"]  # type: ignore[assignment]
    assert pending["count"] == 1  # type: ignore[index]
    assert pending["items"][0]["name"] == VERSION_3  # type: ignore[index]
    assert pending["items"][0]["phase"] == "preflight"  # type: ignore[index]
    assert (registry / VERSION_1).exists()
    assert (registry / VERSION_2).exists()
    assert (registry / VERSION_3).exists()
    assert read_model_pointer(registry, "latest").version == VERSION_1
    assert read_model_pointer(registry, "last_good").version == VERSION_2


def test_auto_without_marker_is_successful_preview_noop_and_writes_output(
    tmp_path: Path,
) -> None:
    registry, _production = _registry(
        tmp_path,
        (VERSION_1, VERSION_2, VERSION_3),
    )
    output = tmp_path / "auto-preview.json"

    exit_code, payload, stderr = _invoke(
        [
            "prune",
            "--scope",
            "registry",
            "--registry",
            str(registry),
            "--registry-keep",
            "1",
            "--auto",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert not stderr
    assert payload["mode"] == "preview"
    assert payload["applied"] is False
    assert payload["reason"] == "approval_required"
    assert payload["approval_required"] == ["registry"]
    assert (registry / VERSION_2).exists()
    assert not (registry / ".retention-approved.json").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_auto_with_corrupt_marker_fails_without_deleting(tmp_path: Path) -> None:
    registry, _production = _registry(
        tmp_path,
        (VERSION_1, VERSION_2, VERSION_3),
    )
    (registry / ".retention-approved.json").write_text("not-json", encoding="utf-8")

    exit_code, payload, stderr = _invoke(
        [
            "prune",
            "--scope",
            "registry",
            "--registry",
            str(registry),
            "--registry-keep",
            "1",
            "--auto",
        ]
    )

    assert exit_code == 2
    assert stderr
    assert payload["error_type"] == "RetentionApprovalError"
    assert (registry / VERSION_2).exists()


def test_scope_all_requires_explicit_tokens_for_both_plans(tmp_path: Path) -> None:
    registry, _production = _registry(tmp_path, (VERSION_1, VERSION_2, VERSION_3))
    runs, master, db = _make_runs(
        tmp_path,
        ["2026-01-01_000000Z", "2026-01-02_000000Z", "2026-01-03_000000Z"],
    )
    arguments = [
        "prune",
        "--scope",
        "all",
        "--registry",
        str(registry),
        "--runs",
        str(runs),
        "--master-manifest",
        str(master),
        "--db",
        str(db),
        "--registry-keep",
        "1",
        "--runs-keep",
        "1",
    ]
    preview_code, preview, _ = _invoke(arguments)
    assert preview_code == 0
    registry_token = preview["plans"]["registry"]["token"]  # type: ignore[index]

    exit_code, error, _ = _invoke([*arguments, "--confirm", f"registry={registry_token}"])

    assert exit_code == 2
    assert error["ok"] is False
    assert (registry / VERSION_2).exists()
    assert (runs / "2026-01-02_000000Z").exists()

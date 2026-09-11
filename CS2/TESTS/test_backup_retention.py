from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from BBDD import backup_retention, build_db, manage_backups
from BBDD.backup_retention import (
    APPROVAL_MARKER_NAME,
    DEFAULT_CONFIG,
    BackupDeletionError,
    BackupRetentionError,
    BackupRetentionConfigError,
    BackupValidation,
    StaleBackupPlanError,
    UnsafeBackupPathError,
    apply_backup_retention,
    load_retention_config,
    plan_backup_retention,
    plan_extra_cleanup,
)

CS2_ROOT = Path(__file__).resolve().parents[1]


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    bbdd = tmp_path / "BBDD"
    backups = bbdd / "backups"
    backups.mkdir(parents=True)
    # Deliberately not SQLite: planning/apply must never open the live DB.
    (bbdd / "cs2.db").write_text("live-sentinel", encoding="utf-8")
    (bbdd / "cs2.db-wal").write_bytes(b"")
    (bbdd / "cs2.db-shm").write_text("sidecar-sentinel", encoding="utf-8")
    (bbdd / "BLACKBOX").mkdir()
    return bbdd, backups


def _write_backup(
    backups: Path,
    name: str,
    content: str | None = None,
    *,
    ledger_rows: int = 1,
) -> Path:
    target = backups / name
    connection = sqlite3.connect(target)
    try:
        connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        connection.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE prediction_ledger(ledger_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO sentinel(value) VALUES (?)", (content if content is not None else name,))
        connection.execute("INSERT INTO matches(match_id) VALUES (1)")
        connection.executemany(
            "INSERT INTO prediction_ledger(ledger_id) VALUES (?)",
            [(row_id,) for row_id in range(1, ledger_rows + 1)],
        )
        connection.commit()
    finally:
        connection.close()
    return target


def _write_config(bbdd: Path, *, indent: int | None = 2) -> Path:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    target = bbdd / "backup_retention.json"
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return target


def _managed_sqlite_layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    bbdd = tmp_path / "BBDD"
    backups = bbdd / "backups"
    backups.mkdir(parents=True)
    live = bbdd / "cs2.db"
    connection = sqlite3.connect(live)
    try:
        connection.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE prediction_ledger(ledger_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO matches(match_id) VALUES (1)")
        connection.execute("INSERT INTO prediction_ledger(ledger_id) VALUES (1)")
        connection.commit()
    finally:
        connection.close()
    return bbdd, backups, live, _write_config(bbdd)


def test_config_defaults_to_one_and_only_two_explicit_extras() -> None:
    config = load_retention_config(DEFAULT_CONFIG)

    assert config.default_keep == 1
    assert config.cleanup_targets == ("cs2_dump.sql", "cs2_integrity_test.db")


def test_preview_is_pure_deterministic_and_orders_by_filename_timestamp(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    oldest = _write_backup(backups, "cs2_20260101_000000Z.db", "old")
    middle = _write_backup(backups, "cs2_pre_schema_20260102_000000Z.db", "middle")
    newest = _write_backup(backups, "cs2_20260103_000000Z.db", "new")
    # copy2 preserves source mtimes in production.  Ensure selection does not
    # accidentally use mtime instead of the timestamp embedded in the name.
    os.utime(oldest, ns=(3_000_000_000, 3_000_000_000))
    os.utime(middle, ns=(2_000_000_000, 2_000_000_000))
    os.utime(newest, ns=(1_000_000_000, 1_000_000_000))
    _write_backup(backups, "manual.db")
    (backups / "notes.txt").write_text("keep", encoding="utf-8")
    (backups / "nested").mkdir()

    before = sorted(path.name for path in backups.iterdir())
    expected_delete_bytes = oldest.stat().st_size + middle.stat().st_size
    first = plan_backup_retention(bbdd, keep_latest=1)
    second = plan_backup_retention(bbdd, keep_latest=1)

    assert first.token == second.token
    assert first.delete_names == (
        "cs2_20260101_000000Z.db",
        "cs2_pre_schema_20260102_000000Z.db",
    )
    assert "cs2_20260103_000000Z.db" in first.keep_names
    assert "manual.db" in first.keep_names
    assert "notes.txt" in first.keep_names
    assert "nested" in first.keep_names
    assert first.delete_bytes == expected_delete_bytes
    assert first.newest_backup_validation is not None
    assert first.newest_backup_validation.ok
    assert first.newest_backup_validation.matches == 1
    assert first.newest_backup_validation.prediction_ledger_rows == 1
    assert sorted(path.name for path in backups.iterdir()) == before
    protected = {entry.name for entry in first.protected}
    assert {"cs2.db", "cs2.db-wal", "cs2.db-shm", "BLACKBOX"} <= protected


def test_confirm_deletes_only_expired_backups_and_keeps_live_state(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    names = [
        "cs2_20260101_000000Z.db",
        "cs2_20260102_000000Z.db",
        "cs2_20260103_000000Z.db",
        "pre_restore_corrupt_20260104_000000Z.db",
    ]
    for name in names:
        _write_backup(backups, name)
    _write_backup(backups, "manual.db")
    live_before = (bbdd / "cs2.db").read_bytes()
    sidecar_before = (bbdd / "cs2.db-shm").read_bytes()

    plan = plan_backup_retention(bbdd, keep_latest=2)
    result = apply_backup_retention(plan, plan.token)

    assert result.deleted == tuple(names[:2])
    assert sorted(path.name for path in backups.iterdir()) == sorted(["manual.db", *names[2:]])
    assert (bbdd / "cs2.db").read_bytes() == live_before
    assert (bbdd / "cs2.db-shm").read_bytes() == sidecar_before
    assert (bbdd / "BLACKBOX").is_dir()


def test_keep_override_is_part_of_token_and_supports_batched_policy(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    for day in range(1, 7):
        _write_backup(backups, f"cs2_202601{day:02d}_000000Z.db")

    keep_five = plan_backup_retention(bbdd, keep_latest=5)
    keep_one = plan_backup_retention(bbdd, keep_latest=1)

    assert keep_five.token != keep_one.token
    assert len(keep_five.delete) == 1
    assert len(keep_one.delete) == 5
    with pytest.raises(StaleBackupPlanError):
        apply_backup_retention(keep_one, keep_five.token)
    assert len(list(backups.glob("*.db"))) == 6


@pytest.mark.parametrize("change", ["candidate", "new_entry", "live"])
def test_stale_filesystem_state_refuses_every_deletion(tmp_path: Path, change: str) -> None:
    bbdd, backups = _layout(tmp_path)
    old = _write_backup(backups, "cs2_20260101_000000Z.db", "old")
    _write_backup(backups, "cs2_20260102_000000Z.db", "new")
    plan = plan_backup_retention(bbdd, keep_latest=1)
    if change == "candidate":
        old.write_text("old-but-changed", encoding="utf-8")
    elif change == "new_entry":
        (backups / "notes-created-after-preview.txt").write_text("new", encoding="utf-8")
    else:
        (bbdd / "cs2.db").write_text("live-changed", encoding="utf-8")

    with pytest.raises(StaleBackupPlanError):
        apply_backup_retention(plan, plan.token)

    assert old.exists()
    assert (backups / "cs2_20260102_000000Z.db").exists()


def test_required_current_snapshot_identity_is_revalidated_before_deletion(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    required = _write_backup(backups, "cs2_20260101_000000Z.db")
    future_named = _write_backup(backups, "cs2_20990101_000000Z.db")
    plan = plan_backup_retention(bbdd, keep_latest=1, required_keep=required)

    assert plan.required_keep is not None
    assert plan.required_keep.name == required.name
    assert plan.keep_names.count(required.name) == 1
    assert future_named.name in plan.delete_names

    required.unlink()
    replacement = _write_backup(backups, required.name)
    with pytest.raises(StaleBackupPlanError):
        apply_backup_retention(plan, plan.token)

    assert replacement.exists()
    assert future_named.exists()


def test_hardlink_to_live_database_safety_blocks_whole_plan(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    linked = backups / "cs2_20260101_000000Z.db"
    try:
        os.link(bbdd / "cs2.db", linked)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are unavailable")
    _write_backup(backups, "cs2_20260102_000000Z.db")
    _write_backup(backups, "cs2_20260103_000000Z.db")

    plan = plan_backup_retention(bbdd, keep_latest=1)

    assert f"hardlink_to_live_database:{linked.name}" in plan.blocked
    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)
    assert linked.exists()
    assert (bbdd / "cs2.db").read_text(encoding="utf-8") == "live-sentinel"


def test_corrupt_newest_backup_blocks_deletion_of_older_valid_copy(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    old = _write_backup(backups, "cs2_20260101_000000Z.db")
    corrupt = backups / "cs2_20260102_000000Z.db"
    corrupt.write_text("not sqlite", encoding="utf-8")

    plan = plan_backup_retention(bbdd, keep_latest=1)

    assert plan.newest_backup_validation is not None
    assert not plan.newest_backup_validation.ok
    assert f"retained_backup_integrity_failed:{corrupt.name}" in plan.blocked
    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)
    assert old.exists()
    assert corrupt.exists()


def test_retained_backup_cannot_regress_sacred_prediction_ledger(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    evidence_richer = _write_backup(
        backups,
        "cs2_20260101_000000Z.db",
        ledger_rows=3,
    )
    retained = _write_backup(
        backups,
        "cs2_20260102_000000Z.db",
        ledger_rows=2,
    )

    plan = plan_backup_retention(bbdd, keep_latest=1)

    assert plan.retained_backup_validation is not None
    assert plan.retained_backup_validation.name == retained.name
    assert plan.retained_backup_validation.prediction_ledger_rows == 2
    counts = {item.name: item.prediction_ledger_rows for item in plan.ledger_counts}
    assert counts == {evidence_richer.name: 3, retained.name: 2}
    assert any(reason.startswith("prediction_ledger_regression:") for reason in plan.blocked)
    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)
    assert evidence_richer.exists()
    assert retained.exists()


@pytest.mark.parametrize(
    "failure",
    ["missing_matches", "missing_ledger", "nonempty_wal", "nonempty_journal"],
)
def test_newest_must_be_a_self_contained_nonempty_cs2_database(tmp_path: Path, failure: str) -> None:
    bbdd, backups = _layout(tmp_path)
    old = _write_backup(backups, "cs2_20260101_000000Z.db")
    newest = backups / "cs2_20260102_000000Z.db"
    if failure == "missing_matches":
        connection = sqlite3.connect(newest)
        try:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
            connection.execute("CREATE TABLE prediction_ledger(ledger_id INTEGER PRIMARY KEY)")
            connection.commit()
        finally:
            connection.close()
    elif failure == "missing_ledger":
        connection = sqlite3.connect(newest)
        try:
            connection.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO matches(match_id) VALUES (1)")
            connection.commit()
        finally:
            connection.close()
    elif failure == "nonempty_wal":
        _write_backup(backups, newest.name)
        Path(str(newest) + "-wal").write_bytes(b"uncheckpointed")
    else:
        _write_backup(backups, newest.name)
        Path(str(newest) + "-journal").write_bytes(b"hot rollback journal")

    plan = plan_backup_retention(bbdd, keep_latest=1)

    assert plan.newest_backup_validation is not None
    assert not plan.newest_backup_validation.ok
    assert f"retained_backup_integrity_failed:{newest.name}" in plan.blocked
    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)
    assert old.exists()
    assert newest.exists()


def test_live_database_symlink_to_backup_blocks_every_deletion_when_supported(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    old = _write_backup(backups, "cs2_20260101_000000Z.db")
    newest = _write_backup(backups, "cs2_20260102_000000Z.db")
    (bbdd / "cs2.db").unlink()
    try:
        os.symlink(newest, bbdd / "cs2.db")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    plan = plan_backup_retention(bbdd, keep_latest=1)

    assert "live_database_unsafe" in plan.blocked
    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)
    assert old.exists()
    assert newest.exists()


def test_symlinked_backups_root_is_rejected_when_supported(tmp_path: Path) -> None:
    bbdd = tmp_path / "BBDD"
    outside = tmp_path / "outside"
    bbdd.mkdir()
    outside.mkdir()
    try:
        os.symlink(outside, bbdd / "backups", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(UnsafeBackupPathError):
        plan_backup_retention(bbdd, keep_latest=1)


def test_existing_lock_blocks_confirmation_without_touching_backups(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    old = _write_backup(backups, "cs2_20260101_000000Z.db")
    newest = _write_backup(backups, "cs2_20260102_000000Z.db")
    plan = plan_backup_retention(bbdd, keep_latest=1)
    lock = backups / ".backup-retention.lock"
    lock.write_text('{"pid":999}', encoding="utf-8")

    with pytest.raises(UnsafeBackupPathError):
        apply_backup_retention(plan, plan.token)

    assert lock.read_text(encoding="utf-8") == '{"pid":999}'
    assert old.exists()
    assert newest.exists()


def test_cleanup_requires_explicit_allowlisted_names_and_keeps_unselected_extra(tmp_path: Path) -> None:
    bbdd, backups = _layout(tmp_path)
    _write_backup(backups, "cs2_20260101_000000Z.db")
    integrity = bbdd / "cs2_integrity_test.db"
    dump = bbdd / "cs2_dump.sql"
    integrity.write_text("scratch db", encoding="utf-8")
    dump.write_text("sql dump", encoding="utf-8")

    plan = plan_extra_cleanup(["cs2_integrity_test.db"], bbdd)

    assert plan.delete_names == ("cs2_integrity_test.db",)
    assert plan.keep_names == ("cs2_dump.sql",)
    result = apply_backup_retention(plan, plan.token)
    assert result.deleted == ("cs2_integrity_test.db",)
    assert not integrity.exists()
    assert dump.read_text(encoding="utf-8") == "sql dump"
    assert (backups / "cs2_20260101_000000Z.db").exists()
    assert (bbdd / "cs2.db").exists()


def test_mid_batch_os_failure_is_explicit_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbdd, backups = _layout(tmp_path)
    first = _write_backup(backups, "cs2_20260101_000000Z.db")
    second = _write_backup(backups, "cs2_20260102_000000Z.db")
    newest = _write_backup(backups, "cs2_20260103_000000Z.db")
    plan = plan_backup_retention(bbdd, keep_latest=1)
    real_unlink = Path.unlink

    def flaky_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == second.name:
            raise PermissionError("simulated locked backup")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(BackupDeletionError) as caught:
        apply_backup_retention(plan, plan.token)

    assert caught.value.deleted == (first.name,)
    assert not first.exists()
    assert second.exists()
    assert newest.exists()
    assert not (backups / ".backup-retention.lock").exists()
    assert (bbdd / "cs2.db").read_text(encoding="utf-8") == "live-sentinel"


def test_disappearing_lock_stops_batch_and_preserves_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbdd, backups = _layout(tmp_path)
    first = _write_backup(backups, "cs2_20260101_000000Z.db")
    second = _write_backup(backups, "cs2_20260102_000000Z.db")
    newest = _write_backup(backups, "cs2_20260103_000000Z.db")
    plan = plan_backup_retention(bbdd, keep_latest=1)
    lock = backups / ".backup-retention.lock"
    real_lexists = os.path.lexists
    real_unlink = Path.unlink
    hide_lock = False

    def simulated_lexists(path: str | os.PathLike[str]) -> bool:
        if hide_lock and Path(path) == lock:
            return False
        return real_lexists(path)

    def unlink_then_lose_lock(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal hide_lock
        real_unlink(path, *args, **kwargs)
        if path.name == first.name:
            hide_lock = True

    monkeypatch.setattr(backup_retention.os.path, "lexists", simulated_lexists)
    monkeypatch.setattr(Path, "unlink", unlink_then_lose_lock)

    try:
        with pytest.raises(BackupDeletionError) as caught:
            apply_backup_retention(plan, plan.token)

        assert caught.value.deleted == (first.name,)
        assert any("mutual exclusion" in note for note in caught.value.__notes__)
        assert not first.exists()
        assert second.exists()
        assert newest.exists()
    finally:
        hide_lock = False
        if real_lexists(lock):
            real_unlink(lock)


@pytest.mark.parametrize("target", ["cs2.db", "../cs2_dump.sql", "BLACKBOX"])
def test_cleanup_rejects_every_non_allowlisted_or_nonlocal_target(tmp_path: Path, target: str) -> None:
    bbdd, _ = _layout(tmp_path)

    with pytest.raises(BackupRetentionError):
        plan_extra_cleanup([target], bbdd)


def test_invalid_config_cannot_expand_cleanup_allowlist(tmp_path: Path) -> None:
    config = tmp_path / "retention.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_keep": 1,
                "cleanup_targets": ["cs2.db"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BackupRetentionConfigError):
        load_retention_config(config)


def test_cli_preview_then_confirm_on_temporary_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bbdd, backups = _layout(tmp_path)
    _write_backup(backups, "cs2_20260101_000000Z.db")
    _write_backup(backups, "cs2_20260102_000000Z.db")

    preview_rc = manage_backups.main(["--bbdd-dir", str(bbdd), "prune", "--keep", "1"])
    preview = json.loads(capsys.readouterr().out)
    token = preview["plan"]["token"]

    assert preview_rc == 0
    assert preview["mode"] == "preview"
    assert preview["plan"]["delete_count"] == 1
    assert (backups / "cs2_20260101_000000Z.db").exists()

    confirmed_rc = manage_backups.main(["--bbdd-dir", str(bbdd), "prune", "--keep", "1", "--confirm", token])
    confirmed = json.loads(capsys.readouterr().out)

    assert confirmed_rc == 0
    assert confirmed["mode"] == "confirmed"
    assert confirmed["deleted"] == ["cs2_20260101_000000Z.db"]
    assert not (backups / "cs2_20260101_000000Z.db").exists()
    assert (backups / APPROVAL_MARKER_NAME).is_file()


def test_cli_auto_without_current_snapshot_is_a_successful_preview_noop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bbdd, backups = _layout(tmp_path)
    oldest = _write_backup(backups, "cs2_20260101_000000Z.db")
    _write_backup(backups, "cs2_20260102_000000Z.db")

    rc = manage_backups.main(["--bbdd-dir", str(bbdd), "prune", "--keep", "1", "--auto"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["mode"] == "preview"
    assert payload["applied"] is False
    assert payload["reason"] == "current_snapshot_required"
    assert oldest.exists()
    assert not (backups / APPROVAL_MARKER_NAME).exists()


def test_manual_confirmation_approves_compatible_auto_retention(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bbdd, backups = _layout(tmp_path)
    oldest = _write_backup(backups, "cs2_20260101_000000Z.db")
    _write_backup(backups, "cs2_20260102_000000Z.db")
    arguments = ["--bbdd-dir", str(bbdd), "prune", "--keep", "1"]

    assert manage_backups.main(arguments) == 0
    token = json.loads(capsys.readouterr().out)["plan"]["token"]
    assert manage_backups.main([*arguments, "--confirm", token]) == 0
    confirmed = json.loads(capsys.readouterr().out)
    marker = backups / APPROVAL_MARKER_NAME

    assert confirmed["mode"] == "confirmed"
    assert marker.is_file()
    assert not oldest.exists()

    previous = backups / "cs2_20260102_000000Z.db"
    newest = _write_backup(backups, "cs2_20260103_000000Z.db")
    assert manage_backups.main([*arguments, "--required-keep", str(newest), "--auto"]) == 0
    automatic = json.loads(capsys.readouterr().out)

    assert automatic["mode"] == "auto"
    assert automatic["applied"] is True
    assert automatic["deleted"] == [previous.name]
    assert not previous.exists()
    assert newest.exists()
    assert marker.is_file()


def test_policy_file_change_requires_a_fresh_manual_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bbdd, backups = _layout(tmp_path)
    config = _write_config(bbdd, indent=2)
    _write_backup(backups, "cs2_20260101_000000Z.db")
    arguments = [
        "--bbdd-dir",
        str(bbdd),
        "--config",
        str(config),
        "prune",
        "--keep",
        "1",
    ]

    assert manage_backups.main(arguments) == 0
    token = json.loads(capsys.readouterr().out)["plan"]["token"]
    assert manage_backups.main([*arguments, "--confirm", token]) == 0
    capsys.readouterr()
    marker_before = (backups / APPROVAL_MARKER_NAME).read_bytes()

    candidate = _write_backup(backups, "cs2_20260102_000000Z.db")
    # Same semantic policy and N, but a changed policy file hash still requires
    # a human to inspect and confirm the new exact contract.
    _write_config(bbdd, indent=4)
    assert manage_backups.main([*arguments, "--required-keep", str(candidate), "--auto"]) == 0
    automatic = json.loads(capsys.readouterr().out)

    assert automatic["mode"] == "preview"
    assert automatic["applied"] is False
    assert automatic["reason"] == "approval_required"
    assert candidate.exists()
    assert (backups / APPROVAL_MARKER_NAME).read_bytes() == marker_before


def test_backup_database_first_run_preserves_every_backup_without_marker(tmp_path: Path) -> None:
    _bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    oldest = _write_backup(backups, "cs2_20260101_000000Z.db")
    previous = _write_backup(backups, "cs2_20260102_000000Z.db")

    result = build_db.backup_database(live, backups)

    retention = result["backup_retention"]
    assert retention["mode"] == "preview"
    assert retention["applied"] is False
    assert retention["reason"] == "approval_required"
    assert oldest.exists()
    assert previous.exists()
    assert Path(result["backup"]).is_file()
    assert not (backups / APPROVAL_MARKER_NAME).exists()


def test_two_backups_in_the_same_second_have_distinct_microsecond_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    moments = iter(
        (
            datetime(2026, 8, 21, 12, 0, 0, 123456, tzinfo=timezone.utc),
            datetime(2026, 8, 21, 12, 0, 0, 654321, tzinfo=timezone.utc),
        )
    )

    class SequencedDatetime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            del tz
            return next(moments)

    monkeypatch.setattr(build_db, "datetime", SequencedDatetime)

    first = build_db.backup_database(live, backups)
    second = build_db.backup_database(live, backups)

    first_path = Path(first["backup"])
    second_path = Path(second["backup"])
    assert first_path.name == "cs2_20260821_120000_123456Z.db"
    assert second_path.name == "cs2_20260821_120000_654321Z.db"
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()


def test_managed_relative_paths_are_normalized_before_required_keep_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = build_db.backup_database(Path("BBDD/cs2.db"), Path("BBDD/backups"))

    created = Path(result["backup"])
    assert created.is_absolute()
    assert created.parent == backups
    assert result["backup_retention"]["required_keep"] == created.name


def test_backup_lock_serializes_snapshot_publish_and_manual_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    previous = _write_backup(backups, "cs2_20260101_000000Z.db")
    real_create = build_db._create_verified_database_backup
    interleaving_checked = False

    def create_with_interleaving(db_path: Path, target: Path) -> dict[str, object]:
        nonlocal interleaving_checked
        with pytest.raises(UnsafeBackupPathError, match="lock already exists"):
            build_db.backup_database(live, backups)
        manual_plan = plan_backup_retention(bbdd, keep_latest=1)
        assert "retention_lock_present" in manual_plan.blocked
        with pytest.raises(UnsafeBackupPathError, match="lock already exists"):
            apply_backup_retention(manual_plan, manual_plan.token)
        interleaving_checked = True
        return real_create(db_path, target)

    monkeypatch.setattr(build_db, "_create_verified_database_backup", create_with_interleaving)

    result = build_db.backup_database(live, backups)

    assert interleaving_checked
    assert previous.exists()
    assert Path(result["backup"]).is_file()
    assert not (backups / ".backup-retention.lock").exists()


def test_sqlite_backup_api_includes_committed_rows_still_in_live_wal(tmp_path: Path) -> None:
    _bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    connection = sqlite3.connect(live)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("INSERT INTO matches(match_id) VALUES (2)")
        connection.execute("INSERT INTO prediction_ledger(ledger_id) VALUES (2)")
        connection.commit()
        wal = Path(str(live) + "-wal")
        assert wal.stat().st_size > 0

        result = build_db.backup_database(live, backups)

        snapshot = Path(result["backup"])
        with closing(sqlite3.connect(snapshot)) as copied:
            assert copied.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert copied.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
            assert copied.execute("SELECT COUNT(*) FROM prediction_ledger").fetchone()[0] == 2
        assert result["backup_validation"]["prediction_ledger_rows"] == 2
    finally:
        connection.close()


def test_live_database_replacement_before_publish_discards_pending_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    replacement = bbdd / "replacement.db"
    with closing(sqlite3.connect(replacement)) as connection:
        connection.execute("CREATE TABLE matches(match_id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE prediction_ledger(ledger_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO matches(match_id) VALUES (99)")
        connection.execute("INSERT INTO prediction_ledger(ledger_id) VALUES (99)")
        connection.commit()
    real_validate = build_db.validate_sqlite_backup

    def replace_live_during_validation(path: Path) -> BackupValidation:
        os.replace(replacement, live)
        return real_validate(path)

    monkeypatch.setattr(build_db, "validate_sqlite_backup", replace_live_during_validation)

    with pytest.raises(RuntimeError, match="identity changed before"):
        build_db.backup_database(live, backups)

    assert not list(backups.glob("cs2_*.db"))
    assert not list(backups.glob("*.pending"))
    assert not (backups / ".backup-retention.lock").exists()
    with closing(sqlite3.connect(live)) as connection:
        assert connection.execute("SELECT match_id FROM matches").fetchone()[0] == 99


def test_backup_database_verifies_then_runs_approved_retention(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bbdd, backups, live, config = _managed_sqlite_layout(tmp_path)
    # Reproduce a future-dated old file: filename order must never evict the
    # snapshot just created by backup_database.
    previous = _write_backup(backups, "cs2_20990101_000000Z.db")
    arguments = ["--bbdd-dir", str(bbdd), "--config", str(config), "prune", "--keep", "1"]
    assert manage_backups.main(arguments) == 0
    token = json.loads(capsys.readouterr().out)["plan"]["token"]
    assert manage_backups.main([*arguments, "--confirm", token]) == 0
    capsys.readouterr()

    result = build_db.backup_database(live, backups)

    validation = result["backup_validation"]
    retention = result["backup_retention"]
    assert validation["ok"] is True
    assert retention["mode"] == "auto"
    assert retention["applied"] is True
    assert retention["deleted"] == [previous.name]
    assert not previous.exists()
    created = Path(result["backup"])
    assert retention["required_keep"] == created.name
    assert created.is_file()
    assert sorted(path.name for path in backups.glob("cs2_*.db")) == [created.name]
    with closing(sqlite3.connect(live)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_successive_operational_snapshots_keep_only_the_newest_backup(
    tmp_path: Path,
) -> None:
    """A normal start/retrain may snapshot repeatedly, but never accumulates files."""

    bbdd, backups, live, config = _managed_sqlite_layout(tmp_path)
    previous = _write_backup(backups, "cs2_20260101_000000Z.db")
    approval = plan_backup_retention(bbdd, keep_latest=1, config_path=config)
    apply_backup_retention(approval, approval.token, record_approval=True)
    live_before = live.read_bytes()

    first_result = build_db.backup_database(live, backups)
    first = Path(first_result["backup"])
    assert first_result["backup_retention"]["mode"] == "auto"
    assert first_result["backup_retention"]["keep_latest"] == 1
    assert first_result["backup_retention"]["deleted"] == [previous.name]
    assert sorted(backups.glob("cs2_*.db")) == [first]

    second_result = build_db.backup_database(live, backups)
    second = Path(second_result["backup"])
    assert second != first
    assert second_result["backup_retention"]["mode"] == "auto"
    assert second_result["backup_retention"]["keep_latest"] == 1
    assert second_result["backup_retention"]["deleted"] == [first.name]
    assert sorted(backups.glob("cs2_*.db")) == [second]
    assert live.read_bytes() == live_before


def test_failed_backup_validation_never_invokes_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbdd, backups, live, _config = _managed_sqlite_layout(tmp_path)
    first = _write_backup(backups, "cs2_20260101_000000Z.db")
    second = _write_backup(backups, "cs2_20260102_000000Z.db")

    monkeypatch.setattr(
        build_db,
        "validate_sqlite_backup",
        lambda _path: BackupValidation(
            name="pending",
            mode="test",
            ok=False,
            result=("simulated verification failure",),
            matches=None,
            prediction_ledger_rows=None,
        ),
    )

    def unexpected_retention(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retention must not run after a failed backup")

    monkeypatch.setattr(build_db, "run_automatic_backup_retention", unexpected_retention)

    with pytest.raises(RuntimeError, match="backup verification failed"):
        build_db.backup_database(live, backups)

    assert first.exists()
    assert second.exists()
    assert sorted(path.name for path in backups.iterdir()) == sorted([first.name, second.name])


def test_cli_reports_structured_partial_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bbdd, backups = _layout(tmp_path)
    first = _write_backup(backups, "cs2_20260101_000000Z.db")
    second = _write_backup(backups, "cs2_20260102_000000Z.db")
    _write_backup(backups, "cs2_20260103_000000Z.db")
    preview = plan_backup_retention(bbdd, keep_latest=1)
    real_unlink = Path.unlink

    def flaky_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name in {second.name, ".backup-retention.lock"}:
            raise PermissionError("simulated locked backup")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    rc = manage_backups.main(["--bbdd-dir", str(bbdd), "prune", "--keep", "1", "--confirm", preview.token])
    captured = capsys.readouterr()
    error = json.loads(captured.err)

    assert rc == 3
    assert not captured.out
    assert error["partial"] is True
    assert error["deleted"] == [first.name]
    assert error["deleted_bytes"] > 0
    assert error["error_type"] == "BackupDeletionError"
    assert any("left in place" in warning for warning in error["warnings"])
    assert (backups / ".backup-retention.lock").exists()


def test_cli_help_documents_preview_first_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(CS2_ROOT / "BBDD" / "manage_backups.py"), "--help"],
        cwd=CS2_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "confirmation" in completed.stdout
    assert "deletion" in completed.stdout
    assert "prune" in completed.stdout
    assert "cleanup" in completed.stdout
    prune_help = subprocess.run(
        [sys.executable, str(CS2_ROOT / "BBDD" / "manage_backups.py"), "prune", "--help"],
        cwd=CS2_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert prune_help.returncode == 0
    assert "--auto" in prune_help.stdout
    assert "--required-keep" in prune_help.stdout

"""Offline storage migration regression tests. Run: python scripts/test_vault.py.

All writes target disposable fixtures. No production DB, model or secret is
opened. This suite is stdlib-only so a clean clone can validate the bootstrap.
"""

import json
from pathlib import Path
import shutil
import subprocess
import tomllib
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import vault_migrate as vault
from vault_paths import state_path


class VaultTests(unittest.TestCase):
    """Migration preserves bytes and fails closed across interruption/relocation."""

    def setUp(self) -> None:
        """Create a project with one representative critical private file."""
        self.temporary = TemporaryDirectory(prefix="vault-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "old location"
        self.source = self.root / "CS2/BBDD/cs2.db"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"binary database fixture\x00\x01")
        self.names = ["CS2/BBDD/cs2.db"]
        mocked = patch.object(vault, "ignored_entries", lambda _: list(self.names))
        mocked.start()
        self.addCleanup(mocked.stop)
        mocked_processes = patch.object(vault, "assert_quiescent")
        mocked_processes.start()
        self.addCleanup(mocked_processes.stop)

    def test_preview_then_move_relocate_and_check(self) -> None:
        """Copied VAULT needs no old root; hashes and original metadata persist."""
        original = vault.digest(self.source)
        self.assertEqual(vault.migrate(self.root)["status"], "planned")
        self.assertTrue(self.source.is_file())
        report = vault.migrate(self.root, apply=True)
        self.assertEqual(report["status"], "migrated")
        self.assertFalse(self.source.exists())
        self.assertEqual(vault.digest(self.root / "VAULT/CS2/BBDD/cs2.db"), original)
        other = self.root.parent / "new location"
        shutil.copytree(self.root / "VAULT", other / "VAULT")
        self.assertEqual(vault.validate_ready(other)["origin_root"], str(self.root))
        self.assertEqual(state_path("CS2/BBDD/cs2.db", repository=other), other / "VAULT/CS2/BBDD/cs2.db")

    def test_conflict_never_overwrites_and_blocks_critical_state(self) -> None:
        """Different destination bytes are never replaced or silently accepted."""
        target = self.root / "VAULT/CS2/BBDD/cs2.db"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"different")
        with self.assertRaises(RuntimeError):
            vault.migrate(self.root, apply=True)
        report = json.loads((self.root / "VAULT/migration.json").read_text(encoding="utf-8"))
        self.assertEqual(report["pending"][0]["reason"], "destination_exists")
        self.assertEqual(target.read_bytes(), b"different")
        self.assertTrue(self.source.is_file())
        with self.assertRaises(RuntimeError):
            vault.validate_ready(self.root)

    def test_failed_rename_retains_original_and_blocks_boot(self) -> None:
        """An access denial is recorded without changing source bytes."""
        with patch.object(vault.os, "rename", side_effect=PermissionError("fixture denial")):
            with self.assertRaises(RuntimeError):
                vault.migrate(self.root, apply=True)
        self.assertTrue(self.source.exists())
        with self.assertRaises(RuntimeError):
            vault.validate_ready(self.root)

    def test_resume_after_interrupted_journal_commit(self) -> None:
        """A verified completed move can be recovered without moving data twice."""
        report = vault.migrate(self.root, apply=True)
        report["status"] = "migrating"
        report["entries"][0]["status"] = "moving"
        vault.save_json(self.root / "VAULT/migration.json", report)
        self.names.clear()
        recovered = vault.migrate(self.root, apply=True)
        self.assertEqual(recovered["entries"][0]["status"], "moved_verified")
        self.assertEqual(vault.validate_ready(self.root)["status"], "migrated")

    def test_missing_copied_database_blocks_start(self) -> None:
        """Missing state cannot be mistaken for a fresh empty installation."""
        vault.migrate(self.root, apply=True)
        (self.root / "VAULT/CS2/BBDD/cs2.db").unlink()
        with self.assertRaises(RuntimeError):
            vault.validate_ready(self.root)

    def test_confined_targets(self) -> None:
        """Reject traversal, drive-relative and absolute targets before moving."""
        for value in ("../other", "C:other", "C:/other", "/other", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vault.confined(self.root, value)

    def test_all_tool_cache_defaults_are_inside_vault(self) -> None:
        """Root invocations and each component must not recreate private caches in code."""
        for component in (".", "CS2", "TENNIS", "CS2/SCRAPER/hltv-scraper-api"):
            directory = vault.ROOT / component
            tools = tomllib.loads((directory / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
            cache_paths = [tools["ruff"]["cache-dir"], tools["mypy"]["cache_dir"]]
            if "pytest" in tools:
                cache_paths.append(tools["pytest"]["ini_options"]["cache_dir"])
            for value in cache_paths:
                with self.subTest(component=component, cache=value):
                    self.assertTrue((directory / value).resolve().is_relative_to(vault.ROOT / "VAULT"))

    def test_empty_leftovers_are_removed_without_removing_files(self) -> None:
        """Empty migration shells disappear; tracked placeholders survive."""
        branch = self.root / "TENNIS/data/old"
        (branch / "empty" / "nested").mkdir(parents=True)
        placeholder = branch / ".gitkeep"
        placeholder.write_bytes(b"")
        vault.remove_empty_residue_directories(self.root, branch)
        self.assertTrue(placeholder.is_file())
        self.assertFalse((branch / "empty").exists())

    def test_empty_cleanup_never_follows_a_symlink(self) -> None:
        """An ignored link cannot cause removal of empty directories elsewhere."""
        branch = self.root / "ignored"
        branch.mkdir()
        outside = self.root.parent / "unrelated"
        (outside / "empty").mkdir(parents=True)
        link = branch / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks is not permitted on this Windows account")
        vault.remove_empty_residue_directories(self.root, branch)
        self.assertTrue((outside / "empty").is_dir())
        self.assertTrue(link.is_symlink())

    def test_noncritical_collision_is_archived_without_overwriting_live_state(self) -> None:
        """A regenerated cache must not prevent migration or replace VAULT data."""
        vault.migrate(self.root, apply=True)
        source = self.root / "TENNIS/.cache/pytest/CACHEDIR.TAG"
        target = self.root / "VAULT/TENNIS/.cache/pytest/CACHEDIR.TAG"
        source.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        source.write_bytes(b"legacy")
        target.write_bytes(b"active")
        self.names[:] = ["TENNIS/.cache/pytest/CACHEDIR.TAG"]
        report = vault.migrate(self.root, apply=True)
        self.assertEqual(report["status"], "migrated")
        self.assertEqual(target.read_bytes(), b"active")
        self.assertFalse(source.exists())
        self.assertEqual(
            (self.root / "VAULT/archive/legacy_collisions/TENNIS/.cache/pytest/CACHEDIR.TAG").read_bytes(), b"legacy"
        )


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is required")
class ResidueAclGuardTests(unittest.TestCase):
    """The admin repair preview refuses broad or newly protected targets."""

    def setUp(self) -> None:
        """Copy only the repair entrypoint into a disposable fake repository."""
        self.temporary = TemporaryDirectory(prefix="acl-preview-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.script = scripts / "repair_vault_residue_acl.ps1"
        shutil.copy2(Path(__file__).with_name(self.script.name), self.script)
        registry = self.root / "VAULT/CS2/MODEL/artifacts/registry"
        (registry / "20260824_125852Z").mkdir(parents=True)
        self.latest = registry / "latest.json"
        self.latest.write_text(json.dumps({"latest": "live"}), encoding="utf-8")
        (registry / "last_good.json").write_text(json.dumps({"last_good": "previous"}), encoding="utf-8")
        self.journal = self.root / "VAULT/migration.json"
        self.journal.write_text(json.dumps({"pending": []}), encoding="utf-8")

    def preview(self) -> subprocess.CompletedProcess[str]:
        """Run without -Apply: this must never request or alter permissions."""
        return subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.script)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_preview_does_not_mutate_files(self) -> None:
        """The exact rejected target is displayed with unchanged pointer bytes."""
        before = self.latest.read_bytes()
        result = self.preview()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("20260824_125852Z", result.stdout)
        self.assertEqual(self.latest.read_bytes(), before)
        self.assertFalse((self.root / "VAULT/verification").exists())

    def test_newly_protected_rejected_version_blocks(self) -> None:
        """A stale cleanup target must not override the current live pointer."""
        self.latest.write_text(json.dumps({"latest": "20260824_125852Z"}), encoding="utf-8")
        result = self.preview()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("now protected", result.stderr)

    def test_unreviewed_journal_path_blocks(self) -> None:
        """The journal cannot expand authority to BBDD or a component root."""
        self.journal.write_text(json.dumps({"pending": [{"source": "TENNIS/BBDD"}]}), encoding="utf-8")
        result = self.preview()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unreviewed residue", result.stderr)


if __name__ == "__main__":
    unittest.main()

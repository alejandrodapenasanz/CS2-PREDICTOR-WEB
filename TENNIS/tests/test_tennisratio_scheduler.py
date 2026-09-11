"""Offline contracts for the optional Windows TennisRatio daily task."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import unittest


TENNIS_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = TENNIS_ROOT / "scripts" / "manage_tennisratio_daily_task.ps1"


class TennisRatioSchedulerContractTests(unittest.TestCase):
    """Keep unattended scheduling thin, deterministic and credential-free."""

    def setUp(self) -> None:
        """Load the PowerShell installer as UTF-8 without executing it."""

        self.script = SCHEDULER_PATH.read_text(encoding="utf-8")

    def test_default_task_runs_absolute_update_only_route_daily(self) -> None:
        """Fix task identity, local time, launcher action and catch-up policy."""

        for fragment in (
            "CS2-Predictor-TennisRatio-Daily",
            "[string]$DailyTime = '06:00'",
            "[IO.Path]::GetFullPath",
            "'run_tennis.ps1'",
            '-File "{0}" -UpdateOnly',
            "New-ScheduledTaskTrigger -Daily",
            "-StartWhenAvailable",
            "-RunOnlyIfNetworkAvailable",
            "-MultipleInstances IgnoreNew",
            "-RestartCount 3",
            "-RestartInterval (New-TimeSpan -Minutes 30)",
            "-ExecutionTimeLimit (New-TimeSpan -Hours 12)",
            "-LogonType Interactive",
            "-RunLevel Limited",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_install_replaces_one_named_task_and_remove_is_idempotent(self) -> None:
        """Require Install/Status/Remove without duplicate task definitions."""

        self.assertIn("[ValidateSet('Install', 'Status', 'Remove')]", self.script)
        self.assertIn("[string]$Mode = 'Status'", self.script)
        self.assertIn("^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$", self.script)
        self.assertIn("Register-ScheduledTask", self.script)
        self.assertIn("-Force | Out-Null", self.script)
        self.assertIn("Get-ScheduledTaskInfo", self.script)
        self.assertIn("Unregister-ScheduledTask", self.script)
        self.assertIn("if ($null -eq $Task)", self.script)
        self.assertEqual(
            len(re.findall(r"(?m)^\s*Register-ScheduledTask\s*`?\s*$", self.script)),
            1,
        )

    def test_installer_does_not_collect_or_embed_credentials(self) -> None:
        """The scheduled task must use the current interactive token only."""

        for forbidden in (
            "Get-Credential",
            "ConvertTo-SecureString",
            "-Password",
            "LogonType Password",
            "LogonType S4U",
            "ServiceAccount",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.script)
        self.assertIn("WindowsIdentity]::GetCurrent().Name", self.script)

    def test_generated_sidecar_database_is_not_versioned(self) -> None:
        """Keep the daily lateral SQLite and its WAL files outside Git."""

        ignored = (TENNIS_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("data/processed/tennisratio.sqlite3*", ignored)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell unavailable")
    def test_installer_has_valid_powershell_syntax(self) -> None:
        """Parse the installer without reading or mutating Task Scheduler."""

        parser_command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{SCHEDULER_PATH}', [ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -gt 0) { "
            "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
        )
        completed = subprocess.run(
            [
                shutil.which("powershell") or "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                parser_command,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

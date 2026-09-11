"""Offline contract tests for TENNIS dependency and PowerShell bootstrapping."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


TENNIS_ROOT = Path(__file__).resolve().parents[1]


class DependencyManifestTests(unittest.TestCase):
    """Keep direct requirements bounded while the generated lock owns pins."""

    def test_direct_requirements_are_complete_and_bounded(self) -> None:
        """Require every audited direct import and reject unbounded roots."""

        raw_lines = (TENNIS_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        requirements = [
            line.strip() for line in raw_lines if line.strip() and not line.lstrip().startswith("#")
        ]
        expected_names = {
            "beautifulsoup4",
            "curl-cffi",
            "joblib",
            "lightgbm",
            "lxml",
            "matplotlib",
            "mypy",
            "numpy",
            "pandas",
            "pyarrow",
            "pytest",
            "requests",
            "ruff",
            "scikit-learn",
            "scrapling",
            "unidecode",
            "urllib3",
        }
        observed_names: set[str] = set()
        for requirement in requirements:
            match = re.match(
                r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?",
                requirement,
            )
            self.assertIsNotNone(match, requirement)
            assert match is not None
            observed_names.add(match.group(1).lower().replace("_", "-"))
            if requirement.startswith("scrapling[fetchers]"):
                self.assertEqual(requirement, "scrapling[fetchers]==0.4.12")
            else:
                self.assertIn(">=", requirement, requirement)
                self.assertIn("<", requirement, requirement)

        self.assertEqual(observed_names, expected_names)
        self.assertIn(
            "pyarrow>=23.0.1,<23.0.2",
            requirements,
            "No ampliar pyarrow sin repetir el smoke de Application Control.",
        )
        self.assertIn(
            "mypy>=1.10.1,<1.10.2",
            requirements,
            "No activar una wheel mypyc bloqueada por Application Control.",
        )


class PowerShellBootstrapTests(unittest.TestCase):
    """Verify fail-fast environment invariants without running the launcher."""

    def setUp(self) -> None:
        """Load the launcher once as UTF-8 text for static assertions."""

        self.launcher_path = TENNIS_ROOT / "run_tennis.ps1"
        self.launcher = self.launcher_path.read_text(encoding="utf-8")

    def test_requires_cpython_313_before_repairing_the_venv(self) -> None:
        """Demand implementation, version, prefix, and safe repair checks."""

        for fragment in (
            "platform.python_implementation()",
            "== 'CPython'",
            "$ExpectedPythonMajor = 3",
            "$ExpectedPythonMinor = 13",
            "pathlib.Path(sys.prefix).resolve()",
            "Find-Cpython313",
            "$previousErrorActionPreference = $ErrorActionPreference",
            "$ErrorActionPreference = 'Continue'",
            "$ErrorActionPreference = $previousErrorActionPreference",
            "'.venv.build'",
            "'.venv.previous'",
            "Remove-TennisManagedDirectory",
            "Remove-Item -LiteralPath $resolved -Recurse -Force",
            "Move-Item -LiteralPath $VenvRoot -Destination $VenvPreviousRoot",
            "-m venv --upgrade $VenvRoot",
            "[switch]$EnvironmentOnly",
            "pipeline omitido por -EnvironmentOnly",
            "Ruta de candidato insegura",
        ):
            self.assertIn(fragment, self.launcher)
        self.assertLess(
            self.launcher.index("$bootstrap = Find-Cpython313"),
            self.launcher.index("New-TennisVenv -BootstrapPython $bootstrap"),
        )
        self.assertNotIn("--clear", self.launcher)
        self.assertNotIn("Reset-TennisVenv", self.launcher)
        self.assertNotIn(".venv.failed.", self.launcher)

    def test_installs_only_the_exact_lock_from_wheels(self) -> None:
        """Forbid hand-written package installs and source distributions."""

        for fragment in (
            "requirements.lock.txt",
            "$RequirementsSource",
            "$script:ManifestCheckCode",
            "metadata.version(name)",
            "--no-deps",
            "--require-hashes",
            "--only-binary=:all:",
            "--requirement', $RequirementsLock",
            "-m pip check",
            "$script:CompiledRuntimeSmokeCode",
            "Test-TennisRuntimeImports",
            "'mypy.main'",
            "'pyarrow'",
            "'pyarrow.parquet'",
            "'lightgbm'",
            'if "--hash=" not in logical:',
            'unexpected = installed_names - set(pins) - {"pip"}',
            "Set-Content -LiteralPath $EnvironmentStamp",
        ):
            self.assertIn(fragment, self.launcher)
        for package_name in ("pandas", "numpy", "lightgbm", "scrapling"):
            self.assertNotIn(f"pip install {package_name}", self.launcher)

    def test_every_real_launcher_run_persists_outcome_and_exit_code(self) -> None:
        """Keep TENNIS stdout/stderr observable beyond the CS2 transcript."""

        for fragment in (
            "$TennisLogsRoot = Join-Path $TennisRoot 'logs'",
            "$LogRetentionKeep = 2",
            '"run_tennis_$TennisLogToken.log"',
            "Start-Transcript -Path $script:TennisLogPath -Force",
            "Stop-Transcript | Out-Null",
            "function Remove-OldTennisLogs",
            "Remove-OldTennisLogs",
            "Complete-TennisLauncher",
            "RUN_END status={0} exit_code={1}",
            "-Outcome 'daily_failed'",
            "-Outcome 'tennisratio_update_failed'",
            "-Outcome 'retrain_failed'",
            "-Outcome 'web_build_failed'",
            "trap {",
            "ERROR no controlado",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.launcher)

        self.assertLess(
            self.launcher.index("Start-Transcript -Path"),
            self.launcher.index("Initialize-TennisEnvironment"),
        )
        script_level_exits = re.findall(
            r"(?m)^\s*exit\s+\$ExitCode\s*$",
            self.launcher,
        )
        self.assertEqual(script_level_exits, ["    exit $ExitCode"])

        ignored = (TENNIS_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("logs/", ignored.splitlines())

    def test_daily_tennisratio_update_uses_fresh_retrain_sources_once(self) -> None:
        """Refresh Sackmann before remapping, then run TennisRatio exactly once."""

        assignment = (
            "$TennisRatioUpdateScript = Join-Path $TennisRoot 'scripts\\update_tennisratio.py'"
        )
        invocation = "& $PythonExecutable $TennisRatioUpdateScript"
        required_paths = """$RequiredPaths = @(
    $RequirementsSource,
    $RequirementsLock,
    $TennisRatioUpdateScript,
    $TennisAbstractUpdateScript
)"""
        for fragment in (
            assignment,
            required_paths,
            "[TENNIS] Actualizando TennisRatio con Scrapling (control automatico diario)",
            invocation,
            "$TennisRatioUpdateExitCode = [int]$LASTEXITCODE",
            "-Outcome 'tennisratio_update_failed'",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.launcher)

        self.assertEqual(self.launcher.count(invocation), 1)
        self.assertLess(
            self.launcher.index("if ($EnvironmentOnly)"),
            self.launcher.index(invocation),
        )
        source_invocation = "& $PythonExecutable $UpdateSourcesScript"
        self.assertIn(source_invocation, self.launcher)
        self.assertLess(self.launcher.index(source_invocation), self.launcher.index(invocation))
        abstract_invocation = (
            "& $PythonExecutable $TennisAbstractUpdateScript @TennisAbstractArguments"
        )
        self.assertLess(self.launcher.index(invocation), self.launcher.index(abstract_invocation))
        self.assertIn("$TennisAbstractArguments += @('--date', $Date)", self.launcher)
        self.assertNotIn("--full-inventory", self.launcher)
        self.assertLess(
            self.launcher.index(invocation),
            self.launcher.index("& $PythonExecutable $DailyScript"),
        )
        training_block = self.launcher.split("$TrainingScripts = @(", 1)[1].split(
            "$DailyArguments = @()", 1
        )[0]
        self.assertNotIn("update_tennisratio.py", training_block)
        self.assertNotIn("if ($Retrain)", training_block)
        self.assertIn("$DailyArguments += '--retrained'", self.launcher)

    def test_update_only_stops_after_one_daily_refresh(self) -> None:
        """Keep the scheduler route isolated from retraining, daily DB and WEB."""

        invocation = "& $PythonExecutable $TennisRatioUpdateScript"
        update_only_block = self.launcher.rsplit("if ($UpdateOnly) {", 1)[-1].split(
            "$TrainingScripts = @(", 1
        )[0]
        for fragment in (
            "[switch]$UpdateOnly",
            "$UpdateOnly -and ($Retrain -or $EnvironmentOnly -or $Date)",
            "-Outcome 'sources_update_only'",
            "--skip-match-charting",
            "& $PythonExecutable $TennisAbstractUpdateScript",
            "-Outcome 'tennis_abstract_update_incomplete'",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.launcher)
        self.assertEqual(self.launcher.count(invocation), 1)
        self.assertIn("Complete-TennisLauncher -ExitCode 0", update_only_block)
        self.assertNotIn("$DailyScript", update_only_block)
        self.assertNotIn("$WebBuildScript", update_only_block)
        self.assertNotIn("$TrainingScripts", update_only_block)

    def test_complete_run_trains_even_without_retrain_switch(self) -> None:
        """Make model refresh part of the default start.ps1 TENNIS route."""

        self.assertIn(
            "((-not [bool]$Date) -or $Retrain)",
            self.launcher,
        )
        training_block = self.launcher.split("$TrainingScripts = @(", 1)[1].split(
            "$DailyArguments = @()", 1
        )[0]
        for fragment in (
            "scripts\\audit_identities.py",
            "scripts\\build_elo.py",
            "scripts\\build_features.py",
            "scripts\\retrain_models.py",
            "& $PythonExecutable $ScriptPath @ScriptArguments",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, training_block)
        self.assertNotIn("if ($Retrain)", training_block)
        self.assertLess(
            self.launcher.index("$TrainingScripts = @("),
            self.launcher.index("& $PythonExecutable $DailyScript"),
        )
        daily_arguments = self.launcher.split("$DailyArguments = @()", 1)[1].split(
            "& $PythonExecutable $DailyScript", 1
        )[0]
        self.assertIn("if ($RunTraining)", daily_arguments)

    def test_embedded_manifest_validator_executes_against_real_files(self) -> None:
        """Execute the embedded Python, not merely PowerShell's outer parser."""

        marker = "$script:ManifestCheckCode = @'\n"
        code = self.launcher.split(marker, 1)[1].split("\n'@", 1)[0]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(TENNIS_ROOT / "requirements.txt"),
                str(TENNIS_ROOT / "requirements.lock.txt"),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(shutil.which("powershell"), "PowerShell unavailable")
    def test_launcher_has_valid_powershell_syntax(self) -> None:
        """Parse the script without executing bootstrap, pip, or pipeline code."""

        parser_command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{self.launcher_path}', [ref]$tokens, [ref]$errors) | Out-Null; "
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

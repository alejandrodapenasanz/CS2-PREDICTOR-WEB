"""Pruebas de los contratos documentales y los lanzadores integrados."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from src.config import PROJECT_ROOT


class DailyDocumentationTests(unittest.TestCase):
    """Evita que la integración raíz diverja de su documentación."""

    def test_integration_documents_both_pipelines_and_retrain(self) -> None:
        """Fija el orden conjunto y el reentreno coordinado."""

        documentation = (
            PROJECT_ROOT / "docs" / "integracion.md"
        ).read_text(encoding="utf-8")
        self.assertIn("start.ps1' -Retrain", documentation)
        self.assertIn("CS2/start.ps1", documentation)
        self.assertIn("TENNIS/run_tennis.ps1", documentation)
        self.assertIn("Para revertir", documentation)

    def test_root_launcher_runs_cs2_then_tennis_and_propagates_retrain(
        self,
    ) -> None:
        """Comprueba orden, fail-fast y separación de argumentos."""

        launcher = (PROJECT_ROOT.parent / "start.ps1").read_text(
            encoding="utf-8"
        )
        required = (
            "$ForwardedArguments = @($args)",
            "-Arguments $ForwardedArguments -Name 'Retrain'",
            "$Cs2Launcher = Join-Path $PSScriptRoot 'CS2\\start.ps1'",
            "$TennisLauncher = Join-Path $PSScriptRoot 'TENNIS\\run_tennis.ps1'",
            "-File $Cs2Launcher @ForwardedArguments",
            "if ($Cs2ExitCode -ne 0)",
            "$TennisArguments += '-Retrain'",
            "-File $TennisLauncher @TennisArguments",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, launcher)
        self.assertLess(
            launcher.index("-File $Cs2Launcher @ForwardedArguments"),
            launcher.index("-File $TennisLauncher @TennisArguments"),
        )

    @unittest.skipUnless(
        shutil.which("powershell.exe"),
        "PowerShell es necesario para validar el lanzador raíz.",
    )
    def test_root_launcher_isolates_children_and_propagates_exit_codes(
        self,
    ) -> None:
        """Ejecuta stubs con rutas con espacios sin tocar los pipelines reales."""

        with TemporaryDirectory(
            dir=PROJECT_ROOT / "tests",
            prefix=".root-launcher-",
        ) as temporary:
            fake_root = Path(temporary) / "repository with spaces"
            cs2_root = fake_root / "CS2"
            tennis_root = fake_root / "TENNIS"
            cs2_root.mkdir(parents=True)
            tennis_root.mkdir(parents=True)
            shutil.copy2(PROJECT_ROOT.parent / "start.ps1", fake_root)

            stub = r"""
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'args.log') `
    -Value ($args -join "`n") -NoNewline
$ExitVariable = if ($PSScriptRoot -like '*\CS2') {
    'CS2_STUB_EXIT'
} else {
    'TENNIS_STUB_EXIT'
}
$ExitValue = [Environment]::GetEnvironmentVariable($ExitVariable)
$Code = if ($ExitValue) { [int]$ExitValue } else { 0 }
exit $Code
""".strip()
            (cs2_root / "start.ps1").write_text(stub, encoding="utf-8")
            (tennis_root / "run_tennis.ps1").write_text(
                stub,
                encoding="utf-8",
            )

            def execute(
                arguments: list[str],
                *,
                cs2_exit: int = 0,
                tennis_exit: int = 0,
            ) -> subprocess.CompletedProcess[str]:
                """Ejecuta el wrapper falso y reinicia sus logs."""

                for log_path in (
                    cs2_root / "args.log",
                    tennis_root / "args.log",
                ):
                    log_path.unlink(missing_ok=True)
                environment = os.environ.copy()
                environment["CS2_STUB_EXIT"] = str(cs2_exit)
                environment["TENNIS_STUB_EXIT"] = str(tennis_exit)
                return subprocess.run(
                    [
                        shutil.which("powershell.exe") or "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(fake_root / "start.ps1"),
                        *arguments,
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            normal = execute(["-SkipScrape", "-MaxMatches", "7"])
            self.assertEqual(normal.returncode, 0, normal.stderr)
            self.assertEqual(
                (cs2_root / "args.log").read_text(encoding="utf-8"),
                "-SkipScrape\n-MaxMatches\n7",
            )
            self.assertEqual(
                (tennis_root / "args.log").read_text(encoding="utf-8"),
                "",
            )

            retrain = execute(["-retrain"])
            self.assertEqual(retrain.returncode, 0, retrain.stderr)
            self.assertEqual(
                (cs2_root / "args.log").read_text(encoding="utf-8"),
                "-retrain",
            )
            self.assertEqual(
                (tennis_root / "args.log").read_text(encoding="utf-8"),
                "-Retrain",
            )

            dry_run = execute(["-DryRun"])
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertFalse((tennis_root / "args.log").exists())

            what_if = execute(["-WhatIf"])
            self.assertEqual(what_if.returncode, 0, what_if.stderr)
            self.assertFalse((tennis_root / "args.log").exists())

            cs2_failure = execute([], cs2_exit=37)
            self.assertEqual(cs2_failure.returncode, 37)
            self.assertFalse((tennis_root / "args.log").exists())

            tennis_failure = execute([], tennis_exit=23)
            self.assertEqual(tennis_failure.returncode, 23)

    def test_launcher_resolves_every_path_from_its_own_location(self) -> None:
        """Comprueba que el lanzador no depende del directorio actual."""

        launcher_path = PROJECT_ROOT / "run_tennis.ps1"
        launcher = launcher_path.read_text(encoding="utf-8")
        self.assertIn("$MyInvocation.MyCommand.Path", launcher)
        self.assertIn("'.venv\\Scripts\\Activate.ps1'", launcher)
        self.assertIn("'scripts\\daily_predictions.py'", launcher)
        self.assertNotIn("start.ps1", launcher)

    def test_readme_documents_full_refresh_and_daily_order(self) -> None:
        """Exige instalación, fuentes, Elo, features, reentreno y predicción."""

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        expected_fragments = (
            "python -m pip install -r",
            "python scripts\\update_sources.py",
            "python scripts\\build_elo.py",
            "python scripts\\build_features.py",
            "python scripts\\retrain_models.py",
            ".\\TENNIS\\run_tennis.ps1",
            "el edge solo se publica si",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)


if __name__ == "__main__":
    unittest.main()

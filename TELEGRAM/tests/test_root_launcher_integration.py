"""Offline behavioral checks for the repository root PowerShell launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


TELEGRAM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TELEGRAM_ROOT.parent
ROOT_LAUNCHER = REPOSITORY_ROOT / "start.ps1"


def _find_powershell() -> str:
    """Return an available PowerShell executable or skip the Windows-only tests."""

    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable on this test host.")
    return executable


def _stub_script(component: str, exit_variable: str) -> str:
    """Build a network-free launcher stub that logs arguments and exits predictably."""

    return (
        f"$Line = '{component}|' + ($args -join ',')\n"
        "[System.IO.File]::AppendAllText("
        "$env:TG_TEST_LOG, $Line + [Environment]::NewLine)\n"
        f"$Code = [int]::Parse($env:{exit_variable})\n"
        "exit $Code\n"
    )


def _prepare_launcher_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the root launcher beside isolated CS2, Telegram, and Tennis stubs."""

    launcher = tmp_path / "start.ps1"
    shutil.copy2(ROOT_LAUNCHER, launcher)
    log_path = tmp_path / "calls.log"

    launchers = {
        tmp_path / "CS2" / "start.ps1": _stub_script("CS2", "TG_TEST_CS2_EXIT"),
        tmp_path / "TELEGRAM" / "run_telegram.ps1": _stub_script("TELEGRAM", "TG_TEST_TELEGRAM_EXIT"),
        tmp_path / "TENNIS" / "run_tennis.ps1": _stub_script("TENNIS", "TG_TEST_TENNIS_EXIT"),
    }
    for path, content in launchers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return launcher, log_path


def _run_launcher(
    tmp_path: Path,
    *arguments: str,
    cs2_exit: int = 0,
    telegram_exit: int = 0,
    tennis_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the copied launcher with deterministic child exit codes."""

    launcher, log_path = _prepare_launcher_tree(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "TG_TEST_LOG": str(log_path),
            "TG_TEST_CS2_EXIT": str(cs2_exit),
            "TG_TEST_TELEGRAM_EXIT": str(telegram_exit),
            "TG_TEST_TENNIS_EXIT": str(tennis_exit),
        }
    )
    completed = subprocess.run(
        [
            _find_powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    return completed, calls


def test_root_launcher_declares_telegram_between_cs2_and_tennis() -> None:
    """Keep the static launcher order and Telegram path explicit."""

    source = ROOT_LAUNCHER.read_text(encoding="utf-8")

    assert "'TELEGRAM\\run_telegram.ps1'" in source
    assert source.index("-File $Cs2Launcher") < source.index("-File $TelegramLauncher")
    assert source.index("-File $TelegramLauncher") < source.index("-File $TennisLauncher")
    assert "exit $TelegramExitCode" in source


def test_telegram_failure_still_runs_tennis_and_retrain_skips_telegram(
    tmp_path: Path,
) -> None:
    """Return Telegram's failure only after a successful Tennis run."""

    completed, calls = _run_launcher(
        tmp_path,
        "-Retrain",
        telegram_exit=7,
        tennis_exit=0,
    )

    assert completed.returncode == 7
    assert calls == ["CS2|-Retrain", "TELEGRAM|", "TENNIS|-Retrain"]


def test_tennis_failure_has_priority_over_telegram_failure(tmp_path: Path) -> None:
    """Return Tennis's code after both downstream launchers have run."""

    completed, calls = _run_launcher(
        tmp_path,
        telegram_exit=7,
        tennis_exit=9,
    )

    assert completed.returncode == 9
    assert calls == ["CS2|", "TELEGRAM|", "TENNIS|"]


@pytest.mark.parametrize("switch_name", ["-DryRun", "-WhatIf"])
def test_preview_switches_skip_telegram_and_tennis(
    tmp_path: Path,
    switch_name: str,
) -> None:
    """Run only CS2 when the root launcher is in preview mode."""

    completed, calls = _run_launcher(tmp_path, switch_name)

    assert completed.returncode == 0
    assert calls == [f"CS2|{switch_name}"]


def test_cs2_failure_stops_before_telegram_and_tennis(tmp_path: Path) -> None:
    """Preserve the existing fail-fast contract for the primary CS2 pipeline."""

    completed, calls = _run_launcher(tmp_path, cs2_exit=5)

    assert completed.returncode == 5
    assert calls == ["CS2|"]

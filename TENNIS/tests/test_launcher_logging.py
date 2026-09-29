"""Native Python diagnostics and exit codes survive PowerShell transcripts."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_streams_are_logged_without_turning_stderr_into_failure(tmp_path, exit_code):
    """A warning stays a warning; a failure retains its traceback and status."""
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "native fixture.py"
    script.write_text(
        "import sys\nprint('fixture stdout: predicción', flush=True)\n"
        "print('fixture stderr diagnostic: calibración', file=sys.stderr, flush=True)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    log = tmp_path / "native transcript.log"

    def quoted(path):
        """Quote a literal path for PowerShell without interpolation."""
        return "'" + str(path).replace("'", "''") + "'"

    command = (
        f". {quoted(root / 'scripts/launcher.helpers.ps1')}; "
        f"Start-Transcript -Path {quoted(log)} | Out-Null; "
        "$ErrorActionPreference='Stop'; "
        "$env:PYTHONIOENCODING='ascii'; "
        f"$code = Invoke-TennisPython -Python {quoted(sys.executable)} -Arguments @({quoted(script)}); "
        "if ($env:PYTHONIOENCODING -ne 'ascii') { throw 'Encoding not restored' }; "
        "Write-Host ('native_exit=' + $code); Stop-Transcript | Out-Null; exit $code"
    )
    completed = subprocess.run(
        [
            shutil.which("powershell"),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == exit_code, completed.stderr
    transcript = log.read_text(encoding="utf-8-sig")
    assert "fixture stdout: predicción" in transcript
    assert "fixture stderr diagnostic: calibración" in transcript
    assert f"native_exit={exit_code}" in transcript


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("abandon", [False, True])
def test_launchers_serialize_and_recover_an_interrupted_owner(tmp_path, abandon):
    """A second launcher waits and then resumes, even after an owner's crash."""
    helper = Path(__file__).resolve().parents[1] / "scripts/launcher.helpers.ps1"

    def quoted(path):
        """Render only fixture/helper literal paths for PowerShell."""
        return "'" + str(path).replace("'", "''") + "'"

    def launch(marker, *, hold):
        """Start one isolated lease owner with controlled standard input."""
        command = (
            f". {quoted(helper)}; $lease=Enter-TennisRun -StateRoot {quoted(tmp_path)}; "
            f"[IO.File]::WriteAllText({quoted(marker)},'acquired'); "
            + ("[Console]::ReadLine() | Out-Null; " if hold else "")
            + "$lease.ReleaseMutex(); $lease.Dispose()"
        )
        return subprocess.Popen(
            [
                shutil.which("powershell"),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    first_marker, second_marker = tmp_path / "first", tmp_path / "second"
    first = launch(first_marker, hold=True)
    second = None
    try:
        deadline = time.monotonic() + 15
        while not first_marker.exists() and time.monotonic() < deadline:
            assert first.poll() is None, first.communicate()
            time.sleep(0.05)
        assert first_marker.exists()
        second = launch(second_marker, hold=False)
        time.sleep(1.5)
        assert second.poll() is None
        assert not second_marker.exists()
        if abandon:
            first.terminate()
            first.communicate(timeout=10)
        else:
            first.communicate(input="release\n", timeout=10)
            assert first.returncode == 0
        stdout, stderr = second.communicate(timeout=10)
        assert second.returncode == 0, (stdout, stderr)
        assert second_marker.read_text() == "acquired"
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

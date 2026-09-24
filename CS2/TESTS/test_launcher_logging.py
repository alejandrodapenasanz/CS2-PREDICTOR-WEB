"""CS2 native stderr remains visible without becoming a synthetic PowerShell error."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_logging_preserves_warnings_and_real_failure(tmp_path, exit_code):
    """Log both UTF-8 streams, throw only on failure, and restore the environment."""
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "native fixture.py"
    fixture.write_text(
        "import sys\nprint('fixture stdout: predicción', flush=True)\n"
        "print('fixture diagnostic: calibración', file=sys.stderr, flush=True)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    transcript_path = tmp_path / "native transcript.log"

    def quoted(path):
        """Render a literal PowerShell path without interpolation."""
        return "'" + str(path).replace("'", "''") + "'"

    command = (
        f". {quoted(root / 'PIPELINE/pipeline.helpers.ps1')}; "
        f"Start-Transcript -Path {quoted(transcript_path)} | Out-Null; "
        "$ErrorActionPreference='Stop'; $env:PYTHONIOENCODING='ascii'; $failed=$false; "
        "try { "
        f"$seconds=Invoke-Native -FilePath {quoted(sys.executable)} -Arguments @({quoted(fixture)}); "
        "if ($seconds -isnot [double]) { throw 'Invalid elapsed duration' } "
        "} catch { $failed=$true; Write-Host ('caught=' + $_.Exception.Message) }; "
        "if ($env:PYTHONIOENCODING -ne 'ascii' -or $ErrorActionPreference -ne 'Stop') "
        "{ throw 'Native preferences not restored' }; "
        "Write-Host ('failed=' + $failed); Stop-Transcript | Out-Null"
    )
    result = subprocess.run(
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
    assert result.returncode == 0, result.stderr
    transcript = transcript_path.read_text(encoding="utf-8-sig")
    assert "fixture stdout: predicción" in transcript
    assert "fixture diagnostic: calibración" in transcript
    assert "NativeCommandError" not in transcript
    assert f"failed={exit_code != 0}" in transcript
    if exit_code:
        assert f"fallo con exit code {exit_code}:" in transcript
    else:
        assert not any(line.startswith("caught=") for line in transcript.splitlines())

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


@pytest.mark.skipif(not shutil.which("powershell"), reason="Windows PowerShell unavailable")
@pytest.mark.parametrize(
    ("probe", "first_import"),
    [("Test-ModelRuntimeImports", "catboost"), ("Test-ScraperRuntimeImports", "curl_cffi")],
)
@pytest.mark.parametrize("mode", ["warning", "missing", "policy"])
def test_runtime_probes_keep_traceback_and_stop_on_os_policy(tmp_path, probe, first_import, mode):
    """Preserve diagnostics/preferences; allow repair only for ordinary import errors."""
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / f"{first_import}.py"
    ending = {
        "warning": "raise SystemExit(0)",
        "missing": "raise ImportError('fixture_missing_runtime_dependency')",
        "policy": "raise OSError(4551, 'fixture_application_control_denied')",
    }[mode]
    fixture.write_text(
        f"import sys\nprint('fixture stderr: diagnóstico completo', file=sys.stderr, flush=True)\n{ending}\n",
        encoding="utf-8",
    )
    transcript_path = tmp_path / "runtime transcript.log"
    log_path = tmp_path / "runtime.jsonl"

    def quoted(path):
        """Render a literal PowerShell argument without interpolation."""
        return "'" + str(path).replace("'", "''") + "'"

    command = (
        f". {quoted(root / 'PIPELINE/pipeline.helpers.ps1')}; "
        f"Start-Transcript -Path {quoted(transcript_path)} | Out-Null; "
        f"Initialize-PipelineLogging -JsonlPath {quoted(log_path)} -Level DEBUG; "
        "$ErrorActionPreference='Stop'; $env:PYTHONIOENCODING='ascii'; "
        "$beforeEncoding=[Console]::OutputEncoding.CodePage; "
        f"$env:PYTHONPATH={quoted(tmp_path)}; $caught=$false; "
        f"try {{ $valid={probe} -Python {quoted(sys.executable)}; "
        "if ($valid -isnot [bool]) { throw 'probe must return a boolean' }; "
        "Write-Host ('valid=' + $valid) "
        "} catch { $caught=$true; Write-Host ('caught=' + $_.Exception.Message) }; "
        "if ($env:PYTHONIOENCODING -ne 'ascii' -or $ErrorActionPreference -ne 'Stop' -or "
        "[Console]::OutputEncoding.CodePage -ne $beforeEncoding) "
        "{ throw 'Probe preferences not restored' }; "
        "Write-Host ('policyStop=' + $caught); Stop-Transcript | Out-Null"
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
    assert result.returncode == 0, result.stdout + result.stderr
    transcript = transcript_path.read_text(encoding="utf-8-sig")
    log = log_path.read_text(encoding="utf-8-sig")
    assert "fixture stderr: diagnóstico completo" in transcript
    assert "NativeCommandError" not in transcript
    assert f"policyStop={mode == 'policy'}" in transcript
    if mode == "policy":
        assert "caught=Windows Application Control" in transcript
        assert "fixture_application_control_denied" in transcript
        assert "fixture_application_control_denied" in log
    else:
        assert f"valid={mode == 'warning'}" in transcript
        if mode == "missing":
            assert "Traceback (most recent call last):" in transcript
            assert "fixture_missing_runtime_dependency" in transcript
            assert "fixture_missing_runtime_dependency" in log

"""Behavioral proof that the import-coverage gate detects missing dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CS2_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CS2_ROOT.parent
CHECKER = REPOSITORY_ROOT / "scripts" / "check_import_coverage.py"
SMOKE = CS2_ROOT / "MODEL" / "smoke_pipeline.py"


def _run(requirements: Path) -> subprocess.CompletedProcess[str]:
    """Run coverage on the deterministic smoke entrypoint."""

    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--requirements",
            str(requirements),
            "--source",
            str(SMOKE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_coverage_fails_if_a_dependency_is_removed_then_passes_restored(
    tmp_path: Path,
) -> None:
    """Remove PyYAML in a temporary manifest, observe failure, then use the original."""

    original = CS2_ROOT / "requirements.txt"
    without_yaml = tmp_path / "requirements-without-yaml.txt"
    lines = original.read_text(encoding="utf-8").splitlines()
    without_yaml.write_text(
        "\n".join(line for line in lines if not line.lower().startswith("pyyaml")) + "\n",
        encoding="utf-8",
    )

    missing = _run(without_yaml)
    assert missing.returncode == 1
    assert "MISSING dependency: import 'yaml' requires 'pyyaml'" in missing.stderr

    restored = _run(original)
    assert restored.returncode == 0, restored.stderr
    assert "import_coverage=ok" in restored.stdout

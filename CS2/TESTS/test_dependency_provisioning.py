"""Verify the locked, component-scoped Python provisioning contracts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


CS2_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CS2_ROOT.parent
HELPERS = CS2_ROOT / "PIPELINE" / "pipeline.helpers.ps1"


def _direct_requirement_names(path: Path) -> set[str]:
    """Return canonical direct project names from a simple input manifest."""
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        assert match, f"Unsupported test requirement line: {line}"
        names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def _powershell_literal(path: Path) -> str:
    """Quote a filesystem path as a single-quoted PowerShell literal."""
    return "'" + str(path).replace("'", "''") + "'"


def test_root_manifest_contains_model_enrichment_and_tooling_only() -> None:
    """Keep CS2 runtime dependencies complete without leaking scraper/API packages."""
    names = _direct_requirement_names(CS2_ROOT / "requirements.txt")
    required = {
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "lightgbm",
        "xgboost",
        "catboost",
        "shap",
        "numba",
        "optuna",
        "pyyaml",
        "requests",
        "parsel",
        "pytest",
        "pytest-cov",
        "ruff",
        "mypy",
    }
    scraper_or_api = {
        "cloudscraper",
        "scrapy",
        "scrapy-impersonate",
        "nodriver",
        "itemadapter",
        "flask",
        "flasgger",
        "pytest-flask",
        "scrapling",
    }
    assert required <= names
    assert names.isdisjoint(scraper_or_api)


def test_launchers_use_locks_without_manual_dependency_lists() -> None:
    """Require both venvs to derive validity and installation from their locks."""
    helper = HELPERS.read_text(encoding="utf-8")
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")
    config = (CS2_ROOT / "PIPELINE" / "pipeline.config.psd1").read_text(encoding="utf-8")

    assert "ModelImports" not in start + config
    assert "ModelPipPackages" not in start + config
    assert "ScraperBaseImports" not in start + config
    assert "ScraperStealthImports" not in start + config
    assert "Ensure-ModelPython -ProjectRoot $Root" in start
    assert "-BaseImports" not in start
    assert "Test-ModelRuntimeImports" in helper
    assert "import shap" in helper
    assert "Test-ScraperRuntimeImports" in helper
    assert "import curl_cffi" in helper
    assert "-StealthImports" not in start
    assert "[string[]]$Imports" not in helper
    assert "[string[]]$PipPackages" not in helper
    assert "--no-deps" in helper
    assert "--only-binary=:all:" in helper
    assert "--require-hashes" in helper
    assert "--find-links" in helper
    assert "requirements.lock.txt" in helper
    assert "Get-SystemPython" not in helper
    assert "$python = Get-ModelBasePython -DryRun:$DryRun" in helper
    assert "unexpected distributions not present in active lock pins" in helper
    assert "Ruta de venv del scraper fuera de su componente" in helper
    assert helper.count("'.venv.build'") == 2
    assert helper.count("'.venv.previous'") == 2
    assert helper.count("Move-Item -LiteralPath $venvDir -Destination $previousDir") == 2
    assert helper.count("Move-Item -LiteralPath $buildDir -Destination $venvDir") == 2
    assert helper.count("Move-Item -LiteralPath $venvDir -Destination $buildDir") == 2
    assert helper.count("Move-Item -LiteralPath $previousDir -Destination $venvDir") == 2
    assert "Remove-Item -LiteralPath $venvDir" not in helper
    assert helper.count("'--force-reinstall'") == 2
    assert "pip', 'install', 'Scrapy'" not in helper
    assert 'pip", "install", "Scrapy"' not in helper


def test_sklearn_manifest_preserves_serialized_production_compatibility() -> None:
    from packaging.requirements import Requirement

    requirements = (CS2_ROOT / "requirements.txt").read_text(encoding="utf-8")
    sklearn = next(Requirement(line) for line in requirements.splitlines() if line.startswith("scikit-learn"))
    assert str(sklearn.specifier) == "==1.8.0"
    lock = (CS2_ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
    assert re.search(r"^scikit-learn==1\.8\.0\s*\\", lock, flags=re.MULTILINE)


def test_python_and_ci_target_exact_313_and_locked_install() -> None:
    """Pin interpreter policy consistently in tooling, training, and CI."""
    pyproject = (CS2_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runner = (CS2_ROOT / "MODEL" / "run_professional_training.py").read_text(encoding="utf-8")
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'target-version = "py313"' in pyproject
    assert 'python_version = "3.13"' in pyproject
    assert "sys.version_info[:2] != (3, 13)" in runner
    assert 'str(ROOT / "requirements.lock.txt")' in runner
    assert 'str(ROOT / "requirements.txt")' not in runner
    assert '"--require-hashes"' in runner
    assert '"pip", "check"' in runner
    assert 'python-version: "3.13"' in workflow
    assert "--no-deps --only-binary=:all: --require-hashes -r requirements.lock.txt" in workflow


def test_documentation_matches_fail_closed_lock_policy() -> None:
    """Keep user-facing commands aligned with the implemented provisioning path."""
    readme = (CS2_ROOT / "README.md").read_text(encoding="utf-8")
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")

    assert "Si no hay Python 3.10-3.13" not in readme
    assert "el scraper degrada" not in readme
    assert "--resolver=backtracking --generate-hashes" in readme
    assert "--allow-unsafe --strip-extras --no-emit-index-url" in readme
    assert '--pip-args="--only-binary=:all:"' in readme
    assert ".venv.build" in readme
    assert ".venv.previous" in readme
    assert "guardas HLTV, deps, exit-codes" not in start


def test_powershell_lock_validator_accepts_derived_hashed_fixture(tmp_path: Path) -> None:
    """Exercise the PowerShell lock validator without network or installation."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")

    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock.txt"
    requirements.write_text("alpha>=1\nbeta==2\n", encoding="utf-8")
    lock.write_text(
        f"alpha==1.5 \\\n    --hash=sha256:{'a' * 64}\nbeta==2.0 \\\n    --hash=sha256:{'b' * 64}\n",
        encoding="utf-8",
    )
    command = (
        f". {_powershell_literal(HELPERS)}; "
        f"Assert-RequirementsLockDerived -Python {_powershell_literal(Path(sys.executable))} "
        f"-RequirementsPath {_powershell_literal(requirements)} -LockPath {_powershell_literal(lock)}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_powershell_lock_validator_rejects_missing_hash(tmp_path: Path) -> None:
    """Reject a lock pin that cannot be verified cryptographically."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")

    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock.txt"
    requirements.write_text("alpha>=1\n", encoding="utf-8")
    lock.write_text("alpha==1.5\n", encoding="utf-8")
    command = (
        f". {_powershell_literal(HELPERS)}; "
        f"Assert-RequirementsLockDerived -Python {_powershell_literal(Path(sys.executable))} "
        f"-RequirementsPath {_powershell_literal(requirements)} -LockPath {_powershell_literal(lock)}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "without sha256 hashes" in (completed.stderr + completed.stdout)


def test_powershell_lock_validator_rejects_stale_direct_pin(tmp_path: Path) -> None:
    """Reject a hashed lock whose direct pin violates the input constraint."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")

    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock.txt"
    requirements.write_text("alpha>=2\n", encoding="utf-8")
    lock.write_text(
        f"alpha==1.5 \\\n    --hash=sha256:{'a' * 64}\n",
        encoding="utf-8",
    )
    command = (
        f". {_powershell_literal(HELPERS)}; "
        f"Assert-RequirementsLockDerived -Python {_powershell_literal(Path(sys.executable))} "
        f"-RequirementsPath {_powershell_literal(requirements)} -LockPath {_powershell_literal(lock)}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    output = re.sub(r"\s+", " ", completed.stderr + completed.stdout)
    assert "incompatible with requirements.txt" in output


def test_locked_environment_rejects_unlocked_distributions(tmp_path: Path) -> None:
    """Reject an interpreter containing distributions beyond active pins and pip."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")

    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(f"pytest=={pytest.__version__}\n", encoding="utf-8")
    command = (
        f". {_powershell_literal(HELPERS)}; "
        f"$valid = Test-LockedEnvironment -Python {_powershell_literal(Path(sys.executable))} "
        f"-LockPath {_powershell_literal(lock)}; if ($valid) {{ exit 1 }}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    output = re.sub(r"\s+", " ", completed.stderr + completed.stdout)
    assert "unexpected distributions not present in active lock pins" in output

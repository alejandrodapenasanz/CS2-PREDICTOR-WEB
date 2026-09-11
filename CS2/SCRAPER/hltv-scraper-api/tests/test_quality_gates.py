"""Regression tests for the scraper quality gates."""

from __future__ import annotations

from pathlib import Path

from scripts.check_import_coverage import DEFAULT_REQUIREMENTS, uncovered_imports
from scripts.check_import_coverage import main as check_imports_main
from scripts.smoke_boot import smoke_cli_entrypoints, smoke_flask, smoke_scrapy


def test_all_direct_imports_are_declared() -> None:
    assert uncovered_imports() == {}


def test_import_coverage_detects_removed_dependency(tmp_path: Path, capsys) -> None:
    """Prove the gate fails when an imported direct dependency is removed."""

    original = DEFAULT_REQUIREMENTS.read_text(encoding="utf-8")
    without_requests = "\n".join(line for line in original.splitlines() if not line.lower().startswith("requests"))
    temporary_requirements = tmp_path / "requirements-without-requests.txt"
    temporary_requirements.write_text(without_requests, encoding="utf-8")

    assert check_imports_main(["--requirements", str(temporary_requirements)]) == 1
    assert "import 'requests' requires 'requests'" in capsys.readouterr().err
    assert check_imports_main(["--requirements", str(DEFAULT_REQUIREMENTS)]) == 0


def test_flask_entrypoint_boots_offline() -> None:
    smoke_flask()


def test_scrapy_entrypoint_boots_offline() -> None:
    smoke_scrapy()


def test_collection_entrypoints_boot_offline() -> None:
    smoke_cli_entrypoints()

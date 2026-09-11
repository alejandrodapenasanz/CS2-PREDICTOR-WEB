"""Offline tests for the scraper dependency and launcher contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

from scripts import collect_hltv_data
from scripts.verify_flasgger_wheel import (
    EXPECTED_SHA256,
    WHEEL_FILENAME,
    WheelVerificationError,
    sha256_file,
    verify_flasgger_wheel,
)

SCRAPER_ROOT = Path(__file__).resolve().parents[1]
CS2_ROOT = SCRAPER_ROOT.parents[1]
PIPELINE_START = CS2_ROOT / "PIPELINE" / "start.py"


def _direct_requirements() -> dict[str, Requirement]:
    """Parse the maintained direct requirements without accessing an index."""

    requirements: dict[str, Requirement] = {}
    for raw_line in (SCRAPER_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        requirement = Requirement(line)
        requirements[requirement.name.lower()] = requirement
    return requirements


def test_vendored_flasgger_wheel_matches_digest_and_metadata() -> None:
    """Accept only the checked-in flasgger wheel with its reviewed digest."""

    wheel = SCRAPER_ROOT / "wheels" / WHEEL_FILENAME

    assert sha256_file(wheel) == EXPECTED_SHA256
    assert verify_flasgger_wheel(wheel) == wheel.resolve()


def test_vendored_wheel_rejects_tampering(tmp_path: Path) -> None:
    """Reject modified bytes even when the allowlisted filename is preserved."""

    original = SCRAPER_ROOT / "wheels" / WHEEL_FILENAME
    tampered = tmp_path / WHEEL_FILENAME
    tampered.write_bytes(original.read_bytes() + b"tampered")

    with pytest.raises(WheelVerificationError, match="SHA-256 mismatch"):
        verify_flasgger_wheel(tampered)


def test_direct_requirements_are_complete_and_bounded() -> None:
    """Declare direct parser/transport imports and bound every dependency."""

    requirements = _direct_requirements()

    for required_name in (
        "flask",
        "flasgger",
        "scrapy",
        "cloudscraper",
        "itemadapter",
        "parsel",
        "requests",
        "scrapy-impersonate",
        "curl-cffi",
        "nodriver",
        "scrapling",
        "pytest",
        "pytest-flask",
        "pytest-cov",
        "mypy",
        "ruff",
    ):
        assert required_name in requirements

    assert "selenium" not in requirements
    assert "scrapy-selenium" not in requirements
    assert str(requirements["flasgger"].specifier) == "==0.9.7.1"
    for requirement in requirements.values():
        specifier = str(requirement.specifier)
        assert "==" in specifier or "<" in specifier


def test_pipeline_and_collection_use_python_module_launcher() -> None:
    """Avoid relocatability-sensitive Scrapy console executables."""

    pipeline_source = PIPELINE_START.read_text(encoding="utf-8")

    assert "SCRAPY_EXE" not in pipeline_source
    assert "PYTHON_EXE = Path(sys.executable).resolve()" in pipeline_source
    assert 'cmd = [str(PYTHON_EXE), "-m", "scrapy", "crawl", spider]' in pipeline_source
    assert collect_hltv_data.scrapy_command("fixture") == [
        collect_hltv_data.sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "fixture",
    ]


def test_installers_consume_only_hashed_lock_and_local_wheels() -> None:
    """Keep Docker and Make off the floating direct-requirements input."""

    dockerfile = (SCRAPER_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (SCRAPER_ROOT / ".dockerignore").read_text(encoding="utf-8")
    makefile = (SCRAPER_ROOT / "Makefile").read_text(encoding="utf-8")

    for content in (dockerfile, makefile):
        assert "-r requirements.lock.txt" in content or '-r "$(LOCKFILE)"' in content
        assert "--require-hashes" in content
        assert "--only-binary=:all:" in content
        assert "--no-deps" in content
        assert "--find-links" in content
    assert "pip install --no-cache-dir -r requirements.txt" not in dockerfile
    assert ".venv/" in dockerignore
    assert "./env/bin/" not in makefile


def test_docs_and_gitignore_preserve_single_wheel_contract() -> None:
    """Document Python 3.13 and allowlist exactly the reviewed local wheel."""

    readme = (SCRAPER_ROOT / "README.md").read_text(encoding="utf-8")
    wheel_readme = (SCRAPER_ROOT / "wheels" / "README.md").read_text(encoding="utf-8")
    gitignore = (SCRAPER_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "CPython 3.13" in readme
    assert "requirements.lock.txt" in readme
    assert "pip-tools==7.5.2" in readme
    exact_compile_command = (
        "py -3.13 -m piptools compile --resolver=backtracking "
        "--generate-hashes --allow-unsafe --strip-extras "
        "--no-emit-index-url --find-links ./wheels --no-emit-find-links "
        "--pip-args '--only-binary=:all:' "
        "--output-file requirements.lock.txt requirements.txt"
    )
    assert exact_compile_command in readme
    assert EXPECTED_SHA256 in wheel_readme
    assert "ca098e10bfbb12f047acc6299cc70a33851943a746e550d86e65e60d4df245fb" in wheel_readme
    for build_contract in (
        "CPython 3.13.15",
        "pip 25.2",
        "setuptools 84.0.0",
        "wheel 0.47.0",
        "SOURCE_DATE_EPOCH=1684430121",
        "--use-pep517 --no-build-isolation --no-deps --no-cache-dir",
        "byte-identical",
    ):
        assert build_contract in wheel_readme
    assert "wheels/*" in gitignore
    assert f"!wheels/{WHEEL_FILENAME}" in gitignore
    assert "!wheels/README.md" in gitignore

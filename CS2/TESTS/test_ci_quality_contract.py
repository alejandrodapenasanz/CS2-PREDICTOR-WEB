"""Protect the three-component CI quality and startup contract."""

from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"


def _run_commands(job: dict[str, object]) -> list[str]:
    """Return all shell commands declared by one workflow job."""

    steps = job["steps"]
    assert isinstance(steps, list)
    return [str(step["run"]) for step in steps if isinstance(step, dict) and "run" in step]


def test_ci_covers_direct_startup_locked_install_and_every_component_gate() -> None:
    """Keep requirements boot validation ahead of reproducible locked gates."""

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    expected_commands = {
        "model-tests": "python scripts/quality_gate.py",
        "scraper-tests": "make PYTHON=python gates",
        "tennis-tests": "python scripts/run_quality_gates.py",
    }

    assert set(jobs) == set(expected_commands)
    for job_name, gate_command in expected_commands.items():
        job = jobs[job_name]
        setup_python = next(step for step in job["steps"] if step.get("uses") == "actions/setup-python@v5")
        assert setup_python["with"]["python-version"] == "3.13"

        commands = _run_commands(job)
        direct_index = next(index for index, command in enumerate(commands) if "-r requirements.txt" in command)
        lock_index = next(
            index
            for index, command in enumerate(commands)
            if "--require-hashes" in command and "requirements.lock.txt" in command
        )
        assert direct_index < lock_index
        assert "pip check" in commands[direct_index]
        assert gate_command in commands

    scraper_commands = "\n".join(_run_commands(jobs["scraper-tests"]))
    assert "--source CS2/PIPELINE/start.py" in scraper_commands
    assert "TESTS/test_hltv_parsers.py" in scraper_commands
    assert "TESTS/test_bbdd_live_pipeline.py" in scraper_commands

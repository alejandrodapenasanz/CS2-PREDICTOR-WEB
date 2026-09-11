#!/usr/bin/env python3
"""Run the local CS2 quality gates with the production CPython interpreter.

The gate covers the maintained lint surface, an incremental format baseline,
configured mypy checks, import coverage, offline tests, real CLI import/boot and
the deterministic seed-42 model smoke. Run from any directory with:

    python CS2/scripts/quality_gate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


CS2_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CS2_ROOT.parent
FORMAT_BASELINE = (
    "MODEL/cs2model/pistol_opponents.py",
    "TESTS/test_pistol_opponents.py",
    "BBDD/round_history_store.py",
    "PIPELINE/round_history.py",
    "MODEL/cs2model/pistols.py",
    "MODEL/run_pistol_ablation.py",
    "TESTS/test_pistol_rounds.py",
    "scripts/quality_gate.py",
    "MODEL/cs2model/config.py",
    "MODEL/cs2model/artifacts.py",
    "MODEL/cs2model/cleanup_pending.py",
    "MODEL/cs2model/odds.py",
    "MODEL/cs2model/economic.py",
    "MODEL/cs2model/drift.py",
    "MODEL/cs2model/enhanced_info.py",
    "MODEL/cs2model/promotion.py",
    "MODEL/cs2model/retention.py",
    "MODEL/cs2model/retrain_policy.py",
    "MODEL/cs2model/reproducibility.py",
    "MODEL/cs2model/roster_rating.py",
    "MODEL/cs2model/segment_calibration.py",
    "MODEL/cs2model/uncertainty.py",
    "MODEL/check_retrain.py",
    "MODEL/audit_promotion_consistency.py",
    "MODEL/manage_models.py",
    "MODEL/monitor_drift.py",
    "MODEL/run_enhanced_info_ablation.py",
    "MODEL/run_segment_interaction_ablation.py",
    "MODEL/evaluate_live_ledger.py",
    "MODEL/measure_segment_calibration.py",
    "MODEL/smoke_pipeline.py",
    "TESTS/test_ci_quality_contract.py",
    "TESTS/test_dependency_provisioning.py",
    "TESTS/test_data_freshness.py",
    "TESTS/test_estimate_uncertainty.py",
    "TESTS/test_enhanced_info_ablation.py",
    "TESTS/test_import_coverage_gate.py",
    "TESTS/test_hltv_cf_refresh.py",
    "TESTS/test_inference_determinism.py",
    "TESTS/test_model_management_cli.py",
    "TESTS/test_model_promotion.py",
    "TESTS/test_odds_router.py",
    "TESTS/test_retention.py",
    "TESTS/test_retrain_policy.py",
    "TESTS/test_roster_sensitive_rating.py",
    "TESTS/test_segment_calibration.py",
    "TESTS/test_training_promotion_wiring.py",
    "TESTS/test_web_uncertainty_display.py",
)


def run(command: list[str], *, cwd: Path = CS2_ROOT) -> None:
    """Run one gate and fail immediately with its native exit status."""

    print(f"[gate] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    """Execute every CS2 gate in deterministic order."""

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("CS2 quality gates require CPython 3.13.")
    python = sys.executable
    run([python, "-m", "ruff", "check", "MODEL", "BBDD", "PIPELINE", "TESTS", "scripts"])
    run([python, "-m", "ruff", "format", "--check", *FORMAT_BASELINE])
    run([python, "-m", "mypy"])
    run(
        [
            python,
            str(REPOSITORY_ROOT / "scripts" / "check_import_coverage.py"),
            "--requirements",
            "CS2/requirements.txt",
            "--source",
            "CS2/MODEL",
            "--source",
            "CS2/BBDD",
            "--source",
            "CS2/PIPELINE",
            "--source",
            "CS2/TESTS",
            "--exclude",
            "CS2/PIPELINE/start.py",
            "--allow-import",
            "BBDD",
            "--allow-import",
            "MODEL",
            "--allow-import",
            "PIPELINE",
        ],
        cwd=REPOSITORY_ROOT,
    )
    for entrypoint in (
        "BBDD/round_history_store.py",
        "MODEL/run_pistol_ablation.py",
        "MODEL/train.py",
        "MODEL/audit_promotion_consistency.py",
        "MODEL/check_retrain.py",
        "MODEL/manage_models.py",
        "MODEL/evaluate_live_ledger.py",
        "MODEL/measure_segment_calibration.py",
        "MODEL/run_enhanced_info_ablation.py",
        "MODEL/run_segment_interaction_ablation.py",
        "BBDD/build_db.py",
        "PIPELINE/enrich_predictions.py",
    ):
        run([python, entrypoint, "--help"])
    with TemporaryDirectory(
        prefix=".pytest-quality-gate-",
        dir=CS2_ROOT,
        ignore_cleanup_errors=True,
    ) as temporary:
        run(
            [
                python,
                "-m",
                "pytest",
                "TESTS/",
                "-q",
                "--ignore=TESTS/test_hltv_parsers.py",
                "--ignore=TESTS/test_bbdd_live_pipeline.py",
                f"--basetemp={Path(temporary) / 'pytest'}",
            ]
        )
    run([python, "MODEL/smoke_pipeline.py"])
    print("cs2_quality_gate=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from MODEL import check_retrain
from MODEL.cs2model.retrain_policy import decide_retrain


CS2_ROOT = Path(__file__).resolve().parents[1]


def artifact(date_max: str | None) -> SimpleNamespace:
    metadata = {} if date_max is None else {"date_max": date_max}
    return SimpleNamespace(metadata=metadata)


def register_attempt(
    registry: Path,
    *,
    version: str,
    date_max: str,
    status: str | None = "reject",
) -> Path:
    version_dir = registry / version
    version_dir.mkdir(parents=True)
    model_path = version_dir / "model.pkl"
    model_path.write_bytes(b"registered-candidate")
    registry_metadata: dict[str, object] = {"promoted": False}
    if status is not None:
        registry_metadata["status"] = status
    payload = {
        "registered_at": f"{version[:4]}-{version[4:6]}-{version[6:8]}T12:00:00Z",
        "artifact": str(model_path),
        "feature_columns": ["elo_diff"],
        "metadata": {"date_max": date_max, "registry": registry_metadata},
        "metrics": {},
    }
    (version_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    (version_dir / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    return version_dir


def test_counts_only_labeled_matches_strictly_after_daily_cutoff() -> None:
    rows = [
        {"id": "before", "date": "2026-01-09", "team1_win": 1},
        {"id": "same-day", "date": "2026-01-10", "team1_win": 0},
        {"id": "new-1", "date": "2026-01-11", "team1_win": 1},
        {"id": "new-2", "date": "2026-01-12T18:00:00Z", "label": 0},
        {"id": "unlabeled", "date": "2026-01-13", "team1_win": None},
        {"id": "invalid-date", "date": "not-a-date", "team1_win": 1},
    ]

    decision = decide_retrain(rows, artifact("2026-01-10"), threshold=2)

    assert decision.should_train is True
    assert decision.n_new_labeled == 2
    assert decision.live_cutoff == "2026-01-10"
    assert decision.attempt_cutoff == "2026-01-10"
    assert decision.cutoff == "2026-01-10"
    assert decision.newest_date == "2026-01-12"
    assert decision.reason == "new_labeled_threshold_reached"


def test_below_threshold_does_not_retrain() -> None:
    decision = decide_retrain(
        [{"date": "2026-01-11", "team1_win": 1}],
        artifact("2026-01-10"),
        threshold=2,
    )

    assert decision.should_train is False
    assert decision.n_new_labeled == 1
    assert decision.reason == "new_labeled_threshold_not_reached"


def test_missing_artifact_trains_even_without_rows() -> None:
    decision = decide_retrain([], None)

    assert decision.should_train is True
    assert decision.n_new_labeled == 0
    assert decision.threshold == 100
    assert decision.live_cutoff is None
    assert decision.attempt_cutoff is None
    assert decision.cutoff is None
    assert decision.newest_date is None
    assert decision.reason == "production_artifact_missing"


@pytest.mark.parametrize("cutoff", [None, "not-a-date"])
def test_artifact_without_valid_cutoff_fails_closed(cutoff: str | None) -> None:
    decision = decide_retrain(
        [{"date": "2026-01-20", "team1_win": 1}],
        artifact(cutoff),
        threshold=1,
    )

    assert decision.should_train is False
    assert decision.n_new_labeled == 0
    assert decision.cutoff is None
    assert decision.attempt_cutoff is None
    assert "fail_closed" in decision.reason


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        decide_retrain([], None, threshold=0)


def test_registered_rejection_prevents_repeat_until_threshold_more_labels(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    register_attempt(
        registry,
        version="20260112_120000Z",
        date_max="2026-01-12",
        status="reject",
    )
    attempted_rows = [
        {"id": "new-1", "date": "2026-01-11", "team1_win": 1},
        {"id": "new-2", "date": "2026-01-12", "team1_win": 0},
    ]

    repeated = decide_retrain(
        attempted_rows,
        artifact("2026-01-10"),
        threshold=2,
        registry=registry,
    )
    after_more_labels = decide_retrain(
        [
            *attempted_rows,
            {"id": "later-1", "date": "2026-01-13", "team1_win": 1},
            {"id": "later-2", "date": "2026-01-14", "team1_win": 0},
        ],
        artifact("2026-01-10"),
        threshold=2,
        registry=registry,
    )

    assert repeated.should_train is False
    assert repeated.n_new_labeled == 0
    assert repeated.live_cutoff == "2026-01-10"
    assert repeated.attempt_cutoff == "2026-01-12"
    assert repeated.cutoff == "2026-01-12"
    assert after_more_labels.should_train is True
    assert after_more_labels.n_new_labeled == 2
    assert after_more_labels.live_cutoff == "2026-01-10"
    assert after_more_labels.attempt_cutoff == "2026-01-12"


def test_registry_accepts_full_historical_structure_without_status(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    register_attempt(
        registry,
        version="20260112_120000Z",
        date_max="2026-01-12",
        status=None,
    )

    decision = decide_retrain(
        [
            {"date": "2026-01-11", "team1_win": 1},
            {"date": "2026-01-12", "team1_win": 0},
        ],
        artifact("2026-01-10"),
        threshold=2,
        registry=registry,
    )

    assert decision.should_train is False
    assert decision.attempt_cutoff == "2026-01-12"


def test_registry_ignores_unknown_malformed_and_future_candidates(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    register_attempt(
        registry,
        version="20260120_120000Z",
        date_max="2099-01-01",
        status="reject",
    )
    register_attempt(
        registry,
        version="20260112_130000Z",
        date_max="2026-01-12",
        status="invented_status",
    )
    register_attempt(
        registry,
        version="20260112_131000Z",
        date_max="2026-01-12",
        status="pending_health_gates",
    )
    malformed = registry / "20260112_140000Z"
    malformed.mkdir()
    (malformed / "metadata.json").write_text("{not-json", encoding="utf-8")
    register_attempt(
        registry,
        version="not-a-canonical-version",
        date_max="2026-01-12",
        status="reject",
    )

    decision = decide_retrain(
        [
            {"date": "2026-01-11", "team1_win": 1},
            {"date": "2026-01-12", "team1_win": 0},
        ],
        artifact("2026-01-10"),
        threshold=2,
        registry=registry,
    )

    assert decision.should_train is True
    assert decision.n_new_labeled == 2
    assert decision.attempt_cutoff == "2026-01-10"


def test_cli_outputs_deterministic_json_and_optional_file(tmp_path: Path) -> None:
    rows = [
        {"date": "2026-02-01", "team1_win": 1},
        {"date": "2026-02-02", "team1_win": 0},
    ]
    output_path = tmp_path / "decision.json"
    registry = tmp_path / "registry"
    registry.mkdir()
    stdout = io.StringIO()
    with (
        patch.object(check_retrain, "_load_rows", return_value=rows),
        patch.object(check_retrain, "load_artifact", return_value=artifact("2026-01-31")),
        redirect_stdout(stdout),
    ):
        exit_code = check_retrain.main(
            [
                "--threshold",
                "2",
                "--registry",
                str(registry),
                "--output",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == stdout.getvalue()
    assert json.loads(stdout.getvalue()) == {
        "cutoff": "2026-01-31",
        "attempt_cutoff": "2026-01-31",
        "live_cutoff": "2026-01-31",
        "n_new_labeled": 2,
        "newest_date": "2026-02-02",
        "reason": "new_labeled_threshold_reached",
        "should_train": True,
        "threshold": 2,
    }


def test_cli_help_exposes_read_only_sources_and_threshold() -> None:
    completed = subprocess.run(
        [sys.executable, str(CS2_ROOT / "MODEL" / "check_retrain.py"), "--help"],
        cwd=CS2_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for option in (
        "--config",
        "--artifact",
        "--registry",
        "--db",
        "--master",
        "--raw",
        "--threshold",
        "--output",
    ):
        assert option in completed.stdout


@pytest.mark.parametrize("operation", ["metadata_stat", "metadata_read", "model_stat", "directory_stat"])
def test_inaccessible_rejected_residue_is_reported_and_skipped(tmp_path: Path, monkeypatch, operation: str) -> None:
    registry = tmp_path / "registry"
    register_attempt(registry, version="20260112_120000Z", date_max="2026-01-12")
    blocked = register_attempt(registry, version="20260114_120000Z", date_max="2026-01-14")
    protected = {name: b"protected pointer" for name in ("latest.json", "last_good.json")}
    for name, content in protected.items():
        (registry / name).write_bytes(content)
    original_stat, original_read = Path.stat, Path.read_text
    stat_target = blocked / ("model.pkl" if operation == "model_stat" else "metadata.json")
    if operation == "directory_stat":
        stat_target = blocked

    def denied_stat(path, *args, **kwargs):
        if operation != "metadata_read" and path == stat_target:
            raise PermissionError(5, "Access denied", str(path))
        return original_stat(path, *args, **kwargs)

    def denied_read(path, *args, **kwargs):
        if operation == "metadata_read" and path == blocked / "metadata.json":
            raise PermissionError(5, "Access denied", str(path))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    monkeypatch.setattr(Path, "read_text", denied_read)
    decision = decide_retrain(
        [{"date": "2026-01-13", "team1_win": 1}, {"date": "2026-01-14", "team1_win": 0}],
        artifact("2026-01-10"),
        threshold=2,
        registry=registry,
    )
    assert decision.should_train
    assert decision.attempt_cutoff == "2026-01-12"
    assert decision.n_new_labeled == 2
    assert len(decision.registry_warnings) == 1
    assert "20260114_120000Z" in decision.registry_warnings[0]
    assert "PermissionError" in decision.as_dict()["registry_warnings"][0]
    for name, content in protected.items():
        assert (registry / name).read_bytes() == content


def test_inaccessible_registry_root_does_not_crash_policy(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    original = Path.iterdir

    def denied(path):
        if path == registry:
            raise PermissionError("registry unavailable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    decision = decide_retrain(
        [{"date": "2026-01-11", "team1_win": 1}], artifact("2026-01-10"), threshold=2, registry=registry
    )
    assert not decision.should_train
    assert decision.attempt_cutoff == "2026-01-10"
    assert "registro no accesible" in decision.registry_warnings[0]


def test_start_pipeline_wires_auto_retrain_after_data_gates_into_same_trainer() -> None:
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")

    assert '$CheckRetrain = Join-Path $Root "MODEL\\check_retrain.py"' in start
    assert '$ModelConfig = Join-Path $Root "MODEL\\config.yaml"' in start
    assert '"--db", $DbPath' in start
    assert '"--artifact", $Artifact' in start
    assert '"--config", $ModelConfig' in start
    assert '"--output", $RetrainDecisionPath' in start
    assert '"retrain_decision_" + $stamp + ".json"' in start
    assert "[bool]$RetrainDecision.should_train" in start
    assert "foreach ($RegistryWarning in @($RetrainDecision.registry_warnings))" in start
    assert "Write-Log ([string]$RegistryWarning) -Level WARN" in start

    decision_position = start.index("# --- Decision de reentreno automatico")
    assert decision_position > start.index('"Health gate datos"')
    assert decision_position < start.index("# --- Etapa 4: entrenamiento")
    assert start.count('Invoke-Native $ModelPython $TrainArgs "Entrenamiento modelo"') == 1


def test_start_manual_and_automatic_retrain_share_the_snapshotting_final_ingest() -> None:
    """Both retrain decisions converge on the same backup-enabled DB write."""

    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")
    final_ingest = 'Invoke-Native $ModelPython @($IngestDb, "--run-dir", $RunDir) "Ingest incremental BBDD"'
    final_stage = start[
        start.index("# --- Etapa 7: ingest final") : start.index("# --- Fase P: BLACKBOX export de respaldo")
    ]

    assert start.count(final_ingest) == 1
    assert start.index("# --- Decision de reentreno automatico") < start.index("# --- Etapa 4: entrenamiento")
    assert start.index("# --- Etapa 4: entrenamiento") < start.index("# --- Etapa 7: ingest final")
    assert start.index(final_ingest) > start.index("# --- Etapa 7: ingest final")
    assert final_ingest in final_stage
    assert '"--no-backup"' not in final_stage

    config = json.loads((CS2_ROOT / "BBDD" / "backup_retention.json").read_text(encoding="utf-8"))
    assert config["default_keep"] == 1


def test_start_auto_retrain_is_db_only_and_dry_run_does_not_read_decision() -> None:
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")
    decision_block = start[
        start.index("# --- Decision de reentreno automatico") : start.index("# --- Etapa 4: entrenamiento")
    ]

    assert "if ((-not $needTrain) -and (-not $NoDb))" in decision_block
    assert "if ($script:DryRun)" in decision_block
    assert "no se escribe ni se lee decision" in decision_block
    assert decision_block.index("if ($script:DryRun)") < decision_block.index("Get-Content -LiteralPath")
    assert "$script:StageTotal += 1" in decision_block
    for field in ("n_new", "threshold", "cutoff", "reason"):
        assert field in decision_block


def test_start_rollback_is_exclusive_and_exits_before_scraper_or_database() -> None:
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")

    assert "[switch]$RollbackModel" in start
    assert '$ManageModels = Join-Path $Root "MODEL\\manage_models.py"' in start
    assert 'Invoke-Native $ModelPython @($ManageModels, "rollback")' in start
    assert "$script:ExitCode = $Cfg.ExitCodes.Train" in start
    assert "'Retrain'" in start
    assert "'BackupBlackbox'" in start
    assert "'RestoreBlackbox'" in start

    rollback_position = start.index("# --- Gestion exclusiva del modelo")
    assert rollback_position < start.index("# --- Fase 0: Python del modelo y del scraper")
    rollback_block = start[rollback_position : start.index("# --- Fase 0: Python del modelo y del scraper")]
    assert "Ensure-ModelPython -ProjectRoot $Root" in rollback_block
    assert "Ensure-ScraperPython" not in rollback_block
    assert "$IngestDb" not in rollback_block
    assert "exit 0" in rollback_block
    assert "DRY-RUN: se ejecutaria:" in rollback_block


def test_start_rollback_dry_run_and_conflict_are_safe() -> None:
    if sys.platform != "win32":
        pytest.skip("The operational CS2 PowerShell launcher is Windows-only")
    shell = "powershell.exe"
    launcher = CS2_ROOT / "start.ps1"

    preview = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-RollbackModel",
            "-DryRun",
        ],
        cwd=CS2_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert preview.returncode == 0, preview.stderr or preview.stdout
    preview_output = preview.stdout + preview.stderr
    assert "manage_models.py rollback" in preview_output
    assert "retencion automatica" not in preview_output.lower()
    assert "Auto-heal" not in preview_output
    assert "Inicializando/sembrando BBDD" not in preview_output
    assert "Scrape online" not in preview_output

    conflict = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-RollbackModel",
            "-Retrain",
            "-DryRun",
        ],
        cwd=CS2_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert conflict.returncode != 0
    assert "modo exclusivo" in (conflict.stdout + conflict.stderr)


def test_start_retention_runs_before_web_and_publishes_cleanup_warning() -> None:
    start = (CS2_ROOT / "start.ps1").read_text(encoding="utf-8")

    retention_position = start.index("# --- Retencion conservadora")
    assert retention_position < start.index('Invoke-Stage -Name "Generando WEB\\data.js compartido"')
    assert retention_position < start.index("# --- Resumen + tabla de tiempos")
    retention_block = start[retention_position : start.index("# --- Resumen + tabla de tiempos")]
    assert '"retention_" + $stamp + ".json"' in retention_block
    assert '"prune"' in retention_block
    assert '"--scope", "all"' in retention_block
    assert '"--auto"' in retention_block
    assert '"--output", $RetentionReportPath' in retention_block
    assert "$script:ExitCode = $Cfg.ExitCodes.Train" in retention_block
    assert "ConvertFrom-Json" in retention_block
    assert "preview/no-op o aplicada segun marker" in retention_block
    assert "LIMPIEZA PENDIENTE" in retention_block
    assert "latest y last_good permanecen protegidos" in retention_block


def test_start_retention_normal_dry_run_does_not_create_report() -> None:
    if sys.platform != "win32":
        pytest.skip("The operational CS2 PowerShell launcher is Windows-only")
    reports_before = {path.name for path in (CS2_ROOT / "PIPELINE" / "logs").glob("retention_*.json")}
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CS2_ROOT / "start.ps1"),
            "-DryRun",
            "-SkipScrape",
        ],
        cwd=CS2_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    output = completed.stdout + completed.stderr
    assert "se evaluaria retencion automatica" in output
    assert "no se escribe ni se aplica el plan" in output
    reports_after = {path.name for path in (CS2_ROOT / "PIPELINE" / "logs").glob("retention_*.json")}
    assert reports_after == reports_before

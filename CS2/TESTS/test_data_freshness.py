from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
FRESHNESS_PATH = ROOT / "PIPELINE" / "freshness.py"
WEB_PATH = REPO_ROOT / "WEB" / "index.html"
BUILD_WEB_PATH = REPO_ROOT / "WEB" / "build_web.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cs2_data_freshness_test", FRESHNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _web_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cs2_cleanup_dashboard_test", BUILD_WEB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_old_successful_batch_is_stale_without_breaking() -> None:
    module = _module()
    manifest = {
        "run_id": "old-run",
        "started_at": "2026-08-23T07:00:00Z",
        "finished_at": "2026-08-23T08:00:00Z",
    }
    config = {"sources": {"hltv": {"label": "HLTV", "cadence": "daily", "threshold_hours": 24.0}}}
    with mock.patch.object(
        module,
        "_read_json",
        side_effect=lambda path: manifest if path.name == "manifest.json" else config,
    ):
        report = module.build_hltv_report(
            Path("old-run"),
            observed_at=datetime(2026, 8, 27, 8, tzinfo=UTC),
        )
    source = report["sources"][0]

    assert source["status"] == "stale"
    assert source["cause"] == "source_not_refreshed"
    assert source["age_hours"] == 96.0
    assert report["has_warning"] is True


def test_failed_attempt_is_distinct_from_a_pipeline_not_run() -> None:
    module = _module()
    source_status = module._status(
        observed_at=datetime(2026, 8, 27, 8, tzinfo=UTC),
        last_success=datetime(2026, 8, 25, 8, tzinfo=UTC),
        threshold_hours=24.0,
        attempt_status="failed",
        attempted_at=datetime(2026, 8, 27, 7, tzinfo=UTC),
        fallback_used=True,
    )
    assert source_status[0] == "stale"
    assert source_status[1] == "source_update_failed"


def test_dashboard_recomputes_age_and_marks_predictions_dynamically() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    assert "nowMs-lastSuccessMs" in html
    assert 'cause = "pipeline_not_run"' in html
    assert 'code:"DATA_STALE"' in html
    assert "value *= .60" in html
    assert "PENDING_CLEANUP" in html
    assert "LIMPIEZA DE ARTEFACTOS" in html


def test_dashboard_payload_exposes_pending_cleanup_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _web_module()
    model_root = tmp_path / "MODEL"
    state = model_root / "artifacts" / "pending_cleanup.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        """{
  "schema_version": 1,
  "updated_at_utc": "2026-08-27T08:00:00Z",
  "items": [{"name": "20260824_125852Z"}]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "MODEL_ROOT", model_root)

    payload = module.pending_cleanup_payload()

    assert payload["status"] == "warning"
    assert payload["count"] == 1
    assert payload["items"][0]["name"] == "20260824_125852Z"

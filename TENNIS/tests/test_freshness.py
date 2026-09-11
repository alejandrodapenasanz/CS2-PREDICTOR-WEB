"""Pruebas del contrato lateral de frescura de TENNIS."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from src import freshness


class FreshnessContractTests(unittest.TestCase):
    def test_old_sources_distinguish_pipeline_not_run_from_update_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ratio = root / "ratio.json"
            ratio.write_text(
                json.dumps({"retrieved_at_utc": "2026-08-24T08:00:00Z"}),
                encoding="utf-8",
            )
            explorer = root / "explorer"
            explorer.mkdir()
            (explorer / "old.metadata.json").write_text(
                json.dumps(
                    {
                        "retrieved_at_utc": "2026-08-24T09:00:00Z",
                        "status_code": 200,
                    }
                ),
                encoding="utf-8",
            )
            feature_manifest = root / "features.json"
            feature_manifest.write_text(
                json.dumps(
                    {
                        "datasets": [{"max_date": "2026-07-01"}],
                        "rankings": [{"max_date": "2026-07-05"}],
                    }
                ),
                encoding="utf-8",
            )
            sackmann = root / "sackmann.json"
            sackmann.write_text(
                json.dumps({"generated_at_utc": "2026-08-24T07:00:00Z"}),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "sources": {
                            "tennisratio": {"label": "TennisRatio", "threshold_hours": 24},
                            "tennis_explorer": {"label": "Tennis Explorer", "threshold_hours": 24},
                            "sackmann": {"label": "Sackmann", "threshold_hours": 1080},
                        }
                    }
                ),
                encoding="utf-8",
            )
            now = datetime(2026, 8, 27, 8, tzinfo=UTC)
            with (
                mock.patch.object(freshness, "TENNISRATIO_LAST_GOOD_PATH", ratio),
                mock.patch.object(freshness, "TENNIS_EXPLORER_DAILY_RAW_DIR", explorer),
                mock.patch.object(freshness, "FEATURE_DATASET_MANIFEST_PATH", feature_manifest),
                mock.patch.object(
                    freshness,
                    "SACKMANN_ACTIVE_MANIFEST_PATH",
                    sackmann,
                ),
                mock.patch.object(freshness, "CONFIG_PATH", config),
            ):
                not_run = freshness.build_freshness_report(
                    observed_at=now,
                    pipeline_last_run_at=datetime(2026, 8, 24, 10, tzinfo=UTC),
                )
                failed = freshness.build_freshness_report(
                    observed_at=now,
                    pipeline_last_run_at=now,
                    attempts={
                        "tennisratio": {
                            "status": "failed",
                            "attempted_at_utc": "2026-08-27T07:30:00Z",
                        }
                    },
                )

            not_run_by_id = {row["source_id"]: row for row in not_run["sources"]}
            failed_by_id = {row["source_id"]: row for row in failed["sources"]}
            self.assertEqual(not_run_by_id["tennisratio"]["cause"], "pipeline_not_run")
            self.assertEqual(failed_by_id["tennisratio"]["cause"], "source_update_failed")
            self.assertEqual(not_run_by_id["sackmann"]["timestamp_kind"], "latest_source_data_date")
            self.assertEqual(not_run_by_id["sackmann"]["last_batch_at_utc"], "2026-08-24T07:00:00Z")

    def test_dashboard_contract_forces_low_confidence_on_stale_or_fallback(self) -> None:
        html = (Path(__file__).resolve().parents[2] / "WEB" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("data_stale_${sourceFamily}", html)
        self.assertIn("agenda_fallback_tennis_explorer", html)
        self.assertIn('return { label: "BAJA", flags }', html)


if __name__ == "__main__":
    unittest.main()

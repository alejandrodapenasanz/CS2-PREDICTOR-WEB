"""Pruebas de keep-list, confirmación y residuos no fatales de modelos."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
for candidate in (PROJECT_ROOT, TESTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.modeling.promotion import active_and_last_good, gate_challenger  # noqa: E402
from src.modeling.retention import (  # noqa: E402
    apply_retention_plan,
    confirm_first_retention,
    plan_model_retention,
    run_automatic_retention,
)
from test_model_promotion import _register_run, _write_config  # noqa: E402


class ModelRetentionTest(unittest.TestCase):
    """Protege punteros y convierte fallos de Windows en avisos."""

    def _store_with_four_runs(self, root: Path) -> tuple[Path, str]:
        """Crea champion más tres challengers sin activarlos."""

        output = root / "models"
        config = _write_config(root / "promotion.json")
        champion = _register_run(
            output,
            "A",
            [0.70, 0.30, 0.70, 0.30],
            created_at="2026-01-01T00:00:00+00:00",
        )
        gate_challenger(champion, output_dir=output, config_path=config)
        for label, month in (("B", 2), ("C", 3), ("D", 4)):
            _register_run(
                output,
                label,
                [0.70, 0.30, 0.70, 0.30],
                created_at=f"2026-{month:02d}-01T00:00:00+00:00",
            )
        return output, champion.fingerprint

    def test_plan_keeps_latest_n_plus_active_and_last_good(self) -> None:
        """La keep-list conserva los dos punteros aunque sean antiguos."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output, champion = self._store_with_four_runs(Path(temporary))

            plan = plan_model_retention(output)

            self.assertEqual(plan.keep_latest, 2)
            self.assertEqual(plan.active_fingerprint, champion)
            self.assertEqual(plan.last_good_fingerprint, champion)
            self.assertIn(champion, plan.keep_fingerprints)
            self.assertEqual(len(plan.keep_fingerprints), 3)
            self.assertEqual(len(plan.delete_fingerprints), 1)

    def test_first_run_only_previews_until_exact_confirmation(self) -> None:
        """Sin marcador aprobado no se elimina ninguna generación."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output, _champion = self._store_with_four_runs(Path(temporary))
            before = {path.name for path in (output / "runs").iterdir()}

            preview = run_automatic_retention(output_dir=output)

            self.assertEqual(preview.mode, "preview")
            self.assertFalse(preview.applied)
            self.assertEqual({path.name for path in (output / "runs").iterdir()}, before)
            applied = confirm_first_retention(preview.plan.token, output_dir=output)
            self.assertTrue(applied.applied)
            self.assertEqual(set(applied.deleted), set(preview.plan.delete_fingerprints))

    def test_delete_failure_is_pending_and_pipeline_stays_green(self) -> None:
        """Un handle bloqueado no toca champion/last_good ni lanza error."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output, champion = self._store_with_four_runs(Path(temporary))
            plan = plan_model_retention(output)
            pending_path = output / "cleanup_pending.json"

            with patch(
                "src.modeling.retention.shutil.rmtree",
                side_effect=PermissionError("archivo bloqueado"),
            ):
                result = apply_retention_plan(
                    plan,
                    confirmation_token=plan.token,
                    pending_path=pending_path,
                )

            self.assertTrue(result.applied)
            self.assertEqual(result.deleted, ())
            self.assertEqual(len(result.pending), 1)
            active, last_good = active_and_last_good(output)
            self.assertEqual(active.fingerprint, champion)
            self.assertEqual(last_good.fingerprint, champion)
            payload = json.loads(pending_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["pending"]), 1)


if __name__ == "__main__":
    unittest.main()

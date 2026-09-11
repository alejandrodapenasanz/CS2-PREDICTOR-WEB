"""Pruebas de puerta común, consistencia, rechazo y rollback de TENNIS."""

from __future__ import annotations

from datetime import date, timedelta
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.artifacts import (  # noqa: E402
    PublishedRun,
    create_staging_directory,
    fingerprint_payload,
    publish_staged_run,
)
from src.modeling.promotion import (  # noqa: E402
    PromotionError,
    gate_challenger,
    has_equivalent_feature_gate,
    rollback_last_good,
)
from src.modeling import artifacts as artifacts_module  # noqa: E402
from src.modeling import training as training_module  # noqa: E402


def _write_config(path: Path) -> Path:
    """Crea márgenes pequeños y un soporte mínimo apto para fixtures."""

    path.write_text(
        json.dumps(
            {
                "log_loss_epsilon": 0.001,
                "brier_epsilon": 0.0005,
                "metric_consistency_atol": 1e-12,
                "minimum_holdout_rows": 4,
                "holdout_seasons": 1,
                "retention_keep_latest": 2,
            }
        ),
        encoding="utf-8",
    )
    return path


def _oof(gender: str, probabilities: list[float], *, suffix: str = "") -> pd.DataFrame:
    """Construye cuatro predicciones OOF de 2025 con identidad estable."""

    labels = [1, 0, 1, 0]
    rows: list[dict[str, object]] = []
    for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
        match_date = date(2025, 1, 1) + timedelta(days=index)
        rows.append(
            {
                "record_id": f"{gender}-2025-{index}{suffix}",
                "gender": gender,
                "match_date": match_date,
                "result_available_date": match_date + timedelta(days=21),
                "tour_level": "ATP Tour" if gender == "M" else "WTA Tour",
                "surface": "Hard",
                "y": label,
                "test_season": 2025,
                "lightgbm_platt": probability,
            }
        )
    return pd.DataFrame(rows)


def _register_run(
    output: Path,
    label: str,
    probabilities: list[float],
    *,
    suffix: str = "",
    created_at: str,
    feature_fingerprint: str | None = None,
) -> PublishedRun:
    """Registra un run mínimo con OOF de ambos géneros."""

    staging = create_staging_directory(output)
    evaluation = staging / "evaluation"
    evaluation.mkdir(parents=True)
    for gender in ("M", "F"):
        _oof(gender, probabilities, suffix=suffix).to_parquet(
            evaluation / f"oof_{gender}.parquet", index=False
        )
    (staging / "model.txt").write_text(label, encoding="utf-8")
    fingerprint = fingerprint_payload(
        {"label": label, "probabilities": probabilities, "suffix": suffix}
    )
    return publish_staged_run(
        staging,
        fingerprint=fingerprint,
        manifest_payload={
            "artifact_version": "test-v1",
            "created_at_utc": created_at,
            "feature_fingerprint": feature_fingerprint,
            "identity": {
                "parameters": {
                    "temporal_evaluation": {
                        "first_test_season": 2025,
                        "last_test_season": 2025,
                    },
                    "calibration": {
                        "method": "Platt",
                        "fold_fit_block": "Y-1",
                        "fold_test_block": "Y",
                        "final_fit_source": "temporal OOF test predictions",
                    },
                }
            },
        },
        output_dir=output,
    )


class ModelPromotionTest(unittest.TestCase):
    """Valida que producción solo cambia por una decisión favorable."""

    def test_rejected_exact_tie_attests_feature_compatibility(self) -> None:
        """Anti-churn rejects the run while its exact gate evidence is usable."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            config = _write_config(root / "promotion.json")
            probabilities = [0.8, 0.2, 0.7, 0.3]
            champion = _register_run(
                root,
                "champion",
                probabilities,
                created_at="2026-01-01T00:00:00+00:00",
                feature_fingerprint="old-features",
            )
            gate_challenger(champion, output_dir=root, config_path=config)
            challenger = _register_run(
                root,
                "challenger",
                probabilities,
                created_at="2026-01-02T00:00:00+00:00",
                feature_fingerprint="new-features",
            )
            result = gate_challenger(
                challenger,
                output_dir=root,
                config_path=config,
            )

            self.assertEqual(result.action, "rejected")
            self.assertFalse(result.changed)
            self.assertTrue(
                has_equivalent_feature_gate(
                    champion_fingerprint=champion.fingerprint,
                    feature_fingerprint="new-features",
                    output_dir=root,
                )
            )

    def test_better_promotes_and_rollback_restores_last_good(self) -> None:
        """Un challenger claramente mejor se activa y puede revertirse."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            output = root / "models"
            config = _write_config(root / "promotion.json")
            champion = _register_run(
                output,
                "champion",
                [0.60, 0.40, 0.60, 0.40],
                created_at="2026-01-01T00:00:00+00:00",
            )
            bootstrap = gate_challenger(champion, output_dir=output, config_path=config)
            challenger = _register_run(
                output,
                "better",
                [0.90, 0.10, 0.90, 0.10],
                created_at="2026-02-01T00:00:00+00:00",
            )

            promoted = gate_challenger(challenger, output_dir=output, config_path=config)
            restored = rollback_last_good(output)

            self.assertEqual(bootstrap.action, "bootstrapped")
            self.assertEqual(promoted.action, "promoted")
            self.assertTrue(promoted.changed)
            self.assertEqual(promoted.active.fingerprint, challenger.fingerprint)
            self.assertEqual(promoted.last_good.fingerprint, champion.fingerprint)
            self.assertIsNotNone(promoted.decision)
            assert promoted.decision is not None
            self.assertEqual(promoted.decision.consistency["status"], "pass")
            self.assertEqual(promoted.decision.n, 8)
            self.assertEqual(promoted.decision.period_start, "2025-01-01")
            self.assertEqual(promoted.decision.period_end, "2025-01-04")
            self.assertLess(promoted.decision.delta_log_loss or 0.0, -0.001)
            self.assertEqual(restored.active.fingerprint, champion.fingerprint)

    def test_worse_is_rejected_without_touching_production(self) -> None:
        """Un challenger peor conserva bytes y fingerprint del champion."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            output = root / "models"
            config = _write_config(root / "promotion.json")
            champion = _register_run(
                output,
                "champion",
                [0.80, 0.20, 0.80, 0.20],
                created_at="2026-01-01T00:00:00+00:00",
            )
            gate_challenger(champion, output_dir=output, config_path=config)
            pointer_before = (output / "manifest.json").read_bytes()
            challenger = _register_run(
                output,
                "worse",
                [0.20, 0.80, 0.20, 0.80],
                created_at="2026-02-01T00:00:00+00:00",
            )

            rejected = gate_challenger(challenger, output_dir=output, config_path=config)

            self.assertEqual(rejected.action, "rejected")
            self.assertFalse(rejected.changed)
            self.assertEqual((output / "manifest.json").read_bytes(), pointer_before)
            self.assertEqual(rejected.active.fingerprint, champion.fingerprint)
            decision_path = output / "promotion" / "decisions" / f"{challenger.fingerprint}.json"
            self.assertTrue(decision_path.is_file())

    def test_mismatched_holdout_is_rejected_before_metrics(self) -> None:
        """La puerta no usa intersección si falta una fila en un artefacto."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            output = root / "models"
            config = _write_config(root / "promotion.json")
            champion = _register_run(
                output,
                "champion",
                [0.70, 0.30, 0.70, 0.30],
                created_at="2026-01-01T00:00:00+00:00",
            )
            gate_challenger(champion, output_dir=output, config_path=config)
            challenger = _register_run(
                output,
                "different-cohort",
                [0.75, 0.25, 0.75, 0.25],
                suffix="-other",
                created_at="2026-02-01T00:00:00+00:00",
            )

            with self.assertRaisesRegex(PromotionError, "exactamente el mismo hold-out"):
                gate_challenger(challenger, output_dir=output, config_path=config)

    def test_training_and_registration_have_no_activation_bypass(self) -> None:
        """Solo promotion.py invoca la primitiva privada de conmutación."""

        training_source = inspect.getsource(training_module)
        registration_source = inspect.getsource(artifacts_module.publish_staged_run)
        self.assertNotIn("_activate_gate_approved_run", training_source)
        self.assertNotIn("_publish_active_manifest", registration_source)
        references: list[str] = []
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            if "_activate_gate_approved_run(" in path.read_text(encoding="utf-8"):
                references.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(
            sorted(references),
            ["src/modeling/artifacts.py", "src/modeling/promotion.py"],
        )


if __name__ == "__main__":
    unittest.main()

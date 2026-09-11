"""Pruebas offline de publicación generacional de datasets de features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.artifacts import (  # noqa: E402
    ACTIVE_POINTER_SCHEMA,
    FeatureArtifactError,
    FeatureRunRetentionPreview,
    activate_feature_run,
    find_verified_feature_run,
    preflight_feature_publication,
    preview_feature_run_retention,
    prune_feature_runs,
    publish_feature_run,
    resolve_active_feature_run,
    verify_feature_run,
)


def _sha256(path: Path) -> str:
    """Calcula el hash de un fixture pequeño."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(label: str) -> str:
    """Deriva un fingerprint de entrada estable para una etiqueta."""

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_staged_run(
    output: Path,
    *,
    fingerprint: str,
    label: str,
    created_at_utc: str = "2026-08-02T00:00:00+00:00",
) -> Path:
    """Crea un workspace mínimo con solo los tres artefactos publicables."""

    workspace = output / f".feature-build-{label}"
    publish = workspace / "publish"
    publish.mkdir(parents=True)
    dataset = publish / "training_M.parquet"
    conflicts = publish / "ranking_conflicts.csv"
    dataset.write_bytes(f"dataset-{label}".encode("utf-8"))
    conflicts.write_text("gender,player_id\n", encoding="utf-8")
    manifest = {
        "schema_version": "tennis-features-v1",
        "created_at_utc": created_at_utc,
        "source_commit": "a" * 40,
        "fingerprint": fingerprint,
        "historical_odds_available": False,
        "model_feature_columns": [],
        "datasets": [
            {
                "gender": "M",
                "output_path": dataset.name,
                "output_size": dataset.stat().st_size,
                "output_sha256": _sha256(dataset),
                "training_rows": 1,
                "min_date": "2024-01-01",
                "max_date": "2024-01-01",
            }
        ],
        "conflict_inventory": {
            "path": conflicts.name,
            "size": conflicts.stat().st_size,
            "sha256": _sha256(conflicts),
            "rows": 0,
        },
    }
    (publish / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return publish


class FeatureArtifactsTest(unittest.TestCase):
    """Comprueba preflight, atomicidad, reactivación e inmutabilidad."""

    def test_publishes_complete_run_and_resolves_active_pointer(self) -> None:
        """Mueve un directorio completo y activa un puntero mínimo al final."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            fingerprint = _fingerprint("first")
            staged = _write_staged_run(
                output,
                fingerprint=fingerprint,
                label="first",
            )

            published = publish_feature_run(
                staged,
                fingerprint=fingerprint,
                output_dir=output,
            )
            active = resolve_active_feature_run(output_dir=output)
            pointer = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(published.skipped)
        self.assertEqual(active.fingerprint, fingerprint)
        self.assertEqual(pointer["pointer_schema_version"], ACTIVE_POINTER_SCHEMA)
        self.assertEqual(pointer["active_run"], f"runs/{fingerprint}")

    def test_reactivates_verified_older_generation_without_rebuilding(self) -> None:
        """Activar A después de B solo cambia el puntero, no los runs."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            fingerprints = []
            for label in ("A", "B"):
                fingerprint = _fingerprint(label)
                fingerprints.append(fingerprint)
                publish_feature_run(
                    _write_staged_run(
                        output,
                        fingerprint=fingerprint,
                        label=label,
                    ),
                    fingerprint=fingerprint,
                    output_dir=output,
                )
            first = find_verified_feature_run(
                fingerprints[0],
                output_dir=output,
            )
            assert first is not None

            activated = activate_feature_run(first, output_dir=output)
            active = resolve_active_feature_run(output_dir=output)

        self.assertTrue(activated.skipped)
        self.assertEqual(active.fingerprint, fingerprints[0])

    def test_automatic_retention_keeps_active_and_latest_previous(self) -> None:
        """El tercer run válido poda el más antiguo tras activar el nuevo."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            published = []
            for day, label in enumerate(("A", "B", "C"), start=1):
                fingerprint = _fingerprint(label)
                published.append(
                    publish_feature_run(
                        _write_staged_run(
                            output,
                            fingerprint=fingerprint,
                            label=label,
                            created_at_utc=(f"2026-08-{day:02d}T00:00:00+00:00"),
                        ),
                        fingerprint=fingerprint,
                        output_dir=output,
                    )
                )

            active = resolve_active_feature_run(output_dir=output)
            preview = preview_feature_run_retention(output_dir=output)

        self.assertIsInstance(preview, FeatureRunRetentionPreview)
        self.assertEqual(active.fingerprint, published[2].fingerprint)
        self.assertEqual(
            {path.name for path in preview.keep_runs},
            {published[1].fingerprint, published[2].fingerprint},
        )
        self.assertEqual(preview.delete_runs, ())
        self.assertFalse(published[0].run_dir.exists())

    def test_stale_retention_preview_cannot_delete_a_new_active_run(self) -> None:
        """Una preview anterior a una reactivación se rechaza sin borrar."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            with mock.patch(
                "src.features.artifacts._prune_after_valid_activation",
                return_value=(),
            ):
                runs = [
                    publish_feature_run(
                        _write_staged_run(
                            output,
                            fingerprint=_fingerprint(label),
                            label=label,
                        ),
                        fingerprint=_fingerprint(label),
                        output_dir=output,
                    )
                    for label in ("A", "B", "C")
                ]
            preview = preview_feature_run_retention(output_dir=output)
            activate_feature_run(runs[0], output_dir=output)

            with self.assertRaisesRegex(FeatureArtifactError, "preview"):
                prune_feature_runs(preview)

            active = resolve_active_feature_run(output_dir=output)
            active_directory_exists = runs[0].run_dir.is_dir()

        self.assertEqual(active.fingerprint, runs[0].fingerprint)
        self.assertTrue(active_directory_exists)

    def test_retention_rejects_links_at_every_destructive_boundary(self) -> None:
        """Output, runs, puntero y contenido enlazados bloquean toda poda."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            with mock.patch(
                "src.features.artifacts._prune_after_valid_activation",
                return_value=(),
            ):
                runs = [
                    publish_feature_run(
                        _write_staged_run(
                            output,
                            fingerprint=_fingerprint(label),
                            label=label,
                        ),
                        fingerprint=_fingerprint(label),
                        output_dir=output,
                    )
                    for label in ("A", "B", "C")
                ]

            blocked_paths = (
                output,
                output / "runs",
                output / "manifest.json",
                runs[0].run_dir,
                runs[0].run_dir / "manifest.json",
            )
            for blocked_path in blocked_paths:
                with self.subTest(blocked_path=blocked_path):
                    with (
                        mock.patch(
                            "src.features.artifacts._is_link_or_junction",
                            side_effect=lambda candidate, blocked=blocked_path: (
                                Path(candidate) == blocked
                            ),
                        ),
                        mock.patch("src.features.artifacts.shutil.rmtree") as remove,
                    ):
                        with self.assertRaisesRegex(
                            FeatureArtifactError,
                            "enlace|junction",
                        ):
                            preview_feature_run_retention(output_dir=output)
                    remove.assert_not_called()

    def test_valid_manifest_mutation_makes_retention_preview_stale(self) -> None:
        """Una mutación aún verificable invalida el sello antes de borrar."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            with mock.patch(
                "src.features.artifacts._prune_after_valid_activation",
                return_value=(),
            ):
                for day, label in enumerate(("A", "B", "C"), start=1):
                    publish_feature_run(
                        _write_staged_run(
                            output,
                            fingerprint=_fingerprint(label),
                            label=label,
                            created_at_utc=f"2026-08-{day:02d}T00:00:00+00:00",
                        ),
                        fingerprint=_fingerprint(label),
                        output_dir=output,
                    )
            preview = preview_feature_run_retention(output_dir=output)
            self.assertEqual(len(preview.delete_runs), 1)
            manifest_path = preview.delete_runs[0] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["audit_note"] = "changed-after-preview"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch("src.features.artifacts.shutil.rmtree") as remove:
                with self.assertRaisesRegex(FeatureArtifactError, "preview"):
                    prune_feature_runs(preview)
            remove.assert_not_called()

    def test_activation_failure_preserves_previous_active_generation(self) -> None:
        """Un fallo del puntero deja A usable y B publicado para reintento."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            first_fingerprint = _fingerprint("stable")
            publish_feature_run(
                _write_staged_run(
                    output,
                    fingerprint=first_fingerprint,
                    label="stable",
                ),
                fingerprint=first_fingerprint,
                output_dir=output,
            )
            pointer_before = (output / "manifest.json").read_bytes()
            second_fingerprint = _fingerprint("pending")
            staged = _write_staged_run(
                output,
                fingerprint=second_fingerprint,
                label="pending",
            )
            real_replace = os.replace

            def fail_only_activation(source: object, target: object) -> None:
                """Simula una ACL rota únicamente en el puntero activo."""

                if Path(target).resolve() == (output / "manifest.json").resolve():
                    raise PermissionError("ACL simulada")
                real_replace(source, target)

            with (
                mock.patch(
                    "src.features.artifacts.os.replace",
                    side_effect=fail_only_activation,
                ),
                mock.patch("src.features.artifacts._prune_after_valid_activation") as prune,
            ):
                with self.assertRaisesRegex(
                    FeatureArtifactError,
                    "activar atómicamente",
                ):
                    publish_feature_run(
                        staged,
                        fingerprint=second_fingerprint,
                        output_dir=output,
                    )
            prune.assert_not_called()

            self.assertEqual(
                (output / "manifest.json").read_bytes(),
                pointer_before,
            )
            self.assertEqual(
                resolve_active_feature_run(output_dir=output).fingerprint,
                first_fingerprint,
            )
            verify_feature_run(
                output / "runs" / second_fingerprint,
                output_dir=output,
            )

    def test_preflight_reports_permissions_before_a_build_workspace_exists(self) -> None:
        """La sonda falla pronto y no crea ningún workspace de cálculo."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            with mock.patch(
                "src.features.artifacts.os.replace",
                side_effect=PermissionError("ACL simulada"),
            ):
                with self.assertRaisesRegex(
                    FeatureArtifactError,
                    "no permite publicar",
                ):
                    preflight_feature_publication(output_dir=output)

            workspaces = list(output.glob(".feature-build-*"))

        self.assertEqual(workspaces, [])

    def test_same_fingerprint_cannot_overwrite_different_artifacts(self) -> None:
        """Detecta no determinismo y conserva inmutable el primer contenido."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "features_active"
            preflight_feature_publication(output_dir=output)
            fingerprint = _fingerprint("same-input")
            first = publish_feature_run(
                _write_staged_run(
                    output,
                    fingerprint=fingerprint,
                    label="first-output",
                ),
                fingerprint=fingerprint,
                output_dir=output,
            )
            original = (first.run_dir / "training_M.parquet").read_bytes()

            with self.assertRaisesRegex(
                FeatureArtifactError,
                "artefactos distintos",
            ):
                publish_feature_run(
                    _write_staged_run(
                        output,
                        fingerprint=fingerprint,
                        label="different-output",
                    ),
                    fingerprint=fingerprint,
                    output_dir=output,
                )

            observed = (first.run_dir / "training_M.parquet").read_bytes()

        self.assertEqual(observed, original)


if __name__ == "__main__":
    unittest.main()

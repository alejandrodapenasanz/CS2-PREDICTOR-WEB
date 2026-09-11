"""Pruebas offline del registro inmutable de challengers de TENNIS."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.artifacts import (  # noqa: E402
    ArtifactError,
    create_staging_directory,
    find_verified_run,
    fingerprint_payload,
    publish_staged_run,
    verify_published_run,
)


class ModelArtifactsTest(unittest.TestCase):
    """Comprueba que registrar nunca equivale a activar."""

    def test_fingerprint_is_canonical_for_mapping_order(self) -> None:
        """El orden de claves JSON no cambia la identidad del run."""

        first = fingerprint_payload({"a": 1, "b": {"c": 2}})
        second = fingerprint_payload({"b": {"c": 2}, "a": 1})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_registers_and_finds_challenger_without_active_pointer(self) -> None:
        """Publicar staging solo crea runs/<fingerprint>."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "models"
            staging = create_staging_directory(output)
            artifact = staging / "models" / "M" / "model.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("modelo", encoding="utf-8")
            fingerprint = fingerprint_payload({"run": "test"})

            registered = publish_staged_run(
                staging,
                fingerprint=fingerprint,
                manifest_payload={"artifact_version": "test-v1"},
                output_dir=output,
            )
            reused = find_verified_run(fingerprint, output_dir=output)

            self.assertFalse(registered.skipped)
            self.assertIsNotNone(reused)
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual(registered.manifest_path, registered.run_dir / "manifest.json")
            verify_published_run(registered.run_dir)

    def test_detects_changed_registered_artifact(self) -> None:
        """Un challenger publicado permanece inmutable y verificable."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "models"
            staging = create_staging_directory(output)
            artifact = staging / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            registered = publish_staged_run(
                staging,
                fingerprint=fingerprint_payload({"run": "corrupt"}),
                manifest_payload={"artifact_version": "test-v1"},
                output_dir=output,
            )
            (registered.run_dir / "artifact.txt").write_text("alterado", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactError, "tamaño|SHA-256"):
                verify_published_run(registered.run_dir)

    def test_duplicate_fingerprint_never_overwrites_run(self) -> None:
        """Un segundo staging no puede sustituir una identidad existente."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "models"
            fingerprint = fingerprint_payload({"run": "same"})
            first = create_staging_directory(output)
            (first / "model.txt").write_text("A", encoding="utf-8")
            publish_staged_run(
                first,
                fingerprint=fingerprint,
                manifest_payload={"artifact_version": "test-v1"},
                output_dir=output,
            )
            second = create_staging_directory(output)
            (second / "model.txt").write_text("B", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactError, "ya está publicado"):
                publish_staged_run(
                    second,
                    fingerprint=fingerprint,
                    manifest_payload={"artifact_version": "test-v1"},
                    output_dir=output,
                )
            payload = json.loads(
                (output / "runs" / fingerprint / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["fingerprint"], fingerprint)


if __name__ == "__main__":
    unittest.main()

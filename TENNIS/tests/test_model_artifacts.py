"""Pruebas offline de publicación inmutable de artefactos."""

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
    activate_published_run,
    create_staging_directory,
    find_verified_run,
    fingerprint_payload,
    publish_staged_run,
    verify_published_run,
)


class ModelArtifactsTest(unittest.TestCase):
    """Comprueba fingerprint, publicación, reutilización e integridad."""

    def test_fingerprint_is_canonical_for_mapping_order(self) -> None:
        """El orden de claves JSON no cambia la identidad del run."""

        first = fingerprint_payload({"a": 1, "b": {"c": 2}})
        second = fingerprint_payload({"b": {"c": 2}, "a": 1})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_publishes_and_finds_an_immutable_verified_run(self) -> None:
        """Mueve staging, activa manifiesto y verifica todos los hashes."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            output = Path(temporary) / "models"
            staging = create_staging_directory(output)
            artifact = staging / "models" / "M" / "model.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("modelo", encoding="utf-8")
            fingerprint = fingerprint_payload({"run": "test"})

            published = publish_staged_run(
                staging,
                fingerprint=fingerprint,
                manifest_payload={"artifact_version": "test-v1"},
                output_dir=output,
            )
            reused = find_verified_run(fingerprint, output_dir=output)

            self.assertFalse(published.skipped)
            self.assertIsNotNone(reused)
            assert reused is not None
            self.assertTrue(reused.skipped)
            self.assertTrue((output / "manifest.json").is_file())
            verify_published_run(reused.run_dir)

    def test_detects_a_changed_published_artifact(self) -> None:
        """Un archivo modificado después de publicar invalida el run."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            output = Path(temporary) / "models"
            staging = create_staging_directory(output)
            artifact = staging / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            fingerprint = fingerprint_payload({"run": "corrupt"})
            published = publish_staged_run(
                staging,
                fingerprint=fingerprint,
                manifest_payload={"artifact_version": "test-v1"},
                output_dir=output,
            )
            changed = published.run_dir / "artifact.txt"
            changed.write_text("alterado", encoding="utf-8")

            with self.assertRaisesRegex(
                ArtifactError, "tamaño|SHA-256"
            ):
                verify_published_run(published.run_dir)

    def test_reactivates_a_verified_older_run(self) -> None:
        """Reutilizar A después de B actualiza el puntero activo a A."""

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tests"
        ) as temporary:
            output = Path(temporary) / "models"
            published_runs = []
            for label in ("A", "B"):
                staging = create_staging_directory(output)
                (staging / "model.txt").write_text(label, encoding="utf-8")
                fingerprint = fingerprint_payload({"run": label})
                published_runs.append(
                    publish_staged_run(
                        staging,
                        fingerprint=fingerprint,
                        manifest_payload={"artifact_version": "test-v1"},
                        output_dir=output,
                    )
                )

            first = find_verified_run(
                published_runs[0].fingerprint,
                output_dir=output,
            )
            assert first is not None
            activated = activate_published_run(first, output_dir=output)
            active = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue(activated.skipped)
            self.assertEqual(active["fingerprint"], first.fingerprint)
            self.assertEqual(
                active["active_run"],
                f"runs/{first.fingerprint}",
            )


if __name__ == "__main__":
    unittest.main()

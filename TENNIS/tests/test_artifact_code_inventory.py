"""Pruebas sintéticas del inventario de código de artefactos persistentes."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_integrity import (  # noqa: E402
    CodeInventoryError,
    build_code_inventory,
)
from src.elo.build import (  # noqa: E402
    ELO_CODE_PATHS,
    VerifiedManifest,
    compute_input_fingerprint,
)
from src.elo.parameters import DEFAULT_ELO_PARAMETERS  # noqa: E402
from src.features.dataset import (  # noqa: E402
    FEATURE_CODE_PATHS,
    _input_fingerprint,
)
from src.features.parameters import DEFAULT_FEATURE_PARAMETERS  # noqa: E402
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402


TESTS_ROOT = Path(__file__).resolve().parent


def _inventory(sha_a: str, sha_b: str) -> tuple[dict[str, object], ...]:
    """Crea un inventario válido sin depender del árbol productivo."""

    return (
        {"path": "src/a.py", "size": 1, "sha256": sha_a * 64},
        {"path": "src/b.py", "size": 2, "sha256": sha_b * 64},
    )


class ArtifactCodeInventoryTest(unittest.TestCase):
    """Garantiza invalidación por código y serialización determinista."""

    def _temporary_root(self) -> TemporaryDirectory[str]:
        """Crea un fixture temporal exclusivamente bajo ``TENNIS/tests``."""

        return TemporaryDirectory(
            prefix=".artifact-code-",
            dir=TESTS_ROOT,
        )

    def test_binary_content_change_changes_inventory_without_version(self) -> None:
        """Cambiar bytes altera SHA-256 aun conservando ruta y versión."""

        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "src" / "formula.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"rating = 1500\n")
            before = build_code_inventory(root, ("src/formula.py",))
            source.write_bytes(b"rating = 1501\n")
            after = build_code_inventory(root, ("src/formula.py",))

        self.assertNotEqual(before, after)
        self.assertNotEqual(before[0]["sha256"], after[0]["sha256"])

    def test_inventory_order_is_deterministic_and_unsafe_paths_fail(self) -> None:
        """Ordena rutas explícitas y rechaza traversal o duplicados."""

        with self._temporary_root() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_bytes(b"a")
            (root / "src" / "b.py").write_bytes(b"b")
            forward = build_code_inventory(
                root,
                ("src/a.py", "src/b.py"),
            )
            reverse = build_code_inventory(
                root,
                ("src/b.py", "src/a.py"),
            )
            with self.assertRaises(CodeInventoryError):
                build_code_inventory(root, ("../outside.py",))
            with self.assertRaises(CodeInventoryError):
                build_code_inventory(root, ("src\\a.py",))
            with self.assertRaises(CodeInventoryError):
                build_code_inventory(root, ("src/a.py", "src/a.py"))

        self.assertEqual(forward, reverse)

    def test_elo_fingerprint_changes_if_only_code_hash_changes(self) -> None:
        """Evita reutilizar Elo si se olvida subir la versión declarada."""

        manifest = VerifiedManifest(
            source_repository="fixture",
            source_commit="a" * 40,
            manifest_path=TESTS_ROOT / "unused.json",
            raw_dir=TESTS_ROOT,
            match_files=(),
        )
        first = compute_input_fingerprint(
            manifest,
            DEFAULT_ELO_PARAMETERS.as_dict(),
            "same-version",
            code_inventory=_inventory("a", "b"),
        )
        changed = compute_input_fingerprint(
            manifest,
            DEFAULT_ELO_PARAMETERS.as_dict(),
            "same-version",
            code_inventory=_inventory("c", "b"),
        )

        self.assertNotEqual(first, changed)

    def test_feature_fingerprint_changes_if_only_code_hash_changes(self) -> None:
        """Evita reutilizar Parquet si cambia una fórmula sin subir versión."""

        shared = {
            "source_commit": "a" * 40,
            "genders": ("M",),
            "inventory": (),
            "feature_parameters": DEFAULT_FEATURE_PARAMETERS,
            "elo_parameters": DEFAULT_ELO_PARAMETERS,
            "identity_exclusion_after_dates": {},
            "source_date_policy": DEFAULT_SOURCE_DATE_POLICY,
            "elo_contracts": {
                "M": {
                    "algorithm_version": "synthetic",
                    "input_fingerprint": "f" * 64,
                }
            },
        }
        first = _input_fingerprint(
            **shared,
            code_inventory=_inventory("a", "b"),
        )
        changed = _input_fingerprint(
            **shared,
            code_inventory=_inventory("a", "c"),
        )

        self.assertNotEqual(first, changed)

    def test_production_lists_include_scripts_identity_and_loaders(self) -> None:
        """Fija las dependencias externas que podrían alterar cada build."""

        expected_elo = {
            "scripts/build_elo.py",
            "src/data_loaders.py",
            "src/identity_integrity.py",
        }
        expected_features = {
            "scripts/build_features.py",
            "src/data_loaders.py",
            "src/identity_integrity.py",
        }

        self.assertTrue(expected_elo.issubset(ELO_CODE_PATHS))
        self.assertTrue(expected_features.issubset(FEATURE_CODE_PATHS))
        self.assertTrue(build_code_inventory(PROJECT_ROOT, ELO_CODE_PATHS))
        self.assertTrue(build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS))


if __name__ == "__main__":
    unittest.main()

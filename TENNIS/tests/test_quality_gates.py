"""Pruebas offline de las puertas locales de calidad de TENNIS."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_import_coverage import (
    discover_third_party_imports,
    missing_import_requirements,
    parse_direct_requirements,
)
from scripts.smoke_project import LAUNCHER_ENTRYPOINTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportCoverageTests(unittest.TestCase):
    """Fija el contrato manifest/import y demuestra su caso negativo."""

    def test_real_direct_requirements_cover_all_external_imports(self) -> None:
        """No permite imports externos productivos sin declaración directa."""

        declared = parse_direct_requirements(PROJECT_ROOT / "requirements.txt")
        imported = discover_third_party_imports(PROJECT_ROOT)
        self.assertEqual(missing_import_requirements(declared, imported), {})

    def test_removing_dependency_fails_and_restoring_it_passes(self) -> None:
        """Quitar pandas de una copia falla; restaurar el manifiesto pasa."""

        source = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        without_pandas = "\n".join(
            line for line in source.splitlines() if not line.startswith("pandas")
        )
        imported = discover_third_party_imports(PROJECT_ROOT)
        with TemporaryDirectory(prefix="tennis-import-contract-") as temporary:
            manifest = Path(temporary) / "requirements.txt"
            manifest.write_text(without_pandas + "\n", encoding="utf-8")
            missing = missing_import_requirements(parse_direct_requirements(manifest), imported)
            self.assertIn("pandas", missing)

            manifest.write_text(source, encoding="utf-8")
            restored = missing_import_requirements(parse_direct_requirements(manifest), imported)
            self.assertEqual(restored, {})


class EntrypointSmokeContractTests(unittest.TestCase):
    """Impide que el smoke se separe de la ruta real de PowerShell."""

    def test_smoke_covers_every_launcher_entrypoint(self) -> None:
        """Todos los scripts ejecutados por run_tennis están en el smoke."""

        launcher = (PROJECT_ROOT / "run_tennis.ps1").read_text(encoding="utf-8")
        for relative_path in LAUNCHER_ENTRYPOINTS:
            self.assertIn(relative_path.replace("/", "\\"), launcher)


if __name__ == "__main__":
    unittest.main()

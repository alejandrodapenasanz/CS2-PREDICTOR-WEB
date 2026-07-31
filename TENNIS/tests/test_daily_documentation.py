"""Pruebas de los contratos documentales y del lanzador de fase 8."""

import unittest

from src.config import PROJECT_ROOT


class DailyDocumentationTests(unittest.TestCase):
    """Evita que la integración no invasiva diverja de su documentación."""

    def test_integration_contains_the_exact_single_start_line(self) -> None:
        """Fija la llamada relativa que el usuario añadirá manualmente."""

        documentation = (
            PROJECT_ROOT / "docs" / "integracion.md"
        ).read_text(encoding="utf-8")
        expected = (
            "& (Join-Path $PSScriptRoot 'TENNIS\\run_tennis.ps1')"
        )
        self.assertIn(expected, documentation)
        self.assertIn("Para revertir", documentation)

    def test_launcher_resolves_every_path_from_its_own_location(self) -> None:
        """Comprueba que el lanzador no depende del directorio actual."""

        launcher_path = PROJECT_ROOT / "run_tennis.ps1"
        launcher = launcher_path.read_text(encoding="utf-8")
        self.assertIn("$MyInvocation.MyCommand.Path", launcher)
        self.assertIn("'.venv\\Scripts\\Activate.ps1'", launcher)
        self.assertIn("'scripts\\daily_predictions.py'", launcher)
        self.assertNotIn("start.ps1", launcher)

    def test_readme_documents_full_refresh_and_daily_order(self) -> None:
        """Exige instalación, fuentes, Elo, features, reentreno y predicción."""

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        expected_fragments = (
            "python -m pip install -r",
            "python scripts\\update_sources.py",
            "python scripts\\build_elo.py",
            "python scripts\\build_features.py",
            "python scripts\\retrain_models.py",
            ".\\TENNIS\\run_tennis.ps1",
            "edge_a/b = P_modelo - P_mercado",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)


if __name__ == "__main__":
    unittest.main()

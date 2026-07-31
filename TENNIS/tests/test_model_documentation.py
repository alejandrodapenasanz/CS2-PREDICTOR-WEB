"""Pruebas del contrato permanente de documentación de fase 7."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelDocumentationTest(unittest.TestCase):
    """Exige docstrings en módulos, clases y funciones nuevas."""

    def test_modeling_modules_have_complete_docstrings(self) -> None:
        """Ningún módulo, clase o callable de modelado queda sin documentar."""

        files = sorted((PROJECT_ROOT / "src" / "modeling").glob("*.py"))
        files.append(PROJECT_ROOT / "scripts" / "retrain_models.py")
        missing: list[str] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if ast.get_docstring(tree) is None:
                missing.append(f"{path.name}:module")
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ) and ast.get_docstring(node) is None:
                    missing.append(f"{path.name}:{node.name}")

        self.assertEqual(missing, [])

    def test_retrain_script_header_explains_contract_and_execution(self) -> None:
        """La cabecera declara función, entradas y forma de ejecución."""

        tree = ast.parse(
            (PROJECT_ROOT / "scripts" / "retrain_models.py").read_text(
                encoding="utf-8"
            )
        )
        header = ast.get_docstring(tree) or ""

        self.assertIn("Qué hace:", header)
        self.assertIn("Qué recibe:", header)
        self.assertIn("Cómo se ejecuta", header)


if __name__ == "__main__":
    unittest.main()

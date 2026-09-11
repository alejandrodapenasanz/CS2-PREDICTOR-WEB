"""Regresiones de permisos para workspaces publicados mediante rename."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.dataset import _create_build_workspace  # noqa: E402
from src.modeling import artifacts as model_artifacts  # noqa: E402


def _windows_acl_is_protected(path: Path) -> bool:
    """Consulta la bandera DACL sin depender del idioma de ``icacls``."""

    command = (
        "[Console]::Write((Get-Acl -LiteralPath "
        "$env:TENNIS_ACL_TEST_PATH)."
        "AreAccessRulesProtected.ToString().ToLowerInvariant())"
    )
    environment = os.environ.copy()
    environment["TENNIS_ACL_TEST_PATH"] = str(path)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            command,
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip() == "true"


@unittest.skipUnless(os.name == "nt", "La regresion comprueba DACL de Windows.")
class ArtifactWorkspacePermissionsTest(unittest.TestCase):
    """Los runs movidos deben ser legibles por procesos posteriores."""

    def test_feature_workspace_inherits_parent_acl(self) -> None:
        """El workspace de features no usa la DACL privada de mkdtemp."""

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "features_active"
            output.mkdir()
            workspace = _create_build_workspace(output)

            self.assertFalse(_windows_acl_is_protected(workspace))

    def test_model_staging_inherits_parent_acl(self) -> None:
        """El staging del modelo mantiene activa la herencia de permisos."""

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            output = temporary_root / "models"
            with mock.patch.object(
                model_artifacts,
                "PROJECT_ROOT",
                temporary_root,
            ):
                staging = model_artifacts.create_staging_directory(output)

            self.assertFalse(_windows_acl_is_protected(staging))


if __name__ == "__main__":
    unittest.main()

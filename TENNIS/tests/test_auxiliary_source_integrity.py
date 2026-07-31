"""Pruebas de integridad de jugadores y rankings usados cada mañana."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.config import PROJECT_ROOT
from src.features import verify_auxiliary_source_inventory
from src.features.dataset import FeatureDatasetSourceError


def _git_blob_sha(content: bytes) -> str:
    """Calcula el identificador de blob Git de unos bytes de fixture."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


class AuxiliarySourceIntegrityTests(unittest.TestCase):
    """Impide cargar auxiliares distintos de los publicados por commit."""

    def test_tampered_ranking_is_rejected_even_with_same_size(self) -> None:
        """Verifica el SHA y no se limita al nombre o al tamaño del archivo."""

        with TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            raw = root / "raw"
            entries = []
            fixtures = (
                ("atp/atp_players.csv", "atp_players", b"id,name\n1,A\n"),
                ("wta/wta_players.csv", "wta_players", b"id,name\n2,B\n"),
                (
                    "atp/atp_rankings_00s.csv",
                    "atp_rankings",
                    b"date,rank\n20200101,1\n",
                ),
                (
                    "wta/wta_rankings_00s.csv",
                    "wta_rankings",
                    b"date,rank\n20200101,1\n",
                ),
            )
            for relative, category, content in fixtures:
                path = raw / Path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                entries.append(
                    {
                        "local_path": relative,
                        "remote_path": relative,
                        "category": category,
                        "size": len(content),
                        "git_blob_sha": _git_blob_sha(content),
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_repository": (
                            "Aneeshers/tennis-sackmann-archive"
                        ),
                        "source_commit": "a" * 40,
                        "files": entries,
                    }
                ),
                encoding="utf-8",
            )

            verified = verify_auxiliary_source_inventory(manifest, raw)
            self.assertEqual(verified.source_commit, "a" * 40)
            self.assertEqual(len(verified.files), 4)

            ranking = raw / "atp" / "atp_rankings_00s.csv"
            ranking.write_bytes(b"date,rank\n20200102,1\n")
            with self.assertRaises(FeatureDatasetSourceError):
                verify_auxiliary_source_inventory(manifest, raw)


if __name__ == "__main__":
    unittest.main()

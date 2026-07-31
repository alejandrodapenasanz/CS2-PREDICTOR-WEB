"""Pruebas offline del downloader oficial del Match Charting Project.

Las pruebas simulan las respuestas REST y raw de GitHub. No requieren red ni
modifican los datos crudos reales del proyecto.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
import unittest
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.match_charting_download import (  # noqa: E402
    ACTIVE_MANIFEST_FILENAME,
    GITHUB_API_ROOT,
    RAW_CONTENT_ROOT,
    SOURCE_REF,
    SOURCE_REPOSITORY,
    VERSIONED_MANIFEST_DIRECTORY,
    DownloadIntegrityError,
    LocalDataConflictError,
    download_match_charting_data,
    fetch_source_inventory,
)


class FakeResponse:
    """Imita la parte de ``requests.Response`` usada por el módulo."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any] | None = None,
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        """Guarda una respuesta JSON o binaria con estado configurable."""

        self._payload = payload
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Lanza ``HTTPError`` cuando el estado simulado no es exitoso."""

        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Mapping[str, Any]:
        """Devuelve el objeto JSON simulado."""

        if self._payload is None:
            raise ValueError("La respuesta simulada no contiene JSON.")
        return self._payload

    def iter_content(self, chunk_size: int) -> Any:
        """Entrega el contenido binario en fragmentos del tamaño solicitado."""

        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def __enter__(self) -> FakeResponse:
        """Devuelve la respuesta al entrar en un contexto."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Finaliza el contexto sin suprimir excepciones."""

        return None


class FakeSession:
    """Sirve respuestas predeterminadas y registra todas las peticiones."""

    def __init__(self, routes: Mapping[str, FakeResponse]) -> None:
        """Copia las rutas simuladas e inicializa el registro de llamadas."""

        self._routes = dict(routes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Devuelve la respuesta asociada o falla ante una URL inesperada."""

        self.calls.append((url, kwargs))
        if url not in self._routes:
            raise AssertionError(f"Petición inesperada en test offline: {url}")
        return self._routes[url]

    def close(self) -> None:
        """Marca la sesión como cerrada."""

        self.closed = True


def _git_blob_sha(content: bytes) -> str:
    """Calcula de forma independiente el SHA-1 de blob para una fixture."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _commit_url() -> str:
    """Devuelve la URL REST que resuelve la rama configurada."""

    return (
        f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/commits/{SOURCE_REF}"
    )


def _tree_url(tree_sha: str) -> str:
    """Devuelve la URL REST del árbol recursivo fijado."""

    return (
        f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/git/trees/"
        f"{tree_sha}?recursive=1"
    )


def _raw_url(commit_sha: str, remote_path: str) -> str:
    """Devuelve la URL raw fijada a commit para una ruta de fixture."""

    encoded_path = quote(remote_path, safe="/")
    return (
        f"{RAW_CONTENT_ROOT}/{SOURCE_REPOSITORY}/{commit_sha}/"
        f"{encoded_path}"
    )


def _make_session(
    *,
    commit_sha: str,
    tree_sha: str,
    selected_contents: Mapping[str, bytes],
    extra_tree_entries: tuple[Mapping[str, Any], ...] = (),
    raw_overrides: Mapping[str, bytes] | None = None,
) -> FakeSession:
    """Construye una sesión con commit, árbol y blobs completamente offline."""

    entries: list[Mapping[str, Any]] = [
        {
            "path": remote_path,
            "type": "blob",
            "size": len(content),
            "sha": _git_blob_sha(content),
        }
        for remote_path, content in selected_contents.items()
    ]
    entries.extend(extra_tree_entries)
    routes: dict[str, FakeResponse] = {
        _commit_url(): FakeResponse(
            payload={
                "sha": commit_sha,
                "commit": {"tree": {"sha": tree_sha}},
            }
        ),
        _tree_url(tree_sha): FakeResponse(
            payload={
                "sha": tree_sha,
                "truncated": False,
                "tree": entries,
            }
        ),
    }
    replacements = raw_overrides or {}
    for remote_path, content in selected_contents.items():
        routes[_raw_url(commit_sha, remote_path)] = FakeResponse(
            content=replacements.get(remote_path, content)
        )
    return FakeSession(routes)


def _minimum_contents(
    *,
    men: bytes = b"men\n",
    women: bytes = b"women\n",
) -> dict[str, bytes]:
    """Devuelve los cuatro archivos mínimos exigidos por el inventario."""

    return {
        "README.md": b"# Match Charting Project\n",
        "data_dictionary.txt": b"match_id - unique match identifier\n",
        "charting-m-matches.csv": men,
        "charting-w-matches.csv": women,
    }


class MatchChartingInventoryTest(unittest.TestCase):
    """Comprueba selección exhaustiva y fijación del árbol remoto."""

    def test_inventory_selects_every_csv_and_documents_but_not_xlsm(
        self,
    ) -> None:
        """Incluye CSV recursivos y documentos, excluyendo el workbook."""

        commit_sha = "a" * 40
        tree_sha = "b" * 40
        contents = _minimum_contents()
        contents["nested/derived.csv"] = b"a,b\n1,2\n"
        session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
            extra_tree_entries=(
                {
                    "path": "MatchChart 0.3.2.xlsm",
                    "type": "blob",
                    "size": 7,
                    "sha": _git_blob_sha(b"workbook"),
                },
                {
                    "path": "LICENSE",
                    "type": "blob",
                    "size": 7,
                    "sha": _git_blob_sha(b"license"),
                },
                {
                    "path": "nested",
                    "type": "tree",
                    "size": 0,
                    "sha": "c" * 40,
                },
            ),
        )

        inventory = fetch_source_inventory(session)

        self.assertEqual(inventory.commit_sha, commit_sha)
        self.assertEqual(inventory.tree_sha, tree_sha)
        self.assertEqual(
            {item.remote_path for item in inventory.files},
            set(contents),
        )
        self.assertIn(
            (_tree_url(tree_sha), {"timeout": (15.0, 180.0)}),
            session.calls,
        )
        self.assertFalse(session.closed)


class MatchChartingDownloadTest(unittest.TestCase):
    """Comprueba integridad, manifiestos e incremento sin acceso a red."""

    def test_initial_download_writes_active_and_versioned_manifests(
        self,
    ) -> None:
        """Descarga el snapshot y publica dos manifiestos equivalentes."""

        commit_sha = "1" * 40
        tree_sha = "2" * 40
        contents = _minimum_contents()
        contents["charting-m-stats-Overview.csv"] = b"match_id,player\n"
        session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
            extra_tree_entries=(
                {
                    "path": "MatchChart 0.3.2.xlsm",
                    "type": "blob",
                    "size": 8,
                    "sha": _git_blob_sha(b"excluded"),
                },
            ),
        )

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory) / "mcp"
            report = download_match_charting_data(
                raw_dir=raw_dir,
                session=session,
            )

            self.assertEqual(report.source_commit, commit_sha)
            self.assertEqual(report.downloaded_count, len(contents))
            self.assertEqual(report.updated_count, 0)
            self.assertEqual(report.skipped_count, 0)
            for remote_path, expected_content in contents.items():
                self.assertEqual(
                    (raw_dir / remote_path).read_bytes(),
                    expected_content,
                )
            self.assertFalse(
                (raw_dir / "MatchChart 0.3.2.xlsm").exists()
            )

            active_path = raw_dir / ACTIVE_MANIFEST_FILENAME
            versioned_path = (
                raw_dir
                / VERSIONED_MANIFEST_DIRECTORY
                / f"{commit_sha}.json"
            )
            self.assertEqual(report.active_manifest_path, active_path)
            self.assertEqual(
                report.versioned_manifest_path,
                versioned_path,
            )
            active_payload = json.loads(
                active_path.read_text(encoding="utf-8")
            )
            versioned_payload = json.loads(
                versioned_path.read_text(encoding="utf-8")
            )
            self.assertEqual(active_payload, versioned_payload)
            self.assertEqual(active_payload["source_commit"], commit_sha)
            self.assertEqual(active_payload["source_tree"], tree_sha)
            self.assertEqual(
                {
                    item["remote_path"]
                    for item in active_payload["files"]
                },
                set(contents),
            )

    def test_new_commit_updates_only_changed_and_new_blobs(
        self,
    ) -> None:
        """Actualiza blobs conocidos, añade nuevos y conserva los idénticos."""

        first_commit = "3" * 40
        first_tree = "4" * 40
        first_contents = _minimum_contents(men=b"men-v1\n")
        first_session = _make_session(
            commit_sha=first_commit,
            tree_sha=first_tree,
            selected_contents=first_contents,
        )

        second_commit = "5" * 40
        second_tree = "6" * 40
        second_contents = dict(first_contents)
        second_contents["charting-m-matches.csv"] = b"men-v2\n"
        second_contents["charting-w-stats-Rally.csv"] = b"match_id,row\n"
        second_session = _make_session(
            commit_sha=second_commit,
            tree_sha=second_tree,
            selected_contents=second_contents,
        )

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory) / "mcp"
            download_match_charting_data(
                raw_dir=raw_dir,
                session=first_session,
            )

            report = download_match_charting_data(
                raw_dir=raw_dir,
                session=second_session,
            )

            self.assertEqual(report.source_commit, second_commit)
            self.assertEqual(
                {path.name for path in report.updated},
                {"charting-m-matches.csv"},
            )
            self.assertEqual(
                {path.name for path in report.downloaded},
                {"charting-w-stats-Rally.csv"},
            )
            self.assertEqual(report.skipped_count, 3)
            self.assertEqual(
                (raw_dir / "charting-m-matches.csv").read_bytes(),
                b"men-v2\n",
            )
            raw_calls = [
                url
                for url, _ in second_session.calls
                if url.startswith(RAW_CONTENT_ROOT)
            ]
            self.assertEqual(
                set(raw_calls),
                {
                    _raw_url(
                        second_commit,
                        "charting-m-matches.csv",
                    ),
                    _raw_url(
                        second_commit,
                        "charting-w-stats-Rally.csv",
                    ),
                },
            )
            self.assertTrue(
                (
                    raw_dir
                    / VERSIONED_MANIFEST_DIRECTORY
                    / f"{first_commit}.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    raw_dir
                    / VERSIONED_MANIFEST_DIRECTORY
                    / f"{second_commit}.json"
                ).is_file()
            )
            active_payload = json.loads(
                (raw_dir / ACTIVE_MANIFEST_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                active_payload["source_commit"],
                second_commit,
            )

    def test_identical_snapshot_performs_no_raw_downloads(self) -> None:
        """Revalida archivos locales y los omite en una segunda ejecución."""

        commit_sha = "7" * 40
        tree_sha = "8" * 40
        contents = _minimum_contents()
        first_session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
        )
        second_session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
        )

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory) / "mcp"
            download_match_charting_data(
                raw_dir=raw_dir,
                session=first_session,
            )
            report = download_match_charting_data(
                raw_dir=raw_dir,
                session=second_session,
            )

            self.assertEqual(report.downloaded_count, 0)
            self.assertEqual(report.updated_count, 0)
            self.assertEqual(report.skipped_count, len(contents))
            self.assertFalse(
                any(
                    url.startswith(RAW_CONTENT_ROOT)
                    for url, _ in second_session.calls
                )
            )

    def test_download_rejects_wrong_size_and_wrong_blob_sha(self) -> None:
        """Rechaza por separado bytes truncados y contenido con SHA distinto."""

        cases = {
            "wrong_size": b"x",
            "wrong_sha": b"wrong",
        }
        for case_name, bad_content in cases.items():
            with self.subTest(case=case_name):
                commit_sha = "9" * 40
                tree_sha = "a" * 40
                contents = _minimum_contents()
                contents["README.md"] = b"right"
                session = _make_session(
                    commit_sha=commit_sha,
                    tree_sha=tree_sha,
                    selected_contents=contents,
                    raw_overrides={"README.md": bad_content},
                )

                with tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parent
                ) as temporary_directory:
                    raw_dir = Path(temporary_directory) / "mcp"
                    with self.assertRaises(DownloadIntegrityError):
                        download_match_charting_data(
                            raw_dir=raw_dir,
                            session=session,
                        )
                    self.assertFalse((raw_dir / "README.md").exists())
                    self.assertFalse(
                        (raw_dir / ACTIVE_MANIFEST_FILENAME).exists()
                    )

    def test_locally_modified_managed_file_is_not_overwritten(self) -> None:
        """Detecta manipulación local aunque el commit remoto no cambie."""

        commit_sha = "b" * 40
        tree_sha = "c" * 40
        contents = _minimum_contents()
        first_session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
        )
        second_session = _make_session(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            selected_contents=contents,
        )

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory) / "mcp"
            download_match_charting_data(
                raw_dir=raw_dir,
                session=first_session,
            )
            managed_path = raw_dir / "charting-m-matches.csv"
            managed_path.write_bytes(b"local edit\n")

            with self.assertRaises(LocalDataConflictError):
                download_match_charting_data(
                    raw_dir=raw_dir,
                    session=second_session,
                )
            self.assertEqual(managed_path.read_bytes(), b"local edit\n")


if __name__ == "__main__":
    unittest.main()

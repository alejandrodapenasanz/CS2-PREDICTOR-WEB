"""Tests sin red de la sincronización incremental de Sackmann.

Las sesiones HTTP falsas reproducen las respuestas de commit, árbol Git y
contenido RAW. Los tests verifican cambios de commit, integridad local,
semántica acotada de ``force`` y conservación de manifiestos por commit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sackmann_download import (  # noqa: E402
    ExistingFileConflictError,
    GITHUB_API_ROOT,
    PLAYER_FILES,
    RAW_CONTENT_ROOT,
    REQUIRED_RANKING_FILES,
    RemovedSourceFileError,
    SOURCE_REF,
    SOURCE_REPOSITORY,
    VERSIONED_MANIFEST_DIRECTORY,
    download_sackmann_data,
)


FIRST_COMMIT = "1" * 40
SECOND_COMMIT = "2" * 40


def _blob_sha(content: bytes) -> str:
    """Calcula para bytes el SHA-1 con el encabezado de blob de Git."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _minimum_valid_files() -> dict[str, bytes]:
    """Crea un inventario mínimo que satisface las familias inspeccionadas."""

    paths = {
        "atp/atp_matches_1968.csv",
        "atp/atp_matches_qual_chall_1978.csv",
        "atp/atp_matches_futures_1991.csv",
        "wta/wta_matches_1968.csv",
        "wta/wta_matches_qual_itf_1968.csv",
    }
    paths.update(PLAYER_FILES)
    paths.update(REQUIRED_RANKING_FILES)
    return {
        path: f"fixture for {path}\n".encode("utf-8")
        for path in sorted(paths)
    }


class FakeResponse:
    """Respuesta HTTP mínima compatible con las operaciones del descargador."""

    def __init__(
        self,
        *,
        json_payload: Mapping[str, Any] | None = None,
        body: bytes | None = None,
    ) -> None:
        """Guarda un payload JSON o un cuerpo binario para una petición falsa."""

        self._json_payload = json_payload
        self._body = body

    def raise_for_status(self) -> None:
        """Simula una respuesta HTTP correcta."""

    def json(self) -> Mapping[str, Any]:
        """Devuelve el objeto JSON configurado para la respuesta."""

        if self._json_payload is None:
            raise ValueError("La respuesta falsa no contiene JSON.")
        return self._json_payload

    def iter_content(self, chunk_size: int) -> Any:
        """Itera el cuerpo en fragmentos para imitar una descarga en streaming."""

        if self._body is None:
            raise AssertionError("La respuesta falsa no contiene bytes.")
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def __enter__(self) -> FakeResponse:
        """Abre el contexto de la respuesta falsa."""

        return self

    def __exit__(self, *args: object) -> None:
        """Cierra el contexto sin suprimir excepciones."""


class FakeGitHubSession:
    """Sesión determinista que sirve un snapshot GitHub enteramente en memoria."""

    def __init__(self, commit_sha: str, files: Mapping[str, bytes]) -> None:
        """Configura el commit, el árbol y los blobs disponibles."""

        self.commit_sha = commit_sha
        self.files = dict(files)
        self.requested_urls: list[str] = []
        self.raw_requests: list[str] = []
        self.closed = False

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        """Responde a las tres clases de URL utilizadas por el módulo."""

        self.requested_urls.append(url)
        commit_url = (
            f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/commits/{SOURCE_REF}"
        )
        tree_url = (
            f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/git/trees/"
            f"{self.commit_sha}?recursive=1"
        )
        raw_prefix = (
            f"{RAW_CONTENT_ROOT}/{SOURCE_REPOSITORY}/{self.commit_sha}/"
        )

        if url == commit_url:
            return FakeResponse(json_payload={"sha": self.commit_sha})
        if url == tree_url:
            tree = [
                {
                    "path": path,
                    "type": "blob",
                    "size": len(content),
                    "sha": _blob_sha(content),
                }
                for path, content in sorted(self.files.items())
            ]
            return FakeResponse(
                json_payload={"truncated": False, "tree": tree}
            )
        if url.startswith(raw_prefix):
            remote_path = url.removeprefix(raw_prefix)
            if remote_path not in self.files:
                raise AssertionError(f"Blob no configurado: {remote_path}")
            self.raw_requests.append(remote_path)
            return FakeResponse(body=self.files[remote_path])
        raise AssertionError(f"URL inesperada en la sesión falsa: {url}")

    def close(self) -> None:
        """Registra el cierre de la sesión falsa."""

        self.closed = True


class SackmannIncrementalDownloadTest(unittest.TestCase):
    """Comprueba sincronización incremental y manifiestos reproducibles."""

    def test_removed_managed_path_aborts_without_deleting_local_file(
        self,
    ) -> None:
        """Una ruta eliminada aguas arriba exige revisión y conserva lo local."""

        removed_path = "atp/atp_rankings_2025.csv"
        initial_files = _minimum_valid_files()
        initial_files[removed_path] = b"partition later removed upstream\n"
        reduced_files = dict(initial_files)
        del reduced_files[removed_path]

        with TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory)
            initial_report = download_sackmann_data(
                raw_dir=raw_dir,
                session=FakeGitHubSession(FIRST_COMMIT, initial_files),
            )
            active_manifest_before = initial_report.manifest_path.read_bytes()

            for force in (False, True):
                with self.subTest(force=force):
                    removal_session = FakeGitHubSession(
                        SECOND_COMMIT,
                        reduced_files,
                    )
                    with self.assertRaises(RemovedSourceFileError):
                        download_sackmann_data(
                            force=force,
                            raw_dir=raw_dir,
                            session=removal_session,
                        )

                    self.assertEqual(removal_session.raw_requests, [])
                    self.assertEqual(
                        (raw_dir / removed_path).read_bytes(),
                        initial_files[removed_path],
                    )
                    self.assertEqual(
                        initial_report.manifest_path.read_bytes(),
                        active_manifest_before,
                    )
                    self.assertFalse(
                        (
                            raw_dir
                            / VERSIONED_MANIFEST_DIRECTORY
                            / f"{SECOND_COMMIT}.json"
                        ).exists()
                    )

    def test_new_commit_downloads_only_added_and_modified_files(self) -> None:
        """Un commit nuevo conserva blobs iguales y baja solo sus diferencias."""

        initial_files = _minimum_valid_files()
        modified_path = "atp/atp_matches_1968.csv"
        added_path = "wta/wta_rankings_2026.csv"

        with TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory)
            initial_session = FakeGitHubSession(FIRST_COMMIT, initial_files)
            initial_report = download_sackmann_data(
                raw_dir=raw_dir,
                session=initial_session,
            )

            self.assertEqual(
                initial_report.downloaded_count,
                len(initial_files),
            )
            self.assertEqual(
                initial_report.manifest_path,
                raw_dir / "sackmann_manifest.json",
            )
            first_manifest_bytes = (
                initial_report.versioned_manifest_path.read_bytes()
            )

            # Simula la migración desde la disposición histórica, que solo
            # conservaba el manifiesto activo.
            initial_report.versioned_manifest_path.unlink()

            updated_files = dict(initial_files)
            updated_files[modified_path] = (
                b"x" * (len(initial_files[modified_path]) - 1) + b"\n"
            )
            updated_files[added_path] = b"new ranking partition\n"
            updated_session = FakeGitHubSession(SECOND_COMMIT, updated_files)
            report = download_sackmann_data(
                raw_dir=raw_dir,
                session=updated_session,
            )

            downloaded_relative = {
                path.relative_to(raw_dir).as_posix()
                for path in report.downloaded
            }
            self.assertEqual(
                downloaded_relative,
                {modified_path, added_path},
            )
            self.assertEqual(
                set(updated_session.raw_requests),
                {modified_path, added_path},
            )
            self.assertEqual(report.skipped_count, len(updated_files) - 2)
            self.assertEqual(
                (raw_dir / modified_path).read_bytes(),
                updated_files[modified_path],
            )

            first_versioned_path = (
                raw_dir
                / VERSIONED_MANIFEST_DIRECTORY
                / f"{FIRST_COMMIT}.json"
            )
            second_versioned_path = (
                raw_dir
                / VERSIONED_MANIFEST_DIRECTORY
                / f"{SECOND_COMMIT}.json"
            )
            self.assertEqual(
                first_versioned_path.read_bytes(),
                first_manifest_bytes,
            )
            self.assertTrue(second_versioned_path.is_file())
            self.assertEqual(report.versioned_manifest_path, second_versioned_path)

            active_payload = json.loads(report.manifest_path.read_text("utf-8"))
            versioned_payload = json.loads(
                second_versioned_path.read_text("utf-8")
            )
            self.assertEqual(active_payload["source_commit"], SECOND_COMMIT)
            self.assertEqual(
                versioned_payload["source_commit"],
                SECOND_COMMIT,
            )

    def test_local_divergence_fails_then_force_repairs_only_that_file(
        self,
    ) -> None:
        """Una edición local exige force, que no vuelve a bajar todo el snapshot."""

        source_files = _minimum_valid_files()
        divergent_path = "wta/wta_players.csv"

        with TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary_directory:
            raw_dir = Path(temporary_directory)
            initial_report = download_sackmann_data(
                raw_dir=raw_dir,
                session=FakeGitHubSession(FIRST_COMMIT, source_files),
            )
            immutable_manifest = (
                initial_report.versioned_manifest_path.read_bytes()
            )
            (raw_dir / divergent_path).write_bytes(b"local manual edit\n")

            conflict_session = FakeGitHubSession(FIRST_COMMIT, source_files)
            with self.assertRaises(ExistingFileConflictError):
                download_sackmann_data(
                    raw_dir=raw_dir,
                    session=conflict_session,
                )
            self.assertEqual(conflict_session.raw_requests, [])

            repair_session = FakeGitHubSession(FIRST_COMMIT, source_files)
            repaired_report = download_sackmann_data(
                force=True,
                raw_dir=raw_dir,
                session=repair_session,
            )

            self.assertEqual(repair_session.raw_requests, [divergent_path])
            self.assertEqual(repaired_report.downloaded_count, 1)
            self.assertEqual(
                repaired_report.skipped_count,
                len(source_files) - 1,
            )
            self.assertEqual(
                (raw_dir / divergent_path).read_bytes(),
                source_files[divergent_path],
            )
            self.assertEqual(
                repaired_report.versioned_manifest_path.read_bytes(),
                immutable_manifest,
            )

            no_op_session = FakeGitHubSession(FIRST_COMMIT, source_files)
            no_op_report = download_sackmann_data(
                force=True,
                raw_dir=raw_dir,
                session=no_op_session,
            )
            self.assertEqual(no_op_report.downloaded_count, 0)
            self.assertEqual(no_op_session.raw_requests, [])


if __name__ == "__main__":
    unittest.main()

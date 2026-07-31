"""Descarga snapshots reproducibles del Match Charting Project.

El módulo descubre todos los CSV, ``README.md`` y ``data_dictionary.txt`` del
repositorio oficial ``JeffSackmann/tennis_MatchChartingProject``. La rama
``master`` se resuelve primero a un commit inmutable y todos los archivos se
descargan desde ese SHA, nunca directamente desde una rama móvil.

Los datos se guardan en un directorio crudo exclusivo para esta fuente. No se
mezclan con los CSV históricos de partidos de Sackmann y este módulo no crea
loaders ni filas canónicas de partidos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import MATCH_CHARTING_RAW_DIR


LOGGER = logging.getLogger(__name__)

SOURCE_REPOSITORY: Final[str] = (
    "JeffSackmann/tennis_MatchChartingProject"
)
SOURCE_REF: Final[str] = "master"
SOURCE_WEB_URL: Final[str] = (
    "https://github.com/JeffSackmann/tennis_MatchChartingProject"
)

GITHUB_API_ROOT: Final[str] = "https://api.github.com"
RAW_CONTENT_ROOT: Final[str] = "https://raw.githubusercontent.com"
REQUEST_TIMEOUT: Final[tuple[float, float]] = (15.0, 180.0)
DOWNLOAD_CHUNK_SIZE: Final[int] = 1024 * 1024

SUPPORT_FILES: Final[frozenset[str]] = frozenset(
    {"README.md", "data_dictionary.txt"}
)
REQUIRED_FILES: Final[frozenset[str]] = frozenset(
    {
        "README.md",
        "data_dictionary.txt",
        "charting-m-matches.csv",
        "charting-w-matches.csv",
    }
)
ACTIVE_MANIFEST_FILENAME: Final[str] = "manifest.json"
VERSIONED_MANIFEST_DIRECTORY: Final[str] = "manifests"
MANIFEST_SCHEMA_VERSION: Final[int] = 1
LICENSE_ID: Final[str] = "CC BY-NC-SA 4.0"

_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


class MatchChartingDownloadError(RuntimeError):
    """Indica un fallo al descubrir, validar o descargar esta fuente."""


class SourceLayoutError(MatchChartingDownloadError):
    """Indica que GitHub no devolvió el inventario esperado."""


class LocalDataConflictError(MatchChartingDownloadError):
    """Indica que un archivo gestionado fue modificado localmente."""


class DownloadIntegrityError(MatchChartingDownloadError):
    """Indica que una descarga no coincide con el tamaño o blob remoto."""


class ManifestError(MatchChartingDownloadError):
    """Indica que un manifiesto local es inválido o contradictorio."""


@dataclass(frozen=True)
class SourceFile:
    """Describe un archivo seleccionado del árbol remoto."""

    remote_path: str
    size: int
    git_blob_sha: str

    @property
    def kind(self) -> str:
        """Devuelve ``csv`` o ``documentation`` según la ruta remota."""

        if self.remote_path.casefold().endswith(".csv"):
            return "csv"
        return "documentation"


@dataclass(frozen=True)
class SourceInventory:
    """Representa el commit, árbol y archivos de un snapshot remoto."""

    commit_sha: str
    tree_sha: str
    files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class DownloadReport:
    """Resume una descarga inicial o actualización incremental."""

    source_commit: str
    downloaded: tuple[Path, ...]
    updated: tuple[Path, ...]
    skipped: tuple[Path, ...]
    active_manifest_path: Path
    versioned_manifest_path: Path

    @property
    def downloaded_count(self) -> int:
        """Devuelve cuántos archivos nuevos se descargaron."""

        return len(self.downloaded)

    @property
    def updated_count(self) -> int:
        """Devuelve cuántos archivos conocidos cambiaron de blob."""

        return len(self.updated)

    @property
    def skipped_count(self) -> int:
        """Devuelve cuántos archivos ya coincidían con el snapshot."""

        return len(self.skipped)


@dataclass(frozen=True)
class PreviousManifest:
    """Contiene la procedencia mínima leída del manifiesto activo."""

    source_commit: str
    blob_shas: Mapping[str, str]


def build_http_session() -> requests.Session:
    """Crea una sesión HTTP identificada y con reintentos conservadores."""

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "TENNIS-Match-Charting-Ingestion/1",
        }
    )
    session.mount("https://", adapter)
    return session


def _request_json(
    session: requests.Session,
    url: str,
) -> Mapping[str, Any]:
    """Obtiene un objeto JSON y traduce fallos HTTP a errores de fuente."""

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MatchChartingDownloadError(
            f"No se pudo consultar la fuente remota: {url}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SourceLayoutError(
            f"La respuesta remota no es un objeto JSON: {url}"
        )
    return payload


def _require_sha(value: object, description: str) -> str:
    """Valida y devuelve un SHA-1 hexadecimal de Git."""

    if not isinstance(value, str) or _SHA_PATTERN.fullmatch(value) is None:
        raise SourceLayoutError(f"{description} no es un SHA Git válido.")
    return value


def _extract_tree_sha(commit_payload: Mapping[str, Any]) -> str:
    """Extrae el SHA del árbol desde la respuesta REST de un commit."""

    commit = commit_payload.get("commit")
    if not isinstance(commit, Mapping):
        raise SourceLayoutError(
            "La respuesta del commit no contiene el objeto commit."
        )
    tree = commit.get("tree")
    if not isinstance(tree, Mapping):
        raise SourceLayoutError(
            "La respuesta del commit no contiene el objeto commit.tree."
        )
    return _require_sha(tree.get("sha"), "El SHA del árbol")


def _is_selected_path(remote_path: str) -> bool:
    """Indica si la ruta es un CSV o uno de los dos documentos admitidos."""

    return (
        remote_path.casefold().endswith(".csv")
        or remote_path in SUPPORT_FILES
    )


def _validate_remote_path(remote_path: str) -> None:
    """Rechaza rutas absolutas, vacías o con segmentos ascendentes."""

    posix_path = PurePosixPath(remote_path)
    if (
        not remote_path
        or posix_path.is_absolute()
        or ".." in posix_path.parts
        or "\\" in remote_path
    ):
        raise SourceLayoutError(
            f"Ruta remota insegura en el inventario: {remote_path!r}."
        )


def _parse_tree_files(
    tree_payload: Mapping[str, Any],
) -> tuple[SourceFile, ...]:
    """Selecciona y valida todos los blobs CSV y documentos requeridos."""

    if tree_payload.get("truncated") is True:
        raise SourceLayoutError(
            "GitHub devolvió un árbol truncado; faltaría parte del inventario."
        )
    raw_entries = tree_payload.get("tree")
    if not isinstance(raw_entries, list):
        raise SourceLayoutError(
            "La respuesta del árbol no contiene una lista tree."
        )

    selected: list[SourceFile] = []
    observed_paths: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or entry.get("type") != "blob":
            continue
        remote_path = entry.get("path")
        if not isinstance(remote_path, str) or not _is_selected_path(
            remote_path
        ):
            continue
        _validate_remote_path(remote_path)
        if remote_path in observed_paths:
            raise SourceLayoutError(
                f"Ruta remota duplicada en el árbol: {remote_path!r}."
            )

        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SourceLayoutError(
                f"Tamaño remoto inválido para {remote_path!r}."
            )
        blob_sha = _require_sha(
            entry.get("sha"),
            f"El SHA de blob de {remote_path!r}",
        )
        observed_paths.add(remote_path)
        selected.append(
            SourceFile(
                remote_path=remote_path,
                size=size,
                git_blob_sha=blob_sha,
            )
        )

    missing = sorted(REQUIRED_FILES.difference(observed_paths))
    if missing:
        raise SourceLayoutError(
            f"Faltan archivos obligatorios del repositorio MCP: {missing}."
        )
    if not any(item.kind == "csv" for item in selected):
        raise SourceLayoutError("El inventario MCP no contiene ningún CSV.")

    return tuple(sorted(selected, key=lambda item: item.remote_path))


def fetch_source_inventory(
    session: requests.Session | None = None,
) -> SourceInventory:
    """Resuelve ``master`` a un commit y devuelve su inventario completo.

    Args:
        session: Sesión HTTP opcional para reutilización o pruebas offline.

    Returns:
        Snapshot inmutable con todos los CSV y los dos documentos admitidos.

    Raises:
        MatchChartingDownloadError: Si GitHub no se puede consultar.
        SourceLayoutError: Si el commit o árbol no tienen el formato esperado.
    """

    owned_session = session is None
    active_session = session or build_http_session()
    try:
        commit_url = (
            f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/commits/"
            f"{SOURCE_REF}"
        )
        commit_payload = _request_json(active_session, commit_url)
        commit_sha = _require_sha(
            commit_payload.get("sha"),
            "El SHA del commit",
        )
        tree_sha = _extract_tree_sha(commit_payload)

        tree_url = (
            f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/git/trees/"
            f"{tree_sha}?recursive=1"
        )
        tree_payload = _request_json(active_session, tree_url)
        files = _parse_tree_files(tree_payload)
        return SourceInventory(
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            files=files,
        )
    finally:
        if owned_session:
            active_session.close()


def _git_blob_sha(path: Path, size: int) -> str:
    """Calcula el SHA-1 que Git asigna al contenido de un archivo local."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(DOWNLOAD_CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(source_root: Path, remote_path: str) -> Path:
    """Resuelve una ruta remota sin permitir que escape del directorio fuente."""

    _validate_remote_path(remote_path)
    root = source_root.resolve()
    destination = (root / Path(remote_path)).resolve()
    if not destination.is_relative_to(root):
        raise SourceLayoutError(
            f"La ruta remota escapa del directorio fuente: {remote_path!r}."
        )
    return destination


def _read_json_object(path: Path) -> Mapping[str, Any]:
    """Lee un archivo JSON local y exige que su raíz sea un objeto."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"No se puede leer el manifiesto {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(f"El manifiesto {path} no contiene un objeto JSON.")
    return payload


def _load_previous_manifest(path: Path) -> PreviousManifest | None:
    """Carga el commit y los blobs del manifiesto activo, si existe."""

    if not path.exists():
        return None
    payload = _read_json_object(path)
    if payload.get("source_repository") != SOURCE_REPOSITORY:
        raise ManifestError(
            f"El manifiesto activo no pertenece a {SOURCE_REPOSITORY}: {path}."
        )
    if payload.get("source_ref") != SOURCE_REF:
        raise ManifestError(
            f"El manifiesto activo no corresponde a {SOURCE_REF}: {path}."
        )
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or _SHA_PATTERN.fullmatch(source_commit) is None
    ):
        raise ManifestError(
            f"El manifiesto activo tiene un commit inválido: {path}."
        )

    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ManifestError(
            f"El manifiesto activo no contiene una lista files: {path}."
        )
    blob_shas: dict[str, str] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise ManifestError(
                f"El manifiesto activo contiene una entrada inválida: {path}."
            )
        remote_path = raw_file.get("remote_path")
        blob_sha = raw_file.get("git_blob_sha")
        if not isinstance(remote_path, str):
            raise ManifestError(
                f"El manifiesto activo contiene una ruta inválida: {path}."
            )
        _validate_remote_path(remote_path)
        if (
            not isinstance(blob_sha, str)
            or _SHA_PATTERN.fullmatch(blob_sha) is None
        ):
            raise ManifestError(
                f"El manifiesto activo contiene un blob inválido: {path}."
            )
        if remote_path in blob_shas:
            raise ManifestError(
                f"El manifiesto activo repite {remote_path!r}: {path}."
            )
        blob_shas[remote_path] = blob_sha
    return PreviousManifest(
        source_commit=source_commit,
        blob_shas=blob_shas,
    )


def _local_blob_sha(path: Path) -> tuple[int, str]:
    """Devuelve el tamaño y SHA Git de un archivo local existente."""

    try:
        size = path.stat().st_size
        return size, _git_blob_sha(path, size)
    except OSError as exc:
        raise MatchChartingDownloadError(
            f"No se pudo validar el archivo local {path}."
        ) from exc


def _requires_download(
    destination: Path,
    source_file: SourceFile,
    previous_blob_sha: str | None,
) -> tuple[bool, bool]:
    """Decide si descargar y si la operación es una actualización.

    Returns:
        ``(requires_download, is_update)``. Un archivo idéntico se omite; un
        archivo que coincide con el manifiesto anterior puede actualizarse.

    Raises:
        LocalDataConflictError: Si el archivo local no coincide ni con el blob
            actual ni con el blob previamente gestionado.
    """

    if not destination.exists():
        return True, False
    if not destination.is_file():
        raise LocalDataConflictError(
            f"La ruta de destino existe pero no es un archivo: {destination}."
        )

    local_size, local_sha = _local_blob_sha(destination)
    if (
        local_size == source_file.size
        and local_sha == source_file.git_blob_sha
    ):
        return False, False
    if previous_blob_sha is not None and local_sha == previous_blob_sha:
        return True, True
    raise LocalDataConflictError(
        f"{destination} fue modificado localmente o carece de procedencia "
        "verificable; no se sobrescribirá."
    )


def _download_one(
    session: requests.Session,
    source_file: SourceFile,
    commit_sha: str,
    destination: Path,
) -> None:
    """Descarga un blob fijado a commit y valida tamaño y SHA antes de moverlo."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.part")
    encoded_path = quote(source_file.remote_path, safe="/")
    url = (
        f"{RAW_CONTENT_ROOT}/{SOURCE_REPOSITORY}/{commit_sha}/"
        f"{encoded_path}"
    )
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {source_file.size}\0".encode("ascii"))
    bytes_written = 0

    try:
        with session.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as target:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE
                ):
                    if not chunk:
                        continue
                    target.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)

        if bytes_written != source_file.size:
            raise DownloadIntegrityError(
                f"{source_file.remote_path}: se esperaban {source_file.size} "
                f"bytes y se recibieron {bytes_written}."
            )
        if digest.hexdigest() != source_file.git_blob_sha:
            raise DownloadIntegrityError(
                f"{source_file.remote_path}: el SHA Git descargado no coincide."
            )
        os.replace(temporary_path, destination)
    except (requests.RequestException, OSError) as exc:
        raise MatchChartingDownloadError(
            f"No se pudo descargar {source_file.remote_path}."
        ) from exc
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning(
                    "No se pudo retirar el temporal %s.",
                    temporary_path,
                )


def _build_manifest_payload(
    inventory: SourceInventory,
) -> dict[str, Any]:
    """Construye el manifiesto inmutable de un snapshot remoto."""

    csv_count = sum(item.kind == "csv" for item in inventory.files)
    documentation_count = len(inventory.files) - csv_count
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_repository": SOURCE_REPOSITORY,
        "source_web_url": SOURCE_WEB_URL,
        "source_ref": SOURCE_REF,
        "source_commit": inventory.commit_sha,
        "source_tree": inventory.tree_sha,
        "license": LICENSE_ID,
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "all_csv": True,
            "support_files": sorted(SUPPORT_FILES),
            "excluded_binary_workbook": True,
        },
        "files": [
            {
                "remote_path": source_file.remote_path,
                "local_path": source_file.remote_path,
                "kind": source_file.kind,
                "size": source_file.size,
                "git_blob_sha": source_file.git_blob_sha,
            }
            for source_file in inventory.files
        ],
        "summary": {
            "files": len(inventory.files),
            "csv": csv_count,
            "documentation": documentation_count,
        },
    }


def _manifest_file_signature(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, int, str], ...]:
    """Extrae una firma comparable de las entradas de un manifiesto."""

    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ManifestError("El manifiesto versionado no contiene files.")
    signature: list[tuple[str, int, str]] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ManifestError(
                "El manifiesto versionado contiene una entrada inválida."
            )
        remote_path = item.get("remote_path")
        size = item.get("size")
        blob_sha = item.get("git_blob_sha")
        if (
            not isinstance(remote_path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(blob_sha, str)
        ):
            raise ManifestError(
                "El manifiesto versionado contiene campos inválidos."
            )
        signature.append((remote_path, size, blob_sha))
    return tuple(signature)


def _validate_versioned_manifest(
    payload: Mapping[str, Any],
    inventory: SourceInventory,
    path: Path,
) -> None:
    """Comprueba que un manifiesto ya archivado describa el mismo snapshot."""

    expected_signature = tuple(
        (
            source_file.remote_path,
            source_file.size,
            source_file.git_blob_sha,
        )
        for source_file in inventory.files
    )
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("source_repository") != SOURCE_REPOSITORY
        or payload.get("source_ref") != SOURCE_REF
        or payload.get("source_commit") != inventory.commit_sha
        or payload.get("source_tree") != inventory.tree_sha
        or _manifest_file_signature(payload) != expected_signature
    ):
        raise ManifestError(
            f"El manifiesto versionado contradice el snapshot remoto: {path}."
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Escribe un objeto JSON mediante reemplazo atómico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ManifestError(f"No se pudo escribir el manifiesto {path}.") from exc
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning(
                    "No se pudo retirar el temporal %s.",
                    temporary_path,
                )


def _write_manifests(
    inventory: SourceInventory,
    source_root: Path,
) -> tuple[Path, Path]:
    """Publica el manifiesto activo y conserva uno inmutable por commit."""

    active_path = source_root / ACTIVE_MANIFEST_FILENAME
    versioned_path = (
        source_root
        / VERSIONED_MANIFEST_DIRECTORY
        / f"{inventory.commit_sha}.json"
    )

    if versioned_path.exists():
        payload = _read_json_object(versioned_path)
        _validate_versioned_manifest(payload, inventory, versioned_path)
    else:
        payload = _build_manifest_payload(inventory)
        _write_json_atomic(versioned_path, payload)

    _write_json_atomic(active_path, payload)
    return active_path, versioned_path


def download_match_charting_data(
    *,
    raw_dir: Path = MATCH_CHARTING_RAW_DIR,
    session: requests.Session | None = None,
) -> DownloadReport:
    """Descarga o actualiza incrementalmente el snapshot público del MCP.

    Los archivos nuevos se descargan. Los archivos que ya coinciden con el
    inventario se omiten. Cuando GitHub publica un blob nuevo, el archivo local
    se reemplaza automáticamente solo si todavía coincide con el blob anotado
    por el manifiesto activo anterior.

    Args:
        raw_dir: Directorio crudo exclusivo de esta fuente. Por defecto es
            ``data/raw/tennis_MatchChartingProject``.
        session: Sesión HTTP opcional para reutilización o pruebas offline.

    Returns:
        Informe con archivos nuevos, actualizados, omitidos y manifiestos.

    Raises:
        SourceLayoutError: Si el inventario remoto es inseguro o incompleto.
        LocalDataConflictError: Si un archivo local fue alterado.
        DownloadIntegrityError: Si tamaño o SHA no coinciden tras descargar.
        ManifestError: Si un manifiesto local es inválido o contradictorio.
    """

    source_root = raw_dir.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    active_manifest_path = source_root / ACTIVE_MANIFEST_FILENAME
    previous_manifest = _load_previous_manifest(active_manifest_path)

    owned_session = session is None
    active_session = session or build_http_session()
    downloaded: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []
    try:
        inventory = fetch_source_inventory(active_session)
        LOGGER.info(
            "MCP resuelto en %s con %d archivos.",
            inventory.commit_sha,
            len(inventory.files),
        )

        for position, source_file in enumerate(inventory.files, start=1):
            destination = _safe_destination(
                source_root,
                source_file.remote_path,
            )
            previous_sha = (
                previous_manifest.blob_shas.get(source_file.remote_path)
                if previous_manifest is not None
                else None
            )
            requires_download, is_update = _requires_download(
                destination,
                source_file,
                previous_sha,
            )
            if not requires_download:
                skipped.append(destination)
                LOGGER.info(
                    "[%d/%d] Conservado %s.",
                    position,
                    len(inventory.files),
                    source_file.remote_path,
                )
                continue

            LOGGER.info(
                "[%d/%d] %s %s.",
                position,
                len(inventory.files),
                "Actualizando" if is_update else "Descargando",
                source_file.remote_path,
            )
            _download_one(
                active_session,
                source_file,
                inventory.commit_sha,
                destination,
            )
            if is_update:
                updated.append(destination)
            else:
                downloaded.append(destination)

        active_path, versioned_path = _write_manifests(
            inventory,
            source_root,
        )
        return DownloadReport(
            source_commit=inventory.commit_sha,
            downloaded=tuple(downloaded),
            updated=tuple(updated),
            skipped=tuple(skipped),
            active_manifest_path=active_path,
            versioned_manifest_path=versioned_path,
        )
    finally:
        if owned_session:
            active_session.close()

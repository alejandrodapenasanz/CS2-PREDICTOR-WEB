"""Descarga incremental y reproducible de los CSV históricos de Jeff Sackmann.

La fuente operativa es el mirror autorizado ``Aneeshers/tennis-sackmann-archive``,
que conserva un snapshot de junio de 2026 de los repositorios originales
``JeffSackmann/tennis_atp`` y ``JeffSackmann/tennis_wta``. El módulo descubre el
árbol remoto antes de descargar, valida las familias y los años observados, y
preserva las rutas ``atp/`` y ``wta/`` bajo ``data/raw``. Cada actualización
compara blobs Git, conserva los no modificados y archiva un manifiesto inmutable
por commit además del manifiesto activo compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Final, Iterable, Mapping

import requests

from src.config import RAW_DATA_DIR
from src.responsible_http import ResponsibleHttpClient, build_http_client


LOGGER = logging.getLogger(__name__)

SOURCE_REPOSITORY: Final[str] = "Aneeshers/tennis-sackmann-archive"
SOURCE_REF: Final[str] = "main"
SOURCE_WEB_URL: Final[str] = "https://github.com/Aneeshers/tennis-sackmann-archive"
ORIGINAL_ATP_URL: Final[str] = "https://github.com/JeffSackmann/tennis_atp"
ORIGINAL_WTA_URL: Final[str] = "https://github.com/JeffSackmann/tennis_wta"
LICENSE_ID: Final[str] = "CC BY-NC-SA 4.0"

GITHUB_API_ROOT: Final[str] = "https://api.github.com"
RAW_CONTENT_ROOT: Final[str] = "https://raw.githubusercontent.com"
REQUEST_TIMEOUT: Final[tuple[float, float]] = (15.0, 180.0)
DOWNLOAD_CHUNK_SIZE: Final[int] = 1024 * 1024
ACTIVE_MANIFEST_FILENAME: Final[str] = "sackmann_manifest.json"
VERSIONED_MANIFEST_DIRECTORY: Final[str] = "sackmann_manifests"

MATCH_FAMILY_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "atp_main": re.compile(r"^atp/atp_matches_(?P<year>\d{4})\.csv$"),
    "atp_qual_chall": re.compile(r"^atp/atp_matches_qual_chall_(?P<year>\d{4})\.csv$"),
    "atp_futures": re.compile(r"^atp/atp_matches_futures_(?P<year>\d{4})\.csv$"),
    "wta_main": re.compile(r"^wta/wta_matches_(?P<year>\d{4})\.csv$"),
    "wta_qual_itf": re.compile(r"^wta/wta_matches_qual_itf_(?P<year>\d{4})\.csv$"),
}

EXPECTED_FIRST_YEAR: Final[dict[str, int]] = {
    "atp_main": 1968,
    "atp_qual_chall": 1978,
    "atp_futures": 1991,
    "wta_main": 1968,
    "wta_qual_itf": 1968,
}

PLAYER_FILES: Final[set[str]] = {
    "atp/atp_players.csv",
    "wta/wta_players.csv",
}

RANKING_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "atp_rankings": re.compile(r"^atp/atp_rankings_[^/]+\.csv$"),
    "wta_rankings": re.compile(r"^wta/wta_rankings_[^/]+\.csv$"),
}

REQUIRED_RANKING_FILES: Final[set[str]] = {
    "atp/atp_rankings_70s.csv",
    "atp/atp_rankings_80s.csv",
    "atp/atp_rankings_90s.csv",
    "atp/atp_rankings_00s.csv",
    "atp/atp_rankings_10s.csv",
    "atp/atp_rankings_20s.csv",
    "atp/atp_rankings_current.csv",
    "wta/wta_rankings_80s.csv",
    "wta/wta_rankings_90s.csv",
    "wta/wta_rankings_00s.csv",
    "wta/wta_rankings_10s.csv",
    "wta/wta_rankings_20s.csv",
    "wta/wta_rankings_current.csv",
}


class SackmannDownloadError(RuntimeError):
    """Indica que no se pudo descubrir, validar o descargar la fuente."""


class SourceLayoutError(SackmannDownloadError):
    """Indica que el árbol remoto no coincide con el formato inspeccionado."""


class RemovedSourceFileError(SourceLayoutError):
    """Indica que desaparecieron rutas gestionadas del inventario remoto."""


class ExistingFileConflictError(SackmannDownloadError):
    """Indica que un archivo local difiere y necesita una descarga forzada."""


class DownloadIntegrityError(SackmannDownloadError):
    """Indica que los bytes descargados no coinciden con el árbol de Git."""


@dataclass(frozen=True)
class SourceFile:
    """Describe un archivo remoto seleccionado para la ingesta."""

    remote_path: str
    size: int
    git_blob_sha: str
    category: str
    year: int | None = None


@dataclass(frozen=True)
class SourceInventory:
    """Representa el commit remoto resuelto y sus archivos seleccionados."""

    commit_sha: str
    files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class ManifestSnapshot:
    """Representa la identidad y los blobs de un manifiesto persistido."""

    source_commit: str | None
    file_shas: Mapping[str, str]
    payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class DownloadReport:
    """Resume el resultado de una ejecución idempotente de descarga."""

    source_commit: str
    downloaded: tuple[Path, ...]
    skipped: tuple[Path, ...]
    manifest_path: Path
    versioned_manifest_path: Path

    @property
    def downloaded_count(self) -> int:
        """Devuelve el número de archivos descargados en esta ejecución."""

        return len(self.downloaded)

    @property
    def skipped_count(self) -> int:
        """Devuelve el número de archivos ya presentes que se conservaron."""

        return len(self.skipped)


def build_http_session() -> ResponsibleHttpClient:
    """Crea una sesión HTTP identificada y con reintentos conservadores."""

    return build_http_client(
        default_headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "CS2-Predictor-TENNIS-Sackmann-Ingestion/2",
        },
        robots_user_agent="CS2-Predictor-TENNIS-Sackmann-Ingestion",
        request_retry_attempts=5,
        request_retry_delay_seconds=0.75,
        request_retry_statuses=(429, 500, 502, 503, 504),
    )


def _request_json(
    session: requests.Session,
    url: str,
) -> Mapping[str, Any]:
    """Obtiene un objeto JSON remoto y traduce errores HTTP a errores de fuente."""

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SackmannDownloadError(f"No se pudo consultar la fuente remota: {url}") from exc
    if not isinstance(payload, Mapping):
        raise SourceLayoutError(f"La respuesta de {url} no es un objeto JSON.")
    return payload


def _classify_remote_path(path: str) -> tuple[str, int | None] | None:
    """Clasifica una ruta remota admitida sin inferir patrones no observados."""

    for family, pattern in MATCH_FAMILY_PATTERNS.items():
        match = pattern.fullmatch(path)
        if match is not None:
            return family, int(match.group("year"))

    if path in PLAYER_FILES:
        return "players", None

    for category, pattern in RANKING_PATTERNS.items():
        if pattern.fullmatch(path) is not None:
            return category, None

    return None


def _validate_years(files: Iterable[SourceFile]) -> None:
    """Valida el primer año inspeccionado y cualquier hueco por familia."""

    for family, expected_first_year in EXPECTED_FIRST_YEAR.items():
        years = sorted(
            source_file.year
            for source_file in files
            if source_file.category == family and source_file.year is not None
        )
        if not years:
            raise SourceLayoutError(f"No se encontró ningún archivo para la familia {family!r}.")
        if years[0] != expected_first_year:
            raise SourceLayoutError(
                f"La familia {family!r} comienza en {years[0]}, pero la fuente "
                f"inspeccionada comenzaba en {expected_first_year}."
            )
        expected_years = set(range(years[0], years[-1] + 1))
        missing_years = sorted(expected_years.difference(years))
        if missing_years:
            raise SourceLayoutError(f"La familia {family!r} tiene años ausentes: {missing_years}.")


def _validate_inventory(files: tuple[SourceFile, ...]) -> None:
    """Comprueba familias, jugadores y archivos de rankings esperados."""

    paths = {source_file.remote_path for source_file in files}
    _validate_years(files)

    missing_players = sorted(PLAYER_FILES.difference(paths))
    if missing_players:
        raise SourceLayoutError(f"Faltan archivos de jugadores esperados: {missing_players}.")

    missing_rankings = sorted(REQUIRED_RANKING_FILES.difference(paths))
    if missing_rankings:
        raise SourceLayoutError(f"Faltan archivos de rankings esperados: {missing_rankings}.")


def fetch_source_inventory(
    session: requests.Session | None = None,
) -> SourceInventory:
    """Descubre el commit actual y selecciona solo los CSV de singles previstos."""

    owned_session = session is None
    active_session = session or build_http_session()
    try:
        commit_url = f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/commits/{SOURCE_REF}"
        commit_payload = _request_json(active_session, commit_url)
        commit_sha = commit_payload.get("sha")
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise SourceLayoutError("GitHub no devolvió un SHA de commit válido para el mirror.")

        tree_url = f"{GITHUB_API_ROOT}/repos/{SOURCE_REPOSITORY}/git/trees/{commit_sha}?recursive=1"
        tree_payload = _request_json(active_session, tree_url)
        if tree_payload.get("truncated") is True:
            raise SourceLayoutError("GitHub devolvió un árbol truncado; no es seguro ingerirlo.")
        raw_entries = tree_payload.get("tree")
        if not isinstance(raw_entries, list):
            raise SourceLayoutError("El árbol de GitHub no contiene una lista tree.")

        selected: list[SourceFile] = []
        for entry in raw_entries:
            if not isinstance(entry, Mapping) or entry.get("type") != "blob":
                continue
            path = entry.get("path")
            size = entry.get("size")
            blob_sha = entry.get("sha")
            if not isinstance(path, str):
                continue
            classification = _classify_remote_path(path)
            if classification is None:
                continue
            if not isinstance(size, int) or size < 0:
                raise SourceLayoutError(f"Tamaño remoto inválido para {path}.")
            if not isinstance(blob_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", blob_sha):
                raise SourceLayoutError(f"SHA de blob inválido para {path}.")
            category, year = classification
            selected.append(
                SourceFile(
                    remote_path=path,
                    size=size,
                    git_blob_sha=blob_sha,
                    category=category,
                    year=year,
                )
            )

        inventory_files = tuple(sorted(selected, key=lambda source_file: source_file.remote_path))
        _validate_inventory(inventory_files)
        return SourceInventory(commit_sha=commit_sha, files=inventory_files)
    finally:
        if owned_session:
            active_session.close()


def _git_blob_sha(path: Path, size: int) -> str:
    """Calcula el SHA-1 de blob empleado por Git para un archivo local."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(raw_dir: Path, remote_path: str) -> Path:
    """Resuelve una ruta remota y evita que pueda escapar de ``raw_dir``."""

    root = raw_dir.resolve()
    destination = (root / Path(remote_path)).resolve()
    if not destination.is_relative_to(root):
        raise SourceLayoutError(f"La ruta remota intenta salir de data/raw: {remote_path!r}.")
    return destination


def _load_previous_manifest(manifest_path: Path) -> ManifestSnapshot:
    """Carga y valida la identidad de un manifiesto local anterior."""

    if not manifest_path.exists():
        return ManifestSnapshot(
            source_commit=None,
            file_shas={},
            payload=None,
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExistingFileConflictError(
            f"El manifiesto local no se puede leer: {manifest_path}."
        ) from exc

    if not isinstance(payload, Mapping):
        raise ExistingFileConflictError(
            f"El manifiesto local no es un objeto JSON: {manifest_path}."
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ExistingFileConflictError(
            f"El manifiesto local no contiene un source_commit válido: {manifest_path}."
        )

    files = payload.get("files")
    if not isinstance(files, list):
        raise ExistingFileConflictError(
            f"El manifiesto local no contiene una lista files: {manifest_path}."
        )
    result: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ExistingFileConflictError(
                f"El manifiesto contiene una entrada files inválida: {manifest_path}."
            )
        remote_path = item.get("remote_path")
        blob_sha = item.get("git_blob_sha")
        if not isinstance(remote_path, str) or not isinstance(blob_sha, str):
            raise ExistingFileConflictError(
                f"El manifiesto contiene una ruta o SHA inválidos: {manifest_path}."
            )
        if re.fullmatch(r"[0-9a-f]{40}", blob_sha) is None:
            raise ExistingFileConflictError(
                f"El manifiesto contiene un SHA de blob inválido para "
                f"{remote_path!r}: {manifest_path}."
            )
        if remote_path in result:
            raise ExistingFileConflictError(
                f"El manifiesto repite la ruta {remote_path!r}: {manifest_path}."
            )
        result[remote_path] = blob_sha
    return ManifestSnapshot(
        source_commit=source_commit,
        file_shas=result,
        payload=payload,
    )


def _should_download_file(
    destination: Path,
    source_file: SourceFile,
    previous_sha: str | None,
    *,
    force: bool,
) -> bool:
    """Decide si descargar tras verificar la integridad histórica local.

    El SHA del manifiesto activo anterior es la referencia para detectar una
    edición local. Un cambio del blob remoto no es un conflicto: se descarga de
    forma incremental. ``force`` solo autoriza reparar divergencias locales y
    no obliga a descargar archivos cuyo contenido ya sea el blob remoto.
    """

    if not destination.exists():
        return True
    if not destination.is_file():
        raise ExistingFileConflictError(f"{destination} existe, pero no es un archivo regular.")

    local_size = destination.stat().st_size
    local_sha = _git_blob_sha(destination, local_size)
    if previous_sha is not None:
        if local_sha != previous_sha:
            if not force:
                raise ExistingFileConflictError(
                    f"{destination} diverge del SHA registrado en el manifiesto "
                    "activo. Ejecuta con --force para repararlo."
                )
            return local_sha != source_file.git_blob_sha
        return previous_sha != source_file.git_blob_sha

    if local_sha == source_file.git_blob_sha:
        return False
    if not force:
        raise ExistingFileConflictError(
            f"{destination} ya existe sin una referencia previa compatible y "
            "no coincide con la fuente. Ejecuta con --force para reemplazarlo."
        )
    return True


def _download_one(
    session: requests.Session,
    source_file: SourceFile,
    commit_sha: str,
    destination: Path,
) -> None:
    """Descarga un archivo a una ruta temporal y verifica tamaño y SHA Git."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.part")
    url = f"{RAW_CONTENT_ROOT}/{SOURCE_REPOSITORY}/{commit_sha}/{source_file.remote_path}"
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {source_file.size}\0".encode("ascii"))
    bytes_written = 0

    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as target:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    target.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
        if bytes_written != source_file.size:
            raise DownloadIntegrityError(
                f"{source_file.remote_path}: se esperaban {source_file.size} bytes "
                f"y se recibieron {bytes_written}."
            )
        if digest.hexdigest() != source_file.git_blob_sha:
            raise DownloadIntegrityError(
                f"{source_file.remote_path}: el SHA del contenido no coincide."
            )
        os.replace(temporary_path, destination)
    except (requests.RequestException, OSError) as exc:
        raise SackmannDownloadError(f"No se pudo descargar {source_file.remote_path}.") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _versioned_manifest_path(raw_dir: Path, commit_sha: str) -> Path:
    """Devuelve la ruta estable del manifiesto correspondiente a un commit."""

    if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
        raise SourceLayoutError(f"No se puede versionar un SHA de commit inválido: {commit_sha!r}.")
    return raw_dir / VERSIONED_MANIFEST_DIRECTORY / f"{commit_sha}.json"


def _inventory_file_shas(inventory: SourceInventory) -> dict[str, str]:
    """Construye el mapa ruta-SHA que identifica los archivos de un inventario."""

    return {source_file.remote_path: source_file.git_blob_sha for source_file in inventory.files}


def _validate_no_removed_source_files(
    previous_manifest: ManifestSnapshot,
    inventory: SourceInventory,
) -> None:
    """Rechaza rutas gestionadas que desaparecen, sin borrar archivos locales."""

    current_paths = set(_inventory_file_shas(inventory))
    removed_paths = sorted(set(previous_manifest.file_shas).difference(current_paths))
    if removed_paths:
        formatted_paths = ", ".join(repr(path) for path in removed_paths)
        raise RemovedSourceFileError(
            "El inventario remoto ha eliminado rutas gestionadas por el "
            f"manifiesto activo: {formatted_paths}. No se ha borrado ningún "
            "archivo local; la actualización requiere revisión explícita."
        )


def _validate_manifest_identity(
    snapshot: ManifestSnapshot,
    *,
    expected_commit: str,
    expected_shas: Mapping[str, str],
    manifest_path: Path,
) -> None:
    """Impide reutilizar una ruta versionada para contenido de otro snapshot."""

    if snapshot.source_commit != expected_commit:
        raise ExistingFileConflictError(
            f"El manifiesto versionado {manifest_path} declara el commit "
            f"{snapshot.source_commit!r}, no {expected_commit!r}."
        )
    if dict(snapshot.file_shas) != dict(expected_shas):
        raise ExistingFileConflictError(
            f"El manifiesto versionado {manifest_path} no coincide con los "
            "blobs esperados para su commit."
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Escribe un objeto JSON mediante reemplazo atómico en su destino."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _build_manifest_payload(
    inventory: SourceInventory,
    raw_dir: Path,
    downloaded: tuple[Path, ...],
    skipped: tuple[Path, ...],
) -> dict[str, Any]:
    """Construye el manifiesto compatible con procedencia, blobs y resumen."""

    downloaded_set = {path.resolve() for path in downloaded}
    return {
        "source_repository": SOURCE_REPOSITORY,
        "source_web_url": SOURCE_WEB_URL,
        "source_ref": SOURCE_REF,
        "source_commit": inventory.commit_sha,
        "original_sources": {
            "atp": ORIGINAL_ATP_URL,
            "wta": ORIGINAL_WTA_URL,
        },
        "license": LICENSE_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "files": [
            {
                "remote_path": source_file.remote_path,
                "local_path": source_file.remote_path,
                "category": source_file.category,
                "year": source_file.year,
                "size": source_file.size,
                "git_blob_sha": source_file.git_blob_sha,
                "status": (
                    "downloaded"
                    if _safe_destination(raw_dir, source_file.remote_path).resolve()
                    in downloaded_set
                    else "skipped"
                ),
            }
            for source_file in inventory.files
        ],
        "summary": {
            "selected": len(inventory.files),
            "downloaded": len(downloaded),
            "skipped": len(skipped),
        },
    }


def _archive_manifest_if_needed(
    snapshot: ManifestSnapshot,
    raw_dir: Path,
) -> Path | None:
    """Archiva un manifiesto activo previo sin sobrescribir snapshots existentes."""

    if snapshot.payload is None or snapshot.source_commit is None:
        return None
    archive_path = _versioned_manifest_path(raw_dir, snapshot.source_commit)
    if archive_path.exists():
        archived_snapshot = _load_previous_manifest(archive_path)
        _validate_manifest_identity(
            archived_snapshot,
            expected_commit=snapshot.source_commit,
            expected_shas=snapshot.file_shas,
            manifest_path=archive_path,
        )
        return archive_path
    _write_json_atomic(archive_path, snapshot.payload)
    return archive_path


def _write_manifests(
    inventory: SourceInventory,
    raw_dir: Path,
    downloaded: tuple[Path, ...],
    skipped: tuple[Path, ...],
    previous_manifest: ManifestSnapshot,
) -> tuple[Path, Path]:
    """Conserva snapshots por commit y actualiza el manifiesto activo compatible."""

    active_path = raw_dir / ACTIVE_MANIFEST_FILENAME
    versioned_path = _versioned_manifest_path(raw_dir, inventory.commit_sha)
    payload = _build_manifest_payload(
        inventory,
        raw_dir,
        downloaded,
        skipped,
    )

    _archive_manifest_if_needed(previous_manifest, raw_dir)
    if versioned_path.exists():
        versioned_snapshot = _load_previous_manifest(versioned_path)
        _validate_manifest_identity(
            versioned_snapshot,
            expected_commit=inventory.commit_sha,
            expected_shas=_inventory_file_shas(inventory),
            manifest_path=versioned_path,
        )
    else:
        _write_json_atomic(versioned_path, payload)

    _write_json_atomic(active_path, payload)
    return active_path, versioned_path


def download_sackmann_data(
    *,
    force: bool = False,
    raw_dir: Path = RAW_DATA_DIR,
    session: requests.Session | None = None,
) -> DownloadReport:
    """Sincroniza los CSV previstos por diferencias de blob respecto al manifiesto.

    Args:
        force: Autoriza reparar archivos que divergen del manifiesto anterior;
            no fuerza la descarga de archivos cuyo blob local ya es correcto.
        raw_dir: Directorio raíz donde se preservarán ``atp/`` y ``wta/``.
        session: Sesión HTTP opcional para reutilización o tests.

    Returns:
        Un resumen inmutable con descargas, omisiones y ambos manifiestos.

    Raises:
        SourceLayoutError: Si faltan familias, años o archivos esperados.
        ExistingFileConflictError: Si un archivo local difiere sin ``force``.
        DownloadIntegrityError: Si un archivo descargado falla la integridad.
    """

    raw_root = raw_dir.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_root / ACTIVE_MANIFEST_FILENAME
    previous_manifest = _load_previous_manifest(manifest_path)

    owned_session = session is None
    active_session = session or build_http_session()
    downloaded: list[Path] = []
    skipped: list[Path] = []
    try:
        inventory = fetch_source_inventory(active_session)
        LOGGER.info(
            "Fuente resuelta en %s con %d archivos seleccionados.",
            inventory.commit_sha,
            len(inventory.files),
        )
        _validate_no_removed_source_files(previous_manifest, inventory)
        versioned_path = _versioned_manifest_path(
            raw_root,
            inventory.commit_sha,
        )
        if versioned_path.exists():
            _validate_manifest_identity(
                _load_previous_manifest(versioned_path),
                expected_commit=inventory.commit_sha,
                expected_shas=_inventory_file_shas(inventory),
                manifest_path=versioned_path,
            )

        pending_downloads: list[tuple[SourceFile, Path]] = []
        for source_file in inventory.files:
            destination = _safe_destination(raw_root, source_file.remote_path)
            if _should_download_file(
                destination,
                source_file,
                previous_manifest.file_shas.get(source_file.remote_path),
                force=force,
            ):
                pending_downloads.append((source_file, destination))
            else:
                skipped.append(destination)

        for position, (source_file, destination) in enumerate(
            pending_downloads,
            start=1,
        ):
            LOGGER.info(
                "[%d/%d] Descargando %s",
                position,
                len(pending_downloads),
                source_file.remote_path,
            )
            _download_one(
                active_session,
                source_file,
                inventory.commit_sha,
                destination,
            )
            downloaded.append(destination)

        downloaded_tuple = tuple(downloaded)
        skipped_tuple = tuple(skipped)
        written_manifest, written_versioned_manifest = _write_manifests(
            inventory,
            raw_root,
            downloaded_tuple,
            skipped_tuple,
            previous_manifest,
        )
        return DownloadReport(
            source_commit=inventory.commit_sha,
            downloaded=downloaded_tuple,
            skipped=skipped_tuple,
            manifest_path=written_manifest,
            versioned_manifest_path=written_versioned_manifest,
        )
    finally:
        if owned_session:
            active_session.close()

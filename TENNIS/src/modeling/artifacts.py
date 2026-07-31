"""Persistencia inmutable y verificada de modelos, métricas y calibradores.

Cada combinación de fuentes, parámetros, versiones y código produce un
fingerprint. Los archivos se escriben primero en staging dentro de
``TENNIS/models`` y el directorio inmutable del run se publica en una única
operación. El manifiesto activo se reemplaza únicamente después de verificar
todos los tamaños y SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Mapping, Sequence

import joblib
import lightgbm
import numpy
import pandas
import pyarrow
import sklearn

from ..config import PHASE7_MODELS_DIR, PROJECT_ROOT
from .data import sha256_file


_MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402


class ArtifactError(RuntimeError):
    """Indica que un run no se puede publicar o verificar."""


@dataclass(frozen=True, slots=True)
class PublishedRun:
    """Identidad y rutas de un run publicado o reutilizado."""

    fingerprint: str
    run_dir: Path
    manifest_path: Path
    skipped: bool
    manifest: Mapping[str, object]


def _ensure_project_path(path: Path, field_name: str) -> Path:
    """Resuelve una ruta y la restringe al árbol ``TENNIS/``."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ArtifactError(
            f"{field_name} debe permanecer dentro de TENNIS/: {resolved}."
        )
    return resolved


def runtime_versions() -> Mapping[str, str]:
    """Devuelve las versiones que afectan serialización y predicción."""

    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serializa un mapping con orden y separadores deterministas."""

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "El payload del fingerprint no es JSON canónico."
        ) from exc
    return text.encode("utf-8")


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    """Calcula SHA-256 sobre una configuración JSON canónica."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def modeling_source_inventory(
    *,
    extra_paths: Sequence[Path] = (),
) -> tuple[Mapping[str, object], ...]:
    """Hashea el código de modelado y scripts explícitos del run."""

    modeling_dir = PROJECT_ROOT / "src" / "modeling"
    paths = sorted(modeling_dir.glob("*.py"))
    paths.extend(Path(path) for path in extra_paths)
    inventory: list[Mapping[str, object]] = []
    seen: set[Path] = set()
    for candidate in paths:
        resolved = _ensure_project_path(candidate, "source_path")
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise ArtifactError(f"No existe el código fuente {resolved}.")
        inventory.append(
            {
                "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if not inventory:
        raise ArtifactError("No se encontró código para versionar el run.")
    return tuple(inventory)


def create_staging_directory(
    output_dir: Path = PHASE7_MODELS_DIR,
) -> Path:
    """Crea un staging único dentro del directorio de modelos."""

    resolved = _ensure_project_path(output_dir, "output_dir")
    resolved.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".staging-", dir=resolved)
    return Path(temporary).resolve()


def dump_joblib_artifact(value: object, path: Path) -> None:
    """Serializa un objeto con compresión moderada dentro de staging."""

    resolved = _ensure_project_path(path, "artifact_path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(value, resolved, compress=3)
    except Exception as exc:
        raise ArtifactError(
            f"No se pudo serializar el artefacto {resolved}."
        ) from exc


def write_json_artifact(
    payload: Mapping[str, object],
    path: Path,
) -> None:
    """Escribe JSON UTF-8 legible, rechazando NaN no estándar."""

    resolved = _ensure_project_path(path, "json_path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        resolved.write_text(text + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"No se pudo escribir {resolved}.") from exc


def _safe_relative_path(root: Path, relative: str) -> Path:
    """Resuelve una ruta POSIX relativa sin traversal."""

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactError(f"Ruta de artefacto insegura: {relative!r}.")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ArtifactError(f"El artefacto sale del run: {relative!r}.")
    return resolved


def _file_inventory(
    root: Path,
    *,
    exclude: Sequence[str] = ("manifest.json",),
) -> tuple[Mapping[str, object], ...]:
    """Inventaría todos los archivos de un staging en orden estable."""

    excluded = set(exclude)
    rows: list[Mapping[str, object]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ArtifactError("El staging no contiene artefactos publicables.")
    return tuple(rows)


def verify_published_run(run_dir: Path) -> Mapping[str, object]:
    """Verifica manifiesto, fingerprint y cada archivo de un run."""

    resolved = _ensure_project_path(run_dir, "run_dir")
    manifest_path = resolved / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"No se pudo leer el manifiesto del run {resolved}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("El manifiesto del run debe ser un objeto JSON.")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("El fingerprint publicado no es válido.")
    if resolved.name != fingerprint:
        raise ArtifactError(
            "El nombre del run no coincide con su fingerprint."
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError("El manifiesto del run no inventaría archivos.")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ArtifactError("Una entrada files no es un objeto.")
        relative = item.get("path")
        expected_size = item.get("size")
        expected_sha = item.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in seen
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ArtifactError("Entrada files inválida o duplicada.")
        seen.add(relative)
        path = _safe_relative_path(resolved, relative)
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ArtifactError(f"Falta o cambió el tamaño de {relative}.")
        if sha256_file(path) != expected_sha:
            raise ArtifactError(f"SHA-256 divergente para {relative}.")
    return payload


def find_verified_run(
    fingerprint: str,
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
) -> PublishedRun | None:
    """Devuelve un run íntegro ya existente o ``None``."""

    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ArtifactError("fingerprint debe ser SHA-256 hexadecimal.")
    resolved = _ensure_project_path(output_dir, "output_dir")
    run_dir = resolved / "runs" / fingerprint
    if not run_dir.exists():
        return None
    payload = verify_published_run(run_dir)
    return PublishedRun(
        fingerprint=fingerprint,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        skipped=True,
        manifest=payload,
    )


def _publish_active_manifest(
    payload: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Reemplaza atómicamente el puntero activo tras publicar el run."""

    active_path = output_dir / "manifest.json"
    temporary = output_dir / ".manifest.tmp"
    write_json_artifact(payload, temporary)
    try:
        os.replace(temporary, active_path)
    except OSError as exc:
        raise ArtifactError(
            f"No se pudo activar el manifiesto {active_path}."
        ) from exc
    return active_path


def publish_staged_run(
    staging_dir: Path,
    *,
    fingerprint: str,
    manifest_payload: Mapping[str, object],
    output_dir: Path = PHASE7_MODELS_DIR,
) -> PublishedRun:
    """Publica un staging inmutable y activa su manifiesto.

    El caller no debe reutilizar ``staging_dir`` después de una publicación
    correcta: el directorio se mueve a ``runs/<fingerprint>``.
    """

    staging = _ensure_project_path(staging_dir, "staging_dir")
    output = _ensure_project_path(output_dir, "output_dir")
    if staging.parent != output:
        raise ArtifactError("staging_dir no pertenece al output_dir.")
    if not staging.name.startswith(".staging-") or not staging.is_dir():
        raise ArtifactError("staging_dir no es un staging válido.")
    if find_verified_run(fingerprint, output_dir=output) is not None:
        raise ArtifactError(
            "El fingerprint ya está publicado; no se sobrescribe."
        )
    inventory = _file_inventory(staging)
    payload = dict(manifest_payload)
    payload["fingerprint"] = fingerprint
    payload["files"] = list(inventory)
    write_json_artifact(payload, staging / "manifest.json")

    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / fingerprint
    if target.exists():
        raise ArtifactError(f"El destino inmutable ya existe: {target}.")
    try:
        os.replace(staging, target)
    except OSError as exc:
        raise ArtifactError(f"No se pudo publicar {target}.") from exc
    verified = verify_published_run(target)
    active_payload = dict(verified)
    active_payload["active_run"] = (
        target.relative_to(output).as_posix()
    )
    active_path = _publish_active_manifest(active_payload, output)
    return PublishedRun(
        fingerprint=fingerprint,
        run_dir=target,
        manifest_path=active_path,
        skipped=False,
        manifest=verified,
    )

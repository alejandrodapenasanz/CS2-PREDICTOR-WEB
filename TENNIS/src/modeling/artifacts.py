"""Persistencia inmutable y verificada de modelos, métricas y calibradores.

Cada combinación de fuentes, parámetros, versiones y código produce un
fingerprint. Los archivos se escriben primero en staging dentro de
``TENNIS/models`` y el directorio inmutable del run se publica en una única
operación. El manifiesto activo se reemplaza únicamente después de verificar
todos los tamaños y SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Final, Mapping, Sequence
import uuid

import joblib
import lightgbm
import numpy
import pandas
import pyarrow
import sklearn

from ..config import PHASE7_MODELS_DIR, PROJECT_ROOT
from ..artifact_integrity import (
    CodeInventoryError,
    build_code_inventory,
    verify_code_inventory,
)
from .data import sha256_file


_MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402


# Lista deliberadamente explicita: un archivo nuevo no entra en produccion
# por el efecto lateral de un glob. Cualquier cambio en entrenamiento,
# serializacion o inferencia diaria invalida el fingerprint del modelo.
MODEL_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/artifact_integrity.py",
    "src/temporal.py",
    "src/modeling/__init__.py",
    "src/modeling/artifacts.py",
    "src/modeling/backtest.py",
    "src/modeling/baselines.py",
    "src/modeling/calibration.py",
    "src/modeling/data.py",
    "src/modeling/estimators.py",
    "src/modeling/evaluation.py",
    "src/modeling/metrics.py",
    "src/modeling/orientation.py",
    "src/modeling/parameters.py",
    "src/modeling/plots.py",
    "src/modeling/preprocessing.py",
    "src/modeling/promotion.py",
    "src/modeling/reporting.py",
    "src/modeling/retention.py",
    "src/modeling/service.py",
    "src/modeling/splits.py",
    "src/modeling/training.py",
    "scripts/retrain_models.py",
    "src/daily_pipeline/__init__.py",
    "src/daily_pipeline/confidence.py",
    "src/daily_pipeline/context.py",
    "src/daily_pipeline/pipeline.py",
    "src/daily_pipeline/tennisratio_overlay.py",
)

# Solo este subconjunto afecta la deserialización y la transformación
# matemática dentro del bundle. La construcción causal de inputs y la
# orquestación diaria conservan sus propios contratos/fingerprints; incluirlas
# aquí inutilizaba un champion aunque la puerta demostrase predicciones
# idénticas sobre el nuevo artefacto de features.
MODEL_INFERENCE_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/temporal.py",
    "src/modeling/baselines.py",
    "src/modeling/calibration.py",
    "src/modeling/estimators.py",
    "src/modeling/orientation.py",
    "src/modeling/parameters.py",
    "src/modeling/preprocessing.py",
    "src/modeling/service.py",
)


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
        raise ArtifactError(f"{field_name} debe permanecer dentro de TENNIS/: {resolved}.")
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
        raise ArtifactError("El payload del fingerprint no es JSON canónico.") from exc
    return text.encode("utf-8")


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    """Calcula SHA-256 sobre una configuración JSON canónica."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def modeling_source_inventory(
    *,
    extra_paths: Sequence[Path] = (),
) -> tuple[Mapping[str, object], ...]:
    """Hashea el contrato explícito de entrenamiento e inferencia.

    ``extra_paths`` se conserva por compatibilidad con la API de reentreno,
    pero solo admite rutas ya incluidas en :data:`MODEL_CODE_PATHS`. Así el
    consumidor conoce de antemano exactamente qué debe volver a verificar.
    """

    allowed = frozenset(MODEL_CODE_PATHS)
    for candidate in extra_paths:
        resolved = _ensure_project_path(candidate, "source_path")
        relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        if relative not in allowed:
            raise ArtifactError(
                f"source_path no pertenece al contrato explícito de código: {relative}."
            )
    try:
        return tuple(build_code_inventory(PROJECT_ROOT, MODEL_CODE_PATHS))
    except CodeInventoryError as exc:
        raise ArtifactError("No se pudo construir el inventario explícito del modelo.") from exc


def verify_model_code_inventory(
    persisted_inventory: object,
) -> tuple[dict[str, object], ...]:
    """Exige compatibilidad exacta del código que ejecuta el bundle.

    El inventario amplio define la identidad del entrenamiento. Para cargar un
    modelo promovido se verifica el subconjunto de inferencia, evitando que un
    cambio exclusivo de la puerta fuerce la activación artificial de un empate.
    """

    if not isinstance(persisted_inventory, list):
        raise ArtifactError("El inventario persistido del modelo no es una lista.")
    by_path = {
        item.get("path"): item
        for item in persisted_inventory
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    missing = sorted(set(MODEL_INFERENCE_CODE_PATHS).difference(by_path))
    if missing:
        raise ArtifactError(
            "El run no acredita todo el contrato de inferencia: " + ", ".join(missing)
        )
    inference_inventory = [by_path[path] for path in sorted(MODEL_INFERENCE_CODE_PATHS)]
    try:
        return verify_code_inventory(
            PROJECT_ROOT,
            MODEL_INFERENCE_CODE_PATHS,
            inference_inventory,
        )
    except CodeInventoryError as exc:
        raise ArtifactError(
            "El run fue producido por código distinto del runtime actual; "
            "reentrene antes de cargarlo."
        ) from exc


def create_staging_directory(
    output_dir: Path = PHASE7_MODELS_DIR,
) -> Path:
    """Crea un staging único dentro del directorio de modelos."""

    resolved = _ensure_project_path(output_dir, "output_dir")
    resolved.mkdir(parents=True, exist_ok=True)
    # ``tempfile.mkdtemp`` crea una DACL privada (0o700) en Windows/Python
    # 3.13. Como el staging completo se renombra al publicar, esa DACL haria
    # ilegible el modelo para procesos posteriores. ``mkdir`` hereda el ACL.
    for _ in range(32):
        candidate = resolved / f".staging-{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise ArtifactError("No se pudo crear el staging heredable del modelo.") from exc
        return candidate.resolve()
    raise ArtifactError("No se pudo reservar un nombre unico para el staging del modelo.")


def dump_joblib_artifact(value: object, path: Path) -> None:
    """Serializa un objeto con compresión moderada dentro de staging."""

    resolved = _ensure_project_path(path, "artifact_path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(value, resolved, compress=3)
    except Exception as exc:
        raise ArtifactError(f"No se pudo serializar el artefacto {resolved}.") from exc


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
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
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
        raise ArtifactError(f"No se pudo leer el manifiesto del run {resolved}.") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("El manifiesto del run debe ser un objeto JSON.")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("El fingerprint publicado no es válido.")
    if resolved.name != fingerprint:
        raise ArtifactError("El nombre del run no coincide con su fingerprint.")
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


def _validate_model_fingerprint(value: object) -> str:
    """Valida la identidad hexadecimal usada como nombre de un run."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError("fingerprint debe ser SHA-256 hexadecimal.")
    return value


def _parse_run_created_at(value: object, *, run_dir: Path) -> datetime:
    """Lee el instante consciente que ordena generaciones inmutables."""

    if not isinstance(value, str):
        raise ArtifactError(f"El run {run_dir.name} no declara created_at_utc.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactError(f"created_at_utc no es ISO válido en {run_dir.name}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactError(f"created_at_utc debe incluir zona horaria en {run_dir.name}.")
    return parsed.astimezone(UTC)


def find_verified_run(
    fingerprint: str,
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
) -> PublishedRun | None:
    """Devuelve un run íntegro ya existente o ``None``."""

    fingerprint = _validate_model_fingerprint(fingerprint)
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
        raise ArtifactError(f"No se pudo activar el manifiesto {active_path}.") from exc
    return active_path


def _activate_gate_approved_run(
    published: PublishedRun,
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
    expected_active_sha256: str | None,
) -> PublishedRun:
    """Conmuta el puntero tras una aprobación emitida por ``promotion``.

    Es deliberadamente privada: entrenamiento solo registra challengers. El
    hash esperado implementa compare-and-swap y evita aplicar una evaluación
    obsoleta si el champion cambió mientras se puntuaba.
    """

    if not isinstance(published, PublishedRun):
        raise TypeError("published debe ser PublishedRun.")
    output = _ensure_project_path(output_dir, "output_dir")
    run_dir = _ensure_project_path(published.run_dir, "run_dir")
    expected_parent = output / "runs"
    if run_dir.parent != expected_parent:
        raise ArtifactError("El run reutilizado no pertenece al output_dir.")
    verified = verify_published_run(run_dir)
    fingerprint = verified.get("fingerprint")
    if fingerprint != published.fingerprint:
        raise ArtifactError("El fingerprint reutilizado no coincide con su manifiesto.")
    current_pointer = output / "manifest.json"
    if expected_active_sha256 is None:
        if current_pointer.exists():
            raise ArtifactError("Ya existe un champion; el bootstrap fue rechazado.")
    elif not current_pointer.is_file() or sha256_file(current_pointer) != expected_active_sha256:
        raise ArtifactError("El champion cambió durante la evaluación.")
    active_payload = dict(verified)
    active_payload["active_run"] = run_dir.relative_to(output).as_posix()
    active_path = _publish_active_manifest(active_payload, output)
    return PublishedRun(
        fingerprint=published.fingerprint,
        run_dir=run_dir,
        manifest_path=active_path,
        skipped=True,
        manifest=verified,
    )


def publish_staged_run(
    staging_dir: Path,
    *,
    fingerprint: str,
    manifest_payload: Mapping[str, object],
    output_dir: Path = PHASE7_MODELS_DIR,
) -> PublishedRun:
    """Registra un staging inmutable sin activar el challenger.

    El caller no debe reutilizar ``staging_dir`` después de una publicación
    correcta: el directorio se mueve a ``runs/<fingerprint>``. Solo la puerta
    de promoción puede crear o cambiar el puntero activo.
    """

    staging = _ensure_project_path(staging_dir, "staging_dir")
    output = _ensure_project_path(output_dir, "output_dir")
    if staging.parent != output:
        raise ArtifactError("staging_dir no pertenece al output_dir.")
    if not staging.name.startswith(".staging-") or not staging.is_dir():
        raise ArtifactError("staging_dir no es un staging válido.")
    if find_verified_run(fingerprint, output_dir=output) is not None:
        raise ArtifactError("El fingerprint ya está publicado; no se sobrescribe.")
    inventory = _file_inventory(staging)
    payload = dict(manifest_payload)
    payload["fingerprint"] = fingerprint
    payload["files"] = list(inventory)
    payload.setdefault("created_at_utc", datetime.now(UTC).isoformat())
    _parse_run_created_at(payload["created_at_utc"], run_dir=staging)
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
    return PublishedRun(
        fingerprint=fingerprint,
        run_dir=target,
        manifest_path=target / "manifest.json",
        skipped=False,
        manifest=verified,
    )

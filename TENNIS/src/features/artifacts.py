"""Publicación generacional, inmutable y verificada de features históricas.

Los artefactos pesados se publican como una generación completa bajo
``runs/<fingerprint>``. El único archivo mutable es ``manifest.json`` en la
raíz del almacén: un puntero pequeño que se reemplaza atómicamente después de
verificar la generación. Así, una interrupción nunca mezcla Parquet de dos
builds ni invalida la generación activa anterior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Final, Mapping, cast

from ..config import FEATURES_PROCESSED_DIR, PROJECT_ROOT


ACTIVE_POINTER_SCHEMA: Final[str] = "tennis-features-active-v1"
ACTIVE_MANIFEST_FILENAME: Final[str] = "manifest.json"
RUNS_DIRECTORY_NAME: Final[str] = "runs"
_SHA256_LENGTH: Final[int] = 64


class FeatureArtifactError(RuntimeError):
    """Indica que un almacén de features no se puede publicar o verificar."""


@dataclass(frozen=True, slots=True)
class PublishedFeatureRun:
    """Describe una generación verificada y su estado de reutilización."""

    fingerprint: str
    run_dir: Path
    manifest_path: Path
    skipped: bool
    manifest: Mapping[str, object]


def _ensure_project_path(path: Path, field_name: str) -> Path:
    """Resuelve una ruta y exige que permanezca dentro de ``TENNIS/``."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise FeatureArtifactError(
            f"{field_name} debe permanecer dentro de TENNIS/: {resolved}."
        )
    return resolved


def _validate_fingerprint(value: object) -> str:
    """Valida un fingerprint SHA-256 hexadecimal en minúsculas."""

    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FeatureArtifactError(
            "El fingerprint de features debe ser SHA-256 hexadecimal."
        )
    return value


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques sobre un artefacto existente."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureArtifactError(
            f"No se pudo leer el artefacto {path}."
        ) from exc
    return digest.hexdigest()


def _load_json_mapping(path: Path, description: str) -> Mapping[str, object]:
    """Lee un objeto JSON o produce un error con contexto estable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureArtifactError(
            f"No se pudo leer {description} {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise FeatureArtifactError(f"{description} debe ser un objeto JSON.")
    return cast(Mapping[str, object], payload)


def _safe_run_artifact(run_dir: Path, relative_path: object) -> Path:
    """Resuelve un artefacto relativo sin aceptar rutas absolutas o traversal."""

    if not isinstance(relative_path, str) or not relative_path:
        raise FeatureArtifactError(
            "La ruta declarada de un artefacto debe ser texto no vacío."
        )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FeatureArtifactError(
            f"Ruta insegura en el run de features: {relative_path!r}."
        )
    candidate = (run_dir / relative).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        raise FeatureArtifactError(
            f"El artefacto sale del run: {relative_path!r}."
        )
    return candidate


def _verify_declared_file(
    run_dir: Path,
    entry: Mapping[str, object],
    *,
    path_key: str,
    size_key: str,
    hash_key: str,
) -> Path:
    """Verifica ruta, tamaño y hash de un archivo declarado en el manifiesto."""

    candidate = _safe_run_artifact(run_dir, entry.get(path_key))
    expected_size = entry.get(size_key)
    expected_hash = entry.get(hash_key)
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_hash, str)
        or len(expected_hash) != _SHA256_LENGTH
        or any(
            character not in "0123456789abcdef"
            for character in expected_hash
        )
    ):
        raise FeatureArtifactError(
            f"Metadatos inválidos para {candidate.name}."
        )
    try:
        exists = candidate.is_file()
        observed_size = candidate.stat().st_size if exists else -1
    except OSError as exc:
        raise FeatureArtifactError(
            f"No se pudo inspeccionar {candidate}."
        ) from exc
    if not exists or observed_size != expected_size:
        raise FeatureArtifactError(
            f"Falta o cambió el tamaño de {candidate}."
        )
    if _sha256_file(candidate) != expected_hash:
        raise FeatureArtifactError(f"SHA-256 divergente para {candidate}.")
    return candidate


def _verify_payload_artifacts(
    artifact_dir: Path,
    payload: Mapping[str, object],
) -> None:
    """Verifica datasets, conflictos y ausencia de archivos no declarados."""

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise FeatureArtifactError(
            "El manifiesto del run no contiene datasets."
        )
    declared: set[Path] = set()
    genders: set[str] = set()
    for raw_entry in datasets:
        if not isinstance(raw_entry, Mapping):
            raise FeatureArtifactError("Una entrada datasets no es un objeto.")
        entry = cast(Mapping[str, object], raw_entry)
        gender = entry.get("gender")
        if gender not in {"M", "F"} or gender in genders:
            raise FeatureArtifactError(
                "Los géneros del manifiesto son inválidos o duplicados."
            )
        genders.add(cast(str, gender))
        path = _verify_declared_file(
            artifact_dir,
            entry,
            path_key="output_path",
            size_key="output_size",
            hash_key="output_sha256",
        )
        if path in declared:
            raise FeatureArtifactError("Un artefacto está declarado dos veces.")
        declared.add(path)

    conflicts = payload.get("conflict_inventory")
    if not isinstance(conflicts, Mapping):
        raise FeatureArtifactError(
            "El manifiesto no declara conflict_inventory."
        )
    conflict_path = _verify_declared_file(
        artifact_dir,
        cast(Mapping[str, object], conflicts),
        path_key="path",
        size_key="size",
        hash_key="sha256",
    )
    if conflict_path in declared:
        raise FeatureArtifactError("El inventario de conflictos está duplicado.")
    declared.add(conflict_path)

    actual = {
        path.resolve()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != ACTIVE_MANIFEST_FILENAME
    }
    if actual != declared:
        extra = sorted(
            str(path.relative_to(artifact_dir)) for path in actual - declared
        )
        missing = sorted(
            str(path.relative_to(artifact_dir)) for path in declared - actual
        )
        raise FeatureArtifactError(
            "El run contiene un inventario inesperado: "
            f"extra={extra}, missing={missing}."
        )


def verify_feature_run(
    run_dir: Path,
    *,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> Mapping[str, object]:
    """Verifica identidad, manifiesto y todos los archivos de un run inmutable."""

    output = _ensure_project_path(output_dir, "output_dir")
    resolved_run = _ensure_project_path(run_dir, "run_dir")
    if resolved_run.parent != output / RUNS_DIRECTORY_NAME:
        raise FeatureArtifactError(
            "El run de features no pertenece al almacén indicado."
        )
    payload = _load_json_mapping(
        resolved_run / ACTIVE_MANIFEST_FILENAME,
        "el manifiesto del run de features",
    )
    fingerprint = _validate_fingerprint(payload.get("fingerprint"))
    if resolved_run.name != fingerprint:
        raise FeatureArtifactError(
            "El nombre del run no coincide con su fingerprint."
        )

    _verify_payload_artifacts(resolved_run, payload)
    return payload


def _active_pointer_payload(
    fingerprint: str,
) -> Mapping[str, object]:
    """Construye el puntero activo mínimo y portable para un fingerprint."""

    checked = _validate_fingerprint(fingerprint)
    return {
        "pointer_schema_version": ACTIVE_POINTER_SCHEMA,
        "fingerprint": checked,
        "active_run": f"{RUNS_DIRECTORY_NAME}/{checked}",
    }


def _write_atomic_json(payload: Mapping[str, object], target: Path) -> None:
    """Escribe y sincroniza JSON temporal antes de reemplazar el destino."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=".feature-manifest-",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except (OSError, TypeError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise FeatureArtifactError(
            f"No se pudo activar atómicamente {target}."
        ) from exc


def resolve_active_feature_run(
    *,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> PublishedFeatureRun:
    """Resuelve el puntero activo y verifica la generación antes de devolverla."""

    output = _ensure_project_path(output_dir, "output_dir")
    pointer_path = output / ACTIVE_MANIFEST_FILENAME
    pointer = _load_json_mapping(pointer_path, "el puntero activo de features")
    if pointer.get("pointer_schema_version") != ACTIVE_POINTER_SCHEMA:
        raise FeatureArtifactError(
            "El manifiesto activo no usa el esquema de puntero esperado."
        )
    fingerprint = _validate_fingerprint(pointer.get("fingerprint"))
    expected_relative = f"{RUNS_DIRECTORY_NAME}/{fingerprint}"
    if pointer.get("active_run") != expected_relative:
        raise FeatureArtifactError(
            "active_run no coincide exactamente con el fingerprint activo."
        )
    run_dir = (output / Path(expected_relative)).resolve()
    payload = verify_feature_run(run_dir, output_dir=output)
    if payload.get("fingerprint") != fingerprint:
        raise FeatureArtifactError(
            "El puntero y el manifiesto inmutable tienen fingerprints distintos."
        )
    return PublishedFeatureRun(
        fingerprint=fingerprint,
        run_dir=run_dir,
        manifest_path=run_dir / ACTIVE_MANIFEST_FILENAME,
        skipped=True,
        manifest=payload,
    )


def find_verified_feature_run(
    fingerprint: str,
    *,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> PublishedFeatureRun | None:
    """Busca una generación por fingerprint sin aceptar una copia corrupta."""

    checked = _validate_fingerprint(fingerprint)
    output = _ensure_project_path(output_dir, "output_dir")
    run_dir = output / RUNS_DIRECTORY_NAME / checked
    if not run_dir.exists():
        return None
    payload = verify_feature_run(run_dir, output_dir=output)
    return PublishedFeatureRun(
        fingerprint=checked,
        run_dir=run_dir,
        manifest_path=run_dir / ACTIVE_MANIFEST_FILENAME,
        skipped=True,
        manifest=payload,
    )


def activate_feature_run(
    published: PublishedFeatureRun,
    *,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> PublishedFeatureRun:
    """Verifica un run existente y reemplaza únicamente el puntero activo."""

    if not isinstance(published, PublishedFeatureRun):
        raise TypeError("published debe ser PublishedFeatureRun.")
    output = _ensure_project_path(output_dir, "output_dir")
    payload = verify_feature_run(published.run_dir, output_dir=output)
    fingerprint = _validate_fingerprint(payload.get("fingerprint"))
    if fingerprint != published.fingerprint:
        raise FeatureArtifactError(
            "El run reutilizado no coincide con su fingerprint declarado."
        )
    _write_atomic_json(
        _active_pointer_payload(fingerprint),
        output / ACTIVE_MANIFEST_FILENAME,
    )
    return PublishedFeatureRun(
        fingerprint=fingerprint,
        run_dir=published.run_dir,
        manifest_path=published.run_dir / ACTIVE_MANIFEST_FILENAME,
        skipped=True,
        manifest=payload,
    )


def preflight_feature_publication(
    *,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> None:
    """Prueba creación, fsync, rename y borrado antes del cálculo costoso.

    Si ya existe un puntero activo, primero verifica la generación y después
    lo reemplaza por bytes semánticamente idénticos. Esto detecta al inicio una
    ACL que impediría la activación final, sin tocar los Parquet publicados.
    """

    output = _ensure_project_path(output_dir, "output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / RUNS_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
    descriptor, raw_probe = tempfile.mkstemp(
        prefix=".feature-preflight-",
        suffix=".tmp",
        dir=output,
    )
    probe = Path(raw_probe)
    moved = probe.with_suffix(".moved")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"tennis-feature-publication-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(probe, moved)
        moved.unlink()
    except OSError as exc:
        for candidate in (probe, moved):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        raise FeatureArtifactError(
            f"El almacén {output} no permite publicar atómicamente."
        ) from exc

    active = output / ACTIVE_MANIFEST_FILENAME
    if active.exists():
        current = resolve_active_feature_run(output_dir=output)
        _write_atomic_json(
            _active_pointer_payload(current.fingerprint),
            active,
        )


def publish_feature_run(
    publish_dir: Path,
    *,
    fingerprint: str,
    output_dir: Path = FEATURES_PROCESSED_DIR,
) -> PublishedFeatureRun:
    """Mueve una generación completa y activa su puntero al final.

    ``publish_dir`` debe ser el hijo ``publish`` de un workspace temporal
    ``.feature-build-*``. Si el fingerprint ya existe, el run se verifica y se
    reactiva; nunca se sobrescribe.
    """

    checked = _validate_fingerprint(fingerprint)
    output = _ensure_project_path(output_dir, "output_dir")
    staged = _ensure_project_path(publish_dir, "publish_dir")
    if (
        staged.name != "publish"
        or not staged.parent.name.startswith(".feature-build-")
        or staged.parent.parent != output
        or not staged.is_dir()
    ):
        raise FeatureArtifactError(
            "publish_dir no es el directorio publicable de un workspace válido."
        )
    staged_payload = _load_json_mapping(
        staged / ACTIVE_MANIFEST_FILENAME,
        "el manifiesto staged de features",
    )
    if _validate_fingerprint(staged_payload.get("fingerprint")) != checked:
        raise FeatureArtifactError(
            "El manifiesto staged no coincide con el fingerprint solicitado."
        )
    _verify_payload_artifacts(staged, staged_payload)

    existing = find_verified_feature_run(checked, output_dir=output)
    if existing is not None:
        comparable_staged = dict(staged_payload)
        comparable_existing = dict(existing.manifest)
        comparable_staged.pop("created_at_utc", None)
        comparable_existing.pop("created_at_utc", None)
        if comparable_staged != comparable_existing:
            raise FeatureArtifactError(
                "El mismo fingerprint produjo artefactos distintos; se rechazó "
                "sobrescribir el run inmutable."
            )
        return activate_feature_run(existing, output_dir=output)

    runs_dir = output / RUNS_DIRECTORY_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / checked
    try:
        os.replace(staged, target)
    except OSError as exc:
        raise FeatureArtifactError(
            f"No se pudo publicar la generación inmutable {target}."
        ) from exc
    payload = verify_feature_run(target, output_dir=output)
    published = PublishedFeatureRun(
        fingerprint=checked,
        run_dir=target,
        manifest_path=target / ACTIVE_MANIFEST_FILENAME,
        skipped=False,
        manifest=payload,
    )
    activated = activate_feature_run(published, output_dir=output)
    return PublishedFeatureRun(
        fingerprint=activated.fingerprint,
        run_dir=activated.run_dir,
        manifest_path=activated.manifest_path,
        skipped=False,
        manifest=activated.manifest,
    )


def resolve_feature_manifest_path(manifest_path: Path) -> Path:
    """Sigue un puntero generacional o conserva un manifiesto plano compatible."""

    resolved = _ensure_project_path(manifest_path, "manifest_path")
    payload = _load_json_mapping(resolved, "el manifiesto de features")
    if payload.get("pointer_schema_version") != ACTIVE_POINTER_SCHEMA:
        return resolved
    active = resolve_active_feature_run(output_dir=resolved.parent)
    return active.manifest_path

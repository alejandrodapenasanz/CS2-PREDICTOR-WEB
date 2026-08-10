"""Inventarios deterministas del codigo que produce artefactos persistentes.

Los fingerprints de datos y parametros no bastan para invalidar un artefacto
cuando cambia una formula sin cambiar su version declarada. Este modulo calcula
un inventario portable de rutas relativas, tamanos y SHA-256 de los archivos de
codigo que intervienen en cada build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Final, Iterable, Mapping, cast


HASH_CHUNK_SIZE: Final[int] = 1024 * 1024
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class CodeInventoryError(RuntimeError):
    """Indica que un inventario de codigo no es seguro o reproducible."""


def _normalise_relative_path(value: object) -> str:
    """Valida una ruta POSIX relativa sin traversal ni alias ambiguos."""

    if not isinstance(value, str) or not value:
        raise CodeInventoryError("Cada ruta de codigo debe ser texto no vacio.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or chr(92) in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CodeInventoryError(
            f"Ruta relativa de codigo no canonica o insegura: {value!r}."
        )
    return value


def canonicalise_code_inventory(
    inventory: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Valida y ordena un inventario para serializacion JSON determinista.

    Args:
        inventory: Entradas con ``path``, ``size`` y ``sha256``.

    Returns:
        Tupla ordenada por ruta con diccionarios de esquema estable.

    Raises:
        CodeInventoryError: Si una entrada es invalida o una ruta se repite.
    """

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_entry in inventory:
        if not isinstance(raw_entry, Mapping):
            raise CodeInventoryError("Cada entrada de codigo debe ser un objeto.")
        path = _normalise_relative_path(raw_entry.get("path"))
        size = raw_entry.get("size")
        sha256 = raw_entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CodeInventoryError(f"Tamano invalido para {path!r}.")
        if (
            not isinstance(sha256, str)
            or _SHA256_PATTERN.fullmatch(sha256) is None
        ):
            raise CodeInventoryError(f"SHA-256 invalido para {path!r}.")
        if path in seen:
            raise CodeInventoryError(f"Ruta de codigo duplicada: {path!r}.")
        seen.add(path)
        entries.append({"path": path, "size": size, "sha256": sha256})
    if not entries:
        raise CodeInventoryError("El inventario de codigo no puede estar vacio.")
    return tuple(sorted(entries, key=lambda entry: str(entry["path"])))


def build_code_inventory(
    project_root: Path,
    relative_paths: Iterable[str],
) -> tuple[dict[str, object], ...]:
    """Calcula SHA-256 binario para una lista explicita de codigo del proyecto.

    Args:
        project_root: Raiz contra la que se resuelven las rutas relativas.
        relative_paths: Rutas POSIX explicitas; nunca se descubre con ``glob``.

    Returns:
        Inventario canonico y portable, ordenado por ruta.

    Raises:
        CodeInventoryError: Si una ruta sale de la raiz, falta o no es archivo.
    """

    root = Path(project_root).resolve()
    entries: list[dict[str, object]] = []
    for raw_path in relative_paths:
        relative_path = _normalise_relative_path(raw_path)
        resolved = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
        if not resolved.is_relative_to(root):
            raise CodeInventoryError(
                f"La ruta de codigo sale de la raiz: {relative_path!r}."
            )
        if not resolved.is_file():
            raise CodeInventoryError(
                f"No existe el archivo de codigo requerido: {relative_path!r}."
            )
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as source:
                for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
            size = resolved.stat().st_size
        except OSError as exc:
            raise CodeInventoryError(
                f"No se puede leer el archivo de codigo: {relative_path!r}."
            ) from exc
        entries.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": digest.hexdigest(),
            }
        )
    return canonicalise_code_inventory(entries)


def verify_code_inventory(
    project_root: Path,
    relative_paths: Iterable[str],
    persisted_inventory: object,
) -> tuple[dict[str, object], ...]:
    """Exige que un inventario persistido coincida con el código actual.

    Args:
        project_root: Raíz portable del proyecto.
        relative_paths: Contrato explícito usado también por el productor.
        persisted_inventory: Valor leído del artefacto persistido.

    Returns:
        Inventario actual canónico cuando coincide exactamente.

    Raises:
        CodeInventoryError: Si falta el inventario, su esquema es inválido o
            cualquier ruta, tamaño o SHA-256 difiere del código actual.
    """

    if not isinstance(persisted_inventory, list):
        raise CodeInventoryError(
            "El artefacto no contiene code_inventory como lista."
        )
    persisted = canonicalise_code_inventory(
        cast(Iterable[Mapping[str, object]], persisted_inventory)
    )
    current = build_code_inventory(project_root, relative_paths)
    if persisted != current:
        persisted_by_path = {
            str(entry["path"]): entry for entry in persisted
        }
        current_by_path = {str(entry["path"]): entry for entry in current}
        changed = sorted(
            path
            for path in set(persisted_by_path) | set(current_by_path)
            if persisted_by_path.get(path) != current_by_path.get(path)
        )
        raise CodeInventoryError(
            "El inventario de código persistido no coincide con el actual: "
            f"{changed}. Reconstruya el artefacto antes de consumirlo."
        )
    return current

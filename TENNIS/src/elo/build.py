"""Orquestación reproducible de la construcción histórica del sistema Elo.

Este módulo toma como única autoridad de entrada el manifiesto Sackmann local.
Antes de exponer una fila valida el commit, la cabecera CSV, el tamaño y el
SHA-1 de blob Git de cada archivo de partidos seleccionado. Las filas se leen
por bloques y reciben procedencia estable, sin descubrir archivos mediante
``glob``.

La publicación de la base SQLite se realiza sobre una ruta temporal situada en
el mismo directorio que el destino. Solo una ejecución completa sustituye el
archivo activo mediante ``os.replace``; un fallo conserva la base anterior.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import TYPE_CHECKING, Any, Final, Iterable, Iterator, Literal, Mapping, cast

import pandas as pd

from src.config import (
    ELO_DATABASE_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from src.artifact_integrity import (
    build_code_inventory,
    canonicalise_code_inventory,
)
from src.data_loaders import MATCH_SOURCE_COLUMNS
from src.temporal import (
    DEFAULT_SOURCE_DATE_POLICY,
    SourceDatePolicy,
)

if TYPE_CHECKING:
    from .types import MatchEvent


Gender = Literal["M", "F"]
GenderSelection = Literal["M", "F", "all"]
LOGGER = logging.getLogger(__name__)

EXPECTED_SOURCE_REPOSITORY: Final[str] = (
    "Aneeshers/tennis-sackmann-archive"
)
DEFAULT_CHUNKSIZE: Final[int] = 50_000
HASH_CHUNK_SIZE: Final[int] = 1024 * 1024
SPOOL_TABLE: Final[str] = "_elo_input_events"
PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_commit",
    "source_path",
    "source_row_number",
)
ELO_CODE_PATHS: Final[tuple[str, ...]] = (
    "config/elo_handoff.json",
    "scripts/build_elo.py",
    "src/artifact_integrity.py",
    "src/data_loaders.py",
    "src/elo/build.py",
    "src/elo/engine.py",
    "src/elo/events.py",
    "src/elo/operational.py",
    "src/elo/parameters.py",
    "src/elo/store.py",
    "src/elo/types.py",
    "src/identity_integrity.py",
    "src/tennisratio/__init__.py",
    "src/tennisratio/service.py",
    "src/tennisratio/store.py",
    "src/tennisratio/types.py",
    "src/temporal.py",
)

MATCH_PATH_LAYOUT: Final[
    dict[str, tuple[Gender, re.Pattern[str]]]
] = {
    "atp_main": (
        "M",
        re.compile(r"^atp/atp_matches_(?P<year>\d{4})\.csv$"),
    ),
    "atp_qual_chall": (
        "M",
        re.compile(
            r"^atp/atp_matches_qual_chall_(?P<year>\d{4})\.csv$"
        ),
    ),
    "atp_futures": (
        "M",
        re.compile(r"^atp/atp_matches_futures_(?P<year>\d{4})\.csv$"),
    ),
    "wta_main": (
        "F",
        re.compile(r"^wta/wta_matches_(?P<year>\d{4})\.csv$"),
    ),
    "wta_qual_itf": (
        "F",
        re.compile(
            r"^wta/wta_matches_qual_itf_(?P<year>\d{4})\.csv$"
        ),
    ),
}

_SHA1_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_UNSIGNED_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+$")


class EloBuildError(RuntimeError):
    """Indica que la orquestación Elo no puede producir una base fiable."""


class ManifestValidationError(EloBuildError):
    """Indica que el manifiesto no cumple el contrato auditado."""


class SourceIntegrityError(EloBuildError):
    """Indica que un CSV local diverge de tamaño, esquema o blob Git."""


class ExistingEloDatabaseError(EloBuildError):
    """Indica que la base activa no puede omitirse ni reemplazarse."""


class EloRuntimeContractError(EloBuildError):
    """Indica que events, engine o store no exponen la interfaz esperada."""


@dataclass(frozen=True)
class VerifiedMatchFile:
    """Describe un CSV de partidos verificado contra el manifiesto."""

    source_path: str
    local_path: Path
    category: str
    gender: Gender
    year: int
    size: int
    git_blob_sha: str


@dataclass(frozen=True)
class VerifiedManifest:
    """Conserva un snapshot Sackmann y sus archivos de partidos verificados."""

    source_repository: str
    source_commit: str
    manifest_path: Path
    raw_dir: Path
    match_files: tuple[VerifiedMatchFile, ...]

    def select_gender(
        self,
        gender: GenderSelection,
    ) -> VerifiedManifest:
        """Devuelve una vista inmutable limitada al género solicitado."""

        selected = _normalise_gender_selection(gender)
        if selected == ("M", "F"):
            return self
        files = tuple(
            match_file
            for match_file in self.match_files
            if match_file.gender in selected
        )
        if not files:
            raise ManifestValidationError(
                f"El manifiesto no contiene partidos para {gender!r}."
            )
        return VerifiedManifest(
            source_repository=self.source_repository,
            source_commit=self.source_commit,
            manifest_path=self.manifest_path,
            raw_dir=self.raw_dir,
            match_files=files,
        )


@dataclass(frozen=True)
class GenderBuildAudit:
    """Resume inclusión, exclusión y persistencia de un universo de género."""

    gender: Gender
    source_rows: int
    included_events: int
    excluded_events: int
    exact_duplicates: int
    date_blocks: int
    rating_rows: int
    exclusions_by_reason: Mapping[str, int]
    base_source_rows: int
    operational_source_rows: int


@dataclass(frozen=True)
class EloBuildReport:
    """Resume una publicación Elo o una omisión idempotente."""

    database_path: Path
    source_commit: str
    algorithm_version: str
    input_fingerprints: Mapping[Gender, str]
    run_ids: Mapping[Gender, str]
    skipped: bool
    audits: tuple[GenderBuildAudit, ...]
    top_ratings: Mapping[Gender, tuple[Mapping[str, object], ...]]
    operational_overlay: Mapping[str, object] | None


def _normalise_gender_selection(
    gender: str,
) -> tuple[Gender, ...]:
    """Valida ``M``, ``F`` o ``all`` sin aceptar alias implícitos."""

    if gender == "M":
        return ("M",)
    if gender == "F":
        return ("F",)
    if gender == "all":
        return ("M", "F")
    raise ValueError("gender debe ser exactamente 'M', 'F' o 'all'.")


def _normalise_identity_exclusion_after_dates(
    rules: Mapping[tuple[Gender, int], date] | None,
) -> Mapping[tuple[Gender, int], date]:
    """Valida reglas causales mediante el mismo contrato que el motor."""

    from .engine import normalise_identity_exclusion_after_dates

    return normalise_identity_exclusion_after_dates(rules)


def _identity_rule_payload(
    rules: Mapping[tuple[Gender, int], date],
) -> list[list[object]]:
    """Serializa reglas causales en orden estable para hash y manifiesto."""

    return [
        [gender, player_id, exclusion_after_date.isoformat()]
        for (gender, player_id), exclusion_after_date in sorted(rules.items())
    ]


def _load_manifest_payload(manifest_path: Path) -> Mapping[str, Any]:
    """Lee el JSON del manifiesto y exige un objeto en el nivel raíz."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(
            f"No se puede leer el manifiesto Sackmann: {manifest_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ManifestValidationError(
            "El manifiesto Sackmann no es un objeto JSON."
        )
    return payload


def _safe_local_path(raw_dir: Path, relative_path: str) -> Path:
    """Resuelve una ruta POSIX del manifiesto sin permitir traversal."""

    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ManifestValidationError(
            f"Ruta local no segura en el manifiesto: {relative_path!r}."
        )
    root = raw_dir.resolve()
    resolved = (root / Path(*pure_path.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise ManifestValidationError(
            f"La ruta local sale de data/raw: {relative_path!r}."
        )
    return resolved


def _git_blob_sha(path: Path, size: int) -> str:
    """Calcula el SHA-1 que Git asigna al contenido exacto de un archivo."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceIntegrityError(f"No se puede leer {path}.") from exc
    return digest.hexdigest()


def _read_csv_header(path: Path) -> tuple[str, ...]:
    """Lee únicamente la primera fila CSV mediante el parser estándar."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            return tuple(next(reader))
    except StopIteration as exc:
        raise SourceIntegrityError(f"El CSV está vacío: {path}.") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SourceIntegrityError(
            f"No se puede leer la cabecera CSV de {path}."
        ) from exc


def _verified_match_file(
    item: Mapping[str, Any],
    *,
    raw_dir: Path,
) -> VerifiedMatchFile | None:
    """Valida una entrada de partidos e ignora categorías auxiliares."""

    category = item.get("category")
    if category not in MATCH_PATH_LAYOUT:
        return None
    if not isinstance(category, str):
        raise ManifestValidationError(
            "Una categoría de partidos no es textual."
        )
    source_path = item.get("remote_path")
    local_path = item.get("local_path")
    if not isinstance(source_path, str) or not isinstance(local_path, str):
        raise ManifestValidationError(
            "Una entrada de partidos carece de remote_path/local_path."
        )
    if source_path != local_path:
        raise ManifestValidationError(
            f"remote_path y local_path difieren para {source_path!r}."
        )

    gender, pattern = MATCH_PATH_LAYOUT[category]
    match = pattern.fullmatch(source_path)
    if match is None:
        raise ManifestValidationError(
            f"La ruta {source_path!r} no coincide con {category!r}."
        )
    path_year = int(match.group("year"))
    manifest_year = item.get("year")
    if (
        isinstance(manifest_year, bool)
        or not isinstance(manifest_year, int)
        or manifest_year != path_year
    ):
        raise ManifestValidationError(
            f"El año de {source_path!r} no coincide con su ruta."
        )

    expected_size = item.get("size")
    expected_sha = item.get("git_blob_sha")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ManifestValidationError(
            f"Tamaño inválido en el manifiesto para {source_path!r}."
        )
    if (
        not isinstance(expected_sha, str)
        or _SHA1_PATTERN.fullmatch(expected_sha) is None
    ):
        raise ManifestValidationError(
            f"SHA de blob inválido para {source_path!r}."
        )

    resolved_path = _safe_local_path(raw_dir, local_path)
    if not resolved_path.is_file():
        raise SourceIntegrityError(
            f"No existe el CSV listado en el manifiesto: {resolved_path}."
        )
    observed_size = resolved_path.stat().st_size
    if observed_size != expected_size:
        raise SourceIntegrityError(
            f"{source_path}: tamaño local {observed_size}; "
            f"manifiesto {expected_size}."
        )
    observed_header = _read_csv_header(resolved_path)
    if observed_header != MATCH_SOURCE_COLUMNS:
        raise SourceIntegrityError(
            f"{source_path}: esquema inesperado. "
            f"Esperado={MATCH_SOURCE_COLUMNS!r}; "
            f"observado={observed_header!r}."
        )
    observed_sha = _git_blob_sha(resolved_path, observed_size)
    if observed_sha != expected_sha:
        raise SourceIntegrityError(
            f"{source_path}: el SHA-1 de blob Git no coincide."
        )
    return VerifiedMatchFile(
        source_path=source_path,
        local_path=resolved_path,
        category=category,
        gender=gender,
        year=path_year,
        size=expected_size,
        git_blob_sha=expected_sha,
    )


def load_verified_manifest(
    manifest_path: Path = SACKMANN_MANIFEST_PATH,
    raw_dir: Path = RAW_DATA_DIR,
) -> VerifiedManifest:
    """Carga el manifiesto y verifica todos los CSV de partidos listados.

    Args:
        manifest_path: JSON activo producido por la ingesta Sackmann.
        raw_dir: Raíz contra la que se resuelven exclusivamente ``local_path``.

    Returns:
        Un manifiesto inmutable ordenado por ruta fuente.

    Raises:
        ManifestValidationError: Si commit, inventario o rutas son inválidos.
        SourceIntegrityError: Si tamaño, cabecera o blob local divergen.
    """

    resolved_manifest = manifest_path.resolve()
    resolved_raw = raw_dir.resolve()
    payload = _load_manifest_payload(resolved_manifest)
    repository = payload.get("source_repository")
    if repository != EXPECTED_SOURCE_REPOSITORY:
        raise ManifestValidationError(
            "El manifiesto no pertenece al mirror Sackmann autorizado."
        )
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or _SHA1_PATTERN.fullmatch(source_commit) is None
    ):
        raise ManifestValidationError(
            "source_commit debe ser un SHA-1 Git hexadecimal de 40 caracteres."
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ManifestValidationError(
            "El manifiesto no contiene una lista files."
        )

    verified: list[VerifiedMatchFile] = []
    seen_source_paths: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise ManifestValidationError(
                f"files[{index}] no es un objeto JSON."
            )
        match_file = _verified_match_file(item, raw_dir=resolved_raw)
        if match_file is None:
            continue
        if match_file.source_path in seen_source_paths:
            raise ManifestValidationError(
                f"Ruta de partidos duplicada: {match_file.source_path!r}."
            )
        seen_source_paths.add(match_file.source_path)
        verified.append(match_file)
    if not verified:
        raise ManifestValidationError(
            "El manifiesto no contiene ningún CSV de partidos."
        )
    return VerifiedManifest(
        source_repository=repository,
        source_commit=source_commit,
        manifest_path=resolved_manifest,
        raw_dir=resolved_raw,
        match_files=tuple(
            sorted(verified, key=lambda match_file: match_file.source_path)
        ),
    )


def iter_manifest_match_frames(
    manifest: VerifiedManifest,
    gender: GenderSelection = "all",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Itera únicamente CSV verificados y añade procedencia fila a fila.

    ``source_row_number`` es el ordinal de registro de datos basado en uno; la
    cabecera queda excluida. El orden es estable por ``source_path`` y por orden
    original de fila, pero no representa orden cronológico global.

    Args:
        manifest: Resultado de :func:`load_verified_manifest`.
        gender: ``M``, ``F`` o ``all``.
        chunksize: Máximo de registros fuente por DataFrame.

    Yields:
        Frames raw con género, familia y las tres columnas de procedencia.
    """

    if not isinstance(manifest, VerifiedManifest):
        raise TypeError("manifest debe ser VerifiedManifest.")
    selected_genders = _normalise_gender_selection(gender)
    if isinstance(chunksize, bool) or not isinstance(chunksize, int):
        raise TypeError("chunksize debe ser un entero positivo.")
    if chunksize <= 0:
        raise ValueError("chunksize debe ser positivo.")

    selected_files = tuple(
        match_file
        for match_file in manifest.match_files
        if match_file.gender in selected_genders
    )
    if not selected_files:
        raise ManifestValidationError(
            f"No hay archivos verificados para gender={gender!r}."
        )

    for match_file in selected_files:
        row_offset = 0
        try:
            chunks = pd.read_csv(
                match_file.local_path,
                dtype="string",
                keep_default_na=False,
                chunksize=chunksize,
                low_memory=False,
            )
            with chunks:
                for frame in chunks:
                    if tuple(frame.columns) != MATCH_SOURCE_COLUMNS:
                        raise SourceIntegrityError(
                            f"El esquema cambió tras verificar "
                            f"{match_file.source_path}."
                        )
                    row_count = len(frame)
                    result = frame.reset_index(drop=True).copy()
                    result["gender"] = pd.Series(
                        [match_file.gender] * row_count,
                        dtype="string",
                    )
                    result["source_family"] = pd.Series(
                        [match_file.category] * row_count,
                        dtype="string",
                    )
                    result["source_commit"] = pd.Series(
                        [manifest.source_commit] * row_count,
                        dtype="string",
                    )
                    result["source_path"] = pd.Series(
                        [match_file.source_path] * row_count,
                        dtype="string",
                    )
                    result["source_row_number"] = pd.Series(
                        range(
                            row_offset + 1,
                            row_offset + row_count + 1,
                        ),
                        dtype="Int64",
                    )
                    row_offset += row_count
                    yield result
        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            raise SourceIntegrityError(
                f"No se puede iterar {match_file.source_path}."
            ) from exc


def _parameter_mapping(parameters: object) -> Mapping[str, Any]:
    """Convierte parámetros dataclass o mapping sin omitir ningún campo."""

    if is_dataclass(parameters) and not isinstance(parameters, type):
        payload = asdict(parameters)
    elif isinstance(parameters, Mapping):
        payload = dict(parameters)
    else:
        raise TypeError("parameters debe ser dataclass o Mapping.")
    if not isinstance(payload, Mapping):
        raise TypeError("Los parámetros no se convirtieron en un Mapping.")
    return payload


def compute_input_fingerprint(
    manifest: VerifiedManifest,
    parameters_mapping: Mapping[str, Any] | object,
    algorithm_version: str,
    identity_exclusion_after_dates: (
        Mapping[tuple[Gender, int], date] | None
    ) = None,
    code_inventory: Iterable[Mapping[str, object]] | None = None,
    source_date_policy: SourceDatePolicy = DEFAULT_SOURCE_DATE_POLICY,
    operational_contract: Mapping[str, object] | None = None,
) -> str:
    """Calcula SHA-256 de fuentes, parámetros, código y cuarentena.

    El inventario es inyectable para pruebas sintéticas. En producción se
    calcula sobre :data:`ELO_CODE_PATHS`, de modo que un cambio de fórmula
    invalida el artefacto aunque se olvide actualizar ``algorithm_version``.
    """

    if not isinstance(manifest, VerifiedManifest):
        raise TypeError("manifest debe ser VerifiedManifest.")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise ValueError("algorithm_version debe ser texto no vacío.")
    if not isinstance(source_date_policy, SourceDatePolicy):
        raise TypeError("source_date_policy debe ser SourceDatePolicy.")
    parameters = _parameter_mapping(parameters_mapping)
    quarantine = _normalise_identity_exclusion_after_dates(
        identity_exclusion_after_dates
    )
    resolved_code_inventory = canonicalise_code_inventory(
        build_code_inventory(PROJECT_ROOT, ELO_CODE_PATHS)
        if code_inventory is None
        else code_inventory
    )
    payload = {
        "source_commit": manifest.source_commit,
        "match_files": [
            {
                "source_path": match_file.source_path,
                "git_blob_sha": match_file.git_blob_sha,
            }
            for match_file in sorted(
                manifest.match_files,
                key=lambda item: item.source_path,
            )
        ],
        "parameters": parameters,
        "algorithm_version": algorithm_version,
        "code_inventory": list(resolved_code_inventory),
        "identity_exclusion_after_dates": _identity_rule_payload(quarantine),
        "source_date_policy": source_date_policy.as_dict(),
        "operational_overlay": (
            None if operational_contract is None else dict(operational_contract)
        ),
    }
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Los parámetros Elo deben ser JSON determinista y finito."
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


def _run_id(gender: Gender, fingerprint: str) -> str:
    """Deriva un identificador de ejecución reproducible por género."""

    return f"elo-{gender.lower()}-{fingerprint[:24]}"


def _staging_path(database_path: Path, fingerprint: str) -> Path:
    """Genera una ruta staging única dentro del directorio de destino."""

    nonce = os.urandom(8).hex()
    return database_path.with_name(
        f".{database_path.name}.{fingerprint[:12]}.{nonce}.staging"
    )


def _cleanup_staging(path: Path) -> None:
    """Elimina solo el staging exacto y sus sidecars SQLite conocidos."""

    for candidate in (
        path,
        path.with_name(f"{path.name}-journal"),
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        candidate.unlink(missing_ok=True)


def _spool_path(database_path: Path, fingerprint: str) -> Path:
    """Genera una ruta efímera para ordenar eventos sin usar memoria total."""

    nonce = os.urandom(8).hex()
    return database_path.with_name(
        f".{database_path.name}.{fingerprint[:12]}.{nonce}.spool"
    )


def _strictly_type_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Tipa fechas e IDs Sackmann sin aceptar formatos implícitos."""

    result = frame.copy()
    raw_dates = result["tourney_date"].astype("string").str.strip()
    valid_shape = raw_dates.str.fullmatch(r"\d{8}", na=False)
    if not bool(valid_shape.all()):
        first_bad = int(result.loc[~valid_shape, "source_row_number"].iloc[0])
        source_path = str(result.loc[~valid_shape, "source_path"].iloc[0])
        raise SourceIntegrityError(
            f"{source_path}, fila {first_bad}: tourney_date no tiene "
            "el formato Sackmann YYYYMMDD."
        )
    try:
        result["tourney_date"] = pd.to_datetime(
            raw_dates,
            format="%Y%m%d",
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise SourceIntegrityError(
            "Una tourney_date Sackmann no representa una fecha válida."
        ) from exc

    for column in ("winner_id", "loser_id"):
        raw_ids = result[column].astype("string").str.strip()
        valid_ids = raw_ids.str.fullmatch(
            _UNSIGNED_INTEGER_PATTERN,
            na=False,
        )
        if not bool(valid_ids.all()):
            first_bad = int(
                result.loc[~valid_ids, "source_row_number"].iloc[0]
            )
            source_path = str(
                result.loc[~valid_ids, "source_path"].iloc[0]
            )
            raise SourceIntegrityError(
                f"{source_path}, fila {first_bad}: {column} no es un "
                "entero no negativo."
            )
        try:
            result[column] = pd.to_numeric(
                raw_ids,
                errors="raise",
            ).astype("int64")
        except (ValueError, TypeError, OverflowError) as exc:
            raise SourceIntegrityError(
                f"{column} excede el rango entero compatible."
            ) from exc
    result["source_row_number"] = result[
        "source_row_number"
    ].astype("int64")
    return result


def _raw_source_record_hashes(frame: pd.DataFrame) -> tuple[str, ...]:
    """Hashea las 49 celdas fuente antes de normalizar fecha o IDs.

    El CSV se lee con ``dtype="string"`` y sin conversión de valores vacíos.
    Por ello este hash distingue, por ejemplo, ``"1"`` de ``"01"`` y texto
    con espacios periféricos. Solo dos filas con las mismas cadenas en todas
    las columnas fuente se consideran copias exactas.
    """

    missing_columns = sorted(
        set(MATCH_SOURCE_COLUMNS).difference(frame.columns)
    )
    if missing_columns:
        raise SourceIntegrityError(
            "No se puede hashear una fila Sackmann; faltan columnas: "
            f"{missing_columns}."
        )

    hashes: list[str] = []
    source_values = frame.loc[:, list(MATCH_SOURCE_COLUMNS)].itertuples(
        index=False,
        name=None,
    )
    for position, values in enumerate(source_values):
        if any(not isinstance(value, str) for value in values):
            raise SourceIntegrityError(
                "El lector raw produjo una celda no textual en la fila "
                f"{position} del chunk."
            )
        payload = tuple(zip(MATCH_SOURCE_COLUMNS, values, strict=True))
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        hashes.append(hashlib.sha256(encoded).hexdigest())
    return tuple(hashes)


def prepare_manifest_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Tipa un chunk raw verificado y añade su identidad exacta de fila.

    Esta frontera pública permite que consumidores causales posteriores, como
    el constructor de features, usen exactamente la misma definición de fecha,
    identificadores y duplicado exacto que la construcción Elo. El hash se
    calcula antes de normalizar sobre las 49 cadenas fuente.

    Args:
        frame: Chunk emitido por :func:`iter_manifest_match_frames`.

    Returns:
        Copia tipada con ``source_record_hash`` SHA-256.
    """

    hashes = _raw_source_record_hashes(frame)
    typed = _strictly_type_event_frame(frame)
    typed["source_record_hash"] = pd.Series(
        hashes,
        index=typed.index,
        dtype="string",
    )
    return typed


def _create_spool(path: Path) -> sqlite3.Connection:
    """Crea el spool SQLite privado con un esquema mínimo y explícito."""

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"""
            CREATE TABLE {SPOOL_TABLE} (
                gender TEXT NOT NULL,
                event_date TEXT NOT NULL,
                source_event_date TEXT NOT NULL,
                winner_id INTEGER NOT NULL,
                loser_id INTEGER NOT NULL,
                surface TEXT,
                tour_level TEXT NOT NULL,
                score TEXT,
                source_commit TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_row_number INTEGER NOT NULL,
                source_record_hash TEXT NOT NULL,
                tourney_id TEXT,
                match_num TEXT,
                round_name TEXT
            )
            """
        )
        connection.commit()
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _spool_manifest_events(
    *,
    manifest: VerifiedManifest,
    gender: GenderSelection,
    chunksize: int,
    spool_path: Path,
    source_date_policy: SourceDatePolicy,
    operational_events: Iterable[MatchEvent] = (),
    base_cutoff_date: date | None = None,
) -> Mapping[Gender, int]:
    """Vuelca eventos ordenados por su fecha causal de disponibilidad."""

    from .events import (
        EventColumns,
        EventValidationError,
        events_from_dataframe,
    )

    columns = EventColumns(
        tour_level="tourney_level",
        source_path="source_path",
        source_row="source_row_number",
        source_record_hash="source_record_hash",
    )
    counts: Counter[Gender] = Counter()
    connection = _create_spool(spool_path)
    insert_sql = f"""
        INSERT INTO {SPOOL_TABLE} (
            gender,
            event_date,
            source_event_date,
            winner_id,
            loser_id,
            surface,
            tour_level,
            score,
            source_commit,
            source_path,
            source_row_number,
            source_record_hash,
            tourney_id,
            match_num,
            round_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        for raw_frame in iter_manifest_match_frames(
            manifest,
            gender=gender,
            chunksize=chunksize,
        ):
            if raw_frame.empty:
                continue
            typed_frame = prepare_manifest_event_frame(raw_frame)
            try:
                events = events_from_dataframe(
                    typed_frame,
                    source_commit=manifest.source_commit,
                    columns=columns,
                )
            except EventValidationError as exc:
                raise SourceIntegrityError(
                    f"Una fila Sackmann no cumple el contrato Elo: {exc}"
                ) from exc
            if base_cutoff_date is not None and any(
                event.result_source_date > base_cutoff_date
                for event in events
            ):
                raise SourceIntegrityError(
                    "El snapshot Sackmann congelado contiene un partido "
                    "posterior al corte fijo del handoff."
                )
            connection.executemany(
                insert_sql,
                (
                    (
                        event.gender,
                        source_date_policy.availability_date(
                            event.date
                        ).isoformat(),
                        event.result_source_date.isoformat(),
                        event.winner_id,
                        event.loser_id,
                        event.surface,
                        event.tour_level,
                        event.score,
                        event.provenance.source_commit,
                        event.provenance.source_path,
                        event.provenance.source_row,
                        event.source_record_hash,
                        event.tourney_id,
                        event.match_num,
                        event.round,
                    )
                    for event in events
                ),
            )
            counts.update(event.gender for event in events)
            connection.commit()
        checked_operational_events = tuple(operational_events)
        connection.executemany(
            insert_sql,
            (
                (
                    event.gender,
                    event.date.isoformat(),
                    event.result_source_date.isoformat(),
                    event.winner_id,
                    event.loser_id,
                    event.surface,
                    event.tour_level,
                    event.score,
                    event.provenance.source_commit,
                    event.provenance.source_path,
                    event.provenance.source_row,
                    event.source_record_hash,
                    event.tourney_id,
                    event.match_num,
                    event.round,
                )
                for event in checked_operational_events
            ),
        )
        counts.update(event.gender for event in checked_operational_events)
        connection.commit()
        connection.execute(
            f"""
            CREATE INDEX idx_elo_spool_order
            ON {SPOOL_TABLE} (
                gender,
                event_date,
                source_path,
                source_row_number
            )
            """
        )
        connection.commit()
    except (sqlite3.Error, OSError) as exc:
        connection.rollback()
        raise EloBuildError(
            f"No se pudo construir el spool Elo {spool_path}."
        ) from exc
    finally:
        connection.close()
    return {"M": int(counts["M"]), "F": int(counts["F"])}


def _iter_spooled_date_frames(
    spool_path: Path,
    *,
    gender: Gender,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """Emite una fecha completa aunque atraviese límites de lectura SQLite."""

    connection = sqlite3.connect(spool_path)
    query = f"""
        SELECT
            gender,
            event_date AS tourney_date,
            source_event_date,
            winner_id,
            loser_id,
            surface,
            tour_level,
            score,
            source_commit,
            source_path,
            source_row_number,
            source_record_hash,
            tourney_id,
            match_num,
            round_name AS round
        FROM {SPOOL_TABLE}
        WHERE gender = ?
        ORDER BY event_date, source_path, source_row_number
    """
    pending: pd.DataFrame | None = None
    try:
        chunks = pd.read_sql_query(
            query,
            connection,
            params=(gender,),
            chunksize=chunksize,
        )
        for chunk in chunks:
            combined = (
                chunk
                if pending is None
                else pd.concat((pending, chunk), ignore_index=True)
            )
            if combined.empty:
                continue
            trailing_date = combined["tourney_date"].iloc[-1]
            ready = combined.loc[
                combined["tourney_date"] != trailing_date
            ]
            pending = combined.loc[
                combined["tourney_date"] == trailing_date
            ].copy()
            for _, date_frame in ready.groupby(
                "tourney_date",
                sort=False,
            ):
                result = date_frame.reset_index(drop=True)
                result["tourney_date"] = pd.to_datetime(
                    result["tourney_date"],
                    format="%Y-%m-%d",
                    errors="raise",
                )
                result["source_event_date"] = pd.to_datetime(
                    result["source_event_date"],
                    format="%Y-%m-%d",
                    errors="raise",
                )
                yield result
        if pending is not None and not pending.empty:
            pending = pending.reset_index(drop=True)
            pending["tourney_date"] = pd.to_datetime(
                pending["tourney_date"],
                format="%Y-%m-%d",
                errors="raise",
            )
            pending["source_event_date"] = pd.to_datetime(
                pending["source_event_date"],
                format="%Y-%m-%d",
                errors="raise",
            )
            yield pending
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise EloBuildError(
            f"No se pudieron reagrupar los eventos Elo de {gender}."
        ) from exc
    finally:
        connection.close()


def _date_block_input_hash(events: tuple[object, ...]) -> str:
    """Resume contenido y procedencia ordenados de un bloque exacto."""

    digest = hashlib.sha256()
    for event in events:
        provenance = getattr(event, "provenance")
        payload = (
            getattr(provenance, "source_commit"),
            getattr(provenance, "source_path"),
            getattr(provenance, "source_row"),
            getattr(event, "source_record_hash"),
        )
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _top_ratings_from_database(
    database_path: Path,
    *,
    run_id: str,
) -> tuple[Mapping[str, object], ...]:
    """Consulta los diez últimos estados con mayor Elo general de un run."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            WITH latest AS (
                SELECT player_id, MAX(state_date) AS state_date
                FROM rating_history
                WHERE run_id = ?
                GROUP BY player_id
            )
            SELECT
                history.player_id,
                history.general_elo,
                history.general_matches,
                history.state_date
            FROM rating_history AS history
            JOIN latest
              ON latest.player_id = history.player_id
             AND latest.state_date = history.state_date
            WHERE history.run_id = ?
            ORDER BY history.general_elo DESC, history.player_id
            LIMIT 10
            """,
            (run_id, run_id),
        ).fetchall()
    except sqlite3.Error as exc:
        raise ExistingEloDatabaseError(
            "No se pudo obtener el top 10 de la base Elo activa."
        ) from exc
    finally:
        connection.close()
    return tuple(
        {
            "player_id": int(row["player_id"]),
            "general_elo": float(row["general_elo"]),
            "general_matches": int(row["general_matches"]),
            "state_date": str(row["state_date"]),
        }
        for row in rows
    )


def build_elo_database(
    *,
    manifest_path: Path = SACKMANN_MANIFEST_PATH,
    raw_dir: Path = RAW_DATA_DIR,
    database_path: Path = ELO_DATABASE_PATH,
    gender: GenderSelection = "all",
    chunksize: int = DEFAULT_CHUNKSIZE,
    force: bool = False,
    parameters: object | None = None,
    identity_exclusion_after_dates: (
        Mapping[tuple[Gender, int], date] | None
    ) = None,
    source_date_policy: SourceDatePolicy = DEFAULT_SOURCE_DATE_POLICY,
    operational_overlay: object | None = None,
) -> EloBuildReport:
    """Construye y publica la base Elo mediante los contratos core/store.

    La integración se importa localmente para que las primitivas de manifiesto
    sigan disponibles durante el desarrollo independiente de ``events``,
    ``engine`` y ``store``. El cuerpo final se completa contra sus interfaces
    públicas, nunca mediante copias locales de sus tipos.

    Args:
        manifest_path: Manifiesto Sackmann que autoriza las entradas.
        raw_dir: Raíz de archivos raw.
        database_path: SQLite final que solo se reemplaza tras éxito completo.
        gender: Universo ``M``, ``F`` o ambos.
        chunksize: Tamaño de lectura de cada CSV.
        force: Reconstruye aunque el fingerprint activo ya exista.
        parameters: ``EloParameters`` opcional; usa sus defaults aprobados.
        identity_exclusion_after_dates: Primera evidencia por identidad; solo
            excluye eventos con fecha estrictamente posterior.
        source_date_policy: Embargo causal aplicado a ``tourney_date`` antes
            de que un resultado pueda modificar un estado Elo.
        operational_overlay: Eventos post-corte ya validados por el adaptador
            multifuente; ``None`` conserva el build Sackmann-only para tests.

    Returns:
        Informe reproducible de runs, auditoría e idempotencia.

    Raises:
        EloRuntimeContractError: Mientras falte una interfaz core requerida.
        EloBuildError: Ante manifiesto, fuente, cálculo o persistencia inválidos.
    """

    from .engine import IDENTITY_EXCLUSION_RULE, EloEngine
    from .events import (
        EventColumns,
        EventValidationError,
        events_from_dataframe,
    )
    from .parameters import (
        ALGORITHM_VERSION,
        DEFAULT_ELO_PARAMETERS,
        EloParameters,
    )
    from .operational import OperationalEloOverlay
    from .store import (
        EloRunNotAvailableError,
        EloStore,
        EloStoreError,
    )

    selected_genders = _normalise_gender_selection(gender)
    if isinstance(chunksize, bool) or not isinstance(chunksize, int):
        raise TypeError("chunksize debe ser un entero positivo.")
    if chunksize <= 0:
        raise ValueError("chunksize debe ser positivo.")
    if not isinstance(force, bool):
        raise TypeError("force debe ser booleano.")
    if not isinstance(source_date_policy, SourceDatePolicy):
        raise TypeError("source_date_policy debe ser SourceDatePolicy.")
    if operational_overlay is not None and not isinstance(
        operational_overlay, OperationalEloOverlay
    ):
        raise TypeError(
            "operational_overlay debe ser OperationalEloOverlay o None."
        )
    resolved_parameters = (
        DEFAULT_ELO_PARAMETERS if parameters is None else parameters
    )
    if not isinstance(resolved_parameters, EloParameters):
        raise TypeError("parameters debe ser EloParameters o None.")
    checked_quarantine = _normalise_identity_exclusion_after_dates(
        identity_exclusion_after_dates
    )

    verified = load_verified_manifest(
        Path(manifest_path),
        Path(raw_dir),
    )
    if (
        operational_overlay is not None
        and operational_overlay.base_source_commit != verified.source_commit
    ):
        raise EloBuildError(
            "El overlay operativo no pertenece al commit Sackmann base."
        )
    gender_manifests = {
        selected_gender: verified.select_gender(selected_gender)
        for selected_gender in selected_genders
    }
    elo_parameter_payload = resolved_parameters.as_dict()
    code_inventory = build_code_inventory(PROJECT_ROOT, ELO_CODE_PATHS)
    persisted_parameter_payload = {
        **elo_parameter_payload,
        "artifact_code_inventory": list(code_inventory),
        "source_date_policy": source_date_policy.as_dict(),
    }
    gender_quarantine_rules = {
        selected_gender: {
            key: exclusion_after_date
            for key, exclusion_after_date in checked_quarantine.items()
            if key[0] == selected_gender
        }
        for selected_gender in selected_genders
    }
    gender_effective_quarantine_rules = {
        selected_gender: {
            key: source_date_policy.availability_date(source_date)
            for key, source_date in gender_quarantine_rules[
                selected_gender
            ].items()
        }
        for selected_gender in selected_genders
    }
    fingerprints: dict[Gender, str] = {
        selected_gender: compute_input_fingerprint(
            gender_manifests[selected_gender],
            elo_parameter_payload,
            ALGORITHM_VERSION,
            gender_quarantine_rules[selected_gender],
            code_inventory,
            source_date_policy,
            (
                None
                if operational_overlay is None
                else operational_overlay.contract_for_gender(
                    selected_gender
                )
            ),
        )
        for selected_gender in selected_genders
    }
    run_ids: dict[Gender, str] = {
        selected_gender: _run_id(
            selected_gender,
            fingerprints[selected_gender],
        )
        for selected_gender in selected_genders
    }

    resolved_database = Path(database_path).resolve()
    if resolved_database.is_file() and not force:
        existing_store = EloStore(resolved_database)
        existing_runs: dict[Gender, Any] = {}
        all_current = True
        try:
            for selected_gender in selected_genders:
                run = existing_store.resolve_complete_run(
                    gender=selected_gender,
                )
                existing_runs[selected_gender] = run
                if (
                    run.input_fingerprint
                    != fingerprints[selected_gender]
                    or run.algorithm_version != ALGORITHM_VERSION
                ):
                    all_current = False
        except (EloStoreError, EloRunNotAvailableError):
            all_current = False
        if all_current:
            current_run_ids = {
                selected_gender: str(
                    existing_runs[selected_gender].run_id
                )
                for selected_gender in selected_genders
            }
            return EloBuildReport(
                database_path=resolved_database,
                source_commit=verified.source_commit,
                algorithm_version=ALGORITHM_VERSION,
                input_fingerprints=fingerprints,
                run_ids=current_run_ids,
                skipped=True,
                audits=(),
                top_ratings={
                    selected_gender: _top_ratings_from_database(
                        resolved_database,
                        run_id=current_run_ids[selected_gender],
                    )
                    for selected_gender in selected_genders
                },
                operational_overlay=(
                    None
                    if operational_overlay is None
                    else operational_overlay.as_dict()
                ),
            )

    if resolved_database.is_file() and selected_genders != ("M", "F"):
        raise ExistingEloDatabaseError(
            "Una reconstrucción parcial no puede reemplazar una base existente. "
            "Use gender='all' o una ruta database_path independiente."
        )

    combined_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolved_database.parent.mkdir(parents=True, exist_ok=True)
    staging_path = _staging_path(
        resolved_database,
        combined_fingerprint,
    )
    spool_path = _spool_path(
        resolved_database,
        combined_fingerprint,
    )
    _cleanup_staging(staging_path)
    _cleanup_staging(spool_path)

    audits: list[GenderBuildAudit] = []
    top_ratings: dict[
        Gender,
        tuple[Mapping[str, object], ...],
    ] = {}
    try:
        source_counts = _spool_manifest_events(
            manifest=verified,
            gender=gender,
            chunksize=chunksize,
            spool_path=spool_path,
            source_date_policy=source_date_policy,
            operational_events=(
                ()
                if operational_overlay is None
                else operational_overlay.events
            ),
            base_cutoff_date=(
                None
                if operational_overlay is None
                else operational_overlay.cutoff_date
            ),
        )
        LOGGER.info(
            "Spool Elo validado: M=%d, F=%d.",
            source_counts.get("M", 0),
            source_counts.get("F", 0),
        )

        store = EloStore(staging_path)
        store.initialize()
        spool_columns = EventColumns(
            tour_level="tour_level",
            source_path="source_path",
            source_row="source_row_number",
            source_record_hash="source_record_hash",
            source_date="source_event_date",
        )
        for selected_gender in selected_genders:
            source_rows = int(source_counts.get(selected_gender, 0))
            operational_source_rows = (
                0
                if operational_overlay is None
                else len(
                    operational_overlay.events_for_gender(selected_gender)
                )
            )
            base_source_rows = source_rows - operational_source_rows
            if source_rows == 0:
                raise EloBuildError(
                    f"No hay partidos fuente para el género {selected_gender}."
                )
            run_id = run_ids[selected_gender]
            store.create_run(
                run_id=run_id,
                gender=selected_gender,
                source_commit=verified.source_commit,
                algorithm_version=ALGORITHM_VERSION,
                input_fingerprint=fingerprints[selected_gender],
                parameters={
                    **persisted_parameter_payload,
                    "identity_exclusion_rule": IDENTITY_EXCLUSION_RULE,
                    "identity_exclusion_after_dates": _identity_rule_payload(
                        gender_quarantine_rules[selected_gender]
                    ),
                    "identity_exclusion_effective_after_dates": (
                        _identity_rule_payload(
                            gender_effective_quarantine_rules[selected_gender]
                        )
                    ),
                    "operational_overlay": (
                        None
                        if operational_overlay is None
                        else operational_overlay.contract_for_gender(
                            selected_gender
                        )
                    ),
                },
            )
            engine = EloEngine(
                resolved_parameters,
                run_id=run_id,
                source_commit=verified.source_commit,
                identity_exclusion_after_dates=(
                    gender_effective_quarantine_rules[selected_gender]
                ),
            )
            included_events = 0
            excluded_events = 0
            date_blocks = 0
            rating_rows = 0
            exclusions: Counter[str] = Counter()
            for date_frame in _iter_spooled_date_frames(
                spool_path,
                gender=selected_gender,
                chunksize=chunksize,
            ):
                try:
                    events = events_from_dataframe(
                        date_frame,
                        source_commit=verified.source_commit,
                        columns=spool_columns,
                    )
                except EventValidationError as exc:
                    raise SourceIntegrityError(
                        "El spool no pudo reconstruir un evento validado: "
                        f"{exc}"
                    ) from exc
                if not events:
                    continue
                match_date = events[0].date
                block = engine.process_date_block(
                    match_date,
                    events,
                )
                store.write_date_block(
                    run_id=run_id,
                    block_date=match_date,
                    input_hash=_date_block_input_hash(events),
                    match_count=block.audit.included,
                    excluded_count=block.audit.excluded,
                    ratings=block.states,
                )
                included_events += block.audit.included
                excluded_events += block.audit.excluded
                exclusions.update(dict(block.audit.excluded_by_reason))
                date_blocks += 1
                rating_rows += len(block.states)
                if date_blocks % 500 == 0:
                    LOGGER.info(
                        "Elo %s: %d fechas, %d incluidos, %d excluidos.",
                        selected_gender,
                        date_blocks,
                        included_events,
                        excluded_events,
                    )
            if date_blocks == 0:
                raise EloBuildError(
                    f"No se procesó ninguna fecha para {selected_gender}."
                )
            store.activate_run(run_id=run_id)
            audits.append(
                GenderBuildAudit(
                    gender=selected_gender,
                    source_rows=source_rows,
                    included_events=included_events,
                    excluded_events=excluded_events,
                    exact_duplicates=int(exclusions["duplicate"]),
                    date_blocks=date_blocks,
                    rating_rows=rating_rows,
                    exclusions_by_reason=dict(sorted(exclusions.items())),
                    base_source_rows=base_source_rows,
                    operational_source_rows=operational_source_rows,
                )
            )
            LOGGER.info(
                "Elo %s completado: %d fechas y %d estados.",
                selected_gender,
                date_blocks,
                rating_rows,
            )

        integrity_connection = sqlite3.connect(staging_path)
        try:
            integrity = integrity_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
        finally:
            integrity_connection.close()
        if integrity is None or integrity[0] != "ok":
            raise EloBuildError(
                f"SQLite integrity_check falló: {integrity!r}."
            )
        for selected_gender in selected_genders:
            top_ratings[selected_gender] = _top_ratings_from_database(
                staging_path,
                run_id=run_ids[selected_gender],
            )
        os.replace(staging_path, resolved_database)
    except BaseException:
        _cleanup_staging(staging_path)
        raise
    finally:
        _cleanup_staging(spool_path)

    return EloBuildReport(
        database_path=resolved_database,
        source_commit=verified.source_commit,
        algorithm_version=ALGORITHM_VERSION,
        input_fingerprints=fingerprints,
        run_ids=run_ids,
        skipped=False,
        audits=tuple(audits),
        top_ratings=top_ratings,
        operational_overlay=(
            None
            if operational_overlay is None
            else operational_overlay.as_dict()
        ),
    )

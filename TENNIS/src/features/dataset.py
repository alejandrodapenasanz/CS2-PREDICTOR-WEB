"""Construcción reproducible de datasets históricos causales por género.

El proceso verifica el snapshot Sackmann, ordena las filas mediante un spool
SQLite, congela cada fecha y genera las features antes de aplicar sus
resultados. Elo y el estado de forma/H2H/descanso avanzan únicamente después de
cerrar todas las filas de ``D``. Los artefactos Parquet se publican como un run
inmutable desde staging y el puntero activo se reemplaza atómicamente al final.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Final, Iterable, Literal, Mapping, Sequence, cast
import uuid

import pandas as pd

from ..artifact_integrity import (
    build_code_inventory,
    canonicalise_code_inventory,
)
from ..config import (
    FEATURES_PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from ..data_loaders import load_players
from ..elo import (
    ALGORITHM_VERSION,
    DEFAULT_ELO_PARAMETERS,
    EloEngine,
    EloParameters,
    EventColumns,
    Gender,
    MatchEvent,
    RatedMatch,
    events_from_dataframe,
    load_verified_manifest,
)
from ..elo.build import (
    ELO_CODE_PATHS,
    EXPECTED_SOURCE_REPOSITORY,
    VerifiedManifest,
    compute_input_fingerprint as compute_elo_input_fingerprint,
)
from ..elo.engine import (
    IDENTITY_EXCLUSION_RULE,
    normalise_identity_exclusion_after_dates,
)
from ..temporal import (
    DEFAULT_SOURCE_DATE_POLICY,
    SourceDatePolicy,
)
from .orientation import orient_match
from .artifacts import (
    FeatureArtifactError,
    activate_feature_run,
    find_verified_feature_run,
    preflight_feature_publication,
    publish_feature_run,
)
from .parameters import (
    DEFAULT_FEATURE_PARAMETERS,
    FEATURE_SCHEMA_VERSION,
    FeatureParameters,
)
from .players import PlayerAgeIndex
from .rankings import RankingIndex
from .schema import (
    AUDIT_COLUMNS,
    TRAINING_COLUMNS,
    validate_training_row,
    training_arrow_schema,
)
from .source import (
    iter_spooled_feature_date_frames,
    spool_manifest_feature_rows,
)
from .state import (
    CausalHistoryState,
    HistoricalMatchResult,
)
from .vector import (
    EloFeatureSnapshot,
    MatchFeatureRequest,
    MODEL_FEATURE_COLUMNS,
    assemble_match_feature_vector,
)


GenderSelection = Literal["M", "F", "all"]
LOGGER = logging.getLogger(__name__)

DEFAULT_BUILD_CHUNKSIZE: Final[int] = 50_000
DEFAULT_PARQUET_BUFFER_ROWS: Final[int] = 10_000
MANIFEST_FILENAME: Final[str] = "manifest.json"
CONFLICTS_FILENAME: Final[str] = "ranking_conflicts.csv"
_SHA1_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_MATCH_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "atp_main",
        "atp_qual_chall",
        "atp_futures",
        "wta_main",
        "wta_qual_itf",
    }
)
_AUXILIARY_CATEGORY_BY_GENDER: Final[
    Mapping[Gender, frozenset[str]]
] = {
    "M": frozenset({"atp_rankings"}),
    "F": frozenset({"wta_rankings"}),
}
_PLAYER_PATH_BY_GENDER: Final[Mapping[Gender, str]] = {
    "M": "atp/atp_players.csv",
    "F": "wta/wta_players.csv",
}
FEATURE_CODE_PATHS: Final[tuple[str, ...]] = (
    "scripts/build_features.py",
    "src/artifact_integrity.py",
    "src/data_loaders.py",
    "src/elo/build.py",
    "src/elo/engine.py",
    "src/elo/events.py",
    "src/elo/parameters.py",
    "src/elo/types.py",
    "src/features/dataset.py",
    "src/features/artifacts.py",
    "src/features/levels.py",
    "src/features/market.py",
    "src/features/orientation.py",
    "src/features/parameters.py",
    "src/features/players.py",
    "src/features/rankings.py",
    "src/features/schema.py",
    "src/features/source.py",
    "src/features/state.py",
    "src/features/vector.py",
    "src/identity_integrity.py",
    "src/temporal.py",
)


class FeatureDatasetError(RuntimeError):
    """Indica que no se pudo construir o verificar el dataset causal."""


class FeatureDatasetSourceError(FeatureDatasetError):
    """Indica una divergencia del manifiesto o de un archivo auxiliar."""


@dataclass(frozen=True, slots=True)
class VerifiedAuxiliaryInventory:
    """Snapshot verificado de maestros y rankings usados en inferencia."""

    source_commit: str
    files: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class RankingBuildAudit:
    """Resume la carga y cuarentena de rankings de un género."""

    gender: Gender
    source_files: int
    source_rows: int
    indexed_rows: int
    players: int
    exact_duplicates: int
    conflict_keys: int
    conflict_observations: int
    min_date: date
    max_date: date


@dataclass(frozen=True, slots=True)
class GenderDatasetAudit:
    """Resume filas, fechas, balance y exclusiones de un Parquet."""

    gender: Gender
    source_rows: int
    training_rows: int
    excluded_rows: int
    date_blocks: int
    min_date: date
    max_date: date
    y_ones: int
    y_rate: float
    output_path: Path
    output_size: int
    output_sha256: str
    exclusions_by_reason: Mapping[str, int]
    canonical_levels: Mapping[str, int]
    missing_rank_sides: int
    missing_age_sides: int


@dataclass(frozen=True, slots=True)
class FeatureDatasetBuildReport:
    """Resultado reproducible de construir o reutilizar los datasets."""

    output_dir: Path
    source_commit: str
    fingerprint: str
    schema_version: str
    skipped: bool
    datasets: tuple[GenderDatasetAudit, ...]
    rankings: tuple[RankingBuildAudit, ...]
    conflict_inventory_path: Path
    conflict_inventory_rows: int


@dataclass(frozen=True, slots=True)
class _SourceContext:
    """Contexto no incluido en ``MatchEvent`` para una fila verificada."""

    source_family: str
    tourney_name: str | None
    best_of: int | None


def _normalise_gender_selection(value: object) -> tuple[Gender, ...]:
    """Valida ``M``, ``F`` o ``all`` sin aceptar alias implícitos."""

    if value == "M":
        return ("M",)
    if value == "F":
        return ("F",)
    if value == "all":
        return ("M", "F")
    raise ValueError("gender debe ser exactamente 'M', 'F' o 'all'.")


def _validated_positive_integer(value: object, field_name: str) -> int:
    """Valida un entero positivo y rechaza booleanos."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} debe ser un entero positivo.")
    return value


def _ensure_project_path(path: Path, field_name: str) -> Path:
    """Restringe entradas y salidas al árbol TENNIS del proyecto."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise FeatureDatasetError(
            f"{field_name} debe permanecer dentro de TENNIS/: {resolved}."
        )
    return resolved


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques sobre un archivo existente."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureDatasetSourceError(
            f"No se pudo hashear {path}."
        ) from exc
    return digest.hexdigest()


def _git_blob_sha(path: Path, expected_size: int) -> str:
    """Calcula la identidad SHA-1 Git de un blob auxiliar."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {expected_size}\0".encode("ascii"))
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureDatasetSourceError(
            f"No se pudo verificar el blob auxiliar {path}."
        ) from exc
    return digest.hexdigest()


def _safe_manifest_local_path(raw_dir: Path, relative_path: str) -> Path:
    """Resuelve una ruta POSIX del manifiesto sin permitir traversal."""

    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FeatureDatasetSourceError(
            f"Ruta auxiliar insegura: {relative_path!r}."
        )
    resolved = (raw_dir / Path(*pure.parts)).resolve()
    if not resolved.is_relative_to(raw_dir.resolve()):
        raise FeatureDatasetSourceError(
            f"La ruta auxiliar sale de data/raw: {relative_path!r}."
        )
    return resolved


def _load_stable_source_inventory(
    manifest_path: Path,
    raw_dir: Path,
    *,
    genders: Sequence[Gender],
    verified_manifest: VerifiedManifest,
) -> tuple[Mapping[str, object], ...]:
    """Valida auxiliares y devuelve un inventario estable para fingerprint."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureDatasetSourceError(
            f"No se pudo leer el manifiesto {manifest_path}."
        ) from exc
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, list):
        raise FeatureDatasetSourceError(
            "El manifiesto no contiene una lista files."
        )
    selected_match_paths = {
        match_file.source_path
        for match_file in verified_manifest.match_files
        if match_file.gender in genders
    }
    required_player_paths = {
        _PLAYER_PATH_BY_GENDER[gender] for gender in genders
    }
    selected_auxiliary_categories = set().union(
        *(_AUXILIARY_CATEGORY_BY_GENDER[gender] for gender in genders)
    )

    inventory: list[Mapping[str, object]] = []
    seen_paths: set[str] = set()
    seen_players: set[str] = set()
    ranking_counts: Counter[Gender] = Counter()
    for item in files:
        if not isinstance(item, Mapping):
            raise FeatureDatasetSourceError(
                "Una entrada files no es un objeto."
            )
        relative_path = item.get("local_path")
        category = item.get("category")
        if not isinstance(relative_path, str) or not isinstance(category, str):
            raise FeatureDatasetSourceError(
                "Una entrada files carece de local_path/category textual."
            )
        selected = (
            relative_path in selected_match_paths
            or relative_path in required_player_paths
            or category in selected_auxiliary_categories
        )
        if not selected:
            continue
        if relative_path in seen_paths:
            raise FeatureDatasetSourceError(
                f"Ruta duplicada en manifiesto: {relative_path!r}."
            )
        seen_paths.add(relative_path)
        expected_size = item.get("size")
        expected_blob = item.get("git_blob_sha")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_blob, str)
            or _SHA1_PATTERN.fullmatch(expected_blob) is None
        ):
            raise FeatureDatasetSourceError(
                f"Metadatos de blob inválidos para {relative_path!r}."
            )
        local_path = _safe_manifest_local_path(raw_dir, relative_path)
        if not local_path.is_file() or local_path.stat().st_size != expected_size:
            raise FeatureDatasetSourceError(
                f"El auxiliar {relative_path!r} falta o cambió de tamaño."
            )
        if relative_path not in selected_match_paths:
            observed_blob = _git_blob_sha(local_path, expected_size)
            if observed_blob != expected_blob:
                raise FeatureDatasetSourceError(
                    f"El blob auxiliar {relative_path!r} no coincide."
                )
        if relative_path in required_player_paths:
            seen_players.add(relative_path)
        if category == "atp_rankings":
            ranking_counts["M"] += 1
        elif category == "wta_rankings":
            ranking_counts["F"] += 1
        inventory.append(
            {
                "category": category,
                "local_path": relative_path,
                "size": expected_size,
                "git_blob_sha": expected_blob,
            }
        )

    if seen_players != required_player_paths:
        raise FeatureDatasetSourceError(
            "Faltan maestros de jugadores en el manifiesto: "
            f"{sorted(required_player_paths.difference(seen_players))}."
        )
    missing_rankings = [
        gender for gender in genders if ranking_counts[gender] == 0
    ]
    if missing_rankings:
        raise FeatureDatasetSourceError(
            f"Faltan rankings en el manifiesto para {missing_rankings}."
        )
    if selected_match_paths.difference(seen_paths):
        raise FeatureDatasetSourceError(
            "El inventario estable omitió CSV de partidos verificados."
        )
    return tuple(
        sorted(
            inventory,
            key=lambda item: str(item["local_path"]),
        )
    )


def verify_stable_source_inventory(
    manifest_path: Path,
    raw_dir: Path,
    *,
    genders: Sequence[Gender] = ("M", "F"),
    verified_manifest: VerifiedManifest | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Verifica blobs de partidos, jugadores y rankings del snapshot activo.

    Esta frontera pública permite que la inferencia diaria aplique el mismo
    contrato de tamaño y SHA-1 Git que la construcción histórica antes de
    cargar maestros o rankings auxiliares.
    """

    selected = tuple(genders)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(gender not in {"M", "F"} for gender in selected)
    ):
        raise ValueError(
            "genders debe contener una selección única de 'M' y/o 'F'."
        )
    resolved_manifest = Path(manifest_path).resolve()
    resolved_raw = Path(raw_dir).resolve()
    verified = (
        load_verified_manifest(resolved_manifest, resolved_raw)
        if verified_manifest is None
        else verified_manifest
    )
    if (
        verified.manifest_path.resolve() != resolved_manifest
        or verified.raw_dir.resolve() != resolved_raw
    ):
        raise FeatureDatasetSourceError(
            "verified_manifest no corresponde a manifest_path/raw_dir."
        )
    return _load_stable_source_inventory(
        resolved_manifest,
        resolved_raw,
        genders=cast(Sequence[Gender], selected),
        verified_manifest=verified,
    )


def verify_auxiliary_source_inventory(
    manifest_path: Path,
    raw_dir: Path,
    *,
    genders: Sequence[Gender] = ("M", "F"),
) -> VerifiedAuxiliaryInventory:
    """Verifica jugadores y rankings sin volver a hashear el histórico.

    Los Parquet y runs Elo ya fijan los partidos históricos por fingerprint.
    La predicción diaria sí vuelve a abrir maestros y rankings, por lo que
    verifica tamaño y SHA-1 Git de esos auxiliares inmediatamente antes de
    cargarlos.
    """

    selected = tuple(genders)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(gender not in {"M", "F"} for gender in selected)
    ):
        raise ValueError(
            "genders debe contener una selección única de 'M' y/o 'F'."
        )
    resolved_manifest = Path(manifest_path).resolve()
    resolved_raw = Path(raw_dir).resolve()
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureDatasetSourceError(
            f"No se pudo leer el manifiesto {resolved_manifest}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise FeatureDatasetSourceError(
            "El manifiesto Sackmann no es un objeto JSON."
        )
    if payload.get("source_repository") != EXPECTED_SOURCE_REPOSITORY:
        raise FeatureDatasetSourceError(
            "El manifiesto no pertenece al mirror Sackmann autorizado."
        )
    commit = payload.get("source_commit")
    if (
        not isinstance(commit, str)
        or _SHA1_PATTERN.fullmatch(commit) is None
    ):
        raise FeatureDatasetSourceError(
            "El manifiesto Sackmann carece de un commit válido."
        )
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise FeatureDatasetSourceError(
            "El manifiesto Sackmann no contiene una lista files."
        )

    required_players = {
        _PLAYER_PATH_BY_GENDER[gender] for gender in selected
    }
    ranking_categories = set().union(
        *(_AUXILIARY_CATEGORY_BY_GENDER[gender] for gender in selected)
    )
    seen_paths: set[str] = set()
    seen_players: set[str] = set()
    ranking_counts: Counter[Gender] = Counter()
    inventory: list[Mapping[str, object]] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise FeatureDatasetSourceError(
                "Una entrada files no es un objeto."
            )
        relative_path = item.get("local_path")
        category = item.get("category")
        if not isinstance(relative_path, str) or not isinstance(category, str):
            raise FeatureDatasetSourceError(
                "Una entrada files carece de local_path/category textual."
            )
        if (
            relative_path not in required_players
            and category not in ranking_categories
        ):
            continue
        if relative_path in seen_paths:
            raise FeatureDatasetSourceError(
                f"Ruta auxiliar duplicada: {relative_path!r}."
            )
        seen_paths.add(relative_path)
        expected_size = item.get("size")
        expected_blob = item.get("git_blob_sha")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_blob, str)
            or _SHA1_PATTERN.fullmatch(expected_blob) is None
        ):
            raise FeatureDatasetSourceError(
                f"Metadatos de blob inválidos para {relative_path!r}."
            )
        local_path = _safe_manifest_local_path(
            resolved_raw,
            relative_path,
        )
        if not local_path.is_file() or local_path.stat().st_size != expected_size:
            raise FeatureDatasetSourceError(
                f"El auxiliar {relative_path!r} falta o cambió de tamaño."
            )
        if _git_blob_sha(local_path, expected_size) != expected_blob:
            raise FeatureDatasetSourceError(
                f"El blob auxiliar {relative_path!r} no coincide."
            )
        if relative_path in required_players:
            seen_players.add(relative_path)
        if category == "atp_rankings":
            ranking_counts["M"] += 1
        elif category == "wta_rankings":
            ranking_counts["F"] += 1
        inventory.append(
            {
                "category": category,
                "local_path": relative_path,
                "size": expected_size,
                "git_blob_sha": expected_blob,
            }
        )
    if seen_players != required_players:
        raise FeatureDatasetSourceError(
            "Faltan maestros de jugadores en el manifiesto: "
            f"{sorted(required_players.difference(seen_players))}."
        )
    missing_rankings = [
        gender for gender in selected if ranking_counts[gender] == 0
    ]
    if missing_rankings:
        raise FeatureDatasetSourceError(
            f"Faltan rankings en el manifiesto para {missing_rankings}."
        )
    return VerifiedAuxiliaryInventory(
        source_commit=commit,
        files=tuple(
            sorted(
                inventory,
                key=lambda item: str(item["local_path"]),
            )
        ),
    )


def _input_fingerprint(
    *,
    source_commit: str,
    genders: Sequence[Gender],
    inventory: Sequence[Mapping[str, object]],
    feature_parameters: FeatureParameters,
    elo_parameters: EloParameters,
    identity_exclusion_after_dates: Mapping[tuple[Gender, int], date],
    source_date_policy: SourceDatePolicy,
    elo_contracts: Mapping[str, Mapping[str, object]],
    code_inventory: Iterable[Mapping[str, object]] | None = None,
) -> str:
    """Resume fuentes, fórmulas, código y esquema de forma reproducible."""

    resolved_code_inventory = canonicalise_code_inventory(
        build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS)
        if code_inventory is None
        else code_inventory
    )
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "genders": list(genders),
        "inventory": list(inventory),
        "feature_parameters": feature_parameters.as_dict(),
        "elo_algorithm_version": ALGORITHM_VERSION,
        "elo_parameters": elo_parameters.as_dict(),
        "training_columns": list(TRAINING_COLUMNS),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "code_inventory": list(resolved_code_inventory),
        "historical_identity_exclusion": "disabled_noncausal_dob_metadata",
        "identity_diagnostic_first_match_dates": [
            [gender, player_id, first_match_date.isoformat()]
            for (gender, player_id), first_match_date in sorted(
                identity_exclusion_after_dates.items()
            )
        ],
        "source_date_policy": source_date_policy.as_dict(),
        "elo_contracts": dict(elo_contracts),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_elo_contracts(
    manifest: VerifiedManifest,
    *,
    genders: Sequence[Gender],
    elo_parameters: EloParameters,
    source_date_policy: SourceDatePolicy,
) -> Mapping[str, Mapping[str, object]]:
    """Reproduce exactamente el run Elo compatible con estas features."""

    code_inventory = build_code_inventory(PROJECT_ROOT, ELO_CODE_PATHS)
    persisted_parameters: Mapping[str, object] = {
        **elo_parameters.as_dict(),
        "artifact_code_inventory": list(code_inventory),
        "source_date_policy": source_date_policy.as_dict(),
        "identity_exclusion_rule": IDENTITY_EXCLUSION_RULE,
        "identity_exclusion_after_dates": [],
        "identity_exclusion_effective_after_dates": [],
    }
    return {
        gender: {
            "input_fingerprint": compute_elo_input_fingerprint(
                manifest.select_gender(gender),
                elo_parameters.as_dict(),
                ALGORITHM_VERSION,
                {},
                code_inventory,
                source_date_policy,
            ),
            "algorithm_version": ALGORITHM_VERSION,
            "parameters": dict(persisted_parameters),
            "source_date_policy": source_date_policy.as_dict(),
            "historical_identity_exclusion": "disabled",
        }
        for gender in genders
    }


def _optional_text(value: object) -> str | None:
    """Convierte texto raw vacío/nulo en ``None`` sin crear contenido."""

    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _source_context_by_provenance(
    frame: pd.DataFrame,
) -> Mapping[tuple[str, int], _SourceContext]:
    """Indexa contexto extra por la misma procedencia estable del evento."""

    contexts: dict[tuple[str, int], _SourceContext] = {}
    columns = (
        "source_path",
        "source_row_number",
        "source_family",
        "tourney_name",
        "best_of",
    )
    for values in frame.loc[:, list(columns)].itertuples(
        index=False,
        name=None,
    ):
        source_path = str(values[0])
        source_row = int(values[1])
        best_of = None if pd.isna(values[4]) else int(values[4])
        key = (source_path, source_row)
        if key in contexts:
            raise FeatureDatasetSourceError(
                f"Procedencia duplicada dentro de la fecha: {key}."
            )
        contexts[key] = _SourceContext(
            source_family=str(values[2]),
            tourney_name=_optional_text(values[3]),
            best_of=best_of,
        )
    return contexts


def _elo_sides(
    rated: RatedMatch,
    player_a_id: int,
) -> tuple[EloFeatureSnapshot, EloFeatureSnapshot]:
    """Orienta los ratings prepartido del ganador/perdedor como A/B."""

    winner = EloFeatureSnapshot.from_pre_match_rating(
        rated.winner_before
    )
    loser = EloFeatureSnapshot.from_pre_match_rating(
        rated.loser_before
    )
    if player_a_id == rated.event.winner_id:
        return winner, loser
    if player_a_id == rated.event.loser_id:
        return loser, winner
    raise FeatureDatasetError(
        "La orientación produjo un jugador A ajeno al evento."
    )


def _training_row(
    *,
    rated: RatedMatch,
    context: _SourceContext,
    history_state: CausalHistoryState,
    ranking_index: RankingIndex,
    age_index: PlayerAgeIndex,
    feature_parameters: FeatureParameters,
    source_date_policy: SourceDatePolicy,
) -> dict[str, object]:
    """Construye una fila completa antes de incorporar el resultado de D."""

    event = rated.event
    orientation = orient_match(
        event.winner_id,
        event.loser_id,
        gender=event.gender,
        source_record_hash=event.source_record_hash,
        seed=feature_parameters.orientation_seed,
    )
    elo_a, elo_b = _elo_sides(rated, orientation.player_a_id)
    history = history_state.snapshot(
        event.gender,
        orientation.player_a_id,
        orientation.player_b_id,
        surface=event.surface,
        as_of_date=event.date,
    )
    ranking_a, ranking_b = ranking_index.get_many(
        (orientation.player_a_id, orientation.player_b_id),
        event.date,
    )
    age_a, age_b = age_index.get_pair(
        event.gender,
        orientation.player_a_id,
        orientation.player_b_id,
        as_of_date=event.date,
    )
    request = MatchFeatureRequest(
        gender=event.gender,
        player_a_id=orientation.player_a_id,
        player_b_id=orientation.player_b_id,
        as_of_date=event.date,
        surface=event.surface,
        tour_level_raw=event.tour_level,
        source_family=context.source_family,
        best_of=context.best_of,
        round=event.round,
    )
    vector = assemble_match_feature_vector(
        request,
        elo_a=elo_a,
        elo_b=elo_b,
        history=history,
        ranking_a=ranking_a,
        ranking_b=ranking_b,
        age_a=age_a,
        age_b=age_b,
    )
    row: dict[str, object] = {
        "record_id": event.source_record_hash,
        "source_commit": event.provenance.source_commit,
        "source_path": event.provenance.source_path,
        "source_row_number": event.provenance.source_row,
        "tourney_id": event.tourney_id,
        "tourney_name": context.tourney_name,
        "match_num": event.match_num,
        "result_available_date": source_date_policy.availability_date(
            event.result_source_date
        ),
    }
    if tuple(row) != AUDIT_COLUMNS:
        raise RuntimeError("La metadata diverge de AUDIT_COLUMNS.")
    row.update(vector.to_dict())
    row["y"] = orientation.y
    validate_training_row(row)
    return row


def _write_rows(
    writer: object,
    rows: list[dict[str, object]],
    schema: object,
) -> None:
    """Convierte un buffer validado en una tabla y lo añade al Parquet."""

    if not rows:
        return
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise FeatureDatasetError(
            "Falta pyarrow; instale TENNIS/requirements.txt."
        ) from exc
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table, row_group_size=len(rows))
    rows.clear()


def _build_gender_parquet(
    *,
    gender: Gender,
    manifest: VerifiedManifest,
    spool_path: Path,
    parquet_path: Path,
    ranking_index: RankingIndex,
    age_index: PlayerAgeIndex,
    feature_parameters: FeatureParameters,
    elo_parameters: EloParameters,
    chunksize: int,
    parquet_buffer_rows: int,
    fingerprint: str,
    identity_exclusion_after_dates: Mapping[tuple[Gender, int], date],
    source_date_policy: SourceDatePolicy,
) -> GenderDatasetAudit:
    """Procesa un universo cronológicamente y escribe su Parquet de staging."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FeatureDatasetError(
            "Falta pyarrow; instale TENNIS/requirements.txt."
        ) from exc

    source_rows = spool_manifest_feature_rows(
        manifest,
        gender=gender,
        spool_path=spool_path,
        chunksize=chunksize,
    )
    schema = training_arrow_schema().with_metadata(
        {
            b"feature_schema_version": FEATURE_SCHEMA_VERSION.encode("ascii"),
            b"input_fingerprint": fingerprint.encode("ascii"),
            b"gender": gender.encode("ascii"),
            b"model_feature_columns": json.dumps(
                MODEL_FEATURE_COLUMNS,
                separators=(",", ":"),
            ).encode("utf-8"),
        }
    )
    engine = EloEngine(
        elo_parameters,
        run_id=f"features-{gender}-{fingerprint[:16]}",
        source_commit=manifest.source_commit,
        identity_exclusion_after_dates={},
    )
    history = CausalHistoryState(
        recent_matches=feature_parameters.recent_matches,
        recent_months=feature_parameters.recent_months,
    )
    event_columns = EventColumns(
        tour_level="tourney_level",
        source_path="source_path",
        source_row="source_row_number",
        source_record_hash="source_record_hash",
    )
    writer = pq.ParquetWriter(
        parquet_path,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    buffer: list[dict[str, object]] = []
    training_rows = 0
    y_ones = 0
    date_blocks = 0
    first_date: date | None = None
    last_date: date | None = None
    exclusions: Counter[str] = Counter()
    canonical_levels: Counter[str] = Counter()
    missing_rank_sides = 0
    missing_age_sides = 0
    pending_by_availability: dict[date, list[MatchEvent]] = {}

    def apply_results_available_before(cutoff: date) -> None:
        """Aplica resultados con disponibilidad estrictamente anterior."""

        available_dates = sorted(
            available_date
            for available_date in pending_by_availability
            if available_date < cutoff
        )
        for available_date in available_dates:
            pending_events = pending_by_availability.pop(available_date)
            effective_events = tuple(
                replace(
                    event,
                    date=available_date,
                    source_date=event.result_source_date,
                )
                for event in pending_events
            )
            available_block = engine.process_date_block(
                available_date,
                effective_events,
            )
            if not available_block.rated_matches:
                continue
            source_dates = {
                rated.event.result_source_date
                for rated in available_block.rated_matches
            }
            if len(source_dates) != 1:
                raise FeatureDatasetError(
                    "Un bloque disponible mezcla tourney_date incompatibles."
                )
            source_match_date = next(iter(source_dates))
            history.apply_date_block(
                source_match_date,
                (
                    HistoricalMatchResult(
                        match_date=rated.event.result_source_date,
                        gender=rated.event.gender,
                        winner_id=rated.event.winner_id,
                        loser_id=rated.event.loser_id,
                        surface=rated.event.surface,
                    )
                    for rated in available_block.rated_matches
                ),
                availability_date=available_date,
            )
    try:
        for date_frame in iter_spooled_feature_date_frames(
            spool_path,
            gender=gender,
            chunksize=chunksize,
        ):
            events = events_from_dataframe(
                date_frame,
                source_commit=manifest.source_commit,
                columns=event_columns,
            )
            if not events:
                continue
            match_date = events[0].date
            apply_results_available_before(match_date)
            contexts = _source_context_by_provenance(date_frame)
            block = engine.preview_date_block(match_date, events)
            date_blocks += 1
            exclusions.update(
                dict(block.audit.excluded_by_reason)
            )
            for rated in block.rated_matches:
                event = rated.event
                if first_date is None:
                    first_date = event.date
                last_date = event.date
                context_key = (
                    event.provenance.source_path,
                    event.provenance.source_row,
                )
                try:
                    context = contexts[context_key]
                except KeyError as exc:
                    raise FeatureDatasetSourceError(
                        "No existe contexto para el evento "
                        f"{context_key!r}."
                    ) from exc
                row = _training_row(
                    rated=rated,
                    context=context,
                    history_state=history,
                    ranking_index=ranking_index,
                    age_index=age_index,
                    feature_parameters=feature_parameters,
                    source_date_policy=source_date_policy,
                )
                buffer.append(row)
                training_rows += 1
                y_ones += int(row["y"])
                canonical_levels[str(row["tour_level"])] += 1
                missing_rank_sides += int(row["ranking_missing_a"])
                missing_rank_sides += int(row["ranking_missing_b"])
                missing_age_sides += int(row["age_missing_a"])
                missing_age_sides += int(row["age_missing_b"])
                available_date = source_date_policy.availability_date(
                    event.result_source_date
                )
                pending_by_availability.setdefault(
                    available_date,
                    [],
                ).append(event)
                if len(buffer) >= parquet_buffer_rows:
                    _write_rows(writer, buffer, schema)
            if date_blocks % 500 == 0:
                LOGGER.info(
                    "Features %s: %d fechas, %d filas.",
                    gender,
                    date_blocks,
                    training_rows,
                )
        _write_rows(writer, buffer, schema)
    finally:
        writer.close()

    if (
        training_rows == 0
        or first_date is None
        or last_date is None
    ):
        raise FeatureDatasetError(
            f"No se produjo ninguna fila de entrenamiento para {gender}."
        )
    excluded_rows = source_rows - training_rows
    if excluded_rows != sum(exclusions.values()):
        raise FeatureDatasetError(
            "La auditoría de exclusiones no reconcilia con las filas fuente: "
            f"source={source_rows}, training={training_rows}, "
            f"reasons={sum(exclusions.values())}."
        )
    output_size = parquet_path.stat().st_size
    output_hash = _sha256_file(parquet_path)
    return GenderDatasetAudit(
        gender=gender,
        source_rows=source_rows,
        training_rows=training_rows,
        excluded_rows=excluded_rows,
        date_blocks=date_blocks,
        min_date=first_date,
        max_date=last_date,
        y_ones=y_ones,
        y_rate=y_ones / training_rows,
        output_path=parquet_path,
        output_size=output_size,
        output_sha256=output_hash,
        exclusions_by_reason=dict(sorted(exclusions.items())),
        canonical_levels=dict(sorted(canonical_levels.items())),
        missing_rank_sides=missing_rank_sides,
        missing_age_sides=missing_age_sides,
    )


def _ranking_audit(index: RankingIndex) -> RankingBuildAudit:
    """Convierte propiedades del índice en un informe serializable."""

    return RankingBuildAudit(
        gender=index.gender,
        source_files=len(index.source_files),
        source_rows=index.source_row_count,
        indexed_rows=index.row_count,
        players=index.player_count,
        exact_duplicates=index.exact_duplicate_count,
        conflict_keys=index.conflict_key_count,
        conflict_observations=index.conflict_observation_count,
        min_date=index.min_ranking_date,
        max_date=index.max_ranking_date,
    )


def _combine_conflict_inventories(
    source_paths: Sequence[Path],
    destination: Path,
    *,
    raw_dir: Path,
) -> int:
    """Combina inventarios por género y relativiza las rutas fuente."""

    fieldnames = (
        "gender",
        "player_id",
        "ranking_date",
        "source_file",
        "source_row",
        "rank",
        "points",
        "tours",
    )
    rows: list[dict[str, str]] = []
    for source_path in source_paths:
        try:
            with source_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as source:
                reader = csv.DictReader(source)
                if tuple(reader.fieldnames or ()) != fieldnames:
                    raise FeatureDatasetError(
                        f"Cabecera de conflictos inesperada: {source_path}."
                    )
                for row in reader:
                    local_source = Path(row["source_file"]).resolve()
                    if not local_source.is_relative_to(raw_dir.resolve()):
                        raise FeatureDatasetError(
                            "Una ruta de conflicto sale de data/raw."
                        )
                    row["source_file"] = local_source.relative_to(
                        raw_dir.resolve()
                    ).as_posix()
                    rows.append(dict(row))
        except OSError as exc:
            raise FeatureDatasetError(
                f"No se pudo combinar {source_path}."
            ) from exc
    rows.sort(
        key=lambda row: (
            row["gender"],
            int(row["player_id"]),
            row["ranking_date"],
            row["source_file"],
            int(row["source_row"]),
        )
    )
    try:
        with destination.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise FeatureDatasetError(
            f"No se pudo escribir {destination}."
        ) from exc
    return len(rows)


def _audit_to_manifest(audit: GenderDatasetAudit) -> Mapping[str, object]:
    """Serializa un informe de dataset con fechas ISO y nombre portable."""

    payload = asdict(audit)
    payload["min_date"] = audit.min_date.isoformat()
    payload["max_date"] = audit.max_date.isoformat()
    payload["output_path"] = audit.output_path.name
    return payload


def _ranking_to_manifest(audit: RankingBuildAudit) -> Mapping[str, object]:
    """Serializa un informe de rankings con fechas ISO."""

    payload = asdict(audit)
    payload["min_date"] = audit.min_date.isoformat()
    payload["max_date"] = audit.max_date.isoformat()
    return payload


def _parse_dataset_audit(
    payload: Mapping[str, object],
    output_dir: Path,
) -> GenderDatasetAudit:
    """Reconstruye un informe validado desde el manifiesto activo."""

    try:
        gender = cast(Gender, str(payload["gender"]))
        output_name = str(payload["output_path"])
        exclusions = dict(cast(Mapping[str, int], payload["exclusions_by_reason"]))
        levels = dict(cast(Mapping[str, int], payload["canonical_levels"]))
        return GenderDatasetAudit(
            gender=gender,
            source_rows=int(payload["source_rows"]),
            training_rows=int(payload["training_rows"]),
            excluded_rows=int(payload["excluded_rows"]),
            date_blocks=int(payload["date_blocks"]),
            min_date=date.fromisoformat(str(payload["min_date"])),
            max_date=date.fromisoformat(str(payload["max_date"])),
            y_ones=int(payload["y_ones"]),
            y_rate=float(payload["y_rate"]),
            output_path=output_dir / output_name,
            output_size=int(payload["output_size"]),
            output_sha256=str(payload["output_sha256"]),
            exclusions_by_reason=exclusions,
            canonical_levels=levels,
            missing_rank_sides=int(payload["missing_rank_sides"]),
            missing_age_sides=int(payload["missing_age_sides"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureDatasetError(
            "Dataset inválido en el manifiesto activo."
        ) from exc


def _parse_ranking_audit(
    payload: Mapping[str, object],
) -> RankingBuildAudit:
    """Reconstruye un informe de ranking desde JSON."""

    try:
        return RankingBuildAudit(
            gender=cast(Gender, str(payload["gender"])),
            source_files=int(payload["source_files"]),
            source_rows=int(payload["source_rows"]),
            indexed_rows=int(payload["indexed_rows"]),
            players=int(payload["players"]),
            exact_duplicates=int(payload["exact_duplicates"]),
            conflict_keys=int(payload["conflict_keys"]),
            conflict_observations=int(payload["conflict_observations"]),
            min_date=date.fromisoformat(str(payload["min_date"])),
            max_date=date.fromisoformat(str(payload["max_date"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureDatasetError(
            "Ranking inválido en el manifiesto activo."
        ) from exc


def _load_current_report(
    manifest_path: Path,
    *,
    artifact_dir: Path,
    output_dir: Path,
    fingerprint: str,
    genders: Sequence[Gender],
) -> FeatureDatasetBuildReport | None:
    """Devuelve un build verificable e idéntico o ``None`` para reconstruir."""

    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != FEATURE_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
        or not isinstance(payload.get("source_commit"), str)
        or _SHA1_PATTERN.fullmatch(
            cast(str, payload.get("source_commit"))
        )
        is None
    ):
        return None
    raw_datasets = payload.get("datasets")
    raw_rankings = payload.get("rankings")
    if not isinstance(raw_datasets, list) or not isinstance(raw_rankings, list):
        return None
    try:
        datasets = tuple(
            _parse_dataset_audit(item, artifact_dir)
            for item in raw_datasets
            if isinstance(item, Mapping)
        )
        rankings = tuple(
            _parse_ranking_audit(item)
            for item in raw_rankings
            if isinstance(item, Mapping)
        )
    except FeatureDatasetError:
        return None
    if (
        tuple(audit.gender for audit in datasets) != tuple(genders)
        or tuple(audit.gender for audit in rankings) != tuple(genders)
    ):
        return None
    for audit in datasets:
        if (
            not audit.output_path.is_file()
            or audit.output_path.stat().st_size != audit.output_size
            or _sha256_file(audit.output_path) != audit.output_sha256
        ):
            return None
    conflicts = payload.get("conflict_inventory")
    if not isinstance(conflicts, Mapping):
        return None
    conflict_path = artifact_dir / str(conflicts.get("path", ""))
    expected_size = conflicts.get("size")
    expected_hash = conflicts.get("sha256")
    if (
        not conflict_path.is_file()
        or not isinstance(expected_size, int)
        or conflict_path.stat().st_size != expected_size
        or not isinstance(expected_hash, str)
        or _sha256_file(conflict_path) != expected_hash
    ):
        return None
    return FeatureDatasetBuildReport(
        output_dir=output_dir,
        source_commit=cast(str, payload["source_commit"]),
        fingerprint=fingerprint,
        schema_version=FEATURE_SCHEMA_VERSION,
        skipped=True,
        datasets=datasets,
        rankings=rankings,
        conflict_inventory_path=conflict_path,
        conflict_inventory_rows=int(conflicts.get("rows", 0)),
    )


def _safe_cleanup_staging(path: Path, output_dir: Path) -> None:
    """Elimina solo un workspace hijo creado por este módulo."""

    resolved = path.resolve()
    parent = output_dir.resolve()
    if (
        resolved.parent != parent
        or not resolved.name.startswith(".feature-build-")
    ):
        raise FeatureDatasetError(
            f"Se rechazó limpiar una ruta de staging insegura: {resolved}."
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _create_build_workspace(output_dir: Path) -> Path:
    """Crea un workspace unico heredando la ACL del almacen.

    En Windows con Python 3.13, ``tempfile.mkdtemp`` crea deliberadamente el
    directorio con una DACL privada equivalente a ``0o700``. Al publicar el
    hijo mediante ``os.replace`` esa DACL viaja al run inmutable y otro
    proceso puede quedar sin lectura. ``Path.mkdir`` usa la herencia normal
    del padre, que es el contrato necesario para los artefactos compartidos.
    """

    resolved_output = output_dir.resolve()
    for _ in range(32):
        candidate = resolved_output / f".feature-build-{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise FeatureDatasetError(
                "No se pudo crear el workspace heredable de features."
            ) from exc
        return candidate
    raise FeatureDatasetError(
        "No se pudo reservar un nombre unico para el workspace de features."
    )


def build_training_datasets(
    *,
    manifest_path: Path = SACKMANN_MANIFEST_PATH,
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = FEATURES_PROCESSED_DIR,
    gender: GenderSelection = "all",
    chunksize: int = DEFAULT_BUILD_CHUNKSIZE,
    parquet_buffer_rows: int = DEFAULT_PARQUET_BUFFER_ROWS,
    force: bool = False,
    feature_parameters: FeatureParameters = DEFAULT_FEATURE_PARAMETERS,
    elo_parameters: EloParameters = DEFAULT_ELO_PARAMETERS,
    identity_exclusion_after_dates: (
        Mapping[tuple[Gender, int], date] | None
    ) = None,
    source_date_policy: SourceDatePolicy = DEFAULT_SOURCE_DATE_POLICY,
) -> FeatureDatasetBuildReport:
    """Construye y publica datasets separados por género sin información futura.

    Args:
        manifest_path: Manifiesto Sackmann activo de la fase 2.
        raw_dir: Raíz local verificada de datos crudos.
        output_dir: Directorio procesado dentro de ``TENNIS/``.
        gender: ``M``, ``F`` o ``all``.
        chunksize: Filas máximas por lectura de CSV/SQLite.
        parquet_buffer_rows: Filas acumuladas antes de cada row group.
        force: Reconstruye aunque el fingerprint activo sea idéntico.
        feature_parameters: Ventanas, edad y semilla aprobadas.
        elo_parameters: Fórmula Elo de la fase 3.
        identity_exclusion_after_dates: Primeras apariciones diagnósticas de
            claves con DOB incompatible. Se incorporan al fingerprint y al
            bloqueo operativo actual, pero nunca seleccionan filas históricas.
        source_date_policy: Embargo causal entre fecha fuente y disponibilidad.

    Returns:
        Informe de publicación, balance, fuentes y cuarentenas.
    """

    genders = _normalise_gender_selection(gender)
    selected_chunksize = _validated_positive_integer(
        chunksize,
        "chunksize",
    )
    selected_buffer = _validated_positive_integer(
        parquet_buffer_rows,
        "parquet_buffer_rows",
    )
    if not isinstance(force, bool):
        raise ValueError("force debe ser booleano.")
    if not isinstance(feature_parameters, FeatureParameters):
        raise TypeError("feature_parameters debe ser FeatureParameters.")
    if not isinstance(elo_parameters, EloParameters):
        raise TypeError("elo_parameters debe ser EloParameters.")
    if not isinstance(source_date_policy, SourceDatePolicy):
        raise TypeError("source_date_policy debe ser SourceDatePolicy.")
    quarantine = normalise_identity_exclusion_after_dates(
        identity_exclusion_after_dates
    )

    resolved_manifest = _ensure_project_path(
        Path(manifest_path),
        "manifest_path",
    )
    resolved_raw = _ensure_project_path(Path(raw_dir), "raw_dir")
    resolved_output = _ensure_project_path(Path(output_dir), "output_dir")
    if (
        genders != ("M", "F")
        and resolved_output == FEATURES_PROCESSED_DIR.resolve()
    ):
        raise FeatureDatasetError(
            "Una construcción parcial no puede reemplazar el manifiesto "
            "canónico de ambos géneros. Use gender='all' o un --output-dir "
            "diagnóstico distinto dentro de TENNIS/."
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    try:
        preflight_feature_publication(output_dir=resolved_output)
    except FeatureArtifactError as exc:
        raise FeatureDatasetError(
            "La publicación generacional de features falló en preflight."
        ) from exc
    verified = load_verified_manifest(
        resolved_manifest,
        resolved_raw,
    )
    inventory = _load_stable_source_inventory(
        resolved_manifest,
        resolved_raw,
        genders=genders,
        verified_manifest=verified,
    )
    code_inventory = build_code_inventory(PROJECT_ROOT, FEATURE_CODE_PATHS)
    elo_contracts = _expected_elo_contracts(
        verified,
        genders=genders,
        elo_parameters=elo_parameters,
        source_date_policy=source_date_policy,
    )
    fingerprint = _input_fingerprint(
        source_commit=verified.source_commit,
        genders=genders,
        inventory=inventory,
        feature_parameters=feature_parameters,
        elo_parameters=elo_parameters,
        identity_exclusion_after_dates=quarantine,
        source_date_policy=source_date_policy,
        elo_contracts=elo_contracts,
        code_inventory=code_inventory,
    )
    existing = find_verified_feature_run(
        fingerprint,
        output_dir=resolved_output,
    )
    if not force:
        if existing is not None:
            activated = activate_feature_run(
                existing,
                output_dir=resolved_output,
            )
            current = _load_current_report(
                activated.manifest_path,
                artifact_dir=activated.run_dir,
                output_dir=resolved_output,
                fingerprint=fingerprint,
                genders=genders,
            )
            if current is None:
                raise FeatureDatasetError(
                    "El run verificado no pudo reconstruir su informe."
                )
            return current

    staging = _create_build_workspace(resolved_output)
    work_dir = staging / "work"
    publish_dir = staging / "publish"
    work_dir.mkdir()
    publish_dir.mkdir()
    dataset_audits: list[GenderDatasetAudit] = []
    ranking_audits: list[RankingBuildAudit] = []
    conflict_parts: list[Path] = []
    try:
        player_frames = [
            load_players(gender_value, raw_dir=resolved_raw)
            for gender_value in genders
        ]
        age_index = PlayerAgeIndex.from_dataframe(
            pd.concat(player_frames, ignore_index=True),
            reference_years=feature_parameters.age_reference_years,
            days_per_year=feature_parameters.days_per_year,
        )
        for selected_gender in genders:
            LOGGER.info(
                "Cargando rankings causales del género %s.",
                selected_gender,
            )
            ranking_index = RankingIndex.from_raw(
                selected_gender,
                raw_dir=resolved_raw,
            )
            ranking_audits.append(_ranking_audit(ranking_index))
            conflict_part = work_dir / (
                f"ranking_conflicts_{selected_gender}.csv"
            )
            ranking_index.write_conflict_inventory(conflict_part)
            conflict_parts.append(conflict_part)
            spool_path = work_dir / f"source_{selected_gender}.sqlite3"
            parquet_path = publish_dir / (
                f"training_{selected_gender}.parquet"
            )
            LOGGER.info(
                "Construyendo features históricas del género %s.",
                selected_gender,
            )
            dataset_audits.append(
                _build_gender_parquet(
                    gender=selected_gender,
                    manifest=verified,
                    spool_path=spool_path,
                    parquet_path=parquet_path,
                    ranking_index=ranking_index,
                    age_index=age_index,
                    feature_parameters=feature_parameters,
                    elo_parameters=elo_parameters,
                    chunksize=selected_chunksize,
                    parquet_buffer_rows=selected_buffer,
                    fingerprint=fingerprint,
                    identity_exclusion_after_dates=quarantine,
                    source_date_policy=source_date_policy,
                )
            )

        combined_conflicts = publish_dir / CONFLICTS_FILENAME
        conflict_rows = _combine_conflict_inventories(
            conflict_parts,
            combined_conflicts,
            raw_dir=resolved_raw,
        )
        conflict_size = combined_conflicts.stat().st_size
        conflict_hash = _sha256_file(combined_conflicts)
        manifest_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_repository": verified.source_repository,
            "source_commit": verified.source_commit,
            "fingerprint": fingerprint,
            "code_inventory": list(code_inventory),
            "genders": list(genders),
            "feature_parameters": feature_parameters.as_dict(),
            "elo_algorithm_version": ALGORITHM_VERSION,
            "elo_parameters": elo_parameters.as_dict(),
            "source_date_policy": source_date_policy.as_dict(),
            "elo_contracts": dict(elo_contracts),
            "training_columns": list(TRAINING_COLUMNS),
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "historical_odds_available": False,
            "identity_quarantine": {
                "keys": [
                    [gender_value, player_id]
                    for gender_value, player_id in sorted(quarantine.keys())
                ],
                "diagnostic_first_match_dates": [
                    {
                        "gender": gender_value,
                        "player_id": player_id,
                        "first_match_date": first_match_date.isoformat(),
                        "causal_evidence_date": None,
                    }
                    for (
                        gender_value,
                        player_id,
                    ), first_match_date in sorted(quarantine.items())
                ],
                "historical_exclusion_rule": "disabled_noncausal_dob_metadata",
                "usage_contract": "current_inference_block_only",
                "rows": len(quarantine),
            },
            "datasets": [
                _audit_to_manifest(audit) for audit in dataset_audits
            ],
            "rankings": [
                _ranking_to_manifest(audit) for audit in ranking_audits
            ],
            "conflict_inventory": {
                "path": CONFLICTS_FILENAME,
                "rows": conflict_rows,
                "size": conflict_size,
                "sha256": conflict_hash,
            },
        }
        staging_manifest = publish_dir / MANIFEST_FILENAME
        staging_manifest.write_text(
            json.dumps(
                manifest_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        published = publish_feature_run(
            publish_dir,
            fingerprint=fingerprint,
            output_dir=resolved_output,
        )
        raw_published_datasets = published.manifest.get("datasets")
        if not isinstance(raw_published_datasets, list):
            raise FeatureDatasetError(
                "El run publicado perdió su inventario de datasets."
            )
        published_audits = [
            _parse_dataset_audit(item, published.run_dir)
            for item in raw_published_datasets
            if isinstance(item, Mapping)
        ]
        conflict_destination = published.run_dir / CONFLICTS_FILENAME
    except BaseException:
        _safe_cleanup_staging(staging, resolved_output)
        raise
    _safe_cleanup_staging(staging, resolved_output)
    return FeatureDatasetBuildReport(
        output_dir=resolved_output,
        source_commit=verified.source_commit,
        fingerprint=fingerprint,
        schema_version=FEATURE_SCHEMA_VERSION,
        skipped=published.skipped,
        datasets=tuple(published_audits),
        rankings=tuple(ranking_audits),
        conflict_inventory_path=conflict_destination,
        conflict_inventory_rows=conflict_rows,
    )

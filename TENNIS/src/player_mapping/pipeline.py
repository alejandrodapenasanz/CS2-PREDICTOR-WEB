"""Orquesta el mapeo causal de una cartelera diaria a IDs Sackmann.

El slug de Tennis Explorer es la identidad primaria y se combina siempre con
el género. Para slugs nuevos, el módulo solo acepta una coincidencia exacta y
única entre apellido normalizado e inicial dentro de jugadores activos antes
de la fecha de la cartelera. Las colisiones y ausencias se conservan como no
resueltas; nunca se aplica matching difuso ni se adivina un jugador.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import (
    PLAYER_MAPPING_DATABASE_PATH,
    PLAYER_OVERRIDES_PATH,
    RAW_DATA_DIR,
    UNRESOLVED_PLAYERS_PATH,
)
from src.data_loaders import DataLoaderError, load_players

from .candidates import load_active_candidate_index
from .normalization import parse_visible_name
from .review import (
    UNRESOLVED_COLUMNS,
    OverrideRecord,
    ReviewKey,
    UnresolvedObservation,
    load_overrides,
    update_unresolved_queue,
)
from .store import MappingRecord, PlayerMappingStore
from .types import (
    ActiveCandidate,
    CandidateIndex,
    Gender,
    NameParsingError,
    PlayerMappingSchemaError,
    PlayerMappingSourceError,
    PlayerMappingValidationError,
)


REQUIRED_SCRAPER_COLUMNS: Final[tuple[str, ...]] = (
    "match_date",
    "tournament",
    "tour_level",
    "gender",
    "player_1_name",
    "player_1_slug",
    "player_2_name",
    "player_2_slug",
)

MAPPING_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "player_1_id",
    "player_2_id",
    "player_1_mapping_method",
    "player_2_mapping_method",
    "mapping_status",
)

MAPPING_STATUS_MAPPED: Final[str] = "mapped"
MAPPING_STATUS_UNMAPPED: Final[str] = "unmapped"


@dataclass(frozen=True, slots=True)
class _MasterPlayer:
    """Identidad mínima del maestro necesaria para validar overrides y caché."""

    gender: Gender
    player_id: int
    player_name: str | None
    ioc: str | None


@dataclass(frozen=True, slots=True)
class _Resolution:
    """Resultado interno de resolver un único lado de un partido."""

    record: MappingRecord | None
    observation: UnresolvedObservation | None


def _validate_pure_date(value: object, *, field_name: str) -> date:
    """Exige una fecha civil sin aceptar conversiones implícitas."""

    if not isinstance(value, date) or isinstance(value, datetime):
        raise PlayerMappingValidationError(
            f"{field_name} debe ser datetime.date, no datetime."
        )
    return value


def _require_scraper_schema(matches: pd.DataFrame) -> None:
    """Valida que el DataFrame contenga el contrato mínimo de la fase 4."""

    if not isinstance(matches, pd.DataFrame):
        raise PlayerMappingValidationError("matches debe ser un DataFrame.")
    missing = sorted(set(REQUIRED_SCRAPER_COLUMNS).difference(matches.columns))
    if missing:
        raise PlayerMappingSchemaError(
            "La cartelera no contiene las columnas requeridas de la fase 4: "
            f"{missing}."
        )


def _resolve_batch_date(
    matches: pd.DataFrame,
    explicit_date: date | None,
) -> date | None:
    """Obtiene el único corte diario y comprueba su coherencia."""

    selected_date = (
        None
        if explicit_date is None
        else _validate_pure_date(explicit_date, field_name="as_of_date")
    )
    if matches.empty:
        return selected_date

    try:
        converted = pd.to_datetime(matches["match_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerMappingSchemaError(
            "match_date contiene fechas inválidas."
        ) from exc
    if converted.isna().any():
        raise PlayerMappingSchemaError("match_date contiene valores nulos.")
    if getattr(converted.dt, "tz", None) is not None:
        raise PlayerMappingSchemaError(
            "match_date debe contener fechas civiles sin zona horaria."
        )
    observed_dates = tuple(
        sorted({timestamp.date() for timestamp in converted})
    )
    if len(observed_dates) != 1:
        raise PlayerMappingSchemaError(
            "El resolvedor acepta una única cartelera diaria por llamada; "
            f"se observaron {len(observed_dates)} fechas."
        )
    observed_date = observed_dates[0]
    if selected_date is not None and selected_date != observed_date:
        raise PlayerMappingSchemaError(
            "as_of_date no coincide con match_date: "
            f"{selected_date.isoformat()} != {observed_date.isoformat()}."
        )
    return observed_date


def _optional_text(value: object) -> str | None:
    """Devuelve texto sin espacios exteriores o ``None`` para un nulo."""

    if pd.isna(value):
        return None
    if not isinstance(value, str):
        raise PlayerMappingSchemaError(
            "Los nombres, slugs, niveles y torneos deben ser texto o nulos."
        )
    stripped = value.strip()
    return stripped or None


def _validate_daily_values(matches: pd.DataFrame) -> None:
    """Rechaza universos, nombres y metadatos diarios incoherentes."""

    if matches.empty:
        return
    if matches["gender"].isna().any():
        raise PlayerMappingSchemaError("gender contiene valores nulos.")
    observed_genders = set(matches["gender"].astype(str))
    if not observed_genders.issubset({"M", "F"}):
        raise PlayerMappingSchemaError(
            "gender solo puede contener exactamente 'M' o 'F'."
        )
    for column in (
        "tournament",
        "tour_level",
        "player_1_name",
        "player_2_name",
    ):
        for value in matches[column]:
            if _optional_text(value) is None:
                raise PlayerMappingSchemaError(
                    f"{column} contiene texto vacío o nulo."
                )


def _coerce_player_id(value: object, *, context: str) -> int:
    """Convierte un ID maestro a entero y rechaza decimales o nulos."""

    if pd.isna(value):
        raise PlayerMappingSchemaError(f"{context} contiene un player_id nulo.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlayerMappingSchemaError(
            f"{context} contiene un player_id no entero."
        ) from exc
    try:
        original_numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlayerMappingSchemaError(
            f"{context} contiene un player_id no numérico."
        ) from exc
    if numeric <= 0 or original_numeric != numeric:
        raise PlayerMappingSchemaError(
            f"{context} contiene un player_id no entero positivo."
        )
    return numeric


def _build_master_lookup(
    players: pd.DataFrame,
) -> dict[tuple[Gender, int], _MasterPlayer]:
    """Construye una tabla separada por género para validar cada ID usado."""

    if not isinstance(players, pd.DataFrame):
        raise PlayerMappingValidationError("players debe ser un DataFrame.")
    required = {"gender", "player_id", "player_name", "ioc"}
    missing = sorted(required.difference(players.columns))
    if missing:
        raise PlayerMappingSchemaError(
            f"El maestro de jugadores carece de columnas: {missing}."
        )

    lookup: dict[tuple[Gender, int], _MasterPlayer] = {}
    for row in players.loc[:, ["gender", "player_id", "player_name", "ioc"]].itertuples(
        index=False
    ):
        if row.gender not in {"M", "F"}:
            raise PlayerMappingSchemaError(
                "players.gender solo puede contener 'M' o 'F'."
            )
        gender: Gender = row.gender
        player_id = _coerce_player_id(
            row.player_id,
            context=f"players[{gender}]",
        )
        key = (gender, player_id)
        if key in lookup:
            raise PlayerMappingSchemaError(
                "El maestro repite la identidad "
                f"(gender={gender}, player_id={player_id})."
            )
        player_name = _optional_text(row.player_name)
        ioc = _optional_text(row.ioc)
        lookup[key] = _MasterPlayer(
            gender=gender,
            player_id=player_id,
            player_name=player_name,
            ioc=ioc,
        )
    return lookup


def _load_master_players(raw_dir: Path) -> pd.DataFrame:
    """Carga el maestro canónico y traduce sus fallos al dominio de mapeo."""

    try:
        return load_players(raw_dir=raw_dir)
    except DataLoaderError as exc:
        raise PlayerMappingSourceError(
            f"No se pudo cargar el maestro Sackmann desde {raw_dir}: {exc}"
        ) from exc


def _validate_override_targets(
    overrides: Mapping[ReviewKey, OverrideRecord],
    master_lookup: Mapping[tuple[Gender, int], _MasterPlayer],
) -> None:
    """Comprueba que cada override apunte a un ID del género declarado."""

    for key, override in overrides.items():
        if key != (override.gender, override.slug):
            raise PlayerMappingSchemaError(
                f"El índice de overrides no coincide con su registro: {key!r}."
            )
        master = master_lookup.get((override.gender, override.player_id))
        if master is None:
            raise PlayerMappingSourceError(
                "El override apunta a un ID inexistente en el género correcto: "
                f"gender={override.gender}, slug={override.slug!r}, "
                f"player_id={override.player_id}."
            )
        if master.player_name is None:
            raise PlayerMappingSourceError(
                "El override apunta a un jugador sin nombre maestro utilizable: "
                f"gender={override.gender}, player_id={override.player_id}."
            )


def _validate_candidate_indexes(
    indexes: Mapping[str, CandidateIndex],
    genders: set[Gender],
) -> dict[Gender, CandidateIndex]:
    """Valida índices inyectados y garantiza aislamiento por género."""

    selected: dict[Gender, CandidateIndex] = {}
    for gender in sorted(genders):
        if gender not in indexes:
            raise PlayerMappingValidationError(
                f"candidate_indexes no contiene el universo {gender}."
            )
        index = indexes[gender]
        if not isinstance(index, dict):
            raise PlayerMappingValidationError(
                f"candidate_indexes[{gender!r}] debe ser un diccionario."
            )
        for candidates in index.values():
            if not isinstance(candidates, tuple):
                raise PlayerMappingValidationError(
                    "Cada grupo de candidatos debe ser una tupla."
                )
            if any(candidate.gender != gender for candidate in candidates):
                raise PlayerMappingSchemaError(
                    "Un índice de candidatos mezcla universos de género."
                )
        selected[gender] = index
    return selected


def _load_candidate_indexes(
    *,
    genders: set[Gender],
    batch_date: date,
    raw_dir: Path,
) -> dict[Gender, CandidateIndex]:
    """Carga únicamente los índices causales necesarios para la cartelera."""

    return {
        gender: load_active_candidate_index(
            gender,
            batch_date,
            raw_dir=raw_dir,
        )
        for gender in sorted(genders)
    }


def _master_for_mapping(
    record: MappingRecord,
    master_lookup: Mapping[tuple[Gender, int], _MasterPlayer],
) -> _MasterPlayer:
    """Detecta cachés que ya no apuntan a una identidad maestra válida."""

    master = master_lookup.get((record.gender, record.player_id))
    if master is None:
        raise PlayerMappingSourceError(
            "La caché contiene un player_id ausente en el género correcto: "
            f"gender={record.gender}, slug={record.slug!r}, "
            f"player_id={record.player_id}."
        )
    return master


def _apply_override(
    *,
    store: PlayerMappingStore,
    existing: MappingRecord | None,
    override: OverrideRecord,
    visible_name: str,
    batch_date: date,
    master_lookup: Mapping[tuple[Gender, int], _MasterPlayer],
) -> MappingRecord:
    """Aplica una intervención manual solo cuando cambia el estado persistido."""

    master = master_lookup[(override.gender, override.player_id)]
    if master.player_name is None:
        raise PlayerMappingSourceError(
            "El jugador del override no tiene nombre maestro utilizable."
        )
    if (
        existing is not None
        and existing.player_id == override.player_id
        and existing.resolution_method == "manual_override"
    ):
        return existing
    return store.apply_manual_override(
        gender=override.gender,
        slug=override.slug,
        player_id=override.player_id,
        visible_name=visible_name,
        sackmann_player_name=master.player_name,
        sackmann_ioc=master.ioc,
        first_resolved_date=batch_date,
        reason=override.reason,
    )


def _unresolved_observation(
    *,
    gender: Gender,
    slug: str | None,
    visible_name: str,
    reason: str,
    candidates: tuple[ActiveCandidate, ...],
    batch_date: date,
    tour_level: str,
    tournament: str,
) -> UnresolvedObservation:
    """Construye una fila de revisión sin exigir que el texto sea parseable."""

    try:
        name_key = parse_visible_name(visible_name)
    except NameParsingError:
        normalized_last_name = None
        first_initial = None
    else:
        normalized_last_name = name_key.normalized_last_name
        first_initial = name_key.first_initial
    return UnresolvedObservation(
        gender=gender,
        slug=slug,
        visible_name=visible_name,
        normalized_last_name=normalized_last_name,
        first_initial=first_initial,
        reason=reason,
        candidate_player_ids=tuple(
            candidate.player_id for candidate in candidates
        ),
        observed_date=batch_date,
        tour_level=tour_level,
        tournament=tournament,
    )


def _resolve_player(
    *,
    store: PlayerMappingStore,
    gender: Gender,
    slug: str | None,
    visible_name: str,
    batch_date: date,
    tour_level: str,
    tournament: str,
    candidate_index: CandidateIndex,
    override: OverrideRecord | None,
    master_lookup: Mapping[tuple[Gender, int], _MasterPlayer],
) -> _Resolution:
    """Resuelve una identidad en orden override, caché y coincidencia exacta."""

    existing = None if slug is None else store.get(gender, slug)
    if override is not None:
        record = _apply_override(
            store=store,
            existing=existing,
            override=override,
            visible_name=visible_name,
            batch_date=batch_date,
            master_lookup=master_lookup,
        )
        return _Resolution(record=record, observation=None)
    if existing is not None:
        _master_for_mapping(existing, master_lookup)
        return _Resolution(record=existing, observation=None)

    if slug is None:
        observation = _unresolved_observation(
            gender=gender,
            slug=None,
            visible_name=visible_name,
            reason="missing_player_slug",
            candidates=(),
            batch_date=batch_date,
            tour_level=tour_level,
            tournament=tournament,
        )
        return _Resolution(record=None, observation=observation)

    try:
        name_key = parse_visible_name(visible_name)
    except NameParsingError:
        observation = _unresolved_observation(
            gender=gender,
            slug=slug,
            visible_name=visible_name,
            reason="invalid_visible_name",
            candidates=(),
            batch_date=batch_date,
            tour_level=tour_level,
            tournament=tournament,
        )
        return _Resolution(record=None, observation=observation)

    candidates = candidate_index.get(name_key, ())
    if len(candidates) == 1:
        candidate = candidates[0]
        record = store.put_automatic(
            gender=gender,
            slug=slug,
            player_id=candidate.player_id,
            visible_name=visible_name,
            sackmann_player_name=candidate.player_name,
            sackmann_ioc=candidate.ioc,
            first_resolved_date=batch_date,
        )
        _master_for_mapping(record, master_lookup)
        return _Resolution(record=record, observation=None)

    reason = (
        "no_active_candidate"
        if not candidates
        else "ambiguous_active_candidates"
    )
    observation = _unresolved_observation(
        gender=gender,
        slug=slug,
        visible_name=visible_name,
        reason=reason,
        candidates=candidates,
        batch_date=batch_date,
        tour_level=tour_level,
        tournament=tournament,
    )
    return _Resolution(record=None, observation=observation)


def _empty_mapped_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Añade las cinco columnas de mapeo con dtypes pandas estables."""

    result = matches.copy()
    result["player_1_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["player_2_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["player_1_mapping_method"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["player_2_mapping_method"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["mapping_status"] = pd.Series(
        MAPPING_STATUS_UNMAPPED,
        index=result.index,
        dtype="string",
    )
    return result


def resolve_scraped_matches(
    matches: pd.DataFrame,
    *,
    as_of_date: date | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    database_path: Path = PLAYER_MAPPING_DATABASE_PATH,
    overrides_path: Path = PLAYER_OVERRIDES_PATH,
    unresolved_path: Path = UNRESOLVED_PLAYERS_PATH,
    players: pd.DataFrame | None = None,
    candidate_indexes: Mapping[str, CandidateIndex] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    """Mapea una cartelera diaria conservando todos sus partidos.

    Args:
        matches: DataFrame diario producido por el scraper de la fase 4.
        as_of_date: Corte explícito opcional; debe coincidir con ``match_date``.
        raw_dir: Raíz de los CSV Sackmann usados para maestro y actividad.
        database_path: SQLite persistente ``(gender, slug) -> player_id``.
        overrides_path: CSV manual con el esquema aprobado de cuatro columnas.
        unresolved_path: Cola CSV de identidades aún pendientes.
        players: Maestro inyectable para tests; por defecto se carga de disco.
        candidate_indexes: Índices inyectables por género para tests.
        clock: Reloj consciente de zona horaria que se transmite al almacén.

    Returns:
        Copia del DataFrame con IDs, método por jugador y ``mapping_status``.

    Raises:
        PlayerMappingError: Ante esquemas, fuentes o decisiones no auditables.
    """

    _require_scraper_schema(matches)
    batch_date = _resolve_batch_date(matches, as_of_date)
    _validate_daily_values(matches)
    result = _empty_mapped_frame(matches)
    if matches.empty:
        return result
    if batch_date is None:
        raise PlayerMappingValidationError(
            "Una cartelera no vacía debe tener una fecha resoluble."
        )

    raw_root = Path(raw_dir)
    player_frame = (
        _load_master_players(raw_root)
        if players is None
        else players.copy()
    )
    master_lookup = _build_master_lookup(player_frame)
    overrides = load_overrides(overrides_path)
    _validate_override_targets(overrides, master_lookup)

    genders: set[Gender] = {
        gender for gender in matches["gender"].astype(str)
    }  # type: ignore[misc]
    indexes = (
        _load_candidate_indexes(
            genders=genders,
            batch_date=batch_date,
            raw_dir=raw_root,
        )
        if candidate_indexes is None
        else _validate_candidate_indexes(candidate_indexes, genders)
    )

    unresolved: list[UnresolvedObservation] = []
    resolved_keys: set[ReviewKey] = set()
    with PlayerMappingStore(database_path, clock=clock) as store:
        for row_index, row in matches.iterrows():
            gender: Gender = str(row["gender"])  # type: ignore[assignment]
            tour_level = _optional_text(row["tour_level"])
            tournament = _optional_text(row["tournament"])
            if tour_level is None or tournament is None:
                raise PlayerMappingSchemaError(
                    "tour_level y tournament deben estar poblados."
                )

            for side in (1, 2):
                visible_name = _optional_text(row[f"player_{side}_name"])
                slug = _optional_text(row[f"player_{side}_slug"])
                if visible_name is None:
                    raise PlayerMappingSchemaError(
                        f"player_{side}_name debe estar poblado."
                    )
                key = None if slug is None else (gender, slug)
                resolution = _resolve_player(
                    store=store,
                    gender=gender,
                    slug=slug,
                    visible_name=visible_name,
                    batch_date=batch_date,
                    tour_level=tour_level,
                    tournament=tournament,
                    candidate_index=indexes[gender],
                    override=None if key is None else overrides.get(key),
                    master_lookup=master_lookup,
                )
                if resolution.record is None:
                    if resolution.observation is None:
                        raise PlayerMappingSchemaError(
                            "Un no-resuelto carece de observación auditable."
                        )
                    unresolved.append(resolution.observation)
                    continue

                record = resolution.record
                result.at[row_index, f"player_{side}_id"] = record.player_id
                result.at[
                    row_index,
                    f"player_{side}_mapping_method",
                ] = record.resolution_method
                resolved_keys.add((record.gender, record.slug))

    result["player_1_id"] = result["player_1_id"].astype("Int64")
    result["player_2_id"] = result["player_2_id"].astype("Int64")
    result["player_1_mapping_method"] = result[
        "player_1_mapping_method"
    ].astype("string")
    result["player_2_mapping_method"] = result[
        "player_2_mapping_method"
    ].astype("string")
    both_mapped = (
        result["player_1_id"].notna() & result["player_2_id"].notna()
    )
    result["mapping_status"] = pd.Series(
        MAPPING_STATUS_UNMAPPED,
        index=result.index,
        dtype="string",
    )
    result.loc[both_mapped, "mapping_status"] = MAPPING_STATUS_MAPPED

    update_unresolved_queue(
        unresolved_path,
        observations=unresolved,
        resolved_keys=resolved_keys,
    )
    return result


def summarize_mapping_coverage(mapped_matches: pd.DataFrame) -> pd.DataFrame:
    """Resume cobertura por género y nivel para slots y partidos completos."""

    if not isinstance(mapped_matches, pd.DataFrame):
        raise PlayerMappingValidationError(
            "mapped_matches debe ser un DataFrame."
        )
    required = {
        "gender",
        "tour_level",
        "player_1_id",
        "player_2_id",
        "player_1_mapping_method",
        "player_2_mapping_method",
        "mapping_status",
    }
    missing = sorted(required.difference(mapped_matches.columns))
    if missing:
        raise PlayerMappingSchemaError(
            f"No se puede resumir la cobertura; faltan columnas: {missing}."
        )
    output_columns = (
        "gender",
        "tour_level",
        "matches",
        "player_slots",
        "automatic_player_slots",
        "automatic_player_slot_pct",
        "mapped_matches",
        "mapped_match_pct",
        "automatic_matches",
        "automatic_match_pct",
    )
    if mapped_matches.empty:
        return pd.DataFrame(columns=output_columns)

    working = mapped_matches.copy()
    working["_automatic_slots"] = (
        working["player_1_mapping_method"]
        .eq("automatic_name")
        .fillna(False)
        .astype(int)
        + working["player_2_mapping_method"]
        .eq("automatic_name")
        .fillna(False)
        .astype(int)
    )
    working["_automatic_match"] = (
        working["player_1_mapping_method"].eq("automatic_name")
        & working["player_2_mapping_method"].eq("automatic_name")
    ).fillna(False).astype(int)
    working["_mapped_match"] = working["mapping_status"].eq(
        MAPPING_STATUS_MAPPED
    ).fillna(False).astype(int)

    summary = (
        working.groupby(
            ["gender", "tour_level"],
            dropna=False,
            observed=True,
        )
        .agg(
            matches=("mapping_status", "size"),
            automatic_player_slots=("_automatic_slots", "sum"),
            mapped_matches=("_mapped_match", "sum"),
            automatic_matches=("_automatic_match", "sum"),
        )
        .reset_index()
        .sort_values(["gender", "tour_level"], kind="stable")
        .reset_index(drop=True)
    )
    summary["player_slots"] = summary["matches"] * 2
    summary["automatic_player_slot_pct"] = (
        100.0
        * summary["automatic_player_slots"]
        / summary["player_slots"]
    )
    summary["mapped_match_pct"] = (
        100.0 * summary["mapped_matches"] / summary["matches"]
    )
    summary["automatic_match_pct"] = (
        100.0 * summary["automatic_matches"] / summary["matches"]
    )
    return summary.loc[:, output_columns]


def load_unresolved_players(
    path: Path = UNRESOLVED_PLAYERS_PATH,
) -> pd.DataFrame:
    """Carga la cola de revisión con tipos estables o devuelve una tabla vacía."""

    csv_path = Path(path)
    if not csv_path.exists():
        text_columns = set(UNRESOLVED_COLUMNS).difference(
            {
                "candidate_count",
                "occurrences",
                "first_seen_date",
                "last_seen_date",
            }
        )
        columns: dict[str, pd.Series] = {
            column: pd.Series(dtype="string")
            for column in text_columns
        }
        columns["candidate_count"] = pd.Series(dtype="Int64")
        columns["occurrences"] = pd.Series(dtype="Int64")
        columns["first_seen_date"] = pd.Series(dtype="datetime64[ns]")
        columns["last_seen_date"] = pd.Series(dtype="datetime64[ns]")
        return pd.DataFrame(columns).loc[:, UNRESOLVED_COLUMNS]
    if not csv_path.is_file():
        raise PlayerMappingSourceError(
            f"La ruta de no resueltos no es un archivo: {csv_path}."
        )
    try:
        frame = pd.read_csv(
            csv_path,
            dtype="string",
            keep_default_na=False,
            na_values=[""],
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise PlayerMappingSourceError(
            f"No se pudo cargar la cola de no resueltos: {csv_path}."
        ) from exc
    if tuple(frame.columns) != UNRESOLVED_COLUMNS:
        raise PlayerMappingSchemaError(
            "La cola de no resueltos no tiene la cabecera aprobada."
        )
    for column in ("candidate_count", "occurrences"):
        try:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            ).astype("Int64")
        except (TypeError, ValueError) as exc:
            raise PlayerMappingSchemaError(
                f"La cola contiene {column} inválido."
            ) from exc
        minimum = 1 if column == "occurrences" else 0
        if frame[column].isna().any() or frame[column].lt(minimum).any():
            raise PlayerMappingSchemaError(
                f"La cola contiene {column} fuera de rango."
            )
    for column in ("first_seen_date", "last_seen_date"):
        try:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise PlayerMappingSchemaError(
                f"La cola contiene {column} inválido."
            ) from exc
        if frame[column].isna().any():
            raise PlayerMappingSchemaError(
                f"La cola contiene {column} nulo."
            )
    return frame

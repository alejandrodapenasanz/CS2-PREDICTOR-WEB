"""API temporal estricta para consultar ratings Elo persistidos.

``get_elo`` y ``get_elos`` solo consultan ejecuciones completas y estados con
``state_date < as_of_date``. Las superficies públicas admitidas son los nombres
de Sackmann ``Hard``, ``Clay``, ``Grass`` y ``Carpet`` sin distinguir
capitalización; la respuesta siempre conserva el nombre canónico.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from typing import Final, Iterable, Mapping, cast

from src.config import ELO_DATABASE_PATH

from .store import (
    DEFAULT_INITIAL_RATING,
    DEFAULT_SURFACE_WEIGHT,
    EloStore,
    EloStoreError,
    EloValidationError,
    RatingHistoryRow,
    RunRecord,
)
from .types import EloSnapshot, Gender, Surface


_SURFACE_FIELDS: Final[
    dict[str, tuple[Surface, str, str]]
] = {
    "hard": ("Hard", "hard_elo", "hard_matches"),
    "clay": ("Clay", "clay_elo", "clay_matches"),
    "grass": ("Grass", "grass_elo", "grass_matches"),
    "carpet": ("Carpet", "carpet_elo", "carpet_matches"),
}


@dataclass(frozen=True)
class EloQuery:
    """Describe una consulta Elo para la API vectorizada."""

    gender: str
    player_id: int
    surface: str | None
    as_of_date: date


@dataclass(frozen=True)
class _ValidatedQuery:
    """Contiene una consulta validada y su posición de salida."""

    index: int
    gender: Gender
    player_id: int
    surface: Surface | None
    as_of_date: date


def _validate_gender(gender: object) -> Gender:
    """Valida el universo de género sin inferirlo desde el jugador."""

    if gender not in {"M", "F"}:
        raise EloValidationError("gender debe ser exactamente 'M' o 'F'.")
    return cast(Gender, gender)


def _validate_player_id(player_id: object) -> int:
    """Valida un ID entero y rechaza booleanos."""

    if isinstance(player_id, bool) or not isinstance(player_id, int):
        raise EloValidationError("player_id debe ser un entero.")
    return player_id


def _validate_as_of_date(as_of_date: object) -> date:
    """Acepta una fecha pura y rechaza datetime y texto."""

    if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
        raise EloValidationError(
            "as_of_date debe ser datetime.date estricto."
        )
    return as_of_date


def _canonical_surface(surface: object) -> Surface | None:
    """Normaliza solo capitalización y devuelve el nombre canónico aprobado."""

    if surface is None:
        return None
    if not isinstance(surface, str):
        raise EloValidationError(
            "surface debe ser Hard, Clay, Grass, Carpet o None."
        )
    definition = _SURFACE_FIELDS.get(surface.casefold())
    if definition is None:
        raise EloValidationError(
            "surface debe ser Hard, Clay, Grass, Carpet o None."
        )
    return definition[0]


def _finite_parameter(
    parameters: Mapping[str, object],
    name: str,
    default: float,
) -> float:
    """Lee un parámetro numérico finito o aplica el valor aprobado."""

    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EloStoreError(
            f"El parámetro {name!r} del run no es numérico."
        )
    result = float(value)
    if not math.isfinite(result):
        raise EloStoreError(
            f"El parámetro {name!r} del run no es finito."
        )
    return result


def _run_query_parameters(run: RunRecord) -> tuple[float, float]:
    """Obtiene rating inicial y peso de superficie desde el run."""

    initial_rating = _finite_parameter(
        run.parameters,
        "initial_rating",
        DEFAULT_INITIAL_RATING,
    )
    surface_weight = _finite_parameter(
        run.parameters,
        "surface_weight",
        DEFAULT_SURFACE_WEIGHT,
    )
    if not 0.0 <= surface_weight <= 1.0:
        raise EloStoreError(
            "surface_weight persistido debe estar entre 0 y 1."
        )
    return initial_rating, surface_weight


def _validate_query(query: EloQuery, index: int) -> _ValidatedQuery:
    """Valida una consulta vectorial conservando su posición."""

    if not isinstance(query, EloQuery):
        raise EloValidationError(
            "queries debe contener exclusivamente EloQuery."
        )
    return _ValidatedQuery(
        index=index,
        gender=_validate_gender(query.gender),
        player_id=_validate_player_id(query.player_id),
        surface=_canonical_surface(query.surface),
        as_of_date=_validate_as_of_date(query.as_of_date),
    )


def _build_snapshot(
    *,
    query: _ValidatedQuery,
    state: RatingHistoryRow | None,
    run: RunRecord,
) -> EloSnapshot:
    """Construye una respuesta y aplica cold start y mezcla documentada."""

    initial_rating, surface_weight = _run_query_parameters(run)
    if state is None:
        state_date = None
        general_elo = initial_rating
        general_matches = 0
        surface_elo_raw = (
            None if query.surface is None else initial_rating
        )
        surface_matches = 0
    else:
        if state.gender != query.gender:
            raise EloStoreError(
                "SQLite devolvió un estado perteneciente a otro género."
            )
        if state.player_id != query.player_id:
            raise EloStoreError(
                "SQLite devolvió un estado perteneciente a otro jugador."
            )
        if state.state_date >= query.as_of_date:
            raise EloStoreError(
                "Violación temporal: state_date no es anterior al corte."
            )
        state_date = state.state_date
        general_elo = state.general_elo
        general_matches = state.general_matches
        if query.surface is None:
            surface_elo_raw = None
            surface_matches = 0
        else:
            _, rating_field, count_field = _SURFACE_FIELDS[
                query.surface.casefold()
            ]
            surface_elo_raw = float(getattr(state, rating_field))
            surface_matches = int(getattr(state, count_field))

    if query.surface is None:
        combined_elo = general_elo
    else:
        if surface_elo_raw is None:
            raise EloStoreError(
                "Falta el rating crudo para una superficie solicitada."
            )
        combined_elo = (
            (1.0 - surface_weight) * general_elo
            + surface_weight * surface_elo_raw
        )
    return EloSnapshot(
        gender=query.gender,
        player_id=query.player_id,
        as_of_date=query.as_of_date,
        state_date=state_date,
        general_elo=float(general_elo),
        surface=query.surface,
        surface_elo_raw=(
            None
            if surface_elo_raw is None
            else float(surface_elo_raw)
        ),
        combined_elo=float(combined_elo),
        general_matches=int(general_matches),
        surface_matches=int(surface_matches),
        is_cold_start=general_matches == 0,
        run_id=run.run_id,
        source_commit=run.source_commit,
    )


def get_elos(
    *,
    queries: Iterable[EloQuery],
    db_path: Path = ELO_DATABASE_PATH,
    run_id: str | None = None,
) -> tuple[EloSnapshot, ...]:
    """Resuelve un lote de consultas agrupando accesos por género y run.

    Args:
        queries: Consultas tipadas que se devolverán en el mismo orden.
        db_path: Archivo SQLite inicializado.
        run_id: Run completo explícito. Si se omite, se usa el activo de cada
            género incluido en el lote.

    Returns:
        Tupla de snapshots Elo en el mismo orden que ``queries``.
    """

    validated_queries = tuple(
        _validate_query(query, index)
        for index, query in enumerate(queries)
    )
    if not validated_queries:
        return ()

    store = EloStore(Path(db_path))
    resolved_runs: dict[tuple[str, str | None], RunRecord] = {}
    grouped_queries: dict[
        tuple[str, str],
        list[_ValidatedQuery],
    ] = {}
    runs_by_group: dict[tuple[str, str], RunRecord] = {}

    for query in validated_queries:
        resolution_key = (query.gender, run_id)
        if resolution_key not in resolved_runs:
            resolved_runs[resolution_key] = store.resolve_complete_run(
                gender=query.gender,
                run_id=run_id,
            )
        run = resolved_runs[resolution_key]
        group_key = (query.gender, run.run_id)
        grouped_queries.setdefault(group_key, []).append(query)
        runs_by_group[group_key] = run

    results: list[EloSnapshot | None] = [None] * len(validated_queries)
    for group_key, group in grouped_queries.items():
        gender, resolved_run_id = group_key
        states = store.fetch_ratings_before(
            run_id=resolved_run_id,
            gender=gender,
            queries=tuple(
                (query.player_id, query.as_of_date)
                for query in group
            ),
        )
        if len(states) != len(group):
            raise EloStoreError(
                "El almacén no devolvió un estado por consulta."
            )
        run = runs_by_group[group_key]
        for query, state in zip(group, states, strict=True):
            results[query.index] = _build_snapshot(
                query=query,
                state=state,
                run=run,
            )

    if any(result is None for result in results):
        raise EloStoreError("No se resolvieron todas las consultas Elo.")
    return cast(tuple[EloSnapshot, ...], tuple(results))


def get_elo(
    *,
    gender: str,
    player_id: int,
    surface: str | None,
    as_of_date: date,
    db_path: Path = ELO_DATABASE_PATH,
    run_id: str | None = None,
) -> EloSnapshot:
    """Devuelve el último Elo estrictamente anterior a ``as_of_date``.

    Un jugador sin estado anterior recibe el rating inicial del run, por
    defecto 1500, con ``is_cold_start=True``. Si ``surface`` es ``None``,
    ``surface_elo_raw`` es nulo y el combinado coincide con el Elo general;
    para una superficie, el combinado usa ``surface_weight`` del run, por
    defecto 0.5.
    """

    return get_elos(
        queries=(
            EloQuery(
                gender=gender,
                player_id=player_id,
                surface=surface,
                as_of_date=as_of_date,
            ),
        ),
        db_path=db_path,
        run_id=run_id,
    )[0]

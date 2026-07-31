"""Índice causal y compacto para los rankings históricos de Sackmann.

El índice conserva únicamente arrays NumPy ordenados y un mapa de segmentos
por jugador. Las consultas aplican siempre el corte estricto
``ranking_date < as_of_date``: un ranking publicado en la propia fecha del
partido no es visible para ese partido.

La auditoría local del 30 de julio de 2026 encontró 3.420.595 filas ATP y
2.154.318 WTA. Hay 181 y 55 copias exactas adicionales, respectivamente, que
se colapsan. También hay 599 claves ATP y 226 WTA ``(player, ranking_date)``
con valores incompatibles, siempre dentro de un mismo CSV fuente. Esas claves
se ponen íntegramente en cuarentena: nunca se elige una fila por orden ni se
promedian valores. La API retrocede al snapshot limpio anterior y expone la
cantidad de fechas conflictivas saltadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Final, Iterable, Literal, Mapping, TypeAlias

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR


Gender: TypeAlias = Literal["M", "F"]
DateInput: TypeAlias = date | str | pd.Timestamp | np.datetime64

ATP_RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "ranking_date",
    "rank",
    "player",
    "points",
)
WTA_RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "ranking_date",
    "rank",
    "player",
    "points",
    "tours",
)
RANKING_COLUMNS_BY_GENDER: Final[dict[Gender, tuple[str, ...]]] = {
    "M": ATP_RANKING_COLUMNS,
    "F": WTA_RANKING_COLUMNS,
}
RANKING_DIRECTORY_BY_GENDER: Final[dict[Gender, str]] = {
    "M": "atp",
    "F": "wta",
}
RANKING_GLOB_BY_GENDER: Final[dict[Gender, str]] = {
    "M": "atp_rankings_*.csv",
    "F": "wta_rankings_*.csv",
}
READ_CHUNK_ROWS: Final[int] = 250_000
MISSING_INTEGER: Final[np.int64] = np.int64(np.iinfo(np.int64).min)


class RankingDataError(RuntimeError):
    """Indica que los rankings locales no se pueden interpretar con seguridad."""


class MissingRankingDataError(RankingDataError):
    """Indica que no existe ningún CSV de rankings para el género solicitado."""


class RankingSchemaError(RankingDataError):
    """Indica que una cabecera o un valor no cumple el esquema inspeccionado."""


class RankingValidationError(ValueError):
    """Indica que un argumento público de consulta no es válido."""


@dataclass(frozen=True, slots=True)
class RankingSnapshot:
    """Representa el último ranking estrictamente anterior a una fecha.

    ``ranking_date``, ``rank`` y ``points`` son nulos cuando no existe una
    observación causal. ``points`` también puede ser nulo en un snapshot
    existente porque Sackmann no lo publica en parte del histórico antiguo.
    """

    gender: Gender
    player_id: int
    as_of_date: date
    ranking_date: date | None
    rank: int | None
    points: int | None
    is_missing: bool
    ranking_age_days: int | None
    conflict_dates_skipped: int


@dataclass(frozen=True, slots=True)
class RankingConflictObservation:
    """Describe una fila fuente perteneciente a una clave en cuarentena."""

    gender: Gender
    player_id: int
    ranking_date: date
    source_file: Path
    source_row: int
    rank: int
    points: int | None
    tours: int | None


@dataclass(frozen=True, slots=True)
class _PlayerSlice:
    """Delimita el segmento de un jugador en los arrays globales ordenados."""

    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _LoadedColumns:
    """Agrupa arrays validados antes de resolver duplicados y conflictos."""

    ranking_days: np.ndarray
    ranks: np.ndarray
    player_ids: np.ndarray
    points: np.ndarray
    tours: np.ndarray
    source_file_ids: np.ndarray
    source_rows: np.ndarray
    source_files: tuple[Path, ...]


def _normalise_gender(gender: str) -> Gender:
    """Normaliza y valida un selector de género."""

    if not isinstance(gender, str):
        raise RankingValidationError("gender debe ser 'M' o 'F'.")
    normalised = gender.strip().upper()
    if normalised not in {"M", "F"}:
        raise RankingValidationError("gender debe ser 'M' o 'F'.")
    return normalised  # type: ignore[return-value]


def _normalise_player_id(player_id: int) -> int:
    """Valida un identificador entero positivo sin aceptar booleanos."""

    if isinstance(player_id, bool) or not isinstance(
        player_id, (int, np.integer)
    ):
        raise RankingValidationError("player_id debe ser un entero positivo.")
    normalised = int(player_id)
    if normalised <= 0:
        raise RankingValidationError("player_id debe ser un entero positivo.")
    return normalised


def _normalise_as_of_date(value: DateInput) -> date:
    """Convierte una fecha diaria explícita y rechaza instantes ambiguos."""

    if isinstance(value, pd.Timestamp):
        if pd.isna(value) or value.time() != datetime.min.time():
            raise RankingValidationError(
                "as_of_date debe ser una fecha válida a medianoche."
            )
        return value.date()
    if isinstance(value, datetime):
        raise RankingValidationError(
            "as_of_date debe ser una fecha diaria, no un datetime."
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            if len(value) != 10:
                raise ValueError
            return date.fromisoformat(value)
        except ValueError as exc:
            raise RankingValidationError(
                "as_of_date de texto debe usar YYYY-MM-DD."
            ) from exc
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise RankingValidationError("as_of_date no puede ser NaT.")
        day_value = value.astype("datetime64[D]")
        if value != day_value:
            raise RankingValidationError(
                "as_of_date debe tener precisión diaria."
            )
        return pd.Timestamp(day_value).date()
    raise RankingValidationError(
        "as_of_date debe ser date, YYYY-MM-DD, Timestamp o datetime64 diario."
    )


def _date_to_day(value: date) -> int:
    """Codifica una fecha como días desde el epoch para búsquedas binarias."""

    return int(np.datetime64(value, "D").astype(np.int64))


def _day_to_date(value: int | np.integer) -> date:
    """Convierte la representación compacta diaria a ``datetime.date``."""

    return pd.Timestamp(np.datetime64(int(value), "D")).date()


def _discover_ranking_files(raw_dir: Path, gender: Gender) -> tuple[Path, ...]:
    """Descubre determinísticamente los CSV inspeccionados de un género."""

    directory = raw_dir.resolve() / RANKING_DIRECTORY_BY_GENDER[gender]
    if not directory.is_dir():
        raise MissingRankingDataError(
            f"No existe el directorio de rankings esperado: {directory}."
        )
    paths = tuple(sorted(directory.glob(RANKING_GLOB_BY_GENDER[gender])))
    if not paths:
        raise MissingRankingDataError(
            f"No se encontraron rankings {RANKING_GLOB_BY_GENDER[gender]!r} "
            f"en {directory}."
        )
    return paths


def _validate_header(
    path: Path,
    expected_columns: tuple[str, ...],
) -> None:
    """Exige que la cabecera coincida exactamente con la fuente inspeccionada."""

    try:
        observed = tuple(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise RankingDataError(
            f"No se pudo leer la cabecera de rankings {path}."
        ) from exc
    if observed != expected_columns:
        raise RankingSchemaError(
            f"Esquema inesperado en {path}. Esperado={expected_columns!r}; "
            f"observado={observed!r}."
        )


def _invalid_value_example(
    values: pd.Series,
    invalid_mask: pd.Series,
    path: Path,
    column: str,
) -> RankingSchemaError:
    """Construye un error localizado para el primer valor inválido."""

    index = invalid_mask[invalid_mask].index[0]
    csv_row = int(index) + 2
    value = values.loc[index]
    return RankingSchemaError(
        f"{path}:{csv_row}: valor inválido en {column!r}: {value!r}."
    )


def _parse_required_positive_integer(
    values: pd.Series,
    *,
    path: Path,
    column: str,
) -> np.ndarray:
    """Parsea una columna entera positiva obligatoria sin coerciones silenciosas."""

    stripped = values.astype("string").str.strip()
    valid_syntax = stripped.str.fullmatch(r"\d+", na=False)
    if not bool(valid_syntax.all()):
        raise _invalid_value_example(
            stripped, ~valid_syntax, path, column
        )
    try:
        parsed = stripped.astype("int64").to_numpy(dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RankingSchemaError(
            f"{path}: la columna {column!r} excede el rango entero."
        ) from exc
    invalid_positive = pd.Series(parsed <= 0, index=values.index)
    if bool(invalid_positive.any()):
        raise _invalid_value_example(
            stripped, invalid_positive, path, column
        )
    return parsed


def _parse_optional_nonnegative_integer(
    values: pd.Series,
    *,
    path: Path,
    column: str,
) -> np.ndarray:
    """Parsea un entero no negativo opcional usando un sentinel interno."""

    stripped = values.astype("string").str.strip()
    present = stripped.notna() & stripped.ne("")
    valid_syntax = ~present | stripped.str.fullmatch(r"\d+", na=False)
    if not bool(valid_syntax.all()):
        raise _invalid_value_example(
            stripped, ~valid_syntax, path, column
        )
    parsed = np.full(len(stripped), MISSING_INTEGER, dtype=np.int64)
    if bool(present.any()):
        try:
            parsed[present.to_numpy()] = (
                stripped.loc[present].astype("int64").to_numpy(dtype=np.int64)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RankingSchemaError(
                f"{path}: la columna {column!r} excede el rango entero."
            ) from exc
    return parsed


def _parse_ranking_dates(values: pd.Series, *, path: Path) -> np.ndarray:
    """Parsea fechas obligatorias en el formato Sackmann ``YYYYMMDD``."""

    stripped = values.astype("string").str.strip()
    valid_syntax = stripped.str.fullmatch(r"\d{8}", na=False)
    if not bool(valid_syntax.all()):
        raise _invalid_value_example(
            stripped, ~valid_syntax, path, "ranking_date"
        )
    parsed = pd.to_datetime(
        stripped,
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_calendar = parsed.isna()
    if bool(invalid_calendar.any()):
        raise _invalid_value_example(
            stripped, invalid_calendar, path, "ranking_date"
        )
    return parsed.to_numpy(dtype="datetime64[D]").astype(np.int32)


def _read_ranking_files(
    paths: tuple[Path, ...],
    gender: Gender,
) -> _LoadedColumns:
    """Lee y valida los CSV por bloques para limitar el pico de memoria."""

    expected_columns = RANKING_COLUMNS_BY_GENDER[gender]
    ranking_days_parts: list[np.ndarray] = []
    ranks_parts: list[np.ndarray] = []
    player_ids_parts: list[np.ndarray] = []
    points_parts: list[np.ndarray] = []
    tours_parts: list[np.ndarray] = []
    file_id_parts: list[np.ndarray] = []
    source_row_parts: list[np.ndarray] = []

    for file_id, path in enumerate(paths):
        _validate_header(path, expected_columns)
        try:
            chunks = pd.read_csv(
                path,
                dtype="string",
                keep_default_na=False,
                na_values=[""],
                chunksize=READ_CHUNK_ROWS,
                low_memory=False,
            )
            with chunks:
                for chunk in chunks:
                    ranking_days_parts.append(
                        _parse_ranking_dates(
                            chunk["ranking_date"],
                            path=path,
                        )
                    )
                    ranks_parts.append(
                        _parse_required_positive_integer(
                            chunk["rank"],
                            path=path,
                            column="rank",
                        )
                    )
                    player_ids_parts.append(
                        _parse_required_positive_integer(
                            chunk["player"],
                            path=path,
                            column="player",
                        )
                    )
                    points_parts.append(
                        _parse_optional_nonnegative_integer(
                            chunk["points"],
                            path=path,
                            column="points",
                        )
                    )
                    if gender == "F":
                        tours_parts.append(
                            _parse_optional_nonnegative_integer(
                                chunk["tours"],
                                path=path,
                                column="tours",
                            )
                        )
                    else:
                        tours_parts.append(
                            np.full(
                                len(chunk),
                                MISSING_INTEGER,
                                dtype=np.int64,
                            )
                        )
                    file_id_parts.append(
                        np.full(len(chunk), file_id, dtype=np.int16)
                    )
                    source_row_parts.append(
                        chunk.index.to_numpy(dtype=np.int64) + 2
                    )
        except RankingDataError:
            raise
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise RankingDataError(
                f"No se pudo leer el CSV de rankings {path}."
            ) from exc

    if not ranking_days_parts:
        raise MissingRankingDataError(
            f"Los CSV de rankings de género {gender} no contienen filas."
        )
    return _LoadedColumns(
        ranking_days=np.concatenate(ranking_days_parts),
        ranks=np.concatenate(ranks_parts),
        player_ids=np.concatenate(player_ids_parts),
        points=np.concatenate(points_parts),
        tours=np.concatenate(tours_parts),
        source_file_ids=np.concatenate(file_id_parts),
        source_rows=np.concatenate(source_row_parts),
        source_files=paths,
    )


def _sort_validate_and_deduplicate(
    loaded: _LoadedColumns,
    gender: Gender,
) -> tuple[
    _LoadedColumns,
    int,
    tuple[RankingConflictObservation, ...],
]:
    """Colapsa copias exactas y pone claves incompatibles en cuarentena."""

    order = np.lexsort((loaded.ranking_days, loaded.player_ids))
    sorted_loaded = _LoadedColumns(
        ranking_days=loaded.ranking_days[order],
        ranks=loaded.ranks[order],
        player_ids=loaded.player_ids[order],
        points=loaded.points[order],
        tours=loaded.tours[order],
        source_file_ids=loaded.source_file_ids[order],
        source_rows=loaded.source_rows[order],
        source_files=loaded.source_files,
    )
    count = len(order)
    same_key = (
        (sorted_loaded.player_ids[1:] == sorted_loaded.player_ids[:-1])
        & (sorted_loaded.ranking_days[1:] == sorted_loaded.ranking_days[:-1])
    )
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(~same_key).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.array([count], dtype=np.int64))
    )

    keep = np.ones(count, dtype=bool)
    exact_duplicate_count = 0
    conflicts: list[RankingConflictObservation] = []
    for start_value, stop_value in zip(starts, stops, strict=True):
        start = int(start_value)
        stop = int(stop_value)
        if stop - start == 1:
            continue
        distinct_values: dict[tuple[int, int, int], int] = {}
        for position in range(start, stop):
            value_key = (
                int(sorted_loaded.ranks[position]),
                int(sorted_loaded.points[position]),
                int(sorted_loaded.tours[position]),
            )
            if value_key in distinct_values:
                exact_duplicate_count += 1
            else:
                distinct_values[value_key] = position
        if len(distinct_values) > 1:
            keep[start:stop] = False
            for position in range(start, stop):
                raw_points = int(sorted_loaded.points[position])
                raw_tours = int(sorted_loaded.tours[position])
                conflicts.append(
                    RankingConflictObservation(
                        gender=gender,
                        player_id=int(sorted_loaded.player_ids[position]),
                        ranking_date=_day_to_date(
                            sorted_loaded.ranking_days[position]
                        ),
                        source_file=sorted_loaded.source_files[
                            int(sorted_loaded.source_file_ids[position])
                        ],
                        source_row=int(
                            sorted_loaded.source_rows[position]
                        ),
                        rank=int(sorted_loaded.ranks[position]),
                        points=(
                            None
                            if raw_points == int(MISSING_INTEGER)
                            else raw_points
                        ),
                        tours=(
                            None
                            if raw_tours == int(MISSING_INTEGER)
                            else raw_tours
                        ),
                    )
                )
            continue
        keep[start + 1 : stop] = False

    deduplicated = _LoadedColumns(
        ranking_days=sorted_loaded.ranking_days[keep],
        ranks=sorted_loaded.ranks[keep],
        player_ids=sorted_loaded.player_ids[keep],
        points=sorted_loaded.points[keep],
        tours=sorted_loaded.tours[keep],
        source_file_ids=sorted_loaded.source_file_ids[keep],
        source_rows=sorted_loaded.source_rows[keep],
        source_files=sorted_loaded.source_files,
    )
    return deduplicated, exact_duplicate_count, tuple(conflicts)


def _make_read_only(array: np.ndarray) -> np.ndarray:
    """Marca un array como no modificable y lo devuelve."""

    array.setflags(write=False)
    return array


class RankingIndex:
    """Índice inmutable de snapshots causales para un único género."""

    __slots__ = (
        "_gender",
        "_ranking_days",
        "_ranks",
        "_points",
        "_player_slices",
        "_conflict_days_by_player",
        "_conflict_observations",
        "_source_files",
        "_source_row_count",
        "_exact_duplicate_count",
        "_min_ranking_date",
        "_max_ranking_date",
    )

    def __init__(
        self,
        *,
        gender: Gender,
        ranking_days: np.ndarray,
        ranks: np.ndarray,
        points: np.ndarray,
        player_slices: Mapping[int, _PlayerSlice],
        conflict_days_by_player: Mapping[int, np.ndarray],
        conflict_observations: tuple[RankingConflictObservation, ...],
        source_files: tuple[Path, ...],
        source_row_count: int,
        exact_duplicate_count: int,
        min_ranking_date: date,
        max_ranking_date: date,
    ) -> None:
        """Construye un índice desde arrays ya validados y ordenados."""

        if not (len(ranking_days) == len(ranks) == len(points)):
            raise RankingValidationError(
                "Los arrays internos de RankingIndex no están alineados."
            )
        self._gender = gender
        self._ranking_days = _make_read_only(ranking_days)
        self._ranks = _make_read_only(ranks)
        self._points = _make_read_only(points)
        self._player_slices = MappingProxyType(dict(player_slices))
        self._conflict_days_by_player = MappingProxyType(
            {
                player_id: _make_read_only(days)
                for player_id, days in conflict_days_by_player.items()
            }
        )
        self._conflict_observations = conflict_observations
        self._source_files = source_files
        self._source_row_count = source_row_count
        self._exact_duplicate_count = exact_duplicate_count
        self._min_ranking_date = min_ranking_date
        self._max_ranking_date = max_ranking_date

    @classmethod
    def from_raw(
        cls,
        gender: str,
        *,
        raw_dir: Path = RAW_DATA_DIR,
    ) -> "RankingIndex":
        """Carga todos los rankings locales de un género con validación estricta.

        Los duplicados totalmente idénticos se colapsan y quedan contabilizados
        en ``exact_duplicate_count``. Si la misma pareja jugador-fecha presenta
        distinto ranking, puntos o —en WTA— número de torneos, todas las filas
        de esa clave se ponen en cuarentena y quedan disponibles para exportar.
        """

        normalised_gender = _normalise_gender(gender)
        paths = _discover_ranking_files(
            Path(raw_dir),
            normalised_gender,
        )
        loaded = _read_ranking_files(paths, normalised_gender)
        source_row_count = len(loaded.ranking_days)
        min_ranking_date = _day_to_date(int(loaded.ranking_days.min()))
        max_ranking_date = _day_to_date(int(loaded.ranking_days.max()))
        deduplicated, exact_duplicate_count, conflict_observations = (
            _sort_validate_and_deduplicate(
                loaded,
                normalised_gender,
            )
        )

        player_ids = deduplicated.player_ids
        if len(player_ids):
            player_starts = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(
                        player_ids[1:] != player_ids[:-1]
                    ).astype(np.int64)
                    + 1,
                )
            )
            player_stops = np.concatenate(
                (
                    player_starts[1:],
                    np.array([len(player_ids)], dtype=np.int64),
                )
            )
            player_slices = {
                int(player_ids[int(start)]): _PlayerSlice(
                    start=int(start),
                    stop=int(stop),
                )
                for start, stop in zip(
                    player_starts,
                    player_stops,
                    strict=True,
                )
            }
        else:
            player_slices = {}
        raw_conflict_days: dict[int, set[int]] = {}
        for observation in conflict_observations:
            raw_conflict_days.setdefault(observation.player_id, set()).add(
                _date_to_day(observation.ranking_date)
            )
        conflict_days_by_player = {
            player_id: np.array(sorted(days), dtype=np.int32)
            for player_id, days in raw_conflict_days.items()
        }
        return cls(
            gender=normalised_gender,
            ranking_days=deduplicated.ranking_days.astype(
                np.int32, copy=False
            ),
            ranks=deduplicated.ranks,
            points=deduplicated.points,
            player_slices=player_slices,
            conflict_days_by_player=conflict_days_by_player,
            conflict_observations=conflict_observations,
            source_files=deduplicated.source_files,
            source_row_count=source_row_count,
            exact_duplicate_count=exact_duplicate_count,
            min_ranking_date=min_ranking_date,
            max_ranking_date=max_ranking_date,
        )

    @property
    def gender(self) -> Gender:
        """Devuelve el único género contenido en el índice."""

        return self._gender

    @property
    def row_count(self) -> int:
        """Devuelve el número de snapshots únicos indexados."""

        return len(self._ranking_days)

    @property
    def source_row_count(self) -> int:
        """Devuelve el número de filas leídas antes de deduplicar."""

        return self._source_row_count

    @property
    def exact_duplicate_count(self) -> int:
        """Devuelve el número de copias exactas descartadas."""

        return self._exact_duplicate_count

    @property
    def player_count(self) -> int:
        """Devuelve el número de jugadores distintos indexados."""

        return len(self._player_slices)

    @property
    def conflict_key_count(self) -> int:
        """Devuelve el número de claves jugador-fecha en cuarentena."""

        return sum(
            len(days) for days in self._conflict_days_by_player.values()
        )

    @property
    def conflict_observation_count(self) -> int:
        """Devuelve el número de filas fuente incluidas en conflictos."""

        return len(self._conflict_observations)

    @property
    def min_ranking_date(self) -> date:
        """Devuelve la primera fecha de ranking disponible."""

        return self._min_ranking_date

    @property
    def max_ranking_date(self) -> date:
        """Devuelve la última fecha de ranking disponible."""

        return self._max_ranking_date

    @property
    def source_files(self) -> tuple[Path, ...]:
        """Devuelve los archivos fuente inspeccionados en orden estable."""

        return self._source_files

    def write_conflict_inventory(self, path: Path) -> Path:
        """Escribe atómicamente un CSV con todas las observaciones en cuarentena.

        La escritura solo ocurre cuando el llamador invoca este método. Primero
        se completa un archivo temporal en el mismo directorio y después se
        sustituye el destino mediante ``os.replace``.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "gender",
                        "player_id",
                        "ranking_date",
                        "source_file",
                        "source_row",
                        "rank",
                        "points",
                        "tours",
                    ),
                )
                writer.writeheader()
                for observation in self._conflict_observations:
                    writer.writerow(
                        {
                            "gender": observation.gender,
                            "player_id": observation.player_id,
                            "ranking_date": (
                                observation.ranking_date.isoformat()
                            ),
                            "source_file": str(observation.source_file),
                            "source_row": observation.source_row,
                            "rank": observation.rank,
                            "points": (
                                ""
                                if observation.points is None
                                else observation.points
                            ),
                            "tours": (
                                ""
                                if observation.tours is None
                                else observation.tours
                            ),
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RankingDataError(
                f"No se pudo escribir el inventario de conflictos "
                f"{destination}."
            ) from exc
        return destination

    def _count_conflicts_skipped(
        self,
        *,
        player_id: int,
        cutoff_day: int,
        selected_day: int | None,
    ) -> int:
        """Cuenta conflictos visibles posteriores al snapshot devuelto."""

        conflict_days = self._conflict_days_by_player.get(player_id)
        if conflict_days is None:
            return 0
        visible_stop = int(
            np.searchsorted(conflict_days, cutoff_day, side="left")
        )
        if selected_day is None:
            return visible_stop
        visible_start = int(
            np.searchsorted(conflict_days, selected_day, side="right")
        )
        return max(0, visible_stop - visible_start)

    def get(
        self,
        player_id: int,
        as_of_date: DateInput,
    ) -> RankingSnapshot:
        """Obtiene el último snapshot con ``ranking_date < as_of_date``."""

        normalised_player_id = _normalise_player_id(player_id)
        normalised_date = _normalise_as_of_date(as_of_date)
        cutoff_day = _date_to_day(normalised_date)
        player_slice = self._player_slices.get(normalised_player_id)
        if player_slice is None:
            return RankingSnapshot(
                gender=self._gender,
                player_id=normalised_player_id,
                as_of_date=normalised_date,
                ranking_date=None,
                rank=None,
                points=None,
                is_missing=True,
                ranking_age_days=None,
                conflict_dates_skipped=self._count_conflicts_skipped(
                    player_id=normalised_player_id,
                    cutoff_day=cutoff_day,
                    selected_day=None,
                ),
            )

        relative_position = int(
            np.searchsorted(
                self._ranking_days[
                    player_slice.start : player_slice.stop
                ],
                cutoff_day,
                side="left",
            )
        )
        if relative_position == 0:
            return RankingSnapshot(
                gender=self._gender,
                player_id=normalised_player_id,
                as_of_date=normalised_date,
                ranking_date=None,
                rank=None,
                points=None,
                is_missing=True,
                ranking_age_days=None,
                conflict_dates_skipped=self._count_conflicts_skipped(
                    player_id=normalised_player_id,
                    cutoff_day=cutoff_day,
                    selected_day=None,
                ),
            )

        position = player_slice.start + relative_position - 1
        selected_day = int(self._ranking_days[position])
        raw_points = int(self._points[position])
        return RankingSnapshot(
            gender=self._gender,
            player_id=normalised_player_id,
            as_of_date=normalised_date,
            ranking_date=_day_to_date(selected_day),
            rank=int(self._ranks[position]),
            points=(
                None
                if raw_points == int(MISSING_INTEGER)
                else raw_points
            ),
            is_missing=False,
            ranking_age_days=cutoff_day - selected_day,
            conflict_dates_skipped=self._count_conflicts_skipped(
                player_id=normalised_player_id,
                cutoff_day=cutoff_day,
                selected_day=selected_day,
            ),
        )

    def get_many(
        self,
        player_ids: Iterable[int],
        as_of_date: DateInput,
    ) -> tuple[RankingSnapshot, ...]:
        """Consulta varios jugadores para una fecha y conserva orden y repetidos."""

        normalised_date = _normalise_as_of_date(as_of_date)
        return tuple(
            self.get(player_id, normalised_date)
            for player_id in player_ids
        )

"""Construcción causal del índice de candidatos Sackmann activos.

La actividad se evalúa respecto a la fecha del partido que se quiere mapear:
un jugador es candidato si aparece en cualquier nivel del mismo género dentro
de ``[D - 3 años naturales, D)``. El límite superior es exclusivo para que un
partido de la propia fecha nunca pueda habilitar a un candidato.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import logging
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import RAW_DATA_DIR
from src.data_loaders import (
    FAMILY_GENDER,
    GENDER_DIRECTORY,
    MATCH_SOURCE_COLUMNS,
    PLAYER_FILENAME,
    PLAYER_SOURCE_COLUMNS,
)
from src.sackmann_download import EXPECTED_FIRST_YEAR, MATCH_FAMILY_PATTERNS

from .normalization import build_sackmann_name_key
from .types import (
    ActiveCandidate,
    CandidateIndex,
    Gender,
    NameKey,
    NameParsingError,
    PlayerMappingSchemaError,
    PlayerMappingSourceError,
    PlayerMappingValidationError,
)


LOGGER = logging.getLogger(__name__)

_PLAYER_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "name_first",
    "name_last",
    "ioc",
)
_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "tourney_date",
    "winner_id",
    "loser_id",
)


def _validate_gender(gender: str) -> Gender:
    """Valida el género sin corregir mayúsculas ni valores desconocidos."""

    if gender not in {"M", "F"}:
        raise PlayerMappingValidationError(
            "gender debe ser exactamente 'M' o 'F'."
        )
    return gender  # type: ignore[return-value]


def _validate_as_of_date(as_of_date: date) -> date:
    """Valida que el corte temporal sea una fecha civil, no un datetime."""

    if not isinstance(as_of_date, date) or isinstance(as_of_date, datetime):
        raise PlayerMappingValidationError(
            "as_of_date debe ser datetime.date, no datetime."
        )
    return as_of_date


def active_window_start(as_of_date: date) -> date:
    """Devuelve el inicio inclusivo de la ventana de tres años naturales.

    Para un 29 de febrero cuyo año de destino no sea bisiesto se utiliza el
    28 de febrero, equivalente al desplazamiento calendario de tres años.

    Args:
        as_of_date: Fecha ``D`` cuyo límite superior será exclusivo.

    Returns:
        Fecha civil correspondiente a ``D - 3 años naturales``.

    Raises:
        PlayerMappingValidationError: Si el argumento no es una fecha civil.
    """

    validated_date = _validate_as_of_date(as_of_date)
    target_year = validated_date.year - 3
    try:
        return validated_date.replace(year=target_year)
    except ValueError:
        return validated_date.replace(year=target_year, day=28)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    """Comprueba que un DataFrame contenga todas las columnas necesarias."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PlayerMappingSchemaError(
            f"{frame_name} no contiene las columnas requeridas: {missing}."
        )


def _validate_gender_column(frame: pd.DataFrame, *, frame_name: str) -> None:
    """Rechaza géneros nulos o ajenos a los dos universos del proyecto."""

    observed = set(frame["gender"].dropna().astype(str))
    if frame["gender"].isna().any() or not observed.issubset({"M", "F"}):
        raise PlayerMappingSchemaError(
            f"{frame_name}.gender solo puede contener 'M' o 'F', sin nulos."
        )


def _coerce_integer_ids(
    values: pd.Series,
    *,
    column_name: str,
    allow_missing: bool,
) -> pd.Series:
    """Convierte identificadores a Int64 y rechaza decimales o texto inválido."""

    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerMappingSchemaError(
            f"{column_name} contiene identificadores no numéricos."
        ) from exc
    non_null = numeric.dropna()
    if not non_null.empty and not (non_null % 1 == 0).all():
        raise PlayerMappingSchemaError(
            f"{column_name} contiene identificadores no enteros."
        )
    if not allow_missing and numeric.isna().any():
        raise PlayerMappingSchemaError(
            f"{column_name} contiene identificadores nulos."
        )
    return numeric.astype("Int64")


def _coerce_match_dates(values: pd.Series) -> pd.Series:
    """Convierte fechas de partidos y rechaza cualquier fecha ausente o inválida."""

    try:
        converted = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerMappingSchemaError(
            "matches.tourney_date contiene fechas inválidas."
        ) from exc
    if converted.isna().any():
        raise PlayerMappingSchemaError(
            "matches.tourney_date contiene fechas nulas."
        )
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_localize(None)
    return converted


def _optional_text(value: object) -> str | None:
    """Convierte un campo textual nullable sin fabricar contenido."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def build_active_candidate_index(
    players: pd.DataFrame,
    matches: pd.DataFrame,
    gender: str,
    as_of_date: date,
) -> CandidateIndex:
    """Agrupa candidatos activos por apellido normalizado e inicial.

    Se consideran conjuntamente todos los niveles presentes en ``matches``.
    Las colisiones se preservan como tuplas de más de un candidato; esta
    función nunca elige por semejanza, ranking, país ni recencia.

    Args:
        players: Maestro Sackmann con identidad, nombre, país y género.
        matches: Partidos unificados con ambos IDs, fecha y género.
        gender: Universo exacto ``"M"`` o ``"F"``.
        as_of_date: Fecha ``D``; solo se observan fechas estrictamente menores.

    Returns:
        Diccionario de :class:`NameKey` a tuplas ordenadas por ``player_id``.

    Raises:
        PlayerMappingValidationError: Si los argumentos básicos son inválidos.
        PlayerMappingSchemaError: Si faltan columnas o los tipos son incoherentes.
        PlayerMappingSourceError: Si un ID activo no existe en el maestro.
    """

    selected_gender = _validate_gender(gender)
    selected_date = _validate_as_of_date(as_of_date)
    if not isinstance(players, pd.DataFrame):
        raise PlayerMappingValidationError("players debe ser un DataFrame.")
    if not isinstance(matches, pd.DataFrame):
        raise PlayerMappingValidationError("matches debe ser un DataFrame.")

    _require_columns(
        players,
        {
            "gender",
            "player_id",
            "player_name",
            "name_first",
            "name_last",
            "ioc",
        },
        frame_name="players",
    )
    _require_columns(
        matches,
        {"gender", "tourney_date", "winner_id", "loser_id"},
        frame_name="matches",
    )
    _validate_gender_column(players, frame_name="players")
    _validate_gender_column(matches, frame_name="matches")

    selected_players = players.loc[
        players["gender"].eq(selected_gender)
    ].copy()
    selected_matches = matches.loc[
        matches["gender"].eq(selected_gender)
    ].copy()
    selected_players["player_id"] = _coerce_integer_ids(
        selected_players["player_id"],
        column_name="players.player_id",
        allow_missing=False,
    )
    if selected_players["player_id"].duplicated().any():
        duplicated_ids = (
            selected_players.loc[
                selected_players["player_id"].duplicated(keep=False),
                "player_id",
            ]
            .astype(str)
            .tolist()
        )
        raise PlayerMappingSchemaError(
            "players contiene player_id duplicados dentro del género "
            f"{selected_gender}: {duplicated_ids[:10]}."
        )

    selected_matches["tourney_date"] = _coerce_match_dates(
        selected_matches["tourney_date"]
    )
    window_start = pd.Timestamp(active_window_start(selected_date))
    exclusive_end = pd.Timestamp(selected_date)
    selected_matches = selected_matches.loc[
        selected_matches["tourney_date"].ge(window_start)
        & selected_matches["tourney_date"].lt(exclusive_end)
    ].copy()
    for column in ("winner_id", "loser_id"):
        selected_matches[column] = _coerce_integer_ids(
            selected_matches[column],
            column_name=f"matches.{column}",
            allow_missing=True,
        )

    appearances = pd.concat(
        [
            selected_matches[["tourney_date", "winner_id"]].rename(
                columns={"winner_id": "player_id"}
            ),
            selected_matches[["tourney_date", "loser_id"]].rename(
                columns={"loser_id": "player_id"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["player_id"])
    if appearances.empty:
        return {}

    last_matches = (
        appearances.groupby("player_id", as_index=False, sort=True)[
            "tourney_date"
        ]
        .max()
        .rename(columns={"tourney_date": "last_match_date"})
    )
    missing_master_ids = sorted(
        set(last_matches["player_id"].astype(int)).difference(
            selected_players["player_id"].astype(int)
        )
    )
    if missing_master_ids:
        raise PlayerMappingSourceError(
            "Hay IDs activos sin fila en el maestro Sackmann del género "
            f"{selected_gender}: {missing_master_ids[:10]}."
        )

    active_players = selected_players.merge(
        last_matches,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    grouped: defaultdict[NameKey, list[ActiveCandidate]] = defaultdict(list)
    skipped_ids: list[int] = []
    for row in active_players.itertuples(index=False):
        name_first = _optional_text(row.name_first)
        name_last = _optional_text(row.name_last)
        player_name = _optional_text(row.player_name)
        if name_first is None or name_last is None or player_name is None:
            skipped_ids.append(int(row.player_id))
            continue
        try:
            name_key = build_sackmann_name_key(name_first, name_last)
        except NameParsingError:
            skipped_ids.append(int(row.player_id))
            continue
        candidate = ActiveCandidate(
            gender=selected_gender,
            player_id=int(row.player_id),
            player_name=player_name,
            name_first=name_first,
            name_last=name_last,
            ioc=_optional_text(row.ioc),
            last_match_date=pd.Timestamp(row.last_match_date).date(),
            name_key=name_key,
        )
        grouped[name_key].append(candidate)

    if skipped_ids:
        LOGGER.warning(
            "Se omitieron %d IDs activos sin nombre utilizable: %s",
            len(skipped_ids),
            skipped_ids[:10],
        )

    return {
        name_key: tuple(
            sorted(candidates, key=lambda candidate: candidate.player_id)
        )
        for name_key, candidates in sorted(
            grouped.items(),
            key=lambda item: (
                item[0].normalized_last_name,
                item[0].first_initial,
            ),
        )
    }


def _validate_csv_header(
    path: Path,
    expected_columns: tuple[str, ...],
) -> None:
    """Comprueba la cabecera completa antes de una lectura selectiva."""

    try:
        observed_columns = tuple(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise PlayerMappingSourceError(
            f"No se pudo leer la cabecera de {path}."
        ) from exc
    if observed_columns != expected_columns:
        raise PlayerMappingSchemaError(
            f"Esquema inesperado en {path}. "
            f"Esperado={expected_columns!r}; observado={observed_columns!r}."
        )


def _load_candidate_players(raw_dir: Path, gender: Gender) -> pd.DataFrame:
    """Lee solo las cuatro columnas del maestro necesarias para candidatos."""

    path = raw_dir / GENDER_DIRECTORY[gender] / PLAYER_FILENAME[gender]
    if not path.is_file():
        raise PlayerMappingSourceError(
            f"No existe el maestro de jugadores esperado: {path}."
        )
    _validate_csv_header(path, PLAYER_SOURCE_COLUMNS)
    try:
        frame = pd.read_csv(
            path,
            usecols=list(_PLAYER_COLUMNS),
            dtype="string",
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
    except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise PlayerMappingSourceError(
            f"No se pudo leer el maestro de jugadores {path}."
        ) from exc
    frame["player_name"] = (
        frame["name_first"]
        .fillna("")
        .str.cat(frame["name_last"].fillna(""), sep=" ")
        .str.strip()
        .replace("", pd.NA)
        .astype("string")
    )
    frame["gender"] = pd.Series(gender, index=frame.index, dtype="string")
    return frame


def _discover_required_match_files(
    raw_dir: Path,
    gender: Gender,
    as_of_date: date,
) -> tuple[Path, ...]:
    """Descubre todas las familias y años que intersectan la ventana activa."""

    start_year = active_window_start(as_of_date).year
    selected: list[tuple[str, int, Path]] = []
    for family, pattern in MATCH_FAMILY_PATTERNS.items():
        if FAMILY_GENDER[family] != gender:
            continue
        directory = raw_dir / GENDER_DIRECTORY[gender]
        if not directory.is_dir():
            raise PlayerMappingSourceError(
                f"No existe el directorio Sackmann esperado: {directory}."
            )
        discovered_by_year: dict[int, Path] = {}
        for path in directory.glob("*.csv"):
            relative_path = path.relative_to(raw_dir).as_posix()
            match = pattern.fullmatch(relative_path)
            if match is not None:
                year = int(match.group("year"))
                if year in discovered_by_year:
                    raise PlayerMappingSourceError(
                        f"La familia {family!r} repite el año {year}."
                    )
                discovered_by_year[year] = path

        first_required_year = max(start_year, EXPECTED_FIRST_YEAR[family])
        if first_required_year > as_of_date.year:
            continue
        required_years = set(
            range(first_required_year, as_of_date.year + 1)
        )
        missing_years = sorted(required_years.difference(discovered_by_year))
        if missing_years:
            raise PlayerMappingSourceError(
                f"La familia {family!r} no contiene los años necesarios "
                f"para la ventana activa: {missing_years}."
            )
        selected.extend(
            (family, year, discovered_by_year[year])
            for year in sorted(required_years)
        )
    return tuple(
        path
        for _, _, path in sorted(
            selected,
            key=lambda item: (item[0], item[1]),
        )
    )


def _load_candidate_matches(
    paths: tuple[Path, ...],
    gender: Gender,
    as_of_date: date,
) -> pd.DataFrame:
    """Lee solo fecha e IDs de los archivos anuales que pueden aportar actividad."""

    frames: list[pd.DataFrame] = []
    window_start = pd.Timestamp(active_window_start(as_of_date))
    exclusive_end = pd.Timestamp(as_of_date)
    for path in paths:
        _validate_csv_header(path, MATCH_SOURCE_COLUMNS)
        try:
            frame = pd.read_csv(
                path,
                usecols=list(_MATCH_COLUMNS),
                dtype="string",
                keep_default_na=False,
                na_values=[""],
                low_memory=False,
            )
            frame["tourney_date"] = pd.to_datetime(
                frame["tourney_date"],
                format="%Y%m%d",
                errors="raise",
            )
        except (
            OSError,
            pd.errors.ParserError,
            UnicodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise PlayerMappingSchemaError(
                f"No se pudo leer y tipar el CSV de partidos {path}."
            ) from exc
        if frame["tourney_date"].isna().any():
            raise PlayerMappingSchemaError(
                f"{path}: tourney_date contiene valores nulos."
            )
        frame = frame.loc[
            frame["tourney_date"].ge(window_start)
            & frame["tourney_date"].lt(exclusive_end)
        ]
        frame["gender"] = pd.Series(
            gender,
            index=frame.index,
            dtype="string",
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            {
                "tourney_date": pd.Series(dtype="datetime64[ns]"),
                "winner_id": pd.Series(dtype="string"),
                "loser_id": pd.Series(dtype="string"),
                "gender": pd.Series(dtype="string"),
            }
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def load_active_candidate_index(
    gender: str,
    as_of_date: date,
    raw_dir: Path = RAW_DATA_DIR,
) -> CandidateIndex:
    """Carga de disco únicamente los datos necesarios y construye el índice.

    Args:
        gender: Universo exacto ``"M"`` o ``"F"``.
        as_of_date: Fecha ``D`` que define la ventana causal.
        raw_dir: Raíz local con los directorios Sackmann ``atp`` y ``wta``.

    Returns:
        Índice de candidatos activos que conserva todas las colisiones.

    Raises:
        PlayerMappingValidationError: Si los argumentos no son válidos.
        PlayerMappingSourceError: Si falta un archivo o año necesario.
        PlayerMappingSchemaError: Si un CSV cambió de esquema o tipo.
    """

    selected_gender = _validate_gender(gender)
    selected_date = _validate_as_of_date(as_of_date)
    if not isinstance(raw_dir, Path):
        raise PlayerMappingValidationError("raw_dir debe ser pathlib.Path.")
    raw_root = raw_dir.resolve()
    if not raw_root.is_dir():
        raise PlayerMappingSourceError(
            f"No existe el directorio de datos crudos: {raw_root}."
        )
    players = _load_candidate_players(raw_root, selected_gender)
    match_paths = _discover_required_match_files(
        raw_root,
        selected_gender,
        selected_date,
    )
    matches = _load_candidate_matches(
        match_paths,
        selected_gender,
        selected_date,
    )
    return build_active_candidate_index(
        players,
        matches,
        selected_gender,
        selected_date,
    )

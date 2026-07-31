"""Loaders tipados para los CSV históricos de partidos y jugadores."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Final, Literal, Sequence

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.config import RAW_DATA_DIR
from src.sackmann_download import EXPECTED_FIRST_YEAR, MATCH_FAMILY_PATTERNS


Gender = Literal["M", "F"]
LOGGER = logging.getLogger(__name__)

MATCH_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
)

PLAYER_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "name_first",
    "name_last",
    "hand",
    "dob",
    "ioc",
    "height",
    "wikidata_id",
)

MATCH_INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "draw_size",
    "match_num",
    "winner_id",
    "winner_ht",
    "loser_id",
    "loser_ht",
    "best_of",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
)

MATCH_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "winner_age",
    "loser_age",
)

MATCH_DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "tour_level",
    "source_family",
    "source_file",
    "source_anomaly",
)

KNOWN_DRAW_SIZE_ANOMALY: Final[dict[str, str]] = {
    "source_file": "wta_matches_1970.csv",
    "tourney_id": "1970-1093",
    "tourney_name": "Rondebosch Exho",
    "tourney_level": "E",
    "tourney_date": "19700406",
    "match_num": "1",
    "raw_value": "exho",
}
KNOWN_DRAW_SIZE_ANOMALY_FLAG: Final[str] = (
    "draw_size_exho_normalized_to_missing"
)

FAMILY_GENDER: Final[dict[str, Gender]] = {
    "atp_main": "M",
    "atp_qual_chall": "M",
    "atp_futures": "M",
    "wta_main": "F",
    "wta_qual_itf": "F",
}

GENDER_DIRECTORY: Final[dict[Gender, str]] = {
    "M": "atp",
    "F": "wta",
}

PLAYER_FILENAME: Final[dict[Gender, str]] = {
    "M": "atp_players.csv",
    "F": "wta_players.csv",
}


class DataLoaderError(RuntimeError):
    """Indica que los datos locales faltan o no cumplen el esquema validado."""


class MissingRawDataError(DataLoaderError):
    """Indica que todavía no se descargaron todos los datos necesarios."""


class SchemaMismatchError(DataLoaderError):
    """Indica que un CSV real no tiene la cabecera inspeccionada."""


@dataclass(frozen=True)
class MatchFile:
    """Describe un archivo local de partidos y su procedencia."""

    path: Path
    gender: Gender
    source_family: str
    year: int


def _normalise_gender(gender: str | None) -> tuple[Gender, ...]:
    """Valida el selector de género y devuelve una tupla determinista."""

    if gender is None:
        return ("M", "F")
    normalised = gender.strip().upper()
    if normalised not in {"M", "F"}:
        raise ValueError("gender debe ser 'M', 'F' o None.")
    return (normalised,)  # type: ignore[return-value]


def _validate_header(path: Path, expected_columns: Sequence[str]) -> None:
    """Comprueba que la cabecera de un CSV coincida exactamente con el esquema."""

    try:
        observed_columns = tuple(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataLoaderError(f"No se pudo leer la cabecera de {path}.") from exc
    if observed_columns != tuple(expected_columns):
        raise SchemaMismatchError(
            f"Esquema inesperado en {path}. "
            f"Esperado={tuple(expected_columns)!r}; observado={observed_columns!r}."
        )


def _discover_match_files(
    raw_dir: Path,
    genders: tuple[Gender, ...],
) -> tuple[MatchFile, ...]:
    """Descubre únicamente las cinco familias de singles inspeccionadas."""

    discovered: list[MatchFile] = []
    for family, pattern in MATCH_FAMILY_PATTERNS.items():
        family_gender = FAMILY_GENDER[family]
        if family_gender not in genders:
            continue
        directory = raw_dir / GENDER_DIRECTORY[family_gender]
        if not directory.is_dir():
            raise MissingRawDataError(
                f"No existe el directorio de datos crudos: {directory}."
            )
        family_files: list[MatchFile] = []
        for path in directory.glob("*.csv"):
            relative_path = path.relative_to(raw_dir).as_posix()
            match = pattern.fullmatch(relative_path)
            if match is None:
                continue
            family_files.append(
                MatchFile(
                    path=path,
                    gender=family_gender,
                    source_family=family,
                    year=int(match.group("year")),
                )
            )
        if not family_files:
            raise MissingRawDataError(
                f"No se encontraron CSV para la familia {family!r} en {directory}."
            )
        years = sorted(match_file.year for match_file in family_files)
        if years[0] != EXPECTED_FIRST_YEAR[family]:
            raise MissingRawDataError(
                f"La familia {family!r} empieza en {years[0]}, no en "
                f"{EXPECTED_FIRST_YEAR[family]}."
            )
        missing_years = sorted(
            set(range(years[0], years[-1] + 1)).difference(years)
        )
        if missing_years:
            raise MissingRawDataError(
                f"La familia {family!r} tiene archivos anuales ausentes: "
                f"{missing_years}."
            )
        discovered.extend(family_files)

    return tuple(
        sorted(
            discovered,
            key=lambda match_file: (
                match_file.gender,
                match_file.source_family,
                match_file.year,
            ),
        )
    )


def _strip_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina espacios periféricos y convierte textos vacíos en valores nulos."""

    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].str.strip().replace("", pd.NA)
    return frame


def _apply_known_match_anomalies(
    frame: pd.DataFrame,
    source_path: Path,
) -> pd.DataFrame:
    """Normaliza únicamente excepciones fuente aprobadas y deja una marca.

    Sackmann publicó ``exho`` en ``draw_size`` para un único partido de
    exhibición de 1970. La excepción se reconoce por la identidad completa de
    la fila, no solo por el literal, para que cualquier caso futuro distinto
    continúe fallando de forma explícita.
    """

    frame["source_anomaly"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    raw_anomaly_mask = frame["draw_size"].eq(
        KNOWN_DRAW_SIZE_ANOMALY["raw_value"]
    ).fillna(False)
    if not raw_anomaly_mask.any():
        return frame

    identity_mask = pd.Series(True, index=frame.index, dtype="boolean")
    for column in (
        "tourney_id",
        "tourney_name",
        "tourney_level",
        "tourney_date",
        "match_num",
    ):
        identity_mask &= frame[column].eq(
            KNOWN_DRAW_SIZE_ANOMALY[column]
        ).fillna(False)
    approved_mask = raw_anomaly_mask & identity_mask
    is_approved_file = (
        source_path.name == KNOWN_DRAW_SIZE_ANOMALY["source_file"]
    )

    if not is_approved_file or int(approved_mask.sum()) != 1:
        raise SchemaMismatchError(
            f"{source_path}: se encontró draw_size='exho' fuera de la "
            "excepción histórica aprobada."
        )
    if int(raw_anomaly_mask.sum()) != 1:
        raise SchemaMismatchError(
            f"{source_path}: se encontraron valores draw_size='exho' "
            "adicionales no aprobados."
        )

    frame.loc[approved_mask, "draw_size"] = pd.NA
    frame.loc[approved_mask, "source_anomaly"] = (
        KNOWN_DRAW_SIZE_ANOMALY_FLAG
    )
    LOGGER.warning(
        "%s: draw_size='exho' normalizado a nulo para %s; el CSV crudo "
        "permanece intacto.",
        source_path,
        KNOWN_DRAW_SIZE_ANOMALY["tourney_id"],
    )
    return frame


def _coerce_integer_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    source_path: Path,
) -> pd.DataFrame:
    """Convierte columnas enteras de forma estricta y conserva nulos con Int64."""

    for column in columns:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
            non_null = numeric.dropna()
            if not non_null.empty and not (non_null % 1 == 0).all():
                raise ValueError("se encontraron valores no enteros")
            frame[column] = numeric.astype("Int64")
        except (TypeError, ValueError) as exc:
            raise SchemaMismatchError(
                f"{source_path}: la columna {column!r} no es entera."
            ) from exc
    return frame


def _coerce_float_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    source_path: Path,
) -> pd.DataFrame:
    """Convierte columnas decimales de forma estricta y conserva valores nulos."""

    for column in columns:
        try:
            frame[column] = pd.to_numeric(
                frame[column], errors="raise"
            ).astype("Float64")
        except (TypeError, ValueError) as exc:
            raise SchemaMismatchError(
                f"{source_path}: la columna {column!r} no es numérica."
            ) from exc
    return frame


def _read_match_file(match_file: MatchFile) -> pd.DataFrame:
    """Lee, valida y tipa un único archivo anual de partidos."""

    _validate_header(match_file.path, MATCH_SOURCE_COLUMNS)
    try:
        frame = pd.read_csv(
            match_file.path,
            dtype="string",
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataLoaderError(
            f"No se pudo cargar el CSV de partidos {match_file.path}."
        ) from exc

    frame = _strip_string_columns(frame)
    frame = _apply_known_match_anomalies(frame, match_file.path)
    frame = _coerce_integer_columns(
        frame,
        MATCH_INTEGER_COLUMNS,
        match_file.path,
    )
    frame = _coerce_float_columns(
        frame,
        MATCH_FLOAT_COLUMNS,
        match_file.path,
    )
    try:
        frame["tourney_date"] = pd.to_datetime(
            frame["tourney_date"],
            format="%Y%m%d",
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(
            f"{match_file.path}: tourney_date contiene fechas inválidas."
        ) from exc

    if frame["tourney_date"].isna().any():
        raise SchemaMismatchError(
            f"{match_file.path}: tourney_date contiene valores nulos."
        )
    if frame["tourney_level"].isna().any():
        raise SchemaMismatchError(
            f"{match_file.path}: tourney_level contiene valores nulos."
        )

    frame["gender"] = pd.Series(
        match_file.gender,
        index=frame.index,
        dtype="string",
    )
    frame["tour_level"] = frame["tourney_level"].astype("string")
    frame["source_family"] = pd.Series(
        match_file.source_family,
        index=frame.index,
        dtype="string",
    )
    frame["source_file"] = pd.Series(
        match_file.path.name,
        index=frame.index,
        dtype="string",
    )
    return frame


def load_matches(
    gender: str | None = None,
    *,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Carga y unifica todos los partidos de singles disponibles.

    ``tour_level`` conserva como texto el código exacto de ``tourney_level`` de
    Sackmann. ``source_family`` permite distinguir el fichero principal de los
    ficheros qual/challenger, futures o qual/ITF sin reinterpretar esos códigos.

    Args:
        gender: ``"M"`` o ``"F"`` para un género; ``None`` para ambos.
        raw_dir: Raíz local que contiene los subdirectorios ``atp`` y ``wta``.

    Returns:
        DataFrame tipado con todas las columnas fuente y cuatro de procedencia.
    """

    genders = _normalise_gender(gender)
    match_files = _discover_match_files(raw_dir.resolve(), genders)
    frames = [_read_match_file(match_file) for match_file in match_files]
    if not frames:
        raise MissingRawDataError("No se encontró ningún partido para cargar.")
    result = pd.concat(frames, ignore_index=True, sort=False)
    if not is_datetime64_any_dtype(result["tourney_date"]):
        raise DataLoaderError("tourney_date no quedó tipada como fecha.")
    if result["gender"].isna().any() or result["tour_level"].isna().any():
        raise DataLoaderError("gender y tour_level deben estar siempre poblados.")
    return result


def _read_players_file(path: Path, gender: Gender) -> pd.DataFrame:
    """Lee y tipa el maestro de jugadores de un género."""

    _validate_header(path, PLAYER_SOURCE_COLUMNS)
    try:
        frame = pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataLoaderError(
            f"No se pudo cargar el CSV de jugadores {path}."
        ) from exc

    frame = _strip_string_columns(frame)
    frame = _coerce_integer_columns(
        frame,
        ("player_id", "height"),
        path,
    )
    raw_dob = frame["dob"].astype("string")
    frame["dob"] = pd.to_datetime(
        raw_dob.where(raw_dob.str.fullmatch(r"\d{8}", na=False)),
        format="%Y%m%d",
        errors="coerce",
    )
    combined_name = (
        frame["name_first"].fillna("")
        .str.cat(frame["name_last"].fillna(""), sep=" ")
        .str.strip()
    )
    frame["player_name"] = combined_name.replace("", pd.NA).astype("string")
    frame["gender"] = pd.Series(gender, index=frame.index, dtype="string")

    if frame["player_id"].isna().any():
        raise SchemaMismatchError(f"{path}: player_id contiene valores nulos.")
    if frame["player_id"].duplicated().any():
        duplicated = (
            frame.loc[frame["player_id"].duplicated(), "player_id"]
            .astype(str)
            .tolist()
        )
        raise SchemaMismatchError(
            f"{path}: player_id duplicados dentro del género: {duplicated[:10]}."
        )
    return frame


def load_players(
    gender: str | None = None,
    *,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Carga los maestros de jugadores separados o unidos por género.

    Args:
        gender: ``"M"`` o ``"F"`` para un género; ``None`` para ambos.
        raw_dir: Raíz local que contiene los subdirectorios ``atp`` y ``wta``.

    Returns:
        DataFrame tipado con los campos fuente, ``player_name`` y ``gender``.
    """

    genders = _normalise_gender(gender)
    frames: list[pd.DataFrame] = []
    for selected_gender in genders:
        path = (
            raw_dir.resolve()
            / GENDER_DIRECTORY[selected_gender]
            / PLAYER_FILENAME[selected_gender]
        )
        if not path.is_file():
            raise MissingRawDataError(
                f"No existe el archivo de jugadores esperado: {path}."
            )
        frames.append(_read_players_file(path, selected_gender))
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_match_coverage(matches: pd.DataFrame) -> pd.DataFrame:
    """Resume número de partidos y rango de fechas por género y código de nivel."""

    required = {"gender", "tour_level", "tourney_date"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise DataLoaderError(
            f"No se puede resumir la cobertura; faltan columnas: {missing}."
        )
    if not is_datetime64_any_dtype(matches["tourney_date"]):
        raise DataLoaderError("tourney_date debe ser datetime para resumir.")

    summary = (
        matches.groupby(["gender", "tour_level"], dropna=False)
        .agg(
            matches=("tourney_date", "size"),
            min_date=("tourney_date", "min"),
            max_date=("tourney_date", "max"),
        )
        .reset_index()
        .sort_values(["gender", "tour_level"], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def audit_match_file_coverage(
    *,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Resume los años de archivos locales y expone huecos sin ocultarlos."""

    match_files = _discover_match_files(raw_dir.resolve(), ("M", "F"))
    records: list[dict[str, object]] = []
    for family in MATCH_FAMILY_PATTERNS:
        family_files = [
            match_file
            for match_file in match_files
            if match_file.source_family == family
        ]
        years = sorted(match_file.year for match_file in family_files)
        missing_years = sorted(
            set(range(years[0], years[-1] + 1)).difference(years)
        )
        records.append(
            {
                "gender": FAMILY_GENDER[family],
                "source_family": family,
                "file_count": len(years),
                "first_file_year": years[0],
                "last_file_year": years[-1],
                "missing_file_years": ",".join(map(str, missing_years)),
            }
        )
    return pd.DataFrame.from_records(records)

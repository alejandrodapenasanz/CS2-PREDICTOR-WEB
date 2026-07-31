"""Particiones temporales expansivas por temporada sin observaciones futuras.

Para una temporada de test ``Y``, el contrato es fijo:

* entrenamiento: temporadas ``<= Y - 2``;
* calibración: temporada ``Y - 1``;
* test: temporada ``Y``.

Las posiciones almacenadas son enteros de posición, no etiquetas del índice de
Pandas. Así, índices repetidos o no consecutivos no pueden hacer que ``loc``
seleccione accidentalmente filas ajenas al bloque previsto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class TemporalSplitError(ValueError):
    """Indica que no puede construirse o validarse un fold causal."""


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """Describe un fold temporal mediante posiciones ``iloc`` inmutables."""

    test_season: int
    train_positions: tuple[int, ...]
    calibration_positions: tuple[int, ...]
    test_positions: tuple[int, ...]

    @property
    def calibration_season(self) -> int:
        """Devuelve la única temporada permitida para calibrar este fold."""

        return self.test_season - 1

    @property
    def train_end_season(self) -> int:
        """Devuelve la última temporada admisible en entrenamiento."""

        return self.test_season - 2

    def take(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extrae copias de train, calibración y test en ese orden."""

        return (
            frame.iloc[list(self.train_positions)].copy(),
            frame.iloc[list(self.calibration_positions)].copy(),
            frame.iloc[list(self.test_positions)].copy(),
        )


def _coerce_dates(
    data: pd.DataFrame | pd.Series | Sequence[object],
    date_column: str,
) -> pd.Series:
    """Convierte la entrada en fechas normalizadas y rechaza valores inválidos."""

    if isinstance(data, pd.DataFrame):
        if date_column not in data.columns:
            raise TemporalSplitError(
                f"No existe la columna temporal {date_column!r}."
            )
        raw_dates = data[date_column]
    elif isinstance(data, pd.Series):
        raw_dates = data
    else:
        raw_dates = pd.Series(data)

    try:
        dates = pd.to_datetime(raw_dates, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise TemporalSplitError(
            "Las fechas contienen valores que no pueden interpretarse."
        ) from exc
    if dates.isna().any():
        raise TemporalSplitError("Las fechas no pueden contener nulos.")
    return dates.reset_index(drop=True)


def _validate_record_ids(
    data: pd.DataFrame | pd.Series | Sequence[object],
    record_id_column: str | None,
) -> None:
    """Comprueba unicidad de la identidad de fila cuando se solicita."""

    if record_id_column is None:
        return
    if not isinstance(data, pd.DataFrame):
        raise TemporalSplitError(
            "record_id_column solo puede usarse con un DataFrame."
        )
    if record_id_column not in data.columns:
        raise TemporalSplitError(
            f"No existe la columna de identidad {record_id_column!r}."
        )
    identifiers = data[record_id_column]
    if identifiers.isna().any():
        raise TemporalSplitError(
            f"{record_id_column!r} contiene identidades nulas."
        )
    duplicated = identifiers.duplicated(keep=False)
    if duplicated.any():
        examples = identifiers.loc[duplicated].astype(str).unique()[:3]
        raise TemporalSplitError(
            "Hay filas repetidas según "
            f"{record_id_column!r}: {examples.tolist()}."
        )


def build_expanding_season_folds(
    data: pd.DataFrame | pd.Series | Sequence[object],
    *,
    date_column: str = "match_date",
    test_seasons: Iterable[int] | None = None,
    record_id_column: str | None = None,
) -> tuple[TemporalFold, ...]:
    """Construye folds expansivos con calibración en la temporada previa.

    Args:
        data: DataFrame, serie o secuencia con una fecha por observación.
        date_column: Nombre de la fecha cuando ``data`` es un DataFrame.
        test_seasons: Temporadas de evaluación. Si se omite, se incluyen todas
            las temporadas observadas que tengan calibración en ``Y-1`` y al
            menos una observación de entrenamiento hasta ``Y-2``.
        record_id_column: Columna opcional cuya unicidad se verifica antes de
            construir los folds.

    Returns:
        Tupla ordenada de folds causales.

    Raises:
        TemporalSplitError: Si faltan bloques requeridos, fechas o identidades.
    """

    _validate_record_ids(data, record_id_column)
    dates = _coerce_dates(data, date_column)
    seasons = dates.dt.year.to_numpy(dtype=np.int64)
    observed = frozenset(int(value) for value in np.unique(seasons))

    if test_seasons is None:
        requested = tuple(
            season
            for season in sorted(observed)
            if season - 1 in observed and np.any(seasons <= season - 2)
        )
    else:
        try:
            requested = tuple(sorted({int(value) for value in test_seasons}))
        except (TypeError, ValueError) as exc:
            raise TemporalSplitError(
                "test_seasons debe contener años enteros."
            ) from exc
    if not requested:
        raise TemporalSplitError(
            "No hay temporadas suficientes para construir ningún fold."
        )

    folds: list[TemporalFold] = []
    for test_season in requested:
        train = tuple(
            int(value)
            for value in np.flatnonzero(seasons <= test_season - 2)
        )
        calibration = tuple(
            int(value)
            for value in np.flatnonzero(seasons == test_season - 1)
        )
        test = tuple(
            int(value)
            for value in np.flatnonzero(seasons == test_season)
        )
        missing: list[str] = []
        if not train:
            missing.append(f"train <= {test_season - 2}")
        if not calibration:
            missing.append(f"calibración = {test_season - 1}")
        if not test:
            missing.append(f"test = {test_season}")
        if missing:
            raise TemporalSplitError(
                f"Fold {test_season} incompleto: falta "
                + ", ".join(missing)
                + "."
            )
        fold = TemporalFold(
            test_season=test_season,
            train_positions=train,
            calibration_positions=calibration,
            test_positions=test,
        )
        validate_temporal_fold(dates, fold)
        folds.append(fold)
    return tuple(folds)


def validate_temporal_fold(
    dates: pd.Series | Sequence[object],
    fold: TemporalFold,
) -> None:
    """Audita separación, temporadas exactas y precedencia de un fold.

    Args:
        dates: Fecha de cada fila en el mismo orden usado al crear el fold.
        fold: Fold que se desea comprobar.

    Raises:
        TemporalSplitError: Si una posición se comparte, queda fuera del
            dataset o viola la precedencia temporal.
    """

    parsed_dates = _coerce_dates(dates, "match_date")
    partitions = {
        "train": fold.train_positions,
        "calibration": fold.calibration_positions,
        "test": fold.test_positions,
    }
    maximum_position = len(parsed_dates) - 1
    for name, positions in partitions.items():
        if not positions:
            raise TemporalSplitError(f"El bloque {name} está vacío.")
        if len(positions) != len(set(positions)):
            raise TemporalSplitError(
                f"El bloque {name} repite posiciones internamente."
            )
        if min(positions) < 0 or max(positions) > maximum_position:
            raise TemporalSplitError(
                f"El bloque {name} contiene posiciones fuera del dataset."
            )

    train_set = set(fold.train_positions)
    calibration_set = set(fold.calibration_positions)
    test_set = set(fold.test_positions)
    if (
        train_set & calibration_set
        or train_set & test_set
        or calibration_set & test_set
    ):
        raise TemporalSplitError(
            "Train, calibración y test deben ser disjuntos."
        )

    train_dates = parsed_dates.iloc[list(fold.train_positions)]
    calibration_dates = parsed_dates.iloc[list(fold.calibration_positions)]
    test_dates = parsed_dates.iloc[list(fold.test_positions)]
    if not (train_dates.dt.year <= fold.train_end_season).all():
        raise TemporalSplitError(
            "Entrenamiento contiene fechas posteriores a Y-2."
        )
    if not (
        calibration_dates.dt.year == fold.calibration_season
    ).all():
        raise TemporalSplitError(
            "Calibración debe contener exclusivamente la temporada Y-1."
        )
    if not (test_dates.dt.year == fold.test_season).all():
        raise TemporalSplitError(
            "Test debe contener exclusivamente la temporada Y."
        )
    if not (
        train_dates.max() < calibration_dates.min()
        and calibration_dates.max() < test_dates.min()
    ):
        raise TemporalSplitError(
            "La precedencia estricta train < calibración < test no se cumple."
        )

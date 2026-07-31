"""Esquema estable del dataset de entrenamiento de la fase 6.

Las columnas de procedencia y objetivo se conservan para auditoría, pero el
modelo de la fase 7 deberá seleccionar exclusivamente
``MODEL_FEATURE_COLUMNS``. Esta separación evita que identificadores o la
orientación del resultado entren accidentalmente como predictores.
"""

from __future__ import annotations

from typing import Final, Mapping

from .vector import MODEL_FEATURE_COLUMNS, VECTOR_COLUMNS


AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "source_commit",
    "source_path",
    "source_row_number",
    "tourney_id",
    "tourney_name",
    "match_num",
)
TARGET_COLUMN: Final[str] = "y"
TRAINING_COLUMNS: Final[tuple[str, ...]] = (
    *AUDIT_COLUMNS,
    *VECTOR_COLUMNS,
    TARGET_COLUMN,
)

_STRING_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "record_id",
        "source_commit",
        "source_path",
        "tourney_id",
        "tourney_name",
        "match_num",
        "gender",
        "surface",
        "tour_level_raw",
        "tour_level",
        "source_family",
        "round",
    }
)
_DATE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "match_date",
        "ranking_date_a",
        "ranking_date_b",
        "birth_date_a",
        "birth_date_b",
    }
)
_TIMESTAMP_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "market_retrieved_at_utc",
        "prediction_as_of_utc",
    }
)
_BOOLEAN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "elo_cold_start_a",
        "elo_cold_start_b",
        "ranking_missing_a",
        "ranking_missing_b",
        "age_missing_a",
        "age_missing_b",
        "age_invalid_for_date_a",
        "age_invalid_for_date_b",
    }
)
_INTEGER_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "source_row_number",
        "player_a_id",
        "player_b_id",
        "best_of",
        "elo_general_matches_a",
        "elo_general_matches_b",
        "elo_general_matches_diff",
        "elo_surface_matches_a",
        "elo_surface_matches_b",
        "elo_surface_matches_diff",
        "recent_n_matches_a",
        "recent_n_matches_b",
        "recent_months_matches_a",
        "recent_months_matches_b",
        "h2h_global_matches",
        "h2h_surface_matches",
        "rest_days_a",
        "rest_days_b",
        "rest_days_diff",
        "rank_a",
        "rank_b",
        "rank_diff",
        "rank_points_a",
        "rank_points_b",
        "rank_points_diff",
        "ranking_age_days_a",
        "ranking_age_days_b",
        "ranking_conflict_dates_skipped_a",
        "ranking_conflict_dates_skipped_b",
    }
)
_FLOAT_COLUMNS: Final[frozenset[str]] = frozenset(
    set(TRAINING_COLUMNS).difference(
        _STRING_COLUMNS,
        _DATE_COLUMNS,
        _TIMESTAMP_COLUMNS,
        _BOOLEAN_COLUMNS,
        _INTEGER_COLUMNS,
        {TARGET_COLUMN},
    )
)
_NON_NULL_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "record_id",
        "source_commit",
        "source_path",
        "source_row_number",
        "gender",
        "match_date",
        "player_a_id",
        "player_b_id",
        "tour_level_raw",
        "tour_level",
        "source_family",
        "elo_general_a",
        "elo_general_b",
        "elo_general_diff",
        "elo_surface_a",
        "elo_surface_b",
        "elo_surface_diff",
        "h2h_global_balance",
        "h2h_global_matches",
        TARGET_COLUMN,
    }
)


class FeatureSchemaError(ValueError):
    """Indica que una fila o el runtime no cumple el esquema declarado."""


def validate_training_row(row: Mapping[str, object]) -> None:
    """Comprueba nombres, orden y etiqueta de una fila antes de persistirla."""

    if tuple(row) != TRAINING_COLUMNS:
        missing = sorted(set(TRAINING_COLUMNS).difference(row))
        extra = sorted(set(row).difference(TRAINING_COLUMNS))
        raise FeatureSchemaError(
            "La fila no coincide con TRAINING_COLUMNS: "
            f"missing={missing}, extra={extra}."
        )
    y = row[TARGET_COLUMN]
    if isinstance(y, bool) or not isinstance(y, int) or y not in {0, 1}:
        raise FeatureSchemaError("y debe ser el entero 0 o 1.")
    for column in _NON_NULL_COLUMNS:
        if row[column] is None:
            raise FeatureSchemaError(
                f"La columna obligatoria {column!r} no puede ser nula."
            )


def training_arrow_schema():
    """Construye el esquema PyArrow sin imponer la dependencia al importar.

    Returns:
        ``pyarrow.Schema`` con orden, tipos y nulabilidad estables.

    Raises:
        FeatureSchemaError: Si PyArrow no está instalado o la clasificación
            interna de columnas quedó incompleta.
    """

    try:
        import pyarrow as pa
    except ImportError as exc:
        raise FeatureSchemaError(
            "Falta pyarrow; instale TENNIS/requirements.txt."
        ) from exc

    classified = (
        _STRING_COLUMNS
        | _DATE_COLUMNS
        | _TIMESTAMP_COLUMNS
        | _BOOLEAN_COLUMNS
        | _INTEGER_COLUMNS
        | _FLOAT_COLUMNS
        | {TARGET_COLUMN}
    )
    if classified != set(TRAINING_COLUMNS):
        raise FeatureSchemaError(
            "La clasificación de tipos no cubre exactamente el dataset."
        )

    fields = []
    for column in TRAINING_COLUMNS:
        if column in _STRING_COLUMNS:
            data_type = pa.string()
        elif column in _DATE_COLUMNS:
            data_type = pa.date32()
        elif column in _TIMESTAMP_COLUMNS:
            data_type = pa.timestamp("us", tz="UTC")
        elif column in _BOOLEAN_COLUMNS:
            data_type = pa.bool_()
        elif column in _INTEGER_COLUMNS:
            data_type = pa.int64()
        elif column == TARGET_COLUMN:
            data_type = pa.int8()
        elif column in _FLOAT_COLUMNS:
            data_type = pa.float64()
        else:
            raise FeatureSchemaError(
                f"No existe tipo Arrow para {column!r}."
            )
        fields.append(
            pa.field(
                column,
                data_type,
                nullable=column not in _NON_NULL_COLUMNS,
            )
        )
    return pa.schema(fields)


if not set(MODEL_FEATURE_COLUMNS).issubset(VECTOR_COLUMNS):
    raise RuntimeError(
        "MODEL_FEATURE_COLUMNS debe ser subconjunto de VECTOR_COLUMNS."
    )

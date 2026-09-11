"""Validación tabular de predicciones, observaciones y estadísticas.

Este módulo solo normaliza valores y determina elegibilidad. No accede a
SQLite y nunca infiere un ganador desde posición, nombre o probabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd

from .identifiers import (
    canonical_date,
    canonical_slug,
    canonical_utc_datetime,
    optional_scalar,
    optional_text,
    required_text,
)
from .types import OperationsValidationError
from ..temporal import DEFAULT_SOURCE_DATE_POLICY


PREDICTION_REQUIRED_COLUMNS = frozenset(
    {
        "prediction_date",
        "prediction_as_of_utc",
        "source_retrieved_at_utc",
        "tournament",
        "gender",
        "status",
        "player_a_slug",
        "player_b_slug",
        "mapping_status",
        "model_probability_a",
        "model_probability_b",
        "prediction_status",
        "model_fingerprint",
        "model_training_max_date",
        "model_training_available_max_date",
    }
)

OBSERVATION_REQUIRED_COLUMNS = frozenset(
    {
        "source_match_id",
        "status",
        "player_1_sets_won",
        "player_2_sets_won",
        "sets_score",
        "winner_side",
        "winner_slug",
        "result_evidence",
    }
)

STATISTICS_REQUIRED_COLUMNS = frozenset(
    {
        "source_match_id",
        "player_slug",
        "gender",
        "as_of_date",
        "statistic_name",
        "statistic_value",
        "source_kind",
    }
)

_FORBIDDEN_STATISTIC_TOKENS = frozenset(
    {
        "prediction",
        "probability",
        "edge",
        "winner",
        "result",
        "settlement",
        "label",
        "target",
    }
)


@dataclass(frozen=True, slots=True)
class PredictionValidity:
    """Valores normalizados y elegibilidad oficial de una predicción."""

    is_valid: bool
    invalid_reason: str | None
    match_date: str
    prediction_as_of_utc: str | None
    source_retrieved_at_utc: str | None
    scheduled_start_utc: str | None
    player_a_slug: str | None
    player_b_slug: str | None
    model_probability_raw_a: float | None
    model_probability_a: float | None
    model_probability_b: float | None
    model_training_max_date: str | None
    model_training_available_max_date: str | None


@dataclass(frozen=True, slots=True)
class ObservationValidity:
    """Resultado normalizado sin derivar identidad desde posiciones."""

    is_valid: bool
    invalid_reason: str | None
    status: str
    player_1_sets_won: int | None
    player_2_sets_won: int | None
    sets_score: str | None
    winner_side: str | None
    winner_slug: str | None
    result_evidence: str | None


def require_frame_columns(
    frame: pd.DataFrame,
    required: frozenset[str],
    *,
    frame_name: str,
) -> None:
    """Exige el esquema de una tabla no vacía."""

    if not isinstance(frame, pd.DataFrame):
        raise OperationsValidationError(
            f"{frame_name} debe ser un DataFrame."
        )
    if frame.empty:
        return
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise OperationsValidationError(
            f"{frame_name} carece de columnas obligatorias: {missing}."
        )


def optional_float(
    value: object,
    field_name: str,
    *,
    probability: bool = False,
) -> float | None:
    """Convierte un número finito y, si procede, exige ``[0, 1]``."""

    scalar = optional_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        raise OperationsValidationError(
            f"{field_name} no puede ser booleano."
        )
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OperationsValidationError(
            f"{field_name} no es numérico."
        ) from exc
    if not math.isfinite(number):
        raise OperationsValidationError(f"{field_name} debe ser finito.")
    if probability and not 0.0 <= number <= 1.0:
        raise OperationsValidationError(
            f"{field_name} debe estar dentro de [0, 1]."
        )
    return number


def optional_integer(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
) -> int | None:
    """Convierte un entero exacto y rechaza booleanos y decimales."""

    scalar = optional_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        raise OperationsValidationError(
            f"{field_name} no puede ser booleano."
        )
    try:
        integer = int(scalar)
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OperationsValidationError(
            f"{field_name} no es un entero."
        ) from exc
    if numeric != integer or (positive and integer <= 0):
        qualifier = "positivo " if positive else ""
        raise OperationsValidationError(
            f"{field_name} debe ser un entero {qualifier}exacto."
        )
    return integer


def _prediction_failure_reasons(
    row: dict[str, object],
    *,
    match_date: str,
    prediction_as_of_utc: str | None,
    source_retrieved_at_utc: str | None,
    scheduled_start_utc: str | None,
    player_a_slug: str | None,
    player_b_slug: str | None,
    probability_a: float | None,
    probability_b: float | None,
    model_training_max_date: str | None,
    model_training_available_max_date: str | None,
) -> tuple[str, ...]:
    """Enumera razones que excluyen una fila de la selección oficial."""

    reasons: list[str] = []
    if optional_text(row.get("prediction_status"), "prediction_status") != (
        "predicted"
    ):
        reasons.append("prediction_status_not_predicted")
    if optional_text(row.get("status"), "status") != "scheduled":
        reasons.append("source_status_not_scheduled")
    if optional_text(row.get("mapping_status"), "mapping_status") != "mapped":
        reasons.append("mapping_not_complete")
    if (
        player_a_slug is None
        or player_b_slug is None
        or player_a_slug == player_b_slug
    ):
        reasons.append("player_slugs_not_distinct")
    if probability_a is None or probability_b is None:
        reasons.append("model_probability_missing")
    elif not math.isclose(
        probability_a + probability_b,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append("model_probabilities_not_complementary")
    if optional_text(row.get("model_fingerprint"), "model_fingerprint") is None:
        reasons.append("model_fingerprint_missing")
    if prediction_as_of_utc is None:
        reasons.append("prediction_timestamp_missing")
    if source_retrieved_at_utc is None:
        reasons.append("source_timestamp_missing")
    if scheduled_start_utc is None:
        reasons.append("scheduled_start_utc_missing")
    if (
        prediction_as_of_utc is not None
        and source_retrieved_at_utc is not None
        and not pd.Timestamp(source_retrieved_at_utc)
        < pd.Timestamp(prediction_as_of_utc)
    ):
        reasons.append("source_not_strictly_before_prediction")
    match_day = date.fromisoformat(match_date)
    if (
        scheduled_start_utc is not None
        and pd.Timestamp(scheduled_start_utc).date() != match_day
    ):
        reasons.append("scheduled_start_date_conflict")
    if (
        prediction_as_of_utc is not None
        and scheduled_start_utc is not None
        and not pd.Timestamp(prediction_as_of_utc)
        < pd.Timestamp(scheduled_start_utc)
    ):
        reasons.append("prediction_not_strictly_before_scheduled_start")
    if (
        prediction_as_of_utc is not None
        and pd.Timestamp(prediction_as_of_utc).date() > match_day
    ):
        reasons.append("prediction_created_after_match_date")
    if (
        source_retrieved_at_utc is not None
        and pd.Timestamp(source_retrieved_at_utc).date() > match_day
    ):
        reasons.append("source_captured_after_match_date")
    if model_training_max_date is None:
        reasons.append("model_training_max_date_missing")
    if model_training_available_max_date is None:
        reasons.append("model_training_available_max_date_missing")
    else:
        available_day = date.fromisoformat(
            model_training_available_max_date
        )
        if not available_day < match_day:
            reasons.append("model_results_not_available_before_match")
        if (
            model_training_max_date is not None
            and DEFAULT_SOURCE_DATE_POLICY.availability_date(
                date.fromisoformat(model_training_max_date)
            )
            != available_day
        ):
            reasons.append("model_training_dates_inconsistent")
    return tuple(reasons)


def validate_prediction_row(
    row: dict[str, object],
) -> PredictionValidity:
    """Normaliza una fila y decide si puede ser predicción oficial."""

    match_date = canonical_date(
        row.get("prediction_date"),
        "prediction_date",
    )
    gender = required_text(row.get("gender"), "gender")
    if gender not in {"M", "F"}:
        raise OperationsValidationError("gender debe ser 'M' o 'F'.")
    prediction_as_of = canonical_utc_datetime(
        row.get("prediction_as_of_utc"),
        "prediction_as_of_utc",
        required=False,
    )
    source_retrieved = canonical_utc_datetime(
        row.get("source_retrieved_at_utc"),
        "source_retrieved_at_utc",
        required=False,
    )
    scheduled_start = canonical_utc_datetime(
        row.get("scheduled_start_utc"),
        "scheduled_start_utc",
        required=False,
    )
    slug_a = canonical_slug(row.get("player_a_slug"), "player_a_slug")
    slug_b = canonical_slug(row.get("player_b_slug"), "player_b_slug")
    raw_probability = optional_float(
        row.get("model_probability_raw_a"),
        "model_probability_raw_a",
        probability=True,
    )
    probability_a = optional_float(
        row.get("model_probability_a"),
        "model_probability_a",
        probability=True,
    )
    probability_b = optional_float(
        row.get("model_probability_b"),
        "model_probability_b",
        probability=True,
    )
    training_date_value = optional_scalar(
        row.get("model_training_max_date")
    )
    model_training_max_date = (
        canonical_date(
            training_date_value,
            "model_training_max_date",
        )
        if training_date_value is not None
        else None
    )
    available_date_value = optional_scalar(
        row.get("model_training_available_max_date")
    )
    model_training_available_max_date = (
        canonical_date(
            available_date_value,
            "model_training_available_max_date",
        )
        if available_date_value is not None
        else None
    )
    reasons = _prediction_failure_reasons(
        row,
        match_date=match_date,
        prediction_as_of_utc=prediction_as_of,
        source_retrieved_at_utc=source_retrieved,
        scheduled_start_utc=scheduled_start,
        player_a_slug=slug_a,
        player_b_slug=slug_b,
        probability_a=probability_a,
        probability_b=probability_b,
        model_training_max_date=model_training_max_date,
        model_training_available_max_date=(
            model_training_available_max_date
        ),
    )
    return PredictionValidity(
        is_valid=not reasons,
        invalid_reason="|".join(reasons) if reasons else None,
        match_date=match_date,
        prediction_as_of_utc=prediction_as_of,
        source_retrieved_at_utc=source_retrieved,
        scheduled_start_utc=scheduled_start,
        player_a_slug=slug_a,
        player_b_slug=slug_b,
        model_probability_raw_a=raw_probability,
        model_probability_a=probability_a,
        model_probability_b=probability_b,
        model_training_max_date=model_training_max_date,
        model_training_available_max_date=(
            model_training_available_max_date
        ),
    )


def _invalid_observation(
    *,
    status: str,
    reason: str,
    first_sets: int | None,
    second_sets: int | None,
    sets_score: str | None,
    winner_side: str | None,
    winner_slug: str | None,
    evidence: str | None,
) -> ObservationValidity:
    """Construye un resultado conservado pero no liquidable."""

    return ObservationValidity(
        is_valid=False,
        invalid_reason=reason,
        status=status,
        player_1_sets_won=first_sets,
        player_2_sets_won=second_sets,
        sets_score=sets_score,
        winner_side=winner_side,
        winner_slug=winner_slug,
        result_evidence=evidence,
    )


def validate_observation_row(
    row: dict[str, object],
) -> ObservationValidity:
    """Valida evidencia terminal sin inferir el ganador desde el marcador."""

    status = required_text(row.get("status"), "status")
    first_sets = optional_integer(
        row.get("player_1_sets_won"),
        "player_1_sets_won",
    )
    second_sets = optional_integer(
        row.get("player_2_sets_won"),
        "player_2_sets_won",
    )
    sets_score = optional_text(row.get("sets_score"), "sets_score")
    winner_side = optional_text(row.get("winner_side"), "winner_side")
    winner_slug = canonical_slug(row.get("winner_slug"), "winner_slug")
    evidence = optional_text(row.get("result_evidence"), "result_evidence")

    if status != "finished":
        if winner_side is not None or winner_slug is not None:
            return _invalid_observation(
                status=status,
                reason="winner_present_for_non_finished_status",
                first_sets=first_sets,
                second_sets=second_sets,
                sets_score=sets_score,
                winner_side=winner_side,
                winner_slug=winner_slug,
                evidence=evidence,
            )
        return _invalid_observation(
            status=status,
            reason="status_not_finished",
            first_sets=first_sets,
            second_sets=second_sets,
            sets_score=sets_score,
            winner_side=None,
            winner_slug=None,
            evidence=evidence,
        )

    missing: list[str] = []
    if first_sets is None or second_sets is None:
        missing.append("sets_won")
    if sets_score is None:
        missing.append("sets_score")
    if winner_side not in {"player_1", "player_2"}:
        missing.append("winner_side")
    if winner_slug is None:
        missing.append("winner_slug")
    if evidence is None:
        missing.append("result_evidence")
    if missing:
        return _invalid_observation(
            status=status,
            reason="missing_terminal_evidence:" + ",".join(missing),
            first_sets=first_sets,
            second_sets=second_sets,
            sets_score=sets_score,
            winner_side=winner_side,
            winner_slug=winner_slug,
            evidence=evidence,
        )

    assert first_sets is not None
    assert second_sets is not None
    if first_sets < 0 or second_sets < 0 or first_sets == second_sets:
        reason = "non_terminal_set_totals"
    elif winner_side == "player_1" and first_sets <= second_sets:
        reason = "winner_side_contradicts_set_totals"
    elif winner_side == "player_2" and second_sets <= first_sets:
        reason = "winner_side_contradicts_set_totals"
    else:
        reason = ""
    if reason:
        return _invalid_observation(
            status=status,
            reason=reason,
            first_sets=first_sets,
            second_sets=second_sets,
            sets_score=sets_score,
            winner_side=winner_side,
            winner_slug=winner_slug,
            evidence=evidence,
        )
    return ObservationValidity(
        is_valid=True,
        invalid_reason=None,
        status=status,
        player_1_sets_won=first_sets,
        player_2_sets_won=second_sets,
        sets_score=sets_score,
        winner_side=winner_side,
        winner_slug=winner_slug,
        result_evidence=evidence,
    )


def validate_statistic_name(name: object) -> str:
    """Rechaza explícitamente predicciones, resultados y labels como stats."""

    text = required_text(name, "statistic_name")
    tokens = {
        token
        for token in text.casefold().replace("-", "_").split("_")
        if token
    }
    forbidden = sorted(tokens.intersection(_FORBIDDEN_STATISTIC_TOKENS))
    if forbidden:
        raise OperationsValidationError(
            "statistic_name intenta persistir señal de predicción/resultado: "
            f"{forbidden}."
        )
    return text


def validate_source_kind(value: object) -> str:
    """Impide declarar predicciones o resultados como fuente estadística."""

    source_kind = required_text(value, "source_kind")
    if source_kind.casefold() in {
        "prediction",
        "settlement",
        "result",
        "label",
    }:
        raise OperationsValidationError(
            "source_kind no puede proceder de predicción, label o resultado."
        )
    return source_kind

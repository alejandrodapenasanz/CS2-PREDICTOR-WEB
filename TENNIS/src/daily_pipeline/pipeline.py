"""Orquestador diario de scraping, mapping, features, modelo y publicación.

La orientación es estable y observable: jugador A es siempre ``player_1`` de
Tennis Explorer y B es ``player_2``. Solo se predicen filas cuyo estado sea
``scheduled`` en el snapshot, ambos jugadores estén mapeados y el modelo haya
sido entrenado exclusivamente con fechas anteriores al partido. El resto se
conserva con probabilidades nulas y razones explícitas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, cast

import numpy as np
import pandas as pd

from ..config import (
    FEATURE_DATASET_MANIFEST_PATH,
    IDENTITY_QUARANTINE_MANIFEST_PATH,
    IDENTITY_QUARANTINE_PATH,
    PREDICTIONS_PROCESSED_DIR,
    PROJECT_ROOT,
)
from ..elo import Gender
from ..features import (
    MatchFeatureRequest,
    calculate_two_way_market_probabilities,
)
from ..identity_integrity import load_identity_quarantine
from ..modeling.data import load_feature_source_manifest
from ..modeling.service import (
    LoadedDeploymentModel,
    load_active_deployment_model,
)
from ..player_mapping import resolve_scraped_matches
from ..tennis_explorer import get_daily_matches
from .confidence import (
    ConfidenceAssessment,
    assess_vector_confidence,
    unavailable_confidence,
)
from .context import (
    DailyFeatureContext,
    build_daily_feature_context,
)


PREDICTION_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "prediction_date",
    "prediction_as_of_utc",
    "source_retrieved_at_utc",
    "source_snapshot_sha256",
    "source_snapshot_age_minutes",
    "source_match_id",
    "match_detail_href",
    "tournament",
    "tour_level",
    "canonical_tour_level",
    "gender",
    "surface",
    "scheduled_time",
    "status",
    "status_evidence",
    "player_a_name",
    "player_b_name",
    "player_a_slug",
    "player_b_slug",
    "player_a_id",
    "player_b_id",
    "mapping_status",
    "identity_quarantined",
    "best_of",
    "round",
    "odds_a",
    "odds_b",
    "model_probability_raw_a",
    "model_probability_a",
    "model_probability_b",
    "market_probability_a",
    "market_probability_b",
    "edge_a",
    "edge_b",
    "market_comparison_status",
    "predicted_winner_name",
    "predicted_winner_slug",
    "predicted_winner_probability",
    "elo_general_a",
    "elo_general_b",
    "elo_surface_a",
    "elo_surface_b",
    "elo_general_matches_a",
    "elo_general_matches_b",
    "elo_surface_matches_a",
    "elo_surface_matches_b",
    "recent_n_win_rate_a",
    "recent_n_win_rate_b",
    "recent_n_matches_a",
    "recent_n_matches_b",
    "recent_months_win_rate_a",
    "recent_months_win_rate_b",
    "recent_months_matches_a",
    "recent_months_matches_b",
    "h2h_global_balance",
    "h2h_global_matches",
    "h2h_surface_balance",
    "h2h_surface_matches",
    "rest_days_a",
    "rest_days_b",
    "ranking_date_a",
    "ranking_date_b",
    "rank_a",
    "rank_b",
    "rank_points_a",
    "rank_points_b",
    "birth_date_a",
    "birth_date_b",
    "age_a",
    "age_b",
    "confidence",
    "confidence_flags",
    "prediction_status",
    "model_profile",
    "model_fingerprint",
    "model_training_max_date",
    "feature_history_max_date",
    "ranking_source_max_date",
    "feature_fingerprint",
)
MAX_DAILY_SNAPSHOT_AGE_MINUTES: Final[int] = 120
_VECTOR_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "elo_general_a",
    "elo_general_b",
    "elo_surface_a",
    "elo_surface_b",
    "elo_general_matches_a",
    "elo_general_matches_b",
    "elo_surface_matches_a",
    "elo_surface_matches_b",
    "recent_n_win_rate_a",
    "recent_n_win_rate_b",
    "recent_n_matches_a",
    "recent_n_matches_b",
    "recent_months_win_rate_a",
    "recent_months_win_rate_b",
    "recent_months_matches_a",
    "recent_months_matches_b",
    "h2h_global_balance",
    "h2h_global_matches",
    "h2h_surface_balance",
    "h2h_surface_matches",
    "rest_days_a",
    "rest_days_b",
    "ranking_date_a",
    "ranking_date_b",
    "rank_a",
    "rank_b",
    "rank_points_a",
    "rank_points_b",
    "birth_date_a",
    "birth_date_b",
    "age_a",
    "age_b",
)


class DailyPredictionError(RuntimeError):
    """Indica que el pipeline no puede producir una salida auditable."""


@dataclass(frozen=True, slots=True)
class DailyPredictionRun:
    """Resultado publicado de una ejecución diaria completa."""

    match_date: date
    prediction_as_of_utc: datetime
    predictions: pd.DataFrame
    output_path: Path | None

    @property
    def predicted_matches(self) -> int:
        """Cuenta filas con una probabilidad calibrada disponible."""

        return int(self.predictions["model_probability_a"].notna().sum())


def _read_clock(clock: Callable[[], datetime] | None) -> datetime:
    """Obtiene un instante consciente de zona y lo normaliza a UTC."""

    observed = clock() if clock is not None else datetime.now().astimezone()
    if not isinstance(observed, datetime):
        raise DailyPredictionError("clock debe devolver datetime.")
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise DailyPredictionError("clock debe devolver un instante con zona.")
    return observed.astimezone(UTC)


def _optional_text(value: object) -> str | None:
    """Convierte un escalar textual y conserva ausencias como ``None``."""

    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    """Convierte un escalar numérico finito y conserva ausencias."""

    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, bool):
        raise DailyPredictionError("Una cuota no puede ser booleana.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DailyPredictionError(f"Valor numérico inválido: {value!r}.") from exc
    if not math.isfinite(result):
        raise DailyPredictionError("Los valores numéricos deben ser finitos.")
    return result


def _aware_datetime(value: object, field_name: str) -> datetime:
    """Convierte un timestamp pandas/Python y exige zona horaria."""

    if value is None or bool(pd.isna(value)):
        raise DailyPredictionError(f"{field_name} no puede ser nulo.")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DailyPredictionError(f"{field_name} no es un timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DailyPredictionError(f"{field_name} debe incluir zona horaria.")
    return parsed.to_pydatetime().astimezone(UTC)


def _validate_mapped_frame(
    mapped_matches: pd.DataFrame,
    match_date: date,
) -> None:
    """Comprueba el contrato combinado de fases 4 y 5 antes de inferir."""

    required = {
        "match_date",
        "tournament",
        "tour_level",
        "gender",
        "surface",
        "scheduled_time",
        "status",
        "player_1_name",
        "player_2_name",
        "player_1_slug",
        "player_2_slug",
        "player_1_id",
        "player_2_id",
        "player_1_odds",
        "player_2_odds",
        "mapping_status",
        "retrieved_at_utc",
        "snapshot_sha256",
        "source_match_id",
        "match_detail_href",
        "status_evidence",
    }
    if not isinstance(mapped_matches, pd.DataFrame):
        raise DailyPredictionError("mapped_matches debe ser un DataFrame.")
    missing = sorted(required.difference(mapped_matches.columns))
    if missing:
        raise DailyPredictionError(
            f"La cartelera mapeada carece de columnas: {missing}."
        )
    if not mapped_matches.index.is_unique:
        raise DailyPredictionError("El índice de la cartelera debe ser único.")
    if mapped_matches.empty:
        return
    dates = pd.to_datetime(
        mapped_matches["match_date"],
        errors="raise",
    ).dt.date
    if not bool((dates == match_date).all()):
        raise DailyPredictionError(
            "La cartelera mezcla fechas o contradice la fecha solicitada."
        )
    genders = set(mapped_matches["gender"].dropna().astype(str).unique())
    if not genders.issubset({"M", "F"}) or mapped_matches["gender"].isna().any():
        raise DailyPredictionError(f"Géneros diarios inválidos: {genders}.")


def _validate_source_match_ids(mapped_matches: pd.DataFrame) -> None:
    """Exige una clave fuente opaca, poblada y única por snapshot."""

    if mapped_matches.empty:
        return
    match_ids = mapped_matches["source_match_id"].astype("string").str.strip()
    if match_ids.isna().any() or match_ids.eq("").any():
        raise DailyPredictionError("source_match_id no puede ser nulo o vacío.")
    if match_ids.duplicated().any():
        raise DailyPredictionError(
            "source_match_id debe ser único dentro del snapshot diario."
        )


def _base_output(
    mapped_matches: pd.DataFrame,
    *,
    match_date: date,
    prediction_as_of_utc: datetime,
) -> pd.DataFrame:
    """Crea la salida de auditoría antes de completar probabilidades."""

    result = pd.DataFrame(index=mapped_matches.index)
    result["prediction_date"] = match_date.isoformat()
    result["prediction_as_of_utc"] = prediction_as_of_utc
    result["source_retrieved_at_utc"] = mapped_matches["retrieved_at_utc"]
    result["source_snapshot_sha256"] = mapped_matches["snapshot_sha256"]
    retrieved = pd.to_datetime(
        mapped_matches["retrieved_at_utc"],
        errors="raise",
        utc=True,
    )
    snapshot_age = (
        pd.Timestamp(prediction_as_of_utc) - retrieved
    ).dt.total_seconds() / 60.0
    if (snapshot_age < 0.0).any():
        raise DailyPredictionError(
            "El snapshot diario no puede ser posterior a la predicción."
        )
    result["source_snapshot_age_minutes"] = snapshot_age.astype(float)
    result["source_match_id"] = mapped_matches["source_match_id"]
    result["match_detail_href"] = mapped_matches["match_detail_href"]
    result["tournament"] = mapped_matches["tournament"]
    result["tour_level"] = mapped_matches["tour_level"]
    result["canonical_tour_level"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["gender"] = mapped_matches["gender"]
    result["surface"] = mapped_matches["surface"]
    result["scheduled_time"] = mapped_matches["scheduled_time"]
    result["status"] = mapped_matches["status"]
    result["status_evidence"] = mapped_matches["status_evidence"]
    result["player_a_name"] = mapped_matches["player_1_name"]
    result["player_b_name"] = mapped_matches["player_2_name"]
    result["player_a_slug"] = mapped_matches["player_1_slug"]
    result["player_b_slug"] = mapped_matches["player_2_slug"]
    result["player_a_id"] = mapped_matches["player_1_id"]
    result["player_b_id"] = mapped_matches["player_2_id"]
    result["mapping_status"] = mapped_matches["mapping_status"]
    result["identity_quarantined"] = False
    result["best_of"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["round"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["odds_a"] = mapped_matches["player_1_odds"]
    result["odds_b"] = mapped_matches["player_2_odds"]
    for column in (
        "model_probability_raw_a",
        "model_probability_a",
        "model_probability_b",
        "market_probability_a",
        "market_probability_b",
        "edge_a",
        "edge_b",
    ):
        result[column] = np.nan
    result["market_comparison_status"] = pd.Series(
        "missing",
        index=result.index,
        dtype="string",
    )
    for column in ("predicted_winner_name", "predicted_winner_slug"):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["predicted_winner_probability"] = np.nan
    date_columns = {
        "ranking_date_a",
        "ranking_date_b",
        "birth_date_a",
        "birth_date_b",
    }
    for column in _VECTOR_AUDIT_COLUMNS:
        result[column] = pd.NaT if column in date_columns else np.nan
    result["confidence"] = pd.Series(
        "UNAVAILABLE",
        index=result.index,
        dtype="string",
    )
    result["confidence_flags"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["prediction_status"] = pd.Series(
        "not_predicted",
        index=result.index,
        dtype="string",
    )
    result["model_profile"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["model_fingerprint"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["model_training_max_date"] = pd.NaT
    result["feature_history_max_date"] = pd.NaT
    result["ranking_source_max_date"] = pd.NaT
    result["feature_fingerprint"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    return result


def _write_assessment(
    output: pd.DataFrame,
    row_index: object,
    assessment: ConfidenceAssessment,
) -> None:
    """Escribe una evaluación de confianza sin alterar probabilidades."""

    output.at[row_index, "confidence"] = assessment.level
    output.at[row_index, "confidence_flags"] = "|".join(assessment.flags)


def _validated_excluded_player_keys(
    keys: frozenset[tuple[str, int]] | set[tuple[str, int]],
) -> frozenset[tuple[Gender, int]]:
    """Valida claves completas de cuarentena sin mezclar géneros."""

    result: set[tuple[Gender, int]] = set()
    for key in keys:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in {"M", "F"}
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or key[1] <= 0
        ):
            raise DailyPredictionError(
                "excluded_player_keys contiene una identidad inválida."
            )
        result.add((cast(Gender, key[0]), key[1]))
    return frozenset(result)


def _row_is_quarantined(
    row: pd.Series,
    excluded_player_keys: frozenset[tuple[Gender, int]],
) -> bool:
    """Indica si cualquiera de los IDs mapeados tiene identidad ambigua."""

    if _optional_text(row["mapping_status"]) != "mapped":
        return False
    gender = cast(Gender, str(row["gender"]))
    return any(
        (gender, int(row[column])) in excluded_player_keys
        for column in ("player_1_id", "player_2_id")
    )


def _initial_row_assessment(
    row: pd.Series,
    *,
    excluded_player_keys: frozenset[tuple[Gender, int]],
) -> ConfidenceAssessment | None:
    """Devuelve una razón fatal previa a features, o ``None`` si es elegible."""

    flags: list[str] = []
    status = _optional_text(row["status"])
    if status != "scheduled":
        flags.append(f"status_{status or 'missing'}")
    if _optional_text(row["mapping_status"]) != "mapped":
        flags.append("player_unmapped")
    elif _row_is_quarantined(row, excluded_player_keys):
        flags.append("identity_quarantined")
    if flags:
        return unavailable_confidence(*flags)
    return None


def _extend_assessment(
    assessment: ConfidenceAssessment,
    *,
    low_flags: Sequence[str] = (),
    warning_flags: Sequence[str] = (),
) -> ConfidenceAssessment:
    """Añade evidencia operativa y degrada la etiqueta si corresponde."""

    flags = tuple(
        dict.fromkeys(
            (
                *assessment.flags,
                *(flag for flag in low_flags if flag),
                *(flag for flag in warning_flags if flag),
            )
        )
    )
    if low_flags:
        level = "LOW"
    elif warning_flags and assessment.level == "HIGH":
        level = "MEDIUM"
    else:
        level = assessment.level
    return ConfidenceAssessment(level=level, flags=flags)


def _model_training_date(model: LoadedDeploymentModel) -> date:
    """Parsea el corte de entrenamiento declarado por un bundle verificado."""

    try:
        return date.fromisoformat(model.training_max_date)
    except ValueError as exc:
        raise DailyPredictionError(
            f"training_max_date inválida para {model.gender}."
        ) from exc


def _target_ids(
    mapped_matches: pd.DataFrame,
    eligible_indexes: Sequence[object],
) -> dict[str, set[int]]:
    """Agrupa IDs mapeados por género para reconstrucción histórica dirigida."""

    targets: dict[str, set[int]] = {}
    for row_index in eligible_indexes:
        row = mapped_matches.loc[row_index]
        gender = str(row["gender"])
        targets.setdefault(gender, set()).update(
            (int(row["player_1_id"]), int(row["player_2_id"]))
        )
    return targets


def predict_mapped_matches(
    mapped_matches: pd.DataFrame,
    *,
    match_date: date,
    prediction_as_of_utc: datetime,
    models: Mapping[Gender, LoadedDeploymentModel],
    feature_context: DailyFeatureContext | None,
    excluded_player_keys: (
        frozenset[tuple[str, int]] | set[tuple[str, int]]
    ) = frozenset(),
) -> pd.DataFrame:
    """Genera probabilidades solo para filas prospectivas y causales."""

    _validate_mapped_frame(mapped_matches, match_date)
    _validate_source_match_ids(mapped_matches)
    quarantine = _validated_excluded_player_keys(excluded_player_keys)
    if (
        prediction_as_of_utc.tzinfo is None
        or prediction_as_of_utc.utcoffset() is None
    ):
        raise DailyPredictionError(
            "prediction_as_of_utc debe incluir zona horaria."
        )
    output = _base_output(
        mapped_matches,
        match_date=match_date,
        prediction_as_of_utc=prediction_as_of_utc,
    )
    if mapped_matches.empty:
        return output.loc[:, list(PREDICTION_OUTPUT_COLUMNS)]

    eligible_by_gender: dict[Gender, list[object]] = {"M": [], "F": []}
    for row_index, row in mapped_matches.iterrows():
        is_quarantined = _row_is_quarantined(row, quarantine)
        output.at[row_index, "identity_quarantined"] = is_quarantined
        initial = _initial_row_assessment(
            row,
            excluded_player_keys=quarantine,
        )
        if initial is not None:
            _write_assessment(output, row_index, initial)
            continue
        gender = cast(Gender, str(row["gender"]))
        model = models.get(gender)
        if model is None:
            _write_assessment(
                output,
                row_index,
                unavailable_confidence("model_missing"),
            )
            continue
        output.at[row_index, "model_profile"] = model.estimator.profile
        output.at[row_index, "model_fingerprint"] = model.run_fingerprint
        if model.estimator.profile == "market_enhanced":
            retrieved = _aware_datetime(
                row["retrieved_at_utc"],
                "retrieved_at_utc",
            )
            if retrieved.date() >= match_date:
                _write_assessment(
                    output,
                    row_index,
                    unavailable_confidence(
                        "market_not_strictly_before_match_date"
                    ),
                )
                continue
        training_date = _model_training_date(model)
        output.at[row_index, "model_training_max_date"] = pd.Timestamp(
            training_date
        )
        if training_date >= match_date:
            _write_assessment(
                output,
                row_index,
                unavailable_confidence("model_not_causal_for_date"),
            )
            continue
        eligible_by_gender[gender].append(row_index)

    causal_indexes = [
        row_index
        for indexes in eligible_by_gender.values()
        for row_index in indexes
    ]
    if causal_indexes and feature_context is None:
        raise DailyPredictionError(
            "Falta feature_context para partidos causalmente elegibles."
        )
    if feature_context is None:
        return output.loc[:, list(PREDICTION_OUTPUT_COLUMNS)]

    for gender, row_indexes in eligible_by_gender.items():
        if not row_indexes:
            continue
        gender_context = feature_context.by_gender.get(gender)
        if gender_context is None:
            raise DailyPredictionError(
                f"Falta contexto causal para el género {gender}."
            )
        model = models[gender]
        vector_rows: list[dict[str, object]] = []
        vector_indexes: list[object] = []
        vector_values: dict[object, dict[str, object]] = {}
        market_statuses: dict[object, str] = {}
        for row_index in row_indexes:
            row = mapped_matches.loc[row_index]
            odds_a = _optional_float(row["player_1_odds"])
            odds_b = _optional_float(row["player_2_odds"])
            has_market = odds_a is not None and odds_b is not None
            market_retrieved = (
                _aware_datetime(
                    row["retrieved_at_utc"],
                    "retrieved_at_utc",
                )
                if has_market
                else None
            )
            market = (
                calculate_two_way_market_probabilities(odds_a, odds_b)
                if has_market
                else None
            )
            market_status = (
                "missing"
                if market is None
                else (
                    "strictly_pre_date"
                    if (
                        market_retrieved is not None
                        and market_retrieved.date() < match_date
                    )
                    else "prestart_unverified"
                )
            )
            use_market_as_feature = (
                model.estimator.profile == "market_enhanced"
                and market_status == "strictly_pre_date"
            )
            surface = _optional_text(row["surface"])
            request = MatchFeatureRequest(
                gender=gender,
                player_a_id=int(row["player_1_id"]),
                player_b_id=int(row["player_2_id"]),
                as_of_date=match_date,
                surface=surface,
                tour_level_raw=str(row["tour_level"]),
                source_family="tennis_explorer",
                best_of=None,
                round=None,
                odds_a=odds_a if use_market_as_feature else None,
                odds_b=odds_b if use_market_as_feature else None,
                market_retrieved_at_utc=(
                    market_retrieved if use_market_as_feature else None
                ),
                prediction_as_of_utc=(
                    prediction_as_of_utc
                    if use_market_as_feature
                    else None
                ),
            )
            values = gender_context.builder.build(request).to_dict()
            if market is not None:
                values["market_probability_a"] = (
                    market.market_probability_a_devig
                )
                values["market_probability_b"] = (
                    market.market_probability_b_devig
                )
                values["market_overround"] = market.overround
                values["market_margin"] = market.margin
            vector_rows.append(values)
            vector_indexes.append(row_index)
            vector_values[row_index] = values
            market_statuses[row_index] = market_status
        model_frame = pd.DataFrame(vector_rows, index=vector_indexes)
        predictions = model.predict(model_frame)
        if not predictions.index.equals(model_frame.index):
            raise DailyPredictionError(
                f"El modelo {gender} alteró el índice de las filas."
            )

        for row_index in vector_indexes:
            raw_probability = float(
                predictions.at[row_index, "model_probability_raw_a"]
            )
            probability_a = float(
                predictions.at[row_index, "model_probability_a"]
            )
            if (
                not math.isfinite(raw_probability)
                or not math.isfinite(probability_a)
                or not 0.0 <= raw_probability <= 1.0
                or not 0.0 <= probability_a <= 1.0
            ):
                raise DailyPredictionError(
                    f"El modelo {gender} devolvió una probabilidad inválida."
                )
            values = vector_values[row_index]
            market_a_value = values["market_probability_a"]
            market_b_value = values["market_probability_b"]
            market_a = (
                None
                if market_a_value is None
                else float(market_a_value)
            )
            market_b = (
                None
                if market_b_value is None
                else float(market_b_value)
            )
            output.at[row_index, "canonical_tour_level"] = values[
                "tour_level"
            ]
            output.at[row_index, "model_probability_raw_a"] = raw_probability
            output.at[row_index, "model_probability_a"] = probability_a
            output.at[row_index, "model_probability_b"] = 1.0 - probability_a
            market_status = market_statuses[row_index]
            output.at[
                row_index,
                "market_comparison_status",
            ] = market_status
            if market_a is not None and market_b is not None:
                output.at[row_index, "market_probability_a"] = market_a
                output.at[row_index, "market_probability_b"] = market_b
                if market_status == "strictly_pre_date":
                    output.at[row_index, "edge_a"] = (
                        probability_a - market_a
                    )
                    output.at[row_index, "edge_b"] = (
                        (1.0 - probability_a) - market_b
                    )
            if probability_a >= 0.5:
                output.at[row_index, "predicted_winner_name"] = output.at[
                    row_index, "player_a_name"
                ]
                output.at[row_index, "predicted_winner_slug"] = output.at[
                    row_index, "player_a_slug"
                ]
                winner_probability = probability_a
            else:
                output.at[row_index, "predicted_winner_name"] = output.at[
                    row_index, "player_b_name"
                ]
                output.at[row_index, "predicted_winner_slug"] = output.at[
                    row_index, "player_b_slug"
                ]
                winner_probability = 1.0 - probability_a
            output.at[
                row_index,
                "predicted_winner_probability",
            ] = winner_probability
            output.at[row_index, "best_of"] = values.get("best_of")
            output.at[row_index, "round"] = values.get("round")
            for column in _VECTOR_AUDIT_COLUMNS:
                output.at[row_index, column] = values.get(column)
            output.at[row_index, "prediction_status"] = "predicted"
            output.at[row_index, "feature_history_max_date"] = pd.Timestamp(
                gender_context.training_metadata.max_date
            )
            output.at[row_index, "ranking_source_max_date"] = pd.Timestamp(
                gender_context.ranking_max_date
            )
            output.at[row_index, "feature_fingerprint"] = (
                feature_context.feature_fingerprint
            )
            assessment = assess_vector_confidence(
                values,
                match_date=match_date,
                history_max_date=gender_context.training_metadata.max_date,
                ranking_max_date=gender_context.ranking_max_date,
            )
            snapshot_age = float(
                output.at[row_index, "source_snapshot_age_minutes"]
            )
            assessment = _extend_assessment(
                assessment,
                low_flags=(
                    (
                        f"source_snapshot_stale_{round(snapshot_age)}m"
                        if snapshot_age > MAX_DAILY_SNAPSHOT_AGE_MINUTES
                        else ""
                    ),
                ),
                warning_flags=(
                    (
                        "market_prestart_unverified"
                        if market_status == "prestart_unverified"
                        else ""
                    ),
                ),
            )
            _write_assessment(output, row_index, assessment)
    return output.loc[:, list(PREDICTION_OUTPUT_COLUMNS)]


def publish_predictions_csv(
    predictions: pd.DataFrame,
    *,
    match_date: date,
    prediction_as_of_utc: datetime,
    output_dir: Path = PREDICTIONS_PROCESSED_DIR,
) -> Path:
    """Publica un CSV timestamped mediante escritura temporal y reemplazo."""

    if tuple(predictions.columns) != PREDICTION_OUTPUT_COLUMNS:
        raise DailyPredictionError(
            "El DataFrame no cumple PREDICTION_OUTPUT_COLUMNS."
        )
    resolved_output = Path(output_dir).resolve()
    if not resolved_output.is_relative_to(PROJECT_ROOT.resolve()):
        raise DailyPredictionError("output_dir debe permanecer dentro de TENNIS/.")
    if (
        prediction_as_of_utc.tzinfo is None
        or prediction_as_of_utc.utcoffset() is None
    ):
        raise DailyPredictionError(
            "prediction_as_of_utc debe incluir zona horaria."
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    timestamp = prediction_as_of_utc.astimezone(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    destination = resolved_output / (
        f"predictions_{match_date.isoformat()}_{timestamp}.csv"
    )
    if destination.exists():
        raise DailyPredictionError(
            f"Ya existe una publicación con el mismo timestamp: {destination}."
        )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=resolved_output,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            predictions.to_csv(
                stream,
                index=False,
                na_rep="",
                float_format="%.8f",
                date_format="%Y-%m-%dT%H:%M:%S%z",
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DailyPredictionError(
            f"No se pudo publicar el CSV diario en {resolved_output}."
        ) from exc
    return destination


def _load_active_quarantine_keys() -> frozenset[tuple[str, int]]:
    """Carga la cuarentena vinculada al manifiesto causal de features."""

    source_manifest = load_feature_source_manifest(
        FEATURE_DATASET_MANIFEST_PATH
    )
    quarantine = load_identity_quarantine(
        expected_source_commit=source_manifest.source_commit,
        csv_path=IDENTITY_QUARANTINE_PATH,
        manifest_path=IDENTITY_QUARANTINE_MANIFEST_PATH,
    )
    return quarantine.keys


def run_daily_prediction_pipeline(
    match_date: date | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
    output_dir: Path = PREDICTIONS_PROCESSED_DIR,
    publish: bool = True,
) -> DailyPredictionRun:
    """Ejecuta las fases 4–7 y devuelve la cartelera completa auditada."""

    observed = _read_clock(clock)
    selected_date = match_date or observed.astimezone().date()
    if isinstance(selected_date, datetime) or not isinstance(selected_date, date):
        raise DailyPredictionError("match_date debe ser datetime.date o None.")

    scraped = get_daily_matches(match_date=selected_date)
    prediction_as_of_utc = _read_clock(clock)
    mapped = resolve_scraped_matches(
        scraped,
        as_of_date=selected_date,
    )
    _validate_mapped_frame(mapped, selected_date)
    _validate_source_match_ids(mapped)
    initially_mappable = (
        mapped["status"].eq("scheduled")
        & mapped["mapping_status"].eq("mapped")
    )
    quarantine = (
        _validated_excluded_player_keys(_load_active_quarantine_keys())
        if bool(initially_mappable.any())
        else frozenset()
    )
    quarantined_rows = pd.Series(
        (
            _row_is_quarantined(row, quarantine)
            for _, row in mapped.iterrows()
        ),
        index=mapped.index,
        dtype=bool,
    )
    eligible_identity = initially_mappable & ~quarantined_rows

    candidate_genders: set[Gender] = {
        cast(Gender, gender)
        for gender in mapped.loc[
            eligible_identity,
            "gender",
        ].astype(str)
    }
    models = {
        gender: load_active_deployment_model(gender)
        for gender in sorted(candidate_genders)
    }
    causal_genders = {
        gender
        for gender, model in models.items()
        if _model_training_date(model) < selected_date
    }
    causal_indexes = mapped.index[
        eligible_identity
        & mapped["gender"].isin(causal_genders)
    ].tolist()
    context = None
    if causal_indexes:
        targets = _target_ids(mapped, causal_indexes)
        causal_models = {
            gender: models[gender]
            for gender in causal_genders
            if gender in targets
        }
        context = build_daily_feature_context(
            as_of_date=selected_date,
            player_ids_by_gender=targets,
            models=causal_models,
        )
    predictions = predict_mapped_matches(
        mapped,
        match_date=selected_date,
        prediction_as_of_utc=prediction_as_of_utc,
        models=models,
        feature_context=context,
        excluded_player_keys=quarantine,
    )
    output_path = (
        publish_predictions_csv(
            predictions,
            match_date=selected_date,
            prediction_as_of_utc=prediction_as_of_utc,
            output_dir=output_dir,
        )
        if publish
        else None
    )
    return DailyPredictionRun(
        match_date=selected_date,
        prediction_as_of_utc=prediction_as_of_utc,
        predictions=predictions,
        output_path=output_path,
    )

"""Orquestador diario de scraping, mapping, features, modelo y publicación.

La orientación es estable y observable: jugador A es siempre ``player_1`` de
Tennis Explorer y B es ``player_2``. Solo se predicen filas cuyo estado sea
``scheduled`` en el snapshot, la inferencia preceda estrictamente al inicio UTC
publicado, ambos jugadores estén mapeados y el modelo haya sido entrenado
exclusivamente con fechas anteriores al partido. El resto se conserva con
probabilidades nulas y razones explícitas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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
    OPERATIONS_DATABASE_PATH,
    PREDICTIONS_PROCESSED_DIR,
    PROJECT_ROOT,
    SURFACE_CATALOG_PATH,
)
from ..elo import Gender
from ..features import (
    MatchFeatureRequest,
    calculate_two_way_market_probabilities,
)
from ..features.match_format import CONTRACT_VERSION as MATCH_FORMAT_CONTRACT
from ..features.match_format import resolve_match_format
from ..freshness import source_threshold_hours
from ..identity_integrity import load_identity_quarantine
from ..modeling.data import load_feature_source_manifest
from ..modeling.service import (
    LoadedDeploymentModel,
    load_active_deployment_model,
)
from ..player_mapping import resolve_scraped_matches
from ..surface_catalog import (
    SurfaceCatalog,
    SurfaceSource,
    build_surface_catalog,
)
from ..tennis_explorer import get_daily_matches
from ..tennisratio import DEFAULT_DB_PATH as TENNISRATIO_DATABASE_PATH
from ..tennisratio import (
    load_active_agenda,
    load_agenda_as_of,
    load_identity_resolutions,
)
from ..tennisratio.agenda_context import restore_agenda_context
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
    "source_family",
    "data_freshness_status",
    "data_freshness_age_hours",
    "data_freshness_cause",
    "fallback_source_used",
    "feature_agenda_retrieved_at_utc",
    "feature_agenda_snapshot_sha256",
    "feature_agenda_source_url",
    "feature_surface",
    "feature_surface_resolution",
    "feature_surface_resolution_reason",
    "feature_surface_captured_at_utc",
    "feature_surface_evidence_id",
    "feature_surface_source_url",
    "feature_surface_source_sha256",
    "surface_catalog_fingerprint",
    "feature_tour_level",
    "feature_odds_a",
    "feature_odds_b",
    "source_match_id",
    "match_detail_href",
    "tournament",
    "tour_level",
    "canonical_tour_level",
    "gender",
    "surface",
    "scheduled_time",
    "scheduled_start_utc",
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
    "best_of_reason",
    "round_reason",
    "match_format_contract",
    "confidence_without_match_format",
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
    "model_training_available_max_date",
    "feature_history_max_date",
    "feature_history_available_max_date",
    "ranking_source_max_date",
    "supplemental_result_rows",
    "supplemental_ranking_rows",
    "overlay_fingerprint",
    "feature_fingerprint",
)
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


def _load_daily_agenda(
    match_date: date,
    *,
    observed_local_date: date,
) -> pd.DataFrame:
    """Load current presentation rows plus a separately frozen feature view."""

    catalog = build_surface_catalog(
        as_of_date=match_date,
        tennisratio_database_path=TENNISRATIO_DATABASE_PATH,
        operations_database_path=OPERATIONS_DATABASE_PATH,
        cache_path=SURFACE_CATALOG_PATH,
    )
    if TENNISRATIO_DATABASE_PATH.is_file():
        agenda = load_active_agenda(
            match_date,
            database_path=TENNISRATIO_DATABASE_PATH,
        )
        if not agenda.empty or match_date == observed_local_date:
            causal_agenda = load_agenda_as_of(
                match_date,
                match_date=match_date,
                database_path=TENNISRATIO_DATABASE_PATH,
            )
            causal_agenda = restore_agenda_context(causal_agenda, TENNISRATIO_DATABASE_PATH)
            return _attach_surface_catalog(
                _attach_feature_agenda(agenda, causal_agenda),
                catalog=catalog,
                match_date=match_date,
            )
    legacy = get_daily_matches(match_date=match_date)
    return _attach_surface_catalog(
        _attach_feature_agenda(legacy, legacy),
        catalog=catalog,
        match_date=match_date,
    )


def _attach_feature_agenda(
    presentation: pd.DataFrame,
    causal: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only an exact pre-date observation as model metadata.

    Current status and odds remain available for presentation and operational
    gating.  Model inputs live in distinct ``feature_*`` columns so a capture
    from the match date cannot masquerade as yesterday's evidence.
    """

    result = presentation.copy()
    result["feature_agenda_retrieved_at_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )
    for column in (
        "feature_agenda_snapshot_sha256",
        "feature_agenda_source_url",
        "feature_surface",
        "feature_tour_level",
        "feature_tournament",
        "feature_tournament_level_source",
        "feature_round_source",
        "feature_round_evidence",
    ):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    for column in ("feature_player_1_odds", "feature_player_2_odds"):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    if result.empty or causal.empty:
        return result
    if causal["source_match_id"].astype("string").duplicated().any():
        raise DailyPredictionError("La agenda causal contiene source_match_id duplicados.")
    causal_lookup = {str(row["source_match_id"]): row for _, row in causal.iterrows()}
    for row_index, current in result.iterrows():
        frozen = causal_lookup.get(str(current["source_match_id"]))
        if frozen is None:
            continue
        identity_columns = ("gender", "player_1_slug", "player_2_slug")
        if any(
            _optional_text(current[column]) != _optional_text(frozen[column])
            for column in identity_columns
        ):
            continue
        result.at[row_index, "feature_agenda_retrieved_at_utc"] = pd.Timestamp(
            frozen["retrieved_at_utc"]
        )
        result.at[row_index, "feature_agenda_snapshot_sha256"] = frozen["snapshot_sha256"]
        result.at[row_index, "feature_agenda_source_url"] = (
            frozen["source_url"] if "source_url" in frozen.index else pd.NA
        )
        result.at[row_index, "feature_surface"] = frozen["surface"]
        result.at[row_index, "feature_tour_level"] = frozen["tour_level"]
        result.at[row_index, "feature_tournament"] = frozen["tournament"]
        for field in ("tournament_level_source", "round_source", "round_evidence"):
            result.at[row_index, f"feature_{field}"] = frozen.get(field, pd.NA)
        result.at[row_index, "feature_player_1_odds"] = frozen["player_1_odds"]
        result.at[row_index, "feature_player_2_odds"] = frozen["player_2_odds"]
    return result


def _attach_surface_catalog(
    agenda: pd.DataFrame,
    *,
    catalog: SurfaceCatalog,
    match_date: date,
) -> pd.DataFrame:
    """Resolve surface only from direct pre-D or exact edition evidence."""

    result = agenda.copy()
    result["feature_surface_captured_at_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )
    for column in (
        "feature_surface_resolution",
        "feature_surface_resolution_reason",
        "feature_surface_evidence_id",
        "feature_surface_source_url",
        "feature_surface_source_sha256",
        "surface_catalog_fingerprint",
    ):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    if result.empty:
        return result
    for row_index, row in result.iterrows():
        source_match_id = str(row["source_match_id"])
        source_family: SurfaceSource = (
            "tennisratio" if source_match_id.startswith("tennisratio:") else "tennis_explorer"
        )
        direct_captured = _optional_feature_agenda_retrieved_at(row)
        resolution = catalog.resolve_for_prediction(
            source_family=source_family,
            source_match_id=source_match_id,
            gender=cast(Gender, str(row["gender"])),
            match_date=match_date,
            tournament=str(row["tournament"]),
            tournament_href=(
                _optional_text(row["tournament_href"]) if "tournament_href" in row.index else None
            ),
            direct_surface=_feature_agenda_value(
                row,
                "feature_surface",
                "surface",
            ),
            direct_captured_at_utc=direct_captured,
            direct_source_url=(
                _optional_text(row["feature_agenda_source_url"])
                if "feature_agenda_source_url" in row.index
                else None
            ),
            direct_source_sha256=(
                _optional_text(row["feature_agenda_snapshot_sha256"])
                if "feature_agenda_snapshot_sha256" in row.index
                else None
            ),
            direct_tour_level=_optional_text(
                _feature_agenda_value(
                    row,
                    "feature_tour_level",
                    "tour_level",
                )
            ),
        )
        result.at[row_index, "feature_surface"] = pd.NA
        result.at[row_index, "surface_catalog_fingerprint"] = catalog.fingerprint
        result.at[row_index, "feature_surface_resolution_reason"] = resolution.reason
        if not resolution.resolved:
            continue
        result.at[row_index, "feature_surface"] = resolution.surface
        result.at[row_index, "feature_surface_resolution"] = resolution.method
        result.at[row_index, "feature_surface_captured_at_utc"] = pd.Timestamp(
            resolution.captured_at_utc
        )
        result.at[row_index, "feature_surface_evidence_id"] = resolution.evidence_id
        result.at[row_index, "feature_surface_source_url"] = resolution.source_url
        result.at[row_index, "feature_surface_source_sha256"] = resolution.source_sha256
        if (
            _optional_text(result.at[row_index, "feature_tour_level"]) is None
            and resolution.tour_level is not None
        ):
            result.at[row_index, "feature_tour_level"] = resolution.tour_level
    return result


def _apply_tennisratio_agenda_identities(
    mapped_matches: pd.DataFrame,
    *,
    as_of_date: date,
) -> pd.DataFrame:
    """Use exact sidecar identities for namespaced agenda participants.

    Identity unlocks historical Elo/ranking/history and is therefore queried
    with the same strict ``< D`` availability rule as every model input.  A
    disagreement or duplicate resolution degrades the row instead of choosing
    one ID silently.
    """

    if mapped_matches.empty or not TENNISRATIO_DATABASE_PATH.is_file():
        return mapped_matches
    ratio_rows = (
        mapped_matches["source_match_id"]
        .astype("string")
        .str.startswith(
            "tennisratio:",
            na=False,
        )
    )
    if not bool(ratio_rows.any()):
        return mapped_matches
    identities = load_identity_resolutions(
        as_of_date=as_of_date,
        database_path=TENNISRATIO_DATABASE_PATH,
    )
    lookup: dict[tuple[str, str], int] = {}
    conflicting_keys: set[tuple[str, str]] = set()
    for identity in identities:
        key = (identity.gender, identity.source_player_key)
        previous = lookup.get(key)
        if previous is not None and previous != identity.sackmann_player_id:
            conflicting_keys.add(key)
            lookup.pop(key, None)
            continue
        if key not in conflicting_keys:
            lookup[key] = identity.sackmann_player_id
    result = mapped_matches.copy()
    for row_index in result.index[ratio_rows]:
        gender = str(result.at[row_index, "gender"])
        conflict = False
        for side in (1, 2):
            slug = _optional_text(result.at[row_index, f"player_{side}_slug"])
            identity_key = None if slug is None else (gender, slug)
            if identity_key in conflicting_keys:
                conflict = True
                break
            source_id = None if identity_key is None else lookup.get(identity_key)
            if source_id is None:
                continue
            existing = result.at[row_index, f"player_{side}_id"]
            if not pd.isna(existing) and int(existing) != source_id:
                conflict = True
                break
            result.at[row_index, f"player_{side}_id"] = source_id
            result.at[
                row_index,
                f"player_{side}_mapping_method",
            ] = "tennisratio_exact_identity"
        if conflict:
            for side in (1, 2):
                result.at[row_index, f"player_{side}_id"] = pd.NA
                result.at[
                    row_index,
                    f"player_{side}_mapping_method",
                ] = "tennisratio_identity_conflict"
            result.at[row_index, "mapping_status"] = "unmapped"
            continue
        both_mapped = all(not pd.isna(result.at[row_index, f"player_{side}_id"]) for side in (1, 2))
        if both_mapped and int(result.at[row_index, "player_1_id"]) == int(
            result.at[row_index, "player_2_id"]
        ):
            for side in (1, 2):
                result.at[row_index, f"player_{side}_id"] = pd.NA
                result.at[
                    row_index,
                    f"player_{side}_mapping_method",
                ] = "tennisratio_identity_conflict"
            both_mapped = False
        result.at[row_index, "mapping_status"] = "mapped" if both_mapped else "unmapped"
    result["player_1_id"] = result["player_1_id"].astype("Int64")
    result["player_2_id"] = result["player_2_id"].astype("Int64")
    result["player_1_mapping_method"] = result["player_1_mapping_method"].astype("string")
    result["player_2_mapping_method"] = result["player_2_mapping_method"].astype("string")
    result["mapping_status"] = result["mapping_status"].astype("string")
    return result


def _resolve_daily_identities(
    agenda: pd.DataFrame,
    *,
    as_of_date: date,
) -> pd.DataFrame:
    """Route each acquisition source through its explicit identity contract."""

    if agenda.empty:
        return resolve_scraped_matches(agenda, as_of_date=as_of_date)
    ratio_rows = (
        agenda["source_match_id"]
        .astype("string")
        .str.startswith(
            "tennisratio:",
            na=False,
        )
    )
    if not bool(ratio_rows.any()):
        return resolve_scraped_matches(agenda, as_of_date=as_of_date)

    frames: list[pd.DataFrame] = []
    legacy_agenda = agenda.loc[~ratio_rows].copy()
    if not legacy_agenda.empty:
        frames.append(
            resolve_scraped_matches(
                legacy_agenda,
                as_of_date=as_of_date,
            )
        )
    ratio_agenda = agenda.loc[ratio_rows].copy()
    ratio_agenda["player_1_id"] = pd.Series(
        pd.NA,
        index=ratio_agenda.index,
        dtype="Int64",
    )
    ratio_agenda["player_2_id"] = pd.Series(
        pd.NA,
        index=ratio_agenda.index,
        dtype="Int64",
    )
    ratio_agenda["player_1_mapping_method"] = pd.Series(
        pd.NA,
        index=ratio_agenda.index,
        dtype="string",
    )
    ratio_agenda["player_2_mapping_method"] = pd.Series(
        pd.NA,
        index=ratio_agenda.index,
        dtype="string",
    )
    ratio_agenda["mapping_status"] = pd.Series(
        "unmapped",
        index=ratio_agenda.index,
        dtype="string",
    )
    frames.append(
        _apply_tennisratio_agenda_identities(
            ratio_agenda,
            as_of_date=as_of_date,
        )
    )
    combined = pd.concat(frames, axis=0, sort=False)
    return combined.loc[agenda.index].copy()


def _prediction_timestamp_after_snapshot(
    observed: datetime,
    scraped: pd.DataFrame,
) -> datetime:
    """Garantiza orden lógico estricto pese a la resolución del reloj.

    La captura ya terminó antes de que esta función sea llamada, pero Windows
    puede devolver el mismo tick para ambos eventos. En ese único caso se usa
    el microtick siguiente a la evidencia fuente; nunca se mueve una captura
    hacia atrás ni se relaja la validación estricta de la BBDD.
    """

    if scraped.empty or "retrieved_at_utc" not in scraped.columns:
        return observed
    retrieved = pd.to_datetime(
        scraped["retrieved_at_utc"],
        errors="raise",
        utc=True,
    )
    if retrieved.isna().any():
        raise DailyPredictionError("retrieved_at_utc no puede ser nulo en un snapshot poblado.")
    latest_source = retrieved.max()
    observed_timestamp = pd.Timestamp(observed)
    if observed_timestamp > latest_source:
        return observed
    logical_timestamp = latest_source + timedelta(microseconds=1)
    return logical_timestamp.to_pydatetime().astimezone(UTC)


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
        result = float(str(value))
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


def _feature_agenda_value(
    row: pd.Series,
    feature_column: str,
    source_column: str,
) -> object:
    """Return the frozen pre-date value, falling back for legacy sources."""

    if feature_column in row.index:
        return row[feature_column]
    return row[source_column]


def _feature_agenda_retrieved_at(row: pd.Series) -> datetime:
    """Return the availability instant of metadata offered to the model."""

    retrieved = _optional_feature_agenda_retrieved_at(row)
    if retrieved is None:
        raise DailyPredictionError("feature_agenda_retrieved_at_utc no puede ser nulo.")
    return retrieved


def _optional_feature_agenda_retrieved_at(row: pd.Series) -> datetime | None:
    """Return frozen agenda availability, or ``None`` when no pre-D row exists."""

    value = _feature_agenda_value(
        row,
        "feature_agenda_retrieved_at_utc",
        "retrieved_at_utc",
    )
    if value is None or bool(pd.isna(value)):
        return None
    return _aware_datetime(value, "feature_agenda_retrieved_at_utc")


def _operational_start_failure(
    row: pd.Series,
    *,
    match_date: date,
    prediction_as_of_utc: datetime,
) -> str | None:
    """Return the fail-closed reason for the current published UTC start."""

    scheduled_value = row["scheduled_start_utc"] if "scheduled_start_utc" in row.index else None
    if scheduled_value is None or bool(pd.isna(scheduled_value)):
        return "scheduled_start_utc_missing"
    try:
        scheduled_start = _aware_datetime(
            scheduled_value,
            "scheduled_start_utc",
        )
    except DailyPredictionError:
        return "scheduled_start_utc_invalid"
    if scheduled_start.date() != match_date:
        return "scheduled_start_date_conflict"
    if prediction_as_of_utc.astimezone(UTC) >= scheduled_start:
        return "prediction_not_strictly_before_scheduled_start"
    return None


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
        raise DailyPredictionError(f"La cartelera mapeada carece de columnas: {missing}.")
    if not mapped_matches.index.is_unique:
        raise DailyPredictionError("El índice de la cartelera debe ser único.")
    if mapped_matches.empty:
        return
    dates = pd.to_datetime(
        mapped_matches["match_date"],
        errors="raise",
    ).dt.date
    if not bool((dates == match_date).all()):
        raise DailyPredictionError("La cartelera mezcla fechas o contradice la fecha solicitada.")
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
        raise DailyPredictionError("source_match_id debe ser único dentro del snapshot diario.")


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
    snapshot_age = (pd.Timestamp(prediction_as_of_utc) - retrieved).dt.total_seconds() / 60.0
    if (snapshot_age < 0.0).any():
        raise DailyPredictionError("El snapshot diario no puede ser posterior a la predicción.")
    result["source_snapshot_age_minutes"] = snapshot_age.astype(float)
    source_match_ids = mapped_matches["source_match_id"].astype("string")
    source_family = source_match_ids.map(
        lambda value: "tennisratio" if str(value).startswith("tennisratio:") else "tennis_explorer"
    ).astype("string")
    result["source_family"] = source_family
    result["data_freshness_age_hours"] = (snapshot_age / 60.0).astype(float)
    thresholds = source_family.map(lambda value: source_threshold_hours(str(value))).astype(float)
    stale = result["data_freshness_age_hours"].gt(thresholds)
    fallback = source_family.eq("tennis_explorer")
    result["data_freshness_status"] = pd.Series(
        np.where(stale, "stale", "fresh"),
        index=result.index,
        dtype="string",
    )
    result["data_freshness_cause"] = pd.Series(
        np.where(
            fallback,
            "fallback_used",
            np.where(stale, "source_not_refreshed", "fresh"),
        ),
        index=result.index,
        dtype="string",
    )
    result["fallback_source_used"] = fallback.astype(bool)
    result["feature_agenda_retrieved_at_utc"] = (
        mapped_matches["feature_agenda_retrieved_at_utc"]
        if "feature_agenda_retrieved_at_utc" in mapped_matches
        else mapped_matches["retrieved_at_utc"]
    )
    result["feature_agenda_snapshot_sha256"] = (
        mapped_matches["feature_agenda_snapshot_sha256"]
        if "feature_agenda_snapshot_sha256" in mapped_matches
        else mapped_matches["snapshot_sha256"]
    )
    result["feature_surface"] = (
        mapped_matches["feature_surface"]
        if "feature_surface" in mapped_matches
        else mapped_matches["surface"]
    )
    result["feature_surface_resolution"] = (
        mapped_matches["feature_surface_resolution"]
        if "feature_surface_resolution" in mapped_matches
        else pd.Series(pd.NA, index=mapped_matches.index, dtype="string")
    )
    result["feature_surface_resolution_reason"] = (
        mapped_matches["feature_surface_resolution_reason"]
        if "feature_surface_resolution_reason" in mapped_matches
        else pd.Series(pd.NA, index=mapped_matches.index, dtype="string")
    )
    result["feature_surface_captured_at_utc"] = (
        mapped_matches["feature_surface_captured_at_utc"]
        if "feature_surface_captured_at_utc" in mapped_matches
        else pd.Series(
            pd.NaT,
            index=mapped_matches.index,
            dtype="datetime64[ns, UTC]",
        )
    )
    for column in (
        "feature_surface_evidence_id",
        "feature_surface_source_url",
        "feature_surface_source_sha256",
        "surface_catalog_fingerprint",
    ):
        result[column] = (
            mapped_matches[column]
            if column in mapped_matches
            else pd.Series(pd.NA, index=mapped_matches.index, dtype="string")
        )
    result["feature_tour_level"] = (
        mapped_matches["feature_tour_level"]
        if "feature_tour_level" in mapped_matches
        else mapped_matches["tour_level"]
    )
    result["feature_odds_a"] = (
        mapped_matches["feature_player_1_odds"]
        if "feature_player_1_odds" in mapped_matches
        else mapped_matches["player_1_odds"]
    )
    result["feature_odds_b"] = (
        mapped_matches["feature_player_2_odds"]
        if "feature_player_2_odds" in mapped_matches
        else mapped_matches["player_2_odds"]
    )
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
    result["scheduled_start_utc"] = (
        mapped_matches["scheduled_start_utc"]
        if "scheduled_start_utc" in mapped_matches
        else pd.Series(
            pd.NaT,
            index=mapped_matches.index,
            dtype="datetime64[ns, UTC]",
        )
    )
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
    for column in (
        "best_of_reason",
        "round_reason",
        "match_format_contract",
        "confidence_without_match_format",
        "feature_agenda_source_url",
    ):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
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
    result["model_training_available_max_date"] = pd.NaT
    result["feature_history_max_date"] = pd.NaT
    result["feature_history_available_max_date"] = pd.NaT
    result["ranking_source_max_date"] = pd.NaT
    result["supplemental_result_rows"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result["supplemental_ranking_rows"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result["overlay_fingerprint"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
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
            raise DailyPredictionError("excluded_player_keys contiene una identidad inválida.")
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
    match_date: date,
    prediction_as_of_utc: datetime,
    excluded_player_keys: frozenset[tuple[Gender, int]],
) -> ConfidenceAssessment | None:
    """Devuelve una razón fatal previa a features, o ``None`` si es elegible.

    La agenda congelada pre-``D`` aporta features, mientras que el instante de
    inicio procede de la agenda operativa actual. La inferencia falla cerrado
    cuando la fuente no publica ese instante UTC o cuando ya se ha alcanzado.
    """

    flags: list[str] = []
    status = _optional_text(row["status"])
    if status != "scheduled":
        flags.append(f"status_{status or 'missing'}")
    if _optional_text(row["mapping_status"]) != "mapped":
        flags.append("player_unmapped")
    elif _row_is_quarantined(row, excluded_player_keys):
        flags.append("identity_quarantined")
    start_failure = _operational_start_failure(
        row,
        match_date=match_date,
        prediction_as_of_utc=prediction_as_of_utc,
    )
    if start_failure is not None:
        flags.append(start_failure)
    try:
        feature_retrieved = _optional_feature_agenda_retrieved_at(row)
    except DailyPredictionError:
        flags.append("feature_agenda_availability_invalid")
    else:
        if feature_retrieved is not None and feature_retrieved.date() >= match_date:
            flags.append("feature_agenda_not_strictly_before_match_date")
    feature_tour_level = _optional_text(
        _feature_agenda_value(
            row,
            "feature_tour_level",
            "tour_level",
        )
    )
    if feature_tour_level is not None and str(row["source_match_id"]).startswith("tennisratio:"):
        gender = _optional_text(row["gender"])
        if (gender == "M" and feature_tour_level == "WTA") or (
            gender == "F" and feature_tour_level in {"ATP", "Challenger"}
        ):
            flags.append("feature_tour_gender_conflict")
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

    cleaned_low_flags = tuple(flag for flag in low_flags if flag)
    cleaned_warning_flags = tuple(flag for flag in warning_flags if flag)
    flags = tuple(
        dict.fromkeys(
            (
                *assessment.flags,
                *cleaned_low_flags,
                *cleaned_warning_flags,
            )
        )
    )
    if cleaned_low_flags:
        level = "LOW"
    elif cleaned_warning_flags and assessment.level == "HIGH":
        level = "MEDIUM"
    else:
        level = assessment.level
    return ConfidenceAssessment(level=level, flags=flags)


def _model_training_date(model: LoadedDeploymentModel) -> date:
    """Parsea el corte de entrenamiento declarado por un bundle verificado."""

    try:
        return date.fromisoformat(model.training_max_date)
    except ValueError as exc:
        raise DailyPredictionError(f"training_max_date inválida para {model.gender}.") from exc


def _model_training_available_date(model: LoadedDeploymentModel) -> date:
    """Parsea el último resultado causalmente disponible del bundle."""

    try:
        return date.fromisoformat(model.training_available_max_date)
    except ValueError as exc:
        raise DailyPredictionError(
            f"training_available_max_date inválida para {model.gender}."
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
        targets.setdefault(gender, set()).update((int(row["player_1_id"]), int(row["player_2_id"])))
    return targets


def predict_mapped_matches(
    mapped_matches: pd.DataFrame,
    *,
    match_date: date,
    prediction_as_of_utc: datetime,
    models: Mapping[Gender, LoadedDeploymentModel],
    feature_context: DailyFeatureContext | None,
    excluded_player_keys: (frozenset[tuple[str, int]] | set[tuple[str, int]]) = frozenset(),
) -> pd.DataFrame:
    """Genera probabilidades solo para filas prospectivas y causales."""

    _validate_mapped_frame(mapped_matches, match_date)
    _validate_source_match_ids(mapped_matches)
    quarantine = _validated_excluded_player_keys(excluded_player_keys)
    if prediction_as_of_utc.tzinfo is None or prediction_as_of_utc.utcoffset() is None:
        raise DailyPredictionError("prediction_as_of_utc debe incluir zona horaria.")
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
            match_date=match_date,
            prediction_as_of_utc=prediction_as_of_utc,
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
        training_date = _model_training_date(model)
        training_available_date = _model_training_available_date(model)
        output.at[row_index, "model_training_max_date"] = pd.Timestamp(training_date)
        output.at[
            row_index,
            "model_training_available_max_date",
        ] = pd.Timestamp(training_available_date)
        if training_available_date >= match_date:
            _write_assessment(
                output,
                row_index,
                unavailable_confidence("model_not_causal_for_date"),
            )
            continue
        eligible_by_gender[gender].append(row_index)

    causal_indexes = [row_index for indexes in eligible_by_gender.values() for row_index in indexes]
    if causal_indexes and feature_context is None:
        raise DailyPredictionError("Falta feature_context para partidos causalmente elegibles.")
    if feature_context is None:
        return output.loc[:, list(PREDICTION_OUTPUT_COLUMNS)]

    for gender, row_indexes in eligible_by_gender.items():
        if not row_indexes:
            continue
        gender_context = feature_context.by_gender.get(gender)
        if gender_context is None:
            raise DailyPredictionError(f"Falta contexto causal para el género {gender}.")
        model = models[gender]
        base_history_available_max_date = feature_context.source_date_policy.availability_date(
            gender_context.training_metadata.max_date
        )
        if base_history_available_max_date != _model_training_available_date(model):
            raise DailyPredictionError(
                f"El corte disponible del histórico no coincide con el modelo activo de {gender}."
            )
        history_available_max_date = getattr(
            gender_context,
            "history_available_max_date",
            base_history_available_max_date,
        )
        history_effective_max_date = getattr(
            gender_context,
            "history_effective_max_date",
            gender_context.training_metadata.max_date,
        )
        if history_available_max_date >= match_date:
            raise DailyPredictionError(
                "El historico servido debe estar disponible antes del partido."
            )
        vector_rows: list[dict[str, object]] = []
        vector_indexes: list[object] = []
        vector_values: dict[object, dict[str, object]] = {}
        agenda_context_flags: dict[object, tuple[str, ...]] = {}
        market_statuses: dict[object, str] = {}
        presentation_markets: dict[
            object,
            tuple[float | None, float | None],
        ] = {}
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
            market = calculate_two_way_market_probabilities(odds_a, odds_b) if has_market else None
            market_status = (
                "missing"
                if market is None
                else (
                    "strictly_pre_date"
                    if (market_retrieved is not None and market_retrieved.date() < match_date)
                    else "prestart_unverified"
                )
            )
            feature_odds_a = _optional_float(
                _feature_agenda_value(
                    row,
                    "feature_player_1_odds",
                    "player_1_odds",
                )
            )
            feature_odds_b = _optional_float(
                _feature_agenda_value(
                    row,
                    "feature_player_2_odds",
                    "player_2_odds",
                )
            )
            feature_market = (
                calculate_two_way_market_probabilities(
                    feature_odds_a,
                    feature_odds_b,
                )
                if feature_odds_a is not None and feature_odds_b is not None
                else None
            )
            feature_retrieved = _optional_feature_agenda_retrieved_at(row)
            use_market_as_feature = (
                model.estimator.profile == "market_enhanced"
                and feature_market is not None
                and feature_retrieved is not None
            )
            surface = _optional_text(
                _feature_agenda_value(
                    row,
                    "feature_surface",
                    "surface",
                )
            )
            tour_level = _optional_text(
                _feature_agenda_value(
                    row,
                    "feature_tour_level",
                    "tour_level",
                )
            )
            surface_resolution = _optional_text(row.get("feature_surface_resolution"))
            has_catalog_context = surface_resolution in {
                "catalog_exact_edition",
                "same_edition_propagation",
            }
            if tour_level is None:
                agenda_context_flags[row_index] = (
                    ("feature_tour_level_missing",)
                    if has_catalog_context
                    else (
                        "feature_agenda_availability_missing",
                        "feature_tour_level_missing",
                    )
                )
            elif feature_retrieved is None and not has_catalog_context:
                agenda_context_flags[row_index] = ("feature_agenda_availability_missing",)
            else:
                agenda_context_flags[row_index] = ()
            source_match_id = str(row["source_match_id"])
            source_family = (
                "tennisratio" if source_match_id.startswith("tennisratio:") else "tennis_explorer"
            )
            request_tour_level = tour_level or ("ATP" if gender == "M" else "WTA")
            match_format = resolve_match_format(
                match_date=match_date,
                captured_at_utc=feature_retrieved,
                gender=gender,
                source_family=source_family,
                tournament_level=row.get("feature_tournament_level_source"),
                tournament=row.get("feature_tournament"),
                round_raw=row.get("feature_round_source"),
                round_evidence=row.get("feature_round_evidence"),
            )
            output.at[row_index, "best_of_reason"] = match_format.best_of_reason
            output.at[row_index, "round_reason"] = match_format.round_reason
            output.at[row_index, "match_format_contract"] = MATCH_FORMAT_CONTRACT
            output.at[row_index, "feature_agenda_source_url"] = row.get(
                "feature_agenda_source_url", pd.NA
            )
            request = MatchFeatureRequest(
                gender=gender,
                player_a_id=int(row["player_1_id"]),
                player_b_id=int(row["player_2_id"]),
                as_of_date=match_date,
                surface=surface,
                tour_level_raw=request_tour_level,
                source_family=source_family,
                best_of=match_format.best_of,
                round=match_format.round,
                odds_a=feature_odds_a if use_market_as_feature else None,
                odds_b=feature_odds_b if use_market_as_feature else None,
                market_retrieved_at_utc=(feature_retrieved if use_market_as_feature else None),
                prediction_as_of_utc=(prediction_as_of_utc if use_market_as_feature else None),
            )
            values = gender_context.builder.build(request).to_dict()
            if tour_level is None:
                values["tour_level_raw"] = None
                values["tour_level"] = None
            if feature_market is not None and use_market_as_feature:
                values["market_probability_a"] = feature_market.market_probability_a_devig
                values["market_probability_b"] = feature_market.market_probability_b_devig
                values["market_overround"] = feature_market.overround
                values["market_margin"] = feature_market.margin
            vector_rows.append(values)
            vector_indexes.append(row_index)
            vector_values[row_index] = values
            market_statuses[row_index] = market_status
            presentation_markets[row_index] = (
                (
                    market.market_probability_a_devig,
                    market.market_probability_b_devig,
                )
                if market is not None
                else (None, None)
            )
        model_frame = pd.DataFrame(vector_rows, index=vector_indexes)
        predictions = model.predict(model_frame, as_of_date=match_date)
        if not predictions.index.equals(model_frame.index):
            raise DailyPredictionError(f"El modelo {gender} alteró el índice de las filas.")

        for row_index in vector_indexes:
            raw_probability = float(predictions.at[row_index, "model_probability_raw_a"])
            probability_a = float(predictions.at[row_index, "model_probability_a"])
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
            market_a, market_b = presentation_markets[row_index]
            output.at[row_index, "canonical_tour_level"] = values["tour_level"]
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
                    output.at[row_index, "edge_a"] = probability_a - market_a
                    output.at[row_index, "edge_b"] = (1.0 - probability_a) - market_b
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
                value = values.get(column)
                if column in {
                    "ranking_date_a",
                    "ranking_date_b",
                    "birth_date_a",
                    "birth_date_b",
                }:
                    value = pd.NaT if value is None else pd.Timestamp(value)
                output.at[row_index, column] = value
            output.at[row_index, "prediction_status"] = "predicted"
            output.at[row_index, "feature_history_max_date"] = pd.Timestamp(
                history_effective_max_date
            )
            output.at[
                row_index,
                "feature_history_available_max_date",
            ] = pd.Timestamp(history_available_max_date)
            output.at[row_index, "ranking_source_max_date"] = pd.Timestamp(
                gender_context.ranking_max_date
            )
            output.at[row_index, "supplemental_result_rows"] = getattr(
                gender_context,
                "supplemental_result_rows",
                0,
            )
            output.at[row_index, "supplemental_ranking_rows"] = getattr(
                gender_context,
                "supplemental_ranking_rows",
                0,
            )
            output.at[row_index, "overlay_fingerprint"] = getattr(
                gender_context,
                "overlay_fingerprint",
                None,
            )
            output.at[row_index, "feature_fingerprint"] = feature_context.feature_fingerprint
            confidence_values = dict(values)
            confidence_values["market_probability_a"] = market_a
            confidence_values["market_probability_b"] = market_b
            assessment = assess_vector_confidence(
                confidence_values,
                match_date=match_date,
                history_available_max_date=history_available_max_date,
                ranking_max_date=gender_context.ranking_max_date,
            )
            baseline_assessment = assess_vector_confidence(
                {**confidence_values, "best_of": None, "round": None},
                match_date=match_date,
                history_available_max_date=history_available_max_date,
                ranking_max_date=gender_context.ranking_max_date,
            )
            snapshot_age = float(output.at[row_index, "source_snapshot_age_minutes"])
            source_family = str(output.at[row_index, "source_family"])
            stale_source = str(output.at[row_index, "data_freshness_status"]) == "stale"
            fallback_source = bool(output.at[row_index, "fallback_source_used"])
            assessment = _extend_assessment(
                assessment,
                low_flags=(
                    *agenda_context_flags[row_index],
                    (
                        f"data_stale_{source_family}_{round(snapshot_age / 60.0)}h"
                        if stale_source
                        else ""
                    ),
                    ("agenda_fallback_tennis_explorer" if fallback_source else ""),
                ),
                warning_flags=(
                    (
                        "market_prestart_unverified"
                        if market_status == "prestart_unverified"
                        else ""
                    ),
                ),
            )
            # The same freshness/context flags apply before and after the fix.
            output.at[row_index, "confidence_without_match_format"] = (
                "LOW" if assessment.level == "LOW" else baseline_assessment.level
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
        raise DailyPredictionError("El DataFrame no cumple PREDICTION_OUTPUT_COLUMNS.")
    resolved_output = Path(output_dir).resolve()
    if not resolved_output.is_relative_to(PROJECT_ROOT.resolve()):
        raise DailyPredictionError("output_dir debe permanecer dentro de TENNIS/.")
    if prediction_as_of_utc.tzinfo is None or prediction_as_of_utc.utcoffset() is None:
        raise DailyPredictionError("prediction_as_of_utc debe incluir zona horaria.")
    resolved_output.mkdir(parents=True, exist_ok=True)
    timestamp = prediction_as_of_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = resolved_output / (f"predictions_{match_date.isoformat()}_{timestamp}.csv")
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

    source_manifest = load_feature_source_manifest(FEATURE_DATASET_MANIFEST_PATH)
    quarantine = load_identity_quarantine(
        expected_source_commit=source_manifest.source_commit,
        csv_path=IDENTITY_QUARANTINE_PATH,
        manifest_path=IDENTITY_QUARANTINE_MANIFEST_PATH,
    )
    return quarantine.keys


def _retrieved_strictly_before_date(
    retrieved_at_utc: pd.Series,
    match_date: date,
) -> pd.Series:
    """Apply the daily ``availability < D`` gate with Pandas 3 dtypes.

    ``Series.dt.date`` may preserve a ``datetime64[s]`` extension dtype in
    Pandas 3, which cannot be compared directly with ``datetime.date``.  A UTC
    midnight Timestamp expresses the same civil-date rule without converting
    the vector to Python objects or admitting any observation from day ``D``.
    """

    cutoff_utc = pd.Timestamp(match_date, tz=UTC)
    return retrieved_at_utc.notna() & retrieved_at_utc.lt(cutoff_utc)


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

    scraped = _load_daily_agenda(
        selected_date,
        observed_local_date=observed.astimezone().date(),
    )
    prediction_as_of_utc = _prediction_timestamp_after_snapshot(
        _read_clock(clock),
        scraped,
    )
    mapped = _resolve_daily_identities(
        scraped,
        as_of_date=selected_date,
    )
    _validate_mapped_frame(mapped, selected_date)
    _validate_source_match_ids(mapped)
    feature_retrieved_column = (
        "feature_agenda_retrieved_at_utc"
        if "feature_agenda_retrieved_at_utc" in mapped
        else "retrieved_at_utc"
    )
    feature_retrieved = pd.to_datetime(
        mapped[feature_retrieved_column],
        errors="coerce",
        utc=True,
    )
    causal_agenda = feature_retrieved.isna() | _retrieved_strictly_before_date(
        feature_retrieved,
        selected_date,
    )
    operationally_prestart = pd.Series(
        (
            _operational_start_failure(
                row,
                match_date=selected_date,
                prediction_as_of_utc=prediction_as_of_utc,
            )
            is None
            for _, row in mapped.iterrows()
        ),
        index=mapped.index,
        dtype=bool,
    )
    initially_mappable = (
        mapped["status"].eq("scheduled")
        & mapped["mapping_status"].eq("mapped")
        & causal_agenda
        & operationally_prestart
    )
    quarantine = (
        _validated_excluded_player_keys(_load_active_quarantine_keys())
        if bool(initially_mappable.any())
        else frozenset()
    )
    quarantined_rows = pd.Series(
        (_row_is_quarantined(row, quarantine) for _, row in mapped.iterrows()),
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
    models = {gender: load_active_deployment_model(gender) for gender in sorted(candidate_genders)}
    causal_genders = {
        gender
        for gender, model in models.items()
        if _model_training_available_date(model) < selected_date
    }
    causal_indexes = mapped.index[
        eligible_identity & mapped["gender"].isin(causal_genders)
    ].tolist()
    context = None
    if causal_indexes:
        targets = _target_ids(mapped, causal_indexes)
        causal_models = {gender: models[gender] for gender in causal_genders if gender in targets}
        context = build_daily_feature_context(
            as_of_date=selected_date,
            player_ids_by_gender=targets,
            models=causal_models,
            tennisratio_database_path=TENNISRATIO_DATABASE_PATH,
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

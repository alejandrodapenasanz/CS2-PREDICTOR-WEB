"""Reconstrucción causal del contexto necesario para predecir una cartelera.

El estado de forma, descanso y H2H no se serializó en la fase 6. Este módulo
lo reconstruye desde los Parquet verificados, leyendo únicamente resultados
cuya ``result_available_date`` es estrictamente anterior al corte y que
involucren a los jugadores de la cartelera. También fija explícitamente los
runs Elo y comprueba que modelo, features, rankings y maestros proceden del
mismo snapshot Sackmann.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, cast

import pandas as pd
import pyarrow.parquet as pq

from ..config import (
    ELO_DATABASE_PATH,
    FEATURE_DATASET_MANIFEST_PATH,
    IDENTITY_QUARANTINE_MANIFEST_PATH,
    IDENTITY_QUARANTINE_PATH,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from ..elo import (
    EloStore,
    Gender,
    Surface,
    normalise_surface,
)
from ..features import (
    CausalHistoryState,
    FeatureParameters,
    HistoricalMatchResult,
    MatchFeatureBuilder,
    PlayerAgeIndex,
    RankingIndex,
    verify_auxiliary_source_inventory,
)
from ..identity_integrity import (
    IdentityIntegrityError,
    load_identity_quarantine,
)
from ..modeling.data import (
    FeatureSourceManifest,
    TrainingDatasetMetadata,
    load_feature_source_manifest,
    verify_training_dataset,
)
from ..modeling.service import LoadedDeploymentModel
from ..modeling.promotion import has_equivalent_feature_gate
from ..temporal import SourceDatePolicy, SourceDatePolicyError
from .tennisratio_overlay import (
    TennisRatioOverlayError,
    build_tennisratio_overlay,
)


_HISTORY_COLUMNS = (
    "record_id",
    "gender",
    "match_date",
    "result_available_date",
    "player_a_id",
    "player_b_id",
    "surface",
    "y",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class DailyContextError(RuntimeError):
    """Indica una incompatibilidad o fuga potencial en el contexto diario."""


@dataclass(frozen=True, slots=True)
class GenderFeatureContext:
    """Dependencias causales verificadas para un único universo de género."""

    gender: Gender
    builder: MatchFeatureBuilder
    training_metadata: TrainingDatasetMetadata
    elo_run_id: str
    elo_max_date: date
    overlay_fingerprint: str | None
    history_effective_max_date: date
    history_available_max_date: date
    ranking_max_date: date
    targeted_history_rows: int
    supplemental_result_rows: int
    supplemental_ranking_rows: int


@dataclass(frozen=True, slots=True)
class DailyFeatureContext:
    """Contexto inmutable compartido por todas las filas de una cartelera."""

    feature_fingerprint: str
    source_commit: str
    feature_parameters: FeatureParameters
    source_date_policy: SourceDatePolicy
    identity_quarantine_keys: frozenset[tuple[str, int]]
    by_gender: Mapping[Gender, GenderFeatureContext]

    def __post_init__(self) -> None:
        """Copia el mapping para impedir cambios después de validarlo."""

        object.__setattr__(
            self,
            "by_gender",
            MappingProxyType(dict(self.by_gender)),
        )
        object.__setattr__(
            self,
            "identity_quarantine_keys",
            frozenset(self.identity_quarantine_keys),
        )


def _validate_cutoff(value: object) -> date:
    """Exige una fecha civil estricta para el corte de predicción."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise DailyContextError("as_of_date debe ser datetime.date estricto.")
    return value


def _validated_target_ids(
    player_ids_by_gender: Mapping[str, set[int] | frozenset[int]],
) -> dict[Gender, tuple[int, ...]]:
    """Normaliza IDs objetivo positivos sin mezclar los dos géneros."""

    unknown = sorted(set(player_ids_by_gender).difference({"M", "F"}))
    if unknown:
        raise DailyContextError(f"Géneros objetivo desconocidos: {unknown}.")
    validated: dict[Gender, tuple[int, ...]] = {}
    for raw_gender, values in player_ids_by_gender.items():
        if not isinstance(values, (set, frozenset)):
            raise DailyContextError(
                "Cada colección de player_id debe ser set o frozenset."
            )
        ids = tuple(sorted(values))
        if not ids:
            continue
        if any(
            isinstance(player_id, bool)
            or not isinstance(player_id, int)
            or player_id <= 0
            for player_id in ids
        ):
            raise DailyContextError(
                f"Los player_id de {raw_gender} deben ser enteros positivos."
            )
        validated[cast(Gender, raw_gender)] = ids
    return validated


def _read_source_commit(manifest_path: Path) -> str:
    """Lee y valida el commit del manifiesto Sackmann actualmente activo."""

    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DailyContextError(
            f"No se pudo leer el manifiesto Sackmann {manifest_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise DailyContextError("El manifiesto Sackmann no es un objeto JSON.")
    commit = payload.get("source_commit")
    if not isinstance(commit, str) or _COMMIT_PATTERN.fullmatch(commit) is None:
        raise DailyContextError(
            "El manifiesto Sackmann no contiene un source_commit válido."
        )
    return commit


def _manifest_quarantine_keys(
    source_manifest: FeatureSourceManifest,
) -> frozenset[tuple[str, int]]:
    """Lee las claves de identidad incorporadas al fingerprint de features."""

    raw = source_manifest.raw_payload.get("identity_quarantine")
    if not isinstance(raw, Mapping):
        raise DailyContextError(
            "El manifiesto de features no declara identity_quarantine."
        )
    keys = raw.get("keys")
    rows = raw.get("rows")
    if not isinstance(keys, list):
        raise DailyContextError(
            "identity_quarantine.keys debe ser una lista publicada."
        )
    parsed: set[tuple[str, int]] = set()
    for key in keys:
        if (
            not isinstance(key, list)
            or len(key) != 2
            or key[0] not in {"M", "F"}
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or key[1] <= 0
        ):
            raise DailyContextError(
                "El manifiesto contiene una clave de identidad inválida."
            )
        parsed.add((str(key[0]), int(key[1])))
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows != len(parsed)
        or len(parsed) != len(keys)
    ):
        raise DailyContextError(
            "El recuento de la cuarentena de features no es coherente."
        )
    return frozenset(parsed)


def _required_parameter(
    payload: Mapping[str, object],
    name: str,
    expected_type: type[int] | tuple[type[int], type[float]],
) -> int | float:
    """Extrae un parámetro numérico sin aceptar booleanos ni coerciones."""

    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise DailyContextError(
            f"feature_parameters.{name} no tiene el tipo publicado esperado."
        )
    return cast(int | float, value)


def _feature_parameters(
    source_manifest: FeatureSourceManifest,
) -> FeatureParameters:
    """Reconstruye exactamente la configuración declarada por la fase 6."""

    raw = source_manifest.raw_payload.get("feature_parameters")
    if not isinstance(raw, Mapping):
        raise DailyContextError(
            "El manifiesto de features no declara feature_parameters."
        )
    return FeatureParameters(
        recent_matches=cast(
            int,
            _required_parameter(raw, "recent_matches", int),
        ),
        recent_months=cast(
            int,
            _required_parameter(raw, "recent_months", int),
        ),
        age_reference_years=float(
            _required_parameter(raw, "age_reference_years", (int, float))
        ),
        days_per_year=float(
            _required_parameter(raw, "days_per_year", (int, float))
        ),
        orientation_seed=cast(
            int,
            _required_parameter(raw, "orientation_seed", int),
        ),
    )


def _source_date_policy(
    source_manifest: FeatureSourceManifest,
) -> SourceDatePolicy:
    """Reconstruye el embargo temporal publicado o falla de forma cerrada."""

    raw = source_manifest.raw_payload.get("source_date_policy")
    if not isinstance(raw, Mapping):
        raise DailyContextError(
            "El manifiesto de features no declara source_date_policy; "
            "reconstruya features y modelos."
        )
    try:
        return SourceDatePolicy.from_mapping(raw)
    except SourceDatePolicyError as exc:
        raise DailyContextError(
            "source_date_policy del manifiesto no es compatible."
        ) from exc


def _elo_contract(
    source_manifest: FeatureSourceManifest,
    gender: Gender,
) -> Mapping[str, object]:
    """Obtiene el contrato Elo exacto publicado para un género."""

    contracts = source_manifest.raw_payload.get("elo_contracts")
    if not isinstance(contracts, Mapping):
        raise DailyContextError(
            "El manifiesto no declara elo_contracts; reconstruya features."
        )
    contract = contracts.get(gender)
    if not isinstance(contract, Mapping):
        raise DailyContextError(
            f"Falta el contrato Elo exacto del género {gender}."
        )
    return contract


def _operational_result_cutoff(
    parameters: Mapping[str, object],
    *,
    base_effective_max_date: date,
    source_commit: str,
) -> date:
    """Recover the fixed persisted handoff without reopening its epoch.

    Legacy Sackmann-only runs have no operational contract and retain the
    historical dataset maximum. Once the contract exists, its commit and
    cutoff are validated before the in-memory sidecar can query newer rows.
    """

    raw = parameters.get("operational_overlay")
    if raw is None:
        return base_effective_max_date
    if not isinstance(raw, Mapping):
        raise DailyContextError(
            "El run Elo contiene un contrato operational_overlay inválido."
        )
    if raw.get("schema") not in {
        "tennis-operational-elo-v1",
        "tennis-operational-elo-v2",
    }:
        raise DailyContextError(
            "El run Elo usa una versión desconocida del handoff operativo."
        )
    if raw.get("base_source_commit") != source_commit:
        raise DailyContextError(
            "El handoff operativo no pertenece al commit Sackmann activo."
        )
    cutoff_text = raw.get("cutoff_date")
    if not isinstance(cutoff_text, str):
        raise DailyContextError(
            "El handoff operativo no declara un cutoff_date válido."
        )
    try:
        cutoff = date.fromisoformat(cutoff_text)
    except ValueError as exc:
        raise DailyContextError(
            "El cutoff_date del handoff operativo no es ISO válido."
        ) from exc
    if cutoff < base_effective_max_date:
        raise DailyContextError(
            "El corte operativo precede datos incluidos en la base Sackmann."
        )
    return cutoff


def _read_target_history(
    metadata: TrainingDatasetMetadata,
    *,
    as_of_date: date,
    player_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Lee resultados disponibles ``< D`` sin requerir ``pyarrow.dataset``.

    Windows Application Control puede bloquear el binario opcional
    ``pyarrow.dataset`` aunque el lector Parquet base sea válido. La lectura
    por lotes conserva el mismo predicado causal y evita cargar el histórico
    completo en memoria o duplicar partidos donde ambos jugadores son objetivo.
    """

    target_ids = frozenset(player_ids)
    cutoff_utc = pd.Timestamp(as_of_date, tz="UTC")
    selected_batches: list[pd.DataFrame] = []
    try:
        parquet = pq.ParquetFile(metadata.path)
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=list(_HISTORY_COLUMNS),
        ):
            frame = batch.to_pandas()
            available = pd.to_datetime(
                frame["result_available_date"],
                errors="raise",
                utc=True,
            )
            involved = frame["player_a_id"].isin(target_ids) | frame[
                "player_b_id"
            ].isin(target_ids)
            selected = frame.loc[available.lt(cutoff_utc) & involved]
            if not selected.empty:
                selected_batches.append(selected)
    except Exception as exc:
        raise DailyContextError(
            f"No se pudo leer el histórico dirigido de {metadata.gender}."
        ) from exc
    if not selected_batches:
        return pd.DataFrame(columns=list(_HISTORY_COLUMNS))
    return pd.concat(selected_batches, ignore_index=True, sort=False)


def rebuild_history_state(
    frame: pd.DataFrame,
    *,
    gender: Gender,
    as_of_date: date,
    target_player_ids: set[int] | frozenset[int],
    feature_parameters: FeatureParameters,
    source_date_policy: SourceDatePolicy,
) -> CausalHistoryState:
    """Reconstruye forma/H2H/descanso sin observar la fecha ``D`` ni el futuro.

    Esta función acepta un DataFrame inyectable para que las pruebas puedan
    demostrar que añadir resultados con disponibilidad ``D`` o posterior no
    cambia el estado. La orientación A/B se invierte mediante ``y`` para
    recuperar ganador y perdedor reales.
    """

    cutoff = _validate_cutoff(as_of_date)
    if not isinstance(source_date_policy, SourceDatePolicy):
        raise DailyContextError(
            "source_date_policy debe ser SourceDatePolicy."
        )
    if gender not in {"M", "F"}:
        raise DailyContextError("gender debe ser exactamente 'M' o 'F'.")
    if not isinstance(frame, pd.DataFrame):
        raise DailyContextError("frame debe ser un DataFrame.")
    missing = sorted(set(_HISTORY_COLUMNS).difference(frame.columns))
    if missing:
        raise DailyContextError(
            f"El histórico dirigido carece de columnas: {missing}."
        )
    target_ids = set(target_player_ids)
    if not target_ids or any(
        isinstance(player_id, bool)
        or not isinstance(player_id, int)
        or player_id <= 0
        for player_id in target_ids
    ):
        raise DailyContextError(
            "target_player_ids debe contener enteros positivos."
        )

    selected = frame.loc[:, list(_HISTORY_COLUMNS)].copy()
    selected["match_date"] = pd.to_datetime(
        selected["match_date"],
        errors="raise",
    ).dt.normalize()
    selected["result_available_date"] = pd.to_datetime(
        selected["result_available_date"],
        errors="raise",
    ).dt.normalize()
    expected_available = selected["match_date"].map(
        lambda value: pd.Timestamp(
            source_date_policy.availability_date(value.date())
        )
    )
    if not selected["result_available_date"].equals(expected_available):
        raise DailyContextError(
            "result_available_date no coincide con source_date_policy."
        )
    selected = selected.loc[
        selected["result_available_date"].dt.date < cutoff
    ].copy()
    if selected.empty:
        return CausalHistoryState(
            recent_matches=feature_parameters.recent_matches,
            recent_months=feature_parameters.recent_months,
        )
    if selected["record_id"].isna().any() or selected["record_id"].duplicated().any():
        raise DailyContextError("record_id debe ser no nulo y único.")
    observed_genders = set(selected["gender"].dropna().astype(str).unique())
    if observed_genders != {gender} or selected["gender"].isna().any():
        raise DailyContextError(
            f"El histórico dirigido mezcla géneros: {observed_genders}."
        )
    observed_targets = set(selected["y"].dropna().unique())
    if (
        selected["y"].isna().any()
        or not observed_targets.issubset({0, 1})
    ):
        raise DailyContextError(
            f"Etiquetas históricas inválidas: {observed_targets}."
        )
    for column in ("player_a_id", "player_b_id"):
        numeric = pd.to_numeric(selected[column], errors="raise")
        if numeric.isna().any() or (numeric <= 0).any():
            raise DailyContextError(f"{column} contiene IDs no positivos.")
        selected[column] = numeric.astype("int64")
    involves_target = (
        selected["player_a_id"].isin(target_ids)
        | selected["player_b_id"].isin(target_ids)
    )
    if not bool(involves_target.all()):
        raise DailyContextError(
            "El histórico dirigido contiene partidos ajenos a los objetivos."
        )

    selected.sort_values(
        ["result_available_date", "record_id"],
        kind="stable",
        inplace=True,
    )
    state = CausalHistoryState(
        recent_matches=feature_parameters.recent_matches,
        recent_months=feature_parameters.recent_months,
    )
    for timestamp, day_frame in selected.groupby(
        "result_available_date",
        sort=False,
        observed=True,
    ):
        available_date = cast(pd.Timestamp, timestamp).date()
        source_dates = tuple(
            cast(pd.Timestamp, value).date()
            for value in day_frame["match_date"].drop_duplicates().tolist()
        )
        if len(source_dates) != 1:
            raise DailyContextError(
                "Un bloque disponible mezcla fechas fuente incompatibles."
            )
        source_match_date = source_dates[0]
        results: list[HistoricalMatchResult] = []
        for row in day_frame.itertuples(index=False):
            player_a_id = int(row.player_a_id)
            player_b_id = int(row.player_b_id)
            if player_a_id == player_b_id:
                raise DailyContextError(
                    f"Partido histórico consigo mismo: {row.record_id!r}."
                )
            winner_id, loser_id = (
                (player_a_id, player_b_id)
                if int(row.y) == 1
                else (player_b_id, player_a_id)
            )
            raw_surface = row.surface
            surface = (
                None
                if raw_surface is None or bool(pd.isna(raw_surface))
                else cast(Surface, normalise_surface(str(raw_surface)))
            )
            results.append(
                HistoricalMatchResult(
                    match_date=source_match_date,
                    gender=gender,
                    winner_id=winner_id,
                    loser_id=loser_id,
                    surface=surface,
                )
            )
        state.apply_date_block(
            source_match_date,
            results,
            availability_date=available_date,
        )
    return state


def _validate_model_source(
    model: LoadedDeploymentModel,
    *,
    source_manifest: FeatureSourceManifest,
    metadata: TrainingDatasetMetadata,
) -> None:
    """Exige que el bundle activo corresponda exactamente al Parquet usado."""

    if model.gender != metadata.gender:
        raise DailyContextError("El modelo activo pertenece a otro género.")
    if (
        model.estimator.dataset_fingerprint != source_manifest.fingerprint
        and not has_equivalent_feature_gate(
            champion_fingerprint=model.run_fingerprint,
            feature_fingerprint=source_manifest.fingerprint,
        )
    ):
        raise DailyContextError(
            f"El modelo {model.gender} no corresponde al dataset de features "
            "activo ni existe una equivalencia exacta acreditada por la puerta."
        )
    if model.training_source_rows != metadata.rows:
        raise DailyContextError(
            f"El modelo {model.gender} no acredita todas las filas fuente."
        )
    if (
        model.training_rows + model.training_excluded_unavailable
        != metadata.rows
    ):
        raise DailyContextError(
            f"El subset causal del modelo {model.gender} no reconcilia."
        )
    if model.source_date_policy != source_manifest.source_date_policy:
        raise DailyContextError(
            f"El modelo {model.gender} usa otra política temporal."
        )
    try:
        model_max_date = date.fromisoformat(model.training_max_date)
    except ValueError as exc:
        raise DailyContextError(
            f"training_max_date inválida en el modelo {model.gender}."
        ) from exc
    if model_max_date != metadata.max_date:
        raise DailyContextError(
            f"El corte del modelo {model.gender} no coincide con sus features."
        )


def build_daily_feature_context(
    *,
    as_of_date: date,
    player_ids_by_gender: Mapping[str, set[int] | frozenset[int]],
    models: Mapping[Gender, LoadedDeploymentModel],
    raw_dir: Path = RAW_DATA_DIR,
    sackmann_manifest_path: Path = SACKMANN_MANIFEST_PATH,
    feature_manifest_path: Path = FEATURE_DATASET_MANIFEST_PATH,
    elo_database_path: Path = ELO_DATABASE_PATH,
    identity_quarantine_path: Path = IDENTITY_QUARANTINE_PATH,
    identity_quarantine_manifest_path: Path = (
        IDENTITY_QUARANTINE_MANIFEST_PATH
    ),
    tennisratio_database_path: Path | None = None,
) -> DailyFeatureContext:
    """Construye builders por género y bloquea cualquier fuente incompatible."""

    cutoff = _validate_cutoff(as_of_date)
    targets = _validated_target_ids(player_ids_by_gender)
    source_manifest = load_feature_source_manifest(feature_manifest_path)
    try:
        verified_sources = verify_auxiliary_source_inventory(
            Path(sackmann_manifest_path),
            Path(raw_dir),
            genders=("M", "F"),
        )
    except Exception as exc:
        raise DailyContextError(
            "Los CSV Sackmann activos no superan la verificación de "
            "integridad de partidos, jugadores y rankings."
        ) from exc
    active_source_commit = verified_sources.source_commit
    if active_source_commit != source_manifest.source_commit:
        raise DailyContextError(
            "El raw Sackmann activo cambió después de construir features; "
            "regenere Elo, features y modelos antes de predecir."
        )
    try:
        quarantine = load_identity_quarantine(
            expected_source_commit=active_source_commit,
            csv_path=Path(identity_quarantine_path),
            manifest_path=Path(identity_quarantine_manifest_path),
        )
    except IdentityIntegrityError as exc:
        raise DailyContextError(
            "La cuarentena de identidades no corresponde al snapshot activo."
        ) from exc
    published_quarantine = _manifest_quarantine_keys(source_manifest)
    if quarantine.keys != published_quarantine:
        raise DailyContextError(
            "La cuarentena activa difiere de la usada para construir features."
        )
    requested_keys = {
        (gender, player_id)
        for gender, player_ids in targets.items()
        for player_id in player_ids
    }
    ambiguous_targets = sorted(requested_keys.intersection(quarantine.keys))
    if ambiguous_targets:
        raise DailyContextError(
            "Se solicitaron identidades Sackmann ambiguas: "
            f"{ambiguous_targets}."
        )
    parameters = _feature_parameters(source_manifest)
    date_policy = _source_date_policy(source_manifest)
    if set(models) != set(targets):
        raise DailyContextError(
            "Debe proporcionarse exactamente un modelo por género objetivo."
        )

    age_index = PlayerAgeIndex.from_raw(
        raw_dir=Path(raw_dir),
        reference_years=parameters.age_reference_years,
        days_per_year=parameters.days_per_year,
    )
    contexts: dict[Gender, GenderFeatureContext] = {}
    elo_store = EloStore(Path(elo_database_path))
    expected_elo_version = source_manifest.raw_payload.get(
        "elo_algorithm_version"
    )
    for gender, player_ids in targets.items():
        metadata = verify_training_dataset(source_manifest, gender)
        model = models[gender]
        _validate_model_source(
            model,
            source_manifest=source_manifest,
            metadata=metadata,
        )
        elo_run = elo_store.resolve_complete_run(gender=gender)
        elo_contract = _elo_contract(source_manifest, gender)
        if elo_run.source_commit != source_manifest.source_commit:
            raise DailyContextError(
                f"El run Elo {gender} procede de otro commit Sackmann."
            )
        if (
            not isinstance(expected_elo_version, str)
            or elo_run.algorithm_version != expected_elo_version
        ):
            raise DailyContextError(
                f"La versión Elo activa de {gender} no coincide con features."
            )
        if elo_contract.get("algorithm_version") != elo_run.algorithm_version:
            raise DailyContextError(
                f"El contrato de algoritmo Elo {gender} no coincide."
            )
        expected_parameters = elo_contract.get("parameters")
        if not isinstance(expected_parameters, Mapping):
            raise DailyContextError(
                f"El contrato Elo de {gender} no declara parámetros."
            )
        exact_elo_run = (
            elo_contract.get("input_fingerprint") == elo_run.input_fingerprint
        )
        active_base_parameters = dict(elo_run.parameters)
        active_operational_contract = active_base_parameters.pop(
            "operational_overlay",
            None,
        )
        if exact_elo_run:
            if dict(elo_run.parameters) != dict(expected_parameters):
                raise DailyContextError(
                    f"Los parámetros/identidad Elo de {gender} no coinciden."
                )
        elif (
            not isinstance(active_operational_contract, Mapping)
            or active_base_parameters != dict(expected_parameters)
        ):
            raise DailyContextError(
                f"El Elo activo de {gender} no es una extensión operativa "
                "exacta de la base usada por las features."
            )
        operational_result_cutoff = _operational_result_cutoff(
            elo_run.parameters,
            base_effective_max_date=metadata.max_date,
            source_commit=elo_run.source_commit,
        )
        if elo_contract.get("source_date_policy") != date_policy.as_dict():
            raise DailyContextError(
                f"La política temporal Elo de {gender} no coincide."
            )
        if elo_contract.get("historical_identity_exclusion") != "disabled":
            raise DailyContextError(
                "La cuarentena DOB no puede seleccionar histórico."
            )
        expected_elo_max_date = date_policy.availability_date(
            metadata.max_date
        )
        if (
            elo_run.max_event_date is None
            or elo_run.max_event_date < expected_elo_max_date
        ):
            raise DailyContextError(
                f"El corte Elo disponible de {gender} no alcanza el "
                "dataset y su embargo."
            )
        ranking_index = RankingIndex.from_raw(gender, raw_dir=Path(raw_dir))
        directed = _read_target_history(
            metadata,
            as_of_date=cutoff,
            player_ids=player_ids,
        )
        history_state = rebuild_history_state(
            directed,
            gender=gender,
            as_of_date=cutoff,
            target_player_ids=frozenset(player_ids),
            feature_parameters=parameters,
            source_date_policy=date_policy,
        )
        base_history_available_date = date_policy.availability_date(
            metadata.max_date
        )
        elo_provider = None
        ranking_provider = ranking_index
        overlay_fingerprint = None
        history_effective_max_date = metadata.max_date
        history_available_max_date = base_history_available_date
        ranking_max_date = ranking_index.max_ranking_date
        supplemental_result_rows = 0
        supplemental_ranking_rows = 0
        if tennisratio_database_path is not None:
            try:
                overlay = build_tennisratio_overlay(
                    gender=gender,
                    as_of_date=cutoff,
                    # Effective overlap and state availability are different
                    # clocks: the sidecar may safely fill the embargo gap when
                    # its own observation became available after the Elo base.
                    base_result_effective_cutoff=operational_result_cutoff,
                    base_history_effective_max_date=metadata.max_date,
                    base_history_available_max_date=(
                        base_history_available_date
                    ),
                    target_player_ids=player_ids,
                    history_state=history_state,
                    ranking_index=ranking_index,
                    elo_store=elo_store,
                    elo_run_id=elo_run.run_id,
                    elo_base_date=cast(date, elo_run.max_event_date),
                    elo_parameters=elo_run.parameters,
                    source_commit=elo_run.source_commit,
                    database_path=Path(tennisratio_database_path),
                )
            except TennisRatioOverlayError as exc:
                raise DailyContextError(
                    "El overlay causal TennisRatio no pudo verificarse."
                ) from exc
            elo_provider = overlay.elo_provider
            ranking_provider = overlay.ranking_provider
            overlay_fingerprint = overlay.overlay_fingerprint
            history_effective_max_date = (
                overlay.history_effective_max_date
            )
            history_available_max_date = (
                overlay.history_available_max_date
            )
            ranking_max_date = overlay.ranking_max_date
            supplemental_result_rows = overlay.supplemental_results
            supplemental_ranking_rows = overlay.supplemental_rankings
        builder = MatchFeatureBuilder(
            history_state=history_state,
            ranking_index=ranking_index,
            age_index=age_index,
            elo_database_path=Path(elo_database_path),
            elo_run_id=elo_run.run_id,
            elo_provider=elo_provider,
            ranking_provider=ranking_provider,
        )
        contexts[gender] = GenderFeatureContext(
            gender=gender,
            builder=builder,
            training_metadata=metadata,
            elo_run_id=elo_run.run_id,
            elo_max_date=cast(date, elo_run.max_event_date),
            overlay_fingerprint=overlay_fingerprint,
            history_effective_max_date=history_effective_max_date,
            history_available_max_date=history_available_max_date,
            ranking_max_date=ranking_max_date,
            targeted_history_rows=len(directed),
            supplemental_result_rows=supplemental_result_rows,
            supplemental_ranking_rows=supplemental_ranking_rows,
        )
    return DailyFeatureContext(
        feature_fingerprint=source_manifest.fingerprint,
        source_commit=source_manifest.source_commit,
        feature_parameters=parameters,
        source_date_policy=date_policy,
        identity_quarantine_keys=quarantine.keys,
        by_gender=contexts,
    )

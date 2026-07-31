"""Construcción modular del vector causal de un partido.

El módulo separa la consulta de fuentes temporales de la composición de
columnas. ``MatchFeatureBuilder`` sirve para una predicción arbitraria y
``assemble_match_feature_vector`` permite al constructor histórico reutilizar
exactamente las mismas fórmulas con los snapshots prepartido del motor Elo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping, cast

from ..config import ELO_DATABASE_PATH
from ..elo import (
    EloQuery,
    EloSnapshot,
    Gender,
    PlayerPreMatchRating,
    Surface,
    get_elos,
    normalise_surface,
)
from .levels import normalize_tour_level
from .market import (
    MarketProbabilities,
    calculate_two_way_market_probabilities,
    validate_market_timestamp,
)
from .players import PlayerAgeIndex, PlayerAgeSnapshot
from .rankings import RankingIndex, RankingSnapshot
from .state import CausalHistoryState, HistorySnapshot


MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "surface",
    "tour_level",
    "tour_level_raw",
    "best_of",
    "round",
    "elo_general_diff",
    "elo_surface_diff",
    "elo_surface_raw_diff",
    "elo_general_matches_diff",
    "elo_surface_matches_diff",
    "elo_cold_start_a",
    "elo_cold_start_b",
    "recent_n_win_rate_diff",
    "recent_n_matches_a",
    "recent_n_matches_b",
    "recent_months_win_rate_diff",
    "recent_months_matches_a",
    "recent_months_matches_b",
    "h2h_global_balance",
    "h2h_global_matches",
    "h2h_surface_balance",
    "h2h_surface_matches",
    "rest_days_a",
    "rest_days_b",
    "rest_days_diff",
    "rank_diff",
    "rank_points_diff",
    "ranking_missing_a",
    "ranking_missing_b",
    "ranking_age_days_a",
    "ranking_age_days_b",
    "ranking_conflict_dates_skipped_a",
    "ranking_conflict_dates_skipped_b",
    "age_diff",
    "age_distance_30_diff",
    "age_missing_a",
    "age_missing_b",
    "odds_a",
    "odds_b",
    "market_probability_a",
    "market_probability_b",
    "market_overround",
    "market_margin",
)

VECTOR_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "match_date",
    "player_a_id",
    "player_b_id",
    "surface",
    "tour_level_raw",
    "tour_level",
    "source_family",
    "best_of",
    "round",
    "elo_general_a",
    "elo_general_b",
    "elo_general_diff",
    "elo_surface_raw_a",
    "elo_surface_raw_b",
    "elo_surface_raw_diff",
    "elo_surface_a",
    "elo_surface_b",
    "elo_surface_diff",
    "elo_general_matches_a",
    "elo_general_matches_b",
    "elo_general_matches_diff",
    "elo_surface_matches_a",
    "elo_surface_matches_b",
    "elo_surface_matches_diff",
    "elo_cold_start_a",
    "elo_cold_start_b",
    "recent_n_win_rate_a",
    "recent_n_win_rate_b",
    "recent_n_win_rate_diff",
    "recent_n_matches_a",
    "recent_n_matches_b",
    "recent_months_win_rate_a",
    "recent_months_win_rate_b",
    "recent_months_win_rate_diff",
    "recent_months_matches_a",
    "recent_months_matches_b",
    "h2h_global_balance",
    "h2h_global_matches",
    "h2h_surface_balance",
    "h2h_surface_matches",
    "rest_days_a",
    "rest_days_b",
    "rest_days_diff",
    "ranking_date_a",
    "ranking_date_b",
    "rank_a",
    "rank_b",
    "rank_diff",
    "rank_points_a",
    "rank_points_b",
    "rank_points_diff",
    "ranking_missing_a",
    "ranking_missing_b",
    "ranking_age_days_a",
    "ranking_age_days_b",
    "ranking_conflict_dates_skipped_a",
    "ranking_conflict_dates_skipped_b",
    "birth_date_a",
    "birth_date_b",
    "age_a",
    "age_b",
    "age_diff",
    "age_distance_30_a",
    "age_distance_30_b",
    "age_distance_30_diff",
    "age_missing_a",
    "age_missing_b",
    "age_invalid_for_date_a",
    "age_invalid_for_date_b",
    "odds_a",
    "odds_b",
    "raw_implied_probability_a",
    "raw_implied_probability_b",
    "market_probability_a",
    "market_probability_b",
    "market_overround",
    "market_margin",
    "market_retrieved_at_utc",
    "prediction_as_of_utc",
    "model_probability_a",
    "edge",
)


class FeatureVectorError(ValueError):
    """Indica que una petición o sus snapshots no están alineados."""


@dataclass(frozen=True, slots=True)
class MatchFeatureRequest:
    """Describe los datos conocidos al solicitar features para A contra B."""

    gender: Gender
    player_a_id: int
    player_b_id: int
    as_of_date: date
    surface: str | None
    tour_level_raw: str
    source_family: str
    best_of: int | None
    round: str | None
    odds_a: float | None = None
    odds_b: float | None = None
    market_retrieved_at_utc: datetime | None = None
    prediction_as_of_utc: datetime | None = None
    model_probability_a: float | None = None

    def __post_init__(self) -> None:
        """Valida identidad, fecha y contexto sin completar valores ausentes."""

        if self.gender not in {"M", "F"}:
            raise FeatureVectorError("gender debe ser exactamente 'M' o 'F'.")
        for field_name, player_id in (
            ("player_a_id", self.player_a_id),
            ("player_b_id", self.player_b_id),
        ):
            if (
                isinstance(player_id, bool)
                or not isinstance(player_id, int)
                or player_id <= 0
            ):
                raise FeatureVectorError(
                    f"{field_name} debe ser un entero positivo."
                )
        if self.player_a_id == self.player_b_id:
            raise FeatureVectorError("A y B deben ser jugadores distintos.")
        if (
            isinstance(self.as_of_date, datetime)
            or not isinstance(self.as_of_date, date)
        ):
            raise FeatureVectorError(
                "as_of_date debe ser datetime.date estricto."
            )
        normalize_tour_level(self.tour_level_raw, self.source_family)
        normalise_surface(self.surface)
        if self.best_of is not None and (
            isinstance(self.best_of, bool)
            or not isinstance(self.best_of, int)
            or self.best_of <= 0
        ):
            raise FeatureVectorError(
                "best_of debe ser un entero positivo o None."
            )
        if self.round is not None and (
            not isinstance(self.round, str) or not self.round
        ):
            raise FeatureVectorError("round debe ser texto no vacío o None.")
        if self.model_probability_a is not None:
            _validated_probability(
                self.model_probability_a,
                "model_probability_a",
            )


@dataclass(frozen=True, slots=True)
class EloFeatureSnapshot:
    """Vista Elo mínima común al servicio persistido y al motor histórico."""

    gender: Gender
    player_id: int
    surface: Surface | None
    general_elo: float
    surface_elo_raw: float | None
    combined_elo: float
    general_matches: int
    surface_matches: int
    is_cold_start: bool

    @classmethod
    def from_elo_snapshot(
        cls,
        snapshot: EloSnapshot,
    ) -> "EloFeatureSnapshot":
        """Adapta una consulta persistida de la fase 3."""

        return cls(
            gender=snapshot.gender,
            player_id=snapshot.player_id,
            surface=snapshot.surface,
            general_elo=snapshot.general_elo,
            surface_elo_raw=snapshot.surface_elo_raw,
            combined_elo=snapshot.combined_elo,
            general_matches=snapshot.general_matches,
            surface_matches=snapshot.surface_matches,
            is_cold_start=snapshot.is_cold_start,
        )

    @classmethod
    def from_pre_match_rating(
        cls,
        rating: PlayerPreMatchRating,
    ) -> "EloFeatureSnapshot":
        """Adapta el snapshot pre-D producido por el motor Elo histórico."""

        return cls(
            gender=rating.gender,
            player_id=rating.player_id,
            surface=rating.surface,
            general_elo=rating.general_elo,
            surface_elo_raw=rating.surface_elo_raw,
            combined_elo=rating.combined_elo,
            general_matches=rating.general_matches,
            surface_matches=rating.surface_matches,
            is_cold_start=rating.general_matches == 0,
        )


@dataclass(frozen=True, slots=True)
class MatchFeatureVector:
    """Vector inmutable por copia y lista explícita de columnas de modelo."""

    values: Mapping[str, object]

    def __post_init__(self) -> None:
        """Copia el mapping para impedir mutaciones externas posteriores."""

        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )

    def to_dict(self) -> dict[str, object]:
        """Devuelve una copia mutable apta para DataFrame o serialización."""

        return dict(self.values)

    def model_values(self) -> dict[str, object]:
        """Selecciona solo el allowlist previsto como entrada del modelo."""

        return {
            column: self.values[column]
            for column in MODEL_FEATURE_COLUMNS
        }


def _validated_probability(value: object, field_name: str) -> float:
    """Exige una probabilidad finita dentro del intervalo cerrado unidad."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise FeatureVectorError(
            f"{field_name} debe ser una probabilidad finita entre 0 y 1."
        )
    return float(value)


def _difference(
    value_a: int | float | None,
    value_b: int | float | None,
) -> int | float | None:
    """Calcula A−B y conserva nulo cuando falta cualquiera de los lados."""

    if value_a is None or value_b is None:
        return None
    return value_a - value_b


def _validate_snapshot_identity(
    *,
    request: MatchFeatureRequest,
    elo_a: EloFeatureSnapshot,
    elo_b: EloFeatureSnapshot,
    history: HistorySnapshot,
    ranking_a: RankingSnapshot,
    ranking_b: RankingSnapshot,
    age_a: PlayerAgeSnapshot,
    age_b: PlayerAgeSnapshot,
) -> None:
    """Comprueba que ningún servicio devolvió otro jugador, género o corte."""

    expected = (
        (elo_a.gender, elo_a.player_id, request.player_a_id),
        (elo_b.gender, elo_b.player_id, request.player_b_id),
        (ranking_a.gender, ranking_a.player_id, request.player_a_id),
        (ranking_b.gender, ranking_b.player_id, request.player_b_id),
        (age_a.gender, age_a.player_id, request.player_a_id),
        (age_b.gender, age_b.player_id, request.player_b_id),
    )
    for observed_gender, observed_player, expected_player in expected:
        if (
            observed_gender != request.gender
            or observed_player != expected_player
        ):
            raise FeatureVectorError(
                "Un snapshot no coincide con el género/jugador solicitado."
            )
    if (
        history.gender != request.gender
        or history.player_a_id != request.player_a_id
        or history.player_b_id != request.player_b_id
        or history.as_of_date != request.as_of_date
    ):
        raise FeatureVectorError(
            "El snapshot histórico no coincide con la petición."
        )
    for snapshot_date in (
        ranking_a.as_of_date,
        ranking_b.as_of_date,
        age_a.as_of_date,
        age_b.as_of_date,
    ):
        if snapshot_date != request.as_of_date:
            raise FeatureVectorError(
                "Todos los snapshots deben usar el mismo as_of_date."
            )


def _market_for_request(
    request: MatchFeatureRequest,
) -> MarketProbabilities:
    """Calcula el mercado y exige procedencia temporal si está disponible."""

    market = calculate_two_way_market_probabilities(
        request.odds_a,
        request.odds_b,
    )
    if market.is_available:
        if (
            request.market_retrieved_at_utc is None
            or request.prediction_as_of_utc is None
        ):
            raise FeatureVectorError(
                "Cuotas completas requieren retrieved_at_utc y "
                "prediction_as_of_utc."
            )
        validate_market_timestamp(
            request.market_retrieved_at_utc,
            request.prediction_as_of_utc,
        )
    elif (
        request.market_retrieved_at_utc is not None
        or request.prediction_as_of_utc is not None
    ):
        raise FeatureVectorError(
            "No se admiten timestamps de mercado sin dos cuotas completas."
        )
    return market


def assemble_match_feature_vector(
    request: MatchFeatureRequest,
    *,
    elo_a: EloFeatureSnapshot,
    elo_b: EloFeatureSnapshot,
    history: HistorySnapshot,
    ranking_a: RankingSnapshot,
    ranking_b: RankingSnapshot,
    age_a: PlayerAgeSnapshot,
    age_b: PlayerAgeSnapshot,
) -> MatchFeatureVector:
    """Compone todas las features con orientación A−B y mercado de-vigado."""

    _validate_snapshot_identity(
        request=request,
        elo_a=elo_a,
        elo_b=elo_b,
        history=history,
        ranking_a=ranking_a,
        ranking_b=ranking_b,
        age_a=age_a,
        age_b=age_b,
    )
    surface = normalise_surface(request.surface)
    if (
        elo_a.surface != surface
        or elo_b.surface != surface
        or history.surface != surface
    ):
        raise FeatureVectorError(
            "La superficie de los snapshots no coincide con la petición."
        )
    level = normalize_tour_level(
        request.tour_level_raw,
        request.source_family,
    )
    market = _market_for_request(request)
    model_probability = request.model_probability_a
    edge = (
        None
        if (
            model_probability is None
            or market.market_probability_a_devig is None
        )
        else model_probability - market.market_probability_a_devig
    )

    values: dict[str, object] = {
        "gender": request.gender,
        "match_date": request.as_of_date,
        "player_a_id": request.player_a_id,
        "player_b_id": request.player_b_id,
        "surface": surface,
        "tour_level_raw": level.raw_level,
        "tour_level": level.canonical_level,
        "source_family": level.source_family,
        "best_of": request.best_of,
        "round": request.round,
        "elo_general_a": elo_a.general_elo,
        "elo_general_b": elo_b.general_elo,
        "elo_general_diff": (
            elo_a.general_elo - elo_b.general_elo
        ),
        "elo_surface_raw_a": elo_a.surface_elo_raw,
        "elo_surface_raw_b": elo_b.surface_elo_raw,
        "elo_surface_raw_diff": _difference(
            elo_a.surface_elo_raw,
            elo_b.surface_elo_raw,
        ),
        "elo_surface_a": elo_a.combined_elo,
        "elo_surface_b": elo_b.combined_elo,
        "elo_surface_diff": (
            elo_a.combined_elo - elo_b.combined_elo
        ),
        "elo_general_matches_a": elo_a.general_matches,
        "elo_general_matches_b": elo_b.general_matches,
        "elo_general_matches_diff": (
            elo_a.general_matches - elo_b.general_matches
        ),
        "elo_surface_matches_a": elo_a.surface_matches,
        "elo_surface_matches_b": elo_b.surface_matches,
        "elo_surface_matches_diff": (
            elo_a.surface_matches - elo_b.surface_matches
        ),
        "elo_cold_start_a": elo_a.is_cold_start,
        "elo_cold_start_b": elo_b.is_cold_start,
        "recent_n_win_rate_a": history.recent_n_win_rate_a,
        "recent_n_win_rate_b": history.recent_n_win_rate_b,
        "recent_n_win_rate_diff": history.recent_n_win_rate_diff,
        "recent_n_matches_a": history.recent_n_matches_a,
        "recent_n_matches_b": history.recent_n_matches_b,
        "recent_months_win_rate_a": history.recent_months_win_rate_a,
        "recent_months_win_rate_b": history.recent_months_win_rate_b,
        "recent_months_win_rate_diff": (
            history.recent_months_win_rate_diff
        ),
        "recent_months_matches_a": history.recent_months_matches_a,
        "recent_months_matches_b": history.recent_months_matches_b,
        "h2h_global_balance": history.h2h_global_balance,
        "h2h_global_matches": history.h2h_global_matches,
        "h2h_surface_balance": history.h2h_surface_balance,
        "h2h_surface_matches": history.h2h_surface_matches,
        "rest_days_a": history.rest_days_a,
        "rest_days_b": history.rest_days_b,
        "rest_days_diff": history.rest_days_diff,
        "ranking_date_a": ranking_a.ranking_date,
        "ranking_date_b": ranking_b.ranking_date,
        "rank_a": ranking_a.rank,
        "rank_b": ranking_b.rank,
        "rank_diff": _difference(ranking_a.rank, ranking_b.rank),
        "rank_points_a": ranking_a.points,
        "rank_points_b": ranking_b.points,
        "rank_points_diff": _difference(
            ranking_a.points,
            ranking_b.points,
        ),
        "ranking_missing_a": ranking_a.is_missing,
        "ranking_missing_b": ranking_b.is_missing,
        "ranking_age_days_a": ranking_a.ranking_age_days,
        "ranking_age_days_b": ranking_b.ranking_age_days,
        "ranking_conflict_dates_skipped_a": (
            ranking_a.conflict_dates_skipped
        ),
        "ranking_conflict_dates_skipped_b": (
            ranking_b.conflict_dates_skipped
        ),
        "birth_date_a": age_a.birth_date,
        "birth_date_b": age_b.birth_date,
        "age_a": age_a.age_years,
        "age_b": age_b.age_years,
        "age_diff": _difference(age_a.age_years, age_b.age_years),
        "age_distance_30_a": age_a.distance_to_reference,
        "age_distance_30_b": age_b.distance_to_reference,
        "age_distance_30_diff": _difference(
            age_a.distance_to_reference,
            age_b.distance_to_reference,
        ),
        "age_missing_a": age_a.is_missing,
        "age_missing_b": age_b.is_missing,
        "age_invalid_for_date_a": age_a.invalid_for_date,
        "age_invalid_for_date_b": age_b.invalid_for_date,
        "odds_a": request.odds_a,
        "odds_b": request.odds_b,
        "raw_implied_probability_a": (
            market.raw_implied_probability_a
        ),
        "raw_implied_probability_b": (
            market.raw_implied_probability_b
        ),
        "market_probability_a": (
            market.market_probability_a_devig
        ),
        "market_probability_b": (
            market.market_probability_b_devig
        ),
        "market_overround": market.overround,
        "market_margin": market.margin,
        "market_retrieved_at_utc": request.market_retrieved_at_utc,
        "prediction_as_of_utc": request.prediction_as_of_utc,
        "model_probability_a": model_probability,
        "edge": edge,
    }
    if tuple(values) != VECTOR_COLUMNS:
        raise RuntimeError(
            "La implementación del vector diverge de VECTOR_COLUMNS."
        )
    return MatchFeatureVector(values)


class MatchFeatureBuilder:
    """Consulta servicios temporales y devuelve un vector para un partido."""

    def __init__(
        self,
        *,
        history_state: CausalHistoryState,
        ranking_index: RankingIndex,
        age_index: PlayerAgeIndex,
        elo_database_path: Path = ELO_DATABASE_PATH,
        elo_run_id: str | None = None,
    ) -> None:
        """Inyecta dependencias para mantener la construcción testeable."""

        if ranking_index.gender not in {"M", "F"}:
            raise FeatureVectorError("RankingIndex contiene género inválido.")
        self.history_state = history_state
        self.ranking_index = ranking_index
        self.age_index = age_index
        self.elo_database_path = Path(elo_database_path)
        self.elo_run_id = elo_run_id

    def build(
        self,
        request: MatchFeatureRequest,
    ) -> MatchFeatureVector:
        """Construye el vector usando solo snapshots estrictamente previos a D."""

        if request.gender != self.ranking_index.gender:
            raise FeatureVectorError(
                "El género de la petición no coincide con RankingIndex."
            )
        surface = normalise_surface(request.surface)
        elo_results = get_elos(
            queries=(
                EloQuery(
                    gender=request.gender,
                    player_id=request.player_a_id,
                    surface=surface,
                    as_of_date=request.as_of_date,
                ),
                EloQuery(
                    gender=request.gender,
                    player_id=request.player_b_id,
                    surface=surface,
                    as_of_date=request.as_of_date,
                ),
            ),
            db_path=self.elo_database_path,
            run_id=self.elo_run_id,
        )
        ranking_a, ranking_b = self.ranking_index.get_many(
            (request.player_a_id, request.player_b_id),
            request.as_of_date,
        )
        age_a, age_b = self.age_index.get_pair(
            request.gender,
            request.player_a_id,
            request.player_b_id,
            as_of_date=request.as_of_date,
        )
        history = self.history_state.snapshot(
            request.gender,
            request.player_a_id,
            request.player_b_id,
            surface=cast(Surface | None, surface),
            as_of_date=request.as_of_date,
        )
        return assemble_match_feature_vector(
            request,
            elo_a=EloFeatureSnapshot.from_elo_snapshot(elo_results[0]),
            elo_b=EloFeatureSnapshot.from_elo_snapshot(elo_results[1]),
            history=history,
            ranking_a=ranking_a,
            ranking_b=ranking_b,
            age_a=age_a,
            age_b=age_b,
        )

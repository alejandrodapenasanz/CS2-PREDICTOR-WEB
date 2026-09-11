"""Pruebas offline de composición y orientación del vector de features."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import sys
from typing import Iterable, cast
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import EloQuery, EloSnapshot, Gender  # noqa: E402
from src.features.players import (  # noqa: E402
    PlayerAgeIndex,
    PlayerAgeSnapshot,
)
from src.features.rankings import (  # noqa: E402
    RankingIndex,
    RankingSnapshot,
)
from src.features.state import (  # noqa: E402
    CausalHistoryState,
    HistorySnapshot,
)
from src.features.vector import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    VECTOR_COLUMNS,
    EloFeatureSnapshot,
    FeatureVectorError,
    MatchFeatureBuilder,
    MatchFeatureRequest,
    assemble_match_feature_vector,
)


CUTOFF = date(2024, 6, 10)


def _request(**overrides: object) -> MatchFeatureRequest:
    """Construye una petición válida y permite sobrescribir campos."""

    values: dict[str, object] = {
        "gender": "M",
        "player_a_id": 1,
        "player_b_id": 2,
        "as_of_date": CUTOFF,
        "surface": "Clay",
        "tour_level_raw": "A",
        "source_family": "atp_main",
        "best_of": 3,
        "round": "R32",
    }
    values.update(overrides)
    return MatchFeatureRequest(**values)  # type: ignore[arg-type]


def _elo(player_id: int, rating: float) -> EloFeatureSnapshot:
    """Crea una vista Elo mínima para tests."""

    return EloFeatureSnapshot(
        gender="M",
        player_id=player_id,
        surface="Clay",
        general_elo=rating,
        surface_elo_raw=rating + 20.0,
        combined_elo=rating + 10.0,
        general_matches=100 if player_id == 1 else 80,
        surface_matches=50 if player_id == 1 else 40,
        is_cold_start=False,
    )


def _history() -> HistorySnapshot:
    """Crea métricas históricas orientadas A contra B."""

    return HistorySnapshot(
        as_of_date=CUTOFF,
        gender="M",
        player_a_id=1,
        player_b_id=2,
        surface="Clay",
        recent_n_win_rate_a=0.7,
        recent_n_matches_a=10,
        recent_n_win_rate_b=0.4,
        recent_n_matches_b=10,
        recent_months_win_rate_a=0.6,
        recent_months_matches_a=15,
        recent_months_win_rate_b=0.5,
        recent_months_matches_b=12,
        h2h_global_balance=0.25,
        h2h_global_matches=8,
        h2h_surface_balance=0.5,
        h2h_surface_matches=4,
        rest_days_a=5,
        rest_days_b=3,
    )


def _ranking(
    player_id: int,
    rank: int | None,
    points: int | None,
) -> RankingSnapshot:
    """Crea un snapshot temporal de ranking."""

    return RankingSnapshot(
        gender="M",
        player_id=player_id,
        as_of_date=CUTOFF,
        ranking_date=(
            date(2024, 6, 3) if rank is not None else None
        ),
        rank=rank,
        points=points,
        is_missing=rank is None,
        ranking_age_days=7 if rank is not None else None,
        conflict_dates_skipped=1 if player_id == 1 else 0,
    )


def _age(player_id: int, age: float | None) -> PlayerAgeSnapshot:
    """Crea una edad orientada para tests."""

    return PlayerAgeSnapshot(
        gender="M",
        player_id=player_id,
        as_of_date=CUTOFF,
        birth_date=(
            date(1994, 1, 1) if age is not None else None
        ),
        age_years=age,
        distance_to_reference=(
            abs(age - 30.0) if age is not None else None
        ),
        is_missing=age is None,
        invalid_for_date=False,
    )


def _assemble(request: MatchFeatureRequest | None = None):
    """Compone el vector estándar de la prueba."""

    return assemble_match_feature_vector(
        request or _request(),
        elo_a=_elo(1, 1700.0),
        elo_b=_elo(2, 1600.0),
        history=_history(),
        ranking_a=_ranking(1, 10, 2500),
        ranking_b=_ranking(2, 40, 1200),
        age_a=_age(1, 30.0),
        age_b=_age(2, 25.0),
    )


class _InjectedEloProvider:
    """Proveedor Elo de prueba que evita cualquier acceso a SQLite."""

    def __init__(self) -> None:
        """Inicializa el registro de consultas recibidas."""

        self.queries: tuple[EloQuery, ...] = ()

    def get_many(
        self,
        queries: Iterable[EloQuery],
    ) -> tuple[EloSnapshot, ...]:
        """Responde snapshots causales alineados con cada consulta."""

        self.queries = tuple(queries)
        return tuple(
            EloSnapshot(
                gender=query.gender,
                player_id=query.player_id,
                as_of_date=query.as_of_date,
                state_date=query.as_of_date - timedelta(days=1),
                general_elo=1700.0 if query.player_id == 1 else 1600.0,
                surface=query.surface,
                surface_elo_raw=(
                    1720.0 if query.player_id == 1 else 1620.0
                ),
                combined_elo=(
                    1710.0 if query.player_id == 1 else 1610.0
                ),
                general_matches=100,
                surface_matches=50,
                is_cold_start=False,
                run_id="injected",
                source_commit="a" * 40,
            )
            for query in self.queries
        )


class _InjectedRankingProvider:
    """Proveedor de ranking de prueba con género explícito."""

    gender: Gender = "M"

    def get_many(
        self,
        player_ids: Iterable[int],
        as_of_date: date,
    ) -> tuple[RankingSnapshot, ...]:
        """Responde rankings alineados con ids y fecha recibidos."""

        if as_of_date != CUTOFF:
            raise AssertionError("El builder alteró el corte solicitado.")
        return tuple(
            _ranking(player_id, 10 if player_id == 1 else 40, 2000)
            for player_id in player_ids
        )


class _InjectedAgeIndex:
    """Adaptador mínimo de edades para aislar el test del builder."""

    def get_pair(
        self,
        gender: str,
        player_a_id: int,
        player_b_id: int,
        *,
        as_of_date: date,
    ) -> tuple[PlayerAgeSnapshot, PlayerAgeSnapshot]:
        """Devuelve el par de edades correspondiente a la petición."""

        if gender != "M" or as_of_date != CUTOFF:
            raise AssertionError("El builder alteró género o corte.")
        return _age(player_a_id, 30.0), _age(player_b_id, 25.0)


class MatchFeatureVectorTest(unittest.TestCase):
    """Comprueba diferencias, nulos, mercado y allowlist."""

    def test_all_directional_features_are_a_minus_b(self) -> None:
        """Aplica orientación A−B de forma consistente en cada familia."""

        values = _assemble().values

        self.assertEqual(values["elo_general_diff"], 100.0)
        self.assertEqual(values["elo_surface_raw_diff"], 100.0)
        self.assertEqual(values["elo_surface_diff"], 100.0)
        self.assertAlmostEqual(values["recent_n_win_rate_diff"], 0.3)
        self.assertAlmostEqual(
            values["recent_months_win_rate_diff"],
            0.1,
        )
        self.assertEqual(values["rest_days_diff"], 2)
        self.assertEqual(values["rank_diff"], -30)
        self.assertEqual(values["rank_points_diff"], 1300)
        self.assertEqual(values["age_diff"], 5.0)
        self.assertEqual(values["age_distance_30_diff"], -5.0)
        self.assertEqual(values["tour_level"], "ATP Tour")

    def test_missing_pair_value_keeps_difference_missing(self) -> None:
        """No imputa ranking, edad ni superficie cuando falta un lado."""

        vector = assemble_match_feature_vector(
            _request(),
            elo_a=_elo(1, 1700.0),
            elo_b=_elo(2, 1600.0),
            history=_history(),
            ranking_a=_ranking(1, 10, 2500),
            ranking_b=_ranking(2, None, None),
            age_a=_age(1, 30.0),
            age_b=_age(2, None),
        )

        self.assertIsNone(vector.values["rank_diff"])
        self.assertIsNone(vector.values["rank_points_diff"])
        self.assertIsNone(vector.values["age_diff"])
        self.assertTrue(vector.values["ranking_missing_b"])
        self.assertTrue(vector.values["age_missing_b"])

    def test_market_is_devigged_and_edge_uses_oriented_a(self) -> None:
        """Normaliza cuotas y resta el mercado A de la probabilidad modelo A."""

        prediction = datetime(2024, 6, 9, 12, tzinfo=timezone.utc)
        retrieved = prediction - timedelta(minutes=5)
        request = _request(
            odds_a=2.0,
            odds_b=4.0,
            market_retrieved_at_utc=retrieved,
            prediction_as_of_utc=prediction,
            model_probability_a=0.7,
        )

        values = _assemble(request).values

        self.assertAlmostEqual(values["market_probability_a"], 2.0 / 3.0)
        self.assertAlmostEqual(values["market_probability_b"], 1.0 / 3.0)
        self.assertAlmostEqual(values["edge"], 0.7 - 2.0 / 3.0)

    def test_historical_absence_of_odds_is_explicit(self) -> None:
        """Conserva mercado, probabilidad modelo y edge nulos en fase 6."""

        values = _assemble().values

        self.assertIsNone(values["market_probability_a"])
        self.assertIsNone(values["model_probability_a"])
        self.assertIsNone(values["edge"])

    def test_model_allowlist_excludes_identity_target_and_future_outputs(
        self,
    ) -> None:
        """Evita que metadata o edge entren por selección accidental."""

        vector = _assemble()
        model_values = vector.model_values()

        self.assertEqual(tuple(model_values), MODEL_FEATURE_COLUMNS)
        for forbidden in (
            "player_a_id",
            "player_b_id",
            "gender",
            "match_date",
            "model_probability_a",
            "edge",
        ):
            self.assertNotIn(forbidden, model_values)
        self.assertTrue(math.isfinite(model_values["elo_general_diff"]))
        self.assertEqual(tuple(vector.values), VECTOR_COLUMNS)

    def test_complete_odds_require_strictly_prior_timestamps(self) -> None:
        """Impide incorporar cuotas sin acreditar cuándo se observaron."""

        with self.assertRaises(FeatureVectorError):
            _assemble(_request(odds_a=2.0, odds_b=2.0))

    def test_misaligned_snapshot_fails_clearly(self) -> None:
        """Rechaza un snapshot de otro jugador antes de generar columnas."""

        with self.assertRaises(FeatureVectorError):
            assemble_match_feature_vector(
                _request(),
                elo_a=_elo(2, 1700.0),
                elo_b=_elo(2, 1600.0),
                history=_history(),
                ranking_a=_ranking(1, 10, 2500),
                ranking_b=_ranking(2, 40, 1200),
                age_a=_age(1, 30.0),
                age_b=_age(2, 25.0),
            )

    def test_builder_uses_injected_elo_and_ranking_providers(self) -> None:
        """El overlay puede sustituir Elo/ranking sin consultar persistencia."""

        elo_provider = _InjectedEloProvider()
        ranking_provider = _InjectedRankingProvider()
        builder = MatchFeatureBuilder(
            history_state=CausalHistoryState(),
            ranking_index=cast(RankingIndex, ranking_provider),
            age_index=cast(PlayerAgeIndex, _InjectedAgeIndex()),
            elo_provider=elo_provider,
            ranking_provider=ranking_provider,
        )

        vector = builder.build(_request())

        self.assertEqual(len(elo_provider.queries), 2)
        self.assertEqual(vector.values["elo_general_diff"], 100.0)
        self.assertEqual(vector.values["rank_diff"], -30)


if __name__ == "__main__":
    unittest.main()

"""Tests causales y de degradación del pipeline diario de la fase 8."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.config import PROJECT_ROOT
from src.daily_pipeline import (
    PREDICTION_OUTPUT_COLUMNS,
    assess_vector_confidence,
    predict_mapped_matches,
    publish_predictions_csv,
    rebuild_history_state,
)
from src.features import FeatureParameters


def _history_frame(include_future: bool = True) -> pd.DataFrame:
    """Construye resultados orientados A/B con una fila futura opcional."""

    rows = [
        {
            "record_id": "r1",
            "gender": "M",
            "match_date": date(2024, 1, 1),
            "player_a_id": 1,
            "player_b_id": 2,
            "surface": "Hard",
            "y": 1,
        },
        {
            "record_id": "r2",
            "gender": "M",
            "match_date": date(2024, 1, 2),
            "player_a_id": 3,
            "player_b_id": 1,
            "surface": "Clay",
            "y": 1,
        },
    ]
    if include_future:
        rows.append(
            {
                "record_id": "future",
                "gender": "M",
                "match_date": date(2024, 1, 3),
                "player_a_id": 1,
                "player_b_id": 2,
                "surface": "Hard",
                "y": 0,
            }
        )
    return pd.DataFrame(rows)


def _mapped_frame() -> pd.DataFrame:
    """Crea una cartelera mínima con predicho, no mapeado y terminado."""

    rows = []
    for status, mapping, first_id, second_id in (
        ("scheduled", "mapped", 1, 2),
        ("scheduled", "unmapped", pd.NA, 4),
        ("finished", "mapped", 5, 6),
    ):
        rows.append(
            {
                "match_date": pd.Timestamp("2024-02-01"),
                "tournament": "Test Open",
                "tour_level": "ATP",
                "gender": "M",
                "surface": "Hard",
                "scheduled_time": "10:00",
                "status": status,
                "status_evidence": f"fixture_{status}",
                "player_1_name": f"A{len(rows)}",
                "player_2_name": f"B{len(rows)}",
                "player_1_slug": f"a-{len(rows)}",
                "player_2_slug": f"b-{len(rows)}",
                "player_1_id": first_id,
                "player_2_id": second_id,
                "player_1_odds": 2.0,
                "player_2_odds": 2.0,
                "mapping_status": mapping,
                "retrieved_at_utc": pd.Timestamp(
                    "2024-02-01T08:00:00Z"
                ),
                "snapshot_sha256": "a" * 64,
                "source_match_id": str(1000 + len(rows)),
                "match_detail_href": (
                    f"/match-detail/?id={1000 + len(rows)}"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["player_1_id"] = frame["player_1_id"].astype("Int64")
    frame["player_2_id"] = frame["player_2_id"].astype("Int64")
    return frame


class _FakeVector:
    """Vector mínimo con la API usada por el orquestador."""

    def __init__(self, values: dict[str, object]) -> None:
        """Conserva valores de prueba por copia."""

        self._values = dict(values)

    def to_dict(self) -> dict[str, object]:
        """Devuelve una copia como hace MatchFeatureVector."""

        return dict(self._values)


class _FakeBuilder:
    """Builder determinista que respeta identidad y mercado del request."""

    def build(self, request: object) -> _FakeVector:
        """Produce las columnas que consumen modelo, output y confianza."""

        return _FakeVector(
            {
                "gender": request.gender,
                "surface": request.surface,
                "tour_level": "ATP Tour",
                "tour_level_raw": request.tour_level_raw,
                "best_of": request.best_of,
                "round": request.round,
                "elo_general_matches_a": 100,
                "elo_general_matches_b": 90,
                "elo_surface_matches_a": 40,
                "elo_surface_matches_b": 35,
                "ranking_missing_a": False,
                "ranking_missing_b": False,
                "ranking_age_days_a": 5,
                "ranking_age_days_b": 5,
                "age_missing_a": False,
                "age_missing_b": False,
                "market_probability_a": 0.5,
                "market_probability_b": 0.5,
            }
        )


class _FakeModel:
    """Modelo calibrado falso con el contrato mínimo del servicio real."""

    def __init__(self, training_max_date: str = "2024-01-20") -> None:
        """Configura un corte causal o no causal para cada prueba."""

        self.gender = "M"
        self.training_max_date = training_max_date
        self.run_fingerprint = "f" * 64
        self.estimator = SimpleNamespace(profile="sports_only")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Devuelve probabilidades deterministas preservando el índice."""

        return pd.DataFrame(
            {
                "model_probability_raw_a": 0.7,
                "model_probability_a": 0.65,
                "market_probability_a": frame["market_probability_a"],
                "edge": 0.15,
            },
            index=frame.index,
        )


def _feature_context() -> object:
    """Crea metadata causal mínima para el género masculino."""

    gender_context = SimpleNamespace(
        builder=_FakeBuilder(),
        training_metadata=SimpleNamespace(max_date=date(2024, 1, 20)),
        ranking_max_date=date(2024, 1, 22),
    )
    return SimpleNamespace(
        feature_fingerprint="d" * 64,
        by_gender={"M": gender_context},
    )


class DailyHistoryRebuildTests(unittest.TestCase):
    """Comprueba reconstrucción dirigida y corte estricto por fecha."""

    def test_future_rows_do_not_change_target_history_snapshot(self) -> None:
        """Añadir D/futuro no modifica forma, H2H ni descanso as-of D."""

        parameters = FeatureParameters()
        snapshots = []
        for include_future in (False, True):
            state = rebuild_history_state(
                _history_frame(include_future),
                gender="M",
                as_of_date=date(2024, 1, 3),
                target_player_ids=frozenset({1, 2}),
                feature_parameters=parameters,
            )
            snapshots.append(
                state.snapshot(
                    "M",
                    1,
                    2,
                    surface="Hard",
                    as_of_date=date(2024, 1, 3),
                )
            )
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[0].recent_n_matches_a, 2)
        self.assertAlmostEqual(snapshots[0].recent_n_win_rate_a, 0.5)
        self.assertEqual(snapshots[0].h2h_global_matches, 1)
        self.assertEqual(snapshots[0].rest_days_a, 1)


class DailyConfidenceTests(unittest.TestCase):
    """Fija los umbrales y prueba que no dependen de la probabilidad."""

    def test_stale_or_poor_history_is_low_confidence(self) -> None:
        """Fuentes obsoletas y poca muestra generan flags explícitos."""

        values = {
            "surface": "Hard",
            "best_of": None,
            "round": None,
            "elo_general_matches_a": 2,
            "elo_general_matches_b": 50,
            "elo_surface_matches_a": 1,
            "elo_surface_matches_b": 20,
            "ranking_missing_a": True,
            "ranking_missing_b": False,
            "ranking_age_days_a": None,
            "ranking_age_days_b": 40,
            "age_missing_a": False,
            "age_missing_b": True,
            "market_probability_a": 0.5,
        }
        assessment = assess_vector_confidence(
            values,
            match_date=date(2024, 2, 1),
            history_max_date=date(2024, 1, 1),
            ranking_max_date=date(2024, 1, 1),
        )
        self.assertEqual(assessment.level, "LOW")
        self.assertIn("limited_general_history_a", assessment.flags)
        self.assertIn("ranking_missing_a", assessment.flags)
        self.assertIn("history_stale_31d", assessment.flags)


class DailyPredictionTableTests(unittest.TestCase):
    """Valida orientación A/B, calibración y degradación por fila."""

    def test_only_scheduled_mapped_match_receives_probabilities(self) -> None:
        """No mapeados y terminados se conservan sin inventar probabilidad."""

        output = predict_mapped_matches(
            _mapped_frame(),
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        self.assertEqual(tuple(output.columns), PREDICTION_OUTPUT_COLUMNS)
        self.assertAlmostEqual(output.loc[0, "model_probability_a"], 0.65)
        self.assertAlmostEqual(output.loc[0, "model_probability_b"], 0.35)
        self.assertTrue(pd.isna(output.loc[0, "edge_a"]))
        self.assertTrue(pd.isna(output.loc[0, "edge_b"]))
        self.assertEqual(
            output.loc[0, "market_comparison_status"],
            "prestart_unverified",
        )
        self.assertIn(
            "market_prestart_unverified",
            output.loc[0, "confidence_flags"],
        )
        self.assertEqual(output.loc[0, "predicted_winner_name"], "A0")
        self.assertAlmostEqual(
            output.loc[0, "predicted_winner_probability"],
            0.65,
        )
        self.assertEqual(output.loc[0, "player_a_name"], "A0")
        self.assertEqual(output.loc[1, "confidence"], "UNAVAILABLE")
        self.assertIn("player_unmapped", output.loc[1, "confidence_flags"])
        self.assertTrue(pd.isna(output.loc[1, "model_probability_a"]))
        self.assertIn("status_finished", output.loc[2, "confidence_flags"])
        self.assertTrue(pd.isna(output.loc[2, "model_probability_a"]))

    def test_model_trained_on_date_is_blocked_as_future_leakage(self) -> None:
        """Un modelo cuyo corte no precede D nunca recibe la fila."""

        frame = _mapped_frame().iloc[[0]].copy()
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel("2024-02-01")},
            feature_context=None,
        )
        self.assertTrue(pd.isna(output.iloc[0]["model_probability_a"]))
        self.assertEqual(output.iloc[0]["confidence"], "UNAVAILABLE")
        self.assertEqual(
            output.iloc[0]["confidence_flags"],
            "model_not_causal_for_date",
        )

    def test_quarantined_identity_degrades_only_its_match(self) -> None:
        """Una identidad ambigua queda sin probabilidad y la jornada continúa."""

        frame = _mapped_frame().iloc[[0]].copy()
        output = predict_mapped_matches(
            frame,
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=None,
            excluded_player_keys={("M", 1)},
        )

        self.assertTrue(output.iloc[0]["identity_quarantined"])
        self.assertTrue(pd.isna(output.iloc[0]["model_probability_a"]))
        self.assertEqual(output.iloc[0]["confidence"], "UNAVAILABLE")
        self.assertEqual(
            output.iloc[0]["confidence_flags"],
            "identity_quarantined",
        )

    def test_csv_is_timestamped_and_round_trips_inside_tennis(self) -> None:
        """La publicación es única, legible y permanece dentro del proyecto."""

        output = predict_mapped_matches(
            _mapped_frame().iloc[[0]].copy(),
            match_date=date(2024, 2, 1),
            prediction_as_of_utc=datetime(
                2024,
                2,
                1,
                9,
                tzinfo=UTC,
            ),
            models={"M": _FakeModel()},
            feature_context=_feature_context(),
        )
        with TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            path = publish_predictions_csv(
                output,
                match_date=date(2024, 2, 1),
                prediction_as_of_utc=datetime(
                    2024,
                    2,
                    1,
                    9,
                    tzinfo=UTC,
                ),
                output_dir=Path(temporary),
            )
            self.assertTrue(path.is_file())
            self.assertIn("predictions_2024-02-01_", path.name)
            loaded = pd.read_csv(path)
            self.assertEqual(tuple(loaded.columns), PREDICTION_OUTPUT_COLUMNS)
            self.assertAlmostEqual(loaded.loc[0, "model_probability_a"], 0.65)


if __name__ == "__main__":
    unittest.main()

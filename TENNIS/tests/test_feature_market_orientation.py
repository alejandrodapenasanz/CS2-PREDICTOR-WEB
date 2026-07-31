"""Pruebas offline del de-vig de cuotas y la orientación estable A/B."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.market import (  # noqa: E402
    InvalidOddsError,
    MarketTimestampError,
    calculate_two_way_market_probabilities,
    validate_market_timestamp,
)
from src.features.orientation import (  # noqa: E402
    DEFAULT_ORIENTATION_SEED,
    Orientation,
    OrientationError,
    orient_match,
)


def _record_hash(index: int) -> str:
    """Genera una identidad SHA-256 reproducible para un partido sintético."""

    return hashlib.sha256(f"match-{index}".encode("utf-8")).hexdigest()


class MarketProbabilitiesTest(unittest.TestCase):
    """Comprueba fórmulas, ausencias y validaciones de cuotas."""

    def test_two_way_devig_normalises_raw_probabilities(self) -> None:
        """Aplica exactamente q=1/cuota y normaliza las dos q para sumar uno."""

        result = calculate_two_way_market_probabilities(2.0, 4.0)

        self.assertTrue(result.is_available)
        self.assertAlmostEqual(result.raw_implied_probability_a, 0.5)
        self.assertAlmostEqual(result.raw_implied_probability_b, 0.25)
        self.assertAlmostEqual(result.market_probability_a_devig, 2.0 / 3.0)
        self.assertAlmostEqual(result.market_probability_b_devig, 1.0 / 3.0)
        self.assertAlmostEqual(
            result.market_probability_a_devig
            + result.market_probability_b_devig,
            1.0,
        )
        self.assertAlmostEqual(result.overround, 0.75)
        self.assertAlmostEqual(result.margin, -0.25)

    def test_bookmaker_margin_is_overround_minus_one(self) -> None:
        """Expone por separado el overround y el margen convencional."""

        result = calculate_two_way_market_probabilities(1.8, 1.8)

        self.assertAlmostEqual(result.overround, 10.0 / 9.0)
        self.assertAlmostEqual(result.margin, 1.0 / 9.0)
        self.assertAlmostEqual(result.market_probability_a_devig, 0.5)
        self.assertAlmostEqual(result.market_probability_b_devig, 0.5)

    def test_any_missing_odd_makes_the_entire_result_missing(self) -> None:
        """No conserva una probabilidad unilateral que no se puede de-vigar."""

        for missing_value in (None, float("nan"), pd.NA):
            with self.subTest(missing_value=repr(missing_value)):
                result = calculate_two_way_market_probabilities(
                    missing_value,
                    2.0,
                )
                self.assertFalse(result.is_available)
                for field in fields(result):
                    self.assertIsNone(getattr(result, field.name))

        reversed_result = calculate_two_way_market_probabilities(2.0, None)
        for field in fields(reversed_result):
            self.assertIsNone(getattr(reversed_result, field.name))

    def test_present_odds_must_be_numeric_finite_and_above_one(self) -> None:
        """Rechaza strings, booleanos, infinitos y límites decimales inválidos."""

        invalid_values = (
            "2.0",
            True,
            float("inf"),
            float("-inf"),
            1.0,
            0.0,
            -2.0,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(InvalidOddsError):
                    calculate_two_way_market_probabilities(value, 2.0)

    def test_present_invalid_odd_is_not_hidden_by_other_missing_odd(self) -> None:
        """Valida cada valor presente aunque la pareja no esté completa."""

        with self.assertRaises(InvalidOddsError):
            calculate_two_way_market_probabilities(None, "not-numeric")
        with self.assertRaises(InvalidOddsError):
            calculate_two_way_market_probabilities(1.0, pd.NA)


class MarketTimestampTest(unittest.TestCase):
    """Comprueba la desigualdad causal estricta de la observación de mercado."""

    def test_strictly_earlier_aware_timestamp_is_valid(self) -> None:
        """Acepta una cuota observada antes del instante de predicción."""

        prediction = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        retrieved = prediction - timedelta(minutes=1)

        self.assertIsNone(validate_market_timestamp(retrieved, prediction))

    def test_equivalent_aware_timezones_compare_by_instant(self) -> None:
        """Compara instantes y no representaciones locales de la hora."""

        madrid = timezone(timedelta(hours=2))
        retrieved = datetime(2026, 7, 30, 13, 59, tzinfo=madrid)
        prediction = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

        self.assertIsNone(validate_market_timestamp(retrieved, prediction))

    def test_equal_or_future_retrieval_is_rejected(self) -> None:
        """Impide usar cuotas observadas en o después del corte predictivo."""

        prediction = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        for retrieved in (
            prediction,
            prediction + timedelta(microseconds=1),
        ):
            with self.subTest(retrieved=retrieved):
                with self.assertRaises(MarketTimestampError):
                    validate_market_timestamp(retrieved, prediction)

    def test_both_timestamps_must_be_timezone_aware(self) -> None:
        """Rechaza cualquier combinación que contenga un datetime ingenuo."""

        aware = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        naive = datetime(2026, 7, 30, 11, 0)

        with self.assertRaises(MarketTimestampError):
            validate_market_timestamp(naive, aware)
        with self.assertRaises(MarketTimestampError):
            validate_market_timestamp(aware, naive)
        with self.assertRaises(MarketTimestampError):
            validate_market_timestamp(naive, naive)


class StableOrientationTest(unittest.TestCase):
    """Comprueba identidad, balance y coherencia de la orientación por hash."""

    def test_default_seed_is_documented_value(self) -> None:
        """Fija la semilla aprobada para reproducibilidad entre ejecuciones."""

        self.assertEqual(DEFAULT_ORIENTATION_SEED, 42)

    def test_same_identity_always_has_same_orientation(self) -> None:
        """No depende de estado global ni del orden de llamadas."""

        kwargs = {
            "winner_id": 101,
            "loser_id": 202,
            "gender": "M",
            "source_record_hash": _record_hash(1),
        }

        first = orient_match(**kwargs)
        for _ in range(20):
            self.assertEqual(orient_match(**kwargs), first)

    def test_reorder_and_future_append_do_not_change_old_matches(self) -> None:
        """La orientación histórica se identifica por partido, no por ordinal."""

        hashes = [_record_hash(index) for index in range(100)]
        baseline = {
            record_hash: orient_match(
                1000 + index,
                2000 + index,
                gender="F",
                source_record_hash=record_hash,
            )
            for index, record_hash in enumerate(hashes)
        }

        reordered_with_future = list(reversed(hashes)) + [
            _record_hash(index) for index in range(100, 150)
        ]
        recalculated = {}
        for record_hash in reordered_with_future:
            if record_hash not in baseline:
                continue
            index = hashes.index(record_hash)
            recalculated[record_hash] = orient_match(
                1000 + index,
                2000 + index,
                gender="F",
                source_record_hash=record_hash,
            )

        self.assertEqual(recalculated, baseline)

    def test_gender_is_part_of_orientation_identity(self) -> None:
        """Separa explícitamente los universos masculino y femenino."""

        male_hashes = []
        female_hashes = []
        decision_differences = 0
        for index in range(128):
            source_hash = _record_hash(index)
            male = orient_match(
                1,
                2,
                gender="M",
                source_record_hash=source_hash,
            )
            female = orient_match(
                1,
                2,
                gender="F",
                source_record_hash=source_hash,
            )
            male_hashes.append(male.orientation_hash)
            female_hashes.append(female.orientation_hash)
            decision_differences += male.swapped != female.swapped

        self.assertTrue(
            all(male != female for male, female in zip(male_hashes, female_hashes))
        )
        self.assertGreater(decision_differences, 0)

    def test_label_exactly_tracks_whether_a_is_original_winner(self) -> None:
        """Prueba la coherencia correcta: y no es independiente del ganador."""

        for index in range(200):
            winner_id = 10_000 + index
            loser_id = 20_000 + index
            result = orient_match(
                winner_id,
                loser_id,
                gender="M",
                source_record_hash=_record_hash(index),
            )

            self.assertEqual(result.y, int(result.player_a_id == winner_id))
            self.assertEqual(result.swapped, result.player_a_id == loser_id)
            self.assertEqual(result.player_b_id, loser_id if result.y else winner_id)

    def test_winner_assignment_is_approximately_balanced(self) -> None:
        """Verifica ~50 % de ganador en A sobre una muestra determinista amplia."""

        labels = [
            orient_match(
                100_000 + index,
                200_000 + index,
                gender="F",
                source_record_hash=_record_hash(index),
            ).y
            for index in range(10_000)
        ]
        winner_in_a_rate = sum(labels) / len(labels)

        self.assertGreater(winner_in_a_rate, 0.47)
        self.assertLess(winner_in_a_rate, 0.53)

    def test_seed_changes_hash_identity(self) -> None:
        """Incluye la semilla en el hash auditable de orientación."""

        source_hash = _record_hash(9)
        seed_42 = orient_match(
            1,
            2,
            gender="M",
            source_record_hash=source_hash,
            seed=42,
        )
        seed_43 = orient_match(
            1,
            2,
            gender="M",
            source_record_hash=source_hash,
            seed=43,
        )

        self.assertNotEqual(seed_42.orientation_hash, seed_43.orientation_hash)

    def test_invalid_identity_and_participants_fail_clearly(self) -> None:
        """Rechaza géneros, hashes, seeds e identificadores inválidos."""

        valid_hash = _record_hash(1)
        invalid_calls = (
            lambda: orient_match(
                1,
                2,
                gender="X",
                source_record_hash=valid_hash,
            ),
            lambda: orient_match(
                1,
                2,
                gender="M",
                source_record_hash="not-a-hash",
            ),
            lambda: orient_match(
                1,
                2,
                gender="M",
                source_record_hash=valid_hash,
                seed=True,
            ),
            lambda: orient_match(
                True,
                2,
                gender="M",
                source_record_hash=valid_hash,
            ),
            lambda: orient_match(
                1,
                1,
                gender="M",
                source_record_hash=valid_hash,
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(OrientationError):
                    invalid_call()

    def test_public_dataclass_rejects_incoherent_label(self) -> None:
        """Impide construir manualmente una etiqueta contraria a swapped."""

        with self.assertRaises(OrientationError):
            Orientation(
                player_a_id=1,
                player_b_id=2,
                y=1,
                swapped=True,
                orientation_hash=_record_hash(1),
            )

"""Pruebas offline del estado causal de forma, H2H y descanso."""

from __future__ import annotations

from datetime import date
import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.state import (  # noqa: E402
    CausalHistoryState,
    HistoricalMatchResult,
    HistoryDateOrderError,
)


def _result(
    match_date: date,
    winner_id: int,
    loser_id: int,
    *,
    gender: str = "M",
    surface: str | None = "Hard",
) -> HistoricalMatchResult:
    """Construye un resultado pequeño y explícito para una prueba."""

    return HistoricalMatchResult(
        match_date=match_date,
        gender=gender,  # type: ignore[arg-type]
        winner_id=winner_id,
        loser_id=loser_id,
        surface=surface,  # type: ignore[arg-type]
    )


class CausalHistoryStateTest(unittest.TestCase):
    """Comprueba ventanas, aislamiento y causalidad por fecha."""

    def test_cold_start_exposes_nan_zero_counts_and_unknown_rest(self) -> None:
        """No inventa porcentajes ni descanso para jugadores sin historia."""

        snapshot = CausalHistoryState().snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 1, 1),
        )

        self.assertTrue(math.isnan(snapshot.recent_n_win_rate_a))
        self.assertTrue(math.isnan(snapshot.recent_months_win_rate_b))
        self.assertEqual(snapshot.recent_n_matches_a, 0)
        self.assertEqual(snapshot.recent_months_matches_b, 0)
        self.assertEqual(snapshot.h2h_global_balance, 0.0)
        self.assertEqual(snapshot.h2h_global_matches, 0)
        self.assertEqual(snapshot.h2h_surface_balance, 0.0)
        self.assertEqual(snapshot.h2h_surface_matches, 0)
        self.assertIsNone(snapshot.rest_days_a)
        self.assertIsNone(snapshot.rest_days_b)

    def test_future_results_cannot_change_a_captured_snapshot(self) -> None:
        """Un bloque posterior no muta la vista inmutable obtenida antes."""

        first_date = date(2024, 1, 1)
        cutoff = date(2024, 1, 10)
        state = CausalHistoryState()
        state.apply_date_block(
            first_date,
            (_result(first_date, 1, 2),),
        )
        before = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )

        state.apply_date_block(
            cutoff,
            (_result(cutoff, 2, 1),),
        )

        self.assertEqual(before.recent_n_win_rate_a, 1.0)
        self.assertEqual(before.recent_n_matches_a, 1)
        self.assertEqual(before.h2h_global_balance, 1.0)
        self.assertEqual(before.rest_days_a, 9)
        with self.assertRaises(HistoryDateOrderError):
            state.snapshot(
                "M",
                1,
                2,
                surface="Hard",
                as_of_date=cutoff,
            )

    def test_future_snapshot_does_not_change_later_intermediate_history(
        self,
    ) -> None:
        """Consultar un corte lejano no poda historia usada después."""

        baseline = CausalHistoryState(
            recent_matches=1,
            recent_months=3,
        )
        probed = CausalHistoryState(
            recent_matches=1,
            recent_months=3,
        )
        january_1 = date(2024, 1, 1)
        january_2 = date(2024, 1, 2)
        february_1 = date(2024, 2, 1)
        for state in (baseline, probed):
            state.apply_date_block(
                january_1,
                (_result(january_1, 1, 2),),
            )
            state.apply_date_block(
                january_2,
                (_result(january_2, 2, 1),),
            )

        probed.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2030, 12, 31),
        )
        for state in (baseline, probed):
            state.apply_date_block(
                february_1,
                (_result(february_1, 1, 2),),
            )

        baseline_snapshot = baseline.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 2, 2),
        )
        probed_snapshot = probed.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 2, 2),
        )

        self.assertEqual(probed_snapshot, baseline_snapshot)
        self.assertEqual(probed_snapshot.recent_months_matches_a, 3)
        self.assertAlmostEqual(
            probed_snapshot.recent_months_win_rate_a,
            2.0 / 3.0,
        )

    def test_same_date_is_frozen_until_the_complete_block_is_applied(
        self,
    ) -> None:
        """Todos los partidos de D parten del mismo estado pre-D."""

        match_date = date(2024, 2, 1)
        state = CausalHistoryState()
        first_view = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=match_date,
        )
        second_view = state.snapshot(
            "M",
            1,
            3,
            surface="Hard",
            as_of_date=match_date,
        )

        self.assertEqual(first_view.recent_n_matches_a, 0)
        self.assertEqual(second_view.recent_n_matches_a, 0)
        state.apply_date_block(
            match_date,
            (
                _result(match_date, 1, 2),
                _result(match_date, 3, 1),
            ),
        )
        after = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 2, 2),
        )
        self.assertEqual(after.recent_n_matches_a, 2)
        self.assertEqual(after.recent_n_win_rate_a, 0.5)
        self.assertEqual(after.h2h_global_matches, 1)

    def test_last_n_includes_the_entire_boundary_date_block(self) -> None:
        """No inventa un orden intradía al cruzar N en una fecha empatada."""

        state = CausalHistoryState(recent_matches=3)
        dates = (
            date(2024, 3, 1),
            date(2024, 3, 2),
            date(2024, 3, 3),
        )
        for match_date in dates:
            state.apply_date_block(
                match_date,
                (
                    _result(match_date, 1, 10),
                    _result(match_date, 11, 1),
                ),
            )

        snapshot = state.snapshot(
            "M",
            1,
            20,
            surface="Hard",
            as_of_date=date(2024, 3, 4),
        )

        self.assertEqual(snapshot.recent_n_matches_a, 4)
        self.assertEqual(snapshot.recent_n_win_rate_a, 0.5)

    def test_calendar_month_window_clamps_at_leap_day(self) -> None:
        """La ventana de tres meses de 31 de mayo empieza el 29 de febrero."""

        state = CausalHistoryState(
            recent_matches=1,
            recent_months=3,
        )
        february_28 = date(2024, 2, 28)
        february_29 = date(2024, 2, 29)
        state.apply_date_block(
            february_28,
            (_result(february_28, 1, 2),),
        )
        state.apply_date_block(
            february_29,
            (_result(february_29, 3, 1),),
        )

        snapshot = state.snapshot(
            "M",
            1,
            9,
            surface="Hard",
            as_of_date=date(2024, 5, 31),
        )

        self.assertEqual(snapshot.recent_months_matches_a, 1)
        self.assertEqual(snapshot.recent_months_win_rate_a, 0.0)
        self.assertEqual(snapshot.recent_n_matches_a, 1)
        self.assertEqual(snapshot.rest_days_a, 92)

    def test_h2h_is_signed_for_a_and_split_by_surface(self) -> None:
        """Orienta el H2H y mantiene acumuladores por superficie."""

        state = CausalHistoryState()
        first = date(2024, 4, 1)
        second = date(2024, 4, 2)
        third = date(2024, 4, 3)
        state.apply_date_block(first, (_result(first, 1, 2),))
        state.apply_date_block(second, (_result(second, 1, 2),))
        state.apply_date_block(
            third,
            (_result(third, 2, 1, surface="Clay"),),
        )
        cutoff = date(2024, 4, 4)

        hard = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )
        reverse = state.snapshot(
            "M",
            2,
            1,
            surface="Hard",
            as_of_date=cutoff,
        )
        clay = state.snapshot(
            "M",
            1,
            2,
            surface="Clay",
            as_of_date=cutoff,
        )
        unknown_surface = state.snapshot(
            "M",
            1,
            2,
            surface=None,
            as_of_date=cutoff,
        )

        self.assertAlmostEqual(hard.h2h_global_balance, 1.0 / 3.0)
        self.assertEqual(hard.h2h_global_matches, 3)
        self.assertEqual(hard.h2h_surface_balance, 1.0)
        self.assertEqual(hard.h2h_surface_matches, 2)
        self.assertAlmostEqual(reverse.h2h_global_balance, -1.0 / 3.0)
        self.assertEqual(reverse.h2h_surface_balance, -1.0)
        self.assertEqual(clay.h2h_surface_balance, -1.0)
        self.assertEqual(clay.h2h_surface_matches, 1)
        self.assertIsNone(unknown_surface.h2h_surface_balance)
        self.assertEqual(unknown_surface.h2h_surface_matches, 0)

    def test_rest_and_difference_properties_are_oriented_a_minus_b(
        self,
    ) -> None:
        """Calcula descanso y diferencias solo con antecedentes conocidos."""

        state = CausalHistoryState()
        first = date(2024, 5, 1)
        second = date(2024, 5, 5)
        state.apply_date_block(first, (_result(first, 1, 3),))
        state.apply_date_block(second, (_result(second, 2, 4),))

        snapshot = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 5, 10),
        )

        self.assertEqual(snapshot.rest_days_a, 9)
        self.assertEqual(snapshot.rest_days_b, 5)
        self.assertEqual(snapshot.rest_days_diff, 4)
        self.assertEqual(snapshot.recent_n_win_rate_diff, 0.0)
        self.assertEqual(snapshot.recent_months_win_rate_diff, 0.0)

    def test_gender_universes_are_independent_even_with_equal_ids(self) -> None:
        """El mismo player_id en M y F nunca comparte forma ni H2H."""

        match_date = date(2024, 6, 1)
        state = CausalHistoryState()
        state.apply_date_block(
            match_date,
            (
                _result(match_date, 1, 2, gender="M"),
                _result(match_date, 2, 1, gender="F"),
            ),
        )
        cutoff = date(2024, 6, 2)

        male = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )
        female = state.snapshot(
            "F",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )

        self.assertEqual(male.recent_n_win_rate_a, 1.0)
        self.assertEqual(male.h2h_global_balance, 1.0)
        self.assertEqual(female.recent_n_win_rate_a, 0.0)
        self.assertEqual(female.h2h_global_balance, -1.0)

    def test_invalid_date_blocks_fail_before_mutating_state(self) -> None:
        """Rechaza fechas mezcladas, vacíos y reaperturas de una fecha."""

        state = CausalHistoryState()
        first = date(2024, 7, 1)
        second = date(2024, 7, 2)

        with self.assertRaises(HistoryDateOrderError):
            state.apply_date_block(
                first,
                (_result(second, 1, 2),),
            )
        self.assertIsNone(state.last_date)
        with self.assertRaises(ValueError):
            state.apply_date_block(first, ())
        self.assertIsNone(state.last_date)

        state.apply_date_block(first, (_result(first, 1, 2),))
        with self.assertRaises(HistoryDateOrderError):
            state.apply_date_block(first, (_result(first, 3, 4),))
        self.assertEqual(state.last_date, first)

    def test_availability_batch_sorts_effective_dates_without_intraday_order(
        self,
    ) -> None:
        """Un lote tardío usa match_date para forma y una sola disponibilidad."""

        first = date(2024, 8, 1)
        second = date(2024, 8, 3)
        available = date(2024, 8, 10)
        results = (
            _result(second, 2, 1),
            _result(first, 1, 2),
            _result(second, 1, 3),
        )
        forward = CausalHistoryState()
        reverse = CausalHistoryState()

        forward.apply_availability_batch(available, results)
        reverse.apply_availability_batch(available, reversed(results))

        with self.assertRaises(HistoryDateOrderError):
            forward.snapshot(
                "M",
                1,
                2,
                surface="Hard",
                as_of_date=available,
            )
        cutoff = date(2024, 8, 11)
        forward_snapshot = forward.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )
        reverse_snapshot = reverse.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=cutoff,
        )

        self.assertEqual(forward_snapshot, reverse_snapshot)
        self.assertEqual(forward_snapshot.recent_n_matches_a, 3)
        self.assertAlmostEqual(forward_snapshot.recent_n_win_rate_a, 2 / 3)
        self.assertEqual(forward_snapshot.h2h_global_matches, 2)
        self.assertEqual(forward_snapshot.rest_days_a, 8)

    def test_late_arrival_is_inserted_by_match_date_and_only_visible_later(
        self,
    ) -> None:
        """Una llegada futura no reescribe la vista previa capturada."""

        available = date(2024, 9, 10)
        state = CausalHistoryState()
        state.apply_availability_batch(
            available,
            (
                _result(date(2024, 9, 1), 1, 2),
                _result(date(2024, 9, 3), 2, 1),
            ),
        )
        before = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 9, 11),
        )

        late_availability = date(2024, 9, 12)
        state.apply_availability_batch(
            late_availability,
            (_result(date(2024, 9, 2), 1, 2),),
        )
        after = state.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=date(2024, 9, 13),
        )

        self.assertEqual(before.recent_n_matches_a, 2)
        self.assertEqual(before.h2h_global_matches, 2)
        self.assertEqual(after.recent_n_matches_a, 3)
        self.assertAlmostEqual(after.recent_n_win_rate_a, 2 / 3)
        self.assertEqual(after.h2h_global_matches, 3)
        self.assertEqual(after.rest_days_a, 10)
        with self.assertRaises(HistoryDateOrderError):
            state.snapshot(
                "M",
                1,
                2,
                surface="Hard",
                as_of_date=date(2024, 9, 11),
            )
        self.assertEqual(before.recent_n_matches_a, 2)


class HistoricalMatchResultValidationTest(unittest.TestCase):
    """Comprueba que la frontera pública no normaliza valores ambiguos."""

    def test_rejects_noncanonical_or_self_match_values(self) -> None:
        """Falla ante superficie desconocida, género inválido y self-match."""

        with self.assertRaises(ValueError):
            _result(date(2024, 1, 1), 1, 2, surface="hard")
        with self.assertRaises(ValueError):
            _result(date(2024, 1, 1), 1, 2, gender="X")
        with self.assertRaises(ValueError):
            _result(date(2024, 1, 1), 1, 1)


if __name__ == "__main__":
    unittest.main()

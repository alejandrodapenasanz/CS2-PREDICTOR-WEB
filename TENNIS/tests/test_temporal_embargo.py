"""Pruebas del embargo causal para fechas de inicio de torneo Sackmann."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import hashlib
import unittest

from src.elo import EloEngine, EventProvenance, MatchEvent
from src.temporal import (
    DEFAULT_SOURCE_DATE_POLICY,
    SourceDatePolicy,
    SourceDatePolicyError,
)


def _event(match_date: date, row: int, winner: int, loser: int) -> MatchEvent:
    """Construye un evento sintético con procedencia y hash válidos."""

    return MatchEvent(
        date=match_date,
        source_date=match_date,
        gender="M",
        winner_id=winner,
        loser_id=loser,
        surface="Hard",
        tour_level="A",
        score="6-4 6-4",
        provenance=EventProvenance("a" * 40, "atp/test.csv", row),
        source_record_hash=hashlib.sha256(str(row).encode()).hexdigest(),
    )


class SourceDatePolicyTest(unittest.TestCase):
    """Fija fórmula, desigualdad estricta y validación del embargo."""

    def test_default_is_21_days_and_equality_remains_unavailable(self) -> None:
        """Un resultado de D entra por primera vez en D+22."""

        policy = DEFAULT_SOURCE_DATE_POLICY
        source = date(2024, 1, 1)
        self.assertEqual(policy.availability_date(source), date(2024, 1, 22))
        self.assertFalse(policy.is_available(source, date(2024, 1, 22)))
        self.assertTrue(policy.is_available(source, date(2024, 1, 23)))

    def test_policy_round_trips_and_rejects_ambiguous_dates(self) -> None:
        """La configuración publicada reconstruye exactamente el contrato."""

        policy = SourceDatePolicy.from_mapping(
            DEFAULT_SOURCE_DATE_POLICY.as_dict()
        )
        self.assertEqual(policy, DEFAULT_SOURCE_DATE_POLICY)
        with self.assertRaises(SourceDatePolicyError):
            SourceDatePolicy(0)
        with self.assertRaises(SourceDatePolicyError):
            policy.availability_date(datetime(2024, 1, 1))


class OverlappingTournamentEloTest(unittest.TestCase):
    """Demuestra que un torneo solapado no observa resultados embargados."""

    def test_preview_queue_blocks_overlap_and_availability_equality(self) -> None:
        """Jan-15/Jan-22 ven cold start; Jan-23 ya ve el resultado Jan-1."""

        engine = EloEngine()
        first = _event(date(2024, 1, 1), 1, 10, 20)
        overlap = _event(date(2024, 1, 15), 2, 10, 30)
        equality = _event(date(2024, 1, 22), 3, 10, 40)
        after = _event(date(2024, 1, 23), 4, 10, 50)

        self.assertEqual(
            engine.preview_date_block(first.date, (first,)).rated_matches[
                0
            ].winner_before.general_matches,
            0,
        )
        self.assertEqual(
            engine.preview_date_block(overlap.date, (overlap,)).rated_matches[
                0
            ].winner_before.general_matches,
            0,
        )
        self.assertEqual(
            engine.preview_date_block(equality.date, (equality,)).rated_matches[
                0
            ].winner_before.general_matches,
            0,
        )
        available = DEFAULT_SOURCE_DATE_POLICY.availability_date(first.date)
        engine.process_date_block(
            available,
            (replace(first, date=available, source_date=first.date),),
        )
        self.assertEqual(
            engine.preview_date_block(after.date, (after,)).rated_matches[
                0
            ].winner_before.general_matches,
            1,
        )

    def test_identity_evidence_equality_is_not_a_historical_selector(self) -> None:
        """El motor genérico conserva igualdad y activa solo al día siguiente."""

        evidence_available = date(2024, 1, 22)
        engine = EloEngine(
            identity_exclusion_after_dates={("M", 10): evidence_available}
        )
        equality = engine.preview_date_block(
            evidence_available,
            (_event(evidence_available, 5, 10, 20),),
        )
        after = engine.preview_date_block(
            date(2024, 1, 23),
            (_event(date(2024, 1, 23), 6, 10, 30),),
        )
        self.assertEqual(equality.audit.included, 1)
        self.assertEqual(after.audit.count("excluded_identity"), 1)


if __name__ == "__main__":
    unittest.main()


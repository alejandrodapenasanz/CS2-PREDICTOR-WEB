"""Pruebas unitarias offline del motor Elo causal por bloques de fecha."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import math
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import (  # noqa: E402
    DateBlockError,
    EloEngine,
    EloParameters,
    EventColumns,
    EventProvenance,
    MatchEvent,
    UnknownSurfaceError,
    events_from_dataframe,
)


def _event(
    *,
    match_date: date,
    gender: str,
    winner_id: int,
    loser_id: int,
    row: int,
    surface: str | None = "Hard",
    level: str = "A",
    score: str | None = "6-4 6-4",
    path: str = "matches.csv",
    record_hash: str | None = None,
) -> MatchEvent:
    """Construye un evento pequeño con procedencia estable para tests."""

    source_record_hash = record_hash or hashlib.sha256(
        repr(
            (
                match_date,
                gender,
                winner_id,
                loser_id,
                surface,
                level,
                score,
                "2024-001",
                str(row),
                "R32",
            )
        ).encode("utf-8")
    ).hexdigest()
    return MatchEvent(
        date=match_date,
        gender=gender,  # type: ignore[arg-type]
        winner_id=winner_id,
        loser_id=loser_id,
        surface=surface,  # type: ignore[arg-type]
        tour_level=level,
        score=score,
        provenance=EventProvenance(
            source_commit="a" * 40,
            source_path=path,
            source_row=row,
        ),
        source_record_hash=source_record_hash,
        tourney_id="2024-001",
        match_num=str(row),
        round="R32",
    )


class EloParametersTest(unittest.TestCase):
    """Comprueba exactamente el núcleo matemático configurado."""

    def test_defaults_match_documented_formula(self) -> None:
        """Usa R0, escala, K dinámico y mezcla solicitados."""

        parameters = EloParameters()

        self.assertEqual(parameters.initial_rating, 1500.0)
        self.assertEqual(parameters.scale, 400.0)
        self.assertAlmostEqual(
            parameters.k_factor(0),
            250.0 / (5.0**0.4),
        )
        self.assertAlmostEqual(
            parameters.expected_score(1900.0, 1500.0),
            10.0 / 11.0,
        )
        self.assertEqual(parameters.combine(1600.0, 1400.0), 1500.0)


class DataFrameEventsTest(unittest.TestCase):
    """Comprueba normalización y procedencia del adaptador DataFrame."""

    def test_builder_normalises_surface_and_preserves_explicit_rows(
        self,
    ) -> None:
        """Conserva commit/ruta/fila aunque cambie el índice del DataFrame."""

        frame = pd.DataFrame(
            {
                "tourney_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02"]
                ),
                "gender": ["m", "F"],
                "winner_id": pd.array([1, 2], dtype="Int64"),
                "loser_id": pd.array([3, 4], dtype="Int64"),
                "surface": ["hArD", pd.NA],
                "tour_level": ["A", "C"],
                "score": ["6-4 6-4", pd.NA],
                "source_file": ["atp.csv", "wta.csv"],
                "raw_row": [91, 17],
            },
            index=[8, 3],
        )

        events = events_from_dataframe(
            frame,
            source_commit="b" * 40,
            columns=EventColumns(source_row="raw_row"),
        )

        self.assertEqual(events[0].surface, "Hard")
        self.assertIsNone(events[1].surface)
        self.assertEqual(events[0].gender, "M")
        self.assertEqual(events[0].provenance.source_row, 91)
        self.assertEqual(events[1].provenance.source_path, "wta.csv")

    def test_builder_rejects_unknown_surface(self) -> None:
        """Falla ante una superficie poblada fuera del vocabulario."""

        frame = pd.DataFrame(
            {
                "tourney_date": [pd.Timestamp("2024-01-01")],
                "gender": ["M"],
                "winner_id": [1],
                "loser_id": [2],
                "surface": ["Synthetic"],
                "tour_level": ["A"],
                "score": [None],
                "source_file": ["atp.csv"],
            }
        )

        with self.assertRaises(UnknownSurfaceError):
            events_from_dataframe(
                frame,
                source_commit="c" * 40,
            )

    def test_builder_accepts_a_path_object_as_common_source(self) -> None:
        """Serializa una ruta común de ``pathlib`` de forma portable."""

        frame = pd.DataFrame(
            {
                "tourney_date": [pd.Timestamp("2024-01-01")],
                "gender": ["M"],
                "winner_id": [1],
                "loser_id": [2],
                "surface": ["Hard"],
                "tour_level": ["A"],
                "score": ["6-4 6-4"],
            }
        )

        event = events_from_dataframe(
            frame,
            source_commit="c" * 40,
            source_path=Path("raw") / "atp.csv",
        )[0]

        self.assertEqual(event.provenance.source_path, "raw/atp.csv")
        self.assertEqual(event.provenance.source_row, 1)

    def test_builder_hashes_content_but_not_provenance(self) -> None:
        """Da el mismo hash a copias y otro hash al contenido distinto."""

        frame = pd.DataFrame(
            {
                "tourney_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-01"]
                ),
                "gender": ["M", "M", "M"],
                "winner_id": [1, 1, 1],
                "loser_id": [2, 2, 2],
                "surface": ["Hard", "Hard", "Hard"],
                "tour_level": ["A", "A", "A"],
                "score": ["6-4 6-4", "6-4 6-4", "7-6 7-6"],
                "source_file": ["one.csv", "copy.csv", "different.csv"],
            }
        )

        events = events_from_dataframe(
            frame,
            source_commit="d" * 40,
        )

        self.assertEqual(
            events[0].source_record_hash,
            events[1].source_record_hash,
        )
        self.assertNotEqual(
            events[0].source_record_hash,
            events[2].source_record_hash,
        )
        self.assertEqual(
            [event.provenance.source_row for event in events],
            [1, 1, 1],
        )


class EloEngineTest(unittest.TestCase):
    """Comprueba aislamiento, causalidad, filtros y determinismo."""

    def test_gender_and_surface_states_are_isolated(self) -> None:
        """El mismo id en M/F y las superficies no comparten estado."""

        match_date = date(2024, 1, 1)
        engine = EloEngine()
        result = engine.process_date_block(
            match_date,
            (
                _event(
                    match_date=match_date,
                    gender="M",
                    winner_id=1,
                    loser_id=2,
                    row=1,
                    surface="Hard",
                ),
                _event(
                    match_date=match_date,
                    gender="F",
                    winner_id=2,
                    loser_id=1,
                    row=2,
                    surface="Clay",
                    path="wta.csv",
                ),
            ),
        )

        male = engine.snapshot(
            "M",
            1,
            as_of_date=date(2024, 1, 2),
            surface="Hard",
        )
        female = engine.snapshot(
            "F",
            1,
            as_of_date=date(2024, 1, 2),
            surface="Clay",
        )
        untouched_clay = engine.snapshot(
            "M",
            1,
            as_of_date=date(2024, 1, 2),
            surface="Clay",
        )
        self.assertGreater(male.general_elo, 1500.0)
        self.assertLess(female.general_elo, 1500.0)
        self.assertEqual(untouched_clay.surface_elo_raw, 1500.0)
        self.assertEqual(untouched_clay.surface_matches, 0)
        self.assertEqual(result.audit.included, 2)
        self.assertTrue(
            all(
                snapshot.as_of_date == date(2024, 1, 2)
                for snapshot in result.snapshots
            )
        )
        self.assertTrue(
            all(
                snapshot.state_date == match_date
                for snapshot in result.snapshots
            )
        )

    def test_same_date_uses_common_snapshot_and_sums_deltas(self) -> None:
        """Dos partidos del mismo jugador usan n=0 y Elo 1500 pre-D."""

        match_date = date(2024, 2, 1)
        events = (
            _event(
                match_date=match_date,
                gender="M",
                winner_id=1,
                loser_id=2,
                row=1,
            ),
            _event(
                match_date=match_date,
                gender="M",
                winner_id=1,
                loser_id=3,
                row=2,
            ),
        )
        engine = EloEngine()

        block = engine.process_date_block(match_date, reversed(events))

        self.assertEqual(len(block.rated_matches), 2)
        for rated in block.rated_matches:
            self.assertEqual(rated.winner_before.general_matches, 0)
            self.assertEqual(rated.winner_before.general_elo, 1500.0)
            self.assertEqual(
                rated.winner_before.surface_matches,
                0,
            )
        expected_delta = EloParameters().k_factor(0) * 0.5 * 2
        winner = engine.snapshot(
            "M",
            1,
            as_of_date=date(2024, 2, 2),
            surface="Hard",
        )
        self.assertAlmostEqual(
            winner.general_elo,
            1500.0 + expected_delta,
        )
        self.assertAlmostEqual(
            winner.surface_elo_raw or math.nan,
            1500.0 + expected_delta,
        )
        self.assertEqual(winner.general_matches, 2)
        self.assertEqual(winner.surface_matches, 2)

    def test_filters_levels_non_matches_self_and_exact_duplicates(self) -> None:
        """Excluye no iniciados y conserva una retirada ya iniciada."""

        match_date = date(2024, 3, 1)
        included_unknown = _event(
            match_date=match_date,
            gender="M",
            winner_id=1,
            loser_id=2,
            row=1,
            score=None,
        )
        duplicate = MatchEvent(
            **{
                **included_unknown.__dict__,
                "provenance": EventProvenance(
                    source_commit="a" * 40,
                    source_path="duplicate.csv",
                    source_row=1,
                ),
            }
        )
        events = (
            included_unknown,
            duplicate,
            _event(
                match_date=match_date,
                gender="M",
                winner_id=3,
                loser_id=4,
                row=3,
                level="e",
            ),
            _event(
                match_date=match_date,
                gender="M",
                winner_id=5,
                loser_id=6,
                row=4,
                score="6-2 1-0 rEt",
            ),
            _event(
                match_date=match_date,
                gender="M",
                winner_id=7,
                loser_id=7,
                row=5,
            ),
        )
        engine = EloEngine()

        block = engine.process_date_block(match_date, events)

        self.assertEqual(block.audit.received, 5)
        self.assertEqual(block.audit.included, 2)
        self.assertEqual(block.audit.count("duplicate"), 1)
        self.assertEqual(block.audit.count("excluded_level"), 1)
        self.assertEqual(block.audit.count("excluded_status"), 0)
        self.assertEqual(block.audit.count("self_match"), 1)
        self.assertIsNone(block.rated_matches[0].event.score)
        self.assertEqual(block.rated_matches[1].event.score, "6-2 1-0 rEt")

    def test_only_identical_source_content_is_deduplicated(self) -> None:
        """Conserva colisiones deportivas si la fila fuente no es idéntica."""

        match_date = date(2024, 3, 2)
        first = _event(
            match_date=match_date,
            gender="M",
            winner_id=1,
            loser_id=2,
            row=1,
            score="6-4 6-4",
            record_hash="1" * 64,
        )
        exact_copy = MatchEvent(
            **{
                **first.__dict__,
                "provenance": EventProvenance(
                    source_commit="a" * 40,
                    source_path="copy.csv",
                    source_row=1,
                ),
            }
        )
        same_sporting_key_different_content = MatchEvent(
            **{
                **first.__dict__,
                "score": "7-6 7-6",
                "source_record_hash": "2" * 64,
                "provenance": EventProvenance(
                    source_commit="a" * 40,
                    source_path="different.csv",
                    source_row=1,
                ),
            }
        )

        block = EloEngine().process_date_block(
            match_date,
            (
                same_sporting_key_different_content,
                exact_copy,
                first,
            ),
        )

        self.assertEqual(block.audit.included, 2)
        self.assertEqual(block.audit.count("duplicate"), 1)
        self.assertEqual(len(block.rated_matches), 2)

    def test_only_non_started_status_markers_are_excluded(self) -> None:
        """Reconoce todos los marcadores sin depender de capitalización."""

        match_date = date(2024, 4, 1)
        markers = ("W/O", "Walkover", "BYE", "RET", "DEF", "ABD", "ABN")
        events = tuple(
            _event(
                match_date=match_date,
                gender="F",
                winner_id=index * 2 + 1,
                loser_id=index * 2 + 2,
                row=index,
                score=f"6-0 {marker}",
            )
            for index, marker in enumerate(markers, start=1)
        )

        block = EloEngine().process_date_block(match_date, events)

        self.assertEqual(block.audit.included, 4)
        self.assertEqual(
            block.audit.count("excluded_status"),
            3,
        )
        self.assertEqual(
            tuple(match.event.score for match in block.rated_matches),
            ("6-0 RET", "6-0 DEF", "6-0 ABD", "6-0 ABN"),
        )

    def test_identity_quarantine_activates_only_after_prior_evidence(
        self,
    ) -> None:
        """Antes y durante la primera evidencia no aplica una regla futura."""

        conflict_date = date(2024, 4, 2)
        engine = EloEngine(
            identity_exclusion_after_dates={("M", 101): conflict_date}
        )
        before = engine.process_date_block(
            date(2024, 4, 1),
            (
                _event(
                    match_date=date(2024, 4, 1),
                    gender="M",
                    winner_id=101,
                    loser_id=201,
                    row=1,
                ),
            ),
        )
        same_day = engine.process_date_block(
            conflict_date,
            (
                _event(
                    match_date=conflict_date,
                    gender="M",
                    winner_id=101,
                    loser_id=202,
                    row=2,
                ),
            ),
        )
        after = engine.process_date_block(
            date(2024, 4, 3),
            (
                _event(
                    match_date=date(2024, 4, 3),
                    gender="M",
                    winner_id=203,
                    loser_id=101,
                    row=3,
                ),
            ),
        )

        self.assertEqual(before.audit.included, 1)
        self.assertEqual(same_day.audit.included, 1)
        self.assertEqual(after.audit.included, 0)
        self.assertEqual(after.audit.count("excluded_identity"), 1)
        self.assertEqual(after.states, ())

    def test_identity_quarantine_dates_are_strictly_typed(self) -> None:
        """Impide cortes ambiguos con hora o mappings mal formados."""

        with self.assertRaises(ValueError):
            EloEngine(
                identity_exclusion_after_dates={
                    ("M", 101): datetime(2024, 4, 2)
                }
            )
        with self.assertRaises(TypeError):
            EloEngine(  # type: ignore[arg-type]
                identity_exclusion_after_dates={(("M", 101), date.today())}
            )

    def test_null_surface_updates_only_general(self) -> None:
        """Una superficie nula no actualiza ninguno de los cuatro pools."""

        match_date = date(2024, 5, 1)
        engine = EloEngine()
        engine.process_date_block(
            match_date,
            (
                _event(
                    match_date=match_date,
                    gender="F",
                    winner_id=1,
                    loser_id=2,
                    row=1,
                    surface=None,
                ),
            ),
        )

        general = engine.snapshot(
            "F",
            1,
            as_of_date=date(2024, 5, 2),
        )
        carpet = engine.snapshot(
            "F",
            1,
            as_of_date=date(2024, 5, 2),
            surface="Carpet",
        )
        self.assertGreater(general.general_elo, 1500.0)
        self.assertIsNone(general.surface_elo_raw)
        self.assertEqual(general.combined_elo, general.general_elo)
        self.assertEqual(carpet.surface_elo_raw, 1500.0)
        self.assertEqual(carpet.surface_matches, 0)

    def test_reordering_events_produces_identical_state(self) -> None:
        """La procedencia fija keeper y orden de suma independientemente."""

        match_date = date(2024, 6, 1)
        events = tuple(
            _event(
                match_date=match_date,
                gender="M",
                winner_id=1,
                loser_id=opponent,
                row=opponent,
                surface="Grass",
            )
            for opponent in (2, 3, 4)
        )
        first = EloEngine().process_date_block(match_date, events)
        second = EloEngine().process_date_block(
            match_date,
            reversed(events),
        )

        self.assertEqual(first.states, second.states)
        self.assertEqual(first.rated_matches, second.rated_matches)

    def test_date_cannot_be_reopened_or_mixed(self) -> None:
        """Rechaza fechas mezcladas y una segunda llamada para la misma D."""

        first_date = date(2024, 7, 1)
        next_date = date(2024, 7, 2)
        engine = EloEngine()

        with self.assertRaises(DateBlockError):
            engine.process_date_block(
                first_date,
                (
                    _event(
                        match_date=next_date,
                        gender="M",
                        winner_id=1,
                        loser_id=2,
                        row=1,
                    ),
                ),
            )
        engine.process_date_block(
            first_date,
            (
                _event(
                    match_date=first_date,
                    gender="M",
                    winner_id=1,
                    loser_id=2,
                    row=1,
                ),
            ),
        )
        with self.assertRaises(DateBlockError):
            engine.process_date_block(
                first_date,
                (
                    _event(
                        match_date=first_date,
                        gender="M",
                        winner_id=3,
                        loser_id=4,
                        row=2,
                    ),
                ),
            )

    def test_unrepresentable_post_date_does_not_mutate_state(self) -> None:
        """Falla antes de publicar estado si D+1 no es representable."""

        engine = EloEngine()
        with self.assertRaises(DateBlockError):
            engine.process_date_block(
                date.max,
                (
                    _event(
                        match_date=date.max,
                        gender="M",
                        winner_id=1,
                        loser_id=2,
                        row=1,
                    ),
                ),
            )

        self.assertIsNone(engine.last_date)
        valid_date = date(2024, 8, 1)
        block = engine.process_date_block(
            valid_date,
            (
                _event(
                    match_date=valid_date,
                    gender="M",
                    winner_id=1,
                    loser_id=2,
                    row=1,
                ),
            ),
        )
        self.assertEqual(block.audit.included, 1)


if __name__ == "__main__":
    unittest.main()

"""Pruebas del catálogo causal de superficies exactas."""

from __future__ import annotations

from datetime import UTC, date, datetime
import unittest

from src.surface_catalog import (
    Surface,
    SurfaceCatalog,
    SurfaceEvidence,
    SurfaceSource,
)


MATCH_DATE = date(2026, 8, 20)


def _evidence(
    *,
    evidence_id: str = "evidence-hard",
    source_family: SurfaceSource = "tennisratio",
    tournament: str = "Exact Open",
    tournament_href: str | None = None,
    surface: Surface = "Hard",
    captured: datetime = datetime(2026, 8, 10, 8, tzinfo=UTC),
) -> SurfaceEvidence:
    """Crea evidencia sintética sin usar heurísticas de nombres."""

    return SurfaceEvidence(
        evidence_id=evidence_id,
        source_family=source_family,
        evidence_kind="direct_agenda",
        source_match_id=f"match:{evidence_id}",
        gender="M",
        edition_year=2026,
        tournament=tournament,
        tournament_href=tournament_href,
        tour_level="ATP",
        surface=surface,
        effective_date=date(2026, 8, 10),
        captured_at_utc=captured,
        source_url="https://example.test/source",
        source_sha256="a" * 64,
    )


def _catalog(*items: SurfaceEvidence) -> SurfaceCatalog:
    """Construye un índice inmutable con fingerprint explícito."""

    return SurfaceCatalog(
        as_of_date=MATCH_DATE,
        evidence=tuple(items),
        fingerprint="f" * 64,
        source_counts={},
    )


class SurfaceCatalogTest(unittest.TestCase):
    """Fija exactitud, causalidad y rechazo ante ambigüedad."""

    def test_exact_same_edition_propagates_surface(self) -> None:
        """Una publicación previa de la misma edición resuelve la superficie."""

        resolution = _catalog(_evidence()).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
        )

        self.assertEqual(resolution.surface, "Hard")
        self.assertEqual(resolution.method, "same_edition_propagation")
        self.assertEqual(resolution.evidence_id, "evidence-hard")

    def test_direct_capture_during_match_day_is_never_used(self) -> None:
        """La granularidad diaria rechaza una superficie capturada durante D."""

        resolution = _catalog().resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
            direct_surface="Clay",
            direct_captured_at_utc=datetime(2026, 8, 20, 7, tzinfo=UTC),
        )

        self.assertIsNone(resolution.surface)
        self.assertEqual(resolution.reason, "no_exact_pre_cutoff_surface_evidence")

    def test_conflict_indoors_and_aggregate_are_unresolved(self) -> None:
        """No elige entre conflictos ni deduce desde etiquetas ambiguas."""

        conflict = _catalog(
            _evidence(),
            _evidence(evidence_id="evidence-clay", surface="Clay"),
        ).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
        )
        indoors = _catalog().resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:indoors",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Indoor City",
            tournament_href=None,
            direct_surface="Indoors",
            direct_captured_at_utc=datetime(2026, 8, 10, 8, tzinfo=UTC),
        )
        aggregate = _catalog(
            _evidence(tournament="Futures"),
        ).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:futures",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Futures",
            tournament_href=None,
        )

        self.assertEqual(conflict.reason, "conflicting_surface_same_tournament_edition")
        self.assertIsNone(indoors.surface)
        self.assertEqual(aggregate.reason, "aggregate_or_missing_exact_tournament_identity")

    def test_city_or_similar_name_never_matches(self) -> None:
        """Una etiqueta parecida no comparte identidad exacta de torneo."""

        resolution = _catalog(_evidence(tournament="Madrid Open")).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:other",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Madrid",
            tournament_href=None,
        )

        self.assertIsNone(resolution.surface)

    def test_future_evidence_cannot_change_past_resolution(self) -> None:
        """Añadir una captura posterior a D deja intacta la resolución as-of."""

        baseline = _catalog(_evidence()).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
        )
        extended = _catalog(
            _evidence(),
            _evidence(
                evidence_id="future-conflict",
                surface="Clay",
                captured=datetime(2026, 8, 21, 8, tzinfo=UTC),
            ),
        ).resolve_for_prediction(
            source_family="tennisratio",
            source_match_id="tennisratio:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
        )

        self.assertEqual(extended, baseline)

    def test_explorer_requires_exact_href_with_edition_year(self) -> None:
        """Explorer no usa un nombre si falta la ficha exacta de la edición."""

        evidence = _evidence(
            source_family="tennis_explorer",
            tournament_href="/montreal/2026/atp-men/",
        )
        resolved = _catalog(evidence).resolve_for_prediction(
            source_family="tennis_explorer",
            source_match_id="explorer:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href="/montreal/2026/atp-men/",
        )
        missing_href = _catalog(evidence).resolve_for_prediction(
            source_family="tennis_explorer",
            source_match_id="explorer:today",
            gender="M",
            match_date=MATCH_DATE,
            tournament="Exact Open",
            tournament_href=None,
        )

        self.assertEqual(resolved.surface, "Hard")
        self.assertIsNone(missing_href.surface)


if __name__ == "__main__":
    unittest.main()

"""Pruebas del handoff causal multifuente del Elo persistido."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import EloEngine  # noqa: E402
from src.elo.operational import (  # noqa: E402
    EloHandoffConfig,
    ExplorerLoadResult,
    ExplorerOperationalResult,
    OperationalEloOverlay,
    assemble_operational_overlay,
    build_operational_elo_overlay_with_fallback,
)
from src.elo.types import Gender  # noqa: E402
from src.surface_catalog import SurfaceCatalog  # noqa: E402
from src.tennisratio import MappedResult  # noqa: E402


CUTOFF = date(2026, 6, 2)
AS_OF = date(2026, 8, 31)
BASE_COMMIT = "8" * 40


def _config() -> EloHandoffConfig:
    """Construye un handoff sintético sin acceder al filesystem."""

    placeholder = PROJECT_ROOT / "unused.sqlite3"
    return EloHandoffConfig(
        schema_version=1,
        base_source_repository="fixture/repository",
        base_source_commit=BASE_COMMIT,
        base_manifest_path=PROJECT_ROOT / "unused.json",
        cutoff_date=CUTOFF,
        tennisratio_database_path=placeholder,
        operations_database_path=placeholder,
        player_mapping_database_path=placeholder,
        source_precedence=("tennisratio", "tennis_explorer"),
    )


def _ratio_result(
    *,
    canonical_id: str = "ratio:1",
    effective_date: date = date(2026, 8, 20),
    available_date: date = date(2026, 8, 21),
    winner_id: int = 1,
    loser_id: int = 2,
) -> MappedResult:
    """Crea un resultado TennisRatio mapeado y terminal."""

    return MappedResult(
        canonical_match_id=canonical_id,
        gender="M",
        effective_date=effective_date,
        available_date=available_date,
        first_seen_at_utc=datetime.combine(
            available_date,
            datetime.min.time(),
            tzinfo=UTC,
        ),
        tournament="Fixture Open",
        tour_level="ATP",
        surface="Hard",
        round="R32",
        winner_sackmann_id=winner_id,
        loser_sackmann_id=loser_id,
        winner_source_key="tennisratio:winner",
        loser_source_key="tennisratio:loser",
        winner_name="Winner",
        loser_name="Loser",
        score="6-4 6-4",
        winner_sets_won=2,
        loser_sets_won=0,
        source_url="https://example.test/ratio",
        source_sha256="a" * 64,
    )


def _explorer_result(
    *,
    source_match_id: str = "explorer:1",
    effective_date: date = date(2026, 8, 20),
    available_date: date = date(2026, 8, 21),
    winner_id: int = 1,
    loser_id: int = 2,
) -> ExplorerOperationalResult:
    """Crea un resultado Tennis Explorer con ambos IDs resueltos."""

    observed = datetime.combine(
        available_date,
        datetime.min.time(),
        tzinfo=UTC,
    )
    return ExplorerOperationalResult(
        source_match_id=source_match_id,
        gender="M",
        effective_date=effective_date,
        available_date=available_date,
        observed_at_utc=observed,
        tournament="Fixture Open",
        tour_level="ATP",
        winner_id=winner_id,
        loser_id=loser_id,
        score="2-0",
        observation_id=f"observation:{source_match_id}",
        payload_sha256="b" * 64,
    )


def _assemble(
    *,
    ratio: tuple[MappedResult, ...] = (),
    ratio_keys: tuple[tuple[Gender, date, str], ...] = (),
    explorer: tuple[ExplorerOperationalResult, ...] = (),
    explorer_eligible: int | None = None,
    surface_catalog: SurfaceCatalog | None = None,
) -> OperationalEloOverlay:
    """Invoca el ensamblador puro con contadores coherentes."""

    return assemble_operational_overlay(
        config=_config(),
        as_of_date=AS_OF,
        ratio_results=ratio,
        ratio_eligible_keys=ratio_keys,
        explorer_load=ExplorerLoadResult(
            results=explorer,
            eligible_records=(len(explorer) if explorer_eligible is None else explorer_eligible),
            omissions=(),
        ),
        surface_catalog=surface_catalog,
    )


class OperationalEloOverlayTest(unittest.TestCase):
    """Valida causalidad, mapping, precedencia y fallback."""

    def test_post_cutoff_result_updates_general_and_exact_surface_elo(self) -> None:
        """Una superficie publicada en el resultado actualiza ambos ratings."""

        result = _ratio_result()
        overlay = _assemble(
            ratio=(result,),
            ratio_keys=(("M", result.effective_date, result.canonical_match_id),),
        )
        self.assertEqual(len(overlay.events), 1)
        event = overlay.events[0]
        self.assertEqual(event.surface, "Hard")
        engine = EloEngine()
        engine.process_date_block(event.date, overlay.events)
        snapshot = engine.snapshot(
            "M",
            1,
            as_of_date=date(2026, 8, 22),
            surface="Hard",
        )
        self.assertGreater(snapshot.general_elo, 1500.0)
        self.assertEqual(snapshot.general_matches, 1)
        self.assertEqual(snapshot.surface_matches, 1)
        self.assertGreater(snapshot.surface_elo_raw or 0.0, 1500.0)
        self.assertEqual(overlay.surface_resolved_events, 1)
        self.assertEqual(overlay.surface_resolution_counts, {"direct_result": 1})

    def test_unmapped_ratio_record_is_omitted_and_logged(self) -> None:
        """Una clave sin resultado mapeado no inventa los player_id."""

        overlay = _assemble(ratio_keys=(("F", date(2026, 8, 20), "ratio:unmapped"),))
        self.assertEqual(overlay.events, ())
        self.assertEqual(len(overlay.omissions), 1)
        self.assertEqual(
            overlay.omissions[0].reason,
            "incomplete_or_ambiguous_mapping",
        )

    def test_ratio_precedence_counts_shared_match_once(self) -> None:
        """El mismo par/fecha en ambas fuentes conserva solo TennisRatio."""

        ratio = _ratio_result()
        explorer = _explorer_result()
        overlay = _assemble(
            ratio=(ratio,),
            ratio_keys=(("M", ratio.effective_date, ratio.canonical_match_id),),
            explorer=(explorer,),
        )
        self.assertEqual(len(overlay.events), 1)
        self.assertEqual(overlay.overlaps_resolved, 1)
        self.assertEqual(
            overlay.events[0].provenance.source_commit,
            "tennisratio-operational-v1",
        )

    def test_cutoff_and_same_day_results_never_enter(self) -> None:
        """Rechaza fecha <= corte y disponibilidad igual al as-of."""

        at_cutoff = _ratio_result(
            canonical_id="ratio:cutoff",
            effective_date=CUTOFF,
            available_date=date(2026, 6, 3),
        )
        seen_today = _ratio_result(
            canonical_id="ratio:today",
            effective_date=date(2026, 8, 30),
            available_date=AS_OF,
        )
        overlay = _assemble(
            ratio=(at_cutoff, seen_today),
            ratio_keys=(
                ("M", at_cutoff.effective_date, at_cutoff.canonical_match_id),
                ("M", seen_today.effective_date, seen_today.canonical_match_id),
            ),
        )
        self.assertEqual(overlay.events, ())
        self.assertTrue(
            all(
                item.reason == "outside_strict_handoff_or_availability"
                for item in overlay.omissions
            )
        )

    def test_same_inputs_produce_identical_fingerprint_and_rating(self) -> None:
        """El ensamblado y el Elo son reproducibles para la misma evidencia."""

        ratio = _ratio_result()
        ratio_keys = (("M", ratio.effective_date, ratio.canonical_match_id),)
        first = _assemble(ratio=(ratio,), ratio_keys=ratio_keys)
        second = _assemble(ratio=(ratio,), ratio_keys=ratio_keys)
        self.assertEqual(first.fingerprint, second.fingerprint)
        engines = (EloEngine(), EloEngine())
        snapshots = []
        for engine, overlay in zip(engines, (first, second), strict=True):
            engine.process_date_block(overlay.events[0].date, overlay.events)
            snapshots.append(
                engine.snapshot(
                    "M",
                    1,
                    as_of_date=date(2026, 8, 22),
                    surface=None,
                )
            )
        self.assertEqual(snapshots[0], snapshots[1])

    def test_surface_catalog_fingerprint_is_part_of_overlay_identity(self) -> None:
        """Cambiar el catálogo cambia el contrato aunque los resultados coincidan."""

        ratio = _ratio_result()
        keys = (("M", ratio.effective_date, ratio.canonical_match_id),)
        catalogs = tuple(
            SurfaceCatalog(
                as_of_date=AS_OF,
                evidence=(),
                fingerprint=marker * 64,
                source_counts={},
            )
            for marker in ("a", "b")
        )
        first = _assemble(
            ratio=(ratio,),
            ratio_keys=keys,
            surface_catalog=catalogs[0],
        )
        second = _assemble(
            ratio=(ratio,),
            ratio_keys=keys,
            surface_catalog=catalogs[1],
        )

        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.events, second.events)

    def test_appending_future_result_cannot_change_current_overlay(self) -> None:
        """An observation outside the as-of cut leaves events and Elo intact."""

        current = _ratio_result()
        future = _ratio_result(
            canonical_id="ratio:future",
            effective_date=AS_OF,
            available_date=date(2026, 9, 1),
            winner_id=2,
            loser_id=1,
        )
        current_keys: tuple[tuple[Gender, date, str], ...] = (
            ("M", current.effective_date, current.canonical_match_id),
        )
        baseline = _assemble(ratio=(current,), ratio_keys=current_keys)
        extended = _assemble(
            ratio=(current, future),
            ratio_keys=(
                *current_keys,
                ("M", future.effective_date, future.canonical_match_id),
            ),
        )

        self.assertEqual(extended.events, baseline.events)
        self.assertEqual(extended.fingerprint, baseline.fingerprint)

    def test_overlay_failure_returns_sackmann_only_contract(self) -> None:
        """Un fallo de la fuente no rompe el build ni conserva eventos parciales."""

        def failing_builder(**_: object) -> OperationalEloOverlay:
            """Simula un fallo antes de completar las dos fuentes."""

            raise OSError("fixture unavailable")

        overlay = build_operational_elo_overlay_with_fallback(
            config=_config(),
            as_of_date=AS_OF,
            builder=failing_builder,
        )
        self.assertEqual(overlay.status, "fallback_sackmann_only")
        self.assertEqual(overlay.events, ())
        self.assertIn("fixture unavailable", overlay.failure or "")


if __name__ == "__main__":
    unittest.main()

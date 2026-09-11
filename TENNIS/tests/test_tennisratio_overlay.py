"""Regression tests for the causal TennisRatio serving overlay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daily_pipeline.tennisratio_overlay import (  # noqa: E402
    TennisRatioOverlay,
    TennisRatioOverlayError,
    build_tennisratio_overlay,
)
from src.daily_pipeline.context import build_daily_feature_context  # noqa: E402
from src.elo import (  # noqa: E402
    DEFAULT_ELO_PARAMETERS,
    EloQuery,
    Gender,
    PlayerEloState,
)
from src.features import CausalHistoryState, RankingSnapshot  # noqa: E402
from src.tennisratio import MappedRanking, MappedResult  # noqa: E402
from src.temporal import DEFAULT_SOURCE_DATE_POLICY  # noqa: E402


BASE_DATE = date(2026, 6, 23)
BASE_EFFECTIVE_CUTOFF = date(2026, 6, 2)
PREDICTION_DATE = date(2026, 8, 23)
AVAILABLE_AT = datetime(2026, 8, 22, 6, tzinfo=UTC)


class _BaseRanking:
    """Minimal causal ranking provider used by the overlay tests."""

    def __init__(self, gender: Gender = "M") -> None:
        """Keep the fake gender universe explicit."""

        self.gender: Gender = gender
        self.max_ranking_date = date(2026, 6, 8)

    def get_many(
        self,
        player_ids: Iterable[int],
        as_of_date: date,
    ) -> tuple[RankingSnapshot, ...]:
        """Return a stable causal base rank for each requested player."""

        ids = tuple(player_ids)
        return tuple(
            RankingSnapshot(
                gender=self.gender,
                player_id=player_id,
                as_of_date=as_of_date,
                ranking_date=self.max_ranking_date,
                rank=100 + player_id,
                points=500,
                is_missing=False,
                ranking_age_days=(as_of_date - self.max_ranking_date).days,
                conflict_dates_skipped=0,
            )
            for player_id in ids
        )


class _BaseEloStore:
    """Return deterministic persisted states for every requested player."""

    def __init__(self, gender: Gender = "M", *, fail: bool = False) -> None:
        """Configure the fake state universe or an internal failure."""

        self.gender: Gender = gender
        self.fail = fail

    def fetch_ratings_before(self, **kwargs: object) -> tuple[PlayerEloState, ...]:
        """Return one base state per requested player."""

        if self.fail:
            raise ValueError("broken persisted Elo state")
        queries = cast(tuple[tuple[int, date], ...], kwargs["queries"])
        return tuple(
            PlayerEloState(
                gender=self.gender,
                player_id=player_id,
                state_date=BASE_DATE,
                general_elo=1500.0,
                hard_elo=1500.0,
                clay_elo=1500.0,
                grass_elo=1500.0,
                carpet_elo=1500.0,
                general_matches=10,
                hard_matches=4,
                clay_matches=3,
                grass_matches=2,
                carpet_matches=1,
            )
            for player_id, _ in queries
        )


def _result(
    *,
    gender: Gender = "M",
    canonical_id: str = "ratio-match-1",
    effective_date: date = date(2026, 8, 20),
    available_date: date = date(2026, 8, 22),
    score: str = "6-4 6-4",
) -> MappedResult:
    """Create one mapped result with configurable causal dates."""

    return MappedResult(
        canonical_match_id=canonical_id,
        gender=gender,
        effective_date=effective_date,
        available_date=available_date,
        first_seen_at_utc=datetime.combine(
            available_date,
            datetime.min.time(),
            tzinfo=UTC,
        ),
        tournament="Cincinnati",
        tour_level="ATP",
        surface="Hard",
        round="R32",
        winner_sackmann_id=1,
        loser_sackmann_id=2,
        winner_source_key="One-A",
        loser_source_key="Two-B",
        winner_name="One A",
        loser_name="Two B",
        score=score,
        winner_sets_won=2,
        loser_sets_won=0,
        source_url="https://www.tennisratio.com/players/One-A.html",
        source_sha256="a" * 64,
    )


def _ranking(
    *,
    gender: Gender = "M",
    rank: int = 7,
    effective_date: date = date(2026, 8, 21),
    source_player_key: str = "One-A",
    source_sha256: str = "b" * 64,
) -> MappedRanking:
    """Create one mapped ranking observation for the requested universe."""

    return MappedRanking(
        gender=gender,
        sackmann_player_id=1,
        source_player_key=source_player_key,
        player_name="One A",
        rank=rank,
        effective_date=effective_date,
        available_date=date(2026, 8, 22),
        first_seen_at_utc=AVAILABLE_AT,
        source_url=(f"https://www.tennisratio.com/players/{source_player_key}.html"),
        source_sha256=source_sha256,
    )


def _build(
    database_path: Path,
    *,
    as_of_date: date,
    results: tuple[MappedResult, ...],
    rankings: tuple[MappedRanking, ...] = (),
    gender: Gender = "M",
    ranking_index: object | None = None,
    elo_store: object | None = None,
    history_state: CausalHistoryState | None = None,
    observed_cutoffs: list[date] | None = None,
    elo_base_date: date = BASE_DATE,
) -> tuple[TennisRatioOverlay, CausalHistoryState]:
    """Build an overlay against injected, as-of-filtered sidecar rows."""

    history = history_state or CausalHistoryState()

    def visible_results(*args: object, **kwargs: object) -> tuple[MappedResult, ...]:
        del args
        cutoffs = kwargs["base_cutoff_by_gender"]
        cutoff = cutoffs[gender]  # type: ignore[index]
        if observed_cutoffs is not None:
            observed_cutoffs.append(cutoff)
        return tuple(
            item
            for item in results
            if item.gender == gender
            and item.effective_date > cutoff
            and item.effective_date < as_of_date
            and item.available_date < as_of_date
        )

    def visible_rankings(*args: object, **kwargs: object) -> tuple[MappedRanking, ...]:
        del args, kwargs
        return tuple(
            item
            for item in rankings
            if item.gender == gender
            and item.effective_date < as_of_date
            and item.available_date < as_of_date
        )

    with (
        mock.patch(
            "src.daily_pipeline.tennisratio_overlay.load_mapped_results",
            side_effect=visible_results,
        ),
        mock.patch(
            "src.daily_pipeline.tennisratio_overlay.load_mapped_rankings",
            side_effect=visible_rankings,
        ),
    ):
        overlay = build_tennisratio_overlay(
            gender=gender,
            as_of_date=as_of_date,
            base_result_effective_cutoff=BASE_EFFECTIVE_CUTOFF,
            base_history_effective_max_date=BASE_EFFECTIVE_CUTOFF,
            base_history_available_max_date=BASE_DATE,
            target_player_ids=(1, 2),
            history_state=history,
            ranking_index=(ranking_index or _BaseRanking(gender)),  # type: ignore[arg-type]
            elo_store=(elo_store or _BaseEloStore(gender)),  # type: ignore[arg-type]
            elo_run_id="base-run",
            elo_base_date=elo_base_date,
            elo_parameters=DEFAULT_ELO_PARAMETERS.as_dict(),
            source_commit="f" * 40,
            database_path=database_path,
        )
    return overlay, history


class TennisRatioOverlayTests(unittest.TestCase):
    """Lock D+1 visibility and future-append invariance."""

    def test_visible_result_advances_history_elo_and_ranking(self) -> None:
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(),),
                rankings=(_ranking(),),
            )

        snapshot = history.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=PREDICTION_DATE,
        )
        self.assertEqual(snapshot.h2h_global_matches, 1)
        self.assertEqual(overlay.history_available_max_date, date(2026, 8, 22))
        self.assertEqual(overlay.ranking_max_date, date(2026, 8, 21))
        self.assertEqual(overlay.supplemental_results, 1)
        self.assertEqual(overlay.supplemental_rankings, 1)
        self.assertRegex(overlay.overlay_fingerprint or "", r"^[0-9a-f]{64}$")
        self.assertIsNotNone(overlay.elo_provider)
        elo = overlay.elo_provider.get_many(  # type: ignore[union-attr]
            (EloQuery("M", 1, "Hard", PREDICTION_DATE),)
        )[0]
        self.assertEqual(elo.general_matches, 11)
        ranking = overlay.ranking_provider.get_many((1,), PREDICTION_DATE)[0]
        self.assertEqual(ranking.rank, 7)
        self.assertIsNone(ranking.points)

    def test_same_day_observation_is_not_visible(self) -> None:
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=date(2026, 8, 22),
                results=(_result(),),
                rankings=(_ranking(),),
            )

        self.assertEqual(overlay.supplemental_results, 0)
        self.assertEqual(overlay.supplemental_rankings, 0)
        self.assertEqual(history.last_date, None)

    def test_persisted_elo_result_still_advances_rebuilt_history(self) -> None:
        """Persisted Elo rows are not replayed, but history is not discarded."""

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(),),
                elo_base_date=date(2026, 8, 22),
            )

        self.assertIsNone(overlay.elo_provider)
        self.assertEqual(overlay.supplemental_results, 1)
        snapshot = history.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=PREDICTION_DATE,
        )
        self.assertEqual(snapshot.h2h_global_matches, 1)

    def test_walkover_does_not_advance_elo_or_history(self) -> None:
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(score="W/O"),),
            )

        self.assertEqual(history.last_date, None)
        self.assertIsNone(overlay.elo_provider)

    def test_appending_future_rows_cannot_change_earlier_snapshot(self) -> None:
        future = _result(
            canonical_id="future",
            effective_date=date(2026, 8, 23),
            available_date=date(2026, 8, 24),
        )
        snapshots = []
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            for rows in ((_result(),), (_result(), future)):
                overlay, history = _build(
                    database_path,
                    as_of_date=PREDICTION_DATE,
                    results=rows,
                )
                history_snapshot = history.snapshot(
                    "M",
                    1,
                    2,
                    surface="Hard",
                    as_of_date=PREDICTION_DATE,
                )
                elo_snapshot = overlay.elo_provider.get_many(  # type: ignore[union-attr]
                    (EloQuery("M", 1, "Hard", PREDICTION_DATE),)
                )[0]
                snapshots.append(
                    (
                        history_snapshot,
                        elo_snapshot,
                        overlay.overlay_fingerprint,
                    )
                )

        self.assertEqual(snapshots[0], snapshots[1])

    def test_effective_cutoff_preserves_the_base_availability_gap(self) -> None:
        """A late-observed result after the base fact cutoff fills the gap."""

        cutoffs: list[date] = []
        gap_result = _result(
            effective_date=date(2026, 6, 10),
            available_date=date(2026, 8, 22),
        )
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(gap_result,),
                observed_cutoffs=cutoffs,
            )

        self.assertEqual(cutoffs, [BASE_EFFECTIVE_CUTOFF])
        self.assertEqual(
            overlay.history_effective_max_date,
            gap_result.effective_date,
        )
        snapshot = history.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=PREDICTION_DATE,
        )
        self.assertEqual(snapshot.h2h_global_matches, 1)
        elo = overlay.elo_provider.get_many(  # type: ignore[union-attr]
            (EloQuery("M", 1, "Hard", PREDICTION_DATE),)
        )[0]
        self.assertEqual(elo.general_matches, 11)

    def test_daily_context_passes_distinct_effective_and_availability_cutoffs(
        self,
    ) -> None:
        """Context wiring cannot regress to using the embargo as overlap."""

        base_parameters = DEFAULT_ELO_PARAMETERS.as_dict()
        parameters = dict(base_parameters)
        source_commit = "f" * 40
        persisted_handoff_cutoff = date(2026, 6, 5)
        parameters["operational_overlay"] = {
            "schema": "tennis-operational-elo-v2",
            "base_source_commit": source_commit,
            "cutoff_date": persisted_handoff_cutoff.isoformat(),
        }
        source_manifest = SimpleNamespace(
            source_commit=source_commit,
            raw_payload={"elo_algorithm_version": "elo-v1"},
            fingerprint="a" * 64,
            source_date_policy=DEFAULT_SOURCE_DATE_POLICY.as_dict(),
        )
        metadata = SimpleNamespace(
            gender="M",
            max_date=BASE_EFFECTIVE_CUTOFF,
            rows=10,
        )
        elo_run = SimpleNamespace(
            source_commit=source_commit,
            algorithm_version="elo-v1",
            input_fingerprint="d" * 64,
            parameters=parameters,
            max_event_date=BASE_DATE,
            run_id="base-run",
        )
        elo_contract = {
            "input_fingerprint": "b" * 64,
            "algorithm_version": elo_run.algorithm_version,
            "parameters": base_parameters,
            "source_date_policy": DEFAULT_SOURCE_DATE_POLICY.as_dict(),
            "historical_identity_exclusion": "disabled",
        }
        ranking_index = _BaseRanking()
        history = CausalHistoryState()
        overlay = TennisRatioOverlay(
            elo_provider=None,
            ranking_provider=ranking_index,  # type: ignore[arg-type]
            overlay_fingerprint="c" * 64,
            history_effective_max_date=BASE_EFFECTIVE_CUTOFF,
            history_available_max_date=BASE_DATE,
            ranking_max_date=ranking_index.max_ranking_date,
            supplemental_results=0,
            supplemental_rankings=0,
        )
        overlay_builder = mock.Mock(return_value=overlay)
        fake_store = mock.Mock()
        fake_store.resolve_complete_run.return_value = elo_run
        with (
            mock.patch(
                "src.daily_pipeline.context.load_feature_source_manifest",
                return_value=source_manifest,
            ),
            mock.patch(
                "src.daily_pipeline.context.verify_auxiliary_source_inventory",
                return_value=SimpleNamespace(source_commit=source_commit),
            ),
            mock.patch(
                "src.daily_pipeline.context.load_identity_quarantine",
                return_value=SimpleNamespace(keys=frozenset()),
            ),
            mock.patch(
                "src.daily_pipeline.context._manifest_quarantine_keys",
                return_value=frozenset(),
            ),
            mock.patch(
                "src.daily_pipeline.context._feature_parameters",
                return_value=SimpleNamespace(
                    age_reference_years=18.0,
                    days_per_year=365.25,
                ),
            ),
            mock.patch(
                "src.daily_pipeline.context._source_date_policy",
                return_value=DEFAULT_SOURCE_DATE_POLICY,
            ),
            mock.patch(
                "src.daily_pipeline.context.PlayerAgeIndex.from_raw",
                return_value=object(),
            ),
            mock.patch(
                "src.daily_pipeline.context.EloStore",
                return_value=fake_store,
            ),
            mock.patch(
                "src.daily_pipeline.context.verify_training_dataset",
                return_value=metadata,
            ),
            mock.patch("src.daily_pipeline.context._validate_model_source"),
            mock.patch(
                "src.daily_pipeline.context._elo_contract",
                return_value=elo_contract,
            ),
            mock.patch(
                "src.daily_pipeline.context.RankingIndex.from_raw",
                return_value=ranking_index,
            ),
            mock.patch(
                "src.daily_pipeline.context._read_target_history",
                return_value=(),
            ),
            mock.patch(
                "src.daily_pipeline.context.rebuild_history_state",
                return_value=history,
            ),
            mock.patch(
                "src.daily_pipeline.context.build_tennisratio_overlay",
                overlay_builder,
            ),
            mock.patch(
                "src.daily_pipeline.context.MatchFeatureBuilder",
                return_value=object(),
            ),
        ):
            context = build_daily_feature_context(
                as_of_date=PREDICTION_DATE,
                player_ids_by_gender={"M": {1, 2}},
                models={"M": mock.Mock()},  # type: ignore[arg-type]
                raw_dir=Path("raw"),
                sackmann_manifest_path=Path("sackmann.json"),
                feature_manifest_path=Path("features.json"),
                elo_database_path=Path("elo.sqlite3"),
                identity_quarantine_path=Path("quarantine.csv"),
                identity_quarantine_manifest_path=Path("quarantine.manifest.json"),
                tennisratio_database_path=Path("ratio.sqlite3"),
            )

        kwargs = overlay_builder.call_args.kwargs
        self.assertEqual(
            kwargs["base_result_effective_cutoff"],
            persisted_handoff_cutoff,
        )
        self.assertEqual(kwargs["elo_base_date"], BASE_DATE)
        self.assertEqual(context.by_gender["M"].overlay_fingerprint, "c" * 64)

    def test_result_aliases_are_counted_once_after_mapping(self) -> None:
        """Different source aliases cannot double-advance Elo or history."""

        alias = replace(
            _result(),
            canonical_match_id="ratio-match-alias",
            winner_source_key="One-Player",
            loser_source_key="Two-Player",
            source_url=("https://www.tennisratio.com/players/One-Player.html"),
            source_sha256="c" * 64,
        )
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(), alias),
            )

        self.assertEqual(overlay.supplemental_results, 1)
        snapshot = history.snapshot(
            "M",
            1,
            2,
            surface="Hard",
            as_of_date=PREDICTION_DATE,
        )
        self.assertEqual(snapshot.h2h_global_matches, 1)
        elo = overlay.elo_provider.get_many(  # type: ignore[union-attr]
            (EloQuery("M", 1, "Hard", PREDICTION_DATE),)
        )[0]
        self.assertEqual(elo.general_matches, 11)

    def test_conflicting_result_aliases_fail_before_history_mutation(self) -> None:
        """Two outcomes for one mapped sporting identity are quarantined."""

        conflict = replace(
            _result(),
            canonical_match_id="ratio-match-conflict",
            winner_sackmann_id=2,
            loser_sackmann_id=1,
            winner_source_key="Two-B",
            loser_source_key="One-A",
            source_sha256="d" * 64,
        )
        history = CausalHistoryState()
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            with self.assertRaises(TennisRatioOverlayError):
                _build(
                    database_path,
                    as_of_date=PREDICTION_DATE,
                    results=(_result(), conflict),
                    history_state=history,
                )

        self.assertIsNone(history.last_date)

    def test_metadata_conflict_is_excluded_without_disabling_overlay(self) -> None:
        """Un alias de torneo ambiguo no inventa contexto ni bloquea el resto."""

        metadata_alias = replace(
            _result(),
            canonical_match_id="ratio-match-metadata-alias",
            tournament="Test Open Challenger",
            source_sha256="e" * 64,
        )
        history = CausalHistoryState()
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, _ = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(), metadata_alias),
                history_state=history,
            )

        self.assertEqual(overlay.supplemental_results, 0)
        self.assertIsNone(overlay.elo_provider)
        self.assertIsNone(history.last_date)

    def test_conflicting_same_date_rankings_are_not_sha_tiebroken(self) -> None:
        """Rank disagreement for one mapped player/date fails closed."""

        rankings = (
            _ranking(rank=7, source_sha256="f" * 64),
            _ranking(
                rank=8,
                source_player_key="One-Player",
                source_sha256="0" * 64,
            ),
        )
        history = CausalHistoryState()
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            with self.assertRaises(TennisRatioOverlayError):
                _build(
                    database_path,
                    as_of_date=PREDICTION_DATE,
                    results=(_result(),),
                    rankings=rankings,
                    history_state=history,
                )

        self.assertIsNone(history.last_date)

    def test_equivalent_ranking_aliases_have_order_independent_provenance(
        self,
    ) -> None:
        """Same-rank aliases deduplicate deterministically before serving."""

        aliases = (
            _ranking(source_sha256="f" * 64),
            _ranking(
                source_player_key="One-Player",
                source_sha256="0" * 64,
            ),
        )
        fingerprints: list[str | None] = []
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            for observations in (aliases, tuple(reversed(aliases))):
                overlay, _ = _build(
                    database_path,
                    as_of_date=PREDICTION_DATE,
                    results=(),
                    rankings=observations,
                )
                self.assertEqual(overlay.supplemental_rankings, 1)
                fingerprints.append(overlay.overlay_fingerprint)

        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_stale_sidecar_ranking_reports_the_base_actually_served(self) -> None:
        """An older lateral row neither replaces nor ages the base source."""

        stale = _ranking(effective_date=date(2026, 6, 1))
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, _ = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(),
                rankings=(stale,),
            )

        snapshot = overlay.ranking_provider.get_many((1,), PREDICTION_DATE)[0]
        self.assertEqual(snapshot.rank, 101)
        self.assertEqual(snapshot.ranking_date, date(2026, 6, 8))
        self.assertEqual(overlay.ranking_max_date, date(2026, 6, 8))
        self.assertEqual(overlay.supplemental_rankings, 0)

    def test_female_overlay_keeps_gender_state_isolated(self) -> None:
        """Female IDs use only female Elo, history and ranking providers."""

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                gender="F",
                results=(_result(gender="F"),),
                rankings=(_ranking(gender="F"),),
            )

        snapshot = history.snapshot(
            "F",
            1,
            2,
            surface="Hard",
            as_of_date=PREDICTION_DATE,
        )
        self.assertEqual(snapshot.h2h_global_matches, 1)
        elo = overlay.elo_provider.get_many(  # type: ignore[union-attr]
            (EloQuery("F", 1, "Hard", PREDICTION_DATE),)
        )[0]
        self.assertEqual(elo.gender, "F")
        self.assertEqual(elo.general_matches, 11)
        ranking = overlay.ranking_provider.get_many((1,), PREDICTION_DATE)[0]
        self.assertEqual(ranking.gender, "F")
        self.assertEqual(ranking.rank, 7)

    def test_internal_failures_are_wrapped_and_history_remains_unmodified(
        self,
    ) -> None:
        """Failures beyond loading surface only through the overlay contract."""

        history = CausalHistoryState()
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            with self.assertRaises(TennisRatioOverlayError) as raised:
                _build(
                    database_path,
                    as_of_date=PREDICTION_DATE,
                    results=(_result(),),
                    elo_store=_BaseEloStore(fail=True),
                    history_state=history,
                )

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIsNone(history.last_date)

    def test_missing_sidecar_has_no_overlay_fingerprint(self) -> None:
        """An absent optional sidecar preserves the immutable base exactly."""

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "missing.sqlite3"
            overlay, history = _build(
                database_path,
                as_of_date=PREDICTION_DATE,
                results=(_result(),),
            )

        self.assertIsNone(overlay.overlay_fingerprint)
        self.assertIsNone(overlay.elo_provider)
        self.assertIsNone(history.last_date)

    def test_defensive_guard_rejects_non_causal_store_output(self) -> None:
        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "sidecar.sqlite3"
            database_path.touch()
            history = CausalHistoryState()
            with (
                mock.patch(
                    "src.daily_pipeline.tennisratio_overlay.load_mapped_results",
                    return_value=(_result(available_date=PREDICTION_DATE),),
                ),
                mock.patch(
                    "src.daily_pipeline.tennisratio_overlay.load_mapped_rankings",
                    return_value=(),
                ),
                self.assertRaises(TennisRatioOverlayError),
            ):
                build_tennisratio_overlay(
                    gender="M",
                    as_of_date=PREDICTION_DATE,
                    base_result_effective_cutoff=BASE_EFFECTIVE_CUTOFF,
                    base_history_effective_max_date=BASE_EFFECTIVE_CUTOFF,
                    base_history_available_max_date=BASE_DATE,
                    target_player_ids=(1, 2),
                    history_state=history,
                    ranking_index=_BaseRanking(),  # type: ignore[arg-type]
                    elo_store=_BaseEloStore(),  # type: ignore[arg-type]
                    elo_run_id="base-run",
                    elo_base_date=BASE_DATE,
                    elo_parameters=DEFAULT_ELO_PARAMETERS.as_dict(),
                    source_commit="f" * 40,
                    database_path=database_path,
                )


if __name__ == "__main__":
    unittest.main()

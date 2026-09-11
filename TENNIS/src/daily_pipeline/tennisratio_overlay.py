"""Causal serving overlay built from the TennisRatio sidecar.

The persisted Elo can already contain the versioned operational handoff. This
module therefore advances the *live Elo* only with observations whose
availability is later than the persisted ``elo_base_date``. The causal history
is rebuilt separately from the frozen feature artifact, so it receives every
post-handoff observation once. A snapshot first seen on ``D`` is usable from
``D + 1`` and an event already persisted can never be applied twice to Elo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, cast

from ..elo import (
    EloEngine,
    EloParameters,
    EloQuery,
    EloSnapshot,
    EloStore,
    EventProvenance,
    Gender,
    MatchEvent,
    PlayerEloState,
    Surface,
    normalise_surface,
)
from ..features import (
    CausalHistoryState,
    HistoricalMatchResult,
    RankingIndex,
    RankingSnapshot,
)
from ..features.vector import EloFeatureProvider, RankingFeatureProvider
from ..tennisratio import (
    MappedRanking,
    MappedResult,
    load_mapped_rankings,
    load_mapped_results,
)


_ELO_PARAMETER_NAMES = (
    "initial_rating",
    "scale",
    "k_numerator",
    "k_offset",
    "k_exponent",
    "surface_weight",
)


class TennisRatioOverlayError(RuntimeError):
    """Raised when a supplemental observation cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class TennisRatioOverlay:
    """Dependencies and freshness cutoffs produced by one causal overlay."""

    elo_provider: EloFeatureProvider | None
    ranking_provider: RankingFeatureProvider
    overlay_fingerprint: str | None
    history_effective_max_date: date
    history_available_max_date: date
    ranking_max_date: date
    supplemental_results: int
    supplemental_rankings: int


@dataclass(frozen=True, slots=True)
class _EngineEloFeatureProvider:
    """Expose an in-memory continued Elo engine through the feature contract."""

    engine: EloEngine

    def get_many(
        self,
        queries: Iterable[EloQuery],
    ) -> tuple[EloSnapshot, ...]:
        """Resolve each query without changing the continued state."""

        materialized = tuple(queries)
        snapshots: list[EloSnapshot] = []
        for query in materialized:
            if not isinstance(query, EloQuery):
                raise TypeError("queries must contain only EloQuery values.")
            snapshots.append(
                self.engine.snapshot(
                    cast(Gender, query.gender),
                    query.player_id,
                    as_of_date=query.as_of_date,
                    surface=cast(Surface | None, query.surface),
                )
            )
        return tuple(snapshots)


class _OverlayRankingProvider:
    """Prefer a newer causal TennisRatio rank over the Sackmann snapshot."""

    def __init__(
        self,
        base: RankingIndex,
        observations: Iterable[MappedRanking],
    ) -> None:
        """Validate observations and retain one unambiguous date per player."""

        self._base = base
        selected: dict[int, MappedRanking] = {}
        for observation in _deduplicate_rankings(
            observations,
            expected_gender=cast(Gender, base.gender),
        ):
            previous = selected.get(observation.sackmann_player_id)
            if previous is None or observation.effective_date > previous.effective_date:
                selected[observation.sackmann_player_id] = observation
        self._observations = selected

    @property
    def gender(self) -> Gender:
        """Return the single gender served by this provider."""

        return cast(Gender, self._base.gender)

    def get_many(
        self,
        player_ids: Iterable[int],
        as_of_date: date,
    ) -> tuple[RankingSnapshot, ...]:
        """Return one strictly pre-date snapshot per requested player."""

        ids = tuple(player_ids)
        base_snapshots = self._base.get_many(ids, as_of_date)
        output: list[RankingSnapshot] = []
        for player_id, base_snapshot in zip(ids, base_snapshots, strict=True):
            candidate = self._selected_candidate(
                player_id,
                base_snapshot,
                as_of_date=as_of_date,
            )
            if candidate is None:
                output.append(base_snapshot)
                continue
            output.append(
                RankingSnapshot(
                    gender=self.gender,
                    player_id=player_id,
                    as_of_date=as_of_date,
                    ranking_date=candidate.effective_date,
                    rank=candidate.rank,
                    points=None,
                    is_missing=False,
                    ranking_age_days=(as_of_date - candidate.effective_date).days,
                    conflict_dates_skipped=0,
                )
            )
        return tuple(output)

    def served_observations(
        self,
        player_ids: Iterable[int],
        as_of_date: date,
    ) -> tuple[MappedRanking, ...]:
        """Return only sidecar rankings that replace the base snapshots."""

        ids = tuple(player_ids)
        base_snapshots = self._base.get_many(ids, as_of_date)
        selected = (
            self._selected_candidate(
                player_id,
                base_snapshot,
                as_of_date=as_of_date,
            )
            for player_id, base_snapshot in zip(
                ids,
                base_snapshots,
                strict=True,
            )
        )
        return tuple(item for item in selected if item is not None)

    def _selected_candidate(
        self,
        player_id: int,
        base_snapshot: RankingSnapshot,
        *,
        as_of_date: date,
    ) -> MappedRanking | None:
        """Choose a strictly causal candidate only when newer than the base."""

        candidate = self._observations.get(player_id)
        if candidate is None:
            return None
        if candidate.effective_date >= as_of_date or candidate.available_date >= as_of_date:
            raise TennisRatioOverlayError("A TennisRatio ranking crossed the strict as-of cutoff.")
        if (
            base_snapshot.ranking_date is not None
            and candidate.effective_date <= base_snapshot.ranking_date
        ):
            return None
        return candidate


def _normalised_token(value: str) -> str:
    """Normalize a source label solely for post-mapping identity checks."""

    return " ".join(value.casefold().split())


def _deduplicate_rankings(
    observations: Iterable[MappedRanking],
    *,
    expected_gender: Gender,
) -> tuple[MappedRanking, ...]:
    """Deduplicate aliases and reject conflicting ranks for player/date."""

    grouped: dict[tuple[int, date], list[MappedRanking]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, MappedRanking):
            raise TypeError("Ranking observations must contain only MappedRanking values.")
        if observation.gender != expected_gender:
            raise TennisRatioOverlayError("TennisRatio ranking observations mix gender universes.")
        grouped[(observation.sackmann_player_id, observation.effective_date)].append(observation)

    output: list[MappedRanking] = []
    for key in sorted(grouped):
        group = grouped[key]
        ranks = {item.rank for item in group}
        if len(ranks) != 1:
            raise TennisRatioOverlayError(
                "Conflicting TennisRatio ranks map to the same player/date: "
                f"player_id={key[0]}, effective_date={key[1].isoformat()}."
            )
        output.append(
            min(
                group,
                key=lambda item: (
                    item.available_date,
                    item.first_seen_at_utc,
                    item.source_player_key,
                    item.source_url,
                    item.source_sha256,
                ),
            )
        )
    return tuple(output)


def _result_identity_key(result: MappedResult) -> tuple[Gender, date, int, int]:
    """Build a source-alias-independent sporting identity after mapping."""

    lower_id, higher_id = sorted((result.winner_sackmann_id, result.loser_sackmann_id))
    return (
        result.gender,
        result.effective_date,
        lower_id,
        higher_id,
    )


def _deduplicate_results(
    results: Iterable[MappedResult],
    *,
    expected_gender: Gender,
) -> tuple[MappedResult, ...]:
    """Collapse aliases, excluding metadata conflicts and rejecting outcomes.

    Two profile perspectives can name the same tournament or circuit level
    differently while agreeing on players, winner, score and set totals. Such
    a group is unusable as an Elo event because its context is ambiguous, but
    it must not disable every other causal result. A disagreement in sporting
    outcome still fails the whole overlay before any state mutation.
    """

    grouped: dict[tuple[Gender, date, int, int], list[MappedResult]] = defaultdict(list)
    canonical_ids: dict[str, tuple[Gender, date, int, int]] = {}
    for result in results:
        if not isinstance(result, MappedResult):
            raise TypeError("Result observations must contain only MappedResult values.")
        if result.gender != expected_gender:
            raise TennisRatioOverlayError("TennisRatio result observations mix gender universes.")
        identity = _result_identity_key(result)
        previous_identity = canonical_ids.setdefault(
            result.canonical_match_id,
            identity,
        )
        if previous_identity != identity:
            raise TennisRatioOverlayError(
                "One TennisRatio canonical_match_id identifies two matches."
            )
        grouped[identity].append(result)

    output: list[MappedResult] = []
    for identity in sorted(grouped):
        group = grouped[identity]
        oriented_winners = {(item.winner_sackmann_id, item.loser_sackmann_id) for item in group}
        event_labels = {
            (_normalised_token(item.tournament), _normalised_token(item.round)) for item in group
        }
        source_contracts = {(item.tour_level, item.surface) for item in group}
        populated_scores = {item.score for item in group if item.score is not None}
        populated_set_totals = {
            (item.winner_sets_won, item.loser_sets_won)
            for item in group
            if item.winner_sets_won is not None or item.loser_sets_won is not None
        }
        if len(oriented_winners) != 1 or len(populated_scores) > 1 or len(populated_set_totals) > 1:
            raise TennisRatioOverlayError(
                "Conflicting TennisRatio observations map to one sporting match."
            )
        if len(event_labels) != 1 or len(source_contracts) != 1:
            continue
        output.append(
            min(
                group,
                key=lambda item: (
                    item.score is None,
                    item.available_date,
                    item.first_seen_at_utc,
                    item.canonical_match_id,
                    item.source_url,
                    item.source_sha256,
                ),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.available_date,
                item.effective_date,
                item.canonical_match_id,
            ),
        )
    )


def _elo_parameters(raw: Mapping[str, object]) -> EloParameters:
    """Rebuild only the mathematical parameters stored in the active run."""

    missing = sorted(set(_ELO_PARAMETER_NAMES).difference(raw))
    if missing:
        raise TennisRatioOverlayError(
            f"The active Elo run lacks parameters required by the overlay: {missing}."
        )
    values: dict[str, float] = {}
    for name in _ELO_PARAMETER_NAMES:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TennisRatioOverlayError(f"The active Elo parameter {name!r} is not numeric.")
        values[name] = float(value)
    return EloParameters(
        initial_rating=values["initial_rating"],
        scale=values["scale"],
        k_numerator=values["k_numerator"],
        k_offset=values["k_offset"],
        k_exponent=values["k_exponent"],
        surface_weight=values["surface_weight"],
    )


def _event_hash(result: MappedResult) -> str:
    """Create a per-match identity; a profile page SHA is not unique enough."""

    payload = {
        "canonical_match_id": result.canonical_match_id,
        "gender": result.gender,
        "effective_date": result.effective_date.isoformat(),
        "available_date": result.available_date.isoformat(),
        "winner_id": result.winner_sackmann_id,
        "loser_id": result.loser_sackmann_id,
        "source_sha256": result.source_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_overlay_fingerprint(
    *,
    gender: Gender,
    as_of_date: date,
    base_result_effective_cutoff: date,
    base_history_effective_max_date: date,
    base_history_available_max_date: date,
    elo_run_id: str,
    elo_base_date: date,
    elo_parameters: EloParameters,
    source_commit: str,
    target_player_ids: tuple[int, ...],
    results: tuple[MappedResult, ...],
    rankings: tuple[MappedRanking, ...],
) -> str:
    """Hash the exact base contract and sidecar evidence served for ``D``."""

    payload = {
        "schema": "tennisratio-overlay-v1",
        "gender": gender,
        "as_of_date": as_of_date.isoformat(),
        "base": {
            "result_effective_cutoff": (base_result_effective_cutoff.isoformat()),
            "history_effective_max_date": (base_history_effective_max_date.isoformat()),
            "history_available_max_date": (base_history_available_max_date.isoformat()),
            "elo_run_id": elo_run_id,
            "elo_base_date": elo_base_date.isoformat(),
            "elo_parameters": elo_parameters.as_dict(),
            "source_commit": source_commit,
        },
        "target_player_ids": list(target_player_ids),
        "results": [
            {
                "sporting_identity": [
                    item.gender,
                    item.effective_date.isoformat(),
                    min(item.winner_sackmann_id, item.loser_sackmann_id),
                    max(item.winner_sackmann_id, item.loser_sackmann_id),
                ],
                "event_label": [
                    _normalised_token(item.tournament),
                    _normalised_token(item.round),
                ],
                "canonical_match_id": item.canonical_match_id,
                "available_date": item.available_date.isoformat(),
                "event_hash": _event_hash(item),
                "source_url": item.source_url,
                "source_sha256": item.source_sha256,
            }
            for item in results
        ],
        "rankings": [
            {
                "player_id": item.sackmann_player_id,
                "rank": item.rank,
                "effective_date": item.effective_date.isoformat(),
                "available_date": item.available_date.isoformat(),
                "source_player_key": item.source_player_key,
                "source_url": item.source_url,
                "source_sha256": item.source_sha256,
            }
            for item in rankings
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_match_events(results: tuple[MappedResult, ...]) -> tuple[MatchEvent, ...]:
    """Adapt mapped sidecar results to availability-dated Elo events."""

    events: list[MatchEvent] = []
    for ordinal, result in enumerate(
        sorted(
            results,
            key=lambda item: (
                item.available_date,
                item.effective_date,
                item.canonical_match_id,
            ),
        ),
        start=1,
    ):
        surface = normalise_surface(result.surface)
        events.append(
            MatchEvent(
                date=result.available_date,
                source_date=result.effective_date,
                gender=result.gender,
                winner_id=result.winner_sackmann_id,
                loser_id=result.loser_sackmann_id,
                surface=cast(Surface | None, surface),
                tour_level=result.tour_level,
                score=result.score,
                round=result.round,
                provenance=EventProvenance(
                    source_commit="tennisratio-sidecar-v1",
                    source_path=result.source_url,
                    source_row=ordinal,
                ),
                source_record_hash=_event_hash(result),
                tourney_id=result.canonical_match_id,
            )
        )
    return tuple(events)


def _continued_elo_provider(
    *,
    gender: Gender,
    results: tuple[MappedResult, ...],
    target_player_ids: tuple[int, ...],
    elo_store: EloStore,
    elo_run_id: str,
    elo_base_date: date,
    elo_parameters: EloParameters,
    source_commit: str,
) -> tuple[EloFeatureProvider | None, tuple[MappedResult, ...]]:
    """Seed the persisted Elo state and apply visible sidecar result batches."""

    if not results:
        return None, ()
    if any(result.available_date <= elo_base_date for result in results):
        raise TennisRatioOverlayError(
            "A supplemental result is not later than the persisted Elo cutoff."
        )
    participant_ids = set(target_player_ids)
    for result in results:
        participant_ids.add(result.winner_sackmann_id)
        participant_ids.add(result.loser_sackmann_id)
    try:
        seed_as_of = elo_base_date + timedelta(days=1)
    except OverflowError as exc:
        raise TennisRatioOverlayError("The Elo base date cannot be advanced.") from exc
    states = elo_store.fetch_ratings_before(
        run_id=elo_run_id,
        gender=gender,
        queries=tuple((player_id, seed_as_of) for player_id in sorted(participant_ids)),
    )
    engine = EloEngine(
        elo_parameters,
        run_id=elo_run_id,
        source_commit=source_commit,
    )
    existing_states: list[PlayerEloState] = []
    for state in states:
        if state is None:
            continue
        if not isinstance(state, PlayerEloState):
            raise TypeError("The persisted Elo provider returned an invalid state.")
        if state.gender != gender:
            raise TennisRatioOverlayError("The persisted Elo provider crossed gender universes.")
        existing_states.append(state)
    if existing_states:
        engine.seed_states(tuple(existing_states), base_date=elo_base_date)

    grouped: dict[date, list[MatchEvent]] = defaultdict(list)
    for event in _as_match_events(results):
        grouped[event.date].append(event)
    included_hashes: set[str] = set()
    for available_date in sorted(grouped):
        block = engine.process_date_block(available_date, grouped[available_date])
        included_hashes.update(rated.event.source_record_hash for rated in block.rated_matches)
    included_results = tuple(result for result in results if _event_hash(result) in included_hashes)
    return _EngineEloFeatureProvider(engine), included_results


def _ratable_results(
    *,
    results: tuple[MappedResult, ...],
    elo_parameters: EloParameters,
    elo_run_id: str,
    source_commit: str,
) -> tuple[MappedResult, ...]:
    """Apply the canonical Elo exclusions before serving Elo or history.

    Persisted operational events must not be replayed into Elo, but history
    still needs the same terminal-result universe. Running the immutable
    exclusion contract on an empty audit engine keeps walkovers and malformed
    events out of both paths without mutating the persisted state.
    """

    if not results:
        return ()
    engine = EloEngine(
        elo_parameters,
        run_id=elo_run_id,
        source_commit=source_commit,
    )
    grouped: dict[date, list[MatchEvent]] = defaultdict(list)
    for event in _as_match_events(results):
        grouped[event.date].append(event)
    included_hashes: set[str] = set()
    for available_date in sorted(grouped):
        block = engine.process_date_block(available_date, grouped[available_date])
        included_hashes.update(rated.event.source_record_hash for rated in block.rated_matches)
    return tuple(result for result in results if _event_hash(result) in included_hashes)


def _apply_history_results(
    *,
    history_state: CausalHistoryState,
    results: tuple[MappedResult, ...],
    target_player_ids: tuple[int, ...],
) -> tuple[MappedResult, ...]:
    """Advance only the directed history needed by today's players."""

    targets = set(target_player_ids)
    selected = tuple(
        result
        for result in results
        if result.winner_sackmann_id in targets or result.loser_sackmann_id in targets
    )
    grouped: dict[date, list[HistoricalMatchResult]] = defaultdict(list)
    for result in selected:
        surface = normalise_surface(result.surface)
        grouped[result.available_date].append(
            HistoricalMatchResult(
                match_date=result.effective_date,
                gender=result.gender,
                winner_id=result.winner_sackmann_id,
                loser_id=result.loser_sackmann_id,
                surface=cast(Surface | None, surface),
            )
        )
    for available_date in sorted(grouped):
        history_state.apply_availability_batch(
            available_date,
            grouped[available_date],
        )
    return selected


def build_tennisratio_overlay(
    *,
    gender: Gender,
    as_of_date: date,
    base_result_effective_cutoff: date,
    base_history_effective_max_date: date,
    base_history_available_max_date: date,
    target_player_ids: tuple[int, ...],
    history_state: CausalHistoryState,
    ranking_index: RankingIndex,
    elo_store: EloStore,
    elo_run_id: str,
    elo_base_date: date,
    elo_parameters: Mapping[str, object],
    source_commit: str,
    database_path: Path,
) -> TennisRatioOverlay:
    """Build a live overlay without mutating historical artifacts or ledgers."""

    if not database_path.is_file():
        return TennisRatioOverlay(
            elo_provider=None,
            ranking_provider=ranking_index,
            overlay_fingerprint=None,
            history_effective_max_date=base_history_effective_max_date,
            history_available_max_date=base_history_available_max_date,
            ranking_max_date=ranking_index.max_ranking_date,
            supplemental_results=0,
            supplemental_rankings=0,
        )
    try:
        results = load_mapped_results(
            as_of_date,
            base_cutoff_by_gender={gender: base_result_effective_cutoff},
            gender=gender,
            database_path=database_path,
        )
        rankings = load_mapped_rankings(
            as_of_date,
            gender=gender,
            database_path=database_path,
        )
    except Exception as exc:
        raise TennisRatioOverlayError(
            "The TennisRatio last-good sidecar could not be queried safely."
        ) from exc
    try:
        canonical_targets = tuple(sorted(set(target_player_ids)))
        if ranking_index.gender != gender:
            raise TennisRatioOverlayError("The base ranking provider crossed gender universes.")
        if any(
            result.gender != gender
            or result.effective_date <= base_result_effective_cutoff
            or result.effective_date > result.available_date
            or result.effective_date >= as_of_date
            or result.available_date >= as_of_date
            for result in results
        ):
            raise TennisRatioOverlayError(
                "The TennisRatio result query returned a non-causal or overlapping observation."
            )
        if any(
            ranking.gender != gender
            or ranking.effective_date > ranking.available_date
            or ranking.effective_date >= as_of_date
            or ranking.available_date >= as_of_date
            for ranking in rankings
        ):
            raise TennisRatioOverlayError(
                "The TennisRatio ranking query returned a non-causal observation."
            )

        deduplicated_results = _deduplicate_results(
            results,
            expected_gender=gender,
        )
        parameters = _elo_parameters(elo_parameters)
        ratable_results = _ratable_results(
            results=deduplicated_results,
            elo_parameters=parameters,
            elo_run_id=elo_run_id,
            source_commit=source_commit,
        )
        elo_results = tuple(item for item in ratable_results if item.available_date > elo_base_date)
        if history_state.last_date is not None and any(
            item.available_date <= history_state.last_date for item in ratable_results
        ):
            raise TennisRatioOverlayError(
                "A supplemental result overlaps the existing history availability state."
            )
        target_set = set(canonical_targets)
        target_rankings = tuple(item for item in rankings if item.sackmann_player_id in target_set)
        ranking_provider = _OverlayRankingProvider(
            ranking_index,
            target_rankings,
        )
        served_rankings = ranking_provider.served_observations(
            canonical_targets,
            as_of_date,
        )
        served_ranking_snapshots = ranking_provider.get_many(
            canonical_targets,
            as_of_date,
        )
        served_ranking_dates = tuple(
            snapshot.ranking_date
            for snapshot in served_ranking_snapshots
            if snapshot.ranking_date is not None
        )
        elo_provider, _ = _continued_elo_provider(
            gender=gender,
            results=elo_results,
            target_player_ids=canonical_targets,
            elo_store=elo_store,
            elo_run_id=elo_run_id,
            elo_base_date=elo_base_date,
            elo_parameters=parameters,
            source_commit=source_commit,
        )
        overlay_fingerprint = _build_overlay_fingerprint(
            gender=gender,
            as_of_date=as_of_date,
            base_result_effective_cutoff=base_result_effective_cutoff,
            base_history_effective_max_date=(base_history_effective_max_date),
            base_history_available_max_date=(base_history_available_max_date),
            elo_run_id=elo_run_id,
            elo_base_date=elo_base_date,
            elo_parameters=parameters,
            source_commit=source_commit,
            target_player_ids=canonical_targets,
            results=ratable_results,
            rankings=served_rankings,
        )
        history_results = _apply_history_results(
            history_state=history_state,
            results=ratable_results,
            target_player_ids=canonical_targets,
        )
        return TennisRatioOverlay(
            elo_provider=elo_provider,
            ranking_provider=ranking_provider,
            overlay_fingerprint=overlay_fingerprint,
            history_effective_max_date=max(
                (
                    base_history_effective_max_date,
                    *(item.effective_date for item in history_results),
                )
            ),
            history_available_max_date=max(
                (
                    base_history_available_max_date,
                    *(item.available_date for item in history_results),
                )
            ),
            ranking_max_date=max(
                served_ranking_dates,
                default=ranking_index.max_ranking_date,
            ),
            supplemental_results=len(ratable_results),
            supplemental_rankings=len(served_rankings),
        )
    except TennisRatioOverlayError:
        raise
    except Exception as exc:
        raise TennisRatioOverlayError(
            "The TennisRatio overlay could not be constructed safely."
        ) from exc


__all__ = [
    "TennisRatioOverlay",
    "TennisRatioOverlayError",
    "build_tennisratio_overlay",
]

"""Motor Elo causal con estados separados por género y superficie.

Cada fecha se procesa como un bloque atómico. Todas las expectativas, K y
deltas del día se calculan contra el mismo snapshot anterior; después se suman
de forma determinista y se publica el estado posterior al bloque.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import groupby
import math
import re
from typing import Iterable, Iterator, Sequence, cast

from .parameters import DEFAULT_ELO_PARAMETERS, EloParameters
from .types import (
    AuditCounts,
    DateBlockResult,
    EloRunResult,
    EloSnapshot,
    EventDecision,
    ExclusionReason,
    Gender,
    MatchEvent,
    PlayerEloState,
    PlayerPreMatchRating,
    RatedMatch,
    Surface,
)


SURFACES: tuple[Surface, ...] = ("Hard", "Clay", "Grass", "Carpet")
EXCLUDED_LEVELS = frozenset({"E", "J"})
_NON_MATCH_STATUS_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:W\s*/\s*O|WALKOVER|BYE)"
    r"(?![A-Z0-9])",
    flags=re.IGNORECASE,
)
_REASON_ORDER: tuple[ExclusionReason, ...] = (
    "excluded_level",
    "excluded_status",
    "excluded_identity",
    "self_match",
    "duplicate",
)


class EloEngineError(RuntimeError):
    """Indica una violación del contrato temporal o del estado Elo."""


class DateBlockError(EloEngineError):
    """Indica que un bloque no contiene una única fecha posterior."""


@dataclass
class _SurfaceState:
    """Mantiene rating y contador mutable de una superficie."""

    rating: float
    matches: int = 0


@dataclass
class _PlayerState:
    """Mantiene el estado mutable completo de un jugador."""

    general_rating: float
    general_matches: int = 0
    surfaces: dict[Surface, _SurfaceState] = field(default_factory=dict)
    state_date: date | None = None


def _empty_audit() -> AuditCounts:
    """Construye una auditoría sin filas."""

    return AuditCounts(
        received=0,
        included=0,
        excluded=0,
        excluded_by_reason=(),
    )


def _audit_from_decisions(
    decisions: Sequence[EventDecision],
) -> AuditCounts:
    """Resume una secuencia de decisiones en orden canónico de motivos."""

    counts: Counter[ExclusionReason] = Counter(
        cast(ExclusionReason, decision.reason)
        for decision in decisions
        if decision.reason is not None
    )
    included = sum(decision.included for decision in decisions)
    excluded_by_reason = tuple(
        (reason, counts[reason])
        for reason in _REASON_ORDER
        if counts[reason]
    )
    return AuditCounts(
        received=len(decisions),
        included=included,
        excluded=len(decisions) - included,
        excluded_by_reason=excluded_by_reason,
    )


def _merge_audits(
    left: AuditCounts,
    right: AuditCounts,
) -> AuditCounts:
    """Suma dos auditorías manteniendo el orden estable de motivos."""

    counts: Counter[ExclusionReason] = Counter(
        dict(left.excluded_by_reason)
    )
    counts.update(dict(right.excluded_by_reason))
    return AuditCounts(
        received=left.received + right.received,
        included=left.included + right.included,
        excluded=left.excluded + right.excluded,
        excluded_by_reason=tuple(
            (reason, counts[reason])
            for reason in _REASON_ORDER
            if counts[reason]
        ),
    )


def _validate_block_date(value: object) -> date:
    """Acepta exclusivamente ``datetime.date`` sin hora."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise DateBlockError("match_date debe ser datetime.date estricto.")
    return value


def _exclusive_as_of_date(state_date: date) -> date:
    """Obtiene el primer corte válido posterior a un estado postfecha."""

    try:
        return state_date + timedelta(days=1)
    except OverflowError as exc:
        raise DateBlockError(
            "date.max no admite un snapshot temporal posterior."
        ) from exc


def _event_exclusion_reason(
    event: MatchEvent,
    excluded_player_keys: frozenset[tuple[Gender, int]],
) -> ExclusionReason | None:
    """Aplica exclusiones auditables antes de deduplicar.

    ``RET``, ``DEF``, ``ABD`` y ``ABN`` conservan el ganador oficial. Usarlos
    para excluir después del partido produciría un backtest más limpio que el
    universo que se intenta predecir.
    """

    if event.tour_level.strip().upper() in EXCLUDED_LEVELS:
        return "excluded_level"
    if (
        event.score is not None
        and _NON_MATCH_STATUS_PATTERN.search(event.score)
    ):
        return "excluded_status"
    if (
        (event.gender, event.winner_id) in excluded_player_keys
        or (event.gender, event.loser_id) in excluded_player_keys
    ):
        return "excluded_identity"
    if event.winner_id == event.loser_id:
        return "self_match"
    return None


def _filter_events(
    events: Sequence[MatchEvent],
    excluded_player_keys: frozenset[tuple[Gender, int]],
) -> tuple[tuple[MatchEvent, ...], tuple[EventDecision, ...]]:
    """Filtra y deduplica eventos con un keeper fijado por procedencia."""

    included: list[MatchEvent] = []
    decisions: list[EventDecision] = []
    seen_keys: set[tuple[object, ...]] = set()
    for event in sorted(events, key=lambda item: item.sort_key):
        reason = _event_exclusion_reason(event, excluded_player_keys)
        if reason is None:
            if event.logical_key in seen_keys:
                reason = "duplicate"
            else:
                seen_keys.add(event.logical_key)
                included.append(event)
        decisions.append(
            EventDecision(
                event=event,
                included=reason is None,
                reason=reason,
            )
        )
    return tuple(included), tuple(decisions)


class EloEngine:
    """Procesa eventos cronológicos sin retener el histórico de bloques."""

    def __init__(
        self,
        parameters: EloParameters = DEFAULT_ELO_PARAMETERS,
        *,
        run_id: str | None = None,
        source_commit: str | None = None,
        excluded_player_keys: Iterable[tuple[Gender, int]] = (),
    ) -> None:
        """Inicializa estado, metadatos y cuarentena de identidad."""

        if not isinstance(parameters, EloParameters):
            raise TypeError("parameters debe ser EloParameters.")
        if run_id is not None and (
            not isinstance(run_id, str) or not run_id.strip()
        ):
            raise ValueError("run_id debe ser texto no vacío o None.")
        if source_commit is not None and (
            not isinstance(source_commit, str) or not source_commit.strip()
        ):
            raise ValueError("source_commit debe ser texto no vacío o None.")
        self.parameters = parameters
        self.run_id = run_id.strip() if run_id is not None else None
        self.source_commit = (
            source_commit.strip() if source_commit is not None else None
        )
        checked_keys: set[tuple[Gender, int]] = set()
        for key in excluded_player_keys:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key[0] not in {"M", "F"}
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
                or key[1] < 0
            ):
                raise ValueError(
                    "excluded_player_keys debe contener pares "
                    "(gender, player_id) válidos."
                )
            checked_keys.add((cast(Gender, key[0]), key[1]))
        self.excluded_player_keys = frozenset(checked_keys)
        self._players: dict[tuple[Gender, int], _PlayerState] = {}
        self._last_date: date | None = None

    @property
    def last_date(self) -> date | None:
        """Devuelve la última fecha cerrada por el motor."""

        return self._last_date

    def _new_player_state(self) -> _PlayerState:
        """Construye un cold start completo con las cuatro superficies."""

        initial = float(self.parameters.initial_rating)
        return _PlayerState(
            general_rating=initial,
            surfaces={
                surface: _SurfaceState(rating=initial)
                for surface in SURFACES
            },
        )

    def _state_for_read(
        self,
        gender: Gender,
        player_id: int,
    ) -> _PlayerState:
        """Devuelve el estado existente o un cold start transitorio."""

        return self._players.get(
            (gender, player_id),
            self._new_player_state(),
        )

    def _pre_match_rating(
        self,
        event: MatchEvent,
        player_id: int,
    ) -> PlayerPreMatchRating:
        """Construye la vista prepartido para un jugador de un evento."""

        state = self._state_for_read(event.gender, player_id)
        if event.surface is None:
            surface_rating = None
            surface_matches = 0
            surface_k = None
        else:
            surface_state = state.surfaces[event.surface]
            surface_rating = surface_state.rating
            surface_matches = surface_state.matches
            surface_k = self.parameters.k_factor(surface_matches)
        return PlayerPreMatchRating(
            gender=event.gender,
            player_id=player_id,
            surface=event.surface,
            general_elo=state.general_rating,
            surface_elo_raw=surface_rating,
            combined_elo=self.parameters.combine(
                state.general_rating,
                surface_rating,
            ),
            general_matches=state.general_matches,
            surface_matches=surface_matches,
            general_k=self.parameters.k_factor(state.general_matches),
            surface_k=surface_k,
        )

    def _rate_event(self, event: MatchEvent) -> RatedMatch:
        """Calcula expectativas y deltas sin mutar ningún estado."""

        winner = self._pre_match_rating(event, event.winner_id)
        loser = self._pre_match_rating(event, event.loser_id)
        combined_expected = self.parameters.expected_score(
            winner.combined_elo,
            loser.combined_elo,
        )
        general_expected = self.parameters.expected_score(
            winner.general_elo,
            loser.general_elo,
        )
        winner_general_delta = winner.general_k * (
            1.0 - general_expected
        )
        loser_general_delta = loser.general_k * (
            0.0 - (1.0 - general_expected)
        )

        if event.surface is None:
            winner_surface_delta = None
            loser_surface_delta = None
        else:
            if (
                winner.surface_elo_raw is None
                or loser.surface_elo_raw is None
                or winner.surface_k is None
                or loser.surface_k is None
            ):
                raise EloEngineError(
                    "El estado de superficie poblada quedó incompleto."
                )
            surface_expected = self.parameters.expected_score(
                winner.surface_elo_raw,
                loser.surface_elo_raw,
            )
            winner_surface_delta = winner.surface_k * (
                1.0 - surface_expected
            )
            loser_surface_delta = loser.surface_k * (
                0.0 - (1.0 - surface_expected)
            )

        return RatedMatch(
            event=event,
            winner_before=winner,
            loser_before=loser,
            winner_expected_probability=combined_expected,
            winner_general_delta=winner_general_delta,
            loser_general_delta=loser_general_delta,
            winner_surface_delta=winner_surface_delta,
            loser_surface_delta=loser_surface_delta,
        )

    def _apply_rated_matches(
        self,
        match_date: date,
        rated_matches: Sequence[RatedMatch],
    ) -> set[tuple[Gender, int]]:
        """Suma deltas determinísticamente y publica el estado postfecha."""

        general_deltas: dict[tuple[Gender, int], list[float]] = defaultdict(
            list
        )
        general_counts: Counter[tuple[Gender, int]] = Counter()
        surface_deltas: dict[
            tuple[Gender, int, Surface],
            list[float],
        ] = defaultdict(list)
        surface_counts: Counter[tuple[Gender, int, Surface]] = Counter()
        affected: set[tuple[Gender, int]] = set()

        for rated in rated_matches:
            event = rated.event
            winner_key = (event.gender, event.winner_id)
            loser_key = (event.gender, event.loser_id)
            affected.update((winner_key, loser_key))
            general_deltas[winner_key].append(rated.winner_general_delta)
            general_deltas[loser_key].append(rated.loser_general_delta)
            general_counts[winner_key] += 1
            general_counts[loser_key] += 1
            if event.surface is not None:
                winner_surface_key = (
                    event.gender,
                    event.winner_id,
                    event.surface,
                )
                loser_surface_key = (
                    event.gender,
                    event.loser_id,
                    event.surface,
                )
                if (
                    rated.winner_surface_delta is None
                    or rated.loser_surface_delta is None
                ):
                    raise EloEngineError(
                        "Faltan deltas de una superficie poblada."
                    )
                surface_deltas[winner_surface_key].append(
                    rated.winner_surface_delta
                )
                surface_deltas[loser_surface_key].append(
                    rated.loser_surface_delta
                )
                surface_counts[winner_surface_key] += 1
                surface_counts[loser_surface_key] += 1

        for player_key in sorted(affected):
            state = self._players.setdefault(
                player_key,
                self._new_player_state(),
            )
            state.general_rating += math.fsum(general_deltas[player_key])
            state.general_matches += general_counts[player_key]
            state.state_date = match_date
            gender, player_id = player_key
            for surface in SURFACES:
                surface_key = (gender, player_id, surface)
                if surface_counts[surface_key] == 0:
                    continue
                surface_state = state.surfaces[surface]
                surface_state.rating += math.fsum(
                    surface_deltas[surface_key]
                )
                surface_state.matches += surface_counts[surface_key]
        return affected

    def _wide_state(
        self,
        gender: Gender,
        player_id: int,
        state_date: date,
    ) -> PlayerEloState:
        """Materializa el estado ancho de un jugador existente."""

        state = self._players[(gender, player_id)]
        return PlayerEloState(
            gender=gender,
            player_id=player_id,
            state_date=state_date,
            general_elo=state.general_rating,
            hard_elo=state.surfaces["Hard"].rating,
            clay_elo=state.surfaces["Clay"].rating,
            grass_elo=state.surfaces["Grass"].rating,
            carpet_elo=state.surfaces["Carpet"].rating,
            general_matches=state.general_matches,
            hard_matches=state.surfaces["Hard"].matches,
            clay_matches=state.surfaces["Clay"].matches,
            grass_matches=state.surfaces["Grass"].matches,
            carpet_matches=state.surfaces["Carpet"].matches,
        )

    def snapshot(
        self,
        gender: Gender,
        player_id: int,
        *,
        as_of_date: date,
        surface: Surface | None = None,
    ) -> EloSnapshot:
        """Consulta el estado actual sin procesar ni anticipar eventos futuros."""

        requested_date = _validate_block_date(as_of_date)
        state = self._state_for_read(gender, player_id)
        if state.state_date is not None and state.state_date >= requested_date:
            raise DateBlockError(
                "as_of_date debe ser posterior al state_date ya incorporado."
            )
        if surface is None:
            surface_rating = None
            surface_matches = 0
            combined = state.general_rating
            cold_start = state.general_matches == 0
        else:
            if surface not in SURFACES:
                raise ValueError(f"Superficie desconocida: {surface!r}.")
            surface_state = state.surfaces[surface]
            surface_rating = surface_state.rating
            surface_matches = surface_state.matches
            combined = self.parameters.combine(
                state.general_rating,
                surface_rating,
            )
            cold_start = (
                state.general_matches == 0
            )
        return EloSnapshot(
            gender=gender,
            player_id=player_id,
            as_of_date=requested_date,
            state_date=state.state_date,
            general_elo=state.general_rating,
            surface=surface,
            surface_elo_raw=surface_rating,
            combined_elo=combined,
            general_matches=state.general_matches,
            surface_matches=surface_matches,
            is_cold_start=cold_start,
            run_id=self.run_id,
            source_commit=self.source_commit,
        )

    def _snapshots_for(
        self,
        keys: Iterable[tuple[Gender, int]],
        *,
        as_of_date: date,
    ) -> tuple[EloSnapshot, ...]:
        """Materializa vistas general y por superficie en orden estable."""

        snapshots: list[EloSnapshot] = []
        for gender, player_id in sorted(set(keys)):
            snapshots.append(
                self.snapshot(
                    gender,
                    player_id,
                    as_of_date=as_of_date,
                    surface=None,
                )
            )
            for surface in SURFACES:
                snapshots.append(
                    self.snapshot(
                        gender,
                        player_id,
                        as_of_date=as_of_date,
                        surface=surface,
                    )
                )
        return tuple(snapshots)

    def process_date_block(
        self,
        match_date: date,
        events: Iterable[MatchEvent],
    ) -> DateBlockResult:
        """Procesa una fecha completa contra un snapshot prefecha común.

        El llamador debe reunir todas las fuentes y chunks de ``match_date`` en
        una sola llamada. Una fecha no puede reabrirse posteriormente.
        """

        validated_date = _validate_block_date(match_date)
        post_as_of_date = _exclusive_as_of_date(validated_date)
        materialised = tuple(events)
        if not materialised:
            raise DateBlockError("Un bloque de fecha no puede estar vacío.")
        if self._last_date is not None and validated_date <= self._last_date:
            raise DateBlockError(
                "Las fechas deben procesarse una sola vez y en orden creciente."
            )
        for event in materialised:
            if not isinstance(event, MatchEvent):
                raise TypeError("events debe contener solo MatchEvent.")
            if event.date != validated_date:
                raise DateBlockError(
                    "Todos los eventos deben coincidir exactamente con "
                    "match_date."
                )

        included, decisions = _filter_events(
            materialised,
            self.excluded_player_keys,
        )
        rated_matches = tuple(self._rate_event(event) for event in included)
        affected = self._apply_rated_matches(
            validated_date,
            rated_matches,
        )
        self._last_date = validated_date
        states = tuple(
            self._wide_state(gender, player_id, validated_date)
            for gender, player_id in sorted(affected)
        )
        snapshots = self._snapshots_for(
            affected,
            as_of_date=post_as_of_date,
        )
        return DateBlockResult(
            date=validated_date,
            rated_matches=rated_matches,
            decisions=decisions,
            states=states,
            snapshots=snapshots,
            audit=_audit_from_decisions(decisions),
        )

    def process(self, events: Iterable[MatchEvent]) -> EloRunResult:
        """Procesa eventos por fecha y devuelve solo el estado final.

        Este método de conveniencia materializa y ordena su entrada. Para el
        histórico completo debe preferirse ``process_date_block`` desde un
        iterador externo por fecha.
        """

        ordered = sorted(tuple(events), key=lambda event: event.sort_key)
        if not ordered:
            return EloRunResult(
                first_date=None,
                last_date=self._last_date,
                processed_dates=0,
                audit=_empty_audit(),
                states=(),
                snapshots=(),
            )

        first_date = ordered[0].date
        processed_dates = 0
        audit = _empty_audit()
        for match_date, grouped in groupby(
            ordered,
            key=lambda event: event.date,
        ):
            block = self.process_date_block(match_date, tuple(grouped))
            processed_dates += 1
            audit = _merge_audits(audit, block.audit)

        if self._last_date is None:
            raise EloEngineError("El motor terminó sin una fecha de estado.")
        keys = tuple(self._players)
        states = tuple(
            self._wide_state(
                gender,
                player_id,
                cast(date, self._players[(gender, player_id)].state_date),
            )
            for gender, player_id in sorted(keys)
        )
        snapshots = self._snapshots_for(
            keys,
            as_of_date=_exclusive_as_of_date(self._last_date),
        )
        return EloRunResult(
            first_date=first_date,
            last_date=self._last_date,
            processed_dates=processed_dates,
            audit=audit,
            states=states,
            snapshots=snapshots,
        )

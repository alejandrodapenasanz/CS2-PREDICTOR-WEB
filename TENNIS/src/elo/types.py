"""Tipos inmutables compartidos por la construcción, motor y consultas Elo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Literal


Gender = Literal["M", "F"]
Surface = Literal["Hard", "Clay", "Grass", "Carpet"]
ExclusionReason = Literal[
    "excluded_level",
    "excluded_status",
    "excluded_identity",
    "duplicate",
    "self_match",
]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, order=True)
class EventProvenance:
    """Identifica de forma estable una fila por commit, ruta y ordinal."""

    source_commit: str
    source_path: str
    source_row: int

    def __post_init__(self) -> None:
        """Valida que la procedencia sea completa y ordenable."""

        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit debe ser texto no vacío.")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("source_path debe ser texto no vacío.")
        if (
            isinstance(self.source_row, bool)
            or not isinstance(self.source_row, int)
            or self.source_row < 1
        ):
            raise ValueError("source_row debe ser un entero positivo.")


@dataclass(frozen=True)
class MatchEvent:
    """Representa un resultado histórico normalizado antes de filtrarlo."""

    date: date
    gender: Gender
    winner_id: int
    loser_id: int
    surface: Surface | None
    tour_level: str
    score: str | None
    provenance: EventProvenance
    source_record_hash: str
    tourney_id: str | None = None
    match_num: str | None = None
    round: str | None = None

    def __post_init__(self) -> None:
        """Valida identidad, vocabularios y hash sin normalizar silenciosamente."""

        if (
            isinstance(self.date, datetime)
            or not isinstance(self.date, date)
        ):
            raise ValueError("date debe ser datetime.date estricto.")
        if self.gender not in {"M", "F"}:
            raise ValueError("gender debe ser exactamente M o F.")
        for name, player_id in (
            ("winner_id", self.winner_id),
            ("loser_id", self.loser_id),
        ):
            if (
                isinstance(player_id, bool)
                or not isinstance(player_id, int)
                or player_id < 0
            ):
                raise ValueError(f"{name} debe ser un entero no negativo.")
        if self.surface not in {None, "Hard", "Clay", "Grass", "Carpet"}:
            raise ValueError(f"Superficie canónica desconocida: {self.surface!r}.")
        if not isinstance(self.tour_level, str) or not self.tour_level:
            raise ValueError("tour_level debe ser texto no vacío.")
        if self.score is not None and not isinstance(self.score, str):
            raise ValueError("score debe ser texto o None.")
        if not isinstance(self.provenance, EventProvenance):
            raise ValueError("provenance debe ser EventProvenance.")
        if (
            not isinstance(self.source_record_hash, str)
            or _SHA256_PATTERN.fullmatch(self.source_record_hash) is None
        ):
            raise ValueError(
                "source_record_hash debe ser un SHA-256 hexadecimal."
            )

    @property
    def logical_key(self) -> tuple[object, ...]:
        """Identifica una copia exacta sin mezclar universos de género."""

        return (
            self.gender,
            self.source_record_hash,
        )

    @property
    def sort_key(self) -> tuple[object, ...]:
        """Devuelve una clave total determinista con procedencia al principio."""

        return (
            self.date,
            self.provenance.source_commit,
            self.provenance.source_path,
            self.provenance.source_row,
            self.source_record_hash,
            self.gender,
            self.winner_id,
            self.loser_id,
            self.tourney_id or "",
            self.match_num or "",
        )


@dataclass(frozen=True)
class PlayerPreMatchRating:
    """Expone el estado prepartido de un jugador y sus K aplicables."""

    gender: Gender
    player_id: int
    surface: Surface | None
    general_elo: float
    surface_elo_raw: float | None
    combined_elo: float
    general_matches: int
    surface_matches: int
    general_k: float
    surface_k: float | None


@dataclass(frozen=True)
class RatedMatch:
    """Conserva expectativa y deltas calculados desde el snapshot común."""

    event: MatchEvent
    winner_before: PlayerPreMatchRating
    loser_before: PlayerPreMatchRating
    winner_expected_probability: float
    winner_general_delta: float
    loser_general_delta: float
    winner_surface_delta: float | None
    loser_surface_delta: float | None


@dataclass(frozen=True)
class EventDecision:
    """Registra si una fila se incluyó y, si no, el motivo normalizado."""

    event: MatchEvent
    included: bool
    reason: ExclusionReason | None


@dataclass(frozen=True)
class AuditCounts:
    """Resume decisiones de inclusión sin depender de diccionarios mutables."""

    received: int
    included: int
    excluded: int
    excluded_by_reason: tuple[tuple[ExclusionReason, int], ...]

    def count(self, reason: ExclusionReason) -> int:
        """Devuelve el número excluido por un motivo concreto."""

        return dict(self.excluded_by_reason).get(reason, 0)


@dataclass(frozen=True)
class EloSnapshot:
    """Contrato temporal común para motor, persistencia y servicio de consulta."""

    gender: Gender
    player_id: int
    as_of_date: date
    state_date: date | None
    general_elo: float
    surface: Surface | None
    surface_elo_raw: float | None
    combined_elo: float
    general_matches: int
    surface_matches: int
    is_cold_start: bool
    run_id: str | None
    source_commit: str | None


@dataclass(frozen=True)
class PlayerEloState:
    """Estado ancho posterior a una fecha, apto para persistencia por jugador."""

    gender: Gender
    player_id: int
    state_date: date
    general_elo: float
    hard_elo: float
    clay_elo: float
    grass_elo: float
    carpet_elo: float
    general_matches: int
    hard_matches: int
    clay_matches: int
    grass_matches: int
    carpet_matches: int


@dataclass(frozen=True)
class DateBlockResult:
    """Resultado autocontenido de cerrar un bloque exacto de fecha."""

    date: date
    rated_matches: tuple[RatedMatch, ...]
    decisions: tuple[EventDecision, ...]
    states: tuple[PlayerEloState, ...]
    snapshots: tuple[EloSnapshot, ...]
    audit: AuditCounts


@dataclass(frozen=True)
class EloRunResult:
    """Resume un procesamiento completo sin retener bloques históricos."""

    first_date: date | None
    last_date: date | None
    processed_dates: int
    audit: AuditCounts
    states: tuple[PlayerEloState, ...]
    snapshots: tuple[EloSnapshot, ...]

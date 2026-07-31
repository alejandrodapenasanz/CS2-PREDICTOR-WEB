"""Estado causal para forma reciente, H2H y descanso.

El motor recibe resultados ya depurados en bloques completos de fecha. Un
``snapshot`` consultado para la fecha ``D`` solo observa bloques cerrados con
fecha estrictamente anterior a ``D``. Los resultados de ``D`` deben aplicarse
después de construir todas las filas de esa fecha mediante
``apply_date_block``.

No existe un orden fiable dentro de una fecha Sackmann. Por ello, la ventana
de últimos partidos conserva entero el bloque de fecha que cruza el límite
configurado; el contador expuesto puede ser mayor que ``recent_matches``.
"""

from __future__ import annotations

from calendar import monthrange
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
import math
from typing import Iterable

from ..elo.types import Gender, Surface


DEFAULT_RECENT_MATCHES = 10
DEFAULT_RECENT_MONTHS = 3
_SURFACES = frozenset({"Hard", "Clay", "Grass", "Carpet"})


class HistoryStateError(RuntimeError):
    """Indica una violación del contrato del estado histórico."""


class HistoryDateOrderError(HistoryStateError):
    """Indica una consulta o actualización temporalmente insegura."""


@dataclass(frozen=True, slots=True)
class HistoricalMatchResult:
    """Representa un resultado elegible que actualizará el estado tras su día."""

    match_date: date
    gender: Gender
    winner_id: int
    loser_id: int
    surface: Surface | None

    def __post_init__(self) -> None:
        """Valida fecha, género, jugadores y superficie sin normalizarlos."""

        _validate_date(self.match_date, field_name="match_date")
        _validate_gender(self.gender)
        _validate_player_id(self.winner_id, field_name="winner_id")
        _validate_player_id(self.loser_id, field_name="loser_id")
        if self.winner_id == self.loser_id:
            raise ValueError("winner_id y loser_id deben ser distintos.")
        _validate_surface(self.surface)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """Expone las métricas históricas de A contra B justo antes de una fecha."""

    as_of_date: date
    gender: Gender
    player_a_id: int
    player_b_id: int
    surface: Surface | None
    recent_n_win_rate_a: float
    recent_n_matches_a: int
    recent_n_win_rate_b: float
    recent_n_matches_b: int
    recent_months_win_rate_a: float
    recent_months_matches_a: int
    recent_months_win_rate_b: float
    recent_months_matches_b: int
    h2h_global_balance: float
    h2h_global_matches: int
    h2h_surface_balance: float | None
    h2h_surface_matches: int
    rest_days_a: int | None
    rest_days_b: int | None

    @property
    def recent_n_win_rate_diff(self) -> float:
        """Devuelve la forma de A menos la de B en últimos partidos."""

        return self.recent_n_win_rate_a - self.recent_n_win_rate_b

    @property
    def recent_months_win_rate_diff(self) -> float:
        """Devuelve la forma de A menos la de B en la ventana de meses."""

        return (
            self.recent_months_win_rate_a
            - self.recent_months_win_rate_b
        )

    @property
    def rest_days_diff(self) -> int | None:
        """Devuelve el descanso de A menos el de B si ambos son conocidos."""

        if self.rest_days_a is None or self.rest_days_b is None:
            return None
        return self.rest_days_a - self.rest_days_b


@dataclass(frozen=True, slots=True)
class _FormDateBlock:
    """Agrega la forma de un jugador en una fecha sin orden intradía."""

    match_date: date
    wins: int
    matches: int


@dataclass(slots=True)
class _PlayerHistory:
    """Mantiene los bloques mínimos necesarios para ambas ventanas de forma."""

    blocks: deque[_FormDateBlock] = field(default_factory=deque)
    retained_matches: int = 0
    last_match_date: date | None = None


@dataclass(slots=True)
class _HeadToHead:
    """Acumula victorias según el orden estable menor/mayor de player_id."""

    lower_wins: int = 0
    higher_wins: int = 0

    @property
    def matches(self) -> int:
        """Devuelve el número total de enfrentamientos acumulados."""

        return self.lower_wins + self.higher_wins


def _validate_date(value: object, *, field_name: str) -> date:
    """Exige un ``datetime.date`` estricto, sin componente horaria."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{field_name} debe ser datetime.date estricto.")
    return value


def _validate_gender(value: object) -> Gender:
    """Exige uno de los dos universos de género admitidos."""

    if value not in {"M", "F"}:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    return value  # type: ignore[return-value]


def _validate_player_id(value: object, *, field_name: str) -> int:
    """Exige un identificador entero no negativo y rechaza booleanos."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} debe ser un entero no negativo.")
    return value


def _validate_surface(value: object) -> Surface | None:
    """Exige una superficie canónica o ``None``."""

    if value is not None and value not in _SURFACES:
        raise ValueError(f"Superficie canónica desconocida: {value!r}.")
    return value  # type: ignore[return-value]


def _subtract_calendar_months(value: date, months: int) -> date:
    """Resta meses naturales y acota el día al último del mes de destino."""

    month_index = value.year * 12 + (value.month - 1) - months
    target_year, zero_based_month = divmod(month_index, 12)
    target_month = zero_based_month + 1
    target_day = min(value.day, monthrange(target_year, target_month)[1])
    return date(target_year, target_month, target_day)


def _win_rate(wins: int, matches: int) -> float:
    """Calcula proporción de victorias o ``NaN`` para un cold start."""

    if matches == 0:
        return math.nan
    return wins / matches


class CausalHistoryState:
    """Procesa forma, H2H y descanso en bloques cronológicos congelados."""

    def __init__(
        self,
        *,
        recent_matches: int = DEFAULT_RECENT_MATCHES,
        recent_months: int = DEFAULT_RECENT_MONTHS,
    ) -> None:
        """Inicializa universos vacíos con ventanas positivas configurables."""

        for name, value in (
            ("recent_matches", recent_matches),
            ("recent_months", recent_months),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} debe ser un entero positivo.")
        self.recent_matches = recent_matches
        self.recent_months = recent_months
        self._players: dict[tuple[Gender, int], _PlayerHistory] = {}
        self._h2h_global: dict[
            tuple[Gender, int, int],
            _HeadToHead,
        ] = {}
        self._h2h_surface: dict[
            tuple[Gender, int, int, Surface],
            _HeadToHead,
        ] = {}
        self._last_date: date | None = None

    @property
    def last_date(self) -> date | None:
        """Devuelve la última fecha aplicada, o ``None`` en cold start."""

        return self._last_date

    def _validate_snapshot_cutoff(self, as_of_date: date) -> None:
        """Impide consultar una fecha ya incorporada al estado."""

        if self._last_date is not None and as_of_date <= self._last_date:
            raise HistoryDateOrderError(
                "as_of_date debe ser posterior a la última fecha cerrada "
                f"({self._last_date.isoformat()}); el estado solo representa "
                "información estrictamente anterior al corte."
            )

    def _prune_player(
        self,
        history: _PlayerHistory,
        *,
        reference_date: date,
    ) -> None:
        """Descarta bloques innecesarios sin romper N ni la ventana mensual."""

        month_cutoff = _subtract_calendar_months(
            reference_date,
            self.recent_months,
        )
        while history.blocks:
            oldest = history.blocks[0]
            can_drop_for_n = (
                history.retained_matches - oldest.matches
                >= self.recent_matches
            )
            outside_month_window = oldest.match_date < month_cutoff
            if not can_drop_for_n or not outside_month_window:
                break
            history.blocks.popleft()
            history.retained_matches -= oldest.matches

    def _player_form(
        self,
        gender: Gender,
        player_id: int,
        *,
        as_of_date: date,
    ) -> tuple[float, int, float, int, int | None]:
        """Calcula ambas ventanas y descanso sin crear ni mutar estado."""

        history = self._players.get((gender, player_id))
        if history is None:
            return math.nan, 0, math.nan, 0, None

        month_cutoff = _subtract_calendar_months(
            as_of_date,
            self.recent_months,
        )
        recent_wins = 0
        recent_count = 0
        month_wins = 0
        month_count = 0

        for block in reversed(history.blocks):
            if recent_count < self.recent_matches:
                recent_wins += block.wins
                recent_count += block.matches
            if block.match_date >= month_cutoff:
                month_wins += block.wins
                month_count += block.matches
            if (
                recent_count >= self.recent_matches
                and block.match_date < month_cutoff
            ):
                break

        if history.last_match_date is None:
            rest_days = None
        else:
            rest_days = (as_of_date - history.last_match_date).days
        return (
            _win_rate(recent_wins, recent_count),
            recent_count,
            _win_rate(month_wins, month_count),
            month_count,
            rest_days,
        )

    @staticmethod
    def _pair_key(
        gender: Gender,
        player_a_id: int,
        player_b_id: int,
    ) -> tuple[Gender, int, int]:
        """Construye la clave H2H sin perder el universo de género."""

        lower_id, higher_id = sorted((player_a_id, player_b_id))
        return gender, lower_id, higher_id

    @staticmethod
    def _signed_h2h(
        record: _HeadToHead | None,
        *,
        player_a_id: int,
        player_b_id: int,
    ) -> tuple[float, int]:
        """Orienta el balance acumulado como victorias de A menos victorias B."""

        if record is None or record.matches == 0:
            return 0.0, 0
        lower_balance = (
            record.lower_wins - record.higher_wins
        ) / record.matches
        if player_a_id < player_b_id:
            return lower_balance, record.matches
        return -lower_balance, record.matches

    def snapshot(
        self,
        gender: Gender,
        player_a_id: int,
        player_b_id: int,
        *,
        surface: Surface | None,
        as_of_date: date,
    ) -> HistorySnapshot:
        """Devuelve forma, H2H y descanso construidos solo con fechas ``< D``."""

        cutoff = _validate_date(as_of_date, field_name="as_of_date")
        checked_gender = _validate_gender(gender)
        checked_a = _validate_player_id(
            player_a_id,
            field_name="player_a_id",
        )
        checked_b = _validate_player_id(
            player_b_id,
            field_name="player_b_id",
        )
        if checked_a == checked_b:
            raise ValueError("player_a_id y player_b_id deben ser distintos.")
        checked_surface = _validate_surface(surface)
        self._validate_snapshot_cutoff(cutoff)

        form_a = self._player_form(
            checked_gender,
            checked_a,
            as_of_date=cutoff,
        )
        form_b = self._player_form(
            checked_gender,
            checked_b,
            as_of_date=cutoff,
        )
        pair_key = self._pair_key(checked_gender, checked_a, checked_b)
        global_balance, global_count = self._signed_h2h(
            self._h2h_global.get(pair_key),
            player_a_id=checked_a,
            player_b_id=checked_b,
        )
        if checked_surface is None:
            surface_balance = None
            surface_count = 0
        else:
            surface_balance, surface_count = self._signed_h2h(
                self._h2h_surface.get((*pair_key, checked_surface)),
                player_a_id=checked_a,
                player_b_id=checked_b,
            )

        return HistorySnapshot(
            as_of_date=cutoff,
            gender=checked_gender,
            player_a_id=checked_a,
            player_b_id=checked_b,
            surface=checked_surface,
            recent_n_win_rate_a=form_a[0],
            recent_n_matches_a=form_a[1],
            recent_n_win_rate_b=form_b[0],
            recent_n_matches_b=form_b[1],
            recent_months_win_rate_a=form_a[2],
            recent_months_matches_a=form_a[3],
            recent_months_win_rate_b=form_b[2],
            recent_months_matches_b=form_b[3],
            h2h_global_balance=global_balance,
            h2h_global_matches=global_count,
            h2h_surface_balance=surface_balance,
            h2h_surface_matches=surface_count,
            rest_days_a=form_a[4],
            rest_days_b=form_b[4],
        )

    def _append_player_block(
        self,
        key: tuple[Gender, int],
        *,
        match_date: date,
        wins: int,
        matches: int,
    ) -> None:
        """Añade el agregado diario de un jugador y limita memoria retenida."""

        history = self._players.setdefault(key, _PlayerHistory())
        history.blocks.append(
            _FormDateBlock(
                match_date=match_date,
                wins=wins,
                matches=matches,
            )
        )
        history.retained_matches += matches
        history.last_match_date = match_date
        self._prune_player(history, reference_date=match_date)

    @staticmethod
    def _record_h2h_result(
        record: _HeadToHead,
        *,
        winner_id: int,
        lower_id: int,
    ) -> None:
        """Incrementa el lado correcto de un acumulador H2H."""

        if winner_id == lower_id:
            record.lower_wins += 1
        else:
            record.higher_wins += 1

    def apply_date_block(
        self,
        match_date: date,
        results: Iterable[HistoricalMatchResult],
    ) -> None:
        """Aplica atómicamente resultados de una única fecha posterior.

        El llamador debe invocar este método solo después de haber generado
        todos los snapshots de ``match_date``. Los resultados deben llegar ya
        filtrados y deduplicados por la capa de ingesta.
        """

        checked_date = _validate_date(match_date, field_name="match_date")
        if self._last_date is not None and checked_date <= self._last_date:
            raise HistoryDateOrderError(
                "Los bloques deben aplicarse en fechas estrictamente "
                f"crecientes; última={self._last_date.isoformat()}, "
                f"recibida={checked_date.isoformat()}."
            )
        materialized = tuple(results)
        if not materialized:
            raise ValueError("results no puede ser vacío.")
        for result in materialized:
            if not isinstance(result, HistoricalMatchResult):
                raise TypeError(
                    "Cada resultado debe ser HistoricalMatchResult."
                )
            if result.match_date != checked_date:
                raise HistoryDateOrderError(
                    "Todos los resultados deben pertenecer exactamente al "
                    f"bloque {checked_date.isoformat()}."
                )

        player_aggregates: dict[
            tuple[Gender, int],
            list[int],
        ] = defaultdict(lambda: [0, 0])
        for result in materialized:
            winner = player_aggregates[(result.gender, result.winner_id)]
            winner[0] += 1
            winner[1] += 1
            loser = player_aggregates[(result.gender, result.loser_id)]
            loser[1] += 1

        for key in sorted(player_aggregates):
            wins, matches = player_aggregates[key]
            self._append_player_block(
                key,
                match_date=checked_date,
                wins=wins,
                matches=matches,
            )

        for result in materialized:
            pair_key = self._pair_key(
                result.gender,
                result.winner_id,
                result.loser_id,
            )
            lower_id = pair_key[1]
            global_record = self._h2h_global.setdefault(
                pair_key,
                _HeadToHead(),
            )
            self._record_h2h_result(
                global_record,
                winner_id=result.winner_id,
                lower_id=lower_id,
            )
            if result.surface is not None:
                surface_record = self._h2h_surface.setdefault(
                    (*pair_key, result.surface),
                    _HeadToHead(),
                )
                self._record_h2h_result(
                    surface_record,
                    winner_id=result.winner_id,
                    lower_id=lower_id,
                )

        self._last_date = checked_date

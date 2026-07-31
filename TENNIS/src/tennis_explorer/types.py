"""Tipos, esquema tabular y excepciones de la ingesta de Tennis Explorer.

El módulo no realiza E/S. Centraliza el contrato público compartido por el
cliente HTTP y el parser para que ambos fallen de forma explícita y produzcan
el mismo esquema estable.
"""

from __future__ import annotations

from typing import Final, Literal


Gender = Literal["M", "F"]
TourLevel = Literal["ATP", "WTA", "Challenger", "ITF"]
Surface = Literal["Hard", "Clay", "Grass", "Carpet"]
MatchStatus = Literal[
    "scheduled",
    "in_progress",
    "finished",
    "walkover",
    "cancelled",
    "unknown",
]
WinnerSide = Literal["player_1", "player_2"]

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "match_date",
    "tournament",
    "tournament_href",
    "tour_level",
    "gender",
    "surface",
    "scheduled_time",
    "player_1_name",
    "player_2_name",
    "player_1_href",
    "player_2_href",
    "player_1_slug",
    "player_2_slug",
    "player_1_has_link",
    "player_2_has_link",
    "player_1_odds",
    "player_2_odds",
    "status",
    "status_evidence",
    "player_1_sets_won",
    "player_2_sets_won",
    "sets_score",
    "winner_side",
    "winner_slug",
    "result_evidence",
    "source_match_id",
    "match_detail_href",
)

OUTPUT_DTYPES: Final[dict[str, str]] = {
    "match_date": "datetime64[ns]",
    "tournament": "string",
    "tournament_href": "string",
    "tour_level": "string",
    "gender": "string",
    "surface": "string",
    "scheduled_time": "string",
    "player_1_name": "string",
    "player_2_name": "string",
    "player_1_href": "string",
    "player_2_href": "string",
    "player_1_slug": "string",
    "player_2_slug": "string",
    "player_1_has_link": "boolean",
    "player_2_has_link": "boolean",
    "player_1_odds": "Float64",
    "player_2_odds": "Float64",
    "status": "string",
    "status_evidence": "string",
    "player_1_sets_won": "Int64",
    "player_2_sets_won": "Int64",
    "sets_score": "string",
    "winner_side": "string",
    "winner_slug": "string",
    "result_evidence": "string",
    "source_match_id": "string",
    "match_detail_href": "string",
}


class TennisExplorerError(RuntimeError):
    """Representa cualquier fallo controlado de la ingesta Tennis Explorer."""


class TennisExplorerSchemaError(TennisExplorerError):
    """Indica que el HTML no coincide con la estructura real inspeccionada."""


class TennisExplorerCacheError(TennisExplorerError):
    """Indica que un snapshot cacheado está incompleto o no es fiable."""


class TennisExplorerHttpError(TennisExplorerError):
    """Indica un error HTTP no atribuible inequívocamente a un bloqueo."""


class TennisExplorerBlockedError(TennisExplorerHttpError):
    """Indica que Tennis Explorer o su WAF parecen haber bloqueado la petición."""

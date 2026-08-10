"""API pública del sistema Elo causal por género y superficie."""

from .build import (
    EloBuildReport,
    GenderBuildAudit,
    build_elo_database,
    iter_manifest_match_frames,
    load_verified_manifest,
    prepare_manifest_event_frame,
)
from .engine import (
    EXCLUDED_LEVELS,
    SURFACES,
    DateBlockError,
    EloEngine,
    EloEngineError,
)
from .events import (
    EventColumns,
    EventValidationError,
    UnknownSurfaceError,
    events_from_dataframe,
    normalise_surface,
)
from .parameters import (
    ALGORITHM_VERSION,
    DEFAULT_ELO_PARAMETERS,
    EloParameters,
)
from .service import EloQuery, get_elo, get_elos
from .store import EloStore, EloStoreError, EloValidationError
from .types import (
    AuditCounts,
    DateBlockResult,
    EloRunResult,
    EloSnapshot,
    EventDecision,
    EventProvenance,
    ExclusionReason,
    Gender,
    MatchEvent,
    PlayerEloState,
    PlayerPreMatchRating,
    PreviewBlockResult,
    RatedMatch,
    Surface,
)


__all__ = [
    "ALGORITHM_VERSION",
    "AuditCounts",
    "DEFAULT_ELO_PARAMETERS",
    "DateBlockError",
    "DateBlockResult",
    "EXCLUDED_LEVELS",
    "EloBuildReport",
    "EloEngine",
    "EloEngineError",
    "EloParameters",
    "EloQuery",
    "EloRunResult",
    "EloSnapshot",
    "EloStore",
    "EloStoreError",
    "EloValidationError",
    "EventColumns",
    "EventDecision",
    "EventProvenance",
    "EventValidationError",
    "ExclusionReason",
    "Gender",
    "GenderBuildAudit",
    "MatchEvent",
    "PlayerEloState",
    "PlayerPreMatchRating",
    "PreviewBlockResult",
    "RatedMatch",
    "SURFACES",
    "Surface",
    "UnknownSurfaceError",
    "events_from_dataframe",
    "build_elo_database",
    "get_elo",
    "get_elos",
    "iter_manifest_match_frames",
    "load_verified_manifest",
    "normalise_surface",
    "prepare_manifest_event_frame",
]

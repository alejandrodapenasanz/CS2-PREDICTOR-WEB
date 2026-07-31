"""API pública del mapeo auditable entre Tennis Explorer y Sackmann."""

from .candidates import (
    active_window_start,
    build_active_candidate_index,
    load_active_candidate_index,
)
from .normalization import (
    build_sackmann_name_key,
    normalize_name_text,
    parse_visible_name,
)
from .pipeline import (
    MAPPING_OUTPUT_COLUMNS,
    REQUIRED_SCRAPER_COLUMNS,
    load_unresolved_players,
    resolve_scraped_matches,
    summarize_mapping_coverage,
)
from .review import (
    OVERRIDE_COLUMNS,
    UNRESOLVED_COLUMNS,
    OverrideRecord,
    PlayerMappingReviewLockError,
    UnresolvedObservation,
    load_overrides,
    update_unresolved_queue,
)
from .store import (
    MappingRecord,
    PlayerMappingStore,
    PlayerMappingStoreError,
)
from .types import (
    ActiveCandidate,
    CandidateIndex,
    Gender,
    NameKey,
    NameParsingError,
    PlayerMappingError,
    PlayerMappingSchemaError,
    PlayerMappingSourceError,
    PlayerMappingValidationError,
)


__all__ = [
    "MAPPING_OUTPUT_COLUMNS",
    "OVERRIDE_COLUMNS",
    "REQUIRED_SCRAPER_COLUMNS",
    "UNRESOLVED_COLUMNS",
    "ActiveCandidate",
    "CandidateIndex",
    "Gender",
    "MappingRecord",
    "NameKey",
    "NameParsingError",
    "OverrideRecord",
    "PlayerMappingError",
    "PlayerMappingReviewLockError",
    "PlayerMappingSchemaError",
    "PlayerMappingSourceError",
    "PlayerMappingStore",
    "PlayerMappingStoreError",
    "PlayerMappingValidationError",
    "UnresolvedObservation",
    "active_window_start",
    "build_active_candidate_index",
    "build_sackmann_name_key",
    "load_active_candidate_index",
    "load_overrides",
    "load_unresolved_players",
    "normalize_name_text",
    "parse_visible_name",
    "resolve_scraped_matches",
    "summarize_mapping_coverage",
    "update_unresolved_queue",
]

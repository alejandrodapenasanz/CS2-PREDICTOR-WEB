"""API pública de features causales y datasets de entrenamiento."""

from .dataset import (
    FeatureDatasetBuildReport,
    FeatureDatasetError,
    GenderDatasetAudit,
    RankingBuildAudit,
    VerifiedAuxiliaryInventory,
    build_training_datasets,
    verify_auxiliary_source_inventory,
    verify_stable_source_inventory,
)
from .levels import (
    TourLevelContext,
    UnknownTourLevelError,
    normalize_tour_level,
)
from .market import (
    InvalidOddsError,
    MarketProbabilities,
    MarketTimestampError,
    calculate_two_way_market_probabilities,
    validate_market_timestamp,
)
from .orientation import (
    DEFAULT_ORIENTATION_SEED,
    Orientation,
    OrientationError,
    orient_match,
)
from .parameters import (
    DEFAULT_FEATURE_PARAMETERS,
    FEATURE_SCHEMA_VERSION,
    FeatureParameters,
)
from .players import (
    PlayerAgeError,
    PlayerAgeIndex,
    PlayerAgeSnapshot,
)
from .rankings import (
    RankingDataError,
    RankingIndex,
    RankingSnapshot,
)
from .schema import (
    AUDIT_COLUMNS,
    TARGET_COLUMN,
    TRAINING_COLUMNS,
)
from .state import (
    CausalHistoryState,
    HistoricalMatchResult,
    HistorySnapshot,
)
from .tennisratio_stats import (
    TennisRatioStatsFeatureError,
    TennisRatioStatsSnapshot,
    build_tennisratio_stats_snapshot,
)
from .vector import (
    MODEL_FEATURE_COLUMNS,
    VECTOR_COLUMNS,
    EloFeatureProvider,
    EloFeatureSnapshot,
    FeatureVectorError,
    MatchFeatureBuilder,
    MatchFeatureRequest,
    MatchFeatureVector,
    RankingFeatureProvider,
    assemble_match_feature_vector,
)


__all__ = [
    "AUDIT_COLUMNS",
    "CausalHistoryState",
    "DEFAULT_FEATURE_PARAMETERS",
    "DEFAULT_ORIENTATION_SEED",
    "EloFeatureProvider",
    "EloFeatureSnapshot",
    "FEATURE_SCHEMA_VERSION",
    "FeatureDatasetBuildReport",
    "FeatureDatasetError",
    "FeatureParameters",
    "FeatureVectorError",
    "GenderDatasetAudit",
    "HistoricalMatchResult",
    "HistorySnapshot",
    "InvalidOddsError",
    "MODEL_FEATURE_COLUMNS",
    "MarketProbabilities",
    "MarketTimestampError",
    "MatchFeatureBuilder",
    "MatchFeatureRequest",
    "MatchFeatureVector",
    "Orientation",
    "OrientationError",
    "PlayerAgeError",
    "PlayerAgeIndex",
    "PlayerAgeSnapshot",
    "RankingBuildAudit",
    "RankingDataError",
    "RankingFeatureProvider",
    "RankingIndex",
    "RankingSnapshot",
    "TARGET_COLUMN",
    "TennisRatioStatsFeatureError",
    "TennisRatioStatsSnapshot",
    "TRAINING_COLUMNS",
    "TourLevelContext",
    "UnknownTourLevelError",
    "VECTOR_COLUMNS",
    "VerifiedAuxiliaryInventory",
    "assemble_match_feature_vector",
    "build_training_datasets",
    "build_tennisratio_stats_snapshot",
    "calculate_two_way_market_probabilities",
    "normalize_tour_level",
    "orient_match",
    "validate_market_timestamp",
    "verify_auxiliary_source_inventory",
    "verify_stable_source_inventory",
]

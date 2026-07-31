"""API pública del pipeline de predicción diaria."""

from .confidence import (
    MAX_HISTORY_AGE_DAYS,
    MAX_RANKING_AGE_DAYS,
    MIN_GENERAL_MATCHES,
    MIN_SURFACE_MATCHES,
    ConfidenceAssessment,
    assess_vector_confidence,
    unavailable_confidence,
)
from .context import (
    DailyContextError,
    DailyFeatureContext,
    GenderFeatureContext,
    build_daily_feature_context,
    rebuild_history_state,
)
from .pipeline import (
    PREDICTION_OUTPUT_COLUMNS,
    DailyPredictionError,
    DailyPredictionRun,
    predict_mapped_matches,
    publish_predictions_csv,
    run_daily_prediction_pipeline,
)


__all__ = [
    "MAX_HISTORY_AGE_DAYS",
    "MAX_RANKING_AGE_DAYS",
    "MIN_GENERAL_MATCHES",
    "MIN_SURFACE_MATCHES",
    "PREDICTION_OUTPUT_COLUMNS",
    "ConfidenceAssessment",
    "DailyContextError",
    "DailyFeatureContext",
    "DailyPredictionError",
    "DailyPredictionRun",
    "GenderFeatureContext",
    "assess_vector_confidence",
    "build_daily_feature_context",
    "predict_mapped_matches",
    "publish_predictions_csv",
    "rebuild_history_state",
    "run_daily_prediction_pipeline",
    "unavailable_confidence",
]

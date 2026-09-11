"""Centraliza las rutas base del proyecto a partir de este archivo."""

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
HTTP_CACHE_DIR: Final[Path] = RAW_DATA_DIR / "_http_cache"
HTTP_CLIENT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "http_client.json"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
ELO_PROCESSED_DIR: Final[Path] = PROCESSED_DATA_DIR / "elo"
ELO_DATABASE_PATH: Final[Path] = ELO_PROCESSED_DIR / "elo.sqlite3"
# ``features`` contiene cuatro artefactos heredados con una ACL de Windows
# irrecuperable tras la migración del repositorio. La ruta activa se mantiene
# separada para que ningún build dependa de poder sobrescribir esos archivos.
FEATURES_PROCESSED_DIR: Final[Path] = PROCESSED_DATA_DIR / "features_active"
FEATURE_DATASET_MANIFEST_PATH: Final[Path] = (
    FEATURES_PROCESSED_DIR / "manifest.json"
)
PREDICTIONS_PROCESSED_DIR: Final[Path] = (
    PROCESSED_DATA_DIR / "predictions"
)
IDENTITY_QUARANTINE_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "identity_quarantine.csv"
)
IDENTITY_QUARANTINE_MANIFEST_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "identity_quarantine.manifest.json"
)
PLAYER_MAPPING_DATABASE_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "player_mapping.sqlite3"
)
UNRESOLVED_PLAYERS_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "unresolved_players.csv"
)
PLAYER_OVERRIDES_PATH: Final[Path] = DATA_DIR / "overrides.csv"
SACKMANN_ATP_RAW_DIR: Final[Path] = RAW_DATA_DIR / "atp"
SACKMANN_WTA_RAW_DIR: Final[Path] = RAW_DATA_DIR / "wta"
SACKMANN_ACTIVE_MANIFEST_PATH: Final[Path] = (
    RAW_DATA_DIR / "sackmann_manifest.json"
)
SACKMANN_MANIFESTS_DIR: Final[Path] = RAW_DATA_DIR / "sackmann_manifests"
SACKMANN_BASE_COMMIT: Final[str] = (
    "83733587353df8a41f2fd4f516147d5aa83f5a8d"
)
SACKMANN_MANIFEST_PATH: Final[Path] = (
    SACKMANN_MANIFESTS_DIR / f"{SACKMANN_BASE_COMMIT}.json"
)
ELO_HANDOFF_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "config" / "elo_handoff.json"
)
ELO_OPERATIONAL_OMISSIONS_PATH: Final[Path] = (
    ELO_PROCESSED_DIR / "operational_omissions.json"
)
SURFACE_CATALOG_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "surface_catalog.json"
)

MATCH_CHARTING_RAW_DIR: Final[Path] = (
    RAW_DATA_DIR / "tennis_MatchChartingProject"
)
MATCH_CHARTING_MANIFEST_PATH: Final[Path] = (
    MATCH_CHARTING_RAW_DIR / "manifest.json"
)
MATCH_CHARTING_MANIFESTS_DIR: Final[Path] = (
    MATCH_CHARTING_RAW_DIR / "manifests"
)

TENNIS_ABSTRACT_RAW_DIR: Final[Path] = RAW_DATA_DIR / "tennis_abstract"
TENNIS_ABSTRACT_ELO_RAW_DIR: Final[Path] = (
    TENNIS_ABSTRACT_RAW_DIR / "elo"
)
TENNIS_ABSTRACT_ELO_MANIFEST_PATH: Final[Path] = (
    TENNIS_ABSTRACT_ELO_RAW_DIR / "manifest.json"
)

TENNIS_EXPLORER_RAW_DIR: Final[Path] = RAW_DATA_DIR / "tennis_explorer"
TENNIS_EXPLORER_DAILY_RAW_DIR: Final[Path] = (
    TENNIS_EXPLORER_RAW_DIR / "daily"
)
TENNIS_EXPLORER_RESULTS_RAW_DIR: Final[Path] = (
    TENNIS_EXPLORER_RAW_DIR / "results"
)
TENNIS_EXPLORER_POLICY_RAW_DIR: Final[Path] = (
    TENNIS_EXPLORER_RAW_DIR / "policy"
)

SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
BBDD_DIR: Final[Path] = PROJECT_ROOT / "BBDD"
OPERATIONS_DATABASE_PATH: Final[Path] = BBDD_DIR / "tennis.sqlite3"
PHASE7_MODELS_DIR: Final[Path] = MODELS_DIR / "phase7"
PHASE7_RUNS_DIR: Final[Path] = PHASE7_MODELS_DIR / "runs"
PHASE7_ACTIVE_MANIFEST_PATH: Final[Path] = (
    PHASE7_MODELS_DIR / "manifest.json"
)
PHASE7_LAST_GOOD_POINTER_PATH: Final[Path] = (
    PHASE7_MODELS_DIR / "last_good.json"
)
PHASE7_PROMOTION_DIR: Final[Path] = PHASE7_MODELS_DIR / "promotion"
PHASE7_PROMOTION_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "config" / "model_promotion.json"
)
PHASE7_CLEANUP_PENDING_PATH: Final[Path] = (
    PHASE7_MODELS_DIR / "cleanup_pending.json"
)
PHASE7_RETENTION_APPROVAL_PATH: Final[Path] = (
    PHASE7_MODELS_DIR / "retention_approval.json"
)

"""Centraliza las rutas base del proyecto a partir de este archivo."""

import json
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
STATE_ROOT: Final[Path] = PROJECT_ROOT.parent / "VAULT" / "TENNIS"


@lru_cache(maxsize=4)
def _recorded_roots(vault: Path) -> tuple[Path, ...]:
    """Read the migration/location contracts once, not once per historical row."""
    roots: list[Path] = []
    journal = vault / "migration.json"
    if journal.is_file():
        roots.append(Path(json.loads(journal.read_text(encoding="utf-8"))["origin_root"]))
    locations = vault / "locations.json"
    if locations.is_file():
        roots.extend(
            Path(value) for value in json.loads(locations.read_text(encoding="utf-8"))["roots"]
        )
    return tuple(roots)


def resolve_state_reference(value: str) -> Path:
    """Relocate recorded data paths without rewriting any historical metadata.

    Only this domain and the exact origins recorded by the file migration are
    accepted. Relative metadata paths retain their original domain-relative
    meaning. Traversal and unrelated absolute paths are rejected.
    """
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if ".." in path.parts:
        raise ValueError("State reference contains traversal")
    if not path.is_absolute() and not PureWindowsPath(value).drive:
        relative = path
    else:
        bases = [STATE_ROOT, PROJECT_ROOT]
        for origin in _recorded_roots(STATE_ROOT.parent):
            bases.extend((origin / "TENNIS", origin / "VAULT" / "TENNIS"))
        relative = None
        for base in bases:
            try:
                relative = path.relative_to(base)
                break
            except ValueError:
                continue
        if relative is None:
            raise ValueError("State reference is outside the recorded TENNIS roots")
    if relative is None:
        raise ValueError("Missing relative state reference")
    result = (STATE_ROOT / relative).resolve()
    if not result.is_relative_to(STATE_ROOT.resolve()):
        raise ValueError("State reference escapes VAULT/TENNIS")
    return result


def relocated_data_path(value: str | Path) -> Path:
    """Rebase stored production data references; preserve explicit fixture roots.

    Containment, hashes and as-of checks remain the caller's responsibility.
    Unrelated absolute paths are not guessed or mapped by basename.
    """
    original = Path(value).resolve()
    try:
        relocated = resolve_state_reference(str(value))
    except ValueError:
        return original
    relative = relocated.relative_to(STATE_ROOT.resolve())
    if relative.parts and relative.parts[0] in {"data", "models", "BBDD", "logs"}:
        return relocated
    return original


DATA_DIR: Final[Path] = STATE_ROOT / "data"
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
FEATURE_DATASET_MANIFEST_PATH: Final[Path] = FEATURES_PROCESSED_DIR / "manifest.json"
PREDICTIONS_PROCESSED_DIR: Final[Path] = PROCESSED_DATA_DIR / "predictions"
IDENTITY_QUARANTINE_PATH: Final[Path] = PROCESSED_DATA_DIR / "identity_quarantine.csv"
IDENTITY_QUARANTINE_MANIFEST_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "identity_quarantine.manifest.json"
)
PLAYER_MAPPING_DATABASE_PATH: Final[Path] = PROCESSED_DATA_DIR / "player_mapping.sqlite3"
UNRESOLVED_PLAYERS_PATH: Final[Path] = PROCESSED_DATA_DIR / "unresolved_players.csv"
PLAYER_OVERRIDES_PATH: Final[Path] = PROJECT_ROOT / "data" / "overrides.csv"
SACKMANN_ATP_RAW_DIR: Final[Path] = RAW_DATA_DIR / "atp"
SACKMANN_WTA_RAW_DIR: Final[Path] = RAW_DATA_DIR / "wta"
SACKMANN_ACTIVE_MANIFEST_PATH: Final[Path] = RAW_DATA_DIR / "sackmann_manifest.json"
SACKMANN_MANIFESTS_DIR: Final[Path] = RAW_DATA_DIR / "sackmann_manifests"
SACKMANN_BASE_COMMIT: Final[str] = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
SACKMANN_MANIFEST_PATH: Final[Path] = SACKMANN_MANIFESTS_DIR / f"{SACKMANN_BASE_COMMIT}.json"
ELO_HANDOFF_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "elo_handoff.json"
ELO_OPERATIONAL_OMISSIONS_PATH: Final[Path] = ELO_PROCESSED_DIR / "operational_omissions.json"
SURFACE_CATALOG_PATH: Final[Path] = PROCESSED_DATA_DIR / "surface_catalog.json"

MATCH_CHARTING_RAW_DIR: Final[Path] = RAW_DATA_DIR / "tennis_MatchChartingProject"
MATCH_CHARTING_MANIFEST_PATH: Final[Path] = MATCH_CHARTING_RAW_DIR / "manifest.json"
MATCH_CHARTING_MANIFESTS_DIR: Final[Path] = MATCH_CHARTING_RAW_DIR / "manifests"

TENNIS_ABSTRACT_RAW_DIR: Final[Path] = RAW_DATA_DIR / "tennis_abstract"
TENNIS_ABSTRACT_ELO_RAW_DIR: Final[Path] = TENNIS_ABSTRACT_RAW_DIR / "elo"
TENNIS_ABSTRACT_ELO_MANIFEST_PATH: Final[Path] = TENNIS_ABSTRACT_ELO_RAW_DIR / "manifest.json"

TENNIS_EXPLORER_RAW_DIR: Final[Path] = RAW_DATA_DIR / "tennis_explorer"
TENNIS_EXPLORER_DAILY_RAW_DIR: Final[Path] = TENNIS_EXPLORER_RAW_DIR / "daily"
TENNIS_EXPLORER_RESULTS_RAW_DIR: Final[Path] = TENNIS_EXPLORER_RAW_DIR / "results"
TENNIS_EXPLORER_POLICY_RAW_DIR: Final[Path] = TENNIS_EXPLORER_RAW_DIR / "policy"

SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
MODELS_DIR: Final[Path] = STATE_ROOT / "models"
BBDD_DIR: Final[Path] = STATE_ROOT / "BBDD"
OPERATIONS_DATABASE_PATH: Final[Path] = BBDD_DIR / "tennis.sqlite3"
PHASE7_MODELS_DIR: Final[Path] = MODELS_DIR / "phase7"
PHASE7_RUNS_DIR: Final[Path] = PHASE7_MODELS_DIR / "runs"
PHASE7_ACTIVE_MANIFEST_PATH: Final[Path] = PHASE7_MODELS_DIR / "manifest.json"
PHASE7_LAST_GOOD_POINTER_PATH: Final[Path] = PHASE7_MODELS_DIR / "last_good.json"
PHASE7_PROMOTION_DIR: Final[Path] = PHASE7_MODELS_DIR / "promotion"
PHASE7_PROMOTION_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "model_promotion.json"
PHASE7_CLEANUP_PENDING_PATH: Final[Path] = PHASE7_MODELS_DIR / "cleanup_pending.json"
PHASE7_RETENTION_APPROVAL_PATH: Final[Path] = PHASE7_MODELS_DIR / "retention_approval.json"

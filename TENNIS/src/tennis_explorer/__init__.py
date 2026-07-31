"""API pública para descargar y parsear agendas de Tennis Explorer."""

from .client import (
    TennisExplorerResultSnapshot,
    build_daily_url,
    get_daily_matches,
    refresh_daily_results,
)
from .parser import empty_matches_dataframe, parse_daily_matches_html
from .types import (
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    TennisExplorerBlockedError,
    TennisExplorerCacheError,
    TennisExplorerError,
    TennisExplorerHttpError,
    TennisExplorerSchemaError,
)


__all__ = [
    "OUTPUT_COLUMNS",
    "OUTPUT_DTYPES",
    "TennisExplorerBlockedError",
    "TennisExplorerCacheError",
    "TennisExplorerError",
    "TennisExplorerHttpError",
    "TennisExplorerResultSnapshot",
    "TennisExplorerSchemaError",
    "build_daily_url",
    "empty_matches_dataframe",
    "get_daily_matches",
    "parse_daily_matches_html",
    "refresh_daily_results",
]

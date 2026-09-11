"""Recover discarded schedule metadata from authenticated immutable raw bytes.

No source row is rewritten. The caller first applies publication/as-of filters;
the original capture timestamp is reused only after SHA and identity checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import gzip
import hashlib
import logging
from pathlib import Path
import sqlite3
from typing import cast

import pandas as pd

from .parser import SCHEDULE_CONTEXT_COLUMNS, parse_agenda_html
from .types import Gender, TennisRatioSchemaError


CONTEXT_COLUMNS = SCHEDULE_CONTEXT_COLUMNS
_LOG = logging.getLogger(__name__)


def restore_agenda_context(frame: pd.DataFrame, database_path: Path) -> pd.DataFrame:
    """Enrich an already selected agenda without changing its causal selection.

    SQL reads only paths for the exact original hashes. Facts are recovered
    from those authenticated bytes, not a newer agenda/result for the match.
    """

    if frame.empty:
        return frame.copy()
    paths = {}
    with sqlite3.connect(
        Path(database_path).resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        for sha in frame["snapshot_sha256"].dropna().unique():
            row = connection.execute(
                "SELECT compressed_path FROM source_snapshots WHERE sha256 = ?",
                (str(sha),),
            ).fetchone()
            if row is not None:
                paths[str(sha)] = Path(row[0])
    recovered = recover_agenda_context(frame.to_dict(orient="records"), paths)
    result = frame.copy()
    for column in CONTEXT_COLUMNS:
        result[column] = pd.array([row.get(column, pd.NA) for row in recovered], dtype="string")
    return result


def recover_agenda_context(
    payloads: Sequence[dict[str, object]],
    snapshot_paths: Mapping[str, Path],
) -> list[dict[str, object]]:
    """Reparse each retained snapshot once; missing/corrupt raw stays unknown."""

    cache: dict[str, dict[str, dict[str, object]]] = {}
    result = []
    for payload in payloads:
        row = dict(payload)
        result.append(row)
        if all(column in row for column in CONTEXT_COLUMNS) and isinstance(
            row.get("tournament_level_source"), str
        ):
            continue
        sha = str(row["snapshot_sha256"])
        path = snapshot_paths.get(sha)
        if sha not in cache:
            cache[sha] = {}
            try:
                if path is None:
                    raise ValueError("raw snapshot path missing")
                raw = gzip.decompress(path.read_bytes())
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise ValueError("raw snapshot SHA mismatch")
                parsed = parse_agenda_html(
                    raw,
                    gender=cast(Gender, str(row["gender"])),
                    source_url=str(row["source_url"]),
                    retrieved_at_utc=datetime.fromisoformat(
                        str(row["retrieved_at_utc"]).replace("Z", "+00:00")
                    ),
                    snapshot_sha256=sha,
                )
                cache[sha] = {
                    str(item["source_match_id"]): item for item in parsed.to_dict(orient="records")
                }
            except (OSError, ValueError, EOFError, TennisRatioSchemaError) as exc:
                _LOG.warning("Schedule metadata unavailable for SHA %s: %s", sha, exc)
        recovered = cache[sha].get(str(row["source_match_id"]))
        if recovered is None:
            continue
        if any(
            str(row[key]) != str(recovered[key])
            for key in ("gender", "player_1_slug", "player_2_slug", "tournament")
        ):
            _LOG.warning("Schedule metadata identity mismatch: %s", row["source_match_id"])
            continue
        if str(row["match_date"])[:10] != str(recovered["match_date"])[:10]:
            continue
        for column in CONTEXT_COLUMNS:
            row[column] = recovered[column]
    return result

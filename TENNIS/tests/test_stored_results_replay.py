"""Regression for unchanged mapped rows with changing replay audit metadata.

All evidence is synthetic and persisted only in a disposable test database.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.config import TESTS_DIR
from src.operations import OperationsStore
from src.operations.result_sources import TennisExplorerMappedResultSnapshot
from src.operations.stored_results import (
    _mapped_metadata,
    reconcile_stored_tennis_explorer_results,
)
from tests.test_operations_daily import NOW, _cross_source_result_snapshot, _prediction_frame


def test_empty_replay_metadata_change_is_additive_and_repeatable() -> None:
    """A legacy empty run does not block revised counts or a new raw capture."""

    day = date(2026, 7, 29)
    raw = _cross_source_result_snapshot(day)
    base = TennisExplorerMappedResultSnapshot(
        match_date=day,
        retrieved_at_utc=raw.retrieved_at_utc,
        source_url=raw.source_url,
        snapshot_sha256="a" * 64,
        matches=raw.matches.iloc[:0].copy(),
        html_path=raw.html_path,
        metadata_path=raw.metadata_path,
        raw_snapshot_sha256=raw.snapshot_sha256,
        official_pending_count=2,
        terminal_result_count=1,
        exact_id_count=0,
        identity_mapped_count=0,
        paired_inferred_count=0,
        unmatched_official_count=2,
        unmapped_player_rows=2,
        ambiguous_pair_rows=0,
    )
    revised = replace(base, official_pending_count=1, unmatched_official_count=1)
    with TemporaryDirectory(dir=TESTS_DIR) as directory:
        database = Path(directory) / "tennis.sqlite3"
        legacy = f"observation-remap:{day}:aaaaaaaaaaaa"
        with OperationsStore(database, clock=lambda: NOW) as store:
            store.register_prediction_run(
                "prediction", _prediction_frame(day, "te:pending"), target_date=day
            )
            store.reconcile_observations(
                legacy,
                base.matches,
                metadata=_mapped_metadata("raw", base),
                observed_at_utc=raw.retrieved_at_utc,
                target_date=day,
            )
            sacred_before = {
                table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
                for table in ("predictions", "observations", "settlements")
            }
        with (
            patch(
                "src.operations.stored_results._load_latest_raw_snapshot", return_value=("raw", raw)
            ),
            patch(
                "src.operations.stored_results.build_tennis_explorer_mapped_result_snapshot",
                return_value=revised,
            ),
        ):
            first = reconcile_stored_tennis_explorer_results(NOW.date(), database_path=database)
            second = reconcile_stored_tennis_explorer_results(NOW.date(), database_path=database)
        assert not first[0].reconciliation.reused
        assert second[0].reconciliation.reused
        with OperationsStore(database, clock=lambda: NOW) as store:
            for table, rows in sacred_before.items():
                assert [
                    tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")
                ] == rows
            assert (
                store.connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE run_id=?", (legacy,)
                ).fetchone()[0]
                == 1
            )
            assert store.connection.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0

"""Tests integrados del manifiesto, build SQLite y garantía anti-fugas Elo."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders import MATCH_SOURCE_COLUMNS  # noqa: E402
from src.elo.build import (  # noqa: E402
    EXPECTED_SOURCE_REPOSITORY,
    SourceIntegrityError,
    build_elo_database,
    load_verified_manifest,
)
from src.elo.service import get_elo  # noqa: E402
from src.elo.store import EloStore  # noqa: E402


def _git_blob_sha(payload: bytes) -> str:
    """Calcula el SHA-1 Git del fixture exactamente como el manifiesto."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _match_row(
    *,
    match_date: str,
    match_num: str,
    winner_id: str,
    loser_id: str,
    score: str = "6-4 6-4",
) -> dict[str, str]:
    """Construye una fila mínima con las 49 columnas reales Sackmann."""

    row = {column: "" for column in MATCH_SOURCE_COLUMNS}
    row.update(
        {
            "tourney_id": f"2024-{match_num}",
            "tourney_name": "Fixture Open",
            "surface": "Hard",
            "draw_size": "32",
            "tourney_level": "A",
            "tourney_date": match_date,
            "match_num": match_num,
            "winner_id": winner_id,
            "winner_name": f"Winner {winner_id}",
            "loser_id": loser_id,
            "loser_name": f"Loser {loser_id}",
            "score": score,
            "best_of": "3",
            "round": "R32",
        }
    )
    return row


def _write_snapshot(
    root: Path,
    *,
    rows: list[dict[str, str]],
    commit: str,
) -> tuple[Path, Path]:
    """Escribe CSV y manifiesto reproducibles dentro del fixture temporal."""

    raw_dir = root / "raw"
    match_path = raw_dir / "atp" / "atp_matches_2024.csv"
    match_path.parent.mkdir(parents=True, exist_ok=True)
    with match_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(MATCH_SOURCE_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = match_path.read_bytes()
    manifest = {
        "source_repository": EXPECTED_SOURCE_REPOSITORY,
        "source_commit": commit,
        "files": [
            {
                "remote_path": "atp/atp_matches_2024.csv",
                "local_path": "atp/atp_matches_2024.csv",
                "category": "atp_main",
                "year": 2024,
                "size": len(payload),
                "git_blob_sha": _git_blob_sha(payload),
            }
        ],
    }
    manifest_path = raw_dir / "sackmann_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return raw_dir, manifest_path


class EloManifestPipelineTest(unittest.TestCase):
    """Valida integridad de entrada, publicación e idempotencia."""

    def _temporary_project(self) -> TemporaryDirectory[str]:
        """Crea artefactos efímeros exclusivamente dentro de TENNIS/tests."""

        return TemporaryDirectory(
            prefix=".elo-pipeline-",
            dir=Path(__file__).resolve().parent,
        )

    def test_manifest_detects_local_bytes_changed_after_snapshot(self) -> None:
        """Rechaza un CSV cuyo tamaño o blob ya no coincide con el manifiesto."""

        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_snapshot(
                root,
                rows=[
                    _match_row(
                        match_date="20240101",
                        match_num="1",
                        winner_id="1",
                        loser_id="2",
                    )
                ],
                commit="a" * 40,
            )
            match_path = raw_dir / "atp" / "atp_matches_2024.csv"
            with match_path.open("a", encoding="utf-8") as destination:
                destination.write("\n")

            with self.assertRaises(SourceIntegrityError):
                load_verified_manifest(manifest_path, raw_dir)

    def test_future_append_does_not_change_earlier_as_of_rating(self) -> None:
        """Prueba que añadir D futura conserva exactamente el Elo as-of previo."""

        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            first_rows = [
                _match_row(
                    match_date="20240101",
                    match_num="1",
                    winner_id="1",
                    loser_id="2",
                ),
                _match_row(
                    match_date="20240102",
                    match_num="2",
                    winner_id="1",
                    loser_id="3",
                ),
            ]
            raw_dir, manifest_path = _write_snapshot(
                root,
                rows=first_rows,
                commit="b" * 40,
            )
            first_database = root / "first.sqlite3"
            first_report = build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=first_database,
                gender="M",
                chunksize=1,
            )
            persisted_run = EloStore(first_database).resolve_complete_run(
                gender="M",
            )
            code_inventory = persisted_run.parameters.get(
                "artifact_code_inventory"
            )
            self.assertIsInstance(code_inventory, list)
            self.assertIn(
                "scripts/build_elo.py",
                {entry["path"] for entry in code_inventory},
            )
            before_append = get_elo(
                gender="M",
                player_id=1,
                surface="Hard",
                as_of_date=date(2024, 1, 24),
                db_path=first_database,
            )
            repeated = build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=first_database,
                gender="M",
                chunksize=2,
            )
            self.assertFalse(first_report.skipped)
            self.assertTrue(repeated.skipped)

            future_rows = first_rows + [
                _match_row(
                    match_date="20240103",
                    match_num="3",
                    winner_id="4",
                    loser_id="1",
                )
            ]
            raw_dir, manifest_path = _write_snapshot(
                root,
                rows=future_rows,
                commit="c" * 40,
            )
            extended_database = root / "extended.sqlite3"
            build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=extended_database,
                gender="M",
                chunksize=2,
            )
            after_append = get_elo(
                gender="M",
                player_id=1,
                surface="hard",
                as_of_date=date(2024, 1, 24),
                db_path=extended_database,
            )
            after_future_match = get_elo(
                gender="M",
                player_id=1,
                surface="Hard",
                as_of_date=date(2024, 1, 25),
                db_path=extended_database,
            )

            comparable_fields = (
                "general_elo",
                "surface_elo_raw",
                "combined_elo",
                "general_matches",
                "surface_matches",
                "state_date",
            )
            for field in comparable_fields:
                self.assertEqual(
                    getattr(before_append, field),
                    getattr(after_append, field),
                )
            self.assertNotEqual(
                after_append.general_elo,
                after_future_match.general_elo,
            )

    def test_deduplication_uses_raw_cells_before_typing(self) -> None:
        """Distingue IDs raw distintos aunque ambos se tipen al mismo entero."""

        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            first = _match_row(
                match_date="20240101",
                match_num="1",
                winner_id="1",
                loser_id="2",
            )
            leading_zero = dict(first)
            leading_zero["winner_id"] = "01"
            raw_dir, manifest_path = _write_snapshot(
                root,
                rows=[first, leading_zero, dict(first)],
                commit="d" * 40,
            )
            database = root / "raw-hash.sqlite3"
            report = build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=database,
                gender="M",
                chunksize=1,
            )

            self.assertEqual(len(report.audits), 1)
            audit = report.audits[0]
            self.assertEqual(audit.source_rows, 3)
            self.assertEqual(audit.included_events, 2)
            self.assertEqual(audit.exact_duplicates, 1)
            snapshot = get_elo(
                gender="M",
                player_id=1,
                surface="Hard",
                as_of_date=date(2024, 1, 23),
                db_path=database,
            )
            self.assertEqual(snapshot.general_matches, 2)
            self.assertEqual(snapshot.surface_matches, 2)

    def test_default_build_does_not_select_history_by_dob_metadata(
        self,
    ) -> None:
        """El contrato productivo vacío conserva todo el histórico deportivo."""

        rows = [
            _match_row(
                match_date="20240101",
                match_num="1",
                winner_id="1",
                loser_id="2",
            ),
            _match_row(
                match_date="20240102",
                match_num="2",
                winner_id="1",
                loser_id="3",
            ),
            _match_row(
                match_date="20240103",
                match_num="3",
                winner_id="1",
                loser_id="4",
            ),
        ]
        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_snapshot(
                root,
                rows=rows,
                commit="e" * 40,
            )
            default_database = root / "default.sqlite3"
            default = build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=default_database,
                gender="M",
                chunksize=1,
            )
            explicit_empty = build_elo_database(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                database_path=root / "explicit-empty.sqlite3",
                gender="M",
                chunksize=2,
                identity_exclusion_after_dates={},
            )
            snapshot = get_elo(
                gender="M",
                player_id=1,
                surface="Hard",
                as_of_date=date(2024, 1, 25),
                db_path=default_database,
            )
            persisted = EloStore(default_database).resolve_complete_run(
                gender="M"
            )

        audit = default.audits[0]
        self.assertEqual(audit.source_rows, 3)
        self.assertEqual(audit.included_events, 3)
        self.assertEqual(audit.excluded_events, 0)
        self.assertNotIn("excluded_identity", audit.exclusions_by_reason)
        self.assertEqual(snapshot.general_matches, 3)
        self.assertEqual(
            persisted.parameters["identity_exclusion_after_dates"],
            [],
        )
        self.assertEqual(
            default.input_fingerprints["M"],
            explicit_empty.input_fingerprints["M"],
        )


if __name__ == "__main__":
    unittest.main()

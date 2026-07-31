"""Tests offline del spool causal y acotado en memoria de la fase 6."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders import MATCH_SOURCE_COLUMNS  # noqa: E402
from src.elo import (  # noqa: E402
    iter_manifest_match_frames,
    load_verified_manifest,
    prepare_manifest_event_frame,
)
from src.elo.build import EXPECTED_SOURCE_REPOSITORY  # noqa: E402
from src.features.source import (  # noqa: E402
    FEATURE_SOURCE_COLUMNS,
    FEATURE_SPOOL_INDEX,
    FeatureSourceValidationError,
    iter_spooled_feature_date_frames,
    spool_manifest_feature_rows,
)


def _git_blob_sha(payload: bytes) -> str:
    """Calcula el SHA-1 Git del fixture exactamente como en fase 3."""

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
    best_of: str = "3",
    tourney_name: str = "Fixture Open",
) -> dict[str, str]:
    """Construye una fila de partido con las 49 columnas Sackmann reales."""

    row = {column: "" for column in MATCH_SOURCE_COLUMNS}
    row.update(
        {
            "tourney_id": f"2024-{match_num}",
            "tourney_name": tourney_name,
            "surface": "Hard",
            "draw_size": "32",
            "tourney_level": "A",
            "tourney_date": match_date,
            "match_num": match_num,
            "winner_id": winner_id,
            "winner_name": f"Winner {winner_id}",
            "loser_id": loser_id,
            "loser_name": f"Loser {loser_id}",
            "score": "6-4 6-4",
            "best_of": best_of,
            "round": "R32",
        }
    )
    return row


def _write_csv(
    raw_dir: Path,
    *,
    relative_path: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    """Escribe un CSV raw y devuelve su entrada exacta de manifiesto."""

    destination = raw_dir.joinpath(*relative_path.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(MATCH_SOURCE_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = destination.read_bytes()
    is_atp = relative_path.startswith("atp/")
    return {
        "remote_path": relative_path,
        "local_path": relative_path,
        "category": "atp_main" if is_atp else "wta_main",
        "year": 2024,
        "size": len(payload),
        "git_blob_sha": _git_blob_sha(payload),
    }


def _write_snapshot(
    root: Path,
    *,
    male_rows: list[dict[str, str]],
    female_rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    """Crea un snapshot local verificable sin ninguna dependencia de red."""

    raw_dir = root / "raw"
    files = [
        _write_csv(
            raw_dir,
            relative_path="atp/atp_matches_2024.csv",
            rows=male_rows,
        )
    ]
    if female_rows is not None:
        files.append(
            _write_csv(
                raw_dir,
                relative_path="wta/wta_matches_2024.csv",
                rows=female_rows,
            )
        )
    manifest_path = raw_dir / "sackmann_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_repository": EXPECTED_SOURCE_REPOSITORY,
                "source_commit": "a" * 40,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_dir, manifest_path


class FeatureSourceSpoolTest(unittest.TestCase):
    """Valida identidad, causalidad, dtypes y aislamiento por género."""

    def _temporary_project(self) -> TemporaryDirectory[str]:
        """Crea artefactos efímeros únicamente dentro de TENNIS/tests."""

        return TemporaryDirectory(
            prefix=".feature-source-",
            dir=Path(__file__).resolve().parent,
        )

    def test_emits_complete_dates_across_chunks_in_stable_order(self) -> None:
        """No divide una fecha aunque cruce chunks y ordena por procedencia."""

        rows = [
            _match_row(
                match_date="20240102",
                match_num="1",
                winner_id="10",
                loser_id="11",
            ),
            _match_row(
                match_date="20240101",
                match_num="2",
                winner_id="20",
                loser_id="21",
                best_of="5",
                tourney_name="First date row",
            ),
            _match_row(
                match_date="20240101",
                match_num="3",
                winner_id="30",
                loser_id="31",
                best_of="",
            ),
            _match_row(
                match_date="20240101",
                match_num="4",
                winner_id="40",
                loser_id="41",
            ),
            _match_row(
                match_date="20240103",
                match_num="5",
                winner_id="50",
                loser_id="51",
            ),
        ]
        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_snapshot(
                root,
                male_rows=rows,
                female_rows=[
                    _match_row(
                        match_date="20240101",
                        match_num="9",
                        winner_id="90",
                        loser_id="91",
                    )
                ],
            )
            manifest = load_verified_manifest(manifest_path, raw_dir)
            spool_path = root / "features.spool.sqlite3"

            inserted = spool_manifest_feature_rows(
                manifest,
                gender="M",
                spool_path=spool_path,
                chunksize=2,
            )
            blocks = list(
                iter_spooled_feature_date_frames(
                    spool_path,
                    gender="M",
                    chunksize=2,
                )
            )

            self.assertEqual(inserted, 5)
            self.assertEqual([len(block) for block in blocks], [3, 1, 1])
            self.assertEqual(
                [block["tourney_date"].iloc[0].date().isoformat()
                 for block in blocks],
                ["2024-01-01", "2024-01-02", "2024-01-03"],
            )
            self.assertEqual(
                blocks[0]["source_row_number"].tolist(),
                [2, 3, 4],
            )
            self.assertEqual(
                blocks[0]["winner_id"].tolist(),
                [20, 30, 40],
            )
            self.assertEqual(list(blocks[0].columns), list(FEATURE_SOURCE_COLUMNS))
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(
                    blocks[0]["tourney_date"]
                )
            )
            self.assertEqual(str(blocks[0]["best_of"].dtype), "Int64")
            self.assertEqual(int(blocks[0]["best_of"].iloc[0]), 5)
            self.assertTrue(pd.isna(blocks[0]["best_of"].iloc[1]))
            self.assertEqual(
                list(
                    iter_spooled_feature_date_frames(
                        spool_path,
                        gender="F",
                        chunksize=1,
                    )
                ),
                [],
            )

            connection = sqlite3.connect(spool_path)
            try:
                journal_mode = connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()[0]
                indexes = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA index_list(_feature_input_rows)"
                    ).fetchall()
                }
            finally:
                connection.close()
            self.assertEqual(str(journal_mode).lower(), "delete")
            self.assertIn(FEATURE_SPOOL_INDEX, indexes)

    def test_preserves_phase3_identity_and_match_context(self) -> None:
        """Conserva hash, procedencia y contexto de la misma fila preparada."""

        row = _match_row(
            match_date="20240229",
            match_num="77",
            winner_id="101",
            loser_id="202",
            best_of="3",
            tourney_name="Identity Championships",
        )
        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_snapshot(
                root,
                male_rows=[row],
            )
            manifest = load_verified_manifest(manifest_path, raw_dir)
            raw_frame = next(
                iter_manifest_match_frames(
                    manifest,
                    gender="M",
                    chunksize=1,
                )
            )
            expected = prepare_manifest_event_frame(raw_frame).iloc[0]
            spool_path = root / "identity.sqlite3"

            spool_manifest_feature_rows(
                manifest,
                gender="M",
                spool_path=spool_path,
                chunksize=1,
            )
            observed = next(
                iter_spooled_feature_date_frames(
                    spool_path,
                    gender="M",
                    chunksize=1,
                )
            ).iloc[0]

            for column in (
                "source_commit",
                "source_path",
                "source_row_number",
                "source_record_hash",
                "tourney_id",
                "tourney_name",
                "match_num",
                "surface",
                "tourney_level",
                "round",
                "winner_id",
                "loser_id",
            ):
                self.assertEqual(observed[column], expected[column])
            self.assertRegex(
                str(observed["source_record_hash"]),
                r"^[0-9a-f]{64}$",
            )

    def test_rejects_invalid_best_of_without_inventing_a_value(self) -> None:
        """Falla claramente si best_of presente no es un entero positivo."""

        for invalid_value in ("0", "-3", "three", "3.0"):
            with self.subTest(best_of=invalid_value):
                with self._temporary_project() as temporary_directory:
                    root = Path(temporary_directory)
                    raw_dir, manifest_path = _write_snapshot(
                        root,
                        male_rows=[
                            _match_row(
                                match_date="20240101",
                                match_num="1",
                                winner_id="1",
                                loser_id="2",
                                best_of=invalid_value,
                            )
                        ],
                    )
                    manifest = load_verified_manifest(
                        manifest_path,
                        raw_dir,
                    )

                    with self.assertRaisesRegex(
                        FeatureSourceValidationError,
                        "best_of",
                    ):
                        spool_manifest_feature_rows(
                            manifest,
                            gender="M",
                            spool_path=root / "invalid.sqlite3",
                            chunksize=1,
                        )

    def test_rejects_invalid_gender_and_chunksize(self) -> None:
        """No acepta universos mezclados ni tamaños de chunk ambiguos."""

        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_snapshot(
                root,
                male_rows=[
                    _match_row(
                        match_date="20240101",
                        match_num="1",
                        winner_id="1",
                        loser_id="2",
                    )
                ],
            )
            manifest = load_verified_manifest(manifest_path, raw_dir)

            with self.assertRaisesRegex(ValueError, "gender"):
                spool_manifest_feature_rows(
                    manifest,
                    gender="all",  # type: ignore[arg-type]
                    spool_path=root / "wrong-gender.sqlite3",
                )
            with self.assertRaisesRegex(ValueError, "chunksize"):
                spool_manifest_feature_rows(
                    manifest,
                    gender="M",
                    spool_path=root / "wrong-chunk.sqlite3",
                    chunksize=True,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()

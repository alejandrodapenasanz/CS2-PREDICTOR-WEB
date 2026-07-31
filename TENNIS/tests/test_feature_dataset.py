"""Pruebas integradas offline del dataset histórico causal en Parquet."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders import (  # noqa: E402
    MATCH_SOURCE_COLUMNS,
    PLAYER_SOURCE_COLUMNS,
)
from src.elo.build import EXPECTED_SOURCE_REPOSITORY  # noqa: E402
from src.config import FEATURES_PROCESSED_DIR  # noqa: E402
from src.features.dataset import (  # noqa: E402
    FeatureDatasetError,
    build_training_datasets,
)
from src.features.schema import TRAINING_COLUMNS  # noqa: E402


TESTS_ROOT = Path(__file__).resolve().parent


def _git_blob_sha(payload: bytes) -> str:
    """Calcula el SHA-1 Git exacto de un fixture."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _manifest_item(
    path: Path,
    raw_dir: Path,
    *,
    category: str,
    year: int | None = None,
) -> dict[str, object]:
    """Construye metadata de manifiesto para un archivo ya escrito."""

    payload = path.read_bytes()
    relative = path.relative_to(raw_dir).as_posix()
    item: dict[str, object] = {
        "remote_path": relative,
        "local_path": relative,
        "category": category,
        "size": len(payload),
        "git_blob_sha": _git_blob_sha(payload),
    }
    if year is not None:
        item["year"] = year
    return item


def _match_row(
    *,
    match_date: str,
    match_num: int,
    winner_id: int,
    loser_id: int,
    surface: str = "Hard",
) -> dict[str, str]:
    """Crea una fila elegible con las 49 columnas Sackmann."""

    row = {column: "" for column in MATCH_SOURCE_COLUMNS}
    row.update(
        {
            "tourney_id": "2024-TEST",
            "tourney_name": "Causal Fixture",
            "surface": surface,
            "draw_size": "32",
            "tourney_level": "A",
            "tourney_date": match_date,
            "match_num": str(match_num),
            "winner_id": str(winner_id),
            "winner_name": f"Winner {winner_id}",
            "loser_id": str(loser_id),
            "loser_name": f"Loser {loser_id}",
            "score": "6-4 6-4",
            "best_of": "3",
            "round": "R32",
        }
    )
    return row


def _write_csv_dicts(
    path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    """Escribe un CSV UTF-8 determinista con salto LF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture_snapshot(
    root: Path,
    *,
    matches: list[dict[str, str]],
) -> tuple[Path, Path]:
    """Crea partidos, jugadores, rankings y manifiesto ATP verificables."""

    raw_dir = root / "raw"
    match_path = raw_dir / "atp" / "atp_matches_2024.csv"
    _write_csv_dicts(
        match_path,
        fieldnames=MATCH_SOURCE_COLUMNS,
        rows=matches,
    )

    involved_ids = sorted(
        {
            int(row[column])
            for row in matches
            for column in ("winner_id", "loser_id")
        }
    )
    player_rows = []
    for player_id in involved_ids:
        player_rows.append(
            {
                "player_id": str(player_id),
                "name_first": "Player",
                "name_last": str(player_id),
                "hand": "R",
                "dob": "19900101",
                "ioc": "ESP",
                "height": "180",
                "wikidata_id": "",
            }
        )
    players_path = raw_dir / "atp" / "atp_players.csv"
    _write_csv_dicts(
        players_path,
        fieldnames=PLAYER_SOURCE_COLUMNS,
        rows=player_rows,
    )

    rankings_path = raw_dir / "atp" / "atp_rankings_fixture.csv"
    ranking_rows = [
        {
            "ranking_date": "20231225",
            "rank": str(index + 1),
            "player": str(player_id),
            "points": str(5000 - index),
        }
        for index, player_id in enumerate(involved_ids)
    ]
    _write_csv_dicts(
        rankings_path,
        fieldnames=("ranking_date", "rank", "player", "points"),
        rows=ranking_rows,
    )

    manifest_path = raw_dir / "sackmann_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_repository": EXPECTED_SOURCE_REPOSITORY,
                "source_commit": "a" * 40,
                "files": [
                    _manifest_item(
                        match_path,
                        raw_dir,
                        category="atp_main",
                        year=2024,
                    ),
                    _manifest_item(
                        players_path,
                        raw_dir,
                        category="players",
                    ),
                    _manifest_item(
                        rankings_path,
                        raw_dir,
                        category="atp_rankings",
                    ),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_dir, manifest_path


class FeatureDatasetIntegrationTest(unittest.TestCase):
    """Comprueba publicación, no-fuga, balance e idempotencia."""

    def _temporary_project(self) -> TemporaryDirectory[str]:
        """Crea un proyecto efímero únicamente dentro de TENNIS/tests."""

        return TemporaryDirectory(
            prefix=".feature-dataset-",
            dir=TESTS_ROOT,
        )

    def test_partial_build_cannot_replace_canonical_two_gender_manifest(
        self,
    ) -> None:
        """Exige un directorio diagnóstico antes de leer o escribir fuentes."""

        with self.assertRaisesRegex(
            FeatureDatasetError,
            "construcción parcial",
        ):
            build_training_datasets(
                manifest_path=TESTS_ROOT / "not-read.json",
                raw_dir=TESTS_ROOT,
                output_dir=FEATURES_PROCESSED_DIR,
                gender="M",
            )

    def test_future_match_does_not_change_existing_training_features(
        self,
    ) -> None:
        """Añadir un resultado posterior conserva exactamente las filas previas."""

        base_matches = [
            _match_row(
                match_date="20240101",
                match_num=1,
                winner_id=1,
                loser_id=2,
            ),
            _match_row(
                match_date="20240110",
                match_num=2,
                winner_id=2,
                loser_id=3,
                surface="Clay",
            ),
        ]
        future = _match_row(
            match_date="20240201",
            match_num=3,
            winner_id=3,
            loser_id=1,
        )
        with self._temporary_project() as base_directory:
            base_root = Path(base_directory)
            raw_dir, manifest_path = _write_fixture_snapshot(
                base_root,
                matches=base_matches,
            )
            report = build_training_datasets(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                output_dir=base_root / "processed",
                gender="M",
                chunksize=1,
                parquet_buffer_rows=1,
            )
            baseline = pd.read_parquet(report.datasets[0].output_path)
            skipped = build_training_datasets(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                output_dir=base_root / "processed",
                gender="M",
                chunksize=2,
                parquet_buffer_rows=2,
            )
            self.assertTrue(skipped.skipped)

            with self._temporary_project() as extended_directory:
                extended_root = Path(extended_directory)
                raw_extended, manifest_extended = _write_fixture_snapshot(
                    extended_root,
                    matches=[*base_matches, future],
                )
                extended_report = build_training_datasets(
                    manifest_path=manifest_extended,
                    raw_dir=raw_extended,
                    output_dir=extended_root / "processed",
                    gender="M",
                    chunksize=2,
                    parquet_buffer_rows=2,
                )
                extended = pd.read_parquet(
                    extended_report.datasets[0].output_path
                )

        old_records = set(baseline["record_id"])
        comparable = extended.loc[
            extended["record_id"].isin(old_records)
        ].sort_values("record_id").reset_index(drop=True)
        expected = baseline.sort_values(
            "record_id"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            comparable,
            expected,
            check_like=False,
        )
        self.assertEqual(tuple(expected.columns), TRAINING_COLUMNS)
        self.assertEqual(report.datasets[0].training_rows, 2)
        self.assertEqual(extended_report.datasets[0].training_rows, 3)

    def test_persisted_labels_are_balanced_and_source_roles_are_absent(
        self,
    ) -> None:
        """Verifica ~50/50 en el artefacto y evita winner/loser como columnas."""

        matches = [
            _match_row(
                match_date="20240101",
                match_num=index,
                winner_id=10_000 + index,
                loser_id=20_000 + index,
            )
            for index in range(1, 1001)
        ]
        with self._temporary_project() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, manifest_path = _write_fixture_snapshot(
                root,
                matches=matches,
            )
            report = build_training_datasets(
                manifest_path=manifest_path,
                raw_dir=raw_dir,
                output_dir=root / "processed",
                gender="M",
                chunksize=137,
                parquet_buffer_rows=211,
            )
            dataset = pd.read_parquet(report.datasets[0].output_path)

        y_rate = float(dataset["y"].mean())
        self.assertGreater(y_rate, 0.46)
        self.assertLess(y_rate, 0.54)
        self.assertAlmostEqual(y_rate, report.datasets[0].y_rate)
        self.assertNotIn("winner_id", dataset.columns)
        self.assertNotIn("loser_id", dataset.columns)
        self.assertNotIn("swapped", dataset.columns)
        self.assertNotIn("orientation_hash", dataset.columns)
        self.assertEqual(set(dataset["y"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()

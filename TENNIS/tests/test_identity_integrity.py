"""Pruebas de detección y persistencia de identidades Sackmann ambiguas."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.config import PROJECT_ROOT
from src.identity_integrity import (
    IdentityConflict,
    IdentityIntegrityError,
    detect_identity_conflicts,
    load_identity_quarantine,
    publish_identity_quarantine,
)


class IdentityIntegrityTests(unittest.TestCase):
    """Valida el corte conservador, el commit y los hashes de cuarentena."""

    def test_detection_flags_only_match_before_tenth_birthday(self) -> None:
        """No intenta dividir carreras: pone en cuarentena la clave completa."""

        manifest = type(
            "Manifest",
            (),
            {
                "source_commit": "a" * 40,
                "match_files": (
                    type("Source", (), {"gender": "M"})(),
                ),
            },
        )()
        players = {
            ("M", 1): ("Ambiguous Player", date(2000, 6, 1)),
            ("M", 2): ("Plausible Player", date(2000, 6, 1)),
        }
        appearances = {
            ("M", 1): (date(2005, 1, 1), date(2024, 1, 1), 20),
            ("M", 2): (date(2012, 1, 1), date(2024, 1, 1), 20),
        }
        with (
            patch(
                "src.identity_integrity.load_verified_manifest",
                return_value=manifest,
            ),
            patch(
                "src.identity_integrity._player_metadata",
                return_value=players,
            ),
            patch(
                "src.identity_integrity._scan_appearances",
                return_value=appearances,
            ),
        ):
            commit, conflicts = detect_identity_conflicts()

        self.assertEqual(commit, "a" * 40)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].player_id, 1)
        self.assertEqual(conflicts[0].reason, "match_before_tenth_birthday")

    def test_publication_round_trip_and_tamper_detection(self) -> None:
        """El CSV solo es aceptado si coincide con su manifiesto y commit."""

        conflict = IdentityConflict(
            gender="F",
            player_id=99,
            player_name="Test Player",
            birth_date=date(2002, 1, 1),
            first_match_date=date(1999, 1, 1),
            last_match_date=date(2025, 1, 1),
            earliest_age_years=-3.0,
            appearances=12,
        )
        with TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            csv_path = root / "identity.csv"
            manifest_path = root / "identity.manifest.json"
            with patch(
                "src.identity_integrity.detect_identity_conflicts",
                return_value=("b" * 40, (conflict,)),
            ):
                report = publish_identity_quarantine(
                    csv_path=csv_path,
                    quarantine_manifest_path=manifest_path,
                    clock=lambda: datetime(2026, 7, 30, 8, tzinfo=UTC),
                )
            loaded = load_identity_quarantine(
                expected_source_commit="b" * 40,
                csv_path=csv_path,
                manifest_path=manifest_path,
            )
            self.assertEqual(report.keys, frozenset({("F", 99)}))
            self.assertEqual(loaded.conflicts, (conflict,))

            csv_path.write_bytes(csv_path.read_bytes() + b"\n")
            with self.assertRaises(IdentityIntegrityError):
                load_identity_quarantine(
                    expected_source_commit="b" * 40,
                    csv_path=csv_path,
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()

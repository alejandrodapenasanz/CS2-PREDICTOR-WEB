"""Pruebas offline del esquema estable del dataset de entrenamiento."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.schema import (  # noqa: E402
    AUDIT_COLUMNS,
    FeatureSchemaError,
    TARGET_COLUMN,
    TRAINING_COLUMNS,
    validate_training_row,
)
from src.features.vector import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    VECTOR_COLUMNS,
)


class FeatureSchemaTest(unittest.TestCase):
    """Comprueba separación de metadata, features y objetivo."""

    def test_training_order_has_audit_vector_and_target_once(self) -> None:
        """Fija una única aparición de cada columna y deja y al final."""

        self.assertEqual(
            TRAINING_COLUMNS,
            (*AUDIT_COLUMNS, *VECTOR_COLUMNS, TARGET_COLUMN),
        )
        self.assertEqual(len(TRAINING_COLUMNS), len(set(TRAINING_COLUMNS)))
        self.assertEqual(TRAINING_COLUMNS[-1], "y")

    def test_model_allowlist_excludes_provenance_identity_and_target(
        self,
    ) -> None:
        """Impide seleccionar columnas de auditoría por el contrato público."""

        self.assertTrue(set(MODEL_FEATURE_COLUMNS).issubset(VECTOR_COLUMNS))
        self.assertTrue(set(MODEL_FEATURE_COLUMNS).isdisjoint(AUDIT_COLUMNS))
        self.assertNotIn(TARGET_COLUMN, MODEL_FEATURE_COLUMNS)
        self.assertNotIn("player_a_id", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("player_b_id", MODEL_FEATURE_COLUMNS)

    def test_row_validator_rejects_order_schema_and_label_errors(self) -> None:
        """Detecta columnas desordenadas, ausencias y y booleana."""

        row = {column: 0 for column in TRAINING_COLUMNS}
        row["record_id"] = "a" * 64
        row["source_commit"] = "b" * 40
        row["source_path"] = "atp/matches.csv"
        row["gender"] = "M"
        row["match_date"] = object()
        row["tour_level_raw"] = "A"
        row["tour_level"] = "ATP Tour"
        row["source_family"] = "atp_main"
        row["y"] = 1
        validate_training_row(row)

        wrong_order = dict(reversed(tuple(row.items())))
        with self.assertRaises(FeatureSchemaError):
            validate_training_row(wrong_order)
        missing = dict(row)
        missing.pop("edge")
        with self.assertRaises(FeatureSchemaError):
            validate_training_row(missing)
        invalid_target = dict(row)
        invalid_target["y"] = True
        with self.assertRaises(FeatureSchemaError):
            validate_training_row(invalid_target)


if __name__ == "__main__":
    unittest.main()

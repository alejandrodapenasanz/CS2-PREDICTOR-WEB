from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model.features import (  # noqa: E402
    SEGMENT_INTERACTION_FEATURE_COLUMNS,
    segment_interaction_features,
)
from run_segment_interaction_ablation import (  # noqa: E402
    MINIMAL_SWISS_COLUMNS,
    VARIANT_COLUMNS,
    validate_common_gate,
)


def _variant_payload(
    holdout_sha256: str = "common-cohort",
    *,
    challenger_log_loss: float = 0.61,
) -> dict[str, Any]:
    return {
        "metrics": {"log_loss": challenger_log_loss, "brier": 0.21},
        "gate": {
            "holdout_sha256": holdout_sha256,
            "incumbent": {"log_loss": 0.62, "brier": 0.22},
            "challenger": {
                "log_loss": challenger_log_loss,
                "brier": 0.21,
            },
        },
    }


def test_minimal_variant_contains_only_the_swiss_interaction() -> None:
    assert MINIMAL_SWISS_COLUMNS == ("elo_diff_x_stage_swiss",)
    assert VARIANT_COLUMNS["minimal_swiss"] == MINIMAL_SWISS_COLUMNS
    assert set(MINIMAL_SWISS_COLUMNS) < set(SEGMENT_INTERACTION_FEATURE_COLUMNS)
    assert VARIANT_COLUMNS["full_interactions"] == tuple(SEGMENT_INTERACTION_FEATURE_COLUMNS)


def test_swiss_interaction_uses_only_frozen_elo_and_stage() -> None:
    swiss = segment_interaction_features({"elo_diff": 125.0, "context_stage_swiss": 1.0})
    non_swiss = segment_interaction_features({"elo_diff": 125.0, "context_stage_group": 1.0})
    reversed_swiss = segment_interaction_features({"elo_diff": -125.0, "context_stage_swiss": 1.0})

    assert swiss[MINIMAL_SWISS_COLUMNS[0]] == 125.0
    assert non_swiss[MINIMAL_SWISS_COLUMNS[0]] == 0.0
    assert reversed_swiss[MINIMAL_SWISS_COLUMNS[0]] == -125.0


def test_gate_rejects_cohort_or_metric_asymmetry() -> None:
    incumbent = {"log_loss": 0.62, "brier": 0.22}
    payloads = {variant: _variant_payload() for variant in ("control_refit", "full_interactions", "minimal_swiss")}

    assert validate_common_gate(payloads, incumbent) == "common-cohort"

    payloads["minimal_swiss"] = _variant_payload("different-cohort")
    with pytest.raises(RuntimeError, match="one cohort"):
        validate_common_gate(payloads, incumbent)

    payloads["minimal_swiss"] = _variant_payload(challenger_log_loss=0.6101)
    payloads["minimal_swiss"]["metrics"]["log_loss"] = 0.61
    with pytest.raises(RuntimeError, match="differs inside and outside"):
        validate_common_gate(payloads, incumbent)

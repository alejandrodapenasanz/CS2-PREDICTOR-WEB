from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from MODEL.cs2model.artifacts import Component, ModelArtifact
from MODEL.cs2model.uncertainty import (
    UNCERTAINTY_METHOD,
    count_history_strictly_before_day,
    estimate_probability_uncertainty,
)
from PIPELINE import enrich_predictions


class _DirectionalEstimator:
    def __init__(self, slope: float) -> None:
        self.slope = slope

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.clip(0.5 + self.slope * matrix[:, 0], 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - positive, positive])


class _ConstantEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def test_low_history_widens_band_and_lowers_confidence() -> None:
    established = estimate_probability_uncertainty(
        0.62,
        0.02,
        team1_history_matches=50,
        team2_history_matches=45,
        component_weights=[0.5, 0.3, 0.2],
    )
    sparse = estimate_probability_uncertainty(
        0.62,
        0.02,
        team1_history_matches=2,
        team2_history_matches=1,
        component_weights=[0.5, 0.3, 0.2],
    )

    assert established.confidence_level == "high"
    assert sparse.confidence_level == "low"
    assert sparse.band_half_width > established.band_half_width
    assert sparse.history_coverage < established.history_coverage


def test_same_day_and_future_appends_do_not_change_asof_coverage_or_band() -> None:
    as_of = datetime(2026, 8, 10, 18, 0)
    prefix = [datetime(2026, 8, 1), datetime(2026, 8, 9)]
    same_day_and_future = [datetime(2026, 8, 10, 8, 0), datetime(2026, 8, 11)]

    before = count_history_strictly_before_day(prefix, as_of)
    after_append = count_history_strictly_before_day([*prefix, *same_day_and_future], as_of)
    assert before == after_append == 2

    baseline = estimate_probability_uncertainty(
        0.61,
        0.03,
        team1_history_matches=before,
        team2_history_matches=before,
        component_weights=[0.6, 0.4],
    )
    appended = estimate_probability_uncertainty(
        0.61,
        0.03,
        team1_history_matches=after_append,
        team2_history_matches=after_append,
        component_weights=[0.6, 0.4],
    )
    assert appended == baseline


def test_member_dispersion_is_computed_after_order_symmetrization() -> None:
    artifact = ModelArtifact(
        feature_columns=["strength"],
        components=[
            Component("steep", _DirectionalEstimator(0.2), None, 0.5),
            Component("mild", _DirectionalEstimator(0.1), None, 0.5),
        ],
    )
    forward = {"strength": 1.0}
    reverse = {"strength": -0.5}

    mean, disagreement = artifact.predict_symmetric_proba_team1_with_uncertainty([forward], [reverse])
    swapped_mean, swapped_disagreement = artifact.predict_symmetric_proba_team1_with_uncertainty([reverse], [forward])

    # Member estimates are 0.65 and 0.575 after symmetrization.
    assert mean[0] == pytest.approx(0.6125)
    assert disagreement[0] == pytest.approx(0.0375)
    assert swapped_mean[0] == pytest.approx(1.0 - mean[0])
    assert swapped_disagreement[0] == pytest.approx(disagreement[0])


def test_new_disagreement_is_separate_from_legacy_staking_value(monkeypatch) -> None:
    artifact = ModelArtifact(
        feature_columns=["strength"],
        components=[
            Component("high", _ConstantEstimator(0.9), None, 0.5),
            Component("low", _ConstantEstimator(0.1), None, 0.5),
        ],
    )
    monkeypatch.setattr(
        enrich_predictions,
        "_match_feature_pair",
        lambda *_args, **_kwargs: ({"strength": 1.0}, {"strength": -1.0}),
    )

    probability, disagreement, legacy_disagreement = enrich_predictions.model_probability_and_uncertainty(
        {"artifact": artifact},
        "alpha",
        "beta",
        datetime(2026, 8, 11),
        "event",
        "bo3",
    )

    # Directional members disagree by 0.4, but both symmetric estimates are 0.5.
    assert probability == pytest.approx(0.5)
    assert disagreement == pytest.approx(0.0)
    assert legacy_disagreement == pytest.approx(0.4)


def test_single_effective_member_does_not_claim_zero_uncertainty() -> None:
    result = estimate_probability_uncertainty(
        0.5,
        0.0,
        team1_history_matches=100,
        team2_history_matches=100,
        component_weights=[1.0, 0.0],
    )

    assert result.effective_members == 1
    assert result.band_half_width == pytest.approx(0.05)
    assert result.confidence_level == "low"


def test_prediction_contract_is_bounded_and_versioned() -> None:
    result = estimate_probability_uncertainty(
        0.98,
        0.08,
        team1_history_matches=12,
        team2_history_matches=15,
        component_weights=[0.7, 0.3],
    )
    fields = result.prediction_fields()

    assert result.lower >= 0.0
    assert result.upper == 1.0
    assert fields["estimate_uncertainty_method"] == UNCERTAINTY_METHOD
    assert fields["estimate_band_half_width"] == pytest.approx(result.band_half_width, abs=1e-6)
    assert math.isclose(float(fields["ensemble_disagreement"]), 0.08)


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_invalid_disagreement_fails_closed(value: float) -> None:
    with pytest.raises(ValueError):
        estimate_probability_uncertainty(
            0.5,
            value,
            team1_history_matches=10,
            team2_history_matches=10,
            component_weights=[1.0, 1.0],
        )

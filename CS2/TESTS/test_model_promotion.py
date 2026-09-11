from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from MODEL.cs2model import promotion as promotion_module
from MODEL.cs2model.promotion import (
    DeploymentRecoveryError,
    PromotionDecision,
    PromotionValidationError,
    assert_same_artifact_same_cohort_metrics,
    bootstrap_candidate,
    compare_candidate_to_incumbent,
    promote_candidate,
    read_model_pointer,
    reference_for_version,
    rollback_last_good,
    sha256_file,
    write_model_pointer,
)


LIVE_VERSION = "20260101_000000Z"
CHALLENGER_VERSION = "20260201_000000Z"


class FeatureProbabilityModel:
    def __init__(self, cutoff: str = "2026-01-01") -> None:
        self.metadata = {"date_max": cutoff}

    def predict_proba_team1(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray([row["incumbent_probability"] for row in feature_rows], dtype=float)


def test_mutating_deployment_api_requires_compare_and_swap_evidence() -> None:
    promotion_parameters = inspect.signature(promote_candidate).parameters
    bootstrap_parameters = inspect.signature(bootstrap_candidate).parameters

    assert promotion_parameters["expected_incumbent"].default is inspect.Parameter.empty
    assert promotion_parameters["expected_candidate_sha256"].default is inspect.Parameter.empty
    assert bootstrap_parameters["expected_candidate_sha256"].default is inspect.Parameter.empty


def cohort(
    incumbent_probabilities: list[float],
    challenger_probabilities: list[float],
    labels: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    challenger: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    for index, (incumbent, candidate, actual) in enumerate(
        zip(incumbent_probabilities, challenger_probabilities, labels, strict=True),
        start=2,
    ):
        match_id = f"match-{index}"
        match_date = f"2026-01-{index:02d}"
        challenger.append(
            {
                "match_id": match_id,
                "date": match_date,
                "actual": actual,
                "prob_team1": candidate,
            }
        )
        features.append(
            {
                "match_id": match_id,
                "date": match_date,
                "actual": actual,
                "features": {"incumbent_probability": incumbent},
            }
        )
    return challenger, features


def make_registry(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    registry = artifacts / "registry"
    for version, payload in (
        (LIVE_VERSION, b"validated-live-model"),
        (CHALLENGER_VERSION, b"improved-challenger-model"),
    ):
        version_dir = registry / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "model.pkl").write_bytes(payload)
    live = reference_for_version(registry, LIVE_VERSION)
    write_model_pointer(registry, "latest", live)
    production = artifacts / "model.pkl"
    production.write_bytes(live.artifact.read_bytes())
    return registry, production


def improved_decision() -> PromotionDecision:
    challenger, features = cohort(
        [0.7, 0.3, 0.7, 0.3],
        [0.8, 0.2, 0.8, 0.2],
        [1, 0, 1, 0],
    )
    return compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=4)


def rejected_decision() -> PromotionDecision:
    challenger, features = cohort(
        [0.8, 0.2, 0.8, 0.2],
        [0.6, 0.4, 0.6, 0.4],
        [1, 0, 1, 0],
    )
    return compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=4)


def snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def test_rejected_challenger_leaves_production_and_pointers_byte_identical(
    tmp_path: Path,
) -> None:
    registry, production = make_registry(tmp_path)
    incumbent = read_model_pointer(registry, "latest")
    candidate = reference_for_version(registry, CHALLENGER_VERSION)
    before = snapshot_files(tmp_path)

    result = promote_candidate(
        registry,
        CHALLENGER_VERSION,
        rejected_decision(),
        production_path=production,
        expected_incumbent=incumbent,
        expected_candidate_sha256=candidate.sha256,
    )

    assert result.action == "rejected"
    assert not result.changed
    assert result.latest.version == LIVE_VERSION
    assert snapshot_files(tmp_path) == before
    assert sha256_file(production) == reference_for_version(registry, LIVE_VERSION).sha256


def test_log_loss_improvement_promotes_and_preserves_outgoing_as_last_good(
    tmp_path: Path,
) -> None:
    registry, production = make_registry(tmp_path)
    incumbent = read_model_pointer(registry, "latest")
    candidate = reference_for_version(registry, CHALLENGER_VERSION)
    decision = improved_decision()
    assert decision.promote
    assert decision.reason == "log_loss_improved"
    assert "n=4" in decision.message
    assert "log_loss incumbent=" in decision.message
    assert "brier incumbent=" in decision.message

    result = promote_candidate(
        registry,
        CHALLENGER_VERSION,
        decision,
        production_path=production,
        expected_incumbent=incumbent,
        expected_candidate_sha256=candidate.sha256,
    )

    assert result.action == "promoted"
    assert result.changed
    assert read_model_pointer(registry, "latest").version == CHALLENGER_VERSION
    assert read_model_pointer(registry, "last_good").version == LIVE_VERSION
    assert sha256_file(production) == reference_for_version(registry, CHALLENGER_VERSION).sha256


def test_bootstrap_initializes_runtime_and_both_validated_pointers(tmp_path: Path) -> None:
    registry = tmp_path / "artifacts" / "registry"
    version_dir = registry / LIVE_VERSION
    version_dir.mkdir(parents=True)
    (version_dir / "model.pkl").write_bytes(b"first-validated-model")
    production = registry.parent / "model.pkl"

    candidate = reference_for_version(registry, LIVE_VERSION)
    result = bootstrap_candidate(
        registry,
        LIVE_VERSION,
        production_path=production,
        expected_candidate_sha256=candidate.sha256,
    )

    assert result.action == "bootstrapped"
    assert result.changed
    assert read_model_pointer(registry, "latest").version == LIVE_VERSION
    assert read_model_pointer(registry, "last_good").version == LIVE_VERSION
    assert sha256_file(production) == reference_for_version(registry, LIVE_VERSION).sha256

    with pytest.raises(PromotionValidationError, match="requires absent"):
        bootstrap_candidate(
            registry,
            LIVE_VERSION,
            production_path=production,
            expected_candidate_sha256=candidate.sha256,
        )


def test_bootstrap_cas_rejects_candidate_changed_after_validation(tmp_path: Path) -> None:
    registry = tmp_path / "artifacts" / "registry"
    version_dir = registry / LIVE_VERSION
    version_dir.mkdir(parents=True)
    candidate = version_dir / "model.pkl"
    candidate.write_bytes(b"first-validated-model")
    expected_candidate = reference_for_version(registry, LIVE_VERSION)
    candidate.write_bytes(b"tampered-after-validation")
    production = registry.parent / "model.pkl"
    before = snapshot_files(tmp_path)

    with pytest.raises(PromotionValidationError, match="candidate changed"):
        bootstrap_candidate(
            registry,
            LIVE_VERSION,
            production_path=production,
            expected_candidate_sha256=expected_candidate.sha256,
        )

    assert snapshot_files(tmp_path) == before
    assert not production.exists()


def test_bootstrap_fails_closed_when_deployment_lock_exists(tmp_path: Path) -> None:
    registry = tmp_path / "artifacts" / "registry"
    version_dir = registry / LIVE_VERSION
    version_dir.mkdir(parents=True)
    (version_dir / "model.pkl").write_bytes(b"first-validated-model")
    lock = registry / ".deployment.lock"
    lock.write_text("pid=someone-else\n", encoding="utf-8")
    production = registry.parent / "model.pkl"
    before = snapshot_files(tmp_path)

    with pytest.raises(PromotionValidationError, match="another model deployment"):
        bootstrap_candidate(
            registry,
            LIVE_VERSION,
            production_path=production,
            expected_candidate_sha256=reference_for_version(registry, LIVE_VERSION).sha256,
        )

    assert snapshot_files(tmp_path) == before
    assert lock.read_text(encoding="utf-8") == "pid=someone-else\n"
    assert not production.exists()


def test_log_loss_tie_uses_minimum_brier_improvement() -> None:
    # Correct-class probabilities have the same product (0.9*0.55 == 0.75*0.66),
    # hence identical log loss, while the challenger has much lower squared error.
    labels = [1, 1, 0, 0] * 2
    incumbent = [0.9, 0.55, 0.1, 0.45] * 2
    challenger_probs = [0.75, 0.66, 0.25, 0.34] * 2
    challenger, features = cohort(incumbent, challenger_probs, labels)

    decision = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=8)

    assert abs(decision.delta_log_loss or 0.0) < 0.001
    assert (decision.delta_brier or 0.0) <= -0.0005
    assert decision.promote
    assert decision.reason == "brier_tiebreak_improved"


def test_insufficient_strictly_post_cutoff_holdout_is_rejected() -> None:
    challenger, features = cohort([0.7, 0.3], [0.8, 0.2], [1, 0])
    challenger.insert(
        0,
        {
            "match_id": "same-day",
            "date": "2026-01-01",
            "actual": 1,
            "prob_team1": 0.99,
        },
    )
    features.insert(
        0,
        {
            "match_id": "same-day",
            "date": "2026-01-01",
            "actual": 1,
            "features": {"incumbent_probability": 0.01},
        },
    )

    decision = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=3)

    assert decision.status == "insufficient"
    assert not decision.promote
    assert decision.n == 2
    assert decision.cutoff == "2026-01-01"
    assert decision.incumbent is None
    assert decision.challenger is None
    assert "n=2" in decision.message


def test_rollback_restores_last_good_and_rotates_outgoing_latest(
    tmp_path: Path,
) -> None:
    registry, production = make_registry(tmp_path)
    incumbent = read_model_pointer(registry, "latest")
    candidate = reference_for_version(registry, CHALLENGER_VERSION)
    promote_candidate(
        registry,
        CHALLENGER_VERSION,
        improved_decision(),
        production_path=production,
        expected_incumbent=incumbent,
        expected_candidate_sha256=candidate.sha256,
    )

    result = rollback_last_good(registry, production_path=production)

    assert result.action == "rolled_back"
    assert result.latest.version == LIVE_VERSION
    assert result.last_good is not None
    assert result.last_good.version == CHALLENGER_VERSION
    assert read_model_pointer(registry, "latest").version == LIVE_VERSION
    assert read_model_pointer(registry, "last_good").version == CHALLENGER_VERSION
    assert sha256_file(production) == reference_for_version(registry, LIVE_VERSION).sha256


def test_appending_unpaired_future_features_cannot_change_frozen_holdout() -> None:
    challenger, features = cohort(
        [0.7, 0.3, 0.7, 0.3],
        [0.8, 0.2, 0.8, 0.2],
        [1, 0, 1, 0],
    )
    baseline = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=4)
    appended = [
        *features,
        {
            "match_id": "future-match",
            "date": "2030-01-01",
            "actual": 0,
            "features": {"incumbent_probability": 0.999999},
        },
    ]

    after_append = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, appended, min_samples=4)

    assert after_append.to_dict() == baseline.to_dict()


def test_prediction_digest_binds_both_models_probabilities() -> None:
    challenger, features = cohort(
        [0.7, 0.3, 0.7, 0.3],
        [0.8, 0.2, 0.8, 0.2],
        [1, 0, 1, 0],
    )
    baseline = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=4)
    changed = [dict(row) for row in challenger]
    changed[0]["prob_team1"] = 0.81

    after_probability_change = compare_candidate_to_incumbent(
        FeatureProbabilityModel(), changed, features, min_samples=4
    )

    assert after_probability_change.holdout_sha256 == baseline.holdout_sha256
    assert after_probability_change.prediction_sha256 != baseline.prediction_sha256


def test_same_artifact_same_cohort_has_identical_metrics_within_documented_tolerance() -> None:
    model = FeatureProbabilityModel()
    features = [
        {"incumbent_probability": 0.8},
        {"incumbent_probability": 0.3},
        {"incumbent_probability": 0.6},
        {"incumbent_probability": 0.1},
    ]
    labels = [1, 0, 1, 0]
    first_pass = model.predict_proba_team1(features)

    audit = assert_same_artifact_same_cohort_metrics(
        model,
        features,
        labels,
        first_pass,
    )

    assert audit == {
        "status": "pass",
        "n": 4,
        "atol": 1e-12,
        "max_abs_probability_delta": 0.0,
        "log_loss_delta": 0.0,
        "brier_delta": 0.0,
    }

    changed = first_pass.copy()
    changed[0] += 1e-6
    with pytest.raises(PromotionValidationError, match="same artifact/same cohort"):
        assert_same_artifact_same_cohort_metrics(model, features, labels, changed)


def test_promotion_report_exposes_same_cohort_metrics_by_router_regime() -> None:
    challenger, features = cohort(
        [0.7, 0.3, 0.7, 0.3],
        [0.8, 0.2, 0.6, 0.4],
        [1, 0, 1, 0],
    )
    challenger[0]["prediction_regime"] = "odds"
    challenger[1]["prediction_regime"] = "odds"
    challenger[2]["prediction_regime"] = "no_odds"
    challenger[3]["prediction_regime"] = "no_odds"

    decision = compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=4)

    assert decision.cohort["date_min"] == "2026-01-02"
    assert decision.cohort["date_max"] == "2026-01-05"
    assert decision.cohort["regime_counts"] == {"no_odds": 2, "odds": 2}
    assert decision.cohort["no_odds_fraction"] == 0.5
    assert decision.by_regime["odds"]["n"] == 2
    assert decision.by_regime["no_odds"]["n"] == 2
    assert "ece_10" in decision.by_regime["odds"]["challenger"]


def test_pairing_rejects_label_mismatch() -> None:
    challenger, features = cohort([0.7], [0.8], [1])
    features[0]["actual"] = 0

    with pytest.raises(PromotionValidationError, match="label mismatch"):
        compare_candidate_to_incumbent(FeatureProbabilityModel(), challenger, features, min_samples=1)


def test_pointer_rejects_tampered_hash_and_unsafe_version(tmp_path: Path) -> None:
    registry, _production = make_registry(tmp_path)
    live_artifact = registry / LIVE_VERSION / "model.pkl"
    live_artifact.write_bytes(b"tampered-after-pointer")

    with pytest.raises(PromotionValidationError, match="sha256"):
        read_model_pointer(registry, "latest")

    outside = tmp_path / "outside" / "model.pkl"
    outside.parent.mkdir()
    outside.write_bytes(b"outside")
    (registry / "latest.json").write_text(
        json.dumps(
            {
                "version": "../outside",
                "artifact": str(outside),
                "sha256": sha256_file(outside),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PromotionValidationError, match="YYYYMMDD_HHMMSSZ"):
        read_model_pointer(registry, "latest")


def test_promotion_cas_rejects_changed_incumbent_without_mutation(tmp_path: Path) -> None:
    registry, production = make_registry(tmp_path)
    expected_incumbent = read_model_pointer(registry, "latest")
    concurrent = reference_for_version(registry, CHALLENGER_VERSION)
    production.write_bytes(concurrent.artifact.read_bytes())
    write_model_pointer(registry, "latest", concurrent)
    before = snapshot_files(tmp_path)

    with pytest.raises(PromotionValidationError, match="latest changed"):
        promote_candidate(
            registry,
            CHALLENGER_VERSION,
            improved_decision(),
            production_path=production,
            expected_incumbent=expected_incumbent,
            expected_candidate_sha256=reference_for_version(registry, CHALLENGER_VERSION).sha256,
        )

    assert snapshot_files(tmp_path) == before


def test_promotion_cas_rejects_candidate_changed_after_evaluation(tmp_path: Path) -> None:
    registry, production = make_registry(tmp_path)
    expected_incumbent = read_model_pointer(registry, "latest")
    expected_candidate = reference_for_version(registry, CHALLENGER_VERSION)
    expected_candidate.artifact.write_bytes(b"tampered-after-evaluation")
    before = snapshot_files(tmp_path)

    with pytest.raises(PromotionValidationError, match="candidate changed"):
        promote_candidate(
            registry,
            CHALLENGER_VERSION,
            improved_decision(),
            production_path=production,
            expected_incumbent=expected_incumbent,
            expected_candidate_sha256=expected_candidate.sha256,
        )

    assert snapshot_files(tmp_path) == before
    assert sha256_file(production) == expected_incumbent.sha256


@pytest.mark.parametrize("failure_stage", ["runtime_copy", "latest_pointer", "post_commit_validation"])
def test_promotion_failure_restores_every_snapshot_and_releases_lock(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, production = make_registry(tmp_path)
    expected_incumbent = read_model_pointer(registry, "latest")
    expected_candidate = reference_for_version(registry, CHALLENGER_VERSION)
    before = snapshot_files(tmp_path)

    if failure_stage == "runtime_copy":
        original_copy = promotion_module._atomic_copy_artifact

        def fail_candidate_copy(source, destination):
            if source.version == CHALLENGER_VERSION:
                raise OSError("injected candidate copy failure")
            return original_copy(source, destination)

        monkeypatch.setattr(promotion_module, "_atomic_copy_artifact", fail_candidate_copy)
    elif failure_stage == "latest_pointer":
        original_write = promotion_module.write_model_pointer

        def fail_latest(registry_dir, pointer_name, reference):
            if pointer_name == "latest" and reference.version == CHALLENGER_VERSION:
                raise OSError("injected latest pointer failure")
            return original_write(registry_dir, pointer_name, reference)

        monkeypatch.setattr(promotion_module, "write_model_pointer", fail_latest)
    else:
        original_verify = promotion_module._verified_production_hash
        calls = 0

        def fail_second_validation(path, expected):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected post-commit validation failure")
            return original_verify(path, expected)

        monkeypatch.setattr(promotion_module, "_verified_production_hash", fail_second_validation)

    with pytest.raises(OSError, match="injected"):
        promote_candidate(
            registry,
            CHALLENGER_VERSION,
            improved_decision(),
            production_path=production,
            expected_incumbent=expected_incumbent,
            expected_candidate_sha256=expected_candidate.sha256,
        )

    assert snapshot_files(tmp_path) == before
    assert not (registry / ".deployment.lock").exists()


def test_failed_recovery_restores_independent_pointers_and_preserves_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, production = make_registry(tmp_path)
    expected_incumbent = read_model_pointer(registry, "latest")
    expected_candidate = reference_for_version(registry, CHALLENGER_VERSION)
    latest_before = (registry / "latest.json").read_bytes()
    last_good_before = (registry / "last_good.json").read_bytes() if (registry / "last_good.json").exists() else None
    original_copy = promotion_module._atomic_copy_artifact
    original_write = promotion_module.write_model_pointer

    def fail_runtime_recovery(source, destination):
        if source.version == LIVE_VERSION:
            raise OSError("injected runtime recovery failure")
        return original_copy(source, destination)

    def fail_latest_commit(registry_dir, pointer_name, reference):
        if pointer_name == "latest" and reference.version == CHALLENGER_VERSION:
            raise OSError("injected latest commit failure")
        return original_write(registry_dir, pointer_name, reference)

    monkeypatch.setattr(promotion_module, "_atomic_copy_artifact", fail_runtime_recovery)
    monkeypatch.setattr(promotion_module, "write_model_pointer", fail_latest_commit)

    with pytest.raises(DeploymentRecoveryError, match="lock preserved"):
        promote_candidate(
            registry,
            CHALLENGER_VERSION,
            improved_decision(),
            production_path=production,
            expected_incumbent=expected_incumbent,
            expected_candidate_sha256=expected_candidate.sha256,
        )

    assert (registry / "latest.json").read_bytes() == latest_before
    assert (
        (registry / "last_good.json").read_bytes() if (registry / "last_good.json").exists() else None
    ) == last_good_before
    assert sha256_file(production) == expected_candidate.sha256
    assert (registry / ".deployment.lock").is_file()

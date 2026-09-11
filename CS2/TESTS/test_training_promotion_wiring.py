from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MODEL"))

import train  # noqa: E402
from cs2model.promotion import (  # noqa: E402
    PromotionValidationError,
    reference_for_version,
    write_model_pointer,
)


class FakeArtifact:
    def __init__(self) -> None:
        self.feature_columns = ["rating_diff"]
        self.metadata = {
            "trained_at": "2026-08-14T12:34:56Z",
            "reproducibility": {"random_seed": 42},
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.metadata, sort_keys=True), encoding="utf-8")
        return destination


class FailingArtifact(FakeArtifact):
    def save(self, path: str | Path) -> Path:
        destination = super().save(path)
        raise OSError(f"injected registry staging failure: {destination.name}")


def test_registering_challenger_never_moves_production_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    monkeypatch.setattr(train, "MODEL_REGISTRY_DIR", registry_root)
    artifact = FakeArtifact()

    result = train.save_model_registry(artifact, {}, [], [], {}, {}, {})

    assert result["version"] == "20260814_123456Z"
    assert Path(result["registered_model"]).is_file()
    assert not (registry_root / "latest.json").exists()
    assert not (registry_root / "last_good.json").exists()

    with pytest.raises(FileExistsError, match="inmutable"):
        train.save_model_registry(artifact, {}, [], [], {}, {}, {})

    if "update_existing" in inspect.signature(train.save_model_registry).parameters:
        with pytest.raises((FileExistsError, ValueError), match="inmutable|mutar"):
            train.save_model_registry(artifact, {}, [], [], {}, {}, {}, update_existing=True)
    assert not (registry_root / "latest.json").exists()


def test_registry_staging_failure_leaves_no_version_or_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    monkeypatch.setattr(train, "MODEL_REGISTRY_DIR", registry_root)

    with pytest.raises(OSError, match="injected registry staging failure"):
        train.save_model_registry(FailingArtifact(), {}, [], [], {}, {}, {})

    assert registry_root.is_dir()
    assert list(registry_root.iterdir()) == []


@pytest.mark.parametrize("pointer_name", ["latest", "last_good"])
def test_registered_live_or_last_good_version_is_immutable(
    pointer_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    monkeypatch.setattr(train, "MODEL_REGISTRY_DIR", registry_root)
    artifact = FakeArtifact()
    result = train.save_model_registry(artifact, {}, [], [], {}, {}, {})
    reference = reference_for_version(registry_root, result["version"])
    write_model_pointer(registry_root, pointer_name, reference)
    before = {path.relative_to(registry_root): path.read_bytes() for path in registry_root.rglob("*") if path.is_file()}

    with pytest.raises(FileExistsError, match="inmutable"):
        train.save_model_registry(artifact, {}, [], [], {}, {}, {})
    if "update_existing" in inspect.signature(train.save_model_registry).parameters:
        with pytest.raises((FileExistsError, ValueError), match="inmutable|mutar"):
            train.save_model_registry(artifact, {}, [], [], {}, {}, {}, update_existing=True)

    after = {path.relative_to(registry_root): path.read_bytes() for path in registry_root.rglob("*") if path.is_file()}
    assert after == before


def test_load_validated_incumbent_rejects_runtime_hash_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    version_dir = registry_root / "20260814_123456Z"
    version_dir.mkdir(parents=True)
    (version_dir / "model.pkl").write_bytes(b"canonical-registry-model")
    reference = reference_for_version(registry_root, version_dir.name)
    write_model_pointer(registry_root, "latest", reference)
    runtime_model = registry_root.parent / "model.pkl"
    runtime_model.write_bytes(b"different-runtime-model")
    monkeypatch.setattr(train, "MODEL_REGISTRY_DIR", registry_root)
    monkeypatch.setattr(train, "ARTIFACT_PATH", runtime_model)

    with pytest.raises(SystemExit, match="SHA-256"):
        train._load_validated_incumbent()


def test_load_validated_incumbent_rejects_pointer_hash_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    version_dir = registry_root / "20260814_123456Z"
    version_dir.mkdir(parents=True)
    registry_model = version_dir / "model.pkl"
    registry_model.write_bytes(b"model-before-pointer")
    reference = reference_for_version(registry_root, version_dir.name)
    write_model_pointer(registry_root, "latest", reference)
    registry_model.write_bytes(b"model-after-pointer-tamper")
    runtime_model = registry_root.parent / "model.pkl"
    runtime_model.write_bytes(registry_model.read_bytes())
    monkeypatch.setattr(train, "MODEL_REGISTRY_DIR", registry_root)
    monkeypatch.setattr(train, "ARTIFACT_PATH", runtime_model)

    with pytest.raises(PromotionValidationError, match="sha256"):
        train._load_validated_incumbent()


def _training_main_ast() -> ast.FunctionDef:
    module = ast.parse(textwrap.dedent(inspect.getsource(train.main)))
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _named_calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_training_promotion_wiring_uses_frozen_candidate_and_cas_inputs() -> None:
    function = _training_main_ast()
    challenger_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "challenger_oos" for target in node.targets)
    ]
    assert len(challenger_assignments) == 1
    challenger_value = challenger_assignments[0].value
    assert isinstance(challenger_value, ast.Name)
    assert challenger_value.id == "promotion_challenger_oos"

    source = ast.unparse(function)
    assert "shadow = ModelArtifact" in source
    assert "select_fold_local_features(X_dicts, y_all, periods, recipe_mask" in source
    assert "prune_exact_redundancies(final_matrix[recipe_mask]" in source
    assert "X_all[recipe_mask]" in source
    assert "y_all[recipe_mask]" in source
    assert "periods[recipe_mask]" in source
    assert "if not selected" in source
    assert "shadow.predict_proba_team1(holdout_features)" in source
    assert "assert_same_artifact_same_cohort_metrics" in source
    assert "'prediction_regime': shadow.prediction_regime" in source
    assert "validate_candidate(Path(args.db), registered_model, out)" in source

    architecture_calls = _named_calls(function, "evaluate_odds_architectures")
    assert len(architecture_calls) == 1
    architecture_call = architecture_calls[0]
    assert ast.unparse(architecture_call.args[4]) == "preds.get(best_name, [])"
    architecture_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in architecture_call.keywords}
    assert architecture_keywords["calibration_fraction"] == ("training_defaults.final_calibration_fraction")
    assert architecture_keywords["recency_half_life"] == "args.recency_half_life"
    assert architecture_keywords["reserve_model"] == "best_name"

    bootstrap_calls = _named_calls(function, "bootstrap_candidate")
    promotion_calls = _named_calls(function, "promote_candidate")
    assert len(bootstrap_calls) == 1
    assert len(promotion_calls) == 1
    bootstrap_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in bootstrap_calls[0].keywords}
    promotion_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in promotion_calls[0].keywords}
    assert bootstrap_keywords["expected_candidate_sha256"] == "candidate_reference.sha256"
    assert promotion_keywords["expected_incumbent"] == "incumbent_reference"
    assert promotion_keywords["expected_candidate_sha256"] == "candidate_reference.sha256"


def test_post_publish_exception_handlers_cannot_rewrite_registry() -> None:
    function = _training_main_ast()
    publish_calls = [
        *_named_calls(function, "bootstrap_candidate"),
        *_named_calls(function, "promote_candidate"),
    ]
    assert len(publish_calls) == 2
    last_publish_line = max(call.lineno for call in publish_calls)
    post_publish_handlers = [
        node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler) and node.lineno > last_publish_line
    ]
    for handler in post_publish_handlers:
        handler_source = ast.unparse(handler)
        assert "save_model_registry" not in handler_source
        assert "artifact.metadata" not in handler_source
        assert "artifact.save" not in handler_source
        assert "write_model_pointer" not in handler_source

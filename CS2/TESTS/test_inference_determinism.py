from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from MODEL.cs2model.artifacts import ColumnSubsetEstimator, Component, ModelArtifact, load_artifact
from MODEL.cs2model.reproducibility import INFERENCE_DETERMINISM_ATOL, INFERENCE_THREAD_ENVIRONMENT
from MODEL.cs2model.reproducibility import force_single_threaded_inference


ROOT = Path(__file__).resolve().parents[1]


def _threaded_artifact(path: Path) -> Path:
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [-0.5, 0.5],
            [0.5, -0.5],
            [1.0, -1.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    forest = RandomForestClassifier(n_estimators=48, n_jobs=-1, random_state=42).fit(X, y)
    nested = ColumnSubsetEstimator(
        estimator=Pipeline([("model", forest)]),
        column_indices=[0, 1],
    )
    return ModelArtifact(
        feature_columns=["strength", "context"],
        components=[Component("forest", nested, None)],
    ).save(path)


def test_load_forces_single_threaded_prediction_and_repeats_within_tolerance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for variable in INFERENCE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(variable, "8")
    path = _threaded_artifact(tmp_path / "model.pkl")

    artifact = load_artifact(path)

    assert artifact is not None
    forest = artifact.components[0].estimator.estimator.named_steps["model"]
    assert forest.n_jobs == 1
    assert all(os.environ[variable] == "1" for variable in INFERENCE_THREAD_ENVIRONMENT)
    row = {"strength": 0.125, "context": -0.25}
    first = artifact.predict_proba_team1([row])
    second = artifact.predict_proba_team1([row])
    np.testing.assert_allclose(first, second, rtol=0.0, atol=INFERENCE_DETERMINISM_ATOL)


def test_inference_thread_traversal_handles_cyclic_containers() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    assert force_single_threaded_inference(cyclic) == 0


def test_loaded_prediction_is_stable_across_fresh_processes(tmp_path: Path) -> None:
    path = _threaded_artifact(tmp_path / "model.pkl")
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from MODEL.cs2model.artifacts import load_artifact

artifact = load_artifact(sys.argv[1])
row = {"strength": 0.125, "context": -0.25}
point = artifact.predict_proba_team1([row])
mean, spread = artifact.predict_proba_team1_with_uncertainty([row])
forest = artifact.components[0].estimator.estimator.named_steps["model"]
print(json.dumps({"point": float(point[0]), "mean": float(mean[0]), "spread": float(spread[0]), "n_jobs": forest.n_jobs}, sort_keys=True))
"""

    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script, str(path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0]["n_jobs"] == outputs[1]["n_jobs"] == 1
    for field in ("point", "mean", "spread"):
        assert abs(outputs[0][field] - outputs[1][field]) <= INFERENCE_DETERMINISM_ATOL

"""Guardado/carga del artefacto de modelo entrenado.

El artefacto contiene TODO lo necesario para puntuar en producción de forma
idéntica al entrenamiento: uno o varios estimadores con su calibrador
isotónico (ensemble ponderado), la lista de features y metadatos. La
reconstrucción del estado cronológico (Glicko/Elo/forma) se hace aparte desde
el histórico, así que NO se serializa aquí.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.pkl"


@dataclass
class Component:
    """Un estimador + su calibrador isotónico + su peso en el ensemble."""

    name: str
    estimator: Any                   # objeto con predict_proba
    calibrator: Any | None           # IsotonicRegression sobre la prob cruda, o None
    weight: float = 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            raw = self.estimator.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return np.clip(raw, 1e-4, 1 - 1e-4)


@dataclass
class ProbabilityColumnEstimator:
    """Platt scaling for an already probabilistic point-in-time rating column."""

    column_index: int
    slope: float = 1.0
    intercept: float = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        centered = np.asarray(X, dtype=float)[:, self.column_index]
        raw = np.clip(np.nan_to_num(centered, nan=0.0) + 0.5, 1e-6, 1.0 - 1e-6)
        logits = np.log(raw / (1.0 - raw))
        values = self.intercept + self.slope * logits
        positive = np.empty_like(values)
        mask = values >= 0.0
        positive[mask] = 1.0 / (1.0 + np.exp(-values[mask]))
        exp_values = np.exp(values[~mask])
        positive[~mask] = exp_values / (1.0 + exp_values)
        positive = np.clip(positive, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class ColumnSubsetEstimator:
    """Adapter allowing one artifact to host components with different inputs."""

    estimator: Any
    column_indices: list[int]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(X)[:, self.column_indices])


@dataclass
class ModelArtifact:
    feature_columns: list[str]
    components: list[Component]
    metadata: dict[str, Any] = field(default_factory=dict)
    rich_target_estimator: Any | None = None
    rich_target_classes: list[str] = field(default_factory=list)

    def _matrix(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        return np.array(
            [[row.get(col, np.nan) for col in self.feature_columns] for row in feature_rows],
            dtype=float,
        )

    def predict_proba_team1(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        """Probabilidad calibrada (ensemble ponderado) de que gane team1."""
        X = self._matrix(feature_rows)
        if len(X) == 0:
            return np.array([])
        total_w = sum(c.weight for c in self.components) or 1.0
        acc = np.zeros(len(X))
        for c in self.components:
            acc += c.weight * c.predict(X)
        return np.clip(acc / total_w, 1e-4, 1 - 1e-4)

    def predict_proba_team1_with_uncertainty(
        self, feature_rows: list[dict[str, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """A2: devuelve (prob_media, std_epistemica).

        La std es la dispersion PONDERADA entre los miembros del ensemble, i.e.
        cuanto discrepan entre si (incertidumbre epistemica). Sirve para apostar
        menos cuando el modelo "no lo tiene claro". Con un solo componente, std=0.
        """
        X = self._matrix(feature_rows)
        if len(X) == 0:
            return np.array([]), np.array([])
        weights = np.array([c.weight for c in self.components], dtype=float)
        total = weights.sum() or 1.0
        weights = weights / total
        preds = np.column_stack([c.predict(X) for c in self.components])  # (n, k)
        mean = np.clip(preds @ weights, 1e-4, 1 - 1e-4)
        if preds.shape[1] < 2:
            return mean, np.zeros(len(X))
        var = ((preds - mean[:, None]) ** 2) @ weights
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict_series_score_distribution(
        self, feature_rows: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """A6: symmetric BO3 scoreline probabilities when the sidecar is active."""
        estimator = getattr(self, "rich_target_estimator", None)
        classes = list(getattr(self, "rich_target_classes", None) or [])
        if estimator is None or not classes:
            return []
        X = self._matrix(feature_rows)
        if len(X) == 0:
            return []

        def aligned(matrix: np.ndarray) -> np.ndarray:
            raw = np.asarray(estimator.predict_proba(matrix), dtype=float)
            estimator_classes = [int(value) for value in estimator.classes_]
            result = np.zeros((len(matrix), len(classes)), dtype=float)
            for source_index, class_index in enumerate(estimator_classes):
                if 0 <= class_index < len(classes):
                    result[:, class_index] = raw[:, source_index]
            totals = result.sum(axis=1, keepdims=True)
            return np.divide(result, totals, out=np.full_like(result, 1.0 / len(classes)), where=totals > 0)

        direct = aligned(X)
        reverse = X.copy()
        directional = set(self.metadata.get("directional_feature_columns") or [])
        for index, column in enumerate(self.feature_columns):
            if column in directional:
                reverse[:, index] *= -1.0
        # Swapping teams reverses 0-2,1-2,2-1,2-0.
        mirrored = aligned(reverse)[:, ::-1]
        probabilities = 0.5 * (direct + mirrored)
        return [
            {name: float(row[index]) for index, name in enumerate(classes)}
            for row in probabilities
        ]

    def save(self, path: str | Path = ARTIFACT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path


def load_artifact(path: str | Path = ARTIFACT_PATH) -> ModelArtifact | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)

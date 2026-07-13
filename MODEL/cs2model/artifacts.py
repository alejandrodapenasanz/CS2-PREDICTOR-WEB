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
class ModelArtifact:
    feature_columns: list[str]
    components: list[Component]
    metadata: dict[str, Any] = field(default_factory=dict)

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

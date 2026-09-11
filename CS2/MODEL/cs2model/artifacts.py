"""Guardado/carga del artefacto de modelo entrenado.

El artefacto contiene TODO lo necesario para puntuar en producción de forma
idéntica al entrenamiento: uno o varios estimadores con su calibrador
isotónico (ensemble ponderado), la lista de features y metadatos. La
reconstrucción del estado cronológico (Glicko/Elo/forma) se hace aparte desde
el histórico, así que NO se serializa aquí.
"""

from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .reproducibility import force_single_threaded_inference

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.pkl"


@dataclass
class Component:
    """Un estimador + su calibrador isotónico + su peso en el ensemble."""

    name: str
    estimator: Any  # objeto con predict_proba
    calibrator: Any | None  # IsotonicRegression sobre la prob cruda, o None
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
    prediction_architecture: str = "model_a_no_odds"
    odds_feature_columns: list[str] = field(default_factory=list)
    odds_components: list[Component] = field(default_factory=list)
    mixed_feature_columns: list[str] = field(default_factory=list)
    mixed_components: list[Component] = field(default_factory=list)

    def _matrix(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        return self._matrix_for_columns(feature_rows, self.feature_columns)

    @staticmethod
    def _matrix_for_columns(feature_rows: list[dict[str, float]], columns: list[str]) -> np.ndarray:
        return np.array(
            [[row.get(col, np.nan) for col in columns] for row in feature_rows],
            dtype=float,
        )

    @staticmethod
    def _valid_odds_row(row: dict[str, float]) -> bool:
        try:
            available = float(row.get("odds_available", 0.0)) >= 0.5
            centered = float(row.get("opening_odds_prob_centered", np.nan))
        except (TypeError, ValueError):
            return False
        return available and np.isfinite(centered) and -0.5 < centered < 0.5

    def prediction_regime(self, row: dict[str, float]) -> str:
        """Return the actual market regime selected for one row."""

        return "odds" if self._valid_odds_row(row) else "no_odds"

    def component_weights_for_row(self, row: dict[str, float]) -> list[float]:
        """Expose normalized weights for uncertainty effective-size accounting."""

        _members, weights = self._routed_members(row)
        return [float(value) for value in weights]

    def _route(self, row: dict[str, float]) -> tuple[list[str], list[Component]]:
        architecture = str(getattr(self, "prediction_architecture", "model_a_no_odds") or "model_a_no_odds")
        if architecture == "router_two_models":
            odds_components = list(getattr(self, "odds_components", []) or [])
            odds_columns = list(getattr(self, "odds_feature_columns", []) or [])
            if self._valid_odds_row(row) and odds_components and odds_columns:
                return odds_columns, odds_components
            return self.feature_columns, self.components
        if architecture == "single_mixed_lgbm":
            mixed_components = list(getattr(self, "mixed_components", []) or [])
            mixed_columns = list(getattr(self, "mixed_feature_columns", []) or [])
            if mixed_components and mixed_columns:
                return mixed_columns, mixed_components
        return self.feature_columns, self.components

    def _routed_members(self, row: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        columns, components = self._route(row)
        if not components:
            raise ValueError("A model route must contain at least one component")
        weights = np.asarray([component.weight for component in components], dtype=float)
        total = float(weights.sum())
        if not np.isfinite(weights).all() or (weights < 0.0).any() or total <= 0.0:
            raise ValueError("Ensemble weights must be finite, non-negative and sum above zero")
        matrix = self._matrix_for_columns([row], columns)
        predictions = np.asarray(
            [float(component.predict(matrix)[0]) for component in components],
            dtype=float,
        )
        return predictions, weights / total

    def _routed_probability_and_uncertainty(
        self, feature_rows: list[dict[str, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        means: list[float] = []
        deviations: list[float] = []
        for row in feature_rows:
            members, weights = self._routed_members(row)
            mean = float(np.clip(members @ weights, 1e-4, 1.0 - 1e-4))
            variance = float(((members - mean) ** 2) @ weights)
            means.append(mean)
            deviations.append(math.sqrt(max(variance, 0.0)))
        return np.asarray(means, dtype=float), np.asarray(deviations, dtype=float)

    def _component_predictions(self, feature_rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated member probabilities and their normalized weights."""
        if not self.components:
            raise ValueError("A model artifact must contain at least one component")
        weights = np.asarray([component.weight for component in self.components], dtype=float)
        total = float(weights.sum())
        if not np.isfinite(weights).all() or (weights < 0.0).any() or total <= 0.0:
            raise ValueError("Ensemble weights must be finite, non-negative and sum above zero")
        X = self._matrix(feature_rows)
        if len(X) == 0:
            return np.empty((0, len(self.components))), weights / total
        predictions = np.column_stack([component.predict(X) for component in self.components])
        return predictions, weights / total

    def predict_proba_team1(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        """Probabilidad calibrada (ensemble ponderado) de que gane team1."""
        means, _deviations = self._routed_probability_and_uncertainty(feature_rows)
        return means

    def predict_proba_team1_with_uncertainty(
        self, feature_rows: list[dict[str, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """A2: devuelve (prob_media, dispersion ponderada del ensemble).

        La dispersion cuantifica cuanto discrepan los miembros calibrados sobre
        el propio estimado. No es un intervalo estadistico; con un solo
        componente vale cero y la capa de cobertura aplica una penalizacion
        separada para no confundirlo con certeza.
        """
        return self._routed_probability_and_uncertainty(feature_rows)

    def predict_symmetric_proba_team1_with_uncertainty(
        self,
        forward_rows: list[dict[str, float]],
        reverse_rows: list[dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order-invariant probability and member dispersion for A vs B.

        Each calibrated member is symmetrized first as
        ``0.5 * (p_j(A,B) + 1 - p_j(B,A))``.  The returned standard deviation
        is therefore the disagreement around the exact probability being
        published, rather than an average of two directional deviations.
        """
        if len(forward_rows) != len(reverse_rows):
            raise ValueError("forward_rows and reverse_rows must have the same length")
        means: list[float] = []
        deviations: list[float] = []
        for forward_row, reverse_row in zip(forward_rows, reverse_rows, strict=True):
            forward, weights = self._routed_members(forward_row)
            reverse, reverse_weights = self._routed_members(reverse_row)
            if len(forward) != len(reverse) or not np.array_equal(weights, reverse_weights):
                raise ValueError("Model route changed between symmetric predictions")
            member_probabilities = 0.5 * (forward + (1.0 - reverse))
            mean = float(np.clip(member_probabilities @ weights, 1e-4, 1.0 - 1e-4))
            variance = float(((member_probabilities - mean) ** 2) @ weights)
            means.append(mean)
            deviations.append(math.sqrt(max(variance, 0.0)))
        return np.asarray(means, dtype=float), np.asarray(deviations, dtype=float)

    def predict_series_score_distribution(self, feature_rows: list[dict[str, float]]) -> list[dict[str, float]]:
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
            result: np.ndarray = np.zeros((len(matrix), len(classes)), dtype=float)
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
        return [{name: float(row[index]) for index, name in enumerate(classes)} for row in probabilities]

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
        artifact = pickle.load(fh)
    force_single_threaded_inference(artifact)
    return artifact

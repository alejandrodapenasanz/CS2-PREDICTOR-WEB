"""Impone antisimetría A/B en evaluación e inferencia probabilística.

La orientación histórica está balanceada, pero un estimador no lineal no
garantiza por sí solo ``P(A,B) = 1 - P(B,A)``. Este módulo invierte todas las
features orientadas, predice ambos órdenes y promedia las probabilidades
complementarias. El mismo contrato se usa en folds y en producción.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol

import numpy as np
import pandas as pd

from .calibration import PlattCalibrator


_SIDE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("elo_cold_start_a", "elo_cold_start_b"),
    ("recent_n_matches_a", "recent_n_matches_b"),
    ("recent_months_matches_a", "recent_months_matches_b"),
    ("rest_days_a", "rest_days_b"),
    ("ranking_missing_a", "ranking_missing_b"),
    ("ranking_age_days_a", "ranking_age_days_b"),
    (
        "ranking_conflict_dates_skipped_a",
        "ranking_conflict_dates_skipped_b",
    ),
    ("age_missing_a", "age_missing_b"),
    ("odds_a", "odds_b"),
    ("market_probability_a", "market_probability_b"),
)
_SIGNED_DIFFERENCES: Final[frozenset[str]] = frozenset(
    {
        "elo_general_diff",
        "elo_surface_diff",
        "elo_surface_raw_diff",
        "elo_general_matches_diff",
        "elo_surface_matches_diff",
        "recent_n_win_rate_diff",
        "recent_months_win_rate_diff",
        "h2h_global_balance",
        "h2h_surface_balance",
        "rest_days_diff",
        "rank_diff",
        "rank_points_diff",
        "age_diff",
        "age_distance_30_diff",
    }
)
_UNCHANGED_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "surface",
        "tour_level",
        "tour_level_raw",
        "best_of",
        "round",
        "h2h_global_matches",
        "h2h_surface_matches",
        "market_overround",
        "market_margin",
    }
)


class OrientationSymmetryError(ValueError):
    """Indica que una matriz no puede invertirse de forma completa."""


class _ProbabilityEstimator(Protocol):
    """Contrato mínimo requerido por la inferencia simétrica."""

    feature_columns: tuple[str, ...]

    def predict_probability(self, frame: pd.DataFrame) -> pd.Series:
        """Devuelve P(A gana) para el orden recibido."""


def _pair_lookup() -> dict[str, str]:
    """Construye el mapping bidireccional de columnas por lado."""

    lookup: dict[str, str] = {}
    for left, right in _SIDE_PAIRS:
        lookup[left] = right
        lookup[right] = left
    return lookup


def reverse_model_orientation(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Invierte exactamente las features A/B activas sin mutar la entrada."""

    if not isinstance(frame, pd.DataFrame):
        raise OrientationSymmetryError("frame debe ser un DataFrame.")
    columns = tuple(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise OrientationSymmetryError(
            "feature_columns debe contener nombres únicos."
        )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise OrientationSymmetryError(
            f"Faltan features para invertir orientación: {missing}."
        )
    pairs = _pair_lookup()
    known = set(pairs) | set(_SIGNED_DIFFERENCES) | set(_UNCHANGED_FEATURES)
    unknown = sorted(set(columns).difference(known))
    if unknown:
        raise OrientationSymmetryError(
            f"No se declaró cómo invertir estas features: {unknown}."
        )

    reversed_frame = frame.copy()
    handled_pairs: set[tuple[str, str]] = set()
    for column in columns:
        partner = pairs.get(column)
        if partner is None:
            continue
        pair = tuple(sorted((column, partner)))
        if pair in handled_pairs:
            continue
        if partner not in columns:
            raise OrientationSymmetryError(
                f"La feature {column!r} requiere también {partner!r}."
            )
        left_values = frame[column].copy()
        right_values = frame[partner].copy()
        reversed_frame[column] = right_values
        reversed_frame[partner] = left_values
        handled_pairs.add(pair)

    for column in set(columns).intersection(_SIGNED_DIFFERENCES):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & numeric.isna()
        if invalid.any():
            raise OrientationSymmetryError(
                f"La diferencia {column!r} contiene valores no numéricos."
            )
        reversed_frame[column] = -numeric
    return reversed_frame


def predict_symmetric_raw(
    estimator: _ProbabilityEstimator,
    frame: pd.DataFrame,
) -> pd.Series:
    """Promedia P(A,B) con el complemento de P(B,A)."""

    forward = estimator.predict_probability(frame).to_numpy(dtype=float)
    reversed_frame = reverse_model_orientation(
        frame,
        estimator.feature_columns,
    )
    backward = estimator.predict_probability(
        reversed_frame
    ).to_numpy(dtype=float)
    symmetric = 0.5 * (forward + (1.0 - backward))
    if (
        symmetric.shape != (len(frame),)
        or not np.isfinite(symmetric).all()
        or np.any((symmetric < 0.0) | (symmetric > 1.0))
    ):
        raise OrientationSymmetryError(
            "La probabilidad simétrica resultó inválida."
        )
    return pd.Series(
        symmetric,
        index=frame.index,
        name="model_probability_a",
        dtype=float,
    )


def fit_symmetric_platt(
    probabilities: Sequence[float] | np.ndarray,
    target: Sequence[int] | np.ndarray,
) -> PlattCalibrator:
    """Ajusta Platt con cada observación y su orientación complementaria."""

    values = np.asarray(probabilities, dtype=float)
    labels = np.asarray(target, dtype=np.int8)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise OrientationSymmetryError(
            "probabilities y target deben ser vectores alineados."
        )
    augmented_probabilities = np.concatenate((values, 1.0 - values))
    augmented_target = np.concatenate((labels, 1 - labels))
    return PlattCalibrator().fit(
        augmented_probabilities,
        augmented_target,
    )


def predict_symmetric_calibrated(
    calibrator: PlattCalibrator,
    symmetric_raw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Calibra ambos órdenes y promedia para conservar antisimetría exacta."""

    values = np.asarray(symmetric_raw, dtype=float)
    if values.ndim != 1:
        raise OrientationSymmetryError(
            "symmetric_raw debe ser un vector unidimensional."
        )
    forward = calibrator.predict_proba(values)
    backward = calibrator.predict_proba(1.0 - values)
    calibrated = 0.5 * (forward + (1.0 - backward))
    if (
        not np.isfinite(calibrated).all()
        or np.any((calibrated < 0.0) | (calibrated > 1.0))
    ):
        raise OrientationSymmetryError(
            "La probabilidad calibrada simétrica resultó inválida."
        )
    return calibrated

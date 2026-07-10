"""Calibration helpers used by trained artifacts.

Classes in this module are intentionally importable from ``cs2model`` so pickle
artifacts can be loaded by production scripts after ``MODEL/train.py`` has run
as ``__main__``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BetaCalibratedClassifier:
    """Beta calibration (Kull & Flach 2017) over a base classifier.

    The calibrator fits a logistic regression on ``[ln(p), -ln(1-p)]`` and
    exposes ``predict_proba`` so it can be stored directly as an artifact
    component.
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self.lr = None

    @staticmethod
    def _feat(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p), -np.log(1.0 - p)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BetaCalibratedClassifier":
        from sklearn.linear_model import LogisticRegression

        p = self.base.predict_proba(X)[:, 1]
        self.lr = LogisticRegression(max_iter=1000).fit(self._feat(p), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base.predict_proba(X)[:, 1]
        p1 = self.lr.predict_proba(self._feat(p))[:, 1]
        return np.column_stack([1.0 - p1, p1])

"""Causal Gaussian state-space rating for team strength."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KalmanRating:
    mean: float = 0.0
    variance: float = 1.0
    games: int = 0


class KalmanTeamRating:
    """Pairwise Kalman filter with process drift and uncertain new teams."""

    def __init__(
        self,
        prior_variance: float = 1.0,
        observation_variance: float = 2.0,
        process_variance: float = 0.01,
    ) -> None:
        self.prior_variance = float(prior_variance)
        self.observation_variance = float(observation_variance)
        self.process_variance = float(process_variance)

    def default(self) -> KalmanRating:
        return KalmanRating(variance=self.prior_variance)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def win_probability(self, a: KalmanRating, b: KalmanRating) -> float:
        scale = math.sqrt(1.0 + a.variance + b.variance)
        return self._sigmoid((a.mean - b.mean) / scale)

    def evolve(self, rating: KalmanRating, elapsed_days: float) -> KalmanRating:
        """Kalman prediction step with variance growth during inactivity."""
        elapsed_weeks = max(0.0, float(elapsed_days)) / 7.0
        return KalmanRating(
            mean=rating.mean,
            variance=min(
                self.prior_variance * 4.0,
                rating.variance + self.process_variance * elapsed_weeks,
            ),
            games=rating.games,
        )

    def update(self, a: KalmanRating, b: KalmanRating, a_won: bool) -> tuple[KalmanRating, KalmanRating]:
        va = min(self.prior_variance * 4.0, a.variance + self.process_variance)
        vb = min(self.prior_variance * 4.0, b.variance + self.process_variance)
        innovation_variance = va + vb + self.observation_variance
        observed_margin = 1.0 if a_won else -1.0
        residual = observed_margin - (a.mean - b.mean)
        gain_a = va / innovation_variance
        gain_b = vb / innovation_variance
        return (
            KalmanRating(
                mean=a.mean + gain_a * residual,
                variance=max(1e-4, va * (1.0 - gain_a)),
                games=a.games + 1,
            ),
            KalmanRating(
                mean=b.mean - gain_b * residual,
                variance=max(1e-4, vb * (1.0 - gain_b)),
                games=b.games + 1,
            ),
        )

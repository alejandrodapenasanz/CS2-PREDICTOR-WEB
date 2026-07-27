"""Online empirical-Bayes Bradley-Terry rating with diagonal Laplace updates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BTRating:
    mean: float = 0.0
    variance: float = 1.0
    games: int = 0


class BayesianBradleyTerry:
    """Causal Bradley-Terry approximation with Gaussian partial pooling.

    Team skills have a shared N(0, prior_variance) prior. Each binary result is
    assimilated with a diagonal Laplace/ADF update. Teams with little evidence
    therefore remain close to the population mean and keep high uncertainty.
    """

    def __init__(
        self,
        prior_variance: float = 1.0,
        performance_variance: float = 1.0,
        process_variance: float = 0.0025,
    ) -> None:
        self.prior_variance = float(prior_variance)
        self.performance_variance = float(performance_variance)
        self.process_variance = float(process_variance)

    def default(self) -> BTRating:
        return BTRating(variance=self.prior_variance)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def win_probability(self, a: BTRating, b: BTRating) -> float:
        # Logistic-Gaussian moment approximation: uncertainty pulls toward 0.5.
        variance = a.variance + b.variance + self.performance_variance
        scale = math.sqrt(1.0 + math.pi * variance / 8.0)
        return self._sigmoid((a.mean - b.mean) / scale)

    def evolve(self, rating: BTRating, elapsed_days: float) -> BTRating:
        """Prediction step: stale evidence becomes more uncertain over time."""
        elapsed_weeks = max(0.0, float(elapsed_days)) / 7.0
        return BTRating(
            mean=rating.mean,
            variance=min(self.prior_variance, rating.variance + self.process_variance * elapsed_weeks),
            games=rating.games,
        )

    def update(self, a: BTRating, b: BTRating, a_won: bool) -> tuple[BTRating, BTRating]:
        va = min(self.prior_variance, a.variance + self.process_variance)
        vb = min(self.prior_variance, b.variance + self.process_variance)
        p = self._sigmoid(a.mean - b.mean)
        curvature = max(p * (1.0 - p), 1e-6)
        denominator = 1.0 + curvature * (va + vb + self.performance_variance)
        residual = (1.0 if a_won else 0.0) - p
        mean_a = a.mean + va * residual / denominator
        mean_b = b.mean - vb * residual / denominator
        var_a = max(1e-4, va * (1.0 - curvature * va / denominator))
        var_b = max(1e-4, vb * (1.0 - curvature * vb / denominator))
        return (
            BTRating(mean=mean_a, variance=var_a, games=a.games + 1),
            BTRating(mean=mean_b, variance=var_b, games=b.games + 1),
        )

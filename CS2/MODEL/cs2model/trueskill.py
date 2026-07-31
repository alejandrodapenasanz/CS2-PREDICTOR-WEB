"""TrueSkill (Herbrich, Minka & Graepel, 2006) a nivel EQUIPO, online/causal.

Rating alternativo a Glicko-2 para el A/B de accuracy. Es INCREMENTAL: cada
partido actualiza solo con su propio resultado y el estado previo, así que es
point-in-time por construcción (cero fuga), a diferencia de WHR, que suaviza el
pasado con partidos futuros (no causal → fugaría si se usa como feature de un
partido pasado).

Modelo 1-contra-1 (equipo vs equipo, sin empates: las series CS2 no empatan).
Referencia: https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/

Se integra como familia AUTO-GATED (`team_trueskill` en train.py): se calcula
siempre y entra al modelo cuando la muestra supera el umbral, sin flags. Sirve
para que el walk-forward mida si aporta sobre Glicko-2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MU0 = 25.0
SIGMA0 = 25.0 / 3.0
BETA = SIGMA0 / 2.0          # varianza del rendimiento por partido
TAU = SIGMA0 / 100.0        # dinámica: crecimiento de sigma entre partidos
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _v_win(t: float) -> float:
    denom = _cdf(t)
    if denom < 1e-12:
        return -t  # límite numérico cuando el ganador era muy improbable
    return _pdf(t) / denom


def _w_win(t: float) -> float:
    v = _v_win(t)
    return v * (v + t)


@dataclass
class TSRating:
    mu: float = MU0
    sigma: float = SIGMA0


class TeamTrueSkill:
    """Motor TrueSkill 1v1 con dinámica por partido."""

    def __init__(self, mu0: float = MU0, sigma0: float = SIGMA0,
                 beta: float = BETA, tau: float = TAU) -> None:
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.beta = beta
        self.tau = tau

    def default(self) -> TSRating:
        return TSRating(self.mu0, self.sigma0)

    def win_probability(self, a: TSRating, b: TSRating) -> float:
        """P(a gana a b) incorporando la incertidumbre de ambos."""
        denom = math.sqrt(2.0 * self.beta * self.beta + a.sigma * a.sigma + b.sigma * b.sigma)
        if denom <= 0.0:
            return 0.5
        return _cdf((a.mu - b.mu) / denom)

    def update(self, winner: TSRating, loser: TSRating) -> tuple[TSRating, TSRating]:
        """Devuelve (ganador, perdedor) actualizados tras un resultado sin empate."""
        sw2 = winner.sigma * winner.sigma + self.tau * self.tau
        sl2 = loser.sigma * loser.sigma + self.tau * self.tau
        c2 = 2.0 * self.beta * self.beta + sw2 + sl2
        c = math.sqrt(c2)
        t = (winner.mu - loser.mu) / c
        v = _v_win(t)
        w = _w_win(t)
        new_winner = TSRating(
            mu=winner.mu + (sw2 / c) * v,
            sigma=math.sqrt(max(1e-6, sw2 * (1.0 - (sw2 / c2) * w))),
        )
        new_loser = TSRating(
            mu=loser.mu - (sl2 / c) * v,
            sigma=math.sqrt(max(1e-6, sl2 * (1.0 - (sl2 / c2) * w))),
        )
        return new_winner, new_loser

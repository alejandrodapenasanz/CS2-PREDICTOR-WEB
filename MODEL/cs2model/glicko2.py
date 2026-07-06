"""Implementación de Glicko-2 (Glickman, 2013) para ratings de equipos de CS2.

Elegido sobre Elo plano porque modela la INCERTIDUMBRE del rating mediante la
rating deviation (RD) y la volatilidad (sigma):

  - Distingue "1500 con 200 partidos" de "1500 con 3 partidos".
  - La RD crece automáticamente con la inactividad: la versión principiada de
    una "red flag" por datos viejos.

Se usa por PERIODOS DE RATING (semanal por defecto, alineado con la
actualización del ranking de HLTV): dentro de un periodo todos los partidos de
un equipo se procesan a la vez sobre el estado de *inicio* de ese periodo, lo
que garantiza point-in-time (los partidos de la misma semana no se ven entre
sí). Entre periodos, los equipos que no juegan ven crecer su RD.

Escala interna de Glicko-2: mu = (r-1500)/173.7178, phi = rd/173.7178.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCALE = 173.7178          # factor de conversión Glicko <-> Glicko-2
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0        # máxima incertidumbre (equipo nuevo)
DEFAULT_SIGMA = 0.06      # volatilidad inicial
MAX_RD = 350.0


@dataclass
class Rating:
    """Estado de rating de una entidad (equipo o jugador)."""

    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    sigma: float = DEFAULT_SIGMA
    last_period: int | None = None      # índice del último periodo con partidos
    games: int = 0                      # nº de resultados procesados (madurez)

    @property
    def mu(self) -> float:
        return (self.rating - DEFAULT_RATING) / SCALE

    @property
    def phi(self) -> float:
        return self.rd / SCALE

    def copy(self) -> "Rating":
        return Rating(self.rating, self.rd, self.sigma, self.last_period, self.games)


@dataclass
class _Match:
    opponent: "Rating"
    score: float        # 1.0 victoria, 0.0 derrota (0.5 empate, no usado en CS2)


class Glicko2:
    """Motor de rating Glicko-2 con periodos de rating.

    Parameters
    ----------
    tau : float
        Constraint de la volatilidad del sistema. Valores típicos 0.3-1.2;
        más bajo = cambios de volatilidad más suaves. 0.5 es el recomendado.
    """

    def __init__(self, tau: float = 0.5, epsilon: float = 1e-6) -> None:
        self.tau = tau
        self.epsilon = epsilon

    # ---- utilidades de la escala Glicko-2 -------------------------------
    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    @staticmethod
    def _expected(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-Glicko2._g(phi_j) * (mu - mu_j)))

    def win_probability(self, a: Rating, b: Rating) -> float:
        """P(a gana a b) incorporando la RD de AMBOS (incertidumbre combinada).

        A mayor incertidumbre conjunta, la probabilidad se acerca a 0.5.
        """
        phi = math.sqrt(a.phi * a.phi + b.phi * b.phi)
        return 1.0 / (1.0 + math.exp(-self._g(phi) * (a.mu - b.mu)))

    # ---- decaimiento de RD por inactividad ------------------------------
    def _decay_rd(self, r: Rating, periods: int) -> None:
        """Aumenta la RD por 'periods' periodos sin jugar (phi* = sqrt(phi^2+periods*sigma^2))."""
        if periods <= 0:
            return
        phi = r.phi
        phi_star = math.sqrt(phi * phi + periods * (r.sigma * r.sigma))
        r.rd = min(phi_star * SCALE, MAX_RD)

    # ---- actualización de un equipo en un periodo -----------------------
    def update(self, r: Rating, matches: list[_Match], period: int) -> Rating:
        """Devuelve un nuevo Rating tras los 'matches' del periodo 'period'.

        Si no hay partidos, solo se aplica el crecimiento de RD por inactividad.
        """
        # Periodos transcurridos desde la última actividad (mínimo 1 si jugó antes).
        if r.last_period is None:
            elapsed = 0
        else:
            elapsed = max(0, period - r.last_period)

        if not matches:
            new = r.copy()
            # Si nunca jugó, no tocamos RD; si jugó, decae por inactividad.
            if r.last_period is not None and elapsed > 0:
                self._decay_rd(new, elapsed)
            return new

        # Pre-decay: la RD que entra al cálculo refleja la inactividad previa.
        pre = r.copy()
        if r.last_period is not None and elapsed > 0:
            self._decay_rd(pre, elapsed)

        mu = pre.mu
        phi = pre.phi

        # v: varianza estimada de la habilidad solo por resultados del periodo.
        v_inv = 0.0
        delta_sum = 0.0
        for m in matches:
            g_j = self._g(m.opponent.phi)
            e_j = self._expected(mu, m.opponent.mu, m.opponent.phi)
            v_inv += g_j * g_j * e_j * (1.0 - e_j)
            delta_sum += g_j * (m.score - e_j)

        if v_inv <= 0.0:
            # Sin información utilizable; solo decae RD.
            new = pre.copy()
            new.last_period = period
            new.games = r.games + len(matches)
            return new

        v = 1.0 / v_inv
        delta = v * delta_sum

        new_sigma = self._new_volatility(phi, v, delta, pre.sigma)

        phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
        new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        new_mu = mu + new_phi * new_phi * delta_sum

        new = Rating(
            rating=new_mu * SCALE + DEFAULT_RATING,
            rd=min(new_phi * SCALE, MAX_RD),
            sigma=new_sigma,
            last_period=period,
            games=r.games + len(matches),
        )
        return new

    def _new_volatility(self, phi: float, v: float, delta: float, sigma: float) -> float:
        """Algoritmo iterativo de Illinois (Glickman) para la nueva sigma."""
        a = math.log(sigma * sigma)
        tau = self.tau
        delta_sq = delta * delta
        phi_sq = phi * phi

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta_sq - phi_sq - v - ex)
            den = 2.0 * (phi_sq + v + ex) ** 2
            return (num / den) - (x - a) / (tau * tau)

        A = a
        if delta_sq > phi_sq + v:
            B = math.log(delta_sq - phi_sq - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau

        fA = f(A)
        fB = f(B)
        iters = 0
        while abs(B - A) > self.epsilon and iters < 100:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC
            iters += 1
        return math.exp(A / 2.0)

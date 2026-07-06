"""cs2model — núcleo de modelado para el predictor de CS2.

Componentes:
  - glicko2:  sistema de rating Glicko-2 (rating + RD + volatilidad), por
              periodos de rating semanales, calculado cronológicamente.
  - dataio:   carga y normalización de resultados scrapeados de HLTV.
  - features: estado cronológico + construcción de features point-in-time,
              compartido por el backtest y la predicción en vivo (cero fuga
              temporal por diseño).
  - artifacts: carga/guardado del modelo entrenado + calibrador.

El principio rector (ver PROJECT.md): la base de datos guarda la verdad de lo
que pasó; las features son vistas calculadas point-in-time, reconstruibles a
cualquier instante del pasado.
"""

from .glicko2 import Glicko2, Rating
from .features import ChronologicalState, FEATURE_COLUMNS, DIFF_COLUMNS, build_training_frame
from . import dataio

__all__ = [
    "Glicko2",
    "Rating",
    "ChronologicalState",
    "FEATURE_COLUMNS",
    "DIFF_COLUMNS",
    "build_training_frame",
    "dataio",
]

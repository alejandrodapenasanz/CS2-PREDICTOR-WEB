"""Tipos públicos del almacén operativo y sus errores controlados.

Las estructuras de resultado permiten integrar la base de datos sin depender
de detalles de SQLite. Todos los contadores describen una transacción ya
confirmada; si la operación falla, no se devuelve ningún resumen parcial.
"""

from __future__ import annotations

from dataclasses import dataclass


class OperationsError(RuntimeError):
    """Error base controlado de la base operativa de tenis."""


class OperationsSchemaError(OperationsError):
    """Indica una versión SQLite o un DataFrame incompatible."""


class OperationsValidationError(OperationsError):
    """Indica un valor que no puede almacenarse de forma auditable."""


class OperationsConflictError(OperationsError):
    """Indica que una identidad inmutable contradice lo ya persistido."""


@dataclass(frozen=True, slots=True)
class PredictionRegistration:
    """Resume el registro transaccional de una ejecución de predicción."""

    run_id: str
    input_rows: int
    matches_inserted: int
    predictions_inserted: int
    valid_predictions: int
    official_predictions_selected: int
    settlements_inserted: int
    queued_rows: int
    reused: bool


@dataclass(frozen=True, slots=True)
class ObservationReconciliation:
    """Resume una ingesta append-only de observaciones de resultados."""

    run_id: str
    input_rows: int
    observations_inserted: int
    settlements_inserted: int
    conflicts_inserted: int
    queued_rows: int
    reused: bool


@dataclass(frozen=True, slots=True)
class StatisticsRegistration:
    """Resume estadísticas prepartido añadidas sin usar predicciones."""

    run_id: str
    input_rows: int
    statistics_inserted: int


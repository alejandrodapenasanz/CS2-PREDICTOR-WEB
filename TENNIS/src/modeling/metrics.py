"""Métricas probabilísticas, segmentación y curvas de fiabilidad.

Las probabilidades ausentes se excluyen de la evaluación y se reflejan
siempre mediante ``n_total``, ``n_evaluated`` y ``coverage``. Esto permite
comparar, por ejemplo, el mercado solo sobre partidos con dos cuotas válidas
sin presentar su subconjunto como si cubriera todo el histórico.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "n_total",
    "n_evaluated",
    "coverage",
    "positive_rate",
    "accuracy",
    "log_loss",
    "brier",
    "auc",
    "auc_defined",
)


class MetricsError(ValueError):
    """Indica que una evaluación contiene un esquema o valores inválidos."""


def _coerce_targets(values: object) -> np.ndarray:
    """Convierte y valida un vector completo de etiquetas binarias."""

    try:
        vector = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise MetricsError("Las etiquetas no son válidas.") from exc
    if vector.ndim != 1:
        raise MetricsError("Las etiquetas deben ser un vector.")
    if vector.size == 0:
        raise MetricsError("No se puede evaluar un conjunto vacío.")
    if pd.isna(vector).any():
        raise MetricsError("Las etiquetas no pueden contener nulos.")
    try:
        numeric = vector.astype(np.int8)
    except (TypeError, ValueError) as exc:
        raise MetricsError("Las etiquetas deben ser 0 o 1.") from exc
    if not np.array_equal(vector, numeric) or not np.isin(
        numeric, (0, 1)
    ).all():
        raise MetricsError("Las etiquetas deben ser exactamente 0 o 1.")
    return numeric


def _coerce_probabilities(values: object, expected_size: int) -> np.ndarray:
    """Convierte probabilidades, preservando ``NaN`` como falta de cobertura."""

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetricsError("Las probabilidades no son numéricas.") from exc
    if vector.ndim != 1 or vector.size != expected_size:
        raise MetricsError(
            "Las probabilidades deben ser un vector del mismo tamaño."
        )
    infinite = np.isinf(vector)
    if infinite.any():
        raise MetricsError("Las probabilidades no pueden ser infinitas.")
    observed = vector[~np.isnan(vector)]
    if ((observed < 0.0) | (observed > 1.0)).any():
        raise MetricsError("Las probabilidades deben estar dentro de [0, 1].")
    return vector


def binary_classification_metrics(
    targets: object,
    probabilities: object,
    *,
    threshold: float = 0.5,
) -> dict[str, int | float | bool]:
    """Calcula accuracy, log-loss, Brier y AUC con cobertura explícita.

    AUC se devuelve como ``NaN`` y ``auc_defined=False`` cuando el subconjunto
    evaluado solo contiene una clase. Log-loss se calcula con ``labels=[0, 1]``
    y por ello sigue siendo informativa en ese caso.
    """

    if not 0.0 <= threshold <= 1.0:
        raise MetricsError("threshold debe estar dentro de [0, 1].")
    y_true = _coerce_targets(targets)
    y_probability = _coerce_probabilities(
        probabilities,
        expected_size=y_true.size,
    )
    available = ~np.isnan(y_probability)
    n_total = int(y_true.size)
    n_evaluated = int(available.sum())
    coverage = float(n_evaluated / n_total)
    if n_evaluated == 0:
        return {
            "n_total": n_total,
            "n_evaluated": 0,
            "coverage": coverage,
            "positive_rate": float("nan"),
            "accuracy": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "auc": float("nan"),
            "auc_defined": False,
        }

    evaluated_targets = y_true[available]
    evaluated_probabilities = y_probability[available]
    predictions = (evaluated_probabilities >= threshold).astype(np.int8)
    has_two_classes = np.unique(evaluated_targets).size == 2
    auc = (
        float(roc_auc_score(evaluated_targets, evaluated_probabilities))
        if has_two_classes
        else float("nan")
    )
    return {
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "coverage": coverage,
        "positive_rate": float(evaluated_targets.mean()),
        "accuracy": float(
            accuracy_score(evaluated_targets, predictions)
        ),
        "log_loss": float(
            log_loss(
                evaluated_targets,
                evaluated_probabilities,
                labels=[0, 1],
            )
        ),
        "brier": float(
            brier_score_loss(
                evaluated_targets,
                evaluated_probabilities,
            )
        ),
        "auc": auc,
        "auc_defined": has_two_classes,
    }


def _validate_evaluation_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """Comprueba que todas las columnas solicitadas existan."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MetricsError(f"Faltan columnas para evaluar: {missing}.")


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    target_column: str = "y",
    group_columns: Sequence[str] = ("tour_level", "surface"),
    model_name: str | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evalúa una predicción globalmente y por cada segmento observado.

    El resultado contiene una fila ``scope='global'`` y una por combinación
    de las columnas de segmento. Los grupos con valores nulos se conservan.
    """

    required = (probability_column, target_column, *group_columns)
    _validate_evaluation_columns(frame, required)
    if frame.empty:
        raise MetricsError("No se puede evaluar un DataFrame vacío.")

    rows: list[dict[str, object]] = []

    def append_metrics(
        subset: pd.DataFrame,
        scope: str,
        group_values: tuple[object, ...],
    ) -> None:
        """Añade las métricas de un subconjunto al acumulador local."""

        row: dict[str, object] = {
            "model": model_name or probability_column,
            "scope": scope,
        }
        row.update(
            {
                column: value
                for column, value in zip(group_columns, group_values)
            }
        )
        row.update(
            binary_classification_metrics(
                subset[target_column].to_numpy(),
                subset[probability_column].to_numpy(),
                threshold=threshold,
            )
        )
        rows.append(row)

    append_metrics(
        frame,
        "global",
        tuple("ALL" for _ in group_columns),
    )
    if group_columns:
        grouper: str | list[str]
        grouper = (
            group_columns[0]
            if len(group_columns) == 1
            else list(group_columns)
        )
        grouped = frame.groupby(grouper, dropna=False, sort=True)
        for keys, subset in grouped:
            values = keys if isinstance(keys, tuple) else (keys,)
            append_metrics(subset, "segment", values)
    ordered = (
        "model",
        "scope",
        *group_columns,
        *METRIC_COLUMNS,
    )
    return pd.DataFrame(rows).loc[:, ordered]


def evaluate_prediction_columns(
    frame: pd.DataFrame,
    prediction_columns: Mapping[str, str],
    *,
    target_column: str = "y",
    group_columns: Sequence[str] = ("tour_level", "surface"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evalúa varios modelos con exactamente el mismo contrato de métricas.

    Args:
        frame: Observaciones con etiqueta, segmentos y predicciones.
        prediction_columns: Mapa ``nombre del modelo -> columna de probabilidad``.
        target_column: Columna binaria observada.
        group_columns: Dimensiones de segmentación.
        threshold: Umbral común para accuracy.
    """

    if not prediction_columns:
        raise MetricsError("Debe proporcionarse al menos una predicción.")
    reports = [
        evaluate_predictions(
            frame,
            probability_column=column,
            target_column=target_column,
            group_columns=group_columns,
            model_name=name,
            threshold=threshold,
        )
        for name, column in prediction_columns.items()
    ]
    return pd.concat(reports, ignore_index=True)


def reliability_curve_quantile(
    targets: object,
    probabilities: object,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Construye una curva de fiabilidad con bins por cuantiles.

    Los límites duplicados se colapsan, como exige ``pandas.qcut``. Si todas
    las probabilidades son iguales se devuelve un único bin. Las ausencias
    quedan reflejadas en las columnas de cobertura de cada fila.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise MetricsError("n_bins debe ser un entero positivo.")
    y_true = _coerce_targets(targets)
    y_probability = _coerce_probabilities(
        probabilities,
        expected_size=y_true.size,
    )
    available = ~np.isnan(y_probability)
    n_total = int(y_true.size)
    n_evaluated = int(available.sum())
    columns = (
        "bin",
        "probability_min",
        "probability_max",
        "mean_predicted",
        "observed_rate",
        "count",
        "n_total",
        "n_evaluated",
        "coverage",
    )
    if n_evaluated == 0:
        return pd.DataFrame(columns=columns)

    evaluated = pd.DataFrame(
        {
            "target": y_true[available],
            "probability": y_probability[available],
        }
    )
    if evaluated["probability"].nunique() == 1:
        evaluated["bin"] = 0
    else:
        quantiles = min(n_bins, n_evaluated)
        evaluated["bin"] = pd.qcut(
            evaluated["probability"],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
        if evaluated["bin"].isna().all():
            evaluated["bin"] = 0
        evaluated["bin"] = evaluated["bin"].astype(int)

    curve = (
        evaluated.groupby("bin", sort=True, observed=True)
        .agg(
            probability_min=("probability", "min"),
            probability_max=("probability", "max"),
            mean_predicted=("probability", "mean"),
            observed_rate=("target", "mean"),
            count=("target", "size"),
        )
        .reset_index()
    )
    curve["bin"] = np.arange(len(curve), dtype=np.int64)
    curve["n_total"] = n_total
    curve["n_evaluated"] = n_evaluated
    curve["coverage"] = n_evaluated / n_total
    return curve.loc[:, columns]


def audit_high_accuracy_segments(
    metrics: pd.DataFrame,
    *,
    threshold: float = 0.85,
    small_sample_threshold: int = 100,
) -> pd.DataFrame:
    """Señala segmentos con accuracy sospechosamente superior al umbral.

    No se descarta ningún segmento por soporte. ``small_sample`` ayuda a
    distinguir un posible artefacto de muestra pequeña, pero todos los casos
    deben revisarse por duplicados, orientación y separación temporal.
    """

    if not 0.0 <= threshold <= 1.0:
        raise MetricsError("threshold debe estar dentro de [0, 1].")
    if (
        isinstance(small_sample_threshold, bool)
        or not isinstance(small_sample_threshold, int)
        or small_sample_threshold <= 0
    ):
        raise MetricsError(
            "small_sample_threshold debe ser un entero positivo."
        )
    _validate_evaluation_columns(
        metrics,
        ("scope", "accuracy", "n_evaluated"),
    )
    suspicious = metrics.loc[
        (metrics["scope"] == "segment")
        & metrics["accuracy"].notna()
        & (metrics["accuracy"] > threshold)
    ].copy()
    suspicious["accuracy_threshold"] = float(threshold)
    suspicious["small_sample"] = (
        suspicious["n_evaluated"] < small_sample_threshold
    )
    suspicious["audit_reason"] = (
        "accuracy superior al umbral; revisar soporte, duplicados, "
        "orientación, columnas y precedencia temporal"
    )
    return suspicious.reset_index(drop=True)

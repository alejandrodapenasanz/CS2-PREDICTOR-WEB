"""Evaluación global, segmentada, calibrada y sobre soporte común.

El evaluador conserva la cobertura nativa de cada baseline y añade poblaciones
de soporte común para comparaciones justas. En particular, un partido sin
ranking o sin dos cuotas de-vigadas nunca se convierte en empate, favorito
ficticio ni error del baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import GenderBacktestResult, MODEL_PROBABILITY_COLUMNS
from .metrics import (
    METRIC_COLUMNS,
    audit_high_accuracy_segments,
    evaluate_prediction_columns,
    reliability_curve_quantile,
)
from .parameters import (
    DEFAULT_TEMPORAL_EVALUATION_PARAMETERS,
    TemporalEvaluationParameters,
)


SEGMENT_COLUMNS: Final[tuple[str, ...]] = ("tour_level", "surface")
_PROBABILITY_METRICS: Final[tuple[str, ...]] = (
    "log_loss",
    "brier",
    "auc",
    "auc_defined",
)


class EvaluationError(RuntimeError):
    """Indica que las predicciones OOF no se pueden evaluar honestamente."""


@dataclass(frozen=True, slots=True)
class MarketCoverageAudit:
    """Resume si existe soporte para comparar modelo y mercado."""

    gender: str
    total_rows: int
    market_rows: int
    coverage: float
    status: str

    def as_dict(self) -> dict[str, object]:
        """Convierte el resumen a una fila serializable."""

        return {
            "gender": self.gender,
            "total_rows": self.total_rows,
            "market_rows": self.market_rows,
            "coverage": self.coverage,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class GenderEvaluationResult:
    """Tablas completas de evaluación para un género."""

    gender: str
    metrics: pd.DataFrame
    reliability: pd.DataFrame
    suspicious_segments: pd.DataFrame
    market_audit: MarketCoverageAudit


def _valid_decisions(series: pd.Series) -> pd.Series:
    """Valida decisiones binarias nullable y devuelve versión numérica."""

    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna()
    if not observed.isin([0, 1]).all():
        raise EvaluationError(
            "El baseline determinista contiene decisiones distintas de 0/1."
        )
    return numeric


def _decision_metrics(
    frame: pd.DataFrame,
    *,
    decision_column: str,
    model_name: str,
    group_columns: Sequence[str] = SEGMENT_COLUMNS,
) -> pd.DataFrame:
    """Calcula exclusivamente accuracy para un baseline determinista."""

    required = {"y", decision_column, *group_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EvaluationError(
            f"Faltan columnas para evaluar decisiones: {missing}."
        )
    if frame.empty:
        raise EvaluationError("No se evalúan decisiones sobre un frame vacío.")
    rows: list[dict[str, object]] = []

    def append(
        subset: pd.DataFrame,
        *,
        scope: str,
        keys: tuple[object, ...],
    ) -> None:
        """Añade una fila global o segmentada al acumulador."""

        target = pd.to_numeric(subset["y"], errors="coerce")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise EvaluationError("El objetivo no es binario completo.")
        decision = _valid_decisions(subset[decision_column])
        available = decision.notna()
        n_total = len(subset)
        n_evaluated = int(available.sum())
        row: dict[str, object] = {
            "model": model_name,
            "scope": scope,
            **{
                column: value
                for column, value in zip(group_columns, keys)
            },
            "n_total": n_total,
            "n_evaluated": n_evaluated,
            "coverage": n_evaluated / n_total,
            "positive_rate": (
                float(target.loc[available].mean())
                if n_evaluated
                else float("nan")
            ),
            "accuracy": (
                float(
                    (
                        decision.loc[available].astype(int)
                        == target.loc[available].astype(int)
                    ).mean()
                )
                if n_evaluated
                else float("nan")
            ),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "auc": float("nan"),
            "auc_defined": False,
        }
        rows.append(row)

    append(
        frame,
        scope="global",
        keys=tuple("ALL" for _ in group_columns),
    )
    grouped = frame.groupby(
        list(group_columns), dropna=False, sort=True
    )
    for keys, subset in grouped:
        values = keys if isinstance(keys, tuple) else (keys,)
        append(subset, scope="segment", keys=values)
    ordered = (
        "model",
        "scope",
        *group_columns,
        *METRIC_COLUMNS,
    )
    return pd.DataFrame(rows).loc[:, ordered]


def _evaluate_population(
    frame: pd.DataFrame,
    *,
    population: str,
    probability_columns: Mapping[str, str],
    include_ranking_favorite: bool,
) -> pd.DataFrame:
    """Evalúa modelos probabilísticos y, opcionalmente, el favorito ranking."""

    if frame.empty:
        return pd.DataFrame()
    metrics = evaluate_prediction_columns(
        frame,
        probability_columns,
        group_columns=SEGMENT_COLUMNS,
    )
    if include_ranking_favorite:
        favorite = _decision_metrics(
            frame,
            decision_column="ranking_favorite_decision",
            model_name="ranking_favorite",
        )
        metrics = pd.concat([metrics, favorite], ignore_index=True)
    metrics.insert(0, "population", population)
    return metrics


def _evaluate_temporal_cells(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evalúa temporada×nivel×superficie solo para auditar extremos."""

    group_columns = ("test_season", *SEGMENT_COLUMNS)
    probability_metrics = evaluate_prediction_columns(
        predictions,
        MODEL_PROBABILITY_COLUMNS,
        group_columns=group_columns,
    )
    ranking_metrics = _decision_metrics(
        predictions,
        decision_column="ranking_favorite_decision",
        model_name="ranking_favorite",
        group_columns=group_columns,
    )
    metrics = pd.concat(
        (probability_metrics, ranking_metrics),
        ignore_index=True,
    )
    metrics.insert(0, "population", "native_temporal_cell")
    return metrics


def _build_reliability(
    predictions: pd.DataFrame,
    *,
    gender: str,
    n_bins: int,
) -> pd.DataFrame:
    """Genera curvas antes/después para los dos estimadores."""

    frames: list[pd.DataFrame] = []
    for model in (
        "logistic_raw",
        "logistic_platt",
        "lightgbm_raw",
        "lightgbm_platt",
    ):
        curve = reliability_curve_quantile(
            predictions["y"].to_numpy(),
            predictions[model].to_numpy(dtype=float),
            n_bins=n_bins,
        )
        curve.insert(0, "model", model)
        curve.insert(0, "gender", gender)
        frames.append(curve)
    if predictions["market_devig"].notna().any():
        curve = reliability_curve_quantile(
            predictions["y"].to_numpy(),
            predictions["market_devig"].to_numpy(
                dtype=float, na_value=np.nan
            ),
            n_bins=n_bins,
        )
        curve.insert(0, "model", "market_devig")
        curve.insert(0, "gender", gender)
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def _augment_suspicious_audit(
    suspicious: pd.DataFrame,
    result: GenderBacktestResult,
) -> pd.DataFrame:
    """Añade comprobaciones estructurales a cada accuracy sospechosa."""

    if suspicious.empty:
        return suspicious.assign(
            gender=pd.Series(dtype=str),
            duplicate_record_ids=pd.Series(dtype=int),
            temporal_contract_passed=pd.Series(dtype=bool),
            forbidden_predictors_present=pd.Series(dtype=str),
            audit_conclusion=pd.Series(dtype=str),
        )
    forbidden = {
        "record_id",
        "gender",
        "match_date",
        "player_a_id",
        "player_b_id",
        "y",
        "model_probability_a",
        "edge",
    }
    present_forbidden = sorted(
        forbidden.intersection(result.feature_columns)
    )
    duplicate_ids = int(
        result.predictions["record_id"].duplicated().sum()
    )
    temporal_passed = all(
        fold.train_max_date < fold.calibration_min_date
        and fold.calibration_max_date < fold.test_min_date
        for fold in result.folds
    )
    audited = suspicious.copy()
    if "gender" in audited.columns:
        observed = set(audited["gender"].dropna().astype(str).unique())
        if observed and observed != {result.gender}:
            raise EvaluationError(
                "La auditoría sospechosa mezcla universos de género."
            )
        audited["gender"] = result.gender
    else:
        audited.insert(0, "gender", result.gender)
    audited["duplicate_record_ids"] = duplicate_ids
    audited["temporal_contract_passed"] = temporal_passed
    audited["forbidden_predictors_present"] = ",".join(present_forbidden)
    audited["audit_conclusion"] = np.where(
        audited["small_sample"],
        "muestra pequeña; no se detectó fuga estructural",
        "sin fuga estructural detectada; requiere revisión estadística",
    )
    return audited


def evaluate_gender_backtest(
    result: GenderBacktestResult,
    *,
    parameters: TemporalEvaluationParameters = (
        DEFAULT_TEMPORAL_EVALUATION_PARAMETERS
    ),
) -> GenderEvaluationResult:
    """Evalúa OOF nativo, ranking común y mercado común.

    La tabla ``native`` conserva la cobertura propia de cada predictor. La
    población ``ranking_common_support`` permite comparar modelos y ranking en
    las mismas filas. ``market_common_support`` solo se crea cuando existe al
    menos una probabilidad de mercado observada.
    """

    predictions = result.predictions
    required = {
        "record_id",
        "gender",
        "y",
        "tour_level",
        "surface",
        "ranking_favorite_decision",
        *MODEL_PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise EvaluationError(
            f"Faltan predicciones OOF para evaluar: {missing}."
        )
    if predictions.empty:
        raise EvaluationError("Las predicciones OOF están vacías.")

    native = _evaluate_population(
        predictions,
        population="native",
        probability_columns=MODEL_PROBABILITY_COLUMNS,
        include_ranking_favorite=True,
    )
    ranking_mask = predictions["ranking_favorite_decision"].notna()
    ranking_common = _evaluate_population(
        predictions.loc[ranking_mask].copy(),
        population="ranking_common_support",
        probability_columns={
            "logistic_platt": "logistic_platt",
            "lightgbm_platt": "lightgbm_platt",
            "ranking_probability": "ranking_probability",
        },
        include_ranking_favorite=True,
    )
    market_mask = predictions["market_devig"].notna()
    market_rows = int(market_mask.sum())
    market_common = _evaluate_population(
        predictions.loc[market_mask].copy(),
        population="market_common_support",
        probability_columns={
            "logistic_platt": "logistic_platt",
            "lightgbm_platt": "lightgbm_platt",
            "market_devig": "market_devig",
        },
        include_ranking_favorite=False,
    )
    metrics = pd.concat(
        [
            frame
            for frame in (native, ranking_common, market_common)
            if not frame.empty
        ],
        ignore_index=True,
    )
    metrics.insert(0, "gender", result.gender)
    market_audit = MarketCoverageAudit(
        gender=result.gender,
        total_rows=len(predictions),
        market_rows=market_rows,
        coverage=market_rows / len(predictions),
        status=(
            "EVALUABLE"
            if market_rows
            else "NO EVALUABLE — cobertura histórica 0 %"
        ),
    )
    reliability = _build_reliability(
        predictions,
        gender=result.gender,
        n_bins=parameters.reliability_bins,
    )
    suspicious = audit_high_accuracy_segments(
        metrics,
        threshold=parameters.suspicious_accuracy_threshold,
        small_sample_threshold=parameters.small_segment_threshold,
    )
    temporal_cells = _evaluate_temporal_cells(predictions)
    temporal_cells.insert(0, "gender", result.gender)
    suspicious_temporal = audit_high_accuracy_segments(
        temporal_cells,
        threshold=parameters.suspicious_accuracy_threshold,
        small_sample_threshold=parameters.small_segment_threshold,
    )
    if not suspicious_temporal.empty:
        suspicious = pd.concat(
            (suspicious, suspicious_temporal),
            ignore_index=True,
            sort=False,
        )
    suspicious = _augment_suspicious_audit(suspicious, result)
    return GenderEvaluationResult(
        gender=result.gender,
        metrics=metrics,
        reliability=reliability,
        suspicious_segments=suspicious,
        market_audit=market_audit,
    )

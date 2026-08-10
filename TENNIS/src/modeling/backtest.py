"""Backtest expansivo por género y ajuste causal de calibradores.

Cada modelo base de un fold se ajusta hasta ``Y-2``. Sus probabilidades sobre
``Y-1`` alimentan exclusivamente un calibrador Platt, y ambos quedan congelados
antes de predecir ``Y``. Las predicciones de test acumuladas son, por tanto,
out-of-fold temporales y pueden auditarse fila a fila.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gc
from typing import Callable, Final, Mapping

import numpy as np
import pandas as pd

from .baselines import (
    RankProbabilityBaseline,
    market_baseline_predictions,
    ranking_favorite_predictions,
)
from .calibration import PlattCalibrator
from .estimators import (
    FittedGenderEstimator,
    fit_gender_estimator,
)
from .parameters import (
    DEFAULT_LIGHTGBM_PARAMETERS,
    DEFAULT_LOGISTIC_PARAMETERS,
    DEFAULT_TEMPORAL_EVALUATION_PARAMETERS,
    LightGBMParameters,
    LogisticParameters,
    TemporalEvaluationParameters,
)
from .orientation import (
    fit_symmetric_platt,
    predict_symmetric_calibrated,
    predict_symmetric_raw,
)
from .preprocessing import FeatureContract, FeatureProfile
from .splits import build_expanding_season_folds


PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "logistic_raw",
    "logistic_platt",
    "lightgbm_raw",
    "lightgbm_platt",
    "ranking_probability",
    "market_devig",
)
MODEL_PROBABILITY_COLUMNS: Final[Mapping[str, str]] = {
    name: name for name in PREDICTION_COLUMNS
}


class BacktestError(RuntimeError):
    """Indica que el backtest no pudo respetar su contrato temporal."""


@dataclass(frozen=True, slots=True)
class FoldAudit:
    """Resume tamaños, cortes y balance de un fold ya ejecutado."""

    gender: str
    test_season: int
    train_rows: int
    calibration_rows: int
    test_rows: int
    train_min_date: date
    train_max_date: date
    train_available_max_date: date
    calibration_min_date: date
    calibration_max_date: date
    calibration_available_max_date: date
    test_min_date: date
    test_max_date: date
    train_positive_rate: float
    calibration_positive_rate: float
    test_positive_rate: float

    def as_dict(self) -> dict[str, object]:
        """Convierte la auditoría a un mapping JSON/CSV serializable."""

        return {
            "gender": self.gender,
            "test_season": self.test_season,
            "train_rows": self.train_rows,
            "calibration_rows": self.calibration_rows,
            "test_rows": self.test_rows,
            "train_min_date": self.train_min_date.isoformat(),
            "train_max_date": self.train_max_date.isoformat(),
            "train_available_max_date": (
                self.train_available_max_date.isoformat()
            ),
            "calibration_min_date": self.calibration_min_date.isoformat(),
            "calibration_max_date": self.calibration_max_date.isoformat(),
            "calibration_available_max_date": (
                self.calibration_available_max_date.isoformat()
            ),
            "test_min_date": self.test_min_date.isoformat(),
            "test_max_date": self.test_max_date.isoformat(),
            "train_positive_rate": self.train_positive_rate,
            "calibration_positive_rate": self.calibration_positive_rate,
            "test_positive_rate": self.test_positive_rate,
        }


@dataclass(frozen=True, slots=True)
class GenderBacktestResult:
    """Predicciones temporales y auditoría de un universo de género."""

    gender: str
    predictions: pd.DataFrame
    folds: tuple[FoldAudit, ...]
    profile: str
    feature_columns: tuple[str, ...]


@dataclass(slots=True)
class GenderFinalModels:
    """Modelos finales y calibradores OOF para un único género."""

    gender: str
    logistic: FittedGenderEstimator
    logistic_calibrator: PlattCalibrator
    lightgbm: FittedGenderEstimator
    lightgbm_calibrator: PlattCalibrator
    ranking_probability: RankProbabilityBaseline
    training_rows: int
    training_source_rows: int
    training_excluded_unavailable: int
    training_max_date: date
    training_available_max_date: date
    training_as_of_date: date
    calibration_rows: int


def _calendar_date(series: pd.Series, operation: str) -> date:
    """Extrae el mínimo o máximo de una serie temporal como fecha."""

    if operation == "min":
        value = series.min()
    elif operation == "max":
        value = series.max()
    else:
        raise ValueError("operation debe ser 'min' o 'max'.")
    if not isinstance(value, pd.Timestamp):
        value = pd.Timestamp(value)
    return value.date()


def _positive_rate(frame: pd.DataFrame) -> float:
    """Calcula la tasa positiva de un bloque binario no vacío."""

    if frame.empty:
        raise BacktestError("No se puede auditar un bloque vacío.")
    target = pd.to_numeric(frame["y"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise BacktestError("El objetivo del fold no es binario completo.")
    return float(target.mean())


def _fold_audit(
    gender: str,
    test_season: int,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> FoldAudit:
    """Construye y revalida la precedencia observada de un fold."""

    train_max = _calendar_date(train["match_date"], "max")
    train_available_max = _calendar_date(
        train["result_available_date"], "max"
    )
    calibration_min = _calendar_date(calibration["match_date"], "min")
    calibration_max = _calendar_date(calibration["match_date"], "max")
    calibration_available_max = _calendar_date(
        calibration["result_available_date"], "max"
    )
    test_min = _calendar_date(test["match_date"], "min")
    if (
        not train_available_max < calibration_min
        or not calibration_available_max < test_min
    ):
        raise BacktestError(
            f"El fold {gender}/{test_season} viola "
            "train < calibración < test."
        )
    return FoldAudit(
        gender=gender,
        test_season=test_season,
        train_rows=len(train),
        calibration_rows=len(calibration),
        test_rows=len(test),
        train_min_date=_calendar_date(train["match_date"], "min"),
        train_max_date=train_max,
        train_available_max_date=train_available_max,
        calibration_min_date=calibration_min,
        calibration_max_date=calibration_max,
        calibration_available_max_date=calibration_available_max,
        test_min_date=test_min,
        test_max_date=_calendar_date(test["match_date"], "max"),
        train_positive_rate=_positive_rate(train),
        calibration_positive_rate=_positive_rate(calibration),
        test_positive_rate=_positive_rate(test),
    )


def _model_fold_probabilities(
    *,
    kind: str,
    gender: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    contract: FeatureContract,
    profile: FeatureProfile,
    logistic_parameters: LogisticParameters,
    lightgbm_parameters: LightGBMParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta base y Platt en bloques disjuntos y predice el test."""

    if kind not in {"logistic", "lightgbm"}:
        raise ValueError("kind debe ser logistic o lightgbm.")
    estimator = fit_gender_estimator(
        train,
        gender=gender,
        kind=kind,  # type: ignore[arg-type]
        contract=contract,
        profile=profile,
        logistic_parameters=logistic_parameters,
        lightgbm_parameters=lightgbm_parameters,
    )
    calibration_raw = predict_symmetric_raw(
        estimator,
        calibration,
    ).to_numpy(dtype=float)
    calibrator = fit_symmetric_platt(
        calibration_raw,
        calibration["y"].to_numpy(dtype=np.int8),
    )
    test_raw = predict_symmetric_raw(
        estimator,
        test,
    ).to_numpy(dtype=float)
    test_calibrated = predict_symmetric_calibrated(
        calibrator,
        test_raw,
    )
    return test_raw, test_calibrated


def _prediction_reference_frame(
    test: pd.DataFrame,
    *,
    test_season: int,
) -> pd.DataFrame:
    """Conserva únicamente identidad, contexto y etiqueta para evaluación."""

    columns = (
        "record_id",
        "gender",
        "match_date",
        "result_available_date",
        "tour_level",
        "surface",
        "y",
    )
    missing = sorted(set(columns).difference(test.columns))
    if missing:
        raise BacktestError(
            f"Faltan referencias en el bloque de test: {missing}."
        )
    output = test.loc[:, list(columns)].copy()
    output["test_season"] = int(test_season)
    return output


def run_gender_backtest(
    frame: pd.DataFrame,
    *,
    gender: str,
    contract: FeatureContract,
    profile: FeatureProfile = "auto",
    evaluation_parameters: TemporalEvaluationParameters = (
        DEFAULT_TEMPORAL_EVALUATION_PARAMETERS
    ),
    logistic_parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS,
    lightgbm_parameters: LightGBMParameters = (
        DEFAULT_LIGHTGBM_PARAMETERS
    ),
    progress: Callable[[str], None] | None = None,
) -> GenderBacktestResult:
    """Ejecuta todos los folds temporales de un género.

    Args:
        frame: Dataset causal completo del género.
        gender: ``M`` o ``F``.
        contract: Allowlist verificada de la fase 6.
        profile: Perfil deportivo o enriquecido con mercado.
        evaluation_parameters: Temporadas y umbrales versionados.
        logistic_parameters: Configuración de la regresión logística.
        lightgbm_parameters: Configuración de LightGBM.
        progress: Callback opcional para mensajes breves por fold.

    Returns:
        Predicciones out-of-fold y auditorías temporales.
    """

    if gender not in {"M", "F"}:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    if frame.empty:
        raise BacktestError(f"El dataset {gender} está vacío.")
    observed = set(frame["gender"].dropna().astype(str).unique())
    if observed != {gender} or frame["gender"].isna().any():
        raise BacktestError(
            f"El backtest {gender} recibió universos {observed}."
        )
    test_seasons = range(
        evaluation_parameters.first_test_season,
        evaluation_parameters.last_test_season + 1,
    )
    folds = build_expanding_season_folds(
        frame,
        test_seasons=test_seasons,
        record_id_column="record_id",
    )
    resolved_profile = contract.resolve_profile(profile)
    feature_columns = contract.columns_for(resolved_profile)
    outputs: list[pd.DataFrame] = []
    audits: list[FoldAudit] = []

    for fold in folds:
        train, calibration, test = fold.take(frame)
        audit = _fold_audit(
            gender,
            fold.test_season,
            train,
            calibration,
            test,
        )
        if progress is not None:
            progress(
                f"{gender} fold {fold.test_season}: "
                f"train={len(train)}, calibración={len(calibration)}, "
                f"test={len(test)}"
            )
        output = _prediction_reference_frame(
            test, test_season=fold.test_season
        )

        logistic_raw, logistic_platt = _model_fold_probabilities(
            kind="logistic",
            gender=gender,
            train=train,
            calibration=calibration,
            test=test,
            contract=contract,
            profile=resolved_profile,
            logistic_parameters=logistic_parameters,
            lightgbm_parameters=lightgbm_parameters,
        )
        output["logistic_raw"] = logistic_raw
        output["logistic_platt"] = logistic_platt
        gc.collect()

        lightgbm_raw, lightgbm_platt = _model_fold_probabilities(
            kind="lightgbm",
            gender=gender,
            train=train,
            calibration=calibration,
            test=test,
            contract=contract,
            profile=resolved_profile,
            logistic_parameters=logistic_parameters,
            lightgbm_parameters=lightgbm_parameters,
        )
        output["lightgbm_raw"] = lightgbm_raw
        output["lightgbm_platt"] = lightgbm_platt
        gc.collect()

        ranking_probability = RankProbabilityBaseline(
            logistic_parameters
        ).fit(train, train["y"]).predict(test)
        output["ranking_probability"] = ranking_probability.probability_a
        ranking_favorite = ranking_favorite_predictions(test)
        output["ranking_favorite_decision"] = (
            ranking_favorite.predicted_y
        )
        market = market_baseline_predictions(test)
        output["market_devig"] = market.probability_a

        outputs.append(output)
        audits.append(audit)
        del train, calibration, test
        gc.collect()

    predictions = pd.concat(outputs, ignore_index=True)
    if predictions["record_id"].duplicated().any():
        raise BacktestError(
            "Una identidad apareció en más de un fold de test."
        )
    if set(predictions["test_season"].unique()) != set(test_seasons):
        raise BacktestError("No se cubrieron todas las temporadas solicitadas.")
    return GenderBacktestResult(
        gender=gender,
        predictions=predictions,
        folds=tuple(audits),
        profile=resolved_profile,
        feature_columns=feature_columns,
    )


def fit_final_gender_models(
    frame: pd.DataFrame,
    *,
    gender: str,
    contract: FeatureContract,
    backtest: GenderBacktestResult,
    profile: FeatureProfile = "auto",
    logistic_parameters: LogisticParameters = DEFAULT_LOGISTIC_PARAMETERS,
    lightgbm_parameters: LightGBMParameters = (
        DEFAULT_LIGHTGBM_PARAMETERS
    ),
    progress: Callable[[str], None] | None = None,
    training_as_of_date: date,
    source_training_rows: int | None = None,
) -> GenderFinalModels:
    """Ajusta modelos finales con todo el histórico y Platt con OOF pasado.

    Los estimadores finales ven todas las filas causales disponibles. Sus
    calibradores no ven predicciones in-sample: se ajustan sobre las
    probabilidades crudas acumuladas por el backtest temporal.
    """

    if frame.empty or backtest.predictions.empty:
        raise BacktestError(
            "El ajuste final requiere histórico y predicciones OOF."
        )
    backtest_predictions = backtest.predictions
    if (
        isinstance(training_as_of_date, datetime)
        or not isinstance(training_as_of_date, date)
    ):
        raise BacktestError("training_as_of_date debe ser date estricto.")
    if source_training_rows is None:
        resolved_source_rows = len(frame)
    elif (
        isinstance(source_training_rows, bool)
        or not isinstance(source_training_rows, int)
        or source_training_rows < len(frame)
    ):
        raise BacktestError(
            "source_training_rows debe ser entero y cubrir el subset causal."
        )
    else:
        resolved_source_rows = source_training_rows
    required_frame = {
        "record_id",
        "gender",
        "match_date",
        "result_available_date",
        "y",
    }
    required_oof = required_frame | {
        "test_season",
        "logistic_raw",
        "lightgbm_raw",
    }
    missing_frame = sorted(required_frame.difference(frame.columns))
    missing_oof = sorted(required_oof.difference(backtest_predictions.columns))
    if missing_frame or missing_oof:
        raise BacktestError(
            "No se puede acreditar la procedencia OOF; faltan columnas: "
            f"histórico={missing_frame}, OOF={missing_oof}."
        )
    if gender not in {"M", "F"}:
        raise BacktestError("gender final debe ser 'M' o 'F'.")
    if set(frame["gender"].dropna().astype(str)) != {gender}:
        raise BacktestError("El histórico final mezcla o contradice el género.")
    if set(backtest_predictions["gender"].dropna().astype(str)) != {gender}:
        raise BacktestError("Las predicciones OOF contradicen el género final.")
    if frame["record_id"].duplicated().any():
        raise BacktestError("El histórico contiene record_id repetidos.")
    if backtest_predictions["record_id"].duplicated().any():
        raise BacktestError("Las predicciones OOF contienen IDs repetidos.")

    reference = frame.set_index("record_id")[[
        "match_date",
        "result_available_date",
        "y",
    ]]
    oof_ids = backtest_predictions["record_id"]
    missing_ids = oof_ids.loc[~oof_ids.isin(reference.index)]
    if not missing_ids.empty:
        raise BacktestError("Las predicciones OOF no pertenecen al histórico.")
    matched = reference.loc[oof_ids.to_list()].reset_index(drop=True)
    reference_dates = pd.to_datetime(matched["match_date"], errors="raise")
    oof_dates = pd.to_datetime(
        backtest_predictions["match_date"], errors="raise"
    ).reset_index(drop=True)
    if not reference_dates.reset_index(drop=True).equals(oof_dates):
        raise BacktestError("Las fechas OOF no coinciden con el histórico.")
    reference_available = pd.to_datetime(
        matched["result_available_date"], errors="raise"
    ).reset_index(drop=True)
    oof_available = pd.to_datetime(
        backtest_predictions["result_available_date"], errors="raise"
    ).reset_index(drop=True)
    if not reference_available.equals(oof_available):
        raise BacktestError(
            "Las fechas de disponibilidad OOF no coinciden con el histórico."
        )
    reference_y = pd.to_numeric(matched["y"], errors="raise").astype("int8")
    oof_y = pd.to_numeric(
        backtest_predictions["y"], errors="raise"
    ).astype("int8").reset_index(drop=True)
    if not reference_y.reset_index(drop=True).equals(oof_y):
        raise BacktestError("Las etiquetas OOF no coinciden con el histórico.")
    seasons = pd.to_numeric(
        backtest_predictions["test_season"], errors="raise"
    ).astype("int64").reset_index(drop=True)
    if not seasons.equals(oof_dates.dt.year.astype("int64")):
        raise BacktestError("test_season OOF no coincide con match_date.")
    fold_by_season = {fold.test_season: fold for fold in backtest.folds}
    if set(seasons.unique()) != set(fold_by_season):
        raise BacktestError("Las temporadas OOF no coinciden con sus folds.")
    for season, fold in fold_by_season.items():
        if not (
            fold.train_available_max_date < fold.calibration_min_date
            and fold.calibration_available_max_date < fold.test_min_date
        ):
            raise BacktestError("Un fold final viola train < calibración < test.")
        season_dates = oof_dates.loc[seasons.eq(season)]
        season_ids = set(
            backtest_predictions.loc[
                seasons.eq(season), "record_id"
            ].astype(str)
        )
        frame_dates = pd.to_datetime(frame["match_date"], errors="raise")
        expected_ids = set(
            frame.loc[
                frame_dates.dt.year.eq(season), "record_id"
            ].astype(str)
        )
        if (
            season_dates.empty
            or len(season_dates) != fold.test_rows
            or season_ids != expected_ids
            or season_dates.min().date() != fold.test_min_date
            or season_dates.max().date() != fold.test_max_date
        ):
            raise BacktestError("El inventario OOF no reconcilia con sus folds.")
    for probability_column in ("logistic_raw", "lightgbm_raw"):
        probabilities = pd.to_numeric(
            backtest_predictions[probability_column], errors="raise"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or np.any((probabilities < 0.0) | (probabilities > 1.0))
        ):
            raise BacktestError(
                f"{probability_column} OOF contiene probabilidades inválidas."
            )
    available_dates = pd.to_datetime(
        frame["result_available_date"], errors="raise"
    ).dt.date
    causal_training = frame.loc[
        available_dates.map(lambda value: value < training_as_of_date)
    ].copy()
    causal_oof = backtest_predictions.loc[
        oof_available.map(lambda value: value.date() < training_as_of_date)
    ].copy()
    if causal_training.empty or causal_oof.empty:
        raise BacktestError(
            "El corte final no deja entrenamiento y calibración OOF causales."
        )
    if progress is not None:
        progress(f"{gender}: ajuste final de regresión logística")
    logistic = fit_gender_estimator(
        causal_training,
        gender=gender,
        kind="logistic",
        contract=contract,
        profile=profile,
        logistic_parameters=logistic_parameters,
        lightgbm_parameters=lightgbm_parameters,
    )
    logistic_calibrator = fit_symmetric_platt(
        causal_oof["logistic_raw"].to_numpy(dtype=float),
        causal_oof["y"].to_numpy(dtype=np.int8),
    )
    if progress is not None:
        progress(f"{gender}: ajuste final de LightGBM")
    lightgbm = fit_gender_estimator(
        causal_training,
        gender=gender,
        kind="lightgbm",
        contract=contract,
        profile=profile,
        logistic_parameters=logistic_parameters,
        lightgbm_parameters=lightgbm_parameters,
    )
    lightgbm_calibrator = fit_symmetric_platt(
        causal_oof["lightgbm_raw"].to_numpy(dtype=float),
        causal_oof["y"].to_numpy(dtype=np.int8),
    )
    rank_probability = RankProbabilityBaseline(
        logistic_parameters
    ).fit(causal_training, causal_training["y"])
    return GenderFinalModels(
        gender=gender,
        logistic=logistic,
        logistic_calibrator=logistic_calibrator,
        lightgbm=lightgbm,
        lightgbm_calibrator=lightgbm_calibrator,
        ranking_probability=rank_probability,
        training_rows=len(causal_training),
        training_source_rows=resolved_source_rows,
        training_excluded_unavailable=(
            resolved_source_rows - len(causal_training)
        ),
        training_max_date=_calendar_date(
            causal_training["match_date"], "max"
        ),
        training_available_max_date=_calendar_date(
            causal_training["result_available_date"], "max"
        ),
        training_as_of_date=training_as_of_date,
        calibration_rows=len(causal_oof),
    )

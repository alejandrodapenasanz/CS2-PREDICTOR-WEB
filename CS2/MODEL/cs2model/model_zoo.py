"""Zoo de algoritmos para el harness anidado (docs/MODEL_UPGRADE.md).

Interfaz comun `ZooModel`: .fit(X, y, sample_weight) / .predict_proba(X) -> P(y=1),
mas una POLITICA DE MISSING declarada por modelo:

  * arboles NaN-native (LightGBM, HistGradientBoosting, XGBoost/CatBoost si estan):
    reciben NaN tal cual -> el arbol aprende la direccion del hueco.
  * modelos lineales / RF / ExtraTrees: imputacion (mediana de train) + FLAGS de
    disponibilidad; el 0 nunca se usa como relleno numerico.

Elo/Glicko/TrueSkill/BT/Kalman se quedan como FEATURES; aqui no compiten como
modelos finales. Los estimadores GBDT opcionales quedan import-guarded: si la
libreria no esta instalada, el modelo no aparece en `available_models()`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# --- dependencias opcionales (guarded) --------------------------------------
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier  # type: ignore
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)


# ===========================================================================
# Preparacion de features segun politica de missing
# ===========================================================================
class _Prep:
    """Aprende en train: imputacion por mediana + flags + (opcional) estandarizado."""

    def __init__(self, standardize: bool) -> None:
        self.standardize = standardize
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import warnings
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columna todo-NaN -> mediana 0
            self.median = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        return self._apply(X, learn=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._apply(np.asarray(X, dtype=float), learn=False)

    def _apply(self, X: np.ndarray, *, learn: bool) -> np.ndarray:
        missing = (~np.isfinite(X)).astype(float)          # flags de disponibilidad
        Xi = np.where(np.isfinite(X), X, self.median)
        out = np.hstack([Xi, missing])
        if self.standardize:
            if learn:
                self.mean = out.mean(axis=0)
                self.std = np.where(out.std(axis=0) < 1e-8, 1.0, out.std(axis=0))
            out = (out - self.mean) / self.std
        return out


# ===========================================================================
# Interfaz comun
# ===========================================================================
class ZooModel:
    name: str = "base"
    missing: str = "impute"   # 'impute' | 'nan'

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ZooModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def optuna_space(trial: Any) -> dict[str, Any]:
        return {}


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)


# ===========================================================================
# Modelos lineales / bosques (impute + flags)
# ===========================================================================
class ElasticNetLogit(ZooModel):
    name = "logistic_en"
    missing = "impute"

    def __init__(self, C: float = 1.0, l1_ratio: float = 0.2) -> None:
        self.C = C
        self.l1_ratio = l1_ratio
        self.prep = _Prep(standardize=True)
        self.clf: LogisticRegression | None = None

    def fit(self, X, y, sample_weight=None):
        import warnings
        Xt = self.prep.fit_transform(X)
        self.clf = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio,
            C=self.C, max_iter=2000, tol=1e-3,
        )
        with warnings.catch_warnings():
            # sklearn>=1.8 depreca 'penalty' (elasticnet sigue funcional hasta 1.10);
            # silenciamos solo ese aviso para no ensuciar el log del harness.
            warnings.simplefilter("ignore", FutureWarning)
            self.clf.fit(Xt, np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return _clip(self.clf.predict_proba(self.prep.transform(X))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}


class _SkForest(ZooModel):
    missing = "impute"
    _cls: type = RandomForestClassifier

    def __init__(self, n_estimators=600, max_depth=None, min_samples_leaf=12, max_features="sqrt"):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, max_features=max_features)
        self.prep = _Prep(standardize=False)
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        Xt = self.prep.fit_transform(X)
        self.clf = self._cls(random_state=42, n_jobs=-1, **self.params)
        self.clf.fit(Xt, np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return _clip(self.clf.predict_proba(self.prep.transform(X))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 40),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
        }


class RandomForestModel(_SkForest):
    name = "random_forest"
    _cls = RandomForestClassifier


class ExtraTreesModel(_SkForest):
    name = "extra_trees"
    _cls = ExtraTreesClassifier


# ===========================================================================
# Arboles NaN-native
# ===========================================================================
class HistGBModel(ZooModel):
    name = "hist_gb"
    missing = "nan"   # HistGradientBoosting soporta NaN nativo

    def __init__(self, learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40):
        self.params = dict(learning_rate=learning_rate, max_iter=max_iter,
                           max_leaf_nodes=max_leaf_nodes, l2_regularization=l2_regularization,
                           min_samples_leaf=min_samples_leaf)
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        self.clf = HistGradientBoostingClassifier(random_state=42, early_stopping=False, **self.params)
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return _clip(self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1200, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
        }


class LightGBMModel(ZooModel):
    name = "lightgbm"
    missing = "nan"

    def __init__(self, learning_rate=0.02, n_estimators=1500, num_leaves=31,
                 min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0):
        self.params = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           subsample=subsample, colsample_bytree=colsample_bytree,
                           reg_lambda=reg_lambda)
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        import warnings
        self.clf = lgb.LGBMClassifier(
            objective="binary", random_state=42, n_jobs=-1, verbose=-1,
            subsample_freq=1, **self.params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return _clip(self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2500, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
        }


class XGBoostModel(ZooModel):
    name = "xgboost"
    missing = "nan"

    def __init__(self, learning_rate=0.02, n_estimators=700, max_depth=4, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=20.0):
        self.params = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                           max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree,
                           reg_lambda=reg_lambda, min_child_weight=min_child_weight)
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        self.clf = xgb.XGBClassifier(
            objective="binary:logistic", random_state=42, n_jobs=-1,
            eval_metric="logloss", tree_method="hist", **self.params,
        )
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return _clip(self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0),
        }


class CatBoostModel(ZooModel):
    name = "catboost"
    missing = "nan"

    def __init__(self, learning_rate=0.02, iterations=2000, depth=5, l2_leaf_reg=6.0):
        self.params = dict(learning_rate=learning_rate, iterations=iterations,
                           depth=depth, l2_leaf_reg=l2_leaf_reg)
        self.clf = None

    def fit(self, X, y, sample_weight=None):
        self.clf = CatBoostClassifier(loss_function="Logloss", random_seed=42,
                                      verbose=False, allow_writing_files=False, **self.params)
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return _clip(self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1])

    @staticmethod
    def optuna_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }


# ===========================================================================
# Registro
# ===========================================================================
_ALL: dict[str, Callable[..., ZooModel]] = {
    "logistic_en": ElasticNetLogit,
    "random_forest": RandomForestModel,
    "extra_trees": ExtraTreesModel,
    "hist_gb": HistGBModel,
}
if HAS_LGB:
    _ALL["lightgbm"] = LightGBMModel
if HAS_XGB:
    _ALL["xgboost"] = XGBoostModel
if HAS_CAT:
    _ALL["catboost"] = CatBoostModel


def available_models() -> list[str]:
    """Modelos utilizables en este entorno (segun dependencias instaladas)."""
    return list(_ALL.keys())


def build_model(name: str, params: dict[str, Any] | None = None) -> ZooModel:
    if name not in _ALL:
        raise KeyError(f"modelo '{name}' no disponible; instalados: {available_models()}")
    return _ALL[name](**(params or {}))


def model_class(name: str) -> Callable[..., ZooModel]:
    return _ALL[name]

"""Versioned runtime configuration for model training and monitoring."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@dataclass(frozen=True)
class TrainingConfig:
    warmup_weeks: int = 10
    min_train_rows: int = 800
    form_half_life_days: float = 120.0
    recency_half_life_days: float = 365.0
    walk_forward_gap_periods: int = 0
    feature_profile: str = "core"
    optuna_trials: int = 8
    optuna_retune_weeks: int = 26
    final_calibration_fraction: float = 0.18
    model_selection_min_history: int = 200


@dataclass(frozen=True)
class FeatureSelectionConfig:
    validation_periods: int = 8
    retune_periods: int = 13
    min_log_loss_gain: float = 0.0005
    min_validation_rows: int = 80
    min_validation_available_rows: int = 20


@dataclass(frozen=True)
class HealthGateConfig:
    max_completed_placeholders: int = 0
    max_foreign_key_errors: int = 0
    max_evaluation_metric_delta: float = 1e-8
    max_log_loss_vs_glicko: float = 0.0
    min_recent_coverage_ratio: float = 0.5
    min_coverage_sample: int = 20


@dataclass(frozen=True)
class DriftConfig:
    rolling_window: int = 200
    reference_window: int = 500
    min_samples: int = 50
    page_hinkley_delta: float = 0.005
    page_hinkley_threshold: float = 5.0
    warning_log_loss_ratio: float = 1.10
    warning_mean_clv: float = -0.02
    regime_map_window_days: int = 120
    regime_previous_window_days: int = 120
    regime_min_observed_maps: int = 20


@dataclass(frozen=True)
class EconomicConfig:
    initial_bankroll: float = 1000.0
    kelly_multiplier: float = 0.25
    max_stake_fraction: float = 0.025
    min_stake_amount: float = 1.0
    max_stake_amount: float = 100.0
    max_profit_amount: float = 5000.0
    min_expected_roi: float = 0.0
    execution_odds_haircut: float = 0.0
    require_closing_odds: bool = False


@dataclass(frozen=True)
class ProjectConfig:
    version: int = 3
    random_seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    health_gates: HealthGateConfig = field(default_factory=HealthGateConfig)
    feature_thresholds: dict[str, int] = field(default_factory=dict)
    estimators: dict[str, dict[str, Any]] = field(default_factory=dict)
    drift: DriftConfig = field(default_factory=DriftConfig)
    economic_backtest: EconomicConfig = field(default_factory=EconomicConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(path or os.environ.get("CS2_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration must be a mapping")
    config = ProjectConfig(
        version=int(raw.get("version", 3)),
        random_seed=int(raw.get("random_seed", 42)),
        training=TrainingConfig(**_mapping(raw.get("training"), "training")),
        feature_selection=FeatureSelectionConfig(
            **_mapping(raw.get("feature_selection"), "feature_selection")
        ),
        health_gates=HealthGateConfig(
            **_mapping(raw.get("health_gates"), "health_gates")
        ),
        feature_thresholds={
            str(key): int(value)
            for key, value in _mapping(raw.get("feature_thresholds"), "feature_thresholds").items()
        },
        estimators={
            str(key): dict(_mapping(value, f"estimators.{key}"))
            for key, value in _mapping(raw.get("estimators"), "estimators").items()
        },
        drift=DriftConfig(**_mapping(raw.get("drift"), "drift")),
        economic_backtest=EconomicConfig(
            **_mapping(raw.get("economic_backtest"), "economic_backtest")
        ),
    )
    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    if config.version < 1:
        raise ValueError("config.version must be >= 1")
    if config.random_seed < 0:
        raise ValueError("random_seed must be >= 0")
    if config.training.min_train_rows < 20:
        raise ValueError("training.min_train_rows must be >= 20")
    if config.training.feature_profile not in {"core", "error-aware"}:
        raise ValueError("training.feature_profile must be core or error-aware")
    if not 0.0 < config.training.final_calibration_fraction < 0.5:
        raise ValueError("training.final_calibration_fraction must be in (0, 0.5)")
    if config.training.model_selection_min_history < 20:
        raise ValueError("training.model_selection_min_history must be >= 20")
    selection = config.feature_selection
    if selection.validation_periods < 2 or selection.retune_periods < 1:
        raise ValueError("feature selection periods are invalid")
    if selection.min_validation_rows < 20:
        raise ValueError("feature selection min_validation_rows must be >= 20")
    if selection.min_validation_available_rows < 1:
        raise ValueError("feature selection availability minimum must be positive")
    if selection.min_log_loss_gain < 0:
        raise ValueError("feature selection min_log_loss_gain must be >= 0")
    gates = config.health_gates
    if gates.max_completed_placeholders < 0 or gates.max_foreign_key_errors < 0:
        raise ValueError("health gate maxima must be >= 0")
    if not 0 < gates.min_recent_coverage_ratio <= 1:
        raise ValueError("health gate coverage ratio must be in (0, 1]")
    if any(value < 0 for value in config.feature_thresholds.values()):
        raise ValueError("feature thresholds must be >= 0")
    if config.drift.min_samples < 10 or config.drift.rolling_window < config.drift.min_samples:
        raise ValueError("drift rolling_window must be >= min_samples >= 10")
    econ = config.economic_backtest
    if econ.initial_bankroll <= 0 or not 0.0 < econ.kelly_multiplier <= 1.0:
        raise ValueError("economic bankroll and Kelly multiplier are invalid")
    if not 0.0 < econ.max_stake_fraction <= 0.25:
        raise ValueError("economic max_stake_fraction must be in (0, 0.25]")
    if not 0.0 <= econ.execution_odds_haircut < 0.25:
        raise ValueError("execution_odds_haircut must be in [0, 0.25)")


_RUNTIME_CONFIG = load_config()


def get_runtime_config() -> ProjectConfig:
    return _RUNTIME_CONFIG


def set_runtime_config(config: ProjectConfig) -> None:
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config


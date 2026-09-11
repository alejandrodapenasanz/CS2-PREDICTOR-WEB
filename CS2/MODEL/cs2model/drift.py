"""Causal drift monitoring for closed out-of-sample predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .config import DriftConfig
from .economic import closing_line_value, safe_decimal_odds


@dataclass
class PageHinkley:
    """One-sided Page-Hinkley detector for increases in a monitored loss."""

    delta: float = 0.005
    threshold: float = 5.0
    count: int = 0
    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        changed = (self.cumulative - self.minimum) > self.threshold
        if changed:
            self.count = 1
            self.mean = value
            self.cumulative = 0.0
            self.minimum = 0.0
        return changed


def binary_log_loss(actual: int, probability: float) -> float:
    p = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return -(actual * math.log(p) + (1 - actual) * math.log(1.0 - p))


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return sum(data) / len(data) if data else None


def _window_comparison(values: list[float], cfg: DriftConfig) -> dict[str, Any]:
    if len(values) < cfg.min_samples:
        return {
            "available": False,
            "n": len(values),
            "reason": f"need_at_least_{cfg.min_samples}_samples",
        }
    current = values[-cfg.rolling_window :]
    reference_end = max(0, len(values) - len(current))
    reference = values[max(0, reference_end - cfg.reference_window) : reference_end]
    current_mean = _mean(current)
    reference_mean = _mean(reference)
    ratio = current_mean / reference_mean if current_mean is not None and reference_mean not in (None, 0.0) else None
    return {
        "available": True,
        "n": len(values),
        "current_n": len(current),
        "reference_n": len(reference),
        "current_mean": current_mean,
        "reference_mean": reference_mean,
        "ratio": ratio,
    }


def _regime_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for record in records:
        value = record.get(field)
        if value in (None, "", "unknown"):
            continue
        groups.setdefault(str(value), []).append(float(record["log_loss"]))
    return {key: {"n": len(values), "mean_log_loss": _mean(values)} for key, values in sorted(groups.items())}


def build_drift_report(
    predictions: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DriftConfig()
    matches = {str(row.get("id")): row for row in match_rows}
    ordered = sorted(predictions, key=lambda row: (str(row.get("date") or ""), str(row.get("match_id") or "")))
    records: list[dict[str, Any]] = []
    loss_detector = PageHinkley(cfg.page_hinkley_delta, cfg.page_hinkley_threshold)
    clv_detector = PageHinkley(cfg.page_hinkley_delta, cfg.page_hinkley_threshold)
    loss_change_points: list[dict[str, Any]] = []
    clv_change_points: list[dict[str, Any]] = []

    for prediction in ordered:
        match_id = str(prediction.get("match_id"))
        match = matches.get(match_id, {})
        actual = int(prediction["actual"])
        probability = float(prediction["prob_team1"])
        loss = binary_log_loss(actual, probability)
        record = {
            "match_id": match_id,
            "date": prediction.get("date"),
            "log_loss": loss,
            "environment": prediction.get("environment"),
            "stage": prediction.get("stage"),
            "patch_version": prediction.get("patch_version") or match.get("patch_version"),
            "map_pool_regime": prediction.get("map_pool_regime") or match.get("map_pool_regime"),
            "clv_price": None,
        }
        if loss_detector.update(loss):
            loss_change_points.append({"index": len(records), "match_id": match_id, "date": record["date"]})

        side = "team1" if probability >= 0.5 else "team2"
        opening = safe_decimal_odds(match.get(f"opening_odds_decimal_{'t1' if side == 'team1' else 't2'}"))
        closing = safe_decimal_odds(match.get(f"closing_odds_decimal_{'t1' if side == 'team1' else 't2'}"))
        if opening is not None and closing is not None:
            clv = closing_line_value(opening, closing)
            record["clv_price"] = clv
            if clv_detector.update(-clv):
                clv_change_points.append({"index": len(records), "match_id": match_id, "date": record["date"]})
        records.append(record)

    losses = [float(record["log_loss"]) for record in records]
    clvs = [float(record["clv_price"]) for record in records if record["clv_price"] is not None]
    loss_window = _window_comparison(losses, cfg)
    clv_window = _window_comparison(clvs, cfg)
    warnings: list[str] = []
    if loss_change_points:
        warnings.append("page_hinkley_log_loss")
    ratio = loss_window.get("ratio")
    if ratio is not None and ratio >= cfg.warning_log_loss_ratio:
        warnings.append("rolling_log_loss_degradation")
    current_clv = clv_window.get("current_mean")
    if current_clv is not None and current_clv <= cfg.warning_mean_clv:
        warnings.append("negative_clv")
    if clv_change_points:
        warnings.append("page_hinkley_clv")

    return {
        "schema_version": 1,
        "method": "closed_oos_rolling_windows_and_page_hinkley",
        "n_predictions": len(records),
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "log_loss": {**loss_window, "page_hinkley_change_points": loss_change_points},
        "clv": {
            **clv_window,
            "coverage": len(clvs) / len(records) if records else 0.0,
            "page_hinkley_change_points": clv_change_points,
            "definition": "opening_decimal_odds / closing_decimal_odds - 1",
        },
        "regimes": {
            "patch_version": _regime_summary(records, "patch_version"),
            "map_pool": _regime_summary(records, "map_pool_regime"),
            "environment": _regime_summary(records, "environment"),
        },
        "thresholds": {
            "rolling_window": cfg.rolling_window,
            "reference_window": cfg.reference_window,
            "min_samples": cfg.min_samples,
            "page_hinkley_delta": cfg.page_hinkley_delta,
            "page_hinkley_threshold": cfg.page_hinkley_threshold,
            "warning_log_loss_ratio": cfg.warning_log_loss_ratio,
            "warning_mean_clv": cfg.warning_mean_clv,
        },
        "note": (
            "Only closed out-of-sample predictions are monitored. Closing odds are audit-only "
            "and never become predictive features."
        ),
    }

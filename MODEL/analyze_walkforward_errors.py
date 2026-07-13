"""Auditoria cientifica de los errores walk-forward del modelo productivo.

Une las predicciones fuera de muestra con las features point-in-time usadas por
el modelo. No modifica ni promociona artefactos: produce evidencia descriptiva,
tests de asociacion y un clasificador auxiliar validado temporalmente.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, pointbiserialr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "MODEL"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from cs2model import dataio
from cs2model.artifacts import load_artifact
from cs2model.features import (
    DIFF_COLUMNS,
    EXTENDED_DIFF_COLUMNS,
    build_training_frame,
)


DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_PREDICTIONS = MODEL_ROOT / "results" / "predictions_walkforward.csv"
DEFAULT_RAW = (
    ROOT
    / "SCRAPPER"
    / "hltv-scraper-api"
    / "hltv_scraper"
    / "data"
    / "raw"
    / "history_10000_2026-06-28"
    / "results_all.json"
)
DEFAULT_OUTPUT = MODEL_ROOT / "results"


def clean(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def logical_match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    teams = tuple(
        sorted(
            (
                (clean(row.get("team1")), int(row.get("score1") or 0)),
                (clean(row.get("team2")), int(row.get("score2") or 0)),
            )
        )
    )
    return (
        str(row.get("date") or "")[:10],
        clean(row.get("event")),
        str(row.get("format") or "").lower(),
        teams,
    )


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def orient_value(feature: str, value: Any, favorite_sign: int, directional: set[str]) -> float:
    number = safe_float(value)
    if number is None:
        return float("nan")
    return number * favorite_sign if feature in directional else number


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    valid = [(index, float(value)) for index, value in enumerate(values) if value is not None and math.isfinite(float(value))]
    if not valid:
        return [None for _ in values]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 1.0
    total = len(ordered)
    for rank_from_end, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = total - rank_from_end + 1
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def read_predictions(path: Path, model_name: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("model") == model_name]
    rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("match_id") or "")))
    return rows


def historical_identity_map(raw_path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not raw_path.exists():
        return {}
    mapping: dict[tuple[Any, ...], dict[str, Any]] = {}
    collisions: set[tuple[Any, ...]] = set()
    for row in dataio.load_results(raw_path, cs2_only=True):
        key = logical_match_key(row)
        if key in mapping:
            collisions.add(key)
        else:
            mapping[key] = row
    for key in collisions:
        mapping.pop(key, None)
    return mapping


def hltv_identity(row: dict[str, Any], historical: dict[tuple[Any, ...], dict[str, Any]]) -> tuple[str | None, str | None]:
    match_id = str(row.get("id") or "")
    source = row if match_id.isdigit() and int(match_id) >= 1_000_000 else historical.get(logical_match_key(row), {})
    hltv_id = str(source.get("id") or "") or None
    link = str(source.get("link") or "") or None
    if link and link.startswith("/"):
        link = "https://www.hltv.org" + link
    if hltv_id and (not hltv_id.isdigit() or int(hltv_id) < 1_000_000):
        hltv_id = None
    return hltv_id, link


def build_audit_frame(
    db_path: Path,
    predictions_path: Path,
    raw_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    artifact = load_artifact()
    if artifact is None:
        raise FileNotFoundError("No existe MODEL/artifacts/model.pkl")
    model_name = str(artifact.metadata.get("production_model") or "")
    predictions = read_predictions(predictions_path, model_name)
    rows = dataio.load_training_rows(db_path=db_path, cs2_only=True)
    half_life = float(artifact.metadata.get("form_half_life_days") or 120.0)
    features, labels, meta, _state = build_training_frame(rows, form_half_life=half_life)
    feature_by_id = {str(item["id"]): feature for item, feature in zip(meta, features)}
    label_by_id = {str(item["id"]): int(label) for item, label in zip(meta, labels)}
    row_by_id = {str(row["id"]): row for row in rows}
    historical = historical_identity_map(raw_path)
    directional = set(DIFF_COLUMNS) | set(EXTENDED_DIFF_COLUMNS)
    availability = [
        "asset_available",
        "analytics_available",
        "player_snapshot_available",
        "ranking_available",
        "roster_available",
        "context_available",
    ]
    records: list[dict[str, Any]] = []
    label_mismatches = 0
    missing_features = 0
    for prediction in predictions:
        match_id = str(prediction.get("match_id") or "")
        feature_row = feature_by_id.get(match_id)
        raw_row = row_by_id.get(match_id)
        if feature_row is None or raw_row is None:
            missing_features += 1
            continue
        probability = float(prediction["prob_team1"])
        actual = int(prediction["actual"])
        if label_by_id.get(match_id) != actual:
            label_mismatches += 1
            continue
        favorite_sign = 1 if probability >= 0.5 else -1
        confidence = max(probability, 1.0 - probability)
        error = int((probability >= 0.5) != bool(actual))
        favorite = raw_row["team1"] if favorite_sign == 1 else raw_row["team2"]
        opponent = raw_row["team2"] if favorite_sign == 1 else raw_row["team1"]
        winner = raw_row["team1"] if actual else raw_row["team2"]
        hltv_id, hltv_url = hltv_identity(raw_row, historical)
        record: dict[str, Any] = {
            "match_id": match_id,
            "hltv_match_id": hltv_id,
            "hltv_url": hltv_url,
            "date": str(prediction.get("date") or raw_row.get("date") or ""),
            "month": str(prediction.get("date") or raw_row.get("date") or "")[:7],
            "event": prediction.get("event") or raw_row.get("event") or "unknown",
            "format": raw_row.get("format") or "unknown",
            "team1": raw_row["team1"],
            "team2": raw_row["team2"],
            "favorite": favorite,
            "opponent": opponent,
            "winner": winner,
            "favorite_side": "team1" if favorite_sign == 1 else "team2",
            "actual_team1": actual,
            "prob_team1": probability,
            "confidence": confidence,
            "expected_error_probability": 1.0 - confidence,
            "error": error,
            "log_loss": -math.log(max(1e-9, confidence if not error else 1.0 - confidence)),
            "opening_odds_available": int(raw_row.get("opening_odds_t1") is not None),
        }
        for feature in artifact.feature_columns:
            record[feature] = orient_value(feature, feature_row.get(feature), favorite_sign, directional)
        for feature in availability:
            record[feature] = float(feature_row.get(feature) or 0.0)
        records.append(record)
    frame = pd.DataFrame(records).sort_values(["date", "match_id"]).reset_index(drop=True)
    metadata = {
        "model": model_name,
        "trained_at": artifact.metadata.get("trained_at"),
        "form_half_life_days": half_life,
        "artifact_feature_count": len(artifact.feature_columns),
        "prediction_rows": len(predictions),
        "audit_rows": len(frame),
        "missing_feature_rows": missing_features,
        "label_mismatches": label_mismatches,
        "orientation": "directional features are multiplied by +1 for a team1 favorite and -1 for a team2 favorite",
    }
    return frame, list(artifact.feature_columns), metadata


def numeric_associations(frame: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    columns = ["confidence", *features]
    usable = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        mask = series.notna()
        if int(mask.sum()) < 100 or int(series[mask].nunique()) < 2:
            continue
        errors = series[mask & (frame["error"] == 1)]
        correct = series[mask & (frame["error"] == 0)]
        if len(errors) < 20 or len(correct) < 20:
            continue
        pooled_var = (
            (len(errors) - 1) * float(errors.var(ddof=1))
            + (len(correct) - 1) * float(correct.var(ddof=1))
        ) / max(1, len(errors) + len(correct) - 2)
        pooled_sd = math.sqrt(max(0.0, pooled_var))
        cohen_d = (float(errors.mean()) - float(correct.mean())) / pooled_sd if pooled_sd else 0.0
        try:
            correlation = float(pointbiserialr(frame.loc[mask, "error"], series[mask]).statistic)
        except ValueError:
            correlation = 0.0
        try:
            p_value = float(mannwhitneyu(errors, correct, alternative="two-sided").pvalue)
        except ValueError:
            p_value = None
        confidence = frame.loc[mask, "confidence"].to_numpy(dtype=float)
        values = series[mask].to_numpy(dtype=float)
        target = frame.loc[mask, "error"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(confidence)), confidence])
        value_residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        target_residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        try:
            if column == "confidence" or np.std(value_residual) <= 1e-12:
                raise ValueError("partial correlation undefined")
            partial_result = pearsonr(value_residual, target_residual)
            partial_r = float(partial_result.statistic)
            partial_p = float(partial_result.pvalue)
        except ValueError:
            partial_r = 0.0
            partial_p = None
        usable.append(
            {
                "feature": column,
                "n": int(mask.sum()),
                "error_mean": float(errors.mean()),
                "correct_mean": float(correct.mean()),
                "error_median": float(errors.median()),
                "correct_median": float(correct.median()),
                "cohen_d": cohen_d,
                "point_biserial_r": correlation,
                "partial_r_controlling_confidence": partial_r,
                "partial_p_value": partial_p,
                "p_value_mann_whitney": p_value,
            }
        )

    if usable:
        matrix_columns = [row["feature"] for row in usable]
        matrix = frame[matrix_columns].replace([np.inf, -np.inf], np.nan)
        imputed = SimpleImputer(strategy="median").fit_transform(matrix)
        discrete = np.array([frame[column].nunique(dropna=True) <= 3 for column in matrix_columns])
        mi = mutual_info_classif(imputed, frame["error"].astype(int), discrete_features=discrete, random_state=42)
        for row, value in zip(usable, mi):
            row["mutual_information"] = float(value)
    adjusted = benjamini_hochberg(row.get("p_value_mann_whitney") for row in usable)
    partial_adjusted = benjamini_hochberg(row.get("partial_p_value") for row in usable)
    for row, q_value, partial_q in zip(usable, adjusted, partial_adjusted):
        row["fdr_q_value"] = q_value
        row["partial_fdr_q_value"] = partial_q
    usable.sort(key=lambda row: abs(float(row["cohen_d"])), reverse=True)
    return usable


def group_stat(frame: pd.DataFrame, segment: str, group: str, mask: pd.Series) -> dict[str, Any] | None:
    mask = mask.fillna(False).astype(bool)
    n = int(mask.sum())
    if n == 0:
        return None
    failures = int(frame.loc[mask, "error"].sum())
    rest_n = len(frame) - n
    rest_failures = int(frame.loc[~mask, "error"].sum())
    rate = failures / n
    baseline = float(frame["error"].mean())
    low, high = wilson_interval(failures, n)
    p_value = None
    if rest_n > 0:
        p_value = float(
            fisher_exact(
                [[failures, n - failures], [rest_failures, rest_n - rest_failures]],
                alternative="two-sided",
            ).pvalue
        )
    return {
        "segment": segment,
        "group": str(group),
        "n": n,
        "failures": failures,
        "failure_rate": rate,
        "failure_rate_ci_low": low,
        "failure_rate_ci_high": high,
        "baseline_failure_rate": baseline,
        "lift_percentage_points": (rate - baseline) * 100.0,
        "risk_ratio": rate / baseline if baseline else None,
        "share_of_all_failures": failures / max(1, int(frame["error"].sum())),
        "p_value_fisher": p_value,
    }


def add_groups(rows: list[dict[str, Any]], frame: pd.DataFrame, segment: str, values: pd.Series, min_n: int = 1) -> None:
    for group in sorted(values.dropna().astype(str).unique()):
        result = group_stat(frame, segment, group, values.astype(str) == group)
        if result and result["n"] >= min_n:
            rows.append(result)


def segment_analysis(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confidence = pd.cut(
        frame["confidence"],
        [0.50, 0.60, 0.70, 0.80, 0.90, 1.01],
        right=False,
        labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
    )
    add_groups(rows, frame, "confidence", confidence)
    add_groups(rows, frame, "format", frame["format"])
    add_groups(rows, frame, "favorite_side", frame["favorite_side"])
    add_groups(rows, frame, "month", frame["month"], min_n=30)
    for column in [
        "event_history_available",
        "opening_odds_available",
        "asset_available",
        "analytics_available",
        "player_snapshot_available",
        "ranking_available",
        "roster_available",
        "context_available",
    ]:
        values = frame[column].fillna(0).ge(0.5).map({True: "available", False: "missing"})
        add_groups(rows, frame, column, values)

    for column in [
        "glicko_diff",
        "elo_diff",
        "glicko_rd_sum",
        "experience_min",
        "format_experience_min",
        "event_experience_min",
        "inactivity_max_days",
    ]:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        try:
            groups = pd.qcut(numeric, 4, duplicates="drop")
        except ValueError:
            continue
        add_groups(rows, frame, f"quartile:{column}", groups)

    add_groups(rows, frame, "event", frame["event"], min_n=20)
    add_groups(rows, frame, "favorite_team", frame["favorite"], min_n=20)
    add_groups(rows, frame, "opponent_team", frame["opponent"], min_n=20)
    adjusted = benjamini_hochberg(row.get("p_value_fisher") for row in rows)
    for row, q_value in zip(rows, adjusted):
        row["fdr_q_value"] = q_value
    return rows


def error_explainer(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    frame = frame.sort_values(["date", "match_id"]).reset_index(drop=True)
    target = frame["error"].astype(int).to_numpy()
    splitter = TimeSeriesSplit(n_splits=5)

    def evaluate(columns: list[str], kind: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        probabilities: list[float] = []
        actuals: list[int] = []
        for train_index, test_index in splitter.split(frame):
            if kind == "ridge_logistic":
                pipeline = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(C=0.005, max_iter=3000),
                )
            else:
                pipeline = make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingClassifier(
                        max_iter=140,
                        max_leaf_nodes=15,
                        learning_rate=0.05,
                        l2_regularization=2.0,
                        random_state=42,
                    ),
                )
            pipeline.fit(frame.iloc[train_index][columns], target[train_index])
            probabilities.extend(pipeline.predict_proba(frame.iloc[test_index][columns])[:, 1].tolist())
            actuals.extend(target[test_index].tolist())
        y_eval = np.asarray(actuals, dtype=int)
        p_eval = np.asarray(probabilities, dtype=float)
        return (
            {
                "n_eval": int(len(y_eval)),
                "roc_auc": float(roc_auc_score(y_eval, p_eval)),
                "average_precision": float(average_precision_score(y_eval, p_eval)),
                "brier": float(brier_score_loss(y_eval, p_eval)),
                "base_error_rate": float(y_eval.mean()),
            },
            y_eval,
            p_eval,
        )

    production_features = [column for column in features if column in frame]
    full_columns = ["confidence", *production_features]
    ridge_full_metrics, y_eval, _p_eval = evaluate(full_columns, "ridge_logistic")
    ridge_feature_metrics, _, _ = evaluate(production_features, "ridge_logistic")
    nonlinear_full_metrics, _, _ = evaluate(full_columns, "hist_gradient_boosting")
    first_eval = len(frame) - len(y_eval)
    confidence_probability = 1.0 - frame.iloc[first_eval:]["confidence"].to_numpy(dtype=float)
    confidence_metrics = {
        "n_eval": int(len(y_eval)),
        "roc_auc": float(roc_auc_score(y_eval, confidence_probability)),
        "average_precision": float(average_precision_score(y_eval, confidence_probability)),
        "brier": float(brier_score_loss(y_eval, confidence_probability)),
        "base_error_rate": float(y_eval.mean()),
    }

    cutoff = int(len(frame) * 0.8)
    final_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=140,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=2.0,
            random_state=42,
        ),
    )
    final_model.fit(frame.iloc[:cutoff][full_columns], target[:cutoff])
    importance = permutation_importance(
        final_model,
        frame.iloc[cutoff:][full_columns],
        target[cutoff:],
        scoring="roc_auc",
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(importance.importances_mean)[::-1]
    permutation = [
        {
            "feature": full_columns[index],
            "roc_auc_drop_mean": float(importance.importances_mean[index]),
            "roc_auc_drop_std": float(importance.importances_std[index]),
        }
        for index in order[:20]
    ]
    best_full_auc = max(ridge_full_metrics["roc_auc"], nonlinear_full_metrics["roc_auc"])
    return {
        "validation": "five expanding chronological splits",
        "confidence_only": confidence_metrics,
        "ridge_production_features_without_confidence": ridge_feature_metrics,
        "ridge_production_features_with_confidence": ridge_full_metrics,
        "nonlinear_production_features_with_confidence": nonlinear_full_metrics,
        "best_incremental_auc_vs_confidence": best_full_auc - confidence_metrics["roc_auc"],
        "permutation_importance_last_20pct": permutation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_correlation(frame: pd.DataFrame, associations: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row["feature"] for row in associations[:12]]
    columns = ["error", *selected]
    correlation = frame[columns].corr(method="spearman").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    for row_index in range(len(columns)):
        for column_index in range(len(columns)):
            value = correlation.iloc[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Correlacion Spearman: error y principales asociaciones")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confidence_errors(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    groups = pd.cut(frame["confidence"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.01], right=False, labels=labels)
    observed = frame.groupby(groups, observed=True)["error"].mean().reindex(labels)
    expected = frame.assign(group=groups).groupby("group", observed=True)["expected_error_probability"].mean().reindex(labels)
    counts = frame.groupby(groups, observed=True)["error"].size().reindex(labels).fillna(0).astype(int)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(x - width / 2, observed * 100.0, width, color="#dc2626", label="Error observado")
    expected_bars = ax.bar(x + width / 2, expected * 100.0, width, color="#2563eb", label="Error esperado por confianza")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"n={n}", ha="center", fontsize=9)
    for observed_bar, expected_bar, n in zip(bars, expected_bars, counts):
        if n < 30:
            observed_bar.set_alpha(0.35)
            expected_bar.set_alpha(0.35)
            observed_bar.set_hatch("//")
            expected_bar.set_hatch("//")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 55)
    ax.set_ylabel("Tasa de error")
    ax.set_xlabel("Confianza asignada al favorito")
    ax.set_title("Fallos walk-forward: observado frente a esperado")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.text(0.01, 0.02, "Franjas con n<30 atenuadas: no concluyentes", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pct(value: Any, digits: int = 1) -> str:
    number = safe_float(value)
    return "-" if number is None else f"{number * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def markdown_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    associations = analysis["numeric_associations"]
    high_confidence_associations = analysis["high_confidence_numeric_associations"]
    segments = analysis["segments"]
    explainer = analysis["error_explainer"]
    failures = analysis["high_confidence_failures"]
    min_risk_n = 100
    risk_segments = [
        row
        for row in segments
        if row["n"] >= min_risk_n
        and row["lift_percentage_points"] > 0
        and row["segment"] not in {"event", "favorite_team", "opponent_team"}
    ]
    risk_segments.sort(key=lambda row: row["lift_percentage_points"], reverse=True)
    format_rows = [row for row in segments if row["segment"] == "format"]
    availability_rows = [
        row
        for row in segments
        if row["segment"].endswith("_available") or row["segment"] == "opening_odds_available"
    ]
    month_rows = [row for row in segments if row["segment"] == "month"]
    confidence_rows = [row for row in segments if row["segment"] == "confidence"]
    lines = [
        "# Auditoria de errores walk-forward\n\n",
        f"Generado: {analysis['generated_at']}\n\n",
        "## Alcance\n\n",
        f"- Modelo: `{analysis['metadata']['model']}` entrenado en `{analysis['metadata']['trained_at']}`.\n",
        f"- Predicciones fuera de muestra: **{summary['n']:,}**.\n",
        f"- Aciertos: **{summary['correct']:,}** ({pct(summary['accuracy'])}).\n",
        f"- Fallos: **{summary['failures']:,}** ({pct(summary['failure_rate'])}).\n",
        f"- Mismatches de etiqueta al unir datos: **{analysis['metadata']['label_mismatches']}**.\n",
        "- Las features direccionales estan orientadas hacia el favorito del modelo; valores positivos significan ventaja del favorito.\n\n",
        "## Hallazgo principal\n\n",
        f"La mayor concentracion esta en partidos de margen pequeno: por debajo de 60% de confianza aparecen "
        f"**{summary['below_60_failures']:,} fallos**, el **{pct(summary['below_60_share_of_failures'])}** de todos los errores. "
        f"Por encima de 80% solo hay **{summary['at_least_80_failures']} fallos** ({pct(summary['at_least_80_share_of_failures'])}).\n\n",
        "Esto no descubre por si solo una feature ausente: confirma que los errores se acumulan donde ambos equipos tienen fuerza parecida. "
        "Las asociaciones de rating, Elo, winrate y score estan muy correlacionadas entre si y describen principalmente ese mismo eje de separacion.\n\n",
        "## Error por confianza\n\n",
        "| Confianza | N | Fallos | Tasa error | Lift vs global | Share fallos |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in confidence_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['failures']} | {pct(row['failure_rate'])} | "
            f"{row['lift_percentage_points']:+.1f} pp | {pct(row['share_of_all_failures'])} |\n"
        )

    lines.extend(
        [
            "\n## Asociaciones numericas principales\n\n",
            "`d` es el tamano del efecto (media fallo - media acierto). En features orientadas, `d<0` significa que los favoritos fallidos tenian una ventaja menor.\n\n",
            "| Feature | Media fallo | Media acierto | Cohen d | r error | r parcial | MI | FDR parcial |\n|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in associations[:18]:
        lines.append(
            f"| {row['feature']} | {num(row['error_mean'])} | {num(row['correct_mean'])} | "
            f"{num(row['cohen_d'])} | {num(row['point_biserial_r'])} | "
            f"{num(row.get('partial_r_controlling_confidence'))} | {num(row.get('mutual_information'), 4)} | "
            f"{num(row.get('partial_fdr_q_value'), 4)} |\n"
        )

    partial_rows = [row for row in associations if row["feature"] != "confidence"]
    partial_rows.sort(key=lambda row: abs(float(row.get("partial_r_controlling_confidence") or 0.0)), reverse=True)
    lines.extend(
        [
            "\n### Asociacion residual tras controlar confianza\n\n",
            "| Feature | r parcial | FDR q |\n|---|---:|---:|\n",
        ]
    )
    for row in partial_rows[:12]:
        lines.append(
            f"| {row['feature']} | {num(row.get('partial_r_controlling_confidence'), 4)} | "
            f"{num(row.get('partial_fdr_q_value'), 4)} |\n"
        )

    significant_partial = [
        row
        for row in partial_rows
        if row.get("partial_fdr_q_value") is not None and float(row["partial_fdr_q_value"]) < 0.05
    ]
    if significant_partial and all(str(row["feature"]).startswith("glicko") for row in significant_partial):
        max_partial = max(abs(float(row["partial_r_controlling_confidence"])) for row in significant_partial)
        lines.append(
            "\nTras controlar la confianza y corregir multiples comparaciones, solo sobreviven las dos "
            f"representaciones casi equivalentes de Glicko (`|r parcial| <= {max_partial:.3f}`). "
            "La asociacion es estadisticamente detectable, pero demasiado pequena para tratarla como "
            "una regla independiente de exclusion.\n"
        )
    elif significant_partial:
        names = ", ".join(f"`{row['feature']}`" for row in significant_partial)
        lines.append(f"\nFeatures que sobreviven FDR al 5% tras controlar confianza: {names}.\n")
    else:
        lines.append("\nNinguna feature conserva una asociacion significativa tras controlar confianza y FDR al 5%.\n")

    lines.extend(
        [
            "\n## Subgrupo de fallos con confianza >=80%\n\n",
            f"En este rango hay **{summary['at_least_80_failures']} fallos de {summary['at_least_80_n']} partidos**. "
            "Se compara solo contra aciertos del mismo rango para no confundir alta confianza con el resto de la muestra.\n\n",
            "| Feature | Media fallo | Media acierto | Cohen d | r parcial | FDR parcial |\n|---|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in high_confidence_associations[:15]:
        lines.append(
            f"| {row['feature']} | {num(row['error_mean'])} | {num(row['correct_mean'])} | "
            f"{num(row['cohen_d'])} | {num(row.get('partial_r_controlling_confidence'))} | "
            f"{num(row.get('partial_fdr_q_value'), 4)} |\n"
        )

    significant_high_confidence = [
        row
        for row in high_confidence_associations
        if row.get("partial_fdr_q_value") is not None and float(row["partial_fdr_q_value"]) < 0.05
    ]
    if significant_high_confidence:
        names = ", ".join(f"`{row['feature']}`" for row in significant_high_confidence)
        lines.append(f"\nEn este subgrupo sobreviven FDR al 5%: {names}.\n")
    else:
        lines.append(
            "\nNinguna feature del subgrupo de alta confianza sobrevive FDR al 5%. Con solo 51 fallos, "
            "los efectos observados son hipotesis para acumular mas muestra, no una explicacion robusta.\n"
        )

    lines.extend(
        [
            "\n## Segmentos con mayor exceso de riesgo\n\n",
            f"Solo se muestran grupos con `n >= {min_risk_n}`. El FDR corrige las multiples comparaciones.\n\n",
            "| Segmento | Grupo | N | Tasa error | Lift | RR | IC95% | FDR q |\n|---|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in risk_segments[:20]:
        lines.append(
            f"| {row['segment']} | {row['group']} | {row['n']} | {pct(row['failure_rate'])} | "
            f"{row['lift_percentage_points']:+.1f} pp | {num(row['risk_ratio'], 2)} | "
            f"{pct(row['failure_rate_ci_low'])}-{pct(row['failure_rate_ci_high'])} | {num(row.get('fdr_q_value'), 4)} |\n"
        )

    lines.extend(["\n## Formato\n\n", "| Formato | N | Tasa error | IC95% | Lift |\n|---|---:|---:|---:|---:|\n"])
    for row in format_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {pct(row['failure_rate'])} | "
            f"{pct(row['failure_rate_ci_low'])}-{pct(row['failure_rate_ci_high'])} | {row['lift_percentage_points']:+.1f} pp |\n"
        )

    lines.extend(
        [
            "\n## Extended features: cobertura exploratoria\n\n",
            "Estas filas no permiten atribucion causal: la cobertura es reciente, pequena y no aleatoria.\n\n",
            "| Feature | Estado | N | Tasa error | Lift | IC95% |\n|---|---|---:|---:|---:|---:|\n",
        ]
    )
    for row in availability_rows:
        if row["group"] != "available":
            continue
        lines.append(
            f"| {row['segment']} | {row['group']} | {row['n']} | {pct(row['failure_rate'])} | "
            f"{row['lift_percentage_points']:+.1f} pp | {pct(row['failure_rate_ci_low'])}-{pct(row['failure_rate_ci_high'])} |\n"
        )

    lines.extend(
        [
            "\n## Estabilidad temporal\n\n",
            "| Mes | N | Tasa error | Lift | IC95% |\n|---|---:|---:|---:|---:|\n",
        ]
    )
    for row in month_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {pct(row['failure_rate'])} | {row['lift_percentage_points']:+.1f} pp | "
            f"{pct(row['failure_rate_ci_low'])}-{pct(row['failure_rate_ci_high'])} |\n"
        )

    confidence_model = explainer["confidence_only"]
    ridge_full = explainer["ridge_production_features_with_confidence"]
    ridge_features = explainer["ridge_production_features_without_confidence"]
    nonlinear_full = explainer["nonlinear_production_features_with_confidence"]
    lines.extend(
        [
            "\n## Se puede predecir el error antes del partido?\n\n",
            f"Validacion: {explainer['validation']}, `n={confidence_model['n_eval']}`.\n\n",
            "| Detector de error | ROC-AUC | Average precision | Brier |\n|---|---:|---:|---:|\n",
            f"| Solo `1 - confianza` | {confidence_model['roc_auc']:.3f} | {confidence_model['average_precision']:.3f} | {confidence_model['brier']:.4f} |\n",
            f"| Ridge: features sin confianza | {ridge_features['roc_auc']:.3f} | {ridge_features['average_precision']:.3f} | {ridge_features['brier']:.4f} |\n",
            f"| Ridge: features + confianza | {ridge_full['roc_auc']:.3f} | {ridge_full['average_precision']:.3f} | {ridge_full['brier']:.4f} |\n",
            f"| HistGradientBoosting: features + confianza | {nonlinear_full['roc_auc']:.3f} | {nonlinear_full['average_precision']:.3f} | {nonlinear_full['brier']:.4f} |\n",
            f"\nMejor incremento AUC frente a usar solo confianza: **{explainer['best_incremental_auc_vs_confidence']:+.3f}**. "
            "Un valor cercano a cero o negativo indica que no hay un patron oculto estable explotable con las features actuales.\n\n",
            "### Permutation importance en el ultimo 20%\n\n",
            "| Feature | Caida AUC | Std |\n|---|---:|---:|\n",
        ]
    )
    for row in explainer["permutation_importance_last_20pct"][:15]:
        lines.append(f"| {row['feature']} | {row['roc_auc_drop_mean']:.4f} | {row['roc_auc_drop_std']:.4f} |\n")

    lines.extend(
        [
            "\n## Fallos de mayor confianza\n\n",
            "| Fecha | Partido | Evento | Formato | Favorito | Confianza | Ganador | HLTV |\n|---|---|---|---|---|---:|---|---|\n",
        ]
    )
    for row in failures[:25]:
        link = f"[link]({row['hltv_url']})" if row.get("hltv_url") else "-"
        lines.append(
            f"| {row['date']} | {row['team1']} vs {row['team2']} | {row['event']} | {row['format']} | "
            f"{row['favorite']} | {pct(row['confidence'])} | {row['winner']} | {link} |\n"
        )

    lines.extend(
        [
            "\n## Interpretacion y siguientes hipotesis\n\n",
            "1. **Margen pequeno, no una causa unica.** La confianza y varias medidas correlacionadas de fuerza explican la mayor parte de la concentracion de fallos. No deben sumarse como si fueran factores independientes.\n",
            "2. **Buscar mejora en los partidos parejos.** Stats de jugador, mapa y veto point-in-time son las candidatas mecanicamente plausibles para separar equipos cuando rating/Elo/forma casi empatan, pero aun no tienen muestra historica suficiente.\n",
            "3. **No hay evidencia de que BO1 sea peor aqui.** Debe leerse la tasa medida y su intervalo, no imponer la intuicion de mayor varianza.\n",
            "4. **Auditar drift y lado.** Meses y `favorite_side` se monitorizan como diagnostico. Solo deben generar cambios si el efecto persiste en ventanas futuras y tras controlar confianza.\n",
            "5. **No reajustar produccion con esta misma muestra.** Cualquier regla nueva debe validarse en un backtest temporal separado; usar estos fallos para crear y evaluar la regla en los mismos datos produciria overfitting.\n\n",
            "Archivos detallados: `walkforward_failure_cases.csv`, `walkforward_failure_numeric.csv`, `walkforward_failure_segments.csv`, `walkforward_failure_audit.json`.\n",
        ]
    )
    return "".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(db_path: Path, predictions_path: Path, raw_path: Path, output_dir: Path) -> dict[str, Any]:
    frame, production_features, metadata = build_audit_frame(db_path, predictions_path, raw_path)
    if frame.empty:
        raise RuntimeError("No hay predicciones walk-forward enlazables")
    associations = numeric_associations(frame, production_features)
    high_confidence_associations = numeric_associations(
        frame[frame["confidence"] >= 0.80].copy(),
        production_features,
    )
    segments = segment_analysis(frame)
    explainer = error_explainer(frame, production_features)
    failures = frame[frame["error"] == 1].sort_values(["confidence", "date"], ascending=[False, False])
    failures_records = failures.to_dict("records")
    total_failures = int(frame["error"].sum())
    below_60 = frame[frame["confidence"] < 0.60]
    at_least_80 = frame[frame["confidence"] >= 0.80]
    summary = {
        "n": int(len(frame)),
        "correct": int(len(frame) - total_failures),
        "failures": total_failures,
        "accuracy": float(1.0 - frame["error"].mean()),
        "failure_rate": float(frame["error"].mean()),
        "below_60_n": int(len(below_60)),
        "below_60_failures": int(below_60["error"].sum()),
        "below_60_share_of_failures": float(below_60["error"].sum() / total_failures),
        "at_least_80_n": int(len(at_least_80)),
        "at_least_80_failures": int(at_least_80["error"].sum()),
        "at_least_80_share_of_failures": float(at_least_80["error"].sum() / total_failures),
    }
    analysis = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": metadata,
        "summary": summary,
        "numeric_associations": associations,
        "high_confidence_numeric_associations": high_confidence_associations,
        "segments": segments,
        "error_explainer": explainer,
        "high_confidence_failures": failures_records[:100],
        "limitations": [
            "Associations are not causal effects.",
            "Team and event rows are repeated observations and not independent units.",
            "Extended-feature coverage is recent, sparse and non-random.",
            "Any new rule requires a separate chronological validation sample.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "walkforward_failure_cases.csv", failures_records)
    write_csv(output_dir / "walkforward_failure_numeric.csv", associations)
    write_csv(output_dir / "walkforward_failure_segments.csv", segments)
    plot_correlation(frame, associations, output_dir / "walkforward_failure_correlation.png")
    plot_confidence_errors(frame, output_dir / "walkforward_failure_by_confidence.png")
    safe_analysis = json_safe(analysis)
    (output_dir / "walkforward_failure_audit.json").write_text(
        json.dumps(safe_analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "WALKFORWARD_FAILURE_AUDIT.md").write_text(
        markdown_report(safe_analysis),
        encoding="utf-8",
    )
    return safe_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita fallos walk-forward del modelo productivo")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    analysis = run(Path(args.db), Path(args.predictions), Path(args.raw), Path(args.output_dir))
    print(
        json.dumps(
            {
                "model": analysis["metadata"]["model"],
                **analysis["summary"],
                "report": str(Path(args.output_dir) / "WALKFORWARD_FAILURE_AUDIT.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

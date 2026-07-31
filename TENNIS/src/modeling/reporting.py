"""Generación del informe Markdown reproducible de la fase 7."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


_MODEL_LABELS: Mapping[str, str] = {
    "logistic_raw": "Logística sin calibrar",
    "logistic_platt": "Logística + Platt",
    "lightgbm_raw": "LightGBM sin calibrar",
    "lightgbm_platt": "LightGBM + Platt",
    "ranking_probability": "Ranking probabilístico",
    "ranking_favorite": "Favorito por ranking",
    "market_devig": "Mercado de-vigado",
}


def _format_value(value: object, column: str) -> str:
    """Formatea una celda sin convertir ausencias en ceros."""

    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return "—"
    if column in {"n_total", "n_evaluated", "count"}:
        return f"{int(value):,}".replace(",", ".")
    if column in {"coverage", "positive_rate"}:
        return f"{100.0 * float(value):.2f} %".replace(".", ",")
    if column in {"accuracy", "auc"}:
        return f"{float(value):.4f}".replace(".", ",")
    if column in {"log_loss", "brier"}:
        return f"{float(value):.6f}".replace(".", ",")
    if isinstance(value, bool):
        return "sí" if value else "no"
    return str(value).replace("|", r"\|")


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    """Convierte un DataFrame pequeño en tabla CommonMark."""

    if frame.empty:
        return "_Sin filas evaluables._"
    header = "| " + " | ".join(labels.get(column, column) for column in columns)
    separator = "| " + " | ".join("---" for _ in columns)
    rows = [header + " |", separator + " |"]
    for _, row in frame.iterrows():
        values = [
            _format_value(row[column], column)
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _global_metrics(
    metrics: pd.DataFrame,
    *,
    gender: str,
    population: str,
) -> pd.DataFrame:
    """Selecciona métricas globales de una población y género."""

    selected = metrics.loc[
        (metrics["gender"] == gender)
        & (metrics["population"] == population)
        & (metrics["scope"] == "global")
    ].copy()
    selected["model"] = selected["model"].map(_MODEL_LABELS).fillna(
        selected["model"]
    )
    return selected


def _segment_metrics(
    metrics: pd.DataFrame,
    *,
    gender: str,
) -> pd.DataFrame:
    """Selecciona el LightGBM calibrado por nivel y superficie."""

    selected = metrics.loc[
        (metrics["gender"] == gender)
        & (metrics["population"] == "native")
        & (metrics["scope"] == "segment")
        & (metrics["model"] == "lightgbm_platt")
    ].copy()
    selected["surface"] = selected["surface"].fillna("Unknown")
    return selected.sort_values(
        ["tour_level", "surface"], kind="stable"
    )


def _calibration_comparison(
    metrics: pd.DataFrame,
    *,
    gender: str,
) -> pd.DataFrame:
    """Construye Brier/log-loss antes y después para ambos algoritmos."""

    global_rows = metrics.loc[
        (metrics["gender"] == gender)
        & (metrics["population"] == "native")
        & (metrics["scope"] == "global")
    ].set_index("model")
    rows: list[dict[str, object]] = []
    for algorithm, raw_name, calibrated_name in (
        ("Regresión logística", "logistic_raw", "logistic_platt"),
        ("LightGBM", "lightgbm_raw", "lightgbm_platt"),
    ):
        raw = global_rows.loc[raw_name]
        calibrated = global_rows.loc[calibrated_name]
        rows.append(
            {
                "algorithm": algorithm,
                "brier_before": raw["brier"],
                "brier_after": calibrated["brier"],
                "brier_delta": calibrated["brier"] - raw["brier"],
                "log_loss_before": raw["log_loss"],
                "log_loss_after": calibrated["log_loss"],
                "log_loss_delta": (
                    calibrated["log_loss"] - raw["log_loss"]
                ),
            }
        )
    return pd.DataFrame(rows)


def render_model_report(
    *,
    fingerprint: str,
    feature_fingerprint: str,
    source_commit: str,
    profile: str,
    evaluation_first_year: int,
    evaluation_last_year: int,
    gender_summaries: Sequence[Mapping[str, object]],
    metrics: pd.DataFrame,
    suspicious: pd.DataFrame,
    market_audits: pd.DataFrame,
    run_relative_path: str,
) -> str:
    """Renderiza el informe completo y honesto de modelado.

    Args:
        fingerprint: Identidad SHA-256 del run de modelos.
        feature_fingerprint: Identidad de los Parquet de fase 6.
        source_commit: Commit Sackmann fijado por la ingesta.
        profile: Perfil de features utilizado.
        evaluation_first_year: Primera temporada de test.
        evaluation_last_year: Última temporada de test.
        gender_summaries: Tamaños y fechas por género.
        metrics: Métricas globales, segmentadas y de soporte común.
        suspicious: Auditoría de segmentos con accuracy mayor del 85 %.
        market_audits: Cobertura observada del mercado.
        run_relative_path: Ruta relativa desde ``docs/`` al run publicado.

    Returns:
        Documento Markdown terminado en salto de línea.
    """

    summary_frame = pd.DataFrame(gender_summaries)
    lines = [
        "# Informe de modelos — fase 7",
        "",
        "## Resultado reproducible",
        "",
        "Se entrenó un LightGBM principal y una regresión logística de "
        "referencia para cada género. La evaluación es exclusivamente "
        "temporal y las probabilidades se calibran con Platt.",
        "",
        "```text",
        f"model_fingerprint: {fingerprint}",
        f"feature_fingerprint: {feature_fingerprint}",
        f"source_commit: {source_commit}",
        f"feature_profile: {profile}",
        f"test_seasons: {evaluation_first_year}-{evaluation_last_year}",
        "fold Y: train <= Y-2; calibración = Y-1; test = Y",
        "```",
        "",
        _markdown_table(
            summary_frame,
            (
                "gender",
                "training_rows",
                "training_max_date",
                "evaluation_rows",
                "folds",
            ),
            {
                "gender": "Género",
                "training_rows": "Filas entrenamiento final",
                "training_max_date": "Última fecha",
                "evaluation_rows": "Predicciones OOF",
                "folds": "Folds",
            },
        ),
        "",
        "Los hiperparámetros fueron fijados antes de observar los bloques de "
        "test. No se hizo selección ni early stopping con calibración/test.",
    ]

    for gender in ("M", "F"):
        label = "hombres" if gender == "M" else "mujeres"
        lines.extend(
            [
                "",
                f"## Resultados globales — {label}",
                "",
                _markdown_table(
                    _global_metrics(
                        metrics, gender=gender, population="native"
                    ),
                    (
                        "model",
                        "n_evaluated",
                        "coverage",
                        "accuracy",
                        "log_loss",
                        "brier",
                        "auc",
                    ),
                    {
                        "test_season": "Temporada",
                        "model": "Método",
                        "n_evaluated": "N",
                        "coverage": "Cobertura",
                        "accuracy": "Accuracy",
                        "log_loss": "Log-loss",
                        "brier": "Brier",
                        "auc": "AUC",
                    },
                ),
                "",
                "### Efecto de la calibración",
                "",
                _markdown_table(
                    _calibration_comparison(metrics, gender=gender),
                    (
                        "algorithm",
                        "brier_before",
                        "brier_after",
                        "brier_delta",
                        "log_loss_before",
                        "log_loss_after",
                        "log_loss_delta",
                    ),
                    {
                        "algorithm": "Modelo",
                        "brier_before": "Brier antes",
                        "brier_after": "Brier después",
                        "brier_delta": "Δ Brier",
                        "log_loss_before": "Log-loss antes",
                        "log_loss_after": "Log-loss después",
                        "log_loss_delta": "Δ log-loss",
                    },
                ),
                "",
                f"![Curva de calibración {gender}]"
                f"(assets/model_calibration_{gender}.png)",
                "",
                "Un delta negativo significa mejora. La curva usa diez bins "
                "por cuantiles y acumula únicamente predicciones OOF.",
                "",
                "### Comparación justa con ranking",
                "",
                _markdown_table(
                    _global_metrics(
                        metrics,
                        gender=gender,
                        population="ranking_common_support",
                    ),
                    (
                        "model",
                        "n_evaluated",
                        "accuracy",
                        "log_loss",
                        "brier",
                        "auc",
                    ),
                    {
                        "model": "Método",
                        "n_evaluated": "N común",
                        "accuracy": "Accuracy",
                        "log_loss": "Log-loss",
                        "brier": "Brier",
                        "auc": "AUC",
                    },
                ),
                "",
                "Esta tabla contiene exactamente los partidos donde el "
                "favorito por ranking es observable; no penaliza rankings "
                "ausentes.",
                "",
                "### LightGBM calibrado por nivel × superficie",
                "",
                _markdown_table(
                    _segment_metrics(metrics, gender=gender),
                    (
                        "tour_level",
                        "surface",
                        "n_evaluated",
                        "accuracy",
                        "log_loss",
                        "brier",
                        "auc",
                    ),
                    {
                        "tour_level": "Nivel",
                        "surface": "Superficie",
                        "n_evaluated": "N",
                        "accuracy": "Accuracy",
                        "log_loss": "Log-loss",
                        "brier": "Brier",
                        "auc": "AUC",
                    },
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Comparación con el mercado",
            "",
            _markdown_table(
                market_audits,
                ("gender", "total_rows", "market_rows", "coverage", "status"),
                {
                    "gender": "Género",
                    "total_rows": "Filas OOF",
                    "market_rows": "Con mercado",
                    "coverage": "Cobertura",
                    "status": "Estado",
                },
            ),
            "",
            "Sackmann no contiene cuotas históricas y la fase 6 declara "
            "`historical_odds_available=false`. Por ello, el favorito por "
            "cuota y el valor añadido frente al mercado son **NO EVALUABLES** "
            "en este run. No se imputó ninguna cuota. El evaluador ya genera "
            "la comparación sobre soporte común cuando existan observaciones "
            "de-vigadas capturadas antes del partido.",
            "",
            "## Auditoría de accuracy superior al 85 %",
            "",
        ]
    )
    if suspicious.empty:
        lines.append(
            "No apareció ningún segmento con accuracy estrictamente superior "
            "al 85 %."
        )
    else:
        display = suspicious.copy()
        if "test_season" not in display.columns:
            display["test_season"] = "ALL"
        else:
            display["test_season"] = display["test_season"].fillna("ALL")
        display["model"] = display["model"].map(_MODEL_LABELS).fillna(
            display["model"]
        )
        lines.extend(
            [
                _markdown_table(
                    display,
                    (
                        "gender",
                        "test_season",
                        "model",
                        "tour_level",
                        "surface",
                        "n_evaluated",
                        "accuracy",
                        "small_sample",
                        "audit_conclusion",
                    ),
                    {
                        "gender": "Género",
                        "model": "Método",
                        "tour_level": "Nivel",
                        "surface": "Superficie",
                        "n_evaluated": "N",
                        "accuracy": "Accuracy",
                        "small_sample": "N<200",
                        "audit_conclusion": "Conclusión",
                    },
                ),
                "",
                "Todos los casos se conservaron. La auditoría comprobó IDs "
                "OOF únicos, precedencia temporal y ausencia de objetivo, "
                "identidades o postdicciones entre las features.",
            ]
        )

    lines.extend(
        [
            "",
            "## Limitaciones",
            "",
            "- No existen cuotas históricas causales: todavía no puede "
            "demostrarse ventaja sobre el mercado ni entrenarse el perfil "
            "`market_enhanced`.",
            "- `tourney_date` suele ser el inicio del torneo. Congelar el "
            "bloque completo evita fugas entre rondas, pero pierde señal.",
            "- El fold de test 2021 se calibra con la temporada atípica 2020. "
            "Se conserva porque mantiene el protocolo y tiene miles de casos.",
            "- La cobertura de rankings, especialmente femenina, varía con el "
            "tiempo. Las comparaciones usan soporte común.",
            "- Hay colisiones históricas de IDs/fechas de nacimiento, valores "
            "extremos de descanso y rankings antiguos. Los nulos se señalan; "
            "no se corrigen biografías ni resultados de forma especulativa.",
            "- 2026 es parcial y no forma parte del backtest principal. Sí "
            "entra al estimador final, porque ya es pasado para futuras "
            "predicciones.",
            "- Los hiperparámetros son conservadores y fijos. La fase no "
            "presenta un barrido de hiperparámetros como si fuera evidencia "
            "independiente.",
            "",
            "## Artefactos detallados",
            "",
            f"Las métricas completas, predicciones OOF, folds, calibradores "
            f"y modelos están en `{run_relative_path}`. `metrics.csv` incluye "
            "todos los modelos y segmentos, además de las poblaciones de "
            "soporte común.",
            "",
        ]
    )
    return "\n".join(lines)


def write_model_report(text: str, path: Path) -> None:
    """Escribe el informe UTF-8 en una ruta proporcionada por el caller."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("El informe no puede estar vacío.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

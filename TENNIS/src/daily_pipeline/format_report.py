"""Coverage of causal match format inputs and their actual confidence impact."""

from __future__ import annotations

import pandas as pd


def match_format_report(frame: pd.DataFrame) -> dict[str, object]:
    """Report denominators explicitly; unpredicted rows are not model inputs."""

    predicted = frame.loc[frame["model_probability_a"].notna()]
    report: dict[str, object] = {"agenda_rows": len(frame), "predicted_rows": len(predicted)}
    for column in ("best_of", "round"):
        populated = int(predicted[column].notna().sum())
        report[column] = {
            "populated": populated,
            "missing": len(predicted) - populated,
            "populated_pct": 100.0 * populated / len(predicted) if len(predicted) else None,
            "missing_flag_rows": int(
                predicted["confidence_flags"].str.contains(f"{column}_missing", na=False).sum()
            ),
            "reasons": predicted[f"{column}_reason"].value_counts().to_dict(),
        }
    report["confidence_transitions"] = {
        f"{before}->{after}": int(count)
        for (before, after), count in predicted.groupby(
            ["confidence_without_match_format", "confidence"], dropna=False
        )
        .size()
        .items()
    }
    return report


def match_format_summary(frame: pd.DataFrame) -> list[str]:
    """Build concise launcher lines with honest N/A when nothing was predicted."""

    predicted = frame.loc[frame["model_probability_a"].notna()]
    lines = []
    for column in ("best_of", "round"):
        n = int(predicted[column].notna().sum())
        pct = f"{100.0 * n / len(predicted):.1f}%" if len(predicted) else "N/A"
        lines.append(
            f"[TENNIS contexto] {column}: {n}/{len(predicted)} predicciones ({pct}); ausentes con flag: {len(predicted) - n}."
        )
    improved = sum(
        {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(row["confidence"]), -1)
        > {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(row["confidence_without_match_format"]), -1)
        for _, row in predicted.iterrows()
    )
    lines.append(
        f"[TENNIS contexto] Predicciones que suben de nivel por formato/ronda: {improved}."
    )
    return lines

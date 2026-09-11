"""Measure calibration segments from closed temporal walk-forward predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from cs2model.config import DEFAULT_CONFIG_PATH, load_config
from cs2model.segment_calibration import segment_calibration_suite


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "MODEL" / "results" / "predictions_walkforward.csv"
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_JSON = ROOT / "MODEL" / "results" / "segment_calibration_backtest.json"
DEFAULT_MARKDOWN = ROOT / "MODEL" / "results" / "SEGMENT_CALIBRATION_BACKTEST.md"


def _match_context(db_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    rows = connection.execute(
        """
        SELECT m.hltv_match_id, m.stage, m.environment, m.best_of,
               mf1.elo - mf2.elo AS elo_diff
        FROM matches AS m
        JOIN match_features AS mf1
          ON mf1.match_id=m.match_id AND mf1.team_id=m.team1_id
        JOIN match_features AS mf2
          ON mf2.match_id=m.match_id AND mf2.team_id=m.team2_id
        """
    ).fetchall()
    connection.close()
    return {
        str(match_id): {
            "stage": stage,
            "environment": environment,
            "format": f"bo{int(best_of)}" if best_of else None,
            "elo_diff": elo_diff,
        }
        for match_id, stage, environment, best_of, elo_diff in rows
    }


def load_walk_forward_rows(csv_path: Path, db_path: Path, model: str) -> list[dict[str, Any]]:
    context = _match_context(db_path)
    selected: list[dict[str, Any]] = []
    models: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_model = str(row.get("model") or "")
            models.add(row_model)
            if row_model != model:
                continue
            match_context = context.get(str(row.get("match_id") or ""), {})
            csv_elo = row.get("elo_diff")
            selected.append(
                {
                    "actual": int(row["actual"]),
                    "prob_team1": float(row["prob_team1"]),
                    "elo_diff": float(csv_elo) if csv_elo not in {None, ""} else match_context.get("elo_diff"),
                    "stage": row.get("stage") or match_context.get("stage"),
                    "environment": row.get("environment") or match_context.get("environment"),
                    "format": row.get("format") or match_context.get("format"),
                    "event_tier": row.get("event_tier"),
                    "date": row.get("date"),
                    "match_id": row.get("match_id"),
                }
            )
    if not selected:
        raise ValueError(f"model {model!r} not found; available={sorted(models)}")
    return selected


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Calibración por segmento — backtest temporal\n",
        f"Modelo/política evaluada: `{report['model']}`. Predicciones OOS: {report['n']}.\n",
        "| Dimensión | Segmento | N | Prob. media | Tasa real | Gap | ECE | Estado |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    calibration = report["segment_calibration"]
    for dimension in ("by_elo_gap", "by_stage", "by_environment"):
        for segment, values in calibration.get(dimension, {}).items():
            if values.get("mean_predicted") is None:
                lines.append(
                    f"| {dimension.removeprefix('by_')} | {segment} | {values.get('n', 0)} "
                    f"| - | - | - | - | {values.get('status', 'unknown')} |"
                )
                continue
            lines.append(
                f"| {dimension.removeprefix('by_')} | {segment} | {values['n']} "
                f"| {values['mean_predicted']:.4f} | {values['observed_rate']:.4f} "
                f"| {values['calibration_gap']:+.4f} | {values['ece_10']:.4f} "
                f"| {values['status']} |"
            )
    lines.extend(
        [
            "",
            "> `insufficient_sample` no se interpreta. `deviation_detected` es una señal descriptiva, "
            "no una orden de cambiar el modelo: el siguiente escalón debe ganar el hold-out temporal "
            "y la puerta champion/challenger.",
            "",
        ]
    )
    return "\n".join(lines)


def measure(csv_path: Path, db_path: Path, config_path: Path, model: str) -> dict[str, Any]:
    rows = load_walk_forward_rows(csv_path, db_path, model)
    policy = load_config(config_path).segment_calibration
    return {
        "source": "closed_temporal_walk_forward_predictions",
        "model": model,
        "n": len(rows),
        "date_min": min(str(row.get("date") or "") for row in rows),
        "date_max": max(str(row.get("date") or "") for row in rows),
        "predictions_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "segment_calibration": segment_calibration_suite(
            rows,
            min_n=policy.min_backtest_segment_rows,
            n_bins=policy.reliability_bins,
            min_absolute_gap=policy.min_absolute_gap,
            min_ece=policy.min_ece,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_CSV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", default="nested_model_policy")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-markdown", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()
    report = measure(Path(args.predictions), Path(args.db), Path(args.config), args.model)
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ablacion temporal de snapshots de jugador frente al nucleo estadistico.

Compara dos modelos logit regulares sobre exactamente los mismos partidos con
snapshot point-in-time apto. En cada bloque temporal ambos se entrenan con las
misma filas previas; el segundo anade solamente PLAYER_FEATURE_COLUMNS.

No promociona artefactos ni cambia produccion: es evidencia exploratoria hasta
que la familia alcance su umbral de activacion automatico.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from cs2model import dataio
from cs2model.features import FEATURE_COLUMNS, PLAYER_FEATURE_COLUMNS, build_training_frame
from train import _fit_base, _matrix, _proba


DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_RESULTS = ROOT / "MODEL" / "results"


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((probability >= 0.5) == y)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def mcnemar_exact_pvalue(base_probability: np.ndarray, player_probability: np.ndarray, y: np.ndarray) -> dict[str, int | float]:
    base_correct = (base_probability >= 0.5) == y
    player_correct = (player_probability >= 0.5) == y
    player_only = int(np.sum(player_correct & ~base_correct))
    base_only = int(np.sum(base_correct & ~player_correct))
    discordant = player_only + base_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(player_only, base_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "player_only_correct": player_only,
        "base_only_correct": base_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_bootstrap(
    y: np.ndarray,
    base_probability: np.ndarray,
    player_probability: np.ndarray,
    draws: int = 5000,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)
    n = len(y)
    deltas: dict[str, list[float]] = {"accuracy": [], "log_loss": [], "brier": []}
    for _ in range(draws):
        sample = rng.integers(0, n, size=n)
        base = metrics(y[sample], base_probability[sample])
        player = metrics(y[sample], player_probability[sample])
        for key in deltas:
            deltas[key].append(player[key] - base[key])
    return {
        key: {
            "mean_delta": float(np.mean(values)),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in deltas.items()
    }


def run_ablation(db_path: Path, warmup: int, block_size: int, bootstrap_draws: int) -> dict[str, Any]:
    rows = dataio.load_training_rows_from_db(db_path, cs2_only=True)
    x_dicts, y_values, meta, _state = build_training_frame(rows)
    eligible = [
        idx for idx, values in enumerate(x_dicts)
        if (values.get("player_snapshot_available") or 0.0) >= 0.5
    ]
    if len(eligible) <= warmup:
        raise ValueError(f"Solo hay {len(eligible)} partidos aptos; se necesitan mas de {warmup}.")

    base_columns = list(FEATURE_COLUMNS)
    player_columns = base_columns + list(PLAYER_FEATURE_COLUMNS)
    base_matrix = _matrix(x_dicts, base_columns)
    player_matrix = _matrix(x_dicts, player_columns)
    y_all = np.asarray(y_values, dtype=int)

    base_probabilities: list[float] = []
    player_probabilities: list[float] = []
    actual: list[int] = []
    cases: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for start in range(warmup, len(eligible), block_size):
        train_indices = np.asarray(eligible[:start], dtype=int)
        test_indices = np.asarray(eligible[start : start + block_size], dtype=int)
        if not len(test_indices):
            continue
        y_train = y_all[train_indices]
        if len(np.unique(y_train)) < 2:
            continue
        base_model = _fit_base("logistic", base_matrix[train_indices], y_train, base_columns)
        player_model = _fit_base("logistic", player_matrix[train_indices], y_train, player_columns)
        base_fold = _proba(base_model, base_matrix[test_indices])
        player_fold = _proba(player_model, player_matrix[test_indices])
        base_probabilities.extend(base_fold.tolist())
        player_probabilities.extend(player_fold.tolist())
        actual.extend(y_all[test_indices].tolist())
        folds.append({"train_rows": int(len(train_indices)), "test_rows": int(len(test_indices))})
        for idx, base_p, player_p in zip(test_indices, base_fold, player_fold, strict=True):
            cases.append(
                {
                    "match_id": meta[idx]["id"],
                    "date": str(meta[idx].get("date") or ""),
                    "team1": meta[idx].get("team1"),
                    "team2": meta[idx].get("team2"),
                    "actual_team1_win": int(y_all[idx]),
                    "base_probability_team1": float(base_p),
                    "player_probability_team1": float(player_p),
                }
            )

    y = np.asarray(actual, dtype=int)
    base_probability = np.asarray(base_probabilities, dtype=float)
    player_probability = np.asarray(player_probabilities, dtype=float)
    if not len(y):
        raise ValueError("No se generaron predicciones temporales validas.")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": {
            "design": "expanding chronological blocks; identical eligible train/test rows in both arms",
            "estimator": "regularized_logistic_with_team_order_augmentation",
            "base_columns": len(base_columns),
            "player_columns_added": len(PLAYER_FEATURE_COLUMNS),
            "warmup_eligible_matches": warmup,
            "test_block_size": block_size,
        },
        "sample": {
            "closed_rows_total": len(rows),
            "eligible_point_in_time_player_rows": len(eligible),
            "evaluated_same_matches": len(y),
            "first_evaluated_date": cases[0]["date"],
            "last_evaluated_date": cases[-1]["date"],
            "folds": folds,
        },
        "base_core": metrics(y, base_probability),
        "with_player_snapshots": metrics(y, player_probability),
        "delta_player_minus_base": {
            key: metrics(y, player_probability)[key] - metrics(y, base_probability)[key]
            for key in ("accuracy", "log_loss", "brier")
        },
        "paired_bootstrap": paired_bootstrap(y, base_probability, player_probability, draws=bootstrap_draws),
        "mcnemar_accuracy": mcnemar_exact_pvalue(base_probability, player_probability, y),
        "cases": cases,
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    base = result["base_core"]
    player = result["with_player_snapshots"]
    delta = result["delta_player_minus_base"]
    sample = result["sample"]
    bootstrap = result["paired_bootstrap"]
    mcnemar = result["mcnemar_accuracy"]
    lines = [
        "# Player Snapshot Ablation\n\n",
        f"Generado: {result['generated_at_utc']}\n\n",
        "## Diseno\n\n",
        "Comparacion walk-forward con bloques cronologicos expansivos. Ambos brazos usan exactamente los mismos partidos aptos, "
        "cortes y filas de entrenamiento; solo cambia la inclusion de las columnas de snapshots de jugador.\n\n",
        f"- Cerrados totales: {sample['closed_rows_total']}\n",
        f"- Aptos point-in-time: {sample['eligible_point_in_time_player_rows']}\n",
        f"- Evaluados en comun fuera de muestra: {sample['evaluated_same_matches']} ({sample['first_evaluated_date']} a {sample['last_evaluated_date']})\n",
        f"- Logistic regularizada; {result['method']['base_columns']} columnas base + {result['method']['player_columns_added']} de jugadores.\n\n",
        "## Resultado\n\n",
        "| Brazo | Accuracy | Log loss | Brier | n |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| Base sin jugadores | {base['accuracy']:.3%} | {base['log_loss']:.4f} | {base['brier']:.4f} | {base['n']} |\n",
        f"| Base + snapshots | {player['accuracy']:.3%} | {player['log_loss']:.4f} | {player['brier']:.4f} | {player['n']} |\n",
        f"| Delta jugador - base | {delta['accuracy']:+.3%} | {delta['log_loss']:+.4f} | {delta['brier']:+.4f} | |\n\n",
        "## Incertidumbre\n\n",
        f"- Bootstrap pareado 95% del delta de accuracy: [{bootstrap['accuracy']['ci_95_low']:+.3%}, {bootstrap['accuracy']['ci_95_high']:+.3%}].\n",
        f"- Bootstrap pareado 95% del delta de log loss: [{bootstrap['log_loss']['ci_95_low']:+.4f}, {bootstrap['log_loss']['ci_95_high']:+.4f}].\n",
        f"- McNemar exacto: jugador acierta solo={mcnemar['player_only_correct']}, base acierta solo={mcnemar['base_only_correct']}, "
        f"p={mcnemar['two_sided_exact_p_value']:.4f}.\n\n",
        "No se cambia produccion con este resultado: la muestra es exploratoria y el umbral operativo sigue siendo 200 partidos cerrados con snapshots point-in-time aptos.\n",
    ]
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablacion temporal base vs stats de jugador.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    print(f"[1/3] Cargando BBDD: {args.db}", flush=True)
    result = run_ablation(args.db, args.warmup, args.block_size, args.bootstrap_draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "player_snapshot_ablation.json"
    markdown_path = args.output_dir / "PLAYER_SNAPSHOT_ABLATION.md"
    print("[2/3] Calculando metricas pareadas e incertidumbre", flush=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print("[3/3] Resultado", flush=True)
    print(render_markdown(result))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI del harness de evaluacion honesto (walk-forward ANIDADO).

Produce, por cada run, un reporte de metricas reproducible:
  - MODEL/results/eval_metrics.json   (todas las metricas + baselines + betting)
  - MODEL/results/eval_calibration.json + reliability_curve.png (si hay matplotlib)
  - MODEL/results/EVAL_REPORT.md
  - MODEL/results/experiments.jsonl   (registro append-only: manifest por run)

Uso:
  python MODEL/evaluate.py --synthetic --window both          # smoke reproducible
  python MODEL/evaluate.py --window expanding                 # datos reales (cs2.db)
  python MODEL/evaluate.py --window both --outer-step 4 --gap 1

El nucleo (cs2model.evaluation) es numpy puro. El fitter de produccion
(LightGBM/super-learner) se puede inyectar en el futuro sin cambiar el protocolo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cs2model.evaluation import EvalConfig, Family, nested_walk_forward

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "MODEL" / "results"
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_CONFIG = ROOT / "MODEL" / "config.yaml"
REGISTRY = OUTPUT_DIR / "experiments.jsonl"


# ===========================================================================
# Configuracion
# ===========================================================================
def load_eval_config(config_path: Path, overrides: dict[str, Any]) -> tuple[EvalConfig, dict[str, int], int]:
    raw = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ev = (raw.get("evaluation") or {})
    seed = int(raw.get("random_seed", 42))
    thresholds = raw.get("feature_thresholds") or {}
    cfg = EvalConfig(
        warmup_periods=int(ev.get("warmup_periods", 10)),
        gap_periods=int(overrides.get("gap", ev.get("gap_periods", 1))),
        outer_step=int(overrides.get("outer_step", ev.get("outer_step", 4))),
        inner_step=int(ev.get("inner_step", 4)),
        inner_warmup=int(ev.get("inner_warmup", 8)),
        window=str(ev.get("window", "expanding")),
        train_width=int(ev.get("train_width", 52)),
        l2_grid=tuple(ev.get("l2_grid", (0.25, 1.0, 4.0))),
        calibration_methods=tuple(ev.get("calibration_methods", ("identity", "platt", "isotonic"))),
        recency_half_life=float(ev.get("recency_half_life", 0.0)),
        seed=seed,
    )
    return cfg, {str(k): int(v) for k, v in thresholds.items()}, seed


# ===========================================================================
# Datos reales (cs2.db) — adaptador; se ejecuta en la maquina del usuario
# ===========================================================================
def load_real_rows(db_path: Path, thresholds: dict[str, int]) -> tuple[list[dict[str, Any]], list[str], list[Family]]:
    """Construye filas para el harness desde cs2.db (features point-in-time).

    Import perezoso del stack pesado (dataio/features) para que el modulo se
    pueda importar sin numpy-heavy/sklearn en entornos ligeros.
    """
    from cs2model import dataio
    from cs2model.features import build_training_frame, BASE_FEATURE_COLUMNS, _period_index
    from cs2model import features as F

    raw_rows = dataio.load_training_rows_from_db(db_path)
    X_dicts, y, meta, _state = build_training_frame(raw_rows)
    odds_by_id = {r["id"]: r for r in raw_rows}

    rows: list[dict[str, Any]] = []
    for i, feats in enumerate(X_dicts):
        m = meta[i]
        odds = odds_by_id.get(m.get("id"), {})
        row = dict(feats)
        row["label"] = int(y[i])
        # ``build_training_frame`` exposes the printable date in ``meta`` but
        # keeps the parsed datetime on the normalized source row. Using the
        # absent ``meta["date_obj"]`` collapses every match into period zero.
        row["period"] = int(_period_index(odds.get("date_obj")))
        row["opening_prob_t1"] = odds.get("opening_odds_t1")
        row["opening_odds_t1"] = odds.get("opening_odds_decimal_t1")
        row["opening_odds_t2"] = odds.get("opening_odds_decimal_t2")
        row["closing_prob_t1"] = odds.get("closing_odds_t1")
        rows.append(row)

    families = _families_from_config(F, thresholds)
    return rows, list(BASE_FEATURE_COLUMNS), families


def _families_from_config(F: Any, thresholds: dict[str, int]) -> list[Family]:
    """Mapea las familias auto-activables a la definicion del harness."""
    table = [
        ("map_box_scores", F.MAP_ASSET_FEATURE_COLUMNS, "asset_available", 200),
        ("event_history", F.EVENT_HISTORY_FEATURE_COLUMNS, "event_history_available", 200),
        ("analytics", F.ANALYTICS_FEATURE_COLUMNS, "analytics_available", 120),
        (
            "analytics_extended",
            F.ANALYTICS_EXTENDED_FEATURE_COLUMNS,
            "analytics_extended_available",
            200,
        ),
        (
            "announced_lineups",
            F.ANNOUNCED_LINEUP_FEATURE_COLUMNS,
            "announced_lineup_available",
            200,
        ),
        ("event_metadata", F.EVENT_METADATA_FEATURE_COLUMNS, "event_metadata_available", 300),
        ("player_snapshots", F.PLAYER_FEATURE_COLUMNS, "player_snapshot_available", 200),
        ("rankings", F.RANKING_FEATURE_COLUMNS, "ranking_available", 200),
        ("roster", F.ROSTER_FEATURE_COLUMNS, "roster_available", 200),
        ("mov_rating", F.MOV_FEATURE_COLUMNS, "mov_available", 800),
        ("team_trueskill", F.TRUESKILL_FEATURE_COLUMNS, "trueskill_available", 800),
        ("player_rating", F.PLAYER_RATING_FEATURE_COLUMNS, "player_skill_available", 200),
        ("strength_of_schedule", F.SOS_FEATURE_COLUMNS, "sos_available", 800),
        (
            "bayesian_bradley_terry",
            F.BAYES_BT_FEATURE_COLUMNS,
            "bayesian_bt_available",
            800,
        ),
        ("kalman_state_space", F.KALMAN_FEATURE_COLUMNS, "kalman_available", 800),
        (
            "bo3_map_compositional",
            F.BO3_COMPOSITIONAL_FEATURE_COLUMNS,
            "bo3_compositional_available",
            500,
        ),
        ("regime", F.REGIME_FEATURE_COLUMNS, "regime_available", 200),
        ("context", F.CONTEXT_FEATURE_COLUMNS, "context_available", 200),
    ]
    out = []
    for name, cols, avail, default in table:
        out.append(Family(name, tuple(cols), avail, int(thresholds.get(name, default))))
    return out


# ===========================================================================
# Datos sinteticos (smoke reproducible, sin cs2.db)
# ===========================================================================
def build_synthetic_rows(seed: int, periods: int = 80, per_week: int = 10) -> tuple[list[dict[str, Any]], list[str], list[Family]]:
    rng = random.Random(seed)
    T = 18
    strength = {t: rng.gauss(0, 1) for t in range(T)}
    rows: list[dict[str, Any]] = []
    for p in range(periods):
        for _ in range(per_week):
            a, b = rng.sample(range(T), 2)
            diff = strength[a] - strength[b]
            prob = 1 / (1 + math.exp(-diff))
            label = 1 if rng.random() < prob else 0
            mkt = min(max(prob + rng.gauss(0, 0.05), 0.02), 0.98)
            close = min(max(prob + rng.gauss(0, 0.03), 0.02), 0.98)
            avail = 1.0 if p >= periods // 2 else 0.0
            rows.append({
                "label": label, "period": p,
                "elo_prob_centered": prob - 0.5 + rng.gauss(0, 0.02),
                "noise1": rng.gauss(0, 1),
                "extra_feat": (diff * 0.5 + rng.gauss(0, 0.3)) if avail else float("nan"),
                "extra_available": avail,
                "opening_prob_t1": mkt, "opening_odds_t1": 1 / mkt, "opening_odds_t2": 1 / (1 - mkt),
                "closing_prob_t1": close,
            })
    families = [Family("extra", ("extra_feat",), "extra_available", threshold=200)]
    return rows, ["elo_prob_centered", "noise1"], families


# ===========================================================================
# Manifest / registro de experimentos
# ===========================================================================
def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps({k: r.get(k) for k in sorted(r)}, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def run_manifest(rows: list[dict[str, Any]], cfg: EvalConfig, args: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for window, res in results.items():
        primary_name = (
            "model"
            if "model" in res.get("models", {})
            else res.get("best_combo")
        )
        mm = res.get("models", {}).get(primary_name, {})
        summary[window] = {
            "primary_model": primary_name,
            "n_eval": res.get("n_eval"),
            "log_loss": mm.get("log_loss"),
            "brier": mm.get("brier"),
            "ece_10": mm.get("ece_10"),
            "roc_auc": mm.get("roc_auc"),
            "accuracy": mm.get("accuracy"),
        }
    return {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "dataset_sha256": _dataset_hash(rows),
        "dataset_rows": len(rows),
        "seed": cfg.seed,
        "config": {
            "warmup_periods": cfg.warmup_periods, "gap_periods": cfg.gap_periods,
            "outer_step": cfg.outer_step, "inner_step": cfg.inner_step,
            "l2_grid": list(cfg.l2_grid), "calibration_methods": list(cfg.calibration_methods),
            "recency_half_life": cfg.recency_half_life,
        },
        "arguments": args,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "metrics": summary,
    }


# ===========================================================================
# Reporte + graficos
# ===========================================================================
def write_reliability_plot(calibration: dict[str, list[dict[str, float]]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfecta")
    for name, bins in calibration.items():
        if not bins:
            continue
        xs = [b["avg_pred"] for b in bins]
        ys = [b["observed"] for b in bins]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_title("Curva de fiabilidad (calibracion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(results: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    lines: list[str] = ["# EVAL_REPORT — walk-forward anidado (medicion honesta)\n"]
    lines.append(f"Generado: {manifest['ts_utc']} · commit: {manifest.get('git_commit')}\n")
    lines.append(f"Dataset: {manifest['dataset_rows']} filas · sha256 {manifest['dataset_sha256'][:12]} · seed {manifest['seed']}\n")
    lines.append("\n> log loss es la metrica PRIMARIA. accuracy es secundaria (el favorito ya acierta ~63-65%).\n")
    for window, res in results.items():
        lines.append(f"\n## Ventana: {window}  (n_eval={res['n_eval']}, folds={res['n_folds']})\n\n")
        lines.append("| Modelo | log_loss | Brier | ECE | ROC-AUC | Accuracy |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        order = ["model", "elo", "market"]
        labels = {"model": "**Modelo (nested)**", "elo": "Baseline Elo", "market": "Baseline mercado (Model B)"}
        for name in order:
            m = res["models"].get(name, {})
            lines.append(f"| {labels[name]} | {_fmt(m.get('log_loss'))} | {_fmt(m.get('brier'))} | "
                         f"{_fmt(m.get('ece_10'))} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('accuracy'))} |\n")
        b = res["betting"]
        lines.append(f"\n**Apuestas vs mercado** (subconjunto con odds={b['subset_with_odds']}): "
                     f"apuestas={b['bets']}, hit={_fmt(b['hit_rate'],3)}, ROI={_fmt(b['roi_on_stake'],4)}, "
                     f"CLV medio={_fmt(b['clv_mean'],4)} (n={b['clv_n']}).\n")
        lines.append(f"\n**Accuracy del favorito recomputada (el '65%' honesto): {_fmt(res['recomputed_favorite_accuracy'],4)}**\n")
        cal = set(d["calibration"] for d in res["decisions"])
        fams = sorted(set(f for d in res["decisions"] for f in d["families"]))
        lines.append(f"\nDecisiones del bucle interno — calibradores usados: {sorted(cal)}; "
                     f"familias activadas: {fams or 'ninguna'}.\n")
    lines.append("\n---\n*Bucle externo insesgado; todas las decisiones en el bucle interno con datos "
                 "estrictamente pasados y gap temporal. Modelo de produccion se reentrena aparte.*\n")
    path.write_text("".join(lines), encoding="utf-8")


# ===========================================================================
# main
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Harness de evaluacion honesto (walk-forward anidado).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--window", choices=["expanding", "sliding", "both"], default="both")
    p.add_argument("--gap", type=int, default=None)
    p.add_argument("--outer-step", type=int, default=None)
    p.add_argument("--synthetic", action="store_true", help="Datos sinteticos deterministas (smoke, sin cs2.db).")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.gap is not None:
        overrides["gap"] = args.gap
    if args.outer_step is not None:
        overrides["outer_step"] = args.outer_step
    cfg, thresholds, seed = load_eval_config(args.config, overrides)

    if args.synthetic:
        rows, base_cols, families = build_synthetic_rows(seed)
    else:
        if not args.db.exists():
            print(f"[evaluate] no existe la BBDD {args.db}. Usa --synthetic para un smoke.")
            return 2
        rows, base_cols, families = load_real_rows(args.db, thresholds)

    windows = ["expanding", "sliding"] if args.window == "both" else [args.window]
    results: dict[str, Any] = {}
    for window in windows:
        import dataclasses
        wcfg = dataclasses.replace(cfg, window=window)
        print(f"[evaluate] nested walk-forward ({window})...", flush=True)
        results[window] = nested_walk_forward(rows, base_cols, families, wcfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_manifest(rows, cfg, vars(args) | {"windows": windows}, results)

    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps({"manifest": manifest, "results": results}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # calibracion + grafico (de la ventana expansiva si existe, si no la primera)
    cal_window = "expanding" if "expanding" in results else windows[0]
    calibration = results[cal_window].get("calibration", {})
    (args.output_dir / "eval_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    plotted = write_reliability_plot(calibration, args.output_dir / "reliability_curve.png")
    write_report(results, manifest, args.output_dir / "EVAL_REPORT.md")

    with (args.output_dir / "experiments.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, default=str) + "\n")

    for window, res in results.items():
        mm = res["models"]["model"]
        print(f"[evaluate] {window}: n_eval={res['n_eval']} log_loss={_fmt(mm.get('log_loss'))} "
              f"auc={_fmt(mm.get('roc_auc'))} fav_acc={_fmt(res['recomputed_favorite_accuracy'])}")
    print(f"[evaluate] artefactos en {args.output_dir} (reliability png: {'si' if plotted else 'omitido, sin matplotlib'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

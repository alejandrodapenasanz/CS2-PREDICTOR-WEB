"""CLI: compara el ZOO de modelos x los 3 block-wise bajo el harness anidado.

Produce el REPORTE comparativo (todos los modelos x todas las metricas WF), elige
el Model A de produccion por log loss, y registra el run.

  python MODEL/compare_models.py --synthetic --n-trials 8           # smoke reproducible
  python MODEL/compare_models.py --window both --n-trials 40        # real (cs2.db)
  python MODEL/compare_models.py --models logistic_en,lightgbm,hist_gb --assemblies indicators,profiles,two_stage

Artefactos en MODEL/results/: MODEL_COMPARISON.md, model_comparison.json y una
linea en experiments.jsonl (manifest por run).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

from cs2model.evaluation import EvalConfig, _blocks
from cs2model.blockwise import run_comparison, ComparisonConfig
from cs2model.model_zoo import available_models

import evaluate as ev  # reutiliza carga de datos, config, manifest, _fmt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "MODEL" / "results"


def _expanding_sliding_equivalent(rows: list[dict[str, Any]], cfg: EvalConfig) -> bool:
    periods = sorted({int(row["period"]) for row in rows})
    common = {
        "warmup": cfg.warmup_periods,
        "step": cfg.outer_step,
        "gap": cfg.gap_periods,
        "train_width": cfg.train_width,
    }
    expanding = _blocks(periods, window="expanding", **common)
    sliding = _blocks(periods, window="sliding", **common)
    return bool(expanding) and expanding == sliding


def _table(res: dict[str, Any]) -> list[str]:
    lines = ["| combo | N | log_loss | Brier | ECE | ROC-AUC | Accuracy |\n",
             "|---|---:|---:|---:|---:|---:|---:|\n"]
    names = [n for n in res["models"] if res["models"][n].get("n")]
    names.sort(key=lambda n: res["models"][n]["log_loss"])
    for n in names:
        m = res["models"][n]
        tag = " **(BEST)**" if n == res.get("best_combo") else (" _(baseline)_" if n in ("elo", "market") else "")
        lines.append(f"| {n}{tag} | {m['n']} | {ev._fmt(m['log_loss'])} | {ev._fmt(m['brier'])} | "
                     f"{ev._fmt(m['ece_10'])} | {ev._fmt(m['roc_auc'])} | {ev._fmt(m['accuracy'])} |\n")
    return lines


def write_report(results: dict[str, dict], manifest: dict, path: Path) -> None:
    out = ["# MODEL_COMPARISON — zoo x block-wise bajo walk-forward anidado\n\n",
           f"Generado: {manifest['ts_utc']} · commit {manifest.get('git_commit')} · "
           f"dataset {manifest['dataset_rows']} filas (sha {manifest['dataset_sha256'][:12]}) · seed {manifest['seed']}\n\n",
           f"Modelos disponibles en el entorno: {', '.join(available_models())}\n\n",
           "> log loss es la metrica PRIMARIA (objetivo: calibracion y CLV, no accuracy bruta).\n"]
    for window, res in results.items():
        out.append(f"\n## Ventana {window} (n_eval={res['n_eval']}, folds={res['n_folds']})\n\n")
        if res.get("reused_equivalent_window"):
            out.append(
                f"> Resultado reutilizado de `{res['reused_equivalent_window']}`: "
                "los conjuntos train/test de todos los folds son exactamente iguales.\n\n"
            )
        out += _table(res)
        out.append(
            f"\n**Estimacion primaria sin sesgo de seleccion: `{res.get('best_combo')}`**\n"
        )
        out.append(
            "\nMejor combo fijo retrospectivo (solo diagnostico, no promocionable con "
            f"este mismo outer test): `{res.get('diagnostic_best_combo')}`.\n"
        )
        cc = res.get("calibrator_choice", {})
        if res.get("best_combo") in cc:
            out.append(f"\nCalibrador elegido en el mejor combo: {cc[res['best_combo']]}\n")
        b = res.get("betting", {})
        if b:
            out.append(f"\nApuestas (mejor combo): bets={b.get('bets')}, ROI={ev._fmt(b.get('roi_on_stake'))}, "
                       f"CLV vs cierre={ev._fmt(b.get('clv_vs_closing_mean'))} (n={b.get('clv_vs_closing_n')}).\n")
        selected_counts: dict[str, int] = {}
        for decision in res.get("decisions", []):
            selected = decision.get("selected_combo")
            if selected:
                selected_counts[selected] = selected_counts.get(selected, 0) + 1
        if selected_counts:
            out.append(f"\nCombos elegidos causalmente por fold: {selected_counts}.\n")
        # resumen de activacion data-driven
        fams = {}
        for d in res["decisions"]:
            for k, v in d.items():
                if isinstance(v, dict) and "families" in v:
                    fams.setdefault(k, set()).update(v["families"])
        if fams:
            out.append("\nActivacion data-driven (familias que entraron por log loss OOS, no por umbral fijo):\n")
            for m, s in fams.items():
                out.append(f"- {m}: {sorted(s) or 'ninguna'}\n")
    out.append("\n---\n*Bucle externo insesgado; familias/hiperparametros/calibrador decididos en el "
               "bucle interno con datos pasados. Model A sin odds; el mercado es baseline.*\n")
    path.write_text("".join(out), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Comparativa zoo x block-wise (walk-forward anidado).")
    p.add_argument("--db", type=Path, default=ROOT / "BBDD" / "cs2.db")
    p.add_argument("--config", type=Path, default=ROOT / "MODEL" / "config.yaml")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--window", choices=["expanding", "sliding", "both"], default="expanding")
    p.add_argument("--models", default="logistic_en,lightgbm")
    p.add_argument("--assemblies", default="indicators,profiles,two_stage")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument(
        "--retune-folds",
        type=int,
        default=6,
        help="Repite Optuna/forward selection cada N folds externos (6 ~= 24 semanas).",
    )
    p.add_argument("--verbose", action="store_true", help="Progreso por fold, modelo y tiempos.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args(argv)

    cfg, thresholds, seed = ev.load_eval_config(args.config, {})
    if args.synthetic:
        rows, base_cols, families = ev.build_synthetic_rows(seed)
    else:
        if not args.db.exists():
            print(f"[compare] no existe {args.db}. Usa --synthetic.")
            return 2
        rows, base_cols, families = ev.load_real_rows(args.db, thresholds)

    want = [m.strip() for m in args.models.split(",") if m.strip()]
    models = tuple(m for m in want if m in available_models())
    skipped = [m for m in want if m not in available_models()]
    if skipped:
        print(f"[compare] omitidos (no instalados en este entorno): {skipped}")
    if not models:
        print("[compare] ningun modelo disponible.")
        return 2
    assemblies = tuple(a.strip() for a in args.assemblies.split(",") if a.strip())
    comp = ComparisonConfig(
        models=models,
        assemblies=assemblies,
        n_trials=args.n_trials,
        verbose=args.verbose,
        retune_folds=max(1, args.retune_folds),
    )

    windows = ["expanding", "sliding"] if args.window == "both" else [args.window]
    results: dict[str, Any] = {}
    for w in windows:
        if (
            w == "sliding"
            and "expanding" in results
            and _expanding_sliding_equivalent(rows, cfg)
        ):
            results[w] = copy.deepcopy(results["expanding"])
            results[w]["window"] = "sliding"
            results[w]["reused_equivalent_window"] = "expanding"
            print(
                "[compare] sliding: mismos bloques train/test que expanding; "
                "reutilizando resultado exacto.",
                flush=True,
            )
            continue
        print(f"[compare] {w}: modelos={models} assemblies={assemblies} n_trials={args.n_trials}...", flush=True)
        results[w] = run_comparison(rows, base_cols, families, dataclasses.replace(cfg, window=w), comp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ev.run_manifest(rows, cfg, vars(args) | {"models": list(models), "assemblies": list(assemblies)}, results)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps({"manifest": manifest, "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(results, manifest, args.output_dir / "MODEL_COMPARISON.md")
    with (args.output_dir / "experiments.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, default=str) + "\n")

    for w, res in results.items():
        print(
            f"[compare] {w}: policy={res.get('best_combo')} "
            f"diagnostic_best={res.get('diagnostic_best_combo')} n_eval={res['n_eval']}"
        )
    print(f"[compare] reporte en {args.output_dir / 'MODEL_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

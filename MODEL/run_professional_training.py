"""Run a full professional CS2 model training sweep and save a Markdown report.

Default behavior:
  - runs MODEL/train.py with CatBoost enabled if installed
  - sweeps form half-life values and walk-forward gaps
  - streams verbose output to the terminal
  - saves the full output to MODEL/results/professional_training_output.md
  - copies each REPORT.md/metrics.json to a timestamped results directory
  - retrains the final production artifact with the best config by log loss

Usage:
    python MODEL/run_professional_training.py
    python MODEL/run_professional_training.py --half-lives 45,60,90 --wf-gaps 0
    python MODEL/run_professional_training.py --skip-final
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "MODEL" / "train.py"
RESULTS_DIR = ROOT / "MODEL" / "results"
BASELINE_NAMES = {"base_rate", "elo", "glicko"}


@dataclass
class RunResult:
    config: str
    half_life: float
    wf_gap: int
    model: str
    log_loss: float
    brier: float
    roc_auc: float
    ece_10: float
    accuracy: float
    returncode: int


def parse_float_list(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("La lista no puede estar vacia.")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("La lista no puede estar vacia.")
    return values


def format_half_life(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


def git_value(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def tee_command(cmd: list[str], report_fh, title: str, env: dict[str, str]) -> int:
    started = time.time()
    report_fh.write(f"\n## {title}\n\n")
    report_fh.write("Command:\n\n")
    report_fh.write("```powershell\n")
    report_fh.write(shell_join(cmd) + "\n")
    report_fh.write("```\n\n")
    report_fh.write("Output:\n\n```text\n")
    report_fh.flush()

    print(f"\n===== {title} =====", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        report_fh.write(line)
        report_fh.flush()
    returncode = process.wait()
    elapsed = time.time() - started
    report_fh.write("\n```\n")
    report_fh.write(f"\nReturn code: `{returncode}`  \n")
    report_fh.write(f"Elapsed: `{elapsed:.1f}s`\n")
    report_fh.flush()
    print(f"===== END {title} rc={returncode} elapsed={elapsed:.1f}s =====", flush=True)
    return returncode


def train_command(args: argparse.Namespace, half_life: float, wf_gap: int) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--form-half-life",
        str(half_life),
        "--wf-gap",
        str(wf_gap),
    ]
    if args.raw:
        cmd += ["--raw", str(Path(args.raw).resolve())]
    if args.warmup_weeks is not None:
        cmd += ["--warmup-weeks", str(args.warmup_weeks)]
    if args.min_train is not None:
        cmd += ["--min-train", str(args.min_train)]
    if args.no_cs2_filter:
        cmd.append("--no-cs2-filter")
    if args.no_catboost:
        cmd.append("--no-catboost")
    if not args.quiet_train:
        cmd.append("--verbose")
    return cmd


def best_from_metrics(metrics_path: Path) -> tuple[str, dict]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates = {
        name: values
        for name, values in metrics.items()
        if name not in BASELINE_NAMES
        and isinstance(values, dict)
        and "log_loss" in values
    }
    if not candidates:
        raise RuntimeError(f"No production candidates found in {metrics_path}")
    return min(candidates.items(), key=lambda item: item[1]["log_loss"])


def copy_current_outputs(run_dir: Path, config: str) -> tuple[Path, Path]:
    report_src = RESULTS_DIR / "REPORT.md"
    metrics_src = RESULTS_DIR / "metrics.json"
    report_dst = run_dir / f"REPORT_{config}.md"
    metrics_dst = run_dir / f"metrics_{config}.json"
    if report_src.exists():
        shutil.copy2(report_src, report_dst)
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dst)
    return report_dst, metrics_dst


def append_result_table(report_fh, rows: list[RunResult]) -> None:
    report_fh.write("\n## Summary sorted by log loss\n\n")
    report_fh.write("| Config | Production model | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |\n")
    report_fh.write("|---|---|---:|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda item: item.log_loss):
        report_fh.write(
            f"| {row.config} | {row.model} | {row.log_loss:.6f} | "
            f"{row.brier:.6f} | {row.roc_auc:.6f} | {row.ece_10:.6f} | "
            f"{row.accuracy:.6f} |\n"
        )
    if rows:
        best = min(rows, key=lambda item: item.log_loss)
        report_fh.write(
            "\nBest config by walk-forward log loss: "
            f"`{best.config}` using `{best.model}` "
            f"(logloss={best.log_loss:.6f}, accuracy={best.accuracy:.6f}).\n"
        )
    report_fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full verbose training sweep for the CS2 predictive model."
    )
    parser.add_argument("--half-lives", type=parse_float_list, default=parse_float_list("45,60,90,120,180"))
    parser.add_argument("--wf-gaps", type=parse_int_list, default=parse_int_list("0,1"))
    parser.add_argument("--output", default=str(Path("MODEL") / "results" / "professional_training_output.md"))
    parser.add_argument("--skip-final", action="store_true",
                        help="Do not retrain the final artifact with the best config.")
    parser.add_argument("--no-catboost", action="store_true",
                        help="Disable CatBoost candidates.")
    parser.add_argument("--quiet-train", action="store_true",
                        help="Do not pass --verbose to MODEL/train.py.")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install requirements.txt and catboost before training.")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--warmup-weeks", type=int, default=None)
    parser.add_argument("--min-train", type=int, default=None)
    parser.add_argument("--no-cs2-filter", action="store_true")
    args = parser.parse_args()

    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"No existe {TRAIN_SCRIPT}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"professional_train_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with report_path.open("w", encoding="utf-8", newline="\n") as report_fh:
        report_fh.write("# Professional CS2 training output\n\n")
        report_fh.write(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`  \n")
        report_fh.write(f"Root: `{ROOT}`  \n")
        report_fh.write(f"Run dir: `{run_dir}`  \n")
        report_fh.write(f"Git branch: `{git_value('branch', '--show-current')}`  \n")
        report_fh.write(f"Git commit: `{git_value('rev-parse', '--short', 'HEAD')}`  \n")
        report_fh.write(f"CatBoost requested: `{not args.no_catboost}`  \n")
        report_fh.write(f"Half-lives: `{', '.join(map(str, args.half_lives))}`  \n")
        report_fh.write(f"WF gaps: `{', '.join(map(str, args.wf_gaps))}`  \n")

        if args.install_deps:
            rc = tee_command(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                report_fh,
                "Install requirements",
                env,
            )
            if rc != 0:
                return rc
            if not args.no_catboost:
                rc = tee_command(
                    [sys.executable, "-m", "pip", "install", "catboost"],
                    report_fh,
                    "Install CatBoost",
                    env,
                )
                if rc != 0:
                    return rc

        if not args.no_catboost:
            try:
                import catboost  # noqa: F401
                report_fh.write("\nCatBoost import: `OK`\n")
            except Exception as exc:
                report_fh.write(f"\nCatBoost import: `FAILED` - `{exc}`\n")
                report_fh.write(
                    "MODEL/train.py will continue without CatBoost candidates unless "
                    "CatBoost is installed in this Python environment.\n"
                )

        rows: list[RunResult] = []
        for gap in args.wf_gaps:
            for half_life in args.half_lives:
                config = f"h{format_half_life(half_life)}_gap{gap}"
                cmd = train_command(args, half_life, gap)
                rc = tee_command(cmd, report_fh, f"TRAIN {config}", env)
                report_dst, metrics_dst = copy_current_outputs(run_dir, config)
                report_fh.write(
                    f"\nCopied report: `{report_dst}`  \n"
                    f"Copied metrics: `{metrics_dst}`\n"
                )
                if rc != 0:
                    report_fh.write(f"\nStopping because `{config}` failed.\n")
                    return rc

                model_name, metrics = best_from_metrics(metrics_dst)
                row = RunResult(
                    config=config,
                    half_life=half_life,
                    wf_gap=gap,
                    model=model_name,
                    log_loss=float(metrics["log_loss"]),
                    brier=float(metrics["brier"]),
                    roc_auc=float(metrics["roc_auc"]),
                    ece_10=float(metrics["ece_10"]),
                    accuracy=float(metrics["accuracy"]),
                    returncode=rc,
                )
                rows.append(row)
                report_fh.write(
                    f"\nResult `{config}`: prod=`{row.model}` "
                    f"logloss=`{row.log_loss:.6f}` acc=`{row.accuracy:.6f}`\n"
                )
                report_fh.flush()

        append_result_table(report_fh, rows)

        if not rows:
            report_fh.write("\nNo completed training runs.\n")
            return 1

        best = min(rows, key=lambda item: item.log_loss)
        if not args.skip_final:
            report_fh.write("\n## Final production training\n\n")
            report_fh.write(
                "Retraining final artifact with the best config by log loss: "
                f"`{best.config}`.\n"
            )
            rc = tee_command(
                train_command(args, best.half_life, best.wf_gap),
                report_fh,
                f"FINAL TRAIN {best.config}",
                env,
            )
            copy_current_outputs(run_dir, "FINAL_BEST")
            if rc != 0:
                return rc

            check_code = (
                "import sys; "
                "from pathlib import Path; "
                "sys.path.insert(0, str(Path('MODEL').resolve())); "
                "from cs2model.artifacts import load_artifact; "
                "a=load_artifact(); "
                "print('production_model:', a.metadata.get('production_model')); "
                "print('calibration:', a.metadata.get('production_calibration')); "
                "print('components:', a.metadata.get('production_components')); "
                "print('catboost:', a.metadata.get('catboost_enabled')); "
                "print('half_life:', a.metadata.get('form_half_life_days')); "
                "print('wf_gap:', a.metadata.get('walk_forward_gap'))"
            )
            rc = tee_command(
                [sys.executable, "-c", check_code],
                report_fh,
                "Validate final artifact",
                env,
            )
            if rc != 0:
                return rc

        archive_path = report_path.with_name(f"{report_path.stem}_{stamp}{report_path.suffix}")
        report_fh.write(f"\nArchived copy: `{archive_path}`\n")

    shutil.copy2(report_path, archive_path)
    print(f"\nOutput saved to: {report_path}", flush=True)
    print(f"Archived copy:  {archive_path}", flush=True)
    print(f"Per-config reports: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

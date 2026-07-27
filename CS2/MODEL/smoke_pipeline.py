"""Fast deterministic end-to-end smoke for the model training pipeline."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def synthetic_results(seed: int = 42, weeks: int = 40) -> list[dict]:
    rng = random.Random(seed)
    teams = [f"Smoke Team {index}" for index in range(12)]
    strengths = {team: (index - 5.5) / 5.0 for index, team in enumerate(teams)}
    start = date(2024, 1, 1)
    rows: list[dict] = []
    match_id = 9_000_000
    for week in range(weeks):
        for slot in range(4):
            team1, team2 = rng.sample(teams, 2)
            probability = 1.0 / (1.0 + pow(2.718281828, -(strengths[team1] - strengths[team2])))
            team1_wins = rng.random() < probability
            score1, score2 = ((2, rng.randint(0, 1)) if team1_wins else (rng.randint(0, 1), 2))
            match_date = start + timedelta(days=week * 7 + slot)
            rows.append(
                {
                    "id": str(match_id),
                    "link": f"/matches/{match_id}/smoke",
                    "map": "bo3",
                    "event": f"Smoke Event {week // 5}",
                    "date": match_date.isoformat(),
                    "team1": {"name": team1, "score": str(score1)},
                    "team2": {"name": team2, "score": str(score2)},
                }
            )
            match_id += 1
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cs2_smoke_") as temp:
        temp_path = Path(temp)
        raw_path = temp_path / "results.json"
        master_path = temp_path / "master.json"
        output_path = temp_path / "output"
        raw_path.write_text(json.dumps(synthetic_results()), encoding="utf-8")
        master_path.write_text("{}", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "MODEL" / "train.py"),
            "--raw", str(raw_path),
            "--output-dir", str(output_path),
            "--master", str(master_path),
            "--algorithms", "logistic",
            "--warmup-weeks", "4",
            "--min-train", "30",
            "--optuna-trials", "0",
            "--no-promote",
            "--smoke",
            "--seed", "42",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
        required = {
            "model.pkl", "metrics.json", "favorite_accuracy_bands.json",
            "experiment_manifest.json", "config.effective.yaml", "drift_report.json",
            "economic_backtest.json",
        }
        missing = sorted(name for name in required if not (output_path / name).exists())
        if missing:
            raise RuntimeError(f"Smoke output missing: {missing}")
        manifest = json.loads((output_path / "experiment_manifest.json").read_text(encoding="utf-8"))
        effective_config = yaml.safe_load(
            (output_path / "config.effective.yaml").read_text(encoding="utf-8")
        )
        metrics = json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))
        if (
            manifest.get("random_seed") != 42
            or effective_config.get("random_seed") != 42
            or effective_config.get("cli_arguments", {}).get("smoke") is not True
            or "logistic_cal" not in metrics
        ):
            raise RuntimeError("Smoke manifest or metrics are invalid")
        print(
            "smoke_pipeline=ok "
            f"rows={manifest['dataset_rows']} dataset={manifest['dataset_sha256'][:12]} "
            f"accuracy={metrics['logistic_cal']['accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

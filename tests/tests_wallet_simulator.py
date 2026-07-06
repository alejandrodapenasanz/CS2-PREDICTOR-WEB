from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from cs2model import dataio
from cs2model.features import build_training_frame
from DAILY_SNAPSHOTS.enrich_predictions import reliability_score


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
DEFAULT_MASTER = ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"
DEFAULT_PREDICTIONS = ROOT / "MODEL" / "results" / "predictions_walkforward.csv"
DEFAULT_GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_predictions(path: Path, model_name: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de predicciones: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") != model_name:
                continue
            match_id = str(row.get("match_id") or "")
            probability = safe_float(row.get("prob_team1"))
            if match_id and probability is not None:
                out[match_id] = {**row, "prob_team1": probability}
    return out


def load_rows(raw_path: Path, master_path: Path) -> list[dict[str, Any]]:
    return dataio.load_training_rows(raw_path, master_path if master_path.exists() else None, cs2_only=True)


def reliability_features_from_training_row(feats: dict[str, float]) -> dict[str, Any]:
    total_matches = max(0.0, safe_float(feats.get("experience_total"), 0.0) or 0.0)
    min_matches = max(0.0, safe_float(feats.get("experience_min"), 0.0) or 0.0)
    max_matches = max(0.0, total_matches - min_matches)
    if (safe_float(feats.get("matches_log_diff"), 0.0) or 0.0) >= 0.0:
        matches_team1, matches_team2 = max_matches, min_matches
    else:
        matches_team1, matches_team2 = min_matches, max_matches

    rd_sum = safe_float(feats.get("glicko_rd_sum"))
    rd_diff = safe_float(feats.get("glicko_rd_diff"))
    if rd_sum is not None and rd_diff is not None:
        rd1 = (rd_sum + rd_diff) / 2.0
        rd2 = (rd_sum - rd_diff) / 2.0
    else:
        rd1 = rd2 = 160.0

    return {
        **feats,
        "matches_team1": matches_team1,
        "matches_team2": matches_team2,
        "glicko_rd_team1": rd1,
        "glicko_rd_team2": rd2,
        "player_coverage_team1": 1.0,
        "player_coverage_team2": 1.0,
    }


def build_reliability_by_match(rows: list[dict[str, Any]]) -> dict[str, float]:
    X_dicts, _y, meta, _state = build_training_frame(rows)
    reliability_by_id: dict[str, float] = {}
    for feats, info in zip(X_dicts, meta):
        entry = {
            "format": info.get("format"),
            "team1": {"name": info.get("team1")},
            "team2": {"name": info.get("team2")},
            "data_quality": {"real_pre_match_snapshot": True},
        }
        reliability_by_id[str(info["id"])] = reliability_score(entry, reliability_features_from_training_row(feats))
    return reliability_by_id


def make_bet_candidates(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    reliability_by_id: dict[str, float],
) -> list[dict[str, Any]]:
    bets: list[dict[str, Any]] = []
    for row in rows:
        match_id = str(row.get("id") or "")
        pred = predictions.get(match_id)
        if not pred:
            continue
        p_team1 = safe_float(pred.get("prob_team1"))
        if p_team1 is None:
            continue

        side = "team1" if p_team1 >= 0.5 else "team2"
        confidence = p_team1 if side == "team1" else 1.0 - p_team1
        decimal_odds = safe_float(row.get("opening_odds_decimal_t1" if side == "team1" else "opening_odds_decimal_t2"))
        if decimal_odds is None or decimal_odds <= 1.0:
            continue

        team1_won = bool(row.get("team1_win"))
        won = team1_won if side == "team1" else not team1_won
        bets.append(
            {
                "match_id": match_id,
                "date": row.get("date") or "",
                "event": row.get("event") or "",
                "side": side,
                "team": row.get("team1" if side == "team1" else "team2"),
                "prob_team1": p_team1,
                "confidence": confidence,
                "reliability": reliability_by_id.get(match_id, 0.0),
                "decimal_odds": decimal_odds,
                "won": won,
            }
        )
    bets.sort(key=lambda bet: (str(bet["date"]), str(bet["match_id"])))
    return bets


def simulate_wallet(
    bets: list[dict[str, Any]],
    initial_wallet: float,
    stake_fraction: float,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    wallet = float(initial_wallet)
    peak = wallet
    max_drawdown = 0.0
    total_staked = 0.0
    profit = 0.0
    wins = 0
    curve = [
        {
            "bet_no": 0,
            "match_id": "inicio",
            "date": "",
            "wallet": wallet,
            "stake": 0.0,
            "pnl": 0.0,
            "won": None,
        }
    ]

    for bet in bets:
        if not predicate(bet):
            continue
        decimal_odds = safe_float(bet.get("decimal_odds"), 0.0) or 0.0
        stake = wallet * stake_fraction
        won = bool(bet.get("won"))
        pnl = stake * (decimal_odds - 1.0) if won else -stake
        wallet += pnl
        peak = max(peak, wallet)
        max_drawdown = max(max_drawdown, (peak - wallet) / peak if peak else 0.0)
        total_staked += stake
        profit += pnl
        wins += 1 if won else 0
        curve.append(
            {
                **bet,
                "bet_no": len(curve),
                "wallet": wallet,
                "stake": stake,
                "pnl": pnl,
                "stake_fraction": stake_fraction,
            }
        )

    n_bets = len(curve) - 1
    return {
        "initial_wallet": initial_wallet,
        "final_wallet": wallet,
        "profit": profit,
        "profit_pct": profit / initial_wallet if initial_wallet else 0.0,
        "n_bets": n_bets,
        "wins": wins,
        "hit_rate": wins / n_bets if n_bets else None,
        "total_staked": total_staked,
        "roi_on_staked": profit / total_staked if total_staked else None,
        "max_drawdown": max_drawdown,
        "stake_fraction": stake_fraction,
        "curve": curve,
    }


def plot_wallet(title: str, simulation: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    curve = simulation["curve"]
    xs = [row["bet_no"] for row in curve]
    ys = [row["wallet"] for row in curve]
    color = "#16a34a" if simulation["final_wallet"] >= simulation["initial_wallet"] else "#dc2626"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.plot(xs, ys, color=color, linewidth=2.4, marker="o", markersize=3.8)
    ax.axhline(simulation["initial_wallet"], color="#64748b", linewidth=1.2, linestyle="--", label="Saldo inicial")
    ax.fill_between(xs, ys, simulation["initial_wallet"], color=color, alpha=0.12)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("Numero de apuesta")
    ax.set_ylabel("Capital")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f} EUR"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    text = (
        f"Apuestas: {simulation['n_bets']}  |  "
        f"Inicial: {simulation['initial_wallet']:.2f} EUR  |  "
        f"Final: {simulation['final_wallet']:.2f} EUR  |  "
        f"Stake: {simulation['stake_fraction'] * 100:.2f}%"
    )
    ax.text(
        0.01,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fmt_money(value: float) -> str:
    return f"{value:,.2f} EUR"


def fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.2f}%"


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera graficos de backtest de cartera para estrategias de favorito del modelo."
    )
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Historico results_all.json")
    parser.add_argument("--master", default=str(DEFAULT_MASTER), help="DAILY_SNAPSHOTS/master/matches.json")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="CSV walk-forward del modelo")
    parser.add_argument("--model", default="ensemble_cal", help="Modelo del CSV a simular")
    parser.add_argument("--initial-wallet", type=float, default=1000.0, help="Saldo inicial de la cartera")
    parser.add_argument("--stake-fraction", type=float, default=0.025, help="Fraccion de banca actual por apuesta")
    parser.add_argument("--graphs-dir", default=str(DEFAULT_GRAPHS_DIR), help="Carpeta de salida de los PNG")
    args = parser.parse_args()

    rows = load_rows(Path(args.raw), Path(args.master))
    predictions = read_predictions(Path(args.predictions), args.model)
    reliability_by_id = build_reliability_by_match(rows)
    candidates = make_bet_candidates(rows, predictions, reliability_by_id)

    strategies: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        (
            "01_always_model_favorite.png",
            "1) Siempre al favorito del modelo",
            lambda _bet: True,
        ),
        (
            "02_model_favorite_conf_gt_60.png",
            "2) Favorito del modelo con confianza > 60%",
            lambda bet: bet["confidence"] > 0.60,
        ),
        (
            "03_model_favorite_conf_gt_70.png",
            "3) Favorito del modelo con confianza > 70%",
            lambda bet: bet["confidence"] > 0.70,
        ),
        (
            "04_model_favorite_conf_gt_70_reliability_gt_70.png",
            "4) Favorito > 70% y fiabilidad > 70%",
            lambda bet: bet["confidence"] > 0.70 and bet["reliability"] > 0.70,
        ),
    ]

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    print(f"Partidos con odds y prediccion usados como universo: {len(candidates)}")
    print(f"Dinero inicial: {fmt_money(args.initial_wallet)}")
    print(f"Stake por apuesta: {fmt_pct(args.stake_fraction)}")
    print("")

    for filename, title, predicate in strategies:
        simulation = simulate_wallet(candidates, args.initial_wallet, args.stake_fraction, predicate)
        output_path = graphs_dir / filename
        plot_wallet(title, simulation, output_path)
        row = {
            "strategy": title,
            "bets": simulation["n_bets"],
            "initial_wallet": simulation["initial_wallet"],
            "final_wallet": simulation["final_wallet"],
            "profit": simulation["profit"],
            "profit_pct": simulation["profit_pct"],
            "hit_rate": simulation["hit_rate"],
            "roi_on_staked": simulation["roi_on_staked"],
            "max_drawdown": simulation["max_drawdown"],
            "graph": str(output_path.resolve()),
        }
        summary.append(row)
        print(title)
        print(f"  Apuestas hechas: {simulation['n_bets']}")
        print(f"  Dinero inicial: {fmt_money(simulation['initial_wallet'])}")
        print(f"  Dinero final: {fmt_money(simulation['final_wallet'])}")
        print(f"  Beneficio: {fmt_money(simulation['profit'])} ({fmt_pct(simulation['profit_pct'])})")
        print(f"  Hit rate: {fmt_pct(simulation['hit_rate'])}")
        print(f"  ROI sobre stake apostado: {fmt_pct(simulation['roi_on_staked'])}")
        print(f"  Max drawdown: {fmt_pct(simulation['max_drawdown'])}")
        print(f"  Grafico: {output_path.resolve()}")
        print("")

    summary_path = graphs_dir / "summary.json"
    write_summary(summary_path, summary)
    print(f"Resumen JSON: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

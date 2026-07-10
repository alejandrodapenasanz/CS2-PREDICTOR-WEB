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
from cs2model.artifacts import load_artifact
from cs2model.features import build_training_frame
from DAILY_SNAPSHOTS.enrich_predictions import (
    model_calibration_factor,
    reliability_score,
)


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
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
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


def load_rows(db_path: Path, raw_path: Path, master_path: Path) -> list[dict[str, Any]]:
    if db_path.exists():
        return dataio.load_training_rows(db_path=db_path, cs2_only=True)
    return dataio.load_training_rows(raw_path, master_path if master_path.exists() else None, cs2_only=True)


def production_model() -> tuple[str, dict[str, Any]]:
    artifact = load_artifact()
    if artifact is None:
        return "ensemble_cal", {}
    return str(artifact.metadata.get("production_model") or "ensemble_cal"), artifact.metadata


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
        odds_team1 = safe_float(row.get("opening_odds_decimal_t1"))
        odds_team2 = safe_float(row.get("opening_odds_decimal_t2"))
        if odds_team1 is None or odds_team2 is None or min(odds_team1, odds_team2) <= 1.0:
            continue
        market_p_team1 = safe_float(row.get("opening_odds_t1"))
        if market_p_team1 is None:
            implied1, implied2 = 1.0 / odds_team1, 1.0 / odds_team2
            market_p_team1 = implied1 / (implied1 + implied2)
        decimal_odds = odds_team1 if side == "team1" else odds_team2

        team1_won = bool(row.get("team1_win"))
        won = team1_won if side == "team1" else not team1_won
        bets.append(
            {
                "match_id": match_id,
                "date": row.get("date") or "",
                "event": row.get("event") or "",
                "side": side,
                "team": row.get("team1" if side == "team1" else "team2"),
                "team1": row.get("team1"),
                "team2": row.get("team2"),
                "prob_team1": p_team1,
                "confidence": confidence,
                "reliability": reliability_by_id.get(match_id, 0.0),
                "decimal_odds": decimal_odds,
                "decimal_odds_team1": odds_team1,
                "decimal_odds_team2": odds_team2,
                "market_prob_team1": market_p_team1,
                "bookmaker_count": int(safe_float(row.get("opening_bookmaker_count"), 0.0) or 0),
                "team1_won": team1_won,
                "won": won,
            }
        )
    # La semilla historica y el ingest diario pueden representar el mismo
    # partido con IDs distintos. Apostarlo dos veces inflaria el backtest.
    unique: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for bet in bets:
        teams = tuple(sorted((str(bet.get("team1") or "").casefold(), str(bet.get("team2") or "").casefold())))
        key = (str(bet.get("date") or ""), teams)
        current = unique.get(key)
        if current is None or len(str(bet["match_id"])) > len(str(current["match_id"])):
            unique[key] = bet
    deduplicated = list(unique.values())
    deduplicated.sort(key=lambda bet: (str(bet["date"]), str(bet["match_id"])))
    return deduplicated


def make_recommended_bets(
    candidates: list[dict[str, Any]],
    model_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calcula el stake solo para el favorito puro del modelo.

    Las cuotas determinan el EV y el tamano, pero nunca cambian el equipo.
    """
    calibration_factor, ece, eval_n = model_calibration_factor(model_metadata)
    recommended: list[dict[str, Any]] = []

    for bet in candidates:
        # El lado viene exclusivamente de prob_team1. El mercado solo aporta
        # la cuota necesaria para medir EV y dimensionar el stake.
        side = str(bet["side"])
        probability = float(bet["confidence"])
        decimal_odds = float(bet["decimal_odds"])
        reliability = float(bet.get("reliability") or 0.05)
        bookmaker_count = int(bet.get("bookmaker_count") or 0)
        consensus_factor = 1.0 if bookmaker_count >= 3 else 0.80 if bookmaker_count == 2 else 0.55
        expected_roi = probability * decimal_odds - 1.0
        raw_kelly = max(0.0, expected_roi / (decimal_odds - 1.0))
        adjusted = raw_kelly * 0.25 * reliability * consensus_factor * calibration_factor
        recommended_fraction = min(adjusted, 0.025)
        if recommended_fraction <= 0.0:
            continue
        recommended.append(
            {
                **bet,
                "side": side,
                "team": bet[side],
                "probability": probability,
                "decimal_odds": decimal_odds,
                "expected_roi": expected_roi,
                "recommended_fraction": recommended_fraction,
                "won": bool(bet["team1_won"]) if side == "team1" else not bool(bet["team1_won"]),
                "calibration_factor": calibration_factor,
                "walk_forward_ece": ece,
                "walk_forward_n": eval_n,
                "stake_method": "quarter_kelly_reliability_consensus_calibration_cap_2_5pct",
            }
        )
    return recommended


def simulate_wallet(
    bets: list[dict[str, Any]],
    initial_wallet: float,
    stake_fraction: float | None,
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
        fraction = stake_fraction
        if fraction is None:
            fraction = safe_float(bet.get("recommended_fraction"), 0.0) or 0.0
        if fraction <= 0.0:
            continue
        stake = wallet * fraction
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
                "stake_fraction": fraction,
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
        "stake_method": "recommended_dynamic" if stake_fraction is None else "fixed_fraction",
        "average_stake_fraction": (
            sum(float(row.get("stake_fraction") or 0.0) for row in curve[1:]) / n_bets
            if n_bets else 0.0
        ),
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

    stake_label = (
        f"dinamico (media {simulation['average_stake_fraction'] * 100:.2f}%)"
        if simulation.get("stake_fraction") is None
        else f"{simulation['stake_fraction'] * 100:.2f}%"
    )
    text = (
        f"Apuestas: {simulation['n_bets']}  |  "
        f"Inicial: {simulation['initial_wallet']:.2f} EUR  |  "
        f"Final: {simulation['final_wallet']:.2f} EUR  |  "
        f"Stake: {stake_label}"
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


def write_bet_ledger(path: Path, simulation: dict[str, Any]) -> None:
    fields = [
        "bet_no", "match_id", "date", "event", "team1", "team2", "side", "team",
        "prob_team1", "probability", "market_prob_team1", "decimal_odds", "expected_roi",
        "reliability", "stake_fraction", "stake",
        "won", "pnl", "wallet",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(simulation["curve"][1:])


def make_accuracy_evaluations(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for row in rows:
        match_id = str(row.get("id") or "")
        prediction = predictions.get(match_id)
        if prediction is None:
            continue
        p_team1 = safe_float(prediction.get("prob_team1"))
        if p_team1 is None:
            continue
        p_team1 = min(max(p_team1, 0.0), 1.0)
        side = "team1" if p_team1 >= 0.5 else "team2"
        confidence = p_team1 if side == "team1" else 1.0 - p_team1
        team1_won = bool(row.get("team1_win"))
        won = team1_won if side == "team1" else not team1_won
        evaluations.append(
            {
                "match_id": match_id,
                "date": row.get("date") or "",
                "team1": row.get("team1"),
                "team2": row.get("team2"),
                "favorite": row.get(side),
                "side": side,
                "prob_team1": p_team1,
                "confidence": confidence,
                "correct": won,
            }
        )
    evaluations.sort(key=lambda row: (str(row["date"]), str(row["match_id"])))
    return evaluations


def summarize_accuracy_bands(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bands = [
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.01, "90-100%"),
    ]
    summary: list[dict[str, Any]] = []
    for lower, upper, label in bands:
        selected = [
            row
            for row in evaluations
            if float(row["confidence"]) >= lower and float(row["confidence"]) < upper
        ]
        n = len(selected)
        correct = sum(1 for row in selected if row["correct"])
        accuracy = correct / n if n else None
        average_confidence = (
            sum(float(row["confidence"]) for row in selected) / n if n else None
        )
        summary.append(
            {
                "band": label,
                "n": n,
                "correct": correct,
                "accuracy": accuracy,
                "average_predicted_probability": average_confidence,
                "calibration_gap": (
                    accuracy - average_confidence
                    if accuracy is not None and average_confidence is not None
                    else None
                ),
            }
        )
    return summary


def write_accuracy_bands(path: Path, bands: list[dict[str, Any]]) -> None:
    fields = [
        "band", "n", "correct", "accuracy",
        "average_predicted_probability", "calibration_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bands)


def plot_accuracy_bands(bands: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["band"]) for row in bands]
    actual = [float(row["accuracy"] or 0.0) * 100.0 for row in bands]
    predicted = [float(row["average_predicted_probability"] or 0.0) * 100.0 for row in bands]
    counts = [int(row["n"]) for row in bands]
    xs = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars_actual = ax.bar(
        [x - width / 2 for x in xs], actual, width,
        label="Accuracy observada", color="#16a34a",
    )
    ax.bar(
        [x + width / 2 for x in xs], predicted, width,
        label="Probabilidad media predicha", color="#2563eb",
    )
    for bar, count in zip(bars_actual, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Accuracy walk-forward por confianza del favorito")
    ax.set_xlabel("Probabilidad asignada al favorito del modelo")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera graficos de backtest de cartera para estrategias de favorito del modelo."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="BBDD SQLite principal")
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Historico results_all.json")
    parser.add_argument("--master", default=str(DEFAULT_MASTER), help="DAILY_SNAPSHOTS/master/matches.json")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="CSV walk-forward del modelo")
    parser.add_argument("--model", default="", help="Modelo del CSV; vacio usa el artefacto productivo")
    parser.add_argument("--initial-wallet", type=float, default=100.0, help="Saldo inicial de la cartera")
    parser.add_argument("--stake-fraction", type=float, default=0.025, help="Fraccion de banca actual por apuesta")
    parser.add_argument("--graphs-dir", default=str(DEFAULT_GRAPHS_DIR), help="Carpeta de salida de los PNG")
    args = parser.parse_args()

    production_name, model_metadata = production_model()
    model_name = args.model or production_name
    rows = load_rows(Path(args.db), Path(args.raw), Path(args.master))
    predictions = read_predictions(Path(args.predictions), model_name)
    reliability_by_id = build_reliability_by_match(rows)
    candidates = make_bet_candidates(rows, predictions, reliability_by_id)
    recommended_bets = make_recommended_bets(candidates, model_metadata)
    accuracy_evaluations = make_accuracy_evaluations(rows, predictions)
    accuracy_bands = summarize_accuracy_bands(accuracy_evaluations)

    strategies: list[
        tuple[str, str, list[dict[str, Any]], float | None, Callable[[dict[str, Any]], bool]]
    ] = [
        (
            "01_always_model_favorite.png",
            "1) Siempre al favorito del modelo",
            candidates,
            args.stake_fraction,
            lambda _bet: True,
        ),
        (
            "02_model_favorite_conf_gt_60.png",
            "2) Favorito del modelo con confianza > 60%",
            candidates,
            args.stake_fraction,
            lambda bet: bet["confidence"] > 0.60,
        ),
        (
            "03_model_favorite_conf_gt_70.png",
            "3) Favorito del modelo con confianza > 70%",
            candidates,
            args.stake_fraction,
            lambda bet: bet["confidence"] > 0.70,
        ),
        (
            "04_model_favorite_conf_gt_70_reliability_gt_70.png",
            "4) Favorito > 70% y fiabilidad > 70%",
            candidates,
            args.stake_fraction,
            lambda bet: bet["confidence"] > 0.70 and bet["reliability"] > 0.70,
        ),
        (
            "05_recommended_positive_ev_dynamic_stake.png",
            "5) Favorito del modelo: EV positivo + stake dinamico",
            recommended_bets,
            None,
            lambda _bet: True,
        ),
    ]

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    print(f"Partidos con odds y prediccion usados como universo: {len(candidates)}")
    print(f"Modelo walk-forward: {model_name}")
    print(f"Favoritos del modelo con EV positivo recomendados: {len(recommended_bets)}")
    print(f"Dinero inicial: {fmt_money(args.initial_wallet)}")
    print(f"Stake escenarios 1-4: {fmt_pct(args.stake_fraction)}")
    print("Stake escenario 5: quarter-Kelly ajustado, cap 2.50%")
    print("")

    for filename, title, strategy_bets, strategy_stake, predicate in strategies:
        simulation = simulate_wallet(strategy_bets, args.initial_wallet, strategy_stake, predicate)
        output_path = graphs_dir / filename
        plot_wallet(title, simulation, output_path)
        ledger_path = output_path.with_suffix(".csv")
        write_bet_ledger(ledger_path, simulation)
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
            "stake_method": simulation["stake_method"],
            "average_stake_fraction": simulation["average_stake_fraction"],
            "graph": str(output_path.resolve()),
            "ledger": str(ledger_path.resolve()),
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
        print(f"  Stake medio: {fmt_pct(simulation['average_stake_fraction'])}")
        print(f"  Grafico: {output_path.resolve()}")
        print(f"  Registro: {ledger_path.resolve()}")
        print("")

    accuracy_csv = graphs_dir / "accuracy_by_confidence_band.csv"
    accuracy_json = graphs_dir / "accuracy_by_confidence_band.json"
    accuracy_graph = graphs_dir / "06_accuracy_by_confidence_band.png"
    write_accuracy_bands(accuracy_csv, accuracy_bands)
    accuracy_json.write_text(
        json.dumps(accuracy_bands, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_accuracy_bands(accuracy_bands, accuracy_graph)
    overall_correct = sum(1 for row in accuracy_evaluations if row["correct"])
    overall_accuracy = (
        overall_correct / len(accuracy_evaluations) if accuracy_evaluations else None
    )
    print("Accuracy walk-forward del favorito por franjas:")
    for band in accuracy_bands:
        print(
            f"  {band['band']}: {band['correct']}/{band['n']} = "
            f"{fmt_pct(band['accuracy'])} "
            f"(prob. media {fmt_pct(band['average_predicted_probability'])})"
        )
    print(
        f"  TOTAL: {overall_correct}/{len(accuracy_evaluations)} = "
        f"{fmt_pct(overall_accuracy)}"
    )
    print(f"  Tabla: {accuracy_csv.resolve()}")
    print(f"  Grafico: {accuracy_graph.resolve()}")
    print("")

    summary_path = graphs_dir / "summary.json"
    write_summary(summary_path, summary)
    print(f"Resumen JSON: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

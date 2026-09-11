"""Realistic moneyline backtesting with vig, limits and closing-line value."""

from __future__ import annotations

import math
import statistics
from typing import Any

from .config import EconomicConfig


def safe_decimal_odds(value: Any) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    return odds if math.isfinite(odds) and odds > 1.0 else None


def devig_two_way(odds1: float, odds2: float) -> dict[str, float]:
    raw1, raw2 = 1.0 / odds1, 1.0 / odds2
    total = raw1 + raw2
    if total <= 0.0:
        raise ValueError("Invalid two-way market")
    return {
        "fair_prob_team1": raw1 / total,
        "fair_prob_team2": raw2 / total,
        "overround": total - 1.0,
    }


def closing_line_value(opening_odds: float, closing_odds: float) -> float:
    """Price CLV. Positive means the bettor obtained a better price than close."""
    if opening_odds <= 1.0 or closing_odds <= 1.0:
        raise ValueError("CLV requires valid decimal odds")
    return opening_odds / closing_odds - 1.0


def _kelly_fraction(probability: float, decimal_odds: float) -> float:
    edge = probability * decimal_odds - 1.0
    if edge <= 0.0 or decimal_odds <= 1.0:
        return 0.0
    return edge / (decimal_odds - 1.0)


def economic_backtest(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    config: EconomicConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EconomicConfig()
    pred_by_id = {str(row["match_id"]): row for row in predictions}
    bankroll = float(cfg.initial_bankroll)
    initial_bankroll = bankroll
    peak = bankroll
    max_drawdown = 0.0
    staked = 0.0
    profit = 0.0
    bets: list[dict[str, Any]] = []
    skipped = {
        "missing_opening_odds": 0,
        "missing_closing_odds_required": 0,
        "non_positive_ev": 0,
        "below_minimum_stake": 0,
    }
    constrained_bets = 0

    for row in rows:
        pred = pred_by_id.get(str(row.get("id")))
        if not pred:
            continue
        p1 = float(pred["prob_team1"])
        p2 = 1.0 - p1
        odds1 = safe_decimal_odds(row.get("opening_odds_decimal_t1"))
        odds2 = safe_decimal_odds(row.get("opening_odds_decimal_t2"))
        if odds1 is None or odds2 is None:
            skipped["missing_opening_odds"] += 1
            continue
        opening_market = devig_two_way(odds1, odds2)
        side = "team1" if p1 >= 0.5 else "team2"
        probability = p1 if side == "team1" else p2
        quoted_odds = odds1 if side == "team1" else odds2
        executed_odds = 1.0 + (quoted_odds - 1.0) * (1.0 - cfg.execution_odds_haircut)
        expected_roi = probability * executed_odds - 1.0
        kelly = _kelly_fraction(probability, executed_odds)
        if kelly <= 0.0 or expected_roi < cfg.min_expected_roi:
            skipped["non_positive_ev"] += 1
            continue

        close1 = safe_decimal_odds(row.get("closing_odds_decimal_t1"))
        close2 = safe_decimal_odds(row.get("closing_odds_decimal_t2"))
        close_side = close1 if side == "team1" else close2
        if cfg.require_closing_odds and close_side is None:
            skipped["missing_closing_odds_required"] += 1
            continue

        requested_fraction = min(cfg.max_stake_fraction, cfg.kelly_multiplier * kelly)
        requested_stake = bankroll * requested_fraction
        payout_limit_stake = cfg.max_profit_amount / max(executed_odds - 1.0, 1e-9)
        stake = min(requested_stake, cfg.max_stake_amount, payout_limit_stake, bankroll)
        if stake + 1e-12 < requested_stake:
            constrained_bets += 1
        if stake < cfg.min_stake_amount:
            skipped["below_minimum_stake"] += 1
            continue

        won = bool(row["team1_win"]) if side == "team1" else not bool(row["team1_win"])
        pnl = stake * (executed_odds - 1.0) if won else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak else 0.0)
        staked += stake
        profit += pnl

        clv_price = None
        clv_probability = None
        closing_overround = None
        if close1 is not None and close2 is not None and close_side is not None:
            closing_market = devig_two_way(close1, close2)
            clv_price = closing_line_value(quoted_odds, close_side)
            close_probability = closing_market["fair_prob_team1" if side == "team1" else "fair_prob_team2"]
            open_probability = opening_market["fair_prob_team1" if side == "team1" else "fair_prob_team2"]
            clv_probability = close_probability - open_probability
            closing_overround = closing_market["overround"]

        bets.append(
            {
                "match_id": str(row.get("id")),
                "date": row.get("date"),
                "side": side,
                "probability": probability,
                "opening_decimal_odds": quoted_odds,
                "decimal_odds": executed_odds,
                "closing_decimal_odds": close_side,
                "expected_roi": expected_roi,
                "opening_overround": opening_market["overround"],
                "closing_overround": closing_overround,
                "clv_price": clv_price,
                "clv_probability": clv_probability,
                "stake_fraction": stake / max(bankroll - pnl, 1e-9),
                "stake": stake,
                "won": won,
                "pnl": pnl,
                "pnl_fraction_start_bankroll": pnl / initial_bankroll,
                "bankroll_after": bankroll,
            }
        )

    wins = sum(1 for bet in bets if bet["won"])
    clv_bets = [bet for bet in bets if bet["clv_price"] is not None]
    average_clv = sum(float(bet["clv_price"]) for bet in clv_bets) / len(clv_bets) if clv_bets else None
    median_clv = statistics.median(float(bet["clv_price"]) for bet in clv_bets) if clv_bets else None
    mean_probability_clv = sum(float(bet["clv_probability"]) for bet in clv_bets) / len(clv_bets) if clv_bets else None
    stake_weighted_clv = (
        sum(float(bet["clv_price"]) * float(bet["stake"]) for bet in clv_bets)
        / sum(float(bet["stake"]) for bet in clv_bets)
        if clv_bets
        else None
    )
    average_vig = sum(float(bet["opening_overround"]) for bet in bets) / len(bets) if bets else None
    return {
        "schema_version": 2,
        "n_bets": len(bets),
        "wins": wins,
        "hit_rate": wins / len(bets) if bets else None,
        "initial_bankroll": initial_bankroll,
        "final_bankroll": bankroll,
        "profit_amount": profit,
        "profit_fraction_start_bankroll": profit / initial_bankroll,
        "total_staked": staked,
        "roi_on_staked": profit / staked if staked else None,
        "max_drawdown": max_drawdown,
        "average_opening_vig": average_vig,
        "clv": {
            "n": len(clv_bets),
            "coverage": len(clv_bets) / len(bets) if bets else 0.0,
            "mean_price_clv": average_clv,
            "median_price_clv": median_clv,
            "mean_probability_clv": mean_probability_clv,
            "stake_weighted_price_clv": stake_weighted_clv,
            "positive_rate": (
                sum(float(bet["clv_price"]) > 0.0 for bet in clv_bets) / len(clv_bets) if clv_bets else None
            ),
        },
        "limits": {
            "constrained_bets": constrained_bets,
            "max_stake_amount": cfg.max_stake_amount,
            "max_profit_amount": cfg.max_profit_amount,
        },
        "skipped": skipped,
        "staking": "model_favorite_only_quarter_kelly_cap_2_5pct",
        "note": (
            "Favorite side is fixed by the model. Opening odds determine EV and stake; "
            "closing odds are audit-only for CLV. Vig, stake and payout limits are explicit."
        ),
        "bets": bets,
    }

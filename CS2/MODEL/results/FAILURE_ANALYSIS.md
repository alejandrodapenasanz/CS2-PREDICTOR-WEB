# Failure analysis - CS2 Predictor

Generated: 2026-07-03T12:25:38Z

## Scientific framing

This report separates observations from hypotheses. A flag co-occurring with a missed prediction is not proof of causality; it is a candidate mechanism to test as more closed matches accumulate.

### Literature anchors

- Walsh & Joshi (2024), Machine learning for sports betting: Calibration/log loss should dominate raw accuracy for betting and Kelly sizing.  
  https://www.sciencedirect.com/science/article/pii/S266682702400015X
- Hegarty & Whelan (2024), testing sports betting market efficiency: Normalized implied probabilities are the correct odds-derived market benchmark.  
  https://www.ucd.ie/economics/t4media/WP24_03.pdf
- Petri et al. (2021), Bandit Modeling of Map Selection in CS:GO: Pick/ban and map selection can move map and match win probability, especially in even matches.  
  https://arxiv.org/abs/2106.08888
- Broms & Nordansjo (2024), Predicting Counter-Strike Matches: HLTV-style player/team variables can predict above 50%, but odds remain a hard benchmark.  
  https://lup.lub.lu.se/student-papers/record/9145457/file/9145459.pdf
- Svec (2022), Predicting Counter-Strike Game Outcomes with ML: Player, roster/team representations and Elo-like baselines are natural CS match features.  
  https://wiki.control.fel.cvut.cz/mediawiki/images/e/e9/P_2022_svec_ondrej.pdf

## Market Blend Policy

- Status: dynamic conservative prior; learned blend disabled until 120 closed point-in-time odds matches.
- Current closed odds matches: 22.

## Closed predictions

| Probability | N | Accuracy | Log loss | Brier | Avg confidence |
|---|---:|---:|---:|---:|---:|
| model | 93 | 0.570 | 0.6678 | 0.2389 | 0.658 |
| risk_adjusted | 93 | 0.570 | 0.6653 | 0.2367 | 0.584 |
| decision | 93 | 0.570 | 0.6628 | 0.2360 | 0.600 |

## Latest snapshot per match

| Probability | N | Accuracy | Log loss | Brier | Avg confidence |
|---|---:|---:|---:|---:|---:|
| model | 42 | 0.548 | 0.7023 | 0.2534 | 0.659 |
| risk_adjusted | 42 | 0.548 | 0.6708 | 0.2398 | 0.588 |
| decision | 42 | 0.548 | 0.6674 | 0.2387 | 0.608 |

## Latest matches with odds only

| Probability | N | Accuracy | Log loss | Brier | Avg confidence |
|---|---:|---:|---:|---:|---:|
| model | 22 | 0.500 | 0.6769 | 0.2432 | 0.633 |
| market | 22 | 0.591 | 0.6899 | 0.2470 | 0.656 |
| decision | 22 | 0.500 | 0.6681 | 0.2389 | 0.628 |

## Where the model misses

| Segment | Group | N | Accuracy | Log loss |
|---|---|---:|---:|---:|
| by_format | bo1 | 11 | 0.545 | 0.6733 |
| by_format | bo3 | 31 | 0.548 | 0.7125 |
| by_confidence | 50-55 | 7 | 0.429 | 0.7080 |
| by_confidence | 55-65 | 15 | 0.533 | 0.7044 |
| by_confidence | 65-80 | 13 | 0.538 | 0.7393 |
| by_confidence | 80+ | 7 | 0.714 | 0.6230 |
| by_reliability | 0-25 | 12 | 0.500 | 0.8084 |
| by_reliability | 45-65 | 5 | 0.800 | 0.6088 |
| by_reliability | 65-85 | 18 | 0.556 | 0.6708 |
| by_reliability | 85-100 | 7 | 0.429 | 0.6680 |
| by_odds | no_odds | 20 | 0.600 | 0.7301 |
| by_odds | odds | 22 | 0.500 | 0.6769 |

## Failure hypotheses from flags

| Hypothesis | N present | Failures | Fail rate | Share of failures |
|---|---:|---:|---:|---:|
| market_sparse_or_missing | 42 | 19 | 0.452 | 1.000 |
| map_veto_pool_uncertain | 42 | 19 | 0.452 | 1.000 |
| roster_player_uncertain | 42 | 19 | 0.452 | 1.000 |
| low_team_history | 42 | 19 | 0.452 | 1.000 |
| incentive_context_unknown | 37 | 17 | 0.459 | 0.895 |
| market_disagreement | 15 | 9 | 0.600 | 0.474 |
| schedule_fatigue | 12 | 7 | 0.583 | 0.368 |
| format_variance | 11 | 5 | 0.455 | 0.263 |
| low_operational_reliability | 8 | 4 | 0.500 | 0.211 |

## High-confidence misses

| Match | Date | Format | Actual | Model | Conf | Rel | Flags |
|---|---|---|---|---|---:|---:|---|
| Spirit HU vs DEFEATERS | 2026-07-02 | bo3 | team2 | team1 | 0.848 | 0.193 | NO_ODDS, LOW_TEAM_HISTORY, CLOSE_ELO, UNKNOWN_EVENT, LOW_PLAYER_STATS, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY |
| BIG Academy vs Wampirki | 2026-06-30 | bo3 | team2 | team1 | 0.802 | 0.193 | NO_ODDS, LOW_TEAM_HISTORY, UNKNOWN_EVENT, LOW_PLAYER_STATS, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| METANOIA Wolves vs VEXA | 2026-07-02 | bo3 | team2 | team1 | 0.735 | 0.782 | NO_ODDS, CLOSE_ELO, UNKNOWN_EVENT, LOW_PLAYER_STATS, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| Wampirki vs Julie&Cie | 2026-07-01 | bo3 | team1 | team2 | 0.725 | 0.193 | NO_ODDS, LOW_TEAM_HISTORY, UNKNOWN_EVENT, LOW_PLAYER_STATS, LOW_MAP_POOL_DATA, STANDIN_RISK, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| 9z vs EYEBALLERS | 2026-07-01 | bo1 | team2 | team1 | 0.717 | 0.820 | LOW_ODDS_SOURCES, ODDS_STRONG_DRIFT, BO1_HIGH_VARIANCE, UNKNOWN_EVENT, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| ALKA vs Red Feet | 2026-06-30 | bo3 | team2 | team1 | 0.693 | 0.468 | LOW_ODDS_SOURCES, LOW_TEAM_HISTORY, CLOSE_ELO, UNKNOWN_EVENT, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| Patins da Ferrari vs GUARA | 2026-07-01 | bo3 | team1 | team2 | 0.655 | 0.193 | LOW_ODDS_SOURCES, LOW_TEAM_HISTORY, UNKNOWN_EVENT, LOW_PLAYER_STATS, LOW_MAP_POOL_DATA, STANDIN_RISK, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN |
| SAW vs Just Players | 2026-07-01 | bo3 | team1 | team2 | 0.652 | 1.000 | LOW_ODDS_SOURCES, UNKNOWN_EVENT, LOW_MAP_POOL_DATA, ROSTER_RECENT_CHANGE, PLAYER_FORM_BACKFILL_ONLY, INCENTIVE_UNKNOWN, FATIGUE_BACK_TO_BACK, HIGH_SCHEDULE_DENSITY |

## Interpretation

- The strongest local warning is calibration under sparse live data: latest closed snapshots have worse log loss than the long walk-forward training metric, so stake sizing should use calibrated/operational probabilities rather than raw accuracy.
- Market odds are a hard benchmark. In the local odds subset, the decision layer (model plus normalized market probability) should be monitored separately from the pure stats model.
- Map pool and veto uncertainty appear in most misses. The literature on CS map selection supports treating map/veto as a first-class variable, but this project still needs more point-in-time map/veto outcomes before training a reliable compositional Bo3 model.
- BO1 and low-roster/player-history cases should remain stake-capped or downgraded; they are not necessarily impossible to predict, but the current evidence is thinner and variance is higher.

## Next tests

- Recompute this report after every scrape/result update; require at least 100-200 closed odds matches before promoting Model B stats+odds as a production model.
- Once analytics/veto coverage has enough closed outcomes, backtest a map-compositional Bo3 model: estimate map win probabilities, simulate likely veto/picks, then aggregate series probability.
- Track error by roster-change recency and player L5/L10 coverage; if the failure rate remains elevated, make roster/player coverage a formal model feature or stronger stake penalty.

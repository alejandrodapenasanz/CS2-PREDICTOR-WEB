# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-13T05:49:11Z
Histórico: 2025-09-11 → 2026-07-12 (9548 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| strength_interactions | OFF | 9548 | 0 | feature_profile_ablation |
| map_box_scores | OFF | 37 | 200 | automatic_at_training_time |
| event_history | ON | 6048 | 200 | automatic_at_training_time |
| analytics | ON | 156 | 120 | automatic_at_training_time |
| player_snapshots | OFF | 119 | 200 | automatic_at_training_time |
| rankings | OFF | 119 | 200 | automatic_at_training_time |
| roster | OFF | 78 | 200 | automatic_at_training_time |
| context | OFF | 121 (LAN 11, online 110) | 200 | automatic_at_training_time |
| opening_odds_model_b | ON | 139 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7112 | 0.5637 | 0.6851 | 0.2460 | 0.4880 | 0.0021 |
| Elo (baseline) | 7112 | 0.6018 | 0.6566 | 0.2324 | 0.6425 | 0.0434 |
| Glicko-2 (baseline) | 7112 | 0.6090 | 0.6589 | 0.2327 | 0.6497 | 0.0492 |
| Logística (Platt) | 7112 | 0.6382 | 0.6352 | 0.2224 | 0.6791 | 0.0102 |
| LightGBM (Platt) | 7112 | 0.6341 | 0.6368 | 0.2232 | 0.6766 | 0.0105 |
| CatBoost (Platt) | 7112 | 0.6382 | 0.6350 | 0.2224 | 0.6797 | 0.0126 |
| XGBoost (Platt) | 7112 | 0.6294 | 0.6412 | 0.2252 | 0.6680 | 0.0158 |
| Random Forest (Platt) | 7112 | 0.6393 | 0.6340 | 0.2218 | 0.6828 | 0.0100 |
| Ensemble LGBM⊕Log (Platt) | 7112 | 0.6398 | 0.6328 | 0.2214 | 0.6829 | 0.0080 |
| Ensemble (isotónica) | 7112 | 0.6398 | 0.6350 | 0.2219 | 0.6816 | 0.0101 |
| Ensemble (beta) | 7112 | 0.6406 | 0.6328 | 0.2214 | 0.6834 | 0.0098 |
| **Ensemble +CatBoost (Platt)** | 7112 | 0.6407 | 0.6326 | 0.2213 | 0.6834 | 0.0109 |
| **Super Learner temporal (Platt)** | 7112 | 0.6419 | 0.6316 | 0.2208 | 0.6862 | 0.0168 |

> Modelo de producción elegido por menor log loss: **super_learner_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| matches_log_diff | 0.1152 |
| format_avg_score_diff | 0.1101 |
| glicko_rd_diff | 0.0916 |
| avg_score_diff | 0.0819 |
| glicko_prob_centered | 0.0726 |
| format_recent_opponent_elo_diff | 0.0510 |
| elo_diff | 0.0444 |
| glicko_diff | 0.0427 |
| recent_opponent_elo_diff | 0.0397 |
| opp_elo_last20_diff | 0.0372 |
| format_last10_score_diff | 0.0369 |
| opp_elo_last5_diff | 0.0365 |
| streak_diff | 0.0357 |
| event_matches_diff | 0.0324 |
| format_h2h_winrate_centered | 0.0296 |

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1471 | 20.7% | 0.519 | 0.692 |
| parejos (55-65%) | 2547 | 35.8% | 0.610 | 0.666 |
| claros (65-80%) | 2665 | 37.5% | 0.708 | 0.597 |
| palizas (>=80%) | 429 | 6.0% | 0.839 | 0.437 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1528 | 2798 | 0.5461 | 0.5485 | -0.0024 |
| 60-70% | 1542 | 2362 | 0.6528 | 0.6488 | +0.0041 |
| 70-80% | 1135 | 1523 | 0.7452 | 0.7442 | +0.0010 |
| 80-90% | 351 | 420 | 0.8357 | 0.8326 | +0.0031 |
| 90-100% | 9 | 9 | 1.0000 | 0.9119 | +0.0881 |
| **Total** | **4565** | **7112** | **0.6419** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 139 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6258 · accuracy 0.6331
- Mercado (odds apertura): log loss 0.5925 · accuracy 0.6978

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

Validación walk-forward solo en partidos con odds de apertura (n_eval=19):

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Mercado apertura | 19 | 0.6316 | 0.5792 | 0.1991 | 0.7440 | 0.1544 |
| Model A en mismos partidos | 19 | 0.7368 | 0.5740 | 0.1952 | 0.7381 | 0.1427 |
| Model B stats + apertura | 19 | 0.6316 | 0.5720 | 0.1953 | 0.7619 | 0.1745 |

Banda de competitividad para Model B:

| Banda | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 2 | 10.5% | 0.000 | 0.752 |
| parejos (55-65%) | 11 | 57.9% | 0.545 | 0.699 |
| claros (65-80%) | 4 | 21.1% | 1.000 | 0.332 |
| palizas (>=80%) | 2 | 10.5% | 1.000 | 0.173 |

SHAP de Model B (top 10):

| Feature | mean |SHAP| |
|---|---:|
| opening_odds_prob_centered | 0.6657 |
| last20_winrate_diff | 0.6169 |
| format_winrate_diff | 0.5527 |
| format_last5_score_diff | 0.3079 |
| elo_diff | 0.3000 |
| format_opp_elo_last5_diff | 0.2841 |
| recent_opponent_elo_diff | 0.2784 |
| format_winrate_trend_5v20_diff | 0.2757 |
| matches_log_diff | 0.2697 |
| glicko_diff | 0.2648 |

> Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.

## Backtest economico

- Apuestas simuladas: 50
- Hit rate: 0.5
- ROI sobre stake: -0.16701175030054966
- Profit sobre banca inicial: -0.15570144849937584
- Max drawdown: 0.2058078987769888
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Siempre apuesta al favorito puro del modelo; las odds de apertura solo filtran EV positivo y dimensionan el stake.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260713_054911Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260713_054911Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- El modelo predice el ganador de la SERIE directamente; la arquitectura composicional Bo3 (P(serie)=P(2-0)+P(2-1)) queda como extensión cuando haya datos por mapa suficientes (PROJECT.md §7.3).

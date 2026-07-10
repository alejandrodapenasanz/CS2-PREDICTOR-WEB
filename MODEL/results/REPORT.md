# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-10T11:17:55Z
Histórico: 2025-09-11 → 2026-07-10 (9492 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| map_box_scores | OFF | 20 | 200 | automatic_at_training_time |
| event_history | ON | 6017 | 200 | automatic_at_training_time |
| analytics | OFF | 100 | 120 | automatic_at_training_time |
| player_snapshots | OFF | 88 | 200 | automatic_at_training_time |
| rankings | OFF | 86 | 200 | automatic_at_training_time |
| roster | OFF | 61 | 200 | automatic_at_training_time |
| context | OFF | 65 (LAN 7, online 58) | 200 | automatic_at_training_time |
| opening_odds_model_b | OFF | 98 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7056 | 0.5632 | 0.6852 | 0.2461 | 0.4874 | 0.0016 |
| Elo (baseline) | 7056 | 0.6027 | 0.6563 | 0.2323 | 0.6430 | 0.0431 |
| Glicko-2 (baseline) | 7056 | 0.6094 | 0.6587 | 0.2326 | 0.6502 | 0.0492 |
| Logística (Platt) | 7056 | 0.6404 | 0.6319 | 0.2210 | 0.6848 | 0.0140 |
| LightGBM (Platt) | 7056 | 0.6404 | 0.6312 | 0.2207 | 0.6871 | 0.0165 |
| CatBoost (Platt) | 7056 | 0.6403 | 0.6302 | 0.2203 | 0.6885 | 0.0228 |
| Ensemble LGBM⊕Log (Platt) | 7056 | 0.6444 | 0.6289 | 0.2196 | 0.6907 | 0.0144 |
| Ensemble (isotónica) | 7056 | 0.6421 | 0.6377 | 0.2204 | 0.6892 | 0.0170 |
| Ensemble (beta) | 7056 | 0.6453 | 0.6287 | 0.2196 | 0.6905 | 0.0147 |
| **Ensemble +CatBoost (Platt)** | 7056 | 0.6441 | 0.6283 | 0.2194 | 0.6917 | 0.0170 |

> Modelo de producción elegido por menor log loss: **ensemble3_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| matches_log_diff | 0.1145 |
| glicko_prob_centered | 0.1046 |
| opp_elo_last20_diff | 0.0918 |
| avg_score_diff | 0.0870 |
| elo_diff | 0.0654 |
| recent_opponent_elo_diff | 0.0591 |
| glicko_rd_diff | 0.0580 |
| format_avg_score_diff | 0.0537 |
| format_last10_score_diff | 0.0352 |
| event_matches_diff | 0.0304 |
| last10_score_diff | 0.0271 |
| format_last5_winrate_diff | 0.0256 |
| glicko_diff | 0.0233 |
| last5_score_diff | 0.0211 |
| format_recent_opponent_elo_diff | 0.0195 |

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1475 | 20.9% | 0.524 | 0.692 |
| parejos (55-65%) | 2589 | 36.7% | 0.607 | 0.667 |
| claros (65-80%) | 2584 | 36.6% | 0.714 | 0.592 |
| palizas (>=80%) | 408 | 5.8% | 0.875 | 0.378 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1545 | 2823 | 0.5473 | 0.5486 | -0.0013 |
| 60-70% | 1516 | 2329 | 0.6509 | 0.6488 | +0.0022 |
| 70-80% | 1127 | 1496 | 0.7533 | 0.7434 | +0.0100 |
| 80-90% | 354 | 404 | 0.8762 | 0.8312 | +0.0450 |
| 90-100% | 3 | 4 | 0.7500 | 0.9112 | -0.1612 |
| **Total** | **4545** | **7056** | **0.6441** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 98 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6279 · accuracy 0.6224
- Mercado (odds apertura): log loss 0.6102 · accuracy 0.7041

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

> Modelo B preparado, pero aun no hay suficientes partidos con odds de apertura para validacion walk-forward fiable. n_odds=98 (mínimo de entrenamiento: 120). No se calcula SHAP de Model B hasta que haya validación walk-forward fiable.

## Backtest economico

- Apuestas simuladas: 39
- Hit rate: 0.5641025641025641
- ROI sobre stake: -0.13864042615769953
- Profit sobre banca inicial: -0.10138573567692788
- Max drawdown: 0.12234876455475097
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Siempre apuesta al favorito puro del modelo; las odds de apertura solo filtran EV positivo y dimensionan el stake.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260710_111755Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260710_111755Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- El modelo predice el ganador de la SERIE directamente; la arquitectura composicional Bo3 (P(serie)=P(2-0)+P(2-1)) queda como extensión cuando haya datos por mapa suficientes (PROJECT.md §7.3).

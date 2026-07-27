# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-26T14:17:35Z
Histórico: 2025-09-11 → 2026-07-25 (9703 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| strength_interactions | OFF | 9703 | 0 | feature_profile_ablation |
| map_box_scores | OFF | 87 | 200 | automatic_at_training_time |
| event_history | ON | 6104 | 200 | automatic_at_training_time |
| analytics | ON | 154 | 120 | automatic_at_training_time |
| analytics_extended | OFF | 127 | 200 | automatic_at_training_time |
| announced_lineups | OFF | 111 | 200 | automatic_at_training_time |
| event_metadata | OFF | 143 | 300 | automatic_at_training_time |
| player_snapshots | OFF | 193 | 200 | automatic_at_training_time |
| rankings | ON | 209 | 200 | automatic_at_training_time |
| roster | OFF | 140 | 200 | automatic_at_training_time |
| mov_rating | ON | 8976 | 800 | automatic_at_training_time |
| team_trueskill | ON | 8976 | 800 | automatic_at_training_time |
| player_rating | OFF | 195 | 200 | automatic_at_training_time |
| strength_of_schedule | ON | 7194 | 800 | automatic_at_training_time |
| bayesian_bradley_terry | ON | 7194 | 800 | automatic_at_training_time |
| kalman_state_space | ON | 7194 | 800 | automatic_at_training_time |
| bo3_map_compositional | OFF | 3 | 500 | automatic_at_training_time |
| regime | ON | 358 | 200 | automatic_at_training_time |
| context | OFF | 276 (LAN 44, online 232) | 200 | automatic_at_training_time |
| feature_pruning | ON | 9703 | 0 | automatic_target_free_exact_redundancy_only |
| optuna_purged_cv | ON | 9703 | 400 | automatic_nested_purged_cv_log_loss |
| rich_target_bo3_scoreline | ON | 8665 | 2000 | automatic_after_walk_forward_multiclass_beats_empirical_baseline |
| opening_odds_model_b | ON | 227 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7267 | 0.5635 | 0.6852 | 0.2460 | 0.4879 | 0.0018 |
| Elo (baseline) | 7267 | 0.6012 | 0.6576 | 0.2328 | 0.6404 | 0.0428 |
| Glicko-2 (baseline) | 7267 | 0.6084 | 0.6595 | 0.2330 | 0.6485 | 0.0492 |
| Bradley-Terry bayesiano (Platt) | 7267 | 0.6067 | 0.6524 | 0.2304 | 0.6505 | 0.0411 |
| Kalman state-space (Platt) | 7267 | 0.6137 | 0.6479 | 0.2284 | 0.6604 | 0.0407 |
| Logística (Platt) | 7267 | 0.6421 | 0.6330 | 0.2213 | 0.6836 | 0.0092 |
| LightGBM (Platt) | 7267 | 0.6392 | 0.6351 | 0.2224 | 0.6792 | 0.0110 |
| CatBoost (Platt) | 7267 | 0.6362 | 0.6356 | 0.2226 | 0.6788 | 0.0112 |
| XGBoost (Platt) | 7267 | 0.6282 | 0.6399 | 0.2247 | 0.6702 | 0.0175 |
| Random Forest (Platt) | 7267 | 0.6407 | 0.6330 | 0.2213 | 0.6844 | 0.0106 |
| Ensemble LGBM⊕Log (Platt) | 7267 | 0.6400 | 0.6311 | 0.2206 | 0.6859 | 0.0090 |
| Ensemble (isotónica) | 7267 | 0.6381 | 0.6372 | 0.2219 | 0.6825 | 0.0113 |
| Ensemble (beta) | 7267 | 0.6419 | 0.6311 | 0.2206 | 0.6864 | 0.0081 |
| **Ensemble +CatBoost (Platt)** | 7267 | 0.6419 | 0.6317 | 0.2209 | 0.6850 | 0.0082 |
| **Super Learner temporal (Platt)** | 7267 | 0.6443 | 0.6300 | 0.2200 | 0.6894 | 0.0120 |

> Modelo de producción elegido por menor log loss: **super_learner_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| trueskill_prob_centered | 0.1630 |
| matches_log_diff | 0.1039 |
| glicko_rd_diff | 0.1007 |
| trueskill_diff | 0.0721 |
| format_avg_score_diff | 0.0601 |
| mov_diff | 0.0560 |
| format_last10_score_diff | 0.0549 |
| format_recent_opponent_elo_diff | 0.0356 |
| format_last5_winrate_diff | 0.0335 |
| format_h2h_winrate_centered | 0.0290 |
| opp_elo_last5_diff | 0.0274 |
| event_matches_diff | 0.0270 |
| streak_diff | 0.0235 |
| recent_opponent_elo_diff | 0.0234 |
| format_last10_winrate_diff | 0.0219 |

## Poda y multicolinealidad (A5)

- Poda automatica sin target: 102 -> 92 columnas.
- Constantes eliminadas: 4.
- Duplicados exactos o de signo opuesto eliminados: 6.
- Diagnostico temporal: VIF + permutation importance en los ultimos 1941 casos + RFE + contraste SHAP.
- Candidatas para revision: analytics_first_ban_pct_diff, analytics_map_played_diff, analytics_insight_score_diff, analytics_common_maps, analytics_available, regime_available, ranking_age_days_max, ranking_hltv_position_advantage, ranking_hltv_points_diff, ranking_valve_position_advantage, ranking_valve_points_diff, ranking_available, ranking_sources, regime_map_pool_size, analytics_first_pick_pct_diff, analytics_map_rows_total, analytics_insight_total.

> Las sugerencias supervisadas son auditoria, no poda automatica: para retirar una feature deben ganar una validacion walk-forward anidada.

## Calibracion por segmento (A3)

| Dimension | Segmento | N | Accuracy | Log loss | Brier | ECE | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| format | bo3 | 6541 | 0.6398 | 0.6334 | 0.2215 | 0.0165 | concluyente |
| format | bo5 | 76 | 0.6579 | 0.6294 | 0.2193 | 0.0657 | concluyente |
| format | bo1 | 650 | 0.6877 | 0.5962 | 0.2047 | 0.0594 | concluyente |
| environment | unknown | 6991 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| environment | lan | 44 | 0.5455 | 0.6993 | 0.2491 | 0.1157 | concluyente |
| environment | online | 232 | 0.6250 | 0.6593 | 0.2327 | 0.0726 | concluyente |
| stage | unknown | 7012 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| stage | group | 109 | 0.5872 | 0.6632 | 0.2355 | 0.0751 | concluyente |
| stage | swiss | 30 | 0.6667 | 0.5665 | 0.1915 | 0.1835 | concluyente |
| stage | quarter | 41 | 0.7805 | 0.5517 | 0.1866 | 0.2285 | concluyente |
| stage | final | 19 | - | - | - | - | muestra <30; no concluyente |
| stage | league | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | semi | 34 | 0.5294 | 0.7927 | 0.2854 | 0.2714 | concluyente |
| stage | ro16 | 16 | - | - | - | - | muestra <30; no concluyente |
| stage | upper_bracket | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | lower_bracket | 2 | - | - | - | - | muestra <30; no concluyente |
| event_tier | unknown | 7267 | - | - | - | - | segmento desconocido; excluido de conclusiones |

Cobertura conocida: format 7267/7267 (100.0%); environment 276/7267 (3.8%); stage 255/7267 (3.5%); event_tier 0/7267 (0.0%).

## Target BO3 enriquecido (A6)

- Estado: ON.
- Cobertura BO3: 8665/2000; minimo por clase: 300.
- Clases: {'0-2': 2184, '1-2': 1642, '2-1': 1787, '2-0': 3052}.
- Scope: scoreline_props_auxiliary_not_winner_model_a.
- Walk-forward: n=6422; log loss multiclase=1.322533; baseline empirico=1.359275; accuracy scoreline=0.410775; accuracy ganador=0.641856.

> auxiliary scoreline model enabled.

## Ratings y optimizacion B8-B11

- B8 Bradley-Terry bayesiano: ON; partial pooling gaussiano, incertidumbre predictiva y candidato calibrado.
- B9 Kalman state-space: ON; deriva de estado, varianza y candidato calibrado.
- B10 Optuna purgado: ON; estudios=2, C final=0.027562018071197132.
- B11 BO3 composicional: OFF; cobertura=3/500.

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1566 | 21.6% | 0.536 | 0.689 |
| parejos (55-65%) | 2727 | 37.5% | 0.602 | 0.669 |
| claros (65-80%) | 2624 | 36.1% | 0.725 | 0.583 |
| palizas (>=80%) | 350 | 4.8% | 0.854 | 0.420 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1673 | 3025 | 0.5531 | 0.5486 | +0.0044 |
| 60-70% | 1563 | 2385 | 0.6553 | 0.6482 | +0.0071 |
| 70-80% | 1147 | 1507 | 0.7611 | 0.7437 | +0.0174 |
| 80-90% | 294 | 345 | 0.8522 | 0.8296 | +0.0225 |
| 90-100% | 5 | 5 | 1.0000 | 0.9060 | +0.0940 |
| **Total** | **4682** | **7267** | **0.6443** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 227 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6474 · accuracy 0.6388
- Mercado (odds apertura): log loss 0.6052 · accuracy 0.7004

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

Validación walk-forward solo en partidos con odds de apertura (n_eval=37):

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Mercado apertura | 37 | 0.6757 | 0.6619 | 0.2283 | 0.6871 | 0.1914 |
| Model A en mismos partidos | 37 | 0.6216 | 0.7269 | 0.2637 | 0.5175 | 0.2768 |
| Model B stats + apertura | 37 | 0.5676 | 0.6796 | 0.2443 | 0.6637 | 0.2068 |

Banda de competitividad para Model B:

| Banda | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| parejos (55-65%) | 22 | 59.5% | 0.500 | 0.731 |
| claros (65-80%) | 14 | 37.8% | 0.643 | 0.634 |
| palizas (>=80%) | 1 | 2.7% | 1.000 | 0.188 |

SHAP de Model B (top 10):

| Feature | mean |SHAP| |
|---|---:|
| opening_odds_prob_centered | 0.7961 |
| last30_winrate_diff | 0.5293 |
| opp_elo_last5_diff | 0.4268 |
| last20_winrate_diff | 0.3765 |
| trueskill_diff | 0.2676 |
| glicko_diff | 0.2158 |
| format_avg_score_diff | 0.2103 |
| format_recent_opponent_elo_diff | 0.2064 |
| format_opp_elo_last5_diff | 0.2036 |
| winrate_diff | 0.1918 |

> Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.

## Backtest economico

- Apuestas simuladas: 82
- Hit rate: 0.43902439024390244
- ROI sobre stake: -0.004392885417115943
- Banca: 1000.0 -> 993.0723287265777
- Profit sobre banca inicial: -0.006927671273422462
- Max drawdown: 0.3100046078607381
- Vig medio de apertura: 0.06698198633143915
- CLV: n=82; cobertura=1.0; media precio=0.015842834047623995; mediana=0.0
- Apuestas limitadas por stake/payout: 0
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Favorite side is fixed by the model. Opening odds determine EV and stake; closing odds are audit-only for CLV. Vig, stake and payout limits are explicit.

## Drift causal (C12)

- Estado: **warning**.
- Predicciones OOS cerradas: 7267.
- Alertas: page_hinkley_log_loss.
- Log loss ventana actual: 0.6762430365222515; referencia: 0.6510079166390257.
- CLV evaluable: 225 (3.1%); media: -0.00030915328211183037.

> Page-Hinkley se aplica al log loss y a -CLV en orden temporal. Las cuotas de cierre solo auditan la ejecucion y nunca son features.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260726_141735Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260726_141735Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- Strength-of-schedule usa Elo del rival y rendimiento real menos esperado, ambos calculados estrictamente antes del partido.
- El modelo A sigue prediciendo el ganador de la serie. El sidecar BO3 estima 0-2/1-2/2-1/2-0 y solo se publica si mejora el baseline multiclase en walk-forward. El modelo por mapa queda pendiente de mas mapstats point-in-time.

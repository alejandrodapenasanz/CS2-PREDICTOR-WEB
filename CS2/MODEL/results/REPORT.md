# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-27T15:34:08Z
Histórico: 2025-09-11 → 2026-07-27 (9725 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| strength_interactions | OFF | 9725 | 0 | feature_profile_ablation |
| map_box_scores | OFF | 112 | 200 | fold_local_temporal_log_loss |
| event_history | ON | 5232 | 200 | fold_local_temporal_log_loss |
| analytics | OFF | 169 | 120 | fold_local_temporal_log_loss |
| analytics_extended | OFF | 157 | 200 | fold_local_temporal_log_loss |
| announced_lineups | OFF | 148 | 200 | fold_local_temporal_log_loss |
| event_metadata | OFF | 158 | 300 | fold_local_temporal_log_loss |
| player_snapshots | OFF | 251 | 200 | fold_local_temporal_log_loss |
| rankings | OFF | 308 | 200 | fold_local_temporal_log_loss |
| roster | OFF | 189 | 200 | fold_local_temporal_log_loss |
| mov_rating | OFF | 8749 | 800 | fold_local_temporal_log_loss |
| team_trueskill | ON | 8749 | 800 | fold_local_temporal_log_loss |
| player_rating | OFF | 237 | 200 | fold_local_temporal_log_loss |
| strength_of_schedule | OFF | 7151 | 800 | fold_local_temporal_log_loss |
| bayesian_bradley_terry | ON | 7151 | 800 | automatic_at_training_time |
| kalman_state_space | ON | 7151 | 800 | automatic_at_training_time |
| bo3_map_compositional | OFF | 5 | 500 | fold_local_temporal_log_loss |
| regime | OFF | 381 | 200 | fold_local_temporal_log_loss |
| context | OFF | 298 (LAN 47, online 251) | 200 | fold_local_temporal_log_loss |
| feature_pruning | OFF | 9725 | 0 | after_fold_local_target_free_exact_redundancy |
| fold_local_selection | ON | 9725 | 80 | nested_outer_train_only |
| optuna_purged_cv | ON | 9725 | 400 | automatic_nested_purged_cv_log_loss |
| rich_target_bo3_scoreline | OFF | 8687 | 2000 | automatic_after_walk_forward_multiclass_beats_empirical_baseline |
| opening_odds_model_b | ON | 237 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7289 | 0.5636 | 0.6851 | 0.2460 | 0.4880 | 0.0019 |
| Elo (baseline) | 7289 | 0.5988 | 0.6582 | 0.2332 | 0.6387 | 0.0437 |
| Glicko-2 (baseline) | 7289 | 0.6087 | 0.6594 | 0.2329 | 0.6491 | 0.0490 |
| Bradley-Terry bayesiano (Platt) | 7289 | 0.6050 | 0.6537 | 0.2311 | 0.6481 | 0.0414 |
| Kalman state-space (Platt) | 7289 | 0.6109 | 0.6495 | 0.2291 | 0.6570 | 0.0410 |
| Logística (Platt) | 7289 | 0.6449 | 0.6317 | 0.2207 | 0.6873 | 0.0126 |
| LightGBM (Platt) | 7289 | 0.6323 | 0.6372 | 0.2234 | 0.6768 | 0.0157 |
| CatBoost (Platt) | 7289 | 0.6349 | 0.6346 | 0.2222 | 0.6809 | 0.0145 |
| XGBoost (Platt) | 7289 | 0.6297 | 0.6402 | 0.2247 | 0.6712 | 0.0112 |
| Random Forest (Platt) | 7289 | 0.6407 | 0.6336 | 0.2216 | 0.6843 | 0.0121 |
| Ensemble LGBM⊕Log (Platt) | 7289 | 0.6414 | 0.6317 | 0.2209 | 0.6862 | 0.0182 |
| Ensemble (isotónica) | 7289 | 0.6369 | 0.6404 | 0.2224 | 0.6825 | 0.0133 |
| Ensemble (beta) | 7289 | 0.6396 | 0.6314 | 0.2207 | 0.6865 | 0.0125 |
| **Ensemble +CatBoost (Platt)** | 7289 | 0.6381 | 0.6318 | 0.2209 | 0.6859 | 0.0159 |
| **Super Learner temporal (Platt)** | 7289 | 0.6445 | 0.6301 | 0.2200 | 0.6904 | 0.0161 |
| **Politica causal de seleccion (L1)** | 7289 | 0.6438 | 0.6309 | 0.2204 | 0.6885 | 0.0162 |

> Metrica primaria sin sesgo de seleccion: **nested_model_policy**. Candidato ajustado para el siguiente periodo: **super_learner_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| trueskill_prob_centered | 0.1572 |
| matches_log_diff | 0.1029 |
| glicko_rd_diff | 0.0855 |
| trueskill_diff | 0.0764 |
| format_avg_score_diff | 0.0656 |
| recent_opponent_elo_diff | 0.0533 |
| format_last10_score_diff | 0.0476 |
| opp_elo_last20_diff | 0.0420 |
| recency_advantage | 0.0397 |
| last10_score_diff | 0.0392 |
| opp_elo_last5_diff | 0.0340 |
| format_last10_winrate_diff | 0.0297 |
| format_last5_winrate_diff | 0.0297 |
| avg_score_diff | 0.0224 |
| format_last5_score_diff | 0.0220 |

## Poda y multicolinealidad (A5)

- Poda automatica sin target: 60 -> 60 columnas.
- Constantes eliminadas: 0.
- Duplicados exactos o de signo opuesto eliminados: 0.
- Diagnostico temporal: VIF + permutation importance en los ultimos 1945 casos + RFE + contraste SHAP.
- Candidatas para revision: context_stage_other, regime_map_pool_size, regime_map_pool_available, regime_patch_known, regime_available, regime_patch_recent, context_available, context_environment_known, context_is_lan, context_is_online, context_stage_known, bo3_map_sample_min, regime_patch_age_log_days, bo3_veto_confidence, bo3_compositional_prob_centered, bo3_compositional_available, bo3_map_probability_spread, context_stage_group, context_stage_qualifier, context_placement_match, context_winners_match, context_opening_match, context_stage_swiss, context_stage_semi, context_stage_quarter, context_stage_final, context_stage_ro16, context_stage_bracket.

> Las sugerencias supervisadas son auditoria, no poda automatica: para retirar una feature deben ganar una validacion walk-forward anidada.

## Calibracion por segmento (A3)

| Dimension | Segmento | N | Accuracy | Log loss | Brier | ECE | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| format | bo3 | 6563 | 0.6412 | 0.6343 | 0.2219 | 0.0160 | concluyente |
| format | bo5 | 76 | 0.6711 | 0.6126 | 0.2118 | 0.0814 | concluyente |
| format | bo1 | 650 | 0.6677 | 0.5987 | 0.2059 | 0.0615 | concluyente |
| environment | unknown | 6991 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| environment | lan | 47 | 0.5957 | 0.6175 | 0.2161 | 0.1292 | concluyente |
| environment | online | 251 | 0.6335 | 0.6456 | 0.2269 | 0.0435 | concluyente |
| stage | unknown | 7014 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| stage | group | 112 | 0.6429 | 0.6247 | 0.2184 | 0.0641 | concluyente |
| stage | swiss | 42 | 0.6429 | 0.6237 | 0.2175 | 0.1506 | concluyente |
| stage | quarter | 41 | 0.7073 | 0.5727 | 0.1963 | 0.1050 | concluyente |
| stage | final | 19 | - | - | - | - | muestra <30; no concluyente |
| stage | league | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | semi | 37 | 0.5676 | 0.6422 | 0.2257 | 0.1023 | concluyente |
| stage | ro16 | 18 | - | - | - | - | muestra <30; no concluyente |
| stage | upper_bracket | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | lower_bracket | 2 | - | - | - | - | muestra <30; no concluyente |
| event_tier | unknown | 7289 | - | - | - | - | segmento desconocido; excluido de conclusiones |

Cobertura conocida: format 7289/7289 (100.0%); environment 298/7289 (4.1%); stage 275/7289 (3.8%); event_tier 0/7289 (0.0%).

## Target BO3 enriquecido (A6)

- Estado: OFF.
- Cobertura BO3: 8687/2000; minimo por clase: 300.
- Clases: {'0-2': 2189, '1-2': 1646, '2-1': 1792, '2-0': 3060}.
- Scope: scoreline_props_auxiliary_not_winner_model_a.
- Walk-forward: n=6444; log loss multiclase=1.481695; baseline empirico=1.359258; accuracy scoreline=0.412477; accuracy ganador=0.644941.

> did not beat empirical scoreline baseline.

## Ratings y optimizacion B8-B11

- B8 Bradley-Terry bayesiano: ON; partial pooling gaussiano, incertidumbre predictiva y candidato calibrado.
- B9 Kalman state-space: ON; deriva de estado, varianza y candidato calibrado.
- B10 Optuna purgado: ON; estudios=2, C final=0.027562018071197132.
- B11 BO3 composicional: OFF; cobertura=5/500.

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1527 | 20.9% | 0.532 | 0.691 |
| parejos (55-65%) | 2693 | 37.0% | 0.594 | 0.675 |
| claros (65-80%) | 2610 | 35.8% | 0.728 | 0.581 |
| palizas (>=80%) | 459 | 6.3% | 0.828 | 0.458 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1640 | 2935 | 0.5588 | 0.5481 | +0.0107 |
| 60-70% | 1504 | 2362 | 0.6367 | 0.6470 | -0.0103 |
| 70-80% | 1169 | 1533 | 0.7626 | 0.7432 | +0.0194 |
| 80-90% | 369 | 447 | 0.8255 | 0.8330 | -0.0075 |
| 90-100% | 11 | 12 | 0.9167 | 0.9122 | +0.0045 |
| **Total** | **4693** | **7289** | **0.6438** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 237 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.643 · accuracy 0.6371
- Mercado (odds apertura): log loss 0.6107 · accuracy 0.6835

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

Validación walk-forward solo en partidos con odds de apertura (n_eval=47):

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Mercado apertura | 47 | 0.5957 | 0.6777 | 0.2378 | 0.6440 | 0.1797 |
| Model A en mismos partidos | 47 | 0.5745 | 0.7527 | 0.2727 | 0.5435 | 0.1791 |
| Model B stats + apertura | 47 | 0.6170 | 0.6688 | 0.2354 | 0.6721 | 0.1533 |

Banda de competitividad para Model B:

| Banda | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 9 | 19.1% | 0.333 | 0.708 |
| parejos (55-65%) | 16 | 34.0% | 0.750 | 0.597 |
| claros (65-80%) | 12 | 25.5% | 0.667 | 0.632 |
| palizas (>=80%) | 10 | 21.3% | 0.600 | 0.793 |

SHAP de Model B (top 10):

| Feature | mean |SHAP| |
|---|---:|
| opening_odds_prob_centered | 0.6304 |
| last30_winrate_diff | 0.4311 |
| opp_elo_last20_diff | 0.3917 |
| last20_winrate_diff | 0.3874 |
| recent_opponent_elo_diff | 0.3476 |
| format_last10_score_diff | 0.2785 |
| winrate_decay_diff | 0.2559 |
| opp_elo_last5_diff | 0.2425 |
| trueskill_diff | 0.2132 |
| trueskill_prob_centered | 0.2122 |

> Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.

## Backtest economico

- Apuestas simuladas: 70
- Hit rate: 0.4714285714285714
- ROI sobre stake: -0.14768598649431347
- Banca: 1000.0 -> 794.0513063023708
- Profit sobre banca inicial: -0.20594869369762922
- Max drawdown: 0.20678539111144315
- Vig medio de apertura: 0.06797820913443899
- CLV: n=33; cobertura=0.4714285714285714; media precio=-0.002796091995902896; mediana=0.0
- Apuestas limitadas por stake/payout: 0
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Favorite side is fixed by the model. Opening odds determine EV and stake; closing odds are audit-only for CLV. Vig, stake and payout limits are explicit.

## Drift causal (C12)

- Estado: **warning**.
- Predicciones OOS cerradas: 7289.
- Alertas: page_hinkley_log_loss.
- Log loss ventana actual: 0.655269555369587; referencia: 0.6447524737449137.
- CLV evaluable: 94 (1.3%); media: -0.002702300529502968.

> Page-Hinkley se aplica al log loss y a -CLV en orden temporal. Las cuotas de cierre solo auditan la ejecucion y nunca son features.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\CS2\MODEL\artifacts\registry\20260727_153408Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\CS2\MODEL\artifacts\registry\20260727_153408Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- Strength-of-schedule usa Elo del rival y rendimiento real menos esperado, ambos calculados estrictamente antes del partido.
- El modelo A sigue prediciendo el ganador de la serie. El sidecar BO3 estima 0-2/1-2/2-1/2-0 y solo se publica si mejora el baseline multiclase en walk-forward. El modelo por mapa queda pendiente de mas mapstats point-in-time.

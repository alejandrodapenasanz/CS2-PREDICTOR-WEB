# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-08-10T06:48:27Z
Histórico: 2025-09-11 → 2026-08-10 (10086 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| strength_interactions | OFF | 10086 | 0 | feature_profile_ablation |
| map_box_scores | OFF | 310 | 200 | fold_local_temporal_log_loss |
| event_history | ON | 5437 | 200 | fold_local_temporal_log_loss |
| analytics | OFF | 512 | 120 | fold_local_temporal_log_loss |
| analytics_extended | OFF | 500 | 200 | fold_local_temporal_log_loss |
| announced_lineups | OFF | 436 | 200 | fold_local_temporal_log_loss |
| event_metadata | OFF | 501 | 300 | fold_local_temporal_log_loss |
| player_snapshots | OFF | 452 | 200 | fold_local_temporal_log_loss |
| rankings | OFF | 520 | 200 | fold_local_temporal_log_loss |
| roster | OFF | 347 | 200 | fold_local_temporal_log_loss |
| mov_rating | OFF | 9065 | 800 | fold_local_temporal_log_loss |
| team_trueskill | ON | 9065 | 800 | fold_local_temporal_log_loss |
| player_rating | OFF | 525 | 200 | fold_local_temporal_log_loss |
| strength_of_schedule | OFF | 7402 | 800 | fold_local_temporal_log_loss |
| bayesian_bradley_terry | ON | 7402 | 800 | automatic_at_training_time |
| kalman_state_space | ON | 7402 | 800 | automatic_at_training_time |
| bo3_map_compositional | OFF | 22 | 500 | fold_local_temporal_log_loss |
| regime | OFF | 742 | 200 | fold_local_temporal_log_loss |
| context | OFF | 659 (LAN 201, online 458) | 200 | fold_local_temporal_log_loss |
| feature_pruning | OFF | 10086 | 0 | after_fold_local_target_free_exact_redundancy |
| fold_local_selection | ON | 10086 | 80 | nested_outer_train_only |
| optuna_purged_cv | OFF | 10086 | 400 | automatic_nested_purged_cv_log_loss |
| rich_target_bo3_scoreline | OFF | 9013 | 2000 | automatic_after_walk_forward_multiclass_beats_empirical_baseline |
| opening_odds_model_b | ON | 497 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7650 | 0.5651 | 0.6848 | 0.2458 | 0.4911 | 0.0033 |
| Elo (baseline) | 7650 | 0.5996 | 0.6575 | 0.2328 | 0.6395 | 0.0444 |
| Glicko-2 (baseline) | 7650 | 0.6108 | 0.6588 | 0.2326 | 0.6496 | 0.0491 |
| Bradley-Terry bayesiano (Platt) | 7650 | 0.6068 | 0.6530 | 0.2307 | 0.6492 | 0.0422 |
| Kalman state-space (Platt) | 7650 | 0.6125 | 0.6484 | 0.2286 | 0.6586 | 0.0417 |
| Logística (Platt) | 7650 | 0.6414 | 0.6314 | 0.2205 | 0.6872 | 0.0132 |
| LightGBM (Platt) | 7650 | 0.6337 | 0.6352 | 0.2224 | 0.6795 | 0.0145 |
| Random Forest (Platt) | 7650 | 0.6426 | 0.6317 | 0.2207 | 0.6866 | 0.0103 |
| Ensemble LGBM⊕Log (Platt) | 7650 | 0.6427 | 0.6301 | 0.2201 | 0.6882 | 0.0138 |
| Ensemble (isotónica) | 7650 | 0.6417 | 0.6376 | 0.2214 | 0.6853 | 0.0141 |
| Ensemble (beta) | 7650 | 0.6417 | 0.6300 | 0.2201 | 0.6884 | 0.0104 |
| **Super Learner temporal (Platt)** | 7650 | 0.6459 | 0.6288 | 0.2194 | 0.6922 | 0.0179 |
| **Politica causal de seleccion (L1)** | 7650 | 0.6443 | 0.6296 | 0.2198 | 0.6903 | 0.0133 |

> Metrica primaria sin sesgo de seleccion: **nested_model_policy**. Candidato ajustado para el siguiente periodo: **super_learner_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| trueskill_prob_centered | 0.1773 |
| matches_log_diff | 0.0970 |
| glicko_rd_diff | 0.0827 |
| trueskill_diff | 0.0741 |
| opp_elo_last20_diff | 0.0558 |
| format_avg_score_diff | 0.0515 |
| last10_score_diff | 0.0503 |
| recent_opponent_elo_diff | 0.0421 |
| recency_advantage | 0.0411 |
| format_last10_score_diff | 0.0378 |
| format_last5_winrate_diff | 0.0326 |
| opp_elo_last5_diff | 0.0322 |
| format_last10_winrate_diff | 0.0219 |
| format_recent_opponent_elo_diff | 0.0215 |
| format_last5_score_diff | 0.0197 |

## Poda y multicolinealidad (A5)

- Poda automatica sin target: 60 -> 60 columnas.
- Constantes eliminadas: 0.
- Duplicados exactos o de signo opuesto eliminados: 0.
- Diagnostico temporal: VIF + permutation importance en los ultimos 2018 casos + RFE + contraste SHAP.
- Candidatas para revision: context_stage_group, regime_map_pool_size, regime_patch_known, regime_available, regime_patch_recent, context_available, context_environment_known, regime_patch_age_log_days, bo3_compositional_prob_centered, bo3_veto_confidence, regime_map_pool_available, bo3_compositional_available, bo3_map_probability_spread, bo3_map_sample_min, context_stage_bracket, context_stage_qualifier, context_placement_match, context_winners_match, context_opening_match, context_stage_swiss, context_stage_semi, context_stage_quarter, context_stage_ro16, context_is_lan, context_is_online, context_stage_known, context_stage_other, context_stage_final.

> Las sugerencias supervisadas son auditoria, no poda automatica: para retirar una feature deben ganar una validacion walk-forward anidada.

## Calibracion por segmento (A3)

| Dimension | Segmento | N | Accuracy | Log loss | Brier | ECE | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| format | bo3 | 6899 | 0.6424 | 0.6322 | 0.2210 | 0.0150 | concluyente |
| format | bo5 | 77 | 0.6364 | 0.6230 | 0.2169 | 0.0644 | concluyente |
| format | bo1 | 674 | 0.6647 | 0.6036 | 0.2078 | 0.0602 | concluyente |
| environment | unknown | 6991 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| environment | lan | 201 | 0.7015 | 0.5461 | 0.1831 | 0.1020 | concluyente |
| environment | online | 458 | 0.6376 | 0.6447 | 0.2265 | 0.0419 | concluyente |
| stage | unknown | 7030 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| stage | group | 251 | 0.7052 | 0.5678 | 0.1924 | 0.0731 | concluyente |
| stage | swiss | 125 | 0.6240 | 0.6345 | 0.2225 | 0.1190 | concluyente |
| stage | quarter | 83 | 0.6867 | 0.6116 | 0.2121 | 0.1064 | concluyente |
| stage | final | 30 | 0.6000 | 0.6800 | 0.2435 | 0.1062 | concluyente |
| stage | league | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | semi | 69 | 0.6522 | 0.6232 | 0.2166 | 0.0701 | concluyente |
| stage | ro16 | 50 | 0.6200 | 0.6493 | 0.2288 | 0.0759 | concluyente |
| stage | upper_bracket | 2 | - | - | - | - | muestra <30; no concluyente |
| stage | lower_bracket | 8 | - | - | - | - | muestra <30; no concluyente |
| event_tier | unknown | 7650 | - | - | - | - | segmento desconocido; excluido de conclusiones |

Cobertura conocida: format 7650/7650 (100.0%); environment 659/7650 (8.6%); stage 620/7650 (8.1%); event_tier 0/7650 (0.0%).

## Target BO3 enriquecido (A6)

- Estado: OFF.
- Cobertura BO3: 9013/2000; minimo por clase: 300.
- Clases: {'0-2': 2264, '1-2': 1703, '2-1': 1855, '2-0': 3191}.
- Scope: scoreline_props_auxiliary_not_winner_model_a.
- Walk-forward: n=6770; log loss multiclase=1.501936; baseline empirico=1.357962; accuracy scoreline=0.410635; accuracy ganador=0.643131.

> did not beat empirical scoreline baseline.

## Ratings y optimizacion B8-B11

- B8 Bradley-Terry bayesiano: ON; partial pooling gaussiano, incertidumbre predictiva y candidato calibrado.
- B9 Kalman state-space: ON; deriva de estado, varianza y candidato calibrado.
- B10 Optuna purgado: OFF; estudios=2, C final=0.5.
- B11 BO3 composicional: OFF; cobertura=22/500.

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1619 | 21.2% | 0.521 | 0.692 |
| parejos (55-65%) | 2813 | 36.8% | 0.604 | 0.670 |
| claros (65-80%) | 2760 | 36.1% | 0.728 | 0.580 |
| palizas (>=80%) | 458 | 6.0% | 0.828 | 0.458 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1715 | 3107 | 0.5520 | 0.5481 | +0.0038 |
| 60-70% | 1595 | 2470 | 0.6457 | 0.6474 | -0.0017 |
| 70-80% | 1240 | 1615 | 0.7678 | 0.7439 | +0.0239 |
| 80-90% | 371 | 448 | 0.8281 | 0.8325 | -0.0043 |
| 90-100% | 8 | 10 | 0.8000 | 0.9148 | -0.1148 |
| **Total** | **4929** | **7650** | **0.6443** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 497 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6156 · accuracy 0.668
- Mercado (odds apertura): log loss 0.5786 · accuracy 0.6982

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

Validación walk-forward solo en partidos con odds de apertura (n_eval=307):

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Mercado apertura | 307 | 0.6938 | 0.5690 | 0.1932 | 0.7708 | 0.0358 |
| Model A en mismos partidos | 307 | 0.6743 | 0.6156 | 0.2132 | 0.7100 | 0.0711 |
| Model B stats + apertura | 307 | 0.6645 | 0.6460 | 0.2239 | 0.6819 | 0.0699 |

Banda de competitividad para Model B:

| Banda | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 67 | 21.8% | 0.492 | 0.697 |
| parejos (55-65%) | 116 | 37.8% | 0.733 | 0.618 |
| claros (65-80%) | 90 | 29.3% | 0.711 | 0.583 |
| palizas (>=80%) | 34 | 11.1% | 0.647 | 0.810 |

SHAP de Model B (top 10):

| Feature | mean |SHAP| |
|---|---:|
| opening_odds_prob_centered | 0.9470 |
| format_recent_opponent_elo_diff | 0.2866 |
| last30_winrate_diff | 0.2842 |
| recent_opponent_elo_diff | 0.2825 |
| last20_winrate_diff | 0.2549 |
| recency_advantage | 0.2061 |
| opp_elo_last20_diff | 0.1991 |
| format_opp_elo_last5_diff | 0.1853 |
| last10_score_diff | 0.1827 |
| trueskill_diff | 0.1693 |

> Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.

## Backtest economico

- Apuestas simuladas: 158
- Hit rate: 0.4873417721518987
- ROI sobre stake: -0.15732019141022982
- Banca: 1000.0 -> 598.194533179908
- Profit sobre banca inicial: -0.40180546682009227
- Max drawdown: 0.4692623983879125
- Vig medio de apertura: 0.07209540804972042
- CLV: n=70; cobertura=0.4430379746835443; media precio=0.005505843187910872; mediana=0.0
- Apuestas limitadas por stake/payout: 0
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Favorite side is fixed by the model. Opening odds determine EV and stake; closing odds are audit-only for CLV. Vig, stake and payout limits are explicit.

## Drift causal (C12)

- Estado: **warning**.
- Predicciones OOS cerradas: 7650.
- Alertas: page_hinkley_log_loss.
- Log loss ventana actual: 0.5598127148717995; referencia: 0.6384216543190163.
- CLV evaluable: 195 (2.5%); media: 0.0003179987498970114.

> Page-Hinkley se aplica al log loss y a -CLV en orden temporal. Las cuotas de cierre solo auditan la ejecucion y nunca son features.

## Model registry

- Artefacto versionado: `C:\dev\CS2-Predictor\CS2\MODEL\artifacts\registry\20260810_064827Z\model.pkl`
- Carpeta: `C:\dev\CS2-Predictor\CS2\MODEL\artifacts\registry\20260810_064827Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- Strength-of-schedule usa Elo del rival y rendimiento real menos esperado, ambos calculados estrictamente antes del partido.
- El modelo A sigue prediciendo el ganador de la serie. El sidecar BO3 estima 0-2/1-2/2-1/2-0 y solo se publica si mejora el baseline multiclase en walk-forward. El modelo por mapa queda pendiente de mas mapstats point-in-time.

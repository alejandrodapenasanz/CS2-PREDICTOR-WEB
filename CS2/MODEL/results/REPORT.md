# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-09-11T08:12:20Z
Histórico: 2025-09-11 → 2026-09-11 (11014 series, era CS2)

## Politica de features opcionales

| Familia | Estado | Cobertura | Umbral | Activacion |
|---|---:|---:|---:|---|
| strength_interactions | OFF | 11014 | 0 | feature_profile_ablation |
| map_box_scores | OFF | 315 | 200 | fold_local_temporal_log_loss |
| event_history | ON | 5441 | 200 | fold_local_temporal_log_loss |
| analytics | OFF | 516 | 120 | fold_local_temporal_log_loss |
| analytics_extended | OFF | 504 | 200 | fold_local_temporal_log_loss |
| announced_lineups | OFF | 440 | 200 | fold_local_temporal_log_loss |
| event_metadata | OFF | 505 | 300 | fold_local_temporal_log_loss |
| player_snapshots | OFF | 454 | 200 | fold_local_temporal_log_loss |
| rankings | OFF | 523 | 200 | fold_local_temporal_log_loss |
| roster | OFF | 350 | 200 | fold_local_temporal_log_loss |
| mov_rating | OFF | 9069 | 800 | fold_local_temporal_log_loss |
| team_trueskill | ON | 9069 | 800 | fold_local_temporal_log_loss |
| player_rating | OFF | 529 | 200 | fold_local_temporal_log_loss |
| strength_of_schedule | OFF | 7405 | 800 | fold_local_temporal_log_loss |
| bayesian_bradley_terry | ON | 8175 | 800 | automatic_at_training_time |
| kalman_state_space | ON | 8175 | 800 | automatic_at_training_time |
| bo3_map_compositional | OFF | 22 | 500 | fold_local_temporal_log_loss |
| regime | OFF | 746 | 200 | fold_local_temporal_log_loss |
| context | OFF | 663 (LAN 201, online 462) | 200 | fold_local_temporal_log_loss |
| segment_interactions | OFF | 11014 | 800 | explicit_experiment_then_fold_local_temporal_log_loss |
| roster_sensitive_rating | ON | 1278 | 0 | standalone_challenger_through_temporal_gate |
| feature_pruning | OFF | 11014 | 0 | after_fold_local_target_free_exact_redundancy |
| fold_local_selection | ON | 11014 | 80 | nested_outer_train_only |
| optuna_purged_cv | ON | 11014 | 400 | automatic_nested_purged_cv_log_loss |
| rich_target_bo3_scoreline | OFF | 9832 | 2000 | automatic_after_walk_forward_multiclass_beats_empirical_baseline |
| opening_odds_model_b | ON | 1111 | 120 | automatic_separate_model_b_evaluation |

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 8578 | 0.5625 | 0.6854 | 0.2461 | 0.4868 | 0.0005 |
| Elo (baseline) | 8578 | 0.6020 | 0.6564 | 0.2323 | 0.6428 | 0.0416 |
| Glicko-2 (baseline) | 8578 | 0.6123 | 0.6584 | 0.2325 | 0.6503 | 0.0472 |
| Glicko-2 actual (Platt, control) | 8578 | 0.6123 | 0.6549 | 0.2315 | 0.6510 | 0.0437 |
| Glicko-2 sensible al roster (sin calibrar) | 8578 | 0.6116 | 0.6591 | 0.2328 | 0.6494 | 0.0474 |
| Glicko-2 sensible al roster (Platt) | 8578 | 0.6116 | 0.6553 | 0.2317 | 0.6501 | 0.0436 |
| Bradley-Terry bayesiano (Platt) | 8578 | 0.6082 | 0.6526 | 0.2305 | 0.6509 | 0.0394 |
| Kalman state-space (Platt) | 8578 | 0.6131 | 0.6480 | 0.2284 | 0.6598 | 0.0390 |
| Logística (Platt) | 8578 | 0.6437 | 0.6302 | 0.2201 | 0.6890 | 0.0119 |
| LightGBM (Platt) | 8578 | 0.6341 | 0.6352 | 0.2225 | 0.6796 | 0.0112 |
| CatBoost (Platt) | 8578 | 0.6353 | 0.6339 | 0.2219 | 0.6821 | 0.0130 |
| XGBoost (Platt) | 8578 | 0.6317 | 0.6383 | 0.2239 | 0.6744 | 0.0104 |
| Random Forest (Platt) | 8578 | 0.6391 | 0.6329 | 0.2213 | 0.6850 | 0.0133 |
| Ensemble LGBM⊕Log (Platt) | 8578 | 0.6409 | 0.6302 | 0.2202 | 0.6885 | 0.0125 |
| Ensemble (isotónica) | 8578 | 0.6374 | 0.6358 | 0.2213 | 0.6860 | 0.0124 |
| Ensemble (beta) | 8578 | 0.6390 | 0.6302 | 0.2202 | 0.6885 | 0.0091 |
| **Ensemble +CatBoost (Platt)** | 8578 | 0.6386 | 0.6305 | 0.2204 | 0.6877 | 0.0134 |
| **Super Learner temporal (Platt)** | 8578 | 0.6440 | 0.6290 | 0.2195 | 0.6918 | 0.0168 |
| **Politica causal de seleccion (L1)** | 8578 | 0.6418 | 0.6296 | 0.2198 | 0.6904 | 0.0114 |

> Metrica primaria sin sesgo de seleccion: **nested_model_policy**. Candidato ajustado para el siguiente periodo: **super_learner_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Challenger de rating sensible al roster

> Beneficio esperado: menos palos gordos y mejor calibracion/robustez tras reconstrucciones profundas; no se espera un salto grande de accuracy media. El rating actual se conserva y el challenger solo puede entrar por seleccion temporal y la puerta champion/challenger.

| Variante | N | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Glicko actual | 8578 | 0.6123 | 0.6584 | 0.2325 | 0.0472 |
| Glicko actual calibrado (control) | 8578 | 0.6123 | 0.6549 | 0.2315 | 0.0437 |
| Glicko-roster | 8578 | 0.6116 | 0.6591 | 0.2328 | 0.0474 |
| Glicko-roster calibrado | 8578 | 0.6116 | 0.6553 | 0.2317 | 0.0436 |

Subgrupo descriptivo de reconstrucciones (N pequeño; no decide):

| Variante | N | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Glicko actual calibrado | 47 | 0.5745 | 0.6296 | 0.2208 | 0.1254 |
| Glicko-roster calibrado | 47 | 0.5319 | 0.6517 | 0.2303 | 0.1382 |

- Filas OOS con cambio de nucleo: 47.
- Eleccion causal interna: `super_learner_cal`.
- Puerta externa: `reject`; promociona=False.

Ejemplos de reconstruccion con mayor cambio de probabilidad:

| Fecha | Partido | Retenidos A/B | P actual | P roster | Delta |
|---|---|---:|---:|---:|---:|
| 2026-08-30 | Sangal vs PURE | 5/0 | 0.8032 | 0.6036 | -0.1996 |
| 2026-09-10 | BESTIA vs Virtus.pro | 3/5 | 0.5131 | 0.3318 | -0.1813 |
| 2026-09-06 | Mai Tai vs NAVI Junior | 1/5 | 0.3521 | 0.5203 | +0.1682 |
| 2026-07-21 | Honvéd vs Julie&Cie | 0/2 | 0.8080 | 0.6424 | -0.1655 |
| 2026-08-03 | 1win vs ex-RUBY | 0/5 | 0.5271 | 0.3831 | -0.1441 |

## Puerta de promocion

- Estado: **reject**; promueve: **False**.
- Holdout comun: n=924, cutoff estrictamente posterior a 2026-08-10.
- Resultado: PROMOTION REJECT: n=924 cutoff=>2026-08-10 log_loss incumbent=0.632855 challenger=0.649382 delta=+0.016527; brier incumbent=0.221526 challenger=0.226536 delta=+0.005010; reason=log_loss_regressed
- Periodo: 2026-08-11 -> 2026-09-11; con odds=611, sin odds=313.

| Regimen | N | Vivo log loss | Challenger log loss | Vivo Brier | Challenger Brier | Challenger ECE |
|---|---:|---:|---:|---:|---:|---:|
| odds | 611 | 0.6427 | 0.6689 | 0.2261 | 0.2342 | 0.1180 |
| no_odds | 313 | 0.6137 | 0.6114 | 0.2125 | 0.2115 | 0.0355 |

- Consistencia mismo shadow/mismas filas: pass; atol=1e-12; delta max P=2.220446049250313e-16.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| trueskill_prob_centered | 0.1826 |
| opp_elo_last20_diff | 0.0947 |
| matches_log_diff | 0.0943 |
| glicko_rd_diff | 0.0813 |
| trueskill_diff | 0.0730 |
| format_last5_winrate_diff | 0.0681 |
| last10_score_diff | 0.0595 |
| opp_elo_last5_diff | 0.0492 |
| recency_advantage | 0.0490 |
| recent_opponent_elo_diff | 0.0460 |
| format_last10_score_diff | 0.0459 |
| format_last5_score_diff | 0.0384 |
| format_avg_score_diff | 0.0329 |
| format_last10_winrate_diff | 0.0297 |
| event_matches_diff | 0.0275 |

## Poda y multicolinealidad (A5)

- Poda automatica sin target: 61 -> 61 columnas.
- Constantes eliminadas: 0.
- Duplicados exactos o de signo opuesto eliminados: 0.
- Diagnostico temporal: VIF + permutation importance en los ultimos 2203 casos + RFE + contraste SHAP.
- Candidatas para revision: context_stage_ro16, context_stage_other, regime_map_pool_available, regime_patch_known, regime_available, regime_patch_recent, context_available, context_environment_known, bo3_map_sample_min, regime_patch_age_log_days, bo3_veto_confidence, bo3_compositional_prob_centered, bo3_compositional_available, bo3_map_probability_spread, context_stage_group, regime_map_pool_size, context_placement_match, context_winners_match, context_opening_match, context_stage_swiss, context_stage_semi, context_stage_quarter, context_is_lan, context_is_online, context_stage_known, context_stage_bracket, context_stage_qualifier, context_stage_final.

> Las sugerencias supervisadas son auditoria, no poda automatica: para retirar una feature deben ganar una validacion walk-forward anidada.

## Calibracion por segmento (A3)

| Dimension | Segmento | N | Prob. media | Tasa real | Gap | ECE | Estado |
|---|---|---:|---:|---:|---:|---:|---|
| elo_gap | 0-50 | 3680 | 0.5428 | 0.5353 | -0.0075 | 0.0140 | compatible_with_sampling_noise |
| elo_gap | 50-100 | 2500 | 0.5568 | 0.5500 | -0.0068 | 0.0161 | compatible_with_sampling_noise |
| elo_gap | 100-200 | 2060 | 0.5825 | 0.6107 | +0.0282 | 0.0307 | compatible_with_sampling_noise |
| elo_gap | 200-300 | 310 | 0.6458 | 0.6613 | +0.0155 | 0.0618 | compatible_with_sampling_noise |
| elo_gap | 300+ | 28 | 0.6209 | 0.6071 | -0.0137 | 0.1267 | insufficient_sample |
| environment | unknown | 6991 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| environment | lan | 317 | 0.5563 | 0.5584 | +0.0021 | 0.0586 | deviation_detected |
| environment | online | 1270 | 0.5611 | 0.5622 | +0.0011 | 0.0169 | compatible_with_sampling_noise |
| stage | unknown | 7045 | - | - | - | - | segmento desconocido; excluido de conclusiones |
| stage | group | 436 | 0.5604 | 0.5459 | -0.0146 | 0.0492 | deviation_detected |
| stage | swiss | 329 | 0.5777 | 0.6201 | +0.0424 | 0.0541 | deviation_detected |
| stage | quarter | 238 | 0.5810 | 0.5924 | +0.0114 | 0.0618 | compatible_with_sampling_noise |
| stage | final | 76 | 0.5216 | 0.5526 | +0.0310 | 0.0439 | insufficient_sample |
| stage | league | 2 | 0.6138 | 0.0000 | -0.6138 | 0.6138 | insufficient_sample |
| stage | semi | 178 | 0.5402 | 0.5281 | -0.0121 | 0.0795 | compatible_with_sampling_noise |
| stage | ro16 | 191 | 0.5755 | 0.5550 | -0.0205 | 0.0866 | compatible_with_sampling_noise |
| stage | upper_bracket | 2 | 0.6965 | 1.0000 | +0.3035 | 0.3035 | insufficient_sample |
| stage | lower_bracket | 81 | 0.5062 | 0.4938 | -0.0123 | 0.0869 | insufficient_sample |
| format | bo3 | 7772 | 0.5589 | 0.5567 | -0.0021 | 0.0130 | compatible_with_sampling_noise |
| format | bo5 | 80 | 0.5453 | 0.5875 | +0.0422 | 0.0744 | insufficient_sample |
| format | bo1 | 726 | 0.5784 | 0.6212 | +0.0428 | 0.0492 | deviation_detected |
| event_tier | unknown | 8578 | - | - | - | - | segmento desconocido; excluido de conclusiones |

> Diagnostico descriptivo sobre predicciones OOS temporales. N minimo=100; los segmentos pequeños no se interpretan. Una señal no cambia el modelo: solo habilita el siguiente experimento y la puerta champion/challenger conserva producción si no mejora.

Cobertura conocida: elo_gap_bin 8578/8578 (100.0%); stage 1533/8578 (17.9%); environment 1587/8578 (18.5%); format 8578/8578 (100.0%); event_tier 0/8578 (0.0%).

## Target BO3 enriquecido (A6)

- Estado: OFF.
- Cobertura BO3: 9832/2000; minimo por clase: 300.
- Clases: {'0-2': 2470, '1-2': 1870, '2-1': 2038, '2-0': 3454}.
- Scope: scoreline_props_auxiliary_not_winner_model_a.
- Walk-forward: n=7589; log loss multiclase=1.503177; baseline empirico=1.359671; accuracy scoreline=0.400448; accuracy ganador=0.638029.

> did not beat empirical scoreline baseline.

## Ratings y optimizacion B8-B11

- B8 Bradley-Terry bayesiano: ON; partial pooling gaussiano, incertidumbre predictiva y candidato calibrado.
- B9 Kalman state-space: ON; deriva de estado, varianza y candidato calibrado.
- B10 Optuna purgado: ON; estudios=2, C final=0.027562018071197132.
- B11 BO3 composicional: OFF; cobertura=22/500.

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1810 | 21.1% | 0.513 | 0.692 |
| parejos (55-65%) | 3158 | 36.8% | 0.599 | 0.672 |
| claros (65-80%) | 3060 | 35.7% | 0.726 | 0.583 |
| palizas (>=80%) | 550 | 6.4% | 0.838 | 0.442 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Accuracy del favorito por probabilidad predicha

| Probabilidad | Aciertos | N | Accuracy | Prob. media | Gap calibracion |
|---|---:|---:|---:|---:|---:|
| 50-60% | 1872 | 3443 | 0.5437 | 0.5485 | -0.0047 |
| 60-70% | 1799 | 2777 | 0.6478 | 0.6471 | +0.0008 |
| 70-80% | 1373 | 1808 | 0.7594 | 0.7434 | +0.0160 |
| 80-90% | 452 | 540 | 0.8370 | 0.8331 | +0.0039 |
| 90-100% | 9 | 10 | 0.9000 | 0.9136 | -0.0136 |
| **Total** | **5505** | **8578** | **0.6418** | - | - |

## Benchmark de mercado (odds de apertura)

Sobre 1111 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6305 · accuracy 0.6355
- Mercado (odds apertura): log loss 0.601 · accuracy 0.6697

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

Validación walk-forward solo en partidos con odds de apertura (n_eval=921):

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Mercado apertura | 921 | 0.6623 | 0.6024 | 0.2081 | 0.7309 | 0.0281 |
| Model A en mismos partidos | 921 | 0.6319 | 0.6336 | 0.2218 | 0.6847 | 0.0349 |
| Model B stats + apertura | 921 | 0.6439 | 0.6507 | 0.2271 | 0.6775 | 0.0319 |

Banda de competitividad para Model B:

| Banda | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 200 | 21.7% | 0.525 | 0.693 |
| parejos (55-65%) | 317 | 34.4% | 0.650 | 0.650 |
| claros (65-80%) | 281 | 30.5% | 0.690 | 0.609 |
| palizas (>=80%) | 123 | 13.4% | 0.715 | 0.677 |

SHAP de Model B (top 10):

| Feature | mean |SHAP| |
|---|---:|
| opening_odds_prob_centered | 0.9095 |
| last10_score_diff | 0.1922 |
| opp_elo_last20_diff | 0.1612 |
| format_opp_elo_last5_diff | 0.1598 |
| last30_winrate_diff | 0.1566 |
| opp_elo_last5_diff | 0.1402 |
| format_matches_log_diff | 0.1398 |
| trueskill_diff | 0.1386 |
| format_recent_opponent_elo_diff | 0.1377 |
| last5_score_diff | 0.1220 |

> Modelo B usa solo opening_odds point-in-time; closing_odds queda excluida del entrenamiento.

## Arquitecturas de producción con/sin odds

> Reanclaje honesto: incluso con opening odds, el techo realista de accuracy ronda aproximadamente el 70%; no es una garantía. La rama sin odds tiene un techo estructuralmente inferior y su objetivo es acercarse lo máximo posible, no alcanzar paridad.

> Esta tabla es un backtest prequential con refit por periodo, no la puntuacion del shadow fijo de la puerta. Reserva=super_learner_cal; calibracion=sigmoid (0.18); half-life=365.0 dias. Para decidir promocion, usa exclusivamente la seccion Puerta de promocion.

Soporte temporal común: n=1371; fracción sin odds=32.82%; arquitectura congelada=router_two_models.

| Puesto | Arquitectura | N | Accuracy | Log loss | Brier | ECE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | router_two_models | 1371 | 0.6521 | 0.6250 | 0.2178 | 0.0141 |
| 2 | single_mixed_lgbm | 1371 | 0.6389 | 0.6296 | 0.2199 | 0.0216 |

Desglose por régimen:

| Arquitectura | Régimen | N | Accuracy | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| router_two_models | odds | 921 | 0.6504 | 0.6308 | 0.2204 | 0.0224 |
| router_two_models | no_odds | 450 | 0.6556 | 0.6133 | 0.2126 | 0.0351 |
| single_mixed_lgbm | odds | 921 | 0.6330 | 0.6351 | 0.2224 | 0.0367 |
| single_mixed_lgbm | no_odds | 450 | 0.6511 | 0.6182 | 0.2148 | 0.0354 |

## Backtest economico

- Apuestas simuladas: 378
- Hit rate: 0.4894179894179894
- ROI sobre stake: -0.11118325871349118
- Banca: 1000.0 -> 436.80142747658044
- Profit sobre banca inicial: -0.5631985725234202
- Max drawdown: 0.563565618654732
- Vig medio de apertura: 0.0706221952422258
- CLV: n=149; cobertura=0.3941798941798942; media precio=0.012484738492597297; mediana=0.0
- Apuestas limitadas por stake/payout: 0
- Staking: model_favorite_only_quarter_kelly_cap_2_5pct

> Favorite side is fixed by the model. Opening odds determine EV and stake; closing odds are audit-only for CLV. Vig, stake and payout limits are explicit.

## Drift causal (C12)

- Estado: **warning**.
- Predicciones OOS cerradas: 8578.
- Alertas: page_hinkley_log_loss.
- Log loss ventana actual: 0.6446186132456266; referencia: 0.6213391438136657.
- CLV evaluable: 418 (4.9%); media: 0.0058103626284888585.

> Page-Hinkley se aplica al log loss y a -CLV en orden temporal. Las cuotas de cierre solo auditan la ejecucion y nunca son features.

## Model registry

- Artefacto versionado: `None`
- Carpeta: `None`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre el último bloque temporal dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- Strength-of-schedule usa Elo del rival y rendimiento real menos esperado, ambos calculados estrictamente antes del partido.
- El modelo A sigue prediciendo el ganador de la serie. El sidecar BO3 estima 0-2/1-2/2-1/2-0 y solo se publica si mejora el baseline multiclase en walk-forward. El modelo por mapa queda pendiente de mas mapstats point-in-time.

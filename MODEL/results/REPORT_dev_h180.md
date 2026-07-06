# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-06T13:19:49Z  
Histórico: 2025-09-11 → 2026-07-06 (9444 series, era CS2)

## Politica de features opcionales

- Analytics HLTV: OFF (53/120 filas cerradas). Analytics captured and stored, but excluded from training until enough closed point-in-time rows exist.
- Contexto torneo LAN/online/fase: OFF (118/200 contexto; LAN=37/50, online=81/50). Tournament context captured and stored, but excluded from production training until LAN/online coverage is balanced enough.

## Resultados walk-forward (ventana expansiva, paso semanal)

| Modelo | N | Accuracy | Log loss | Brier | ROC-AUC | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Base rate (baseline) | 7008 | 0.5622 | 0.6855 | 0.2462 | 0.4864 | 0.0006 |
| Elo (baseline) | 7008 | 0.6025 | 0.6569 | 0.2325 | 0.6423 | 0.0425 |
| Glicko-2 (baseline) | 7008 | 0.6092 | 0.6596 | 0.2330 | 0.6493 | 0.0498 |
| Logística (Platt) | 7008 | 0.6411 | 0.6331 | 0.2215 | 0.6833 | 0.0168 |
| LightGBM (Platt) | 7008 | 0.6390 | 0.6319 | 0.2209 | 0.6865 | 0.0161 |
| Ensemble LGBM⊕Log (Platt) | 7008 | 0.6427 | 0.6299 | 0.2201 | 0.6896 | 0.0155 |
| Ensemble (isotónica) | 7008 | 0.6394 | 0.6401 | 0.2212 | 0.6871 | 0.0183 |
| Ensemble (beta) | 7008 | 0.6445 | 0.6299 | 0.2201 | 0.6895 | 0.0147 |

> Modelo de producción elegido por menor log loss: **ensemble_beta**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| glicko_prob_centered | 0.1149 |
| glicko_rd_diff | 0.1057 |
| avg_score_diff | 0.1053 |
| matches_log_diff | 0.0851 |
| opp_elo_last20_diff | 0.0839 |
| format_avg_score_diff | 0.0728 |
| recent_opponent_elo_diff | 0.0595 |
| last10_score_diff | 0.0587 |
| streak_diff | 0.0375 |
| elo_diff | 0.0311 |
| format_last5_winrate_diff | 0.0302 |
| opp_elo_last5_diff | 0.0280 |
| event_matches_diff | 0.0276 |
| h2h_winrate_centered | 0.0230 |
| glicko_diff | 0.0228 |

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1649 | 23.5% | 0.529 | 0.692 |
| parejos (55-65%) | 2586 | 36.9% | 0.619 | 0.663 |
| claros (65-80%) | 2298 | 32.8% | 0.713 | 0.595 |
| palizas (>=80%) | 475 | 6.8% | 0.857 | 0.404 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Benchmark de mercado (odds de apertura)

Sobre 69 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.6612 · accuracy 0.5652
- Mercado (odds apertura): log loss 0.6454 · accuracy 0.6957

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

> Modelo B preparado, pero aun no hay suficientes partidos con odds de apertura para validacion walk-forward fiable. n_odds=69 (mínimo de entrenamiento: 120). No se calcula SHAP de Model B hasta que haya validación walk-forward fiable.

## Backtest economico

- Apuestas simuladas: 57
- Hit rate: 0.45614035087719296
- ROI sobre stake: 0.040278889827625876
- Profit sobre banca inicial: 0.04772302946092433
- Max drawdown: 0.12843933071918778
- Staking: quarter_kelly_cap_2_5pct

> Solo usa odds de apertura guardadas; n pequeno hasta acumular mas mercado.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260706_131949Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260706_131949Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- El modelo predice el ganador de la SERIE directamente; la arquitectura composicional Bo3 (P(serie)=P(2-0)+P(2-1)) queda como extensión cuando haya datos por mapa suficientes (PROJECT.md §7.3).

# Informe de entrenamiento — modelo CS2 (Glicko-2 + LightGBM)

Generado: 2026-07-06T13:12:24Z  
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
| Logística (Platt) | 7008 | 0.6404 | 0.6329 | 0.2214 | 0.6838 | 0.0162 |
| LightGBM (Platt) | 7008 | 0.6384 | 0.6346 | 0.2222 | 0.6820 | 0.0167 |
| Ensemble sin calibrar | 7008 | 0.6378 | 0.6319 | 0.2208 | 0.6883 | 0.0170 |
| **Ensemble (Platt)** | 7008 | 0.6481 | 0.6304 | 0.2203 | 0.6886 | 0.0166 |

> Modelo de producción elegido por menor log loss: **ensemble_cal**.

> Log loss y Brier son el objetivo (probabilidades calibradas), no solo accuracy.
> El baseline 'elige al favorito' (Elo/Glicko) ya acierta ~63-65%; el modelo aporta si lo supera en log loss/Brier/AUC.

## Importancia de features (SHAP, |valor| medio)

| Feature | mean |SHAP| |
|---|---:|
| glicko_prob_centered | 0.1723 |
| opp_elo_last20_diff | 0.1034 |
| matches_log_diff | 0.0990 |
| glicko_rd_diff | 0.0988 |
| avg_score_diff | 0.0933 |
| last10_score_diff | 0.0721 |
| format_avg_score_diff | 0.0681 |
| last20_winrate_diff | 0.0679 |
| streak_diff | 0.0604 |
| recent_opponent_elo_diff | 0.0472 |
| glicko_diff | 0.0387 |
| event_matches_diff | 0.0375 |
| elo_diff | 0.0352 |
| last30_winrate_diff | 0.0326 |
| format_last10_score_diff | 0.0307 |

## Acierto por competitividad (¿hay skill o son palizas?)

| Banda (confianza) | N | % | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| coinflip (<55%) | 1484 | 21.2% | 0.550 | 0.691 |
| parejos (55-65%) | 2597 | 37.1% | 0.607 | 0.668 |
| claros (65-80%) | 2522 | 36.0% | 0.715 | 0.591 |
| palizas (>=80%) | 405 | 5.8% | 0.852 | 0.415 |

> Log loss de un coinflip puro = ln(2) ≈ 0.693. El valor del modelo se concentra en la banda 55-80%; en los partidos genuinamente parejos la incertidumbre es irreducible.

## Benchmark de mercado (odds de apertura)

Sobre 69 partidos con odds guardadas (ILUSTRATIVO, n pequeño):

- Modelo: log loss 0.665 · accuracy 0.5942
- Mercado (odds apertura): log loss 0.6454 · accuracy 0.6957

> Ilustrativo (n pequeño). El modelo NO usa odds como feature; esto mide si bate al mercado.

## Modelo B (stats + odds de apertura)

> Modelo B preparado, pero aun no hay suficientes partidos con odds de apertura para validacion walk-forward fiable. n_odds=69 (mínimo de entrenamiento: 120). No se calcula SHAP de Model B hasta que haya validación walk-forward fiable.

## Backtest economico

- Apuestas simuladas: 53
- Hit rate: 0.4339622641509434
- ROI sobre stake: 0.022296633898432706
- Profit sobre banca inicial: 0.026262768288872307
- Max drawdown: 0.1506721376986616
- Staking: quarter_kelly_cap_2_5pct

> Solo usa odds de apertura guardadas; n pequeno hasta acumular mas mercado.

## Model registry

- Artefacto versionado: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260706_131224Z\model.pkl`
- Carpeta: `C:\Users\aleja\OneDrive - Universitat Oberta de Catalunya\CS2-Predictor\MODEL\artifacts\registry\20260706_131224Z`

## Notas metodológicas

- Cero fuga temporal: cada feature se calcula solo con partidos anteriores (Glicko/Elo cronológicos, forma con decay).
- Glicko-2 con periodos de rating semanales; la RD crece con la inactividad (incertidumbre por datos viejos).
- Calibración Platt (sigmoid) ajustada sobre holdout aleatorio dentro de cada fold; se compara contra isotónica/sin-calibrar y se elige por log loss walk-forward.
- Augmentación por simetría A↔B para una frontera antisimétrica sin sesgo de lado.
- Model B usa solo odds de apertura con timestamp; las odds de cierre se guardan como auditoría/benchmark, no como feature.
- El modelo predice el ganador de la SERIE directamente; la arquitectura composicional Bo3 (P(serie)=P(2-0)+P(2-1)) queda como extensión cuando haya datos por mapa suficientes (PROJECT.md §7.3).

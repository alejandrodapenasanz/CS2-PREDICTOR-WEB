# Backtest experimental: toda informacion disponible

Generado: 2026-07-04T14:00:47Z

## Cobertura

- Total series CS2: 9418
- Partidos con mapstats/player stats del mismo partido archivados: 92
- Partidos con historial previo de assets para ambos equipos: 38
- Partidos con Analytics point-in-time: 27
- Partidos con odds de apertura/snapshot: 55

## Resultados

| Experimento | n | Accuracy | Log loss | Nota |
|---|---:|---:|---:|---|
| pre_match_sin_assets | 6982 | 0.6448 | 0.6333 | Modelo logistico walk-forward sin stats/mapstats. |
| pre_match_con_assets_rolling | 6982 | 0.6442 | 0.6333 | Incluye historiales previos de mapstats/player stats; no usa boxscore del mismo partido. |
| pre_match_assets_analytics_forzado | 6982 | 0.6434 | 0.6332 | Analytics forzado aunque no llega al umbral productivo de 120 filas. |
| pre_match_assets_analytics_odds_forzado | 6982 | 0.6444 | 0.6333 | Incluye odds point-in-time; diagnostico, no modelo estadistico puro. |
| pre_match_con_assets_solo_filas_con_cobertura | 38 | 0.6842 | 0.5964 | Subset honesto: solo partidos donde ambos equipos tenian asset history previo. |
| post_veto_walk_forward | 92 | 0.6630 | 0.6461 | Escenario post-veto con veto/mapas reales; entrenamiento cronologico, muestra muy pequena. |
| post_veto_asset_only_expanding_min30 | 62 | 0.4839 | 1.1645 | Diagnostico solo sobre 92 filas con assets; min_train=30, no robusto. |

## Decision

No se promueve el entrenamiento experimental. La muestra point-in-time para assets/Analytics/odds esta por debajo del umbral minimo razonable (120 filas cerradas por bloque) y el modo post-veto solo sirve como senal exploratoria. El modelo productivo queda como estaba.

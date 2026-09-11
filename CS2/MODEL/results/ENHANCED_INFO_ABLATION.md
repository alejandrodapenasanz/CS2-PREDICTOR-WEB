# Ablación causal de enhanced-info

Ejecución: 2026-08-26. Semilla: 42. Artefacto vivo comparado:
`20260810_064827Z`, con corte de entrenamiento `2026-08-10` y arquitectura
real `model_a_no_odds`.

## Contrato de disponibilidad

`enhanced_available = 1` solo si al menos una de estas fuentes tiene contenido
utilizable y un timestamp de disponibilidad estrictamente anterior al kickoff:

| Fuente | Evidencia exigida |
|---|---|
| Opening odds | Primera captura válida de dos cuotas coherentes, de-vigada, con `captured_at_utc < kickoff_utc`. |
| Snapshots de jugadores | La captura más nueva usada para ambos equipos es `< kickoff_utc`; se mantiene el mínimo de cobertura y mapas. |
| Rankings | Join as-of con `bisect_left`: las capturas iguales al kickoff y posteriores quedan fuera. |
| Analytics | Último snapshot disponible con `match_analytics_snapshots.captured_at_utc < matches.datetime_utc`. |
| Contexto | `matches.prematch_captured_at_utc < kickoff_utc`; un `context_json` sin esa evidencia no marca la fila. |
| Alineación anunciada | Snapshot completo 5v5 de ambos lados con un timestamp compartido `< kickoff_utc`. |

Box scores, resultados, alineaciones reales observadas al terminar y cualquier
payload conocido en el kickoff o después no cuentan. Cada fila conserva los
flags `enhanced_source_*` y la evidencia temporal usada para auditoría.

## Warm-up y políticas

Se cargaron y conservaron 10.414 partidos crudos. Las features se reconstruyeron
una sola vez, en orden cronológico, sobre las 10.414 filas; después se aplicaron
las máscaras de entrenamiento. Por tanto, excluir una fila del learner no la
elimina del historial usado por Elo, Glicko, forma, fuerza de rivales o H2H.

Las políticas fueron:

- `all`: todas las filas, peso 1.
- `only_enhanced`: solo filas con `enhanced_available=1`, peso 1.
- `recency_weighted`: todas las filas con
  `max(1e-3, 0.5 ** (age_days / 365))`.

La evaluación usa exactamente los mismos 324 partidos del 2026-08-11 al
2026-08-26 para vivo y challengers. Accuracy es informativa; la puerta decide
por log-loss y, dentro del margen, por Brier.

## Cobertura

Hay 1.061/10.414 filas enhanced (10,19 %). Los conteos por fuente se solapan:

| Fuente | Filas |
|---|---:|
| Opening odds | 719 |
| Snapshots de jugadores | 684 |
| Rankings | 758 |
| Analytics | 840 |
| Contexto | 986 |
| Alineación anunciada | 737 |

## Resultado de la puerta

| Variante | Filas de ajuste | Accuracy | Log-loss | Brier | Decisión |
|---|---:|---:|---:|---:|---|
| Vivo | 10.090 | 62,04 % | 0,648969 | 0,228952 | Referencia |
| `all` | 10.090 | 62,35 % | 0,648702 | 0,228807 | Rechazada |
| `only_enhanced` | 737 | 60,80 % | 0,667428 | 0,236703 | Rechazada |
| `recency_weighted` | 10.090 | 61,73 % | **0,648004** | **0,228474** | Rechazada por margen |

`recency_weighted` ganó la comparación interna, pero sus mejoras frente al vivo
(`-0,000965` en log-loss y `-0,000478` en Brier) no alcanzan los márgenes
anti-churn configurados (`0,001` y `0,0005`). No se creó ni publicó un nuevo
artefacto. `latest` sigue en `20260810_064827Z` y `last_good` en
`20260803_064101Z`.

La referencia larga de producto sigue siendo 64,40 % (5.097/7.914); no se usa
para decidir esta promoción porque no es la misma cohorte.

## Sesgo observado

La cobertura enhanced está fuertemente concentrada en datos recientes y en
partidos cuyo entorno fue identificado: 59,64 % en los últimos 90 días frente a
0 % entre 91 y 365 días; 99,61 % en LAN, 100 % en online y 0,80 % cuando el
entorno es desconocido. `event_tier` está ausente en este histórico y no permite
un análisis útil por tier.

Esto explica el riesgo de especialización de `only_enhanced` y su peor
generalización. La muestra se evaluó en el hold-out completo, nunca solo en las
filas cubiertas.

El detalle ejecutable y los hashes de cohorte/predicciones están en
`MODEL/results/enhanced_info_ablation/enhanced_info_ablation.json`.

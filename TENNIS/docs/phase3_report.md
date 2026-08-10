# Informe del sistema Elo de la fase 3

> **El informe Elo original queda sustituido por este cierre de fase 9.** El
> contrato anterior (`sackmann-elo-v2`) usaba `tourney_date` como fecha efectiva
> inmediata y excluía varios resultados de partidos iniciados. Las cifras de
> aquel run no describen el sistema activo.

## Contrato activo

- Algoritmo: `sackmann-elo-v5`.
- Fuente temporal: `tourney_date` es la fecha aproximada de inicio del torneo,
  no la fecha real de cada ronda.
- Política: `sackmann-tourney-start-embargo-v1`, con embargo de 21 días.
- Disponibilidad: `result_available_date=tourney_date+21 días`; un resultado
  solo es utilizable si `result_available_date < as_of_date`. La igualdad se
  excluye y el primer corte utilizable es `tourney_date+22`.
- `RET`, `DEF`, `ABD` y `ABN` se incluyen cuando el partido se inició y existe
  un ganador oficial. `W/O`, `Walkover` y `BYE` se excluyen.
- El diagnóstico DOB de identidades no selecciona eventos históricos. Sus 90
  claves solo pueden bloquear o degradar inferencia operativa actual.
- Los universos `M` y `F` son completamente independientes. Dentro de cada
  género, todos los niveles elegibles alimentan el mismo pool.

La fórmula aprobada es rating inicial 1500,
`K(n)=250/(n+5)^0.4` y mezcla 50/50 entre Elo general y Elo puro de superficie.
El contrato matemático completo está en [`elo.md`](elo.md).

## Ejecución Elo v5 verificada

| Campo | Hombres (`M`) | Mujeres (`F`) |
|---|---:|---:|
| Commit Sackmann | `83733587353df8a41f2fd4f516147d5aa83f5a8d` | `83733587353df8a41f2fd4f516147d5aa83f5a8d` |
| Inicio UTC | 2026-08-09 13:02:58 | 2026-08-09 13:09:33 |
| Fin UTC | 2026-08-09 13:09:33 | 2026-08-09 13:15:17 |
| `run_id` | `elo-m-837d7bd8d6c48ae3f17f3e9c` | `elo-f-ec4cffd280c630cab1de03b8` |
| Fingerprint de entrada | `837d7bd8d6c48ae3f17f3e9c0beb953d79a4bd0da4c0568f7326261914e14fe9` | `ec4cffd280c630cab1de03b8ed276a16b8a63b823111f1e7b6bfdfcb75bc2072` |
| Rango fuente | 1967-12-28 a 2026-06-01 | 1967-12-25 a 2026-06-02 |
| Rango efectivo | 1968-01-18 a 2026-06-22 | 1968-01-15 a 2026-06-23 |
| Bloques efectivos | 4.644 | 4.333 |
| Partidos incluidos | 959.020 | 769.835 |
| Filas excluidas | 4.195 | 16.822 |
| Integridad SQLite | `PRAGMA integrity_check = ok` | `PRAGMA integrity_check = ok` |

Las exclusiones se reconciliaron con el manifiesto de features generado desde
el mismo universo: en `M`, 4.188 estados excluidos y 7 autopartidos; en `F`,
11.882 duplicados, 4.639 estados excluidos, 289 niveles excluidos y 12
autopartidos. Los contadores de `date_blocks` suman exactamente los partidos
incluidos de cada género.

## Top 10 de Elo general

Es un top histórico por último estado, sin decaimiento por inactividad. Por eso
puede contener leyendas retiradas: sirve como prueba de cordura del rating, no
como ranking de jugadores activos.

| # | Hombres | Elo | Estado | Mujeres | Elo | Estado |
|---:|---|---:|---|---|---:|---|
| 1 | Jannik Sinner | 2832,51 | 2026-06-15 | Aryna Sabalenka | 2757,58 | 2026-06-15 |
| 2 | Carlos Alcaraz | 2764,66 | 2026-05-04 | Justine Henin | 2728,74 | 2011-02-07 |
| 3 | Roger Federer | 2654,82 | 2021-07-19 | Ashleigh Barty | 2704,74 | 2022-02-07 |
| 4 | Novak Djokovic | 2654,46 | 2026-06-15 | Steffi Graf | 2697,98 | 1999-08-23 |
| 5 | Rafael Nadal | 2585,12 | 2024-12-10 | Elena Rybakina | 2697,82 | 2026-06-15 |
| 6 | Alexander Zverev | 2585,07 | 2026-06-15 | Lindsay Davenport | 2646,73 | 2008-09-15 |
| 7 | Juan Martín del Potro | 2540,29 | 2022-02-28 | Iga Swiatek | 2627,75 | 2026-06-15 |
| 8 | Robin Söderling | 2533,21 | 2011-08-01 | Mirra Andreeva | 2627,29 | 2026-06-15 |
| 9 | Arthur Fils | 2527,30 | 2026-05-27 | Coco Gauff | 2626,49 | 2026-06-15 |
| 10 | Jack Draper | 2501,22 | 2026-05-04 | Jessica Pegula | 2621,01 | 2026-06-15 |

## Ejemplo reproducible de `get_elo`

Para Carlos Alcaraz (`player_id=207989`, `M`) sobre hard:

| `as_of_date` | `state_date` usada | Elo general | Elo hard | Elo combinado | Partidos general/hard |
|---|---|---:|---:|---:|---:|
| 2020-01-01 | 2019-10-28 | 1983,7371 | 1696,7762 | 1840,2567 | 31 / 5 |
| 2026-01-01 | 2025-11-30 | 2764,6241 | 2600,7016 | 2682,6629 | 438 / 205 |

En ambos cortes se cumple `state_date < as_of_date`, el `run_id` es
`elo-m-837d7bd8d6c48ae3f17f3e9c` y el commit coincide con el que consumen las
features. La consulta se obtiene mediante `src.elo.service.get_elo`; un jugador
sin estado anterior recibe 1500 y queda marcado como cold start.

## Validación

- La política de embargo prueba igualdad excluida y primera disponibilidad en
  `D+22` (`tests/test_temporal_embargo.py`).
- La invariancia as-of se prueba calculando el mismo corte antes y después de
  añadir resultados futuros (`tests/test_elo_engine.py` y
  `tests/test_elo_pipeline.py`).
- La separación por género, mezcla de superficie, estados prepartido,
  persistencia, hashes e inventario de código tienen cobertura focal.
- Resultado automatizado final, incluidos los tests focales y la suite completa:
  `329/329 tests correctos`.

El cierre consolidado y sus limitaciones están en [`audit.md`](audit.md).

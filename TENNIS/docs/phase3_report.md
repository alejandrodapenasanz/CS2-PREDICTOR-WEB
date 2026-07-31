# Informe del sistema Elo de la fase 3

## Ejecución auditada

- Fecha de cálculo y auditoría: 30 de julio de 2026.
- Fuente canónica: `Aneeshers/tennis-sackmann-archive`.
- Commit fuente: `83733587353df8a41f2fd4f516147d5aa83f5a8d`.
- Versión del algoritmo: `sackmann-elo-v2`.
- Base publicada: `data/processed/elo/elo.sqlite3`.
- Tamaño de la base: 403.329.024 bytes, aproximadamente 384,6 MiB.
- `PRAGMA integrity_check`: `ok`.
- Violaciones detectadas por `PRAGMA foreign_key_check`: 0.
- Segunda ejecución con la misma entrada: omitida de forma idempotente.

Los parámetros activos son rating inicial 1500, escala logística 400,
`K(n) = 250 / (n + 5) ** 0.4` y mezcla 50/50 entre Elo general y Elo puro de
superficie. La fórmula, la congelación por fecha y todas las decisiones de
elegibilidad están en [`elo.md`](elo.md).

## Runs activos

| Género | `run_id` | Fingerprint de entrada | Rango de eventos |
|---|---|---|---|
| M | `elo-m-c94ef28d4994dcda3c94d8e3` | `c94ef28d4994dcda3c94d8e38b0de631a610e677648b8add83c7da7b2b04cc27` | 1967-12-28 a 2026-06-01 |
| F | `elo-f-4eecb96b73f7f08f40ec2f33` | `4eecb96b73f7f08f40ec2f33b36dfbc74f4486942fc1102bf5c6afd453c2b681` | 1967-12-25 a 2026-06-02 |

La base contiene dos runs completos y activos, 8.977 bloques de fecha y
1.679.231 estados históricos de jugador.

## Auditoría de inclusión

| Género | Filas fuente | Incluidas | Excluidas | Duplicados exactos | Nivel `E/J` | Estado del marcador | Mismo jugador | Bloques | Estados |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 963.215 | 929.149 | 34.066 | 0 | 0 | 34.059 | 7 | 4.644 | 897.240 |
| F | 786.657 | 751.126 | 35.531 | 11.724 | 289 | 23.507 | 11 | 4.333 | 781.991 |
| **Total** | **1.749.872** | **1.680.275** | **69.597** | **11.724** | **289** | **57.566** | **18** | **8.977** | **1.679.231** |

Los motivos son mutuamente excluyentes y se asignan por prioridad: nivel,
estado explícito, mismo jugador y, finalmente, copia exacta. Por eso una copia
de una fila que ya contiene `RET` se contabiliza como estado, no también como
duplicado. `sackmann-elo-v2` decide la igualdad sobre las 49 cadenas raw antes
del tipado.

## Top 10 de Elo general masculino

Es el top absoluto del último estado disponible de cada jugador, sin filtro de
actividad ni decaimiento por inactividad.

| Puesto | Jugador | ID | Elo general | Partidos | Último estado |
|---:|---|---:|---:|---:|---:|
| 1 | Jannik Sinner | 206173 | 2848,145 | 532 | 2026-05-25 |
| 2 | Carlos Alcaraz | 207989 | 2754,480 | 451 | 2026-04-13 |
| 3 | Roger Federer | 103819 | 2656,167 | 1.563 | 2021-06-28 |
| 4 | Novak Djokovic | 104925 | 2656,026 | 1.439 | 2026-05-25 |
| 5 | Rafael Nadal | 104745 | 2583,098 | 1.377 | 2024-11-19 |
| 6 | Alexander Zverev | 100644 | 2576,970 | 877 | 2026-05-25 |
| 7 | Juan Martin del Potro | 105223 | 2549,358 | 719 | 2022-02-07 |
| 8 | Arthur Fils | 209950 | 2538,284 | 302 | 2026-04-22 |
| 9 | Robin Soderling | 104417 | 2534,690 | 587 | 2011-07-11 |
| 10 | Jack Draper | 207733 | 2514,767 | 338 | 2026-03-18 |

## Top 10 de Elo general femenino

| Puesto | Jugadora | ID | Elo general | Partidos | Último estado |
|---:|---|---:|---:|---:|---:|
| 1 | Aryna Sabalenka | 214544 | 2759,521 | 687 | 2026-05-25 |
| 2 | Justine Henin | 200003 | 2736,862 | 621 | 2011-01-17 |
| 3 | Ashleigh Barty | 202458 | 2728,607 | 375 | 2022-01-17 |
| 4 | Steffi Graf | 200414 | 2726,050 | 1.018 | 1999-06-21 |
| 5 | Elena Rybakina | 214981 | 2707,010 | 542 | 2026-05-25 |
| 6 | Lindsay Davenport | 200128 | 2673,114 | 928 | 2008-08-25 |
| 7 | Iga Swiatek | 216347 | 2640,048 | 508 | 2026-05-25 |
| 8 | Coco Gauff | 221103 | 2634,295 | 412 | 2026-05-25 |
| 9 | Mirra Andreeva | 259799 | 2625,604 | 237 | 2026-05-25 |
| 10 | Jessica Pegula | 202468 | 2619,849 | 720 | 2026-05-25 |

Los 20 IDs se resolvieron contra el maestro de jugadores de su propio género.
La presencia de figuras retiradas es esperable bajo la decisión aprobada de
mostrar el top absoluto sin regresión por inactividad; no representa un ranking
de jugadores activos.

## Ejemplo de consulta `get_elo` as-of

Consulta de Novak Djokovic, ID 104925, universo `M`, superficie `Hard`:

| `as_of_date` | Estado recuperado | Elo general | Elo Hard puro | Combinado | Partidos generales | Partidos Hard |
|---:|---:|---:|---:|---:|---:|---:|
| 2010-01-01 | 2009-11-22 | 2636,998744 | 2546,323114 | 2591,660929 | 409 | 217 |
| 2025-01-01 | 2024-12-30 | 2708,915097 | 2628,529786 | 2668,722441 | 1.381 | 830 |

En ambos casos `state_date < as_of_date`. La consulta de 2025 no modifica ni
reescribe el resultado histórico de 2010.

## Validación

La ejecución final de:

```powershell
python -m unittest discover -s tests -v
```

produjo 47 tests correctos. Entre ellos:

- añadir partidos futuros deja exactamente igual un Elo `as_of` anterior;
- todos los partidos con la misma fecha usan el snapshot previo común;
- una fecha no puede reabrirse ni persistirse fuera de orden;
- el mismo `player_id` en `M` y `F` permanece aislado;
- cada superficie mantiene su rating y contador independientes;
- solo una copia raw exacta se deduplica;
- un bloque SQLite fallido se revierte como una unidad;
- un run incompleto nunca sustituye al activo.

La revisión de cordura de los nombres queda expuesta en los dos top 10
anteriores para validación humana, tal como exige esta fase.


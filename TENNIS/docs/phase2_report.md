# Informe de ingesta de la fase 2

## Ejecución auditada

- Fecha de auditoría: 30 de julio de 2026.
- Fuente canónica:
  `Aneeshers/tennis-sackmann-archive`.
- Commit:
  `83733587353df8a41f2fd4f516147d5aa83f5a8d`.
- CSV seleccionados y verificados: 277.
- CSV de partidos: 262.
- Partidos cargados: 1.749.872.
- Jugadores: 66.912 hombres y 70.571 mujeres.
- Excepciones fuente normalizadas: 1.

## Fuentes auxiliares actualizadas

| Fuente | Revisión/fecha | Resultado |
|---|---|---|
| Match Charting Project | commit `2c59eef194967e688b69e73df344184a06322cd8` | 40 CSV y 2 documentos descargados y verificados |
| Tennis Abstract Elo ATP | rating 2026-07-27; recuperado 2026-07-30 07:49:55 UTC | 548 jugadores; HTML y metadata activados |
| Tennis Abstract Elo WTA | rating 2026-07-27; recuperado 2026-07-30 07:49:55 UTC | 544 jugadoras; HTML y metadata activados |

Una segunda ejecución inmediata produjo 0 descargas Sackmann, 0 descargas
Match Charting y 0 GET a Tennis Abstract.

Match Charting sigue fuera del DataFrame canónico. Su auditoría encontró 7.566
filas masculinas pero 7.565 `match_id` únicos, y 4.080 filas femeninas pero
4.073 IDs únicos. Dos filas masculinas y nueve femeninas tienen menos campos
que la cabecera. No se corrigieron ni fusionaron; deberán resolverse
explícitamente durante el mapeo de la fase 5.

## Partidos por género y `tour_level`

`tour_level` es una copia textual exacta de `tourney_level`; los códigos no
documentados no se traducen.

| Género | `tour_level` | Partidos | Fecha mínima | Fecha máxima |
|---|---:|---:|---:|---:|
| F | `10` | 152.232 | 1995-12-11 | 2016-12-19 |
| F | `100` | 10.622 | 2007-04-23 | 2026-06-01 |
| F | `15` | 108.219 | 2012-05-28 | 2026-06-01 |
| F | `20` | 1.237 | 1996-03-11 | 2003-05-05 |
| F | `25` | 120.926 | 1994-01-31 | 2023-12-18 |
| F | `35` | 17.438 | 2024-01-01 | 2026-05-25 |
| F | `35+H` | 31 | 2025-06-16 | 2025-06-16 |
| F | `40` | 2.741 | 2023-01-02 | 2023-12-25 |
| F | `50` | 27.973 | 1994-05-02 | 2026-06-01 |
| F | `50+H` | 184 | 2025-02-24 | 2025-09-22 |
| F | `60` | 15.129 | 2017-01-23 | 2023-12-11 |
| F | `75` | 13.120 | 1996-04-15 | 2026-06-01 |
| F | `80` | 3.098 | 2017-04-10 | 2023-10-23 |
| F | `C` | 79.898 | 1972-08-07 | 2026-06-02 |
| F | `CC` | 1.913 | 1996-03-03 | 2003-03-03 |
| F | `D` | 12.246 | 1968-05-21 | 2026-04-10 |
| F | `E` | 283 | 1970-04-06 | 1987-08-28 |
| F | `F` | 533 | 1975-04-01 | 2025-11-01 |
| F | `G` | 41.355 | 1968-01-19 | 2026-05-25 |
| F | `I` | 25.832 | 2009-01-05 | 2026-05-18 |
| F | `J` | 6 | 1969-02-13 | 1973-08-30 |
| F | `O` | 672 | 1968-10-14 | 2024-07-29 |
| F | `P` | 18.179 | 2009-01-12 | 2026-05-17 |
| F | `PM` | 7.782 | 2009-03-09 | 2026-05-05 |
| F | `T1` | 7.273 | 1985-02-05 | 2008-10-06 |
| F | `T2` | 8.091 | 2000-01-07 | 2008-10-20 |
| F | `T3` | 7.982 | 2000-01-03 | 2008-10-27 |
| F | `T4` | 5.311 | 2000-01-03 | 2008-09-29 |
| F | `T5` | 1.728 | 1995-01-02 | 2005-01-10 |
| F | `W` | 94.623 | 1967-12-25 | 2024-11-25 |
| M | `15` | 58.502 | 2020-01-06 | 2026-06-01 |
| M | `25` | 35.072 | 2020-01-06 | 2026-06-01 |
| M | `A` | 146.224 | 1967-12-28 | 2026-05-17 |
| M | `C` | 208.233 | 1978-01-08 | 2026-06-01 |
| M | `D` | 15.170 | 1968-03-20 | 2026-02-07 |
| M | `F` | 624 | 1982-10-14 | 2025-12-17 |
| M | `G` | 39.835 | 1968-01-19 | 2026-05-25 |
| M | `M` | 29.710 | 1968-05-06 | 2026-05-06 |
| M | `O` | 64 | 2024-07-29 | 2024-07-29 |
| M | `S` | 429.781 | 1990-12-29 | 2019-12-30 |

Totales: 786.657 partidos femeninos y 963.215 masculinos.

## Cobertura de archivos

| Género | Familia | Archivos | Primer año | Último año | Años de archivo ausentes |
|---|---|---:|---:|---:|---|
| M | `atp_main` | 59 | 1968 | 2026 | Ninguno |
| M | `atp_qual_chall` | 49 | 1978 | 2026 | Ninguno |
| M | `atp_futures` | 36 | 1991 | 2026 | Ninguno |
| F | `wta_main` | 59 | 1968 | 2026 | Ninguno |
| F | `wta_qual_itf` | 59 | 1968 | 2026 | Ninguno |

## Huecos y cautelas visibles

- Los archivos de 2026 son parciales: sus últimas fechas observadas se sitúan
  entre mayo y comienzos de junio.
- Los archivos cuyo nombre comienza en 1968 contienen algunas semanas de torneo
  fechadas a finales de diciembre de 1967. No se corrigieron ni reasignaron.
- Los rankings ATP observados abarcan 1973-08-27 a 2026-06-08 sin años civiles
  completamente ausentes, pero su cadencia inicial es irregular: solo hay 7,
  10, 7 y 2 fechas de ranking en 1973, 1974, 1975 y 1976, respectivamente.
- Los rankings WTA observados comienzan el 1984-01-02; no hay rankings femeninos
  anteriores en el snapshot.
- `W`, `CC`, `35+H` y `50+H` no están definidos por el diccionario Sackmann
  preservado. Se mantienen como valores opacos.
- `O` aparece solo en eventos olímpicos durante la auditoría, pero se conserva
  sin traducir porque el diccionario no lo define.
- `tourney_date` suele ser el inicio de la semana del torneo, no la fecha exacta
  del partido. No permite imponer orden dentro de un mismo evento.

## Anomalía histórica

`wta/wta_matches_1970.csv`, fila CSV 1819, contiene `draw_size="exho"` para
`Rondebosch Exho`. Es la única anomalía numérica encontrada. El loader la
convierte a nulo con una marca explícita y deja intacto el CSV crudo. El detalle
y la regla exacta están en
[`data_dictionary.md`](data_dictionary.md#excepción-fuente-aprobada).

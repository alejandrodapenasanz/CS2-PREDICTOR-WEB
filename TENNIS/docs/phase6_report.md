# Informe de la fase 6: features causales

## Resultado reproducible

La construcción completa terminó sobre el snapshot Sackmann verificado:

```text
source_commit:
83733587353df8a41f2fd4f516147d5aa83f5a8d

feature_fingerprint:
c08239fe01d4607f734bfecf33b8e17aae1d4f8194ad5b7ccd2e35b4a2cf1e59

schema:
tennis-features-v1
```

Se publicaron dos datasets independientes:

| Género | Filas fuente | Filas entrenamiento | Excluidas | Proporción `y=1` | Rango `tourney_date` | Tamaño (bytes) |
|---|---:|---:|---:|---:|---|---:|
| M | 963.215 | 929.149 | 34.066 | 0,499458 | 1967-12-28 a 2026-06-01 | 161.312.442 |
| F | 786.657 | 751.126 | 35.531 | 0,500449 | 1967-12-25 a 2026-06-02 | 129.569.818 |

Las 929.149 y 751.126 identidades `record_id` son únicas dentro de sus
respectivos Parquet: el número de IDs distintos coincide exactamente con el
número de filas.

## Exclusiones reconciliadas

Cada fila fuente se reconcilia como fila de entrenamiento o como una exclusión
auditada:

| Motivo | M | F |
|---|---:|---:|
| Estado no jugado/completo (`W/O`, `Walkover`, `BYE`, `RET`, `DEF`, `ABD`, `ABN`) | 34.059 | 23.507 |
| Duplicado exacto | 0 | 11.724 |
| Nivel excluido (`E` o `J`) | 0 | 289 |
| Mismo ID como ganador y perdedor | 7 | 11 |
| **Total** | **34.066** | **35.531** |

Un marcador ausente no se excluye automáticamente: Elo solo requiere ganador y
perdedor distintos si no existe una marca explícita de partido no válido.

## Distribución por nivel canónico

Estas son las categorías exactas escritas en los Parquet:

| Nivel | M | F |
|---|---:|---:|
| `ATP Tour` | 171.710 | 0 |
| `WTA Tour` | 0 | 172.147 |
| `Grand Slam` | 38.632 | 40.766 |
| `Challenger` | 200.829 | 66.576 |
| `ITF` | 503.130 | 458.881 |
| `Team` | 14.785 | 12.101 |
| `Other` | 63 | 655 |
| **Total** | **929.149** | **751.126** |

El normalizador se auditó contra las 54 combinaciones reales de familia fuente
y código crudo del snapshot. Cualquier combinación nueva provoca un error en
vez de caer en una categoría genérica silenciosa. El inventario completo y sus
decisiones están en `docs/features.md`.

## Rankings y cuarentena

| Género | Filas fuente | Indexadas | Duplicados exactos | Claves conflictivas | Observaciones conflictivas |
|---|---:|---:|---:|---:|---:|
| M | 3.420.595 | 3.419.216 | 181 | 599 | 1.198 |
| F | 2.154.318 | 2.153.811 | 55 | 226 | 452 |

Las 825 claves conflictivas aportan 1.650 observaciones al inventario
`data/processed/features/ranking_conflicts.csv`. Todas las observaciones de una
clave `(player_id, ranking_date)` incompatible se ponen en cuarentena. La
consulta retrocede al último snapshot limpio con `ranking_date < D`; no elige
una fila arbitraria. El vector registra además la antigüedad del snapshot y el
número de fechas conflictivas visibles que tuvo que saltar.

El dataset contiene 150.489 lados masculinos y 333.850 lados femeninos sin
ranking causal disponible. Permanecen nulos y están acompañados por flags de
ausencia.

Hubo 497 lados masculinos y 188 femeninos cuyo snapshot retrocedió sobre al
menos una fecha conflictiva. El máximo observado de
`ranking_conflict_dates_skipped` fue 210 en M y 62 en F; estos contadores
permiten que la fase de modelado mida o filtre explícitamente esos fallbacks.

La auditoría temporal encontró:

```text
filas con ranking_date_A >= match_date: 0
filas con ranking_date_B >= match_date: 0
```

## Auditoría anti-fugas y de calidad

| Comprobación | M | F | Resultado |
|---|---:|---:|---|
| `record_id` distintos | 929.149 | 751.126 | Igual al total de filas |
| Rankings con fecha `>= D` | 0 | 0 | Correcto |
| Descansos presentes `<= 0` | 0 | 0 | Correcto |
| Lados con edad inválida para `D` | 632 | 25 | Degradados a nulo y marcados |
| Valores no nulos de cuotas/mercado/modelo/edge | 0 | 0 | Correcto: no existen cuotas históricas Sackmann |
| Discrepancias Elo en 100 filas aleatorias | 0 | 0 | General y combinado iguales a la base activa de fase 3 |

En total hay 19.406 lados masculinos y 58.801 femeninos sin edad utilizable;
los 632 y 25 casos con nacimiento incompatible con la fecha son un subconjunto
explícitamente marcado. Se concentran en IDs históricos reutilizados o en
maestros cuya fecha de nacimiento corresponde a una persona posterior al
partido; por ejemplo, el ID ATP `103616` aparece en partidos de 1968 pero su
DOB maestra es de 1980. Los 657 casos quedaron con edad nula y flag de
invalidez; no se reasignaron IDs ni se corrigieron o imputaron fechas
biográficas.

La ausencia de cuotas históricas es visible: cuotas, probabilidades implícitas
brutas, probabilidades de-vigadas, overround, margen, timestamps de mercado,
probabilidad del modelo y `edge` permanecen nulos. Las fases posteriores solo
podrán rellenarlos con observaciones recuperadas antes del instante de
predicción.

La comprobación Elo consultó, para 100 filas aleatorias de cada género, la base
activa de la fase 3 mediante `get_elos()` con el mismo jugador, superficie y
corte `D`. Tanto el rating general como el combinado por superficie coincidieron
en las 200 comparaciones; hubo cero discrepancias.

Una segunda invocación con las mismas fuentes y parámetros devolvió
`skipped=True`. El constructor verificó el fingerprint, los tamaños y los
SHA-256 publicados y reutilizó los artefactos sin reconstruirlos.

La ejecución de cierre de la suite completa obtuvo:

```text
198/198 tests correctos
```

Las pruebas cubren, entre otros contratos, invariancia al añadir partidos
posteriores, congelación de todos los partidos de una misma fecha, límites de
meses naturales, orientación determinista y balanceada, H2H y descanso,
rankings estrictamente anteriores con conflictos en cuarentena, edad, cuotas y
timestamps, allowlist, esquema Arrow, fingerprint, idempotencia y CLI.

## Vector comentado: Federer–Nadal, Wimbledon 2008

El ejemplo se leyó directamente de `training_M.parquet`. Corresponde a Roger
Federer como A (`103819`) y Rafael Nadal como B (`104745`), final de Wimbledon
codificada con `tourney_date=2008-06-23`.

| Grupo | A: Federer | B: Nadal | A−B / valor orientado |
|---|---:|---:|---:|
| Elo general | 2.646,880732 | 2.656,038663 | −9,157931 |
| Elo puro en hierba | 2.323,332838 | 2.131,693777 | +191,639062 |
| Elo efectivo 50/50 | 2.485,106785 | 2.393,866220 | +91,240565 |
| Partidos Elo general | 775 | 462 | +313 |
| Partidos Elo hierba | 89 | 29 | +60 |
| Forma últimos N | 0,916667 (12) | 1,000000 (12) | −0,083333 |
| Forma últimos 3 meses | 0,838710 (31) | 0,942857 (35) | −0,104147 |
| Descanso | 14 días | 14 días | 0 |
| Ranking causal | 1 | 2 | −1 |
| Puntos de ranking | 6.900 | 5.755 | +1.145 |
| Edad | 26,875295 | 22,056579 | +4,818716 |
| Distancia a 30 | 3,124705 | 7,943421 | −4,818716 |

Contexto y otras features:

```text
surface = Grass
tour_level_raw = G
tour_level = Grand Slam
source_family = atp_main
best_of = 5
round = F

h2h_global_balance = -0,294118  (17 partidos)
h2h_surface_balance = 1,000000  (2 partidos en hierba)

ranking_date_A = ranking_date_B = 2008-06-16
ranking_age_days_A = ranking_age_days_B = 7
ranking_conflict_dates_skipped_A = 0
ranking_conflict_dates_skipped_B = 0

odds_A = odds_B = null
market_probability_A = market_probability_B = null
model_probability_A = null
edge = null
y = 0
```

El H2H global negativo indica ventaja histórica de B desde la orientación A/B,
mientras que el H2H de hierba favorecía a A en los dos precedentes disponibles.
Los contadores de forma reciente son 12, no 10, porque el bloque completo de la
fecha que cruza el límite se conserva para no inventar un orden intradía.

`y=0` significa que A perdió y B ganó, coherente con la orientación aleatoria
estable. Ninguna feature contiene ese resultado: todos los estados usados son
anteriores a `2008-06-23`.

## Limitación temporal conocida

El dato Sackmann de esta final ilustra la principal cautela del dataset:
`tourney_date=2008-06-23` es el inicio aproximado del torneo, no el 6 de julio,
fecha real de la final. Las rondas anteriores de Wimbledon comparten el mismo
corte y no alimentan Elo, forma, H2H ni descanso de la final. Esta decisión
conservadora evita usar el resultado de una ronda posterior como si precediera
a otra cuando la fuente no proporciona fechas reales por partido.

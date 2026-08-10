# Features causales y dataset de entrenamiento

## Propósito y contrato temporal

La fase 6 construye un vector de características para un partido entre los
jugadores `A` y `B` y dos datasets históricos independientes, uno masculino y
otro femenino. Salvo las variables de contexto, una diferencia se orienta
siempre como:

```text
diff = valor_A - valor_B
```

Si falta cualquiera de los dos valores, la diferencia queda nula; no se imputa
un valor silenciosamente. Los identificadores, valores individuales y metadatos
de procedencia se conservan para auditoría, pero no forman parte automáticamente
de la entrada del modelo.

La regla temporal es estricta: para un partido cuya fecha fuente es `D`, todos
los estados deportivos y rankings cumplen:

```text
fecha_observación < D
```

`tourney_date` suele ser el inicio aproximado del torneo, no el día real de la
ronda. El esquema v2 aplica la política compartida
`sackmann-tourney-start-embargo-v1`:

```text
result_available_date = tourney_date + 21 días
resultado_utilizable_en_D ⇔ result_available_date < D
```

La igualdad queda excluida: un resultado fechado en `T` entra por primera vez
en un vector de `T+22`. El constructor recorre `tourney_date` en orden, genera
el vector de cada partido mediante una previsualización que no muta el estado y
mantiene sus resultados en una cola pendiente. Antes de abrir un bloque `D`,
solo aplica las colas con `result_available_date < D` a Elo, forma, H2H y
descanso. No usa `round`, `match_num`, el orden del CSV ni la posición de una
fila para inventar orden intratorneo.

El Parquet conserva ambas columnas: `match_date`/`tourney_date` como fecha
fuente del ejemplo y `result_available_date` como frontera causal de su label.
El embargo es conservador, pero no garantiza cubrir torneos excepcionalmente
largos o reprogramados; esa incertidumbre residual se declara en la auditoría.

Los universos `M` y `F` permanecen completamente separados. Dentro de cada
género, todos los niveles elegibles alimentan un único estado histórico.

## Parámetros versionados

Los valores aprobados se declaran en `src/features/parameters.py`, forman parte
del fingerprint del dataset y no se ocultan en el código:

| Parámetro | Valor | Uso |
|---|---:|---|
| `recent_matches` | 10 | Forma en los últimos partidos |
| `recent_months` | 3 | Forma en meses naturales |
| `age_reference_years` | 30,0 | Referencia de la transformación `Age.30` |
| `days_per_year` | 365,2425 | Conversión de días a edad decimal |
| `orientation_seed` | 42 | Orientación estable de A/B |

El esquema publicado es `tennis-features-v2`.

## Orientación reproducible de A y B

La fuente histórica identifica ganador y perdedor, lo que filtraría la etiqueta
si se conservara ese orden. Cada registro se orienta independientemente con:

```text
orientation_hash =
    SHA256(JSON_compacto((42, gender, source_record_hash)))
swapped = bit_menos_significativo(orientation_hash)
```

`source_record_hash` es el SHA-256 de las 49 cadenas crudas de la fila Sackmann,
antes de tipar o normalizar. Si `swapped` es falso, `A` es el ganador y `y=1`;
si es verdadero, `A` es el perdedor y `y=0`. La orientación no depende de un
generador secuencial: reordenar filas o añadir partidos posteriores no cambia
una decisión ya tomada.

La etiqueta es, por definición, equivalente a «A ganó». La comprobación de
balance correcta es que el ganador quede asignado a A aproximadamente en el
50 % de las filas, no que `y` sea independiente de ese hecho.

## Familias de features

### Elo general y por superficie

La fase 6 reutiliza exactamente el motor de la fase 3. Cada jugador comienza en
1500 y mantiene un Elo general y un Elo puro para `Hard`, `Clay`, `Grass` y
`Carpet`. Para ratings `R_i` y `R_j`:

```text
E_i = 1 / (1 + 10 ** ((R_j - R_i) / 400))
K_i(n_i) = 250 / (n_i + 5) ** 0.4
R_i' = R_i + K_i(n_i) * (S_i - E_i)
```

`n_i` solo cuenta resultados elegibles cuya fecha de disponibilidad sea
estrictamente anterior al corte. El general se actualiza con todo partido
elegible cuando termina su embargo; el estado de superficie solo con partidos
de esa superficie. La mezcla usada como Elo efectivo de superficie es:

```text
R_superficie_efectivo =
    0,5 * R_general + 0,5 * R_superficie_puro
```

Si no existe superficie, el rating efectivo es el general. El dataset conserva:

- Elo general de A/B y su diferencia;
- Elo puro de superficie de A/B y su diferencia;
- Elo efectivo combinado de A/B y su diferencia;
- contadores general y de superficie, sus diferencias y flags de *cold start*.

En los nombres de columnas, `elo_surface_raw_*` es el Elo puro y
`elo_surface_*` es el combinado 50/50. Los dos géneros nunca comparten estado.

### Forma reciente

Para cada jugador se almacenan bloques completos de resultados ya disponibles,
con número de victorias y partidos:

```text
win_rate = victorias / partidos
```

La ventana de últimos `N=10` partidos se recorre desde la fecha anterior más
reciente. Como no hay orden fiable dentro de una fecha, se incorpora entero el
bloque diario que cruza el límite. Por tanto, `recent_n_matches_*` puede ser
mayor que 10; ese contador hace visible el tamaño real del denominador.

La ventana de `M=3` meses usa la fecha fuente de los resultados que ya superaron
el embargo:

```text
[D - 3 meses naturales, D)
```

La resta usa meses de calendario y, si el día no existe en el mes de destino,
lo acota a su último día. Para un jugador sin historia, la tasa es `NaN` y el
contador es cero. Se guardan tasas y contadores individuales, y las diferencias
de tasas A−B.

### Head-to-head

El balance global as-of es:

```text
h2h_global_balance =
    (victorias_A - victorias_B) / enfrentamientos
```

La versión por superficie aplica la misma fórmula usando solo esa superficie.
Sin enfrentamientos, el balance es `0,0` y el contador es cero. Si la superficie
del partido es desconocida, el balance por superficie es nulo y su contador es
cero. El signo se recalcula para la orientación A/B, con independencia del
orden estable usado internamente para almacenar el par.

### Descanso

Para cada lado, considerando solo resultados cuyo embargo ya terminó:

```text
rest_days = D - fecha_último_partido_anterior
rest_days_diff = rest_days_A - rest_days_B
```

El valor es nulo si el jugador aún no tiene un partido elegible anterior, y la
diferencia es nula si falta cualquiera de los lados. Como el estado de `D` se
aplica al cerrar el bloque, no se generan descansos cero por partidos de la
misma fecha.

### Ranking y puntos

El índice de rankings está separado por género y obtiene por búsqueda binaria
el último snapshot que cumple:

```text
ranking_date < D
```

Se guardan fecha, `rank`, `rank_points`, antigüedad
`D - ranking_date`, flags de ausencia, diferencias A−B y el número de fechas
conflictivas que hubo que saltar. En `rank_diff`, un valor positivo significa
que A tiene un número de ranking peor que B.

Los duplicados exactamente iguales para una misma clave
`(player_id, ranking_date)` se colapsan. Si esa clave contiene valores
incompatibles de ranking, puntos o —en WTA— número de torneos, se ponen en
cuarentena todas sus observaciones. Nunca se escoge «la primera», «la última»,
la mejor ni un promedio: el índice retrocede al snapshot limpio anterior. Si no
existe, ranking y puntos quedan nulos. El inventario reproducible se publica en
`data/processed/features_active/runs/<fingerprint>/ranking_conflicts.csv`.

### Edad y `Age.30`

La edad decimal as-of se calcula desde la fecha de nacimiento:

```text
age = (D - birth_date).days / 365,2425
age_diff = age_A - age_B
Age.30_i = abs(age_i - 30)
age_distance_30_diff = Age.30_A - Age.30_B
```

Esta es la transformación `Age.30` de la literatura: penaliza la distancia a
30 en ambos sentidos y representa un intervalo óptimo aproximado de 28–32 años.
La definición se basa en el artículo primario
[Machine learning for the prediction of professional tennis matches](https://journals.sagepub.com/doi/10.3233/JSA-240670).

Una fecha de nacimiento ausente queda nula. Si `birth_date >= D`, se marca
`age_invalid_for_date`, la edad se degrada a nulo y la construcción continúa;
no se fuerza una edad imposible.

### Contexto del partido y auditoría de niveles

Se conservan `surface`, `best_of`, `round`, `source_family`,
`tour_level_raw` y una categoría `tour_level` normalizada. La allowlist es
cerrada sobre las 54 combinaciones `(source_family, tourney_level)` observadas
en el snapshot fuente activo:

| Familia fuente | Códigos crudos | Categoría canónica |
|---|---|---|
| `atp_main` | `A`, `F`, `M` | `ATP Tour` |
| `atp_main` | `G` | `Grand Slam` |
| `atp_main` | `D` | `Team` |
| `atp_main` | `O` | `Other` |
| `atp_qual_chall` | `A`, `M` | `ATP Tour` |
| `atp_qual_chall` | `C` | `Challenger` |
| `atp_qual_chall` | `G` | `Grand Slam` |
| `atp_futures` | `S`, `15`, `25` | `ITF` |
| `wta_main` | `F`, `I`, `P`, `PM`, `T1`, `T2`, `T3`, `T4`, `T5`, `W` | `WTA Tour` |
| `wta_main` | `G` | `Grand Slam` |
| `wta_main` | `D` | `Team` |
| `wta_main` | `35+H`, `50+H`, `CC` | `ITF` |
| `wta_main` | `E`, `J`, `O` | `Other` |
| `wta_qual_itf` | `I`, `P`, `PM`, `T1`, `T2`, `T3`, `T4`, `T5`, `W` | `WTA Tour` |
| `wta_qual_itf` | `G` | `Grand Slam` |
| `wta_qual_itf` | `C` | `Challenger` |
| `wta_qual_itf` | `10`, `100`, `15`, `20`, `25`, `35`, `40`, `50`, `60`, `75`, `80` | `ITF` |
| `wta_qual_itf` | `E` | `Other` |

El inventario contiene 6 + 4 + 3 + 18 + 23 = 54 combinaciones. No existe una
regla comodín: una combinación futura no auditada detiene la construcción con
un error claro. El código crudo nunca se pierde. `C` femenino agrupa circuitos
secundarios históricos y WTA 125 modernos; además, el snapshot contiene 31
filas de Buenos Aires 125 de 2024 codificadas como `W`. Esas particularidades
son la razón de conservar también `tour_level_raw`.

Los niveles `E` y `J`, aunque se pueden describir como contexto, son excluidos
por la elegibilidad Elo y no producen filas de entrenamiento. También se
excluyen walkovers (`W/O`, `Walkover`), `BYE`, auto-partidos y duplicados
exactos. `RET`, `DEF`, `ABD` y `ABN` sí se conservan: acreditan un partido
iniciado con ganador oficial y solo actualizan el estado cuando termina su
embargo.

La cuarentena DOB de identidades **no filtra el histórico**. Su fecha de
nacimiento procede del maestro actual y no permite saber cuándo se descubrió
una colisión; usarla como selector retrospectivo sería otra fuga. El manifiesto
publica esas claves solo como diagnóstico ligado al artefacto y declara
`historical_exclusion_rule=disabled_noncausal_dob_metadata`. La inferencia
operativa actual puede bloquear o degradar una clave ya conocida, pero Elo,
features, backtest y reentreno conservan todas sus filas históricas. Esto evita
reescribir el pasado, aunque no elimina la posible contaminación por IDs
reutilizados.

### Cuotas, mercado y edge

Para dos cuotas decimales completas y válidas, ambas estrictamente mayores que
uno:

```text
q_A = 1 / odds_A
q_B = 1 / odds_B
overround = q_A + q_B
margin = overround - 1
p_market_A = q_A / overround
p_market_B = q_B / overround
```

Así, las probabilidades de-vigadas suman uno. Si falta una sola cuota, todos los
campos de mercado quedan nulos: no se conserva un lado parcial. Una observación
de mercado exige timestamps con zona horaria y el corte:

```text
market_retrieved_at_utc < prediction_as_of_utc
```

La igualdad y las cuotas recuperadas después del instante de predicción se
rechazan. Cuando existan ambas probabilidades:

```text
edge = model_probability_A - market_probability_A
```

Sackmann no contiene cuotas históricas verificables. Por ello, en los Parquet
de esta fase `odds_*`, probabilidades de mercado, `model_probability_a` y
`edge` son nulos. Las fases 7–8 los rellenarán solo cuando exista una
observación causal trazable; no se fabrican cuotas retroactivas.

## Allowlist de entrada al modelo

La constante `MODEL_FEATURE_COLUMNS` es la única lista autorizada como entrada
de la fase 7:

```text
surface, tour_level, tour_level_raw, best_of, round,
elo_general_diff, elo_surface_diff, elo_surface_raw_diff,
elo_general_matches_diff, elo_surface_matches_diff,
elo_cold_start_a, elo_cold_start_b,
recent_n_win_rate_diff, recent_n_matches_a, recent_n_matches_b,
recent_months_win_rate_diff, recent_months_matches_a,
recent_months_matches_b,
h2h_global_balance, h2h_global_matches,
h2h_surface_balance, h2h_surface_matches,
rest_days_a, rest_days_b, rest_days_diff,
rank_diff, rank_points_diff, ranking_missing_a, ranking_missing_b,
ranking_age_days_a, ranking_age_days_b,
ranking_conflict_dates_skipped_a, ranking_conflict_dates_skipped_b,
age_diff, age_distance_30_diff, age_missing_a, age_missing_b,
odds_a, odds_b, market_probability_a, market_probability_b,
market_overround, market_margin
```

Quedan fuera de esa allowlist `record_id`, procedencia, nombres, IDs de
jugadores, fechas de nacimiento, valores individuales que no se han aprobado
como features, `y`, `model_probability_a` y `edge`. Esto impide que columnas de
auditoría o el propio resultado entren por selección accidental de «todas las
columnas».

## Construcción, persistencia e idempotencia

El constructor:

1. verifica el manifiesto Sackmann, el commit y los blobs requeridos;
2. vuelca cada género a un SQLite temporal ordenable sin cargar todo el
   histórico en memoria;
3. previsualiza cada bloque fuente y aplica únicamente resultados cuyo embargo
   ya terminó, con el mismo contrato Elo v5;
4. escribe por buffers un Parquet Zstandard con esquema Arrow explícito;
5. verifica un preflight barato de escritura, `fsync`, rename y borrado;
6. mueve el directorio publicable completo a un run inmutable;
7. verifica hashes y sustituye atómicamente el puntero activo al final.

Los artefactos son:

```text
data/processed/features_active/manifest.json
data/processed/features_active/runs/<fingerprint>/manifest.json
data/processed/features_active/runs/<fingerprint>/training_M.parquet
data/processed/features_active/runs/<fingerprint>/training_F.parquet
data/processed/features_active/runs/<fingerprint>/ranking_conflicts.csv
```

Cada `record_id` es el `source_record_hash` estable. El Parquet también conserva
`source_commit`, `source_path` y `source_row_number`. El fingerprint SHA-256
incluye commit, inventario y blobs fuente, géneros, parámetros de features,
política de fecha fuente, contrato Elo v5 exacto, esquema completo, allowlist
del modelo e inventario de código. También fija el snapshot diagnóstico de
identidades para que el contexto operativo sea reproducible; esas claves no
seleccionan filas históricas. El manifiesto declara de forma explícita que la
exclusión histórica está desactivada.

Sin `--force`, si fingerprint, tamaños y SHA-256 de todos los artefactos
publicados coinciden, la ejecución verifica el run y lo reactiva sin
recalcular. Con `--force` se reconstruye en staging y se exige que el mismo
fingerprint produzca exactamente el mismo manifiesto salvo el timestamp; nunca
se sobrescribe un run. Un build incompleto o una ACL dañada en el puntero no
mezclan artefactos ni reemplazan al activo anterior. Todas las rutas recibidas
por la API/CLI se restringen al árbol `TENNIS/`.

El loader de entrenamiento y el contexto diario recalculan
`FEATURE_CODE_PATHS` y exigen igualdad exacta con `code_inventory` del run. Un
inventario ausente, mal formado o con cualquier hash distinto falla cerrado y
obliga a reconstruir features; una subida de versión declarativa no puede
ocultar que las fórmulas ejecutables cambiaron.

Desde `TENNIS/`, en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\build_features.py
```

También se puede ejecutar sin activar:

```powershell
.\.venv\Scripts\python.exe scripts\build_features.py
```

Opciones principales:

```text
--gender {M,F,all}
--manifest PATH
--raw-dir PATH
--output-dir PATH
--chunksize INTEGER_POSITIVO
--parquet-buffer-rows INTEGER_POSITIVO
--force
```

`--force` reconstruye deliberadamente un fingerprint ya publicado; no cambia
por sí mismo parámetros ni fuentes.

El directorio canónico `data/processed/features_active` solo admite
`--gender all`. Una construcción parcial debe usar un `--output-dir`
diagnóstico distinto dentro de `TENNIS/`; así no puede sustituir el manifiesto
activo de los dos universos con un único género.

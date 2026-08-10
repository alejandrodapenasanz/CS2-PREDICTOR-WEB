# Diccionario de datos

## Procedencia y alcance

Los campos fuente proceden de los CSV de Jeff Sackmann conservados en el
snapshot autorizado
[`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive).
La atribución original corresponde a Jeff Sackmann / Tennis Abstract. Los datos
se distribuyen con licencia **CC BY-NC-SA 4.0**: requieren atribución, permiten
solo uso no comercial y exigen compartir las redistribuciones bajo la misma
licencia.

La fase 2 carga partidos individuales de singles de estas familias:

| `source_family` | Género | Archivos | Contenido |
|---|---:|---|---|
| `atp_main` | `M` | `atp_matches_YYYY.csv` | Cuadros principales ATP y competiciones incluidas por Sackmann. |
| `atp_qual_chall` | `M` | `atp_matches_qual_chall_YYYY.csv` | Qualifying de tour y cuadros principales Challenger. |
| `atp_futures` | `M` | `atp_matches_futures_YYYY.csv` | Futures, Satellites e ITF masculinos. |
| `wta_main` | `F` | `wta_matches_YYYY.csv` | Cuadros principales WTA y competiciones incluidas por Sackmann. |
| `wta_qual_itf` | `F` | `wta_matches_qual_itf_YYYY.csv` | Qualifying WTA, ITF y qualifying ITF disponible. |

No se incluyen dobles ni `atp_matches_amateur.csv`, porque quedan fuera del
alcance de singles ATP/Challenger/ITF y WTA/ITF definido para el predictor.

## Columnas derivadas de partidos

| Columna | Tipo del loader | Descripción |
|---|---|---|
| `gender` | `string` | Universo de género: `M` para ATP masculino y `F` para WTA femenino. Es parte de la identidad del jugador y nunca debe mezclarse entre universos. |
| `tour_level` | `string` | Copia textual exacta de `tourney_level`. No colapsa códigos históricos ni importes ITF. |
| `source_family` | `string` | Familia de archivo de la tabla anterior; permite distinguir qualifying, Challenger e ITF sin inferirlo desde el código del torneo. |
| `source_file` | `string` | Nombre del CSV anual del que procede la fila. |
| `source_anomaly` | `string` | Marca una excepción fuente aprobada y documentada; es nula para las demás filas. |

## Identificación del torneo y del partido

| Columna | Tipo del loader | Descripción |
|---|---|---|
| `tourney_id` | `string` | Identificador del torneo. Los primeros cuatro caracteres suelen representar el año, pero el resto no tiene un formato único. |
| `tourney_name` | `string` | Nombre del torneo según la fuente. |
| `surface` | `string` | Superficie indicada por Sackmann, cuando está disponible. |
| `draw_size` | `Int64` | Tamaño nominal del cuadro; puede estar redondeado a una potencia de dos. La excepción histórica documentada al final de este archivo se carga como nulo. |
| `tourney_level` | `string` | Código de nivel original de Sackmann, preservado sin reinterpretación. |
| `tourney_date` | `datetime64[ns]` | Fecha `YYYYMMDD` convertida a fecha real. Normalmente es el lunes o inicio aproximado del torneo, no necesariamente el día del partido. |
| `match_num` | `Int64` | Identificador del partido dentro del torneo; puede ser secuencial, descendente o arbitrario. No implica orden cronológico. |
| `score` | `string` | Marcador publicado, incluidos indicadores como `RET` o `W/O`. |
| `best_of` | `Int64` | Número máximo de sets, normalmente 3 o 5. |
| `round` | `string` | Código de ronda original, por ejemplo `Q1`, `R32`, `QF`, `SF`, `F` o `RR`. |
| `minutes` | `Int64` | Duración del partido en minutos, si está disponible. |

## Nivel del torneo

`tour_level` conserva exactamente el código fuente de `tourney_level`. Esto es
importante porque el mismo fichero puede contener varios niveles y porque los
Futures/ITF modernos usan importes numéricos en lugar de una sola etiqueta.

El diccionario original de Sackmann documenta:

| Código | Definición documentada |
|---|---|
| `G` | Grand Slam. |
| `M` | Masters 1000 masculino. |
| `A` | Otros eventos masculinos de nivel tour. |
| `C` | Challenger masculino; también aparece en datos WTA y se conserva sin ampliar su significado. |
| `S` | Satellite/ITF masculino histórico. |
| `F` | Finals y otros eventos de cierre de temporada. |
| `D` | Davis Cup en hombres; Fed/Federation/Billie Jean King Cup y algunas competiciones por equipos históricas en mujeres. |
| `P` | Premier WTA. |
| `PM` | Premier Mandatory WTA. |
| `I` | International WTA. |
| `T1`…`T5` | Categorías históricas Tier I…Tier V de WTA. |
| Valor numérico (`10`, `15`, `25`, etc.) | Nivel ITF expresado por Sackmann mediante el premio en miles. Se conserva como texto. |
| `E` | Exhibición. |
| `J` | Juniors. |
| `T` | Team tennis, reservado por el diccionario aunque no aparecía allí todavía. |

El snapshot contiene además códigos no definidos por el diccionario preservado:

| Código observado | Género/familia observada | Tratamiento |
|---|---|---|
| `O` | ATP y WTA; únicamente eventos olímpicos en la auditoría | Se conserva como `O`; la asociación olímpica es una observación, no una recodificación oficial. |
| `W` | WTA principal y qualifying/ITF | Se conserva como valor opaco; su uso histórico no representa una jerarquía estable. |
| `CC` | WTA entre 1996 y 2003 | Se conserva como valor opaco. |
| `35+H`, `50+H` | WTA 2025 | Se conservan literalmente como texto. |

Los niveles numéricos observados son `15` y `25` en ATP, y `10`, `15`, `20`,
`25`, `35`, `40`, `50`, `60`, `75`, `80` y `100` en WTA. No se deduce el
significado de ningún código opaco desde el nombre del torneo.

## Participantes y rankings dentro de cada partido

Las columnas con prefijo `winner_` describen al ganador; las equivalentes con
prefijo `loser_` describen al perdedor.

| Sufijo/columna | Tipo del loader | Descripción |
|---|---|---|
| `winner_id`, `loser_id` | `Int64` | Identificador Sackmann del jugador. La clave segura es `(gender, player_id)`. |
| `winner_seed`, `loser_seed` | `string` | Cabeza de serie, si está disponible. |
| `winner_entry`, `loser_entry` | `string` | Tipo de entrada, por ejemplo `WC`, `Q`, `LL`, `PR` o `ITF`. |
| `winner_name`, `loser_name` | `string` | Nombre del jugador incorporado en el CSV del partido. |
| `winner_hand`, `loser_hand` | `string` | Mano: `R`, `L`, `U` u otro valor original de la fuente. |
| `winner_ht`, `loser_ht` | `Int64` | Altura en centímetros, si está disponible. |
| `winner_ioc`, `loser_ioc` | `string` | Código de país de tres caracteres. |
| `winner_age`, `loser_age` | `Float64` | Edad en años calculada por la fuente respecto a `tourney_date`. |
| `winner_rank`, `loser_rank` | `Int64` | Ranking ATP/WTA en `tourney_date` o en la fecha de ranking más reciente anterior disponible. |
| `winner_rank_points`, `loser_rank_points` | `Int64` | Puntos del ranking asociados, si están disponibles. |

Los rankings embebidos no demuestran por sí solos el requisito temporal estricto
`ranking_date < D`. Las fases de features deberán reconstruir esa unión desde los
CSV de rankings y exigir una fecha estrictamente anterior a la predicción.

## Estadísticas del partido

Los prefijos `w_` y `l_` significan ganador y perdedor. Son totales enteros, no
porcentajes.

| Sufijo | Tipo del loader | Descripción |
|---|---|---|
| `ace` | `Int64` | Aces. |
| `df` | `Int64` | Dobles faltas. |
| `svpt` | `Int64` | Puntos jugados al servicio. |
| `1stIn` | `Int64` | Primeros servicios dentro. |
| `1stWon` | `Int64` | Puntos ganados con primer servicio. |
| `2ndWon` | `Int64` | Puntos ganados con segundo servicio. |
| `SvGms` | `Int64` | Juegos al servicio. |
| `bpSaved` | `Int64` | Puntos de break salvados. |
| `bpFaced` | `Int64` | Puntos de break afrontados. |

Por tanto, las columnas completas son `w_ace`, `w_df`, `w_svpt`, `w_1stIn`,
`w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced` y sus equivalentes
`l_...`.

Estas estadísticas describen el propio partido y no podrán utilizarse como
features para predecir ese mismo partido.

## Maestro de jugadores

El loader conserva los ocho campos reales y añade dos columnas explícitas:

| Columna | Tipo del loader | Descripción |
|---|---|---|
| `player_id` | `Int64` | Identificador Sackmann. Debe combinarse con `gender` como clave. |
| `name_first` | `string` | Nombre, que puede faltar. |
| `name_last` | `string` | Apellidos, que pueden faltar. |
| `hand` | `string` | Mano original (`R`, `L`, `U`, `A` u otro valor publicado). |
| `dob` | `datetime64[ns]` | Fecha de nacimiento. Fechas parciales o calendáricamente inválidas se convierten en `NaT` sin corregirlas. |
| `ioc` | `string` | Código de país de tres caracteres. |
| `height` | `Int64` | Altura en centímetros, si está disponible. |
| `wikidata_id` | `string` | Identificador de Wikidata, si está disponible. |
| `player_name` | `string` | Unión limpia de `name_first` y `name_last`, conservando nombres parciales. |
| `gender` | `string` | `M` o `F`, derivado exclusivamente del maestro ATP/WTA de origen. |

## Nota temporal

`tourney_date` identifica normalmente el inicio de la semana del torneo. No
permite ordenar con seguridad las rondas dentro del mismo día o evento. Elo y
features conservan esa fecha como `source_date`, pero el resultado solo se
vuelve utilizable en `result_available_date=source_date+21 días` y únicamente
si `result_available_date < D`. Los partidos con la misma fecha de
disponibilidad se congelan y aplican como un bloque; una consulta para `D` solo
recupera estados efectivos con `state_date < D`. No se infiere orden a partir
de la ronda, `match_num`, el orden del CSV ni el nombre del archivo.

## Excepción fuente aprobada

La auditoría de los 262 CSV y 1.749.872 filas encontró una única celda no
convertible dentro de las columnas numéricas:

| Archivo | Fila CSV | Torneo | Columna | Valor crudo |
|---|---:|---|---|---|
| `wta/wta_matches_1970.csv` | 1819 | `Rondebosch Exho` (`1970-1093`) | `draw_size` | `exho` |

La estructura de esa fila tiene exactamente las 49 columnas esperadas y
`tourney_level = E`; no es un desplazamiento del CSV. Con autorización expresa,
el loader convierte solo esa celda a `pd.NA` y asigna
`source_anomaly = "draw_size_exho_normalized_to_missing"`. La identificación
exige simultáneamente archivo, torneo, nombre, nivel, fecha, número de partido y
valor literal. Cualquier otro `draw_size="exho"` continúa provocando
`SchemaMismatchError`. El CSV de `data/raw` nunca se modifica.

## Histórico Elo propio

La fase 3 genera `data/processed/elo/elo.sqlite3` desde los partidos canónicos y
el manifiesto Sackmann verificado. Este artefacto procesado se ignora en Git y
puede reconstruirse con `python scripts/build_elo.py`. El cálculo mantiene runs
independientes por género y nunca mezcla un mismo `player_id` entre `M` y `F`.
La fórmula, los parámetros por defecto, el tratamiento por superficie y los
motivos de exclusión están documentados en [`elo.md`](elo.md).

Los parámetros configurables se persisten en `runs.parameters_json`:

| Clave | Valor por defecto | Uso |
|---|---:|---|
| `initial_rating` | `1500.0` | Estado inicial general y de cada superficie. |
| `scale` | `400.0` | Escala de la expectativa logística. |
| `k_numerator` | `250.0` | Numerador de `K(n)`. |
| `k_offset` | `5.0` | Desplazamiento del contador previo en `K(n)`. |
| `k_exponent` | `0.4` | Exponente de decaimiento de `K(n)`. |
| `surface_weight` | `0.5` | Peso del Elo puro de superficie en el rating combinado. |

### Tablas SQLite

SQLite usa journal `DELETE`, un único escritor y cuatro tablas principales:

| Tabla | Clave | Contenido |
|---|---|---|
| `runs` | `run_id` | Género, commit fuente, versión del algoritmo, fingerprint de entrada, parámetros JSON, estado `building`/`complete`/`failed`, timestamps, rango de fechas y motivo de fallo. |
| `date_blocks` | (`run_id`, `block_date`) | Checkpoint de cada fecha: hash de entrada, partidos incluidos y exclusiones. |
| `rating_history` | (`run_id`, `gender`, `player_id`, `state_date`) | Estado ancho posterior a cerrar una fecha, con Elo general, cuatro Elo puros de superficie y sus cinco contadores. |
| `active_runs` | `gender` | Puntero al único run completo activo de cada universo y fecha de activación. |

Una fila de `rating_history` representa el estado posterior al bloque
`state_date`. Su contrato es:

| Columna | Tipo SQLite | Descripción |
|---|---|---|
| `run_id` | `TEXT` | Ejecución reproducible a la que pertenece el estado. |
| `gender` | `TEXT` | `M` o `F`; debe coincidir con el género del run. |
| `player_id` | `INTEGER` | Identificador Sackmann, único solo junto con `gender`. |
| `state_date` | `TEXT` | Fecha ISO del bloque ya cerrado. Solo es consultable si es estrictamente anterior al corte solicitado. |
| `general_elo` | `REAL` | Elo actualizado con todos los partidos elegibles. |
| `hard_elo`, `clay_elo`, `grass_elo`, `carpet_elo` | `REAL` | Elo puro acumulado para cada superficie. |
| `general_matches` | `INTEGER` | Partidos elegibles acumulados en el Elo general. |
| `hard_matches`, `clay_matches`, `grass_matches`, `carpet_matches` | `INTEGER` | Partidos elegibles acumulados en cada Elo puro de superficie. |

La escritura de cada bloque es transaccional y exige fechas estrictamente
crecientes dentro del run. Un run en construcción no es consultable ni puede
reemplazar al activo hasta completarse y activarse en una sola transacción.

### Contrato de `get_elo` y `get_elos`

`get_elo()` devuelve un `EloSnapshot` para una consulta. `get_elos()` recibe un
iterable de `EloQuery`, devuelve una tupla en el mismo orden y agrupa los
accesos por género y run. Ambas funciones aceptan opcionalmente `db_path` y
`run_id`; sin `run_id`, usan el run completo activo de cada género.

La semántica temporal es estricta:

```text
state_date < as_of_date
```

No se admite `state_date <= as_of_date`. `as_of_date` debe ser
`datetime.date`, no texto ni `datetime`. `gender` debe ser exactamente `M` o
`F`; `surface` admite `Hard`, `Clay`, `Grass`, `Carpet` o `None`, normalizando
solo la capitalización.

| Campo de `EloSnapshot` | Tipo Python | Descripción |
|---|---|---|
| `gender` | `Literal["M", "F"]` | Universo consultado. |
| `player_id` | `int` | Jugador consultado. |
| `as_of_date` | `date` | Corte exclusivo solicitado. |
| `state_date` | `date \| None` | Fecha del último estado estrictamente anterior; nula en cold start. |
| `general_elo` | `float` | Elo general efectivo en el estado recuperado. |
| `surface` | `str \| None` | Superficie canónica solicitada. |
| `surface_elo_raw` | `float \| None` | Elo puro de superficie; nulo cuando `surface=None`. |
| `combined_elo` | `float` | Mezcla configurada de Elo general y de superficie; coincide con el general cuando `surface=None`. |
| `general_matches` | `int` | Contador general previo al corte. |
| `surface_matches` | `int` | Contador de la superficie solicitada; cero cuando `surface=None`. |
| `is_cold_start` | `bool` | Indica ausencia de partidos generales anteriores al corte. |
| `run_id` | `str \| None` | Ejecución completa usada para la respuesta. |
| `source_commit` | `str \| None` | Commit Sackmann fijado por esa ejecución. |

Si no existe historia anterior, la API devuelve los valores iniciales
persistidos en el run, contadores cero e `is_cold_start=True`. Las suites
`test_elo_engine.py`, `test_elo_pipeline.py` y `test_elo_store.py` cubren la
fórmula, el aislamiento, la regla temporal, la reproducibilidad, la
persistencia transaccional y las consultas individual y por lotes.

## Cartelera diaria de Tennis Explorer

`get_daily_matches()` devuelve únicamente singles ATP, WTA, Challenger e ITF
de la fecha solicitada. Este DataFrame diario no se concatena directamente con
el histórico Sackmann: aunque ambos usan `gender` y `tour_level`, el histórico
conserva códigos fuente detallados y la cartelera usa las cuatro etiquetas
operativas `ATP`, `WTA`, `Challenger` e `ITF`.

| Columna | Tipo pandas | Descripción |
|---|---|---|
| `match_date` | `datetime64[ns]` | Fecha diaria exacta solicitada a Tennis Explorer. |
| `tournament` | `string` | Texto visible del encabezado del torneo. En Futures masculinos puede ser el agregado literal `Futures 2026`. |
| `tournament_href` | `string` | Ruta relativa del torneo; nula cuando la fuente no publica enlace. |
| `tour_level` | `string` | Nivel diario normalizado: `ATP`, `WTA`, `Challenger` o `ITF`. |
| `gender` | `string` | `M` para singles masculino o `F` para singles femenino, derivado del marcador HTML del torneo. |
| `surface` | `string` | `Hard`, `Clay`, `Grass` o `Carpet`, anulable. Se une por `tournament_href` desde el catálogo del mismo HTML; nunca se infiere por calendario. `Indoors` queda nulo porque acredita recinto, no material. |
| `scheduled_time` | `string` | Hora local visible, anulable. No se convierte a UTC porque la celda no aporta una zona inequívoca. |
| `player_1_name`, `player_2_name` | `string` | Textos visibles de las dos anclas de jugador, sin añadir la cabeza de serie situada fuera del enlace. |
| `player_1_href`, `player_2_href` | `string` | Rutas relativas `/player/<slug>/`; nulas cuando la fuente no enlaza al participante. |
| `player_1_slug`, `player_2_slug` | `string` | Segmentos estables extraídos de las rutas de jugador y destinados al mapeo de la fase 5. |
| `player_1_has_link`, `player_2_has_link` | `boolean` | Indican explícitamente si la fuente proporcionó una ficha enlazada. |
| `player_1_odds`, `player_2_odds` | `Float64` | Cuotas decimales H/A en el orden de las filas; nulas si la celda estaba vacía. |
| `status` | `string` | `scheduled`, `in_progress`, `finished`, `walkover`, `cancelled` o `unknown`, siempre con criterio conservador. |
| `status_evidence` | `string` | Evidencia directa o motivo de ambigüedad usado para asignar `status`. |
| `player_1_sets_won`, `player_2_sets_won` | `Int64` | Contadores numéricos observados; pueden representar un marcador parcial y ser nulos. |
| `sets_score` | `string` | Par de sets observado en orientación de la página, por ejemplo `2-1`; anulable. |
| `winner_side` | `string` | `player_1` o `player_2` solo para un final convencional con evidencia completa y slug; anulable. |
| `winner_slug` | `string` | Slug del ganador demostrable; nunca se infiere para walkover, cancelación, retirada no caracterizada o marcador parcial. |
| `result_evidence` | `string` | Motivo auditable por el que existe o no existe `winner_slug`. |
| `source_match_id` | `string` | Identificador fuente opaco extraído del parámetro `id` del enlace de detalle. |
| `match_detail_href` | `string` | Ruta relativa obligatoria `/match-detail/?id=...`; su ausencia se trata como cambio de esquema. |
| `source_url` | `string` | URL `type=all` exacta usada para la fecha. |
| `retrieved_at_utc` | `datetime64[ns, UTC]` | Instante de adquisición del HTML. |
| `snapshot_sha256` | `string` | SHA-256 de los bytes HTML validados. |

Una fila con `status="unknown"` no debe reinterpretarse posteriormente como
programada, cancelada o en juego sin una nueva evidencia fechada. Las cuotas
solo estuvieron demostrablemente disponibles en `retrieved_at_utc`; no pueden
usarse para reconstruir predicciones anteriores a ese instante.

Los dobles y UTR se excluyen. La página diaria no desagrega la sede ni la
superficie de los Futures masculinos y el proyecto no las infiere. La
procedencia, los selectores y el contrato completo están en
[`tennis_explorer.md`](tennis_explorer.md).

## Mapeo de identidades Tennis Explorer → Sackmann

La fase 5 usa `(gender, slug)` como clave entre fuentes. Un mismo texto visible
no constituye una identidad y un `player_id` Sackmann solo es seguro junto con
su género. La coincidencia automática se restringe a jugadores del mismo
género con al menos un partido en cualquier nivel dentro de
`[D - 3 años naturales, D)`, y solo acepta una clave exacta y única de apellido
normalizado e inicial. El contrato completo está en
[`player_mapping.md`](player_mapping.md).

### Columnas añadidas a la cartelera

El resultado conserva todas las columnas de `get_daily_matches()` y añade:

| Columna | Tipo pandas | Descripción |
|---|---|---|
| `player_1_id`, `player_2_id` | `Int64` | ID Sackmann anulable del participante. La identidad completa sigue siendo `(gender, player_id)`. |
| `player_1_mapping_method`, `player_2_mapping_method` | `string` | `automatic_name` o `manual_override`; nula cuando el lado no se pudo mapear. Una lectura de caché conserva el método de origen. |
| `mapping_status` | `string` | `mapped` si ambos IDs existen; `unmapped` si falta al menos uno. No se elimina la fila no mapeada. |

Ningún valor nulo se sustituye por un candidato probable.

### Overrides manuales

`data/overrides.csv` es un artefacto versionado con cuatro columnas exactas y
en este orden:

| Columna | Tipo lógico | Descripción |
|---|---|---|
| `gender` | `TEXT` | `M` o `F`; primer componente de la clave. |
| `slug` | `TEXT` | Slug no vacío de Tennis Explorer; segundo componente de la clave. |
| `player_id` | entero positivo | ID que debe existir en el maestro Sackmann del mismo género. |
| `reason` | `TEXT` | Justificación manual no vacía que se copia a la auditoría. |

Las claves `(gender, slug)` duplicadas, las columnas adicionales, el orden
distinto o una fila inválida detienen la carga del archivo. Un override puede
insertar una asociación o corregir la vigente; no modifica el CSV crudo de
ninguna fuente.

### Base `player_mapping.sqlite3`

`data/processed/player_mapping.sqlite3` es una caché procesada ignorada por Git.
SQLite usa journal `DELETE`. La tabla vigente `player_mappings` tiene clave
primaria `(gender, slug)`:

| Columna | Tipo SQLite | Descripción |
|---|---|---|
| `gender` | `TEXT` | Universo `M` o `F`. |
| `slug` | `TEXT` | Identidad estable dentro del género; no admite `/`, query ni fragmento. |
| `player_id` | `INTEGER` | ID Sackmann positivo. |
| `visible_name` | `TEXT` | Último texto de Tennis Explorer persistido con la resolución. |
| `sackmann_player_name` | `TEXT` | Nombre completo del maestro Sackmann validado. |
| `sackmann_ioc` | `TEXT` nullable | IOC Sackmann en mayúsculas; se conserva para auditoría, no se usa como desempate porque la página diaria actual no publica país de jugador. |
| `resolution_method` | `TEXT` | `automatic_name` o `manual_override`. |
| `first_resolved_date` | `TEXT` | Fecha ISO del primer mapping de la clave; una corrección posterior la conserva. |
| `updated_at_utc` | `TEXT` | Timestamp UTC ISO de la última escritura autorizada. |

Una inserción automática nunca actualiza una clave preexistente. La tabla
`mapping_audit` es append-only y sus triggers rechazan `UPDATE` y `DELETE`:

| Columna | Tipo SQLite | Descripción |
|---|---|---|
| `audit_id` | `INTEGER` | Secuencia autoincremental del evento. |
| `event_type` | `TEXT` | `automatic_insert`, `manual_insert` o `manual_correction`. |
| `gender`, `slug` | `TEXT` | Clave afectada. |
| `old_player_id`, `new_player_id` | `INTEGER` nullable/no nulo | ID anterior, si existía, y nuevo ID. |
| `old_visible_name`, `new_visible_name` | `TEXT` nullable/no nulo | Texto anterior y posterior. |
| `old_sackmann_player_name`, `new_sackmann_player_name` | `TEXT` nullable/no nulo | Nombre Sackmann anterior y posterior. |
| `old_sackmann_ioc`, `new_sackmann_ioc` | `TEXT` nullable | IOC anterior y posterior. |
| `old_resolution_method`, `new_resolution_method` | `TEXT` nullable/no nulo | Método anterior y método vigente después del evento. |
| `old_first_resolved_date`, `new_first_resolved_date` | `TEXT` nullable/no nulo | Fecha inicial anterior y posterior. |
| `old_updated_at_utc`, `new_updated_at_utc` | `TEXT` nullable/no nulo | Timestamps de estado anterior y posterior. |
| `reason` | `TEXT` | `automatic_name` para inserciones automáticas o motivo no vacío del override. |
| `changed_at_utc` | `TEXT` | Timestamp UTC del evento de auditoría. |

### Cola `unresolved_players.csv`

`data/processed/unresolved_players.csv` es una vista persistente de pendientes
actuales, ignorada por Git. Se reemplaza atómicamente bajo un lock local y se
deduplica por `(gender, slug)`. Si falta el slug, la clave de revisión usa
género, apellido normalizado e inicial. Si el nombre tampoco es parseable, usa
internamente `unparsed:<texto visible>` solo para deduplicar la revisión.

| Columna | Tipo CSV | Descripción |
|---|---|---|
| `gender` | texto | `M` o `F`. |
| `slug` | texto nullable | Slug de Tennis Explorer; vacío solo cuando la fuente no publicó enlace. |
| `visible_name` | texto | Nombre abreviado más reciente observado. |
| `normalized_last_name` | texto nullable | Apellido tras `Unidecode`, normalización de puntuación y espacios; vacío si el texto no cumple `Apellido I.`. |
| `first_initial` | carácter nullable | Inicial normalizada del nombre; vacía junto con el apellido cuando el texto no es parseable. |
| `reason` | texto | Motivo explícito de no resolución. |
| `candidate_count` | entero no negativo | Número derivado de `candidate_player_ids`. |
| `candidate_player_ids` | lista textual | IDs candidatos separados por `|`, vacía si no había ninguno. |
| `first_seen_date`, `last_seen_date` | fecha ISO | Primera y última fecha diaria en que se observó el pendiente. |
| `observed_dates` | lista textual | Fechas ISO distintas, ordenadas y separadas por `|`. |
| `occurrences` | entero positivo | Número de fechas distintas; repetir la misma fecha no lo incrementa. |
| `last_tour_level` | texto | Nivel operativo de la observación más reciente. |
| `last_tournament` | texto | Torneo de la observación más reciente. |

Cuando una clave se resuelve por caché, coincidencia exacta u override, se
retira de la cola. Así, el archivo representa trabajo pendiente actual y no un
histórico; la historia de mappings resueltos permanece en SQLite.

## Fuentes auxiliares

Match Charting Project y los snapshots Elo de Tennis Abstract se conservan
separados del DataFrame canónico de partidos. Sus esquemas, política de
actualización y restricciones temporales están documentados en
[`source_freshness.md`](source_freshness.md).

### DataFrame Elo externo de Tennis Abstract

`load_latest_elo(gender)` devuelve la última tabla web validada para un único
universo. Estos campos son una referencia externa y no reemplazan los ratings
propios construidos en la fase 3:

| Columna | Tipo | Descripción |
|---|---|---|
| `gender` | `string` | `M` para la página ATP o `F` para la página WTA. |
| `rating_date` | `datetime64[ns]` | Fecha `Last update` publicada en el informe. |
| `elo_rank` | `Int64` | Puesto del Elo general de Tennis Abstract. |
| `player_name` | `string` | Nombre mostrado en la tabla. No es todavía una identidad Sackmann. |
| `player_url` | `string` | URL pública enlazada por la celda del jugador; no se solicita automáticamente. |
| `age` | `Float64` | Edad publicada en años. |
| `elo` | `Float64` | Rating Elo general externo. |
| `hard_elo_rank`, `hard_elo` | `Int64`, `Float64` | Puesto y rating externo para pista dura. |
| `clay_elo_rank`, `clay_elo` | `Int64`, `Float64` | Puesto y rating externo para tierra. |
| `grass_elo_rank`, `grass_elo` | `Int64`, `Float64` | Puesto y rating externo para hierba. |
| `peak_elo` | `Float64` | Máximo Elo publicado. |
| `peak_month` | `datetime64[ns]` | Mes `YYYY-MM` del máximo, representado por su primer día. |
| `official_rank` | `Int64` | Ranking ATP/WTA mostrado en el mismo informe. |
| `log_diff` | `Float64` | Diferencia logarítmica publicada entre ranking oficial y Elo. |
| `source_url` | `string` | Una de las dos URLs auditadas y permitidas. |
| `retrieved_at_utc` | `datetime64[ns, UTC]` | Instante real de adquisición del HTML. |

Para una predicción en `T`, esta tabla solo es admisible si
`retrieved_at_utc < T`. No se utilizará como si hubiera estado disponible en
fechas históricas anteriores.

## Dataset causal de features

La fase 6 publica dos archivos con el mismo esquema Arrow:

- `data/processed/features_active/runs/<fingerprint>/training_M.parquet`;
- `data/processed/features_active/runs/<fingerprint>/training_F.parquet`.

Cada fila representa un partido elegible y se calcula antes de incorporar
cualquier resultado cuya `tourney_date` sea `D`. `MODEL_FEATURE_COLUMNS`
define el allowlist de entradas del futuro modelo. Las columnas de procedencia,
identidad y objetivo permanecen en el Parquet para auditoría, pero no forman
parte de ese allowlist.

### Procedencia, identidad y objetivo

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `record_id` | `string` no nulo | SHA-256 de las 49 cadenas raw Sackmann. Es la misma identidad de copia exacta usada por Elo. |
| `source_commit` | `string` no nulo | Commit inmutable del snapshot Sackmann. |
| `source_path` | `string` no nulo | Ruta POSIX del CSV dentro de `data/raw`. |
| `source_row_number` | `int64` no nulo | Ordinal uno-basado de la fila de datos, sin contar la cabecera. |
| `tourney_id` | `string` nullable | Identificador de torneo conservado de la fuente. |
| `tourney_name` | `string` nullable | Nombre fuente del torneo; solo metadata, no feature del modelo. |
| `match_num` | `string` nullable | Identificador raw del partido dentro del torneo. |
| `gender` | `string` no nulo | Universo `M` o `F`; cada Parquet contiene uno solo. |
| `match_date` | `date32` no nulo | `tourney_date` Sackmann. Suele ser la fecha de inicio del torneo, no la hora real de cada ronda. |
| `player_a_id`, `player_b_id` | `int64` no nulo | Participantes después de la orientación estable A/B. Los IDs no entran al modelo. |
| `y` | `int8` no nulo | `1` si ganó A y `0` si ganó B. |

No se persisten `winner_id`, `loser_id`, `swapped` ni `orientation_hash`.
`record_id` y los campos de procedencia también quedan fuera del allowlist del
modelo.

### Contexto del partido

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `surface` | `string` nullable | `Hard`, `Clay`, `Grass`, `Carpet` o nulo. |
| `tour_level_raw` | `string` no nulo | Código `tourney_level` exacto de Sackmann. |
| `tour_level` | `string` no nulo | Categoría auditada: `Grand Slam`, `ATP Tour`, `WTA Tour`, `Challenger`, `ITF`, `Team` u `Other`. |
| `source_family` | `string` no nulo | `atp_main`, `atp_qual_chall`, `atp_futures`, `wta_main` o `wta_qual_itf`; participa en el mapeo cerrado de nivel. |
| `best_of` | `int64` nullable | Número de sets del formato. Un raw vacío queda nulo; otro valor presente debe ser entero positivo. |
| `round` | `string` nullable | Ronda raw normalizada únicamente para convertir vacío en nulo. |

### Elo prepartido

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `elo_general_a`, `elo_general_b` | `float64` no nulo | Elo general de cada jugador con estados de fecha `< D`. |
| `elo_general_diff` | `float64` no nulo | `elo_general_a - elo_general_b`. |
| `elo_surface_raw_a`, `elo_surface_raw_b` | `float64` nullable | Elo puro de la superficie; nulo si `surface` es nula. |
| `elo_surface_raw_diff` | `float64` nullable | Diferencia A−B de Elo puro; nula si falta cualquier lado. |
| `elo_surface_a`, `elo_surface_b` | `float64` no nulo | Elo efectivo: `0,5 × general + 0,5 × superficie`; sin superficie coincide con el general. |
| `elo_surface_diff` | `float64` no nulo | Diferencia A−B del Elo efectivo usado para esa superficie. |
| `elo_general_matches_a`, `elo_general_matches_b` | `int64` | Partidos previos que alimentaron el Elo general de cada lado. |
| `elo_general_matches_diff` | `int64` | Diferencia A−B de esos contadores. |
| `elo_surface_matches_a`, `elo_surface_matches_b` | `int64` | Partidos previos en la superficie consultada; cero si es nula. |
| `elo_surface_matches_diff` | `int64` | Diferencia A−B de los contadores de superficie. |
| `elo_cold_start_a`, `elo_cold_start_b` | `boolean` | Indican que el jugador aún no tenía partidos Elo anteriores a `D`. |

### Forma, H2H y descanso

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `recent_n_win_rate_a`, `recent_n_win_rate_b` | `float64` | Porcentaje de victorias en los últimos 10 partidos previos. Es `NaN` con cero observaciones. |
| `recent_n_win_rate_diff` | `float64` | Forma de A menos forma de B; puede ser `NaN` en cold start. |
| `recent_n_matches_a`, `recent_n_matches_b` | `int64` | Exposición real de la ventana. Puede superar 10 para conservar completo el bloque de fecha que cruza el límite. |
| `recent_months_win_rate_a`, `recent_months_win_rate_b` | `float64` | Porcentaje de victorias en `[D - 3 meses naturales, D)`. |
| `recent_months_win_rate_diff` | `float64` | Diferencia A−B de la forma en tres meses. |
| `recent_months_matches_a`, `recent_months_matches_b` | `int64` | Partidos observados en la ventana natural de cada jugador. |
| `h2h_global_balance` | `float64` no nulo | `(victorias_A - victorias_B) / enfrentamientos` usando solo fechas `< D`; cero sin H2H. |
| `h2h_global_matches` | `int64` no nulo | Número de enfrentamientos globales previos. |
| `h2h_surface_balance` | `float64` nullable | Balance H2H equivalente limitado a la superficie; nulo si la superficie es desconocida y cero si no hubo enfrentamientos. |
| `h2h_surface_matches` | `int64` | Número de H2H previos en la superficie. |
| `rest_days_a`, `rest_days_b` | `int64` nullable | Días desde el último bloque de partido anterior de cada lado; nulo sin historia. |
| `rest_days_diff` | `int64` nullable | `rest_days_a - rest_days_b`; nulo si falta cualquier lado. |

### Ranking y puntos prepartido

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `ranking_date_a`, `ranking_date_b` | `date32` nullable | Fecha del último snapshot limpio que cumple estrictamente `ranking_date < D`. |
| `rank_a`, `rank_b` | `int64` nullable | Puesto oficial de cada lado; un número menor es mejor. |
| `rank_diff` | `int64` nullable | `rank_a - rank_b`. |
| `rank_points_a`, `rank_points_b` | `int64` nullable | Puntos publicados por Sackmann; pueden faltar en parte del histórico antiguo. |
| `rank_points_diff` | `int64` nullable | `rank_points_a - rank_points_b`. |
| `ranking_missing_a`, `ranking_missing_b` | `boolean` | Indican que no había snapshot limpio anterior al corte. |
| `ranking_age_days_a`, `ranking_age_days_b` | `int64` nullable | Antigüedad en días del snapshot devuelto respecto de `D`. |
| `ranking_conflict_dates_skipped_a`, `ranking_conflict_dates_skipped_b` | `int64` | Fechas conflictivas visibles que se omitieron al retroceder al snapshot limpio. |

Una clave `(player_id, ranking_date)` con distintos `rank`, `points` o `tours`
se elimina completa del índice; nunca se selecciona la primera, última o mejor
fila.

### Edad

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `birth_date_a`, `birth_date_b` | `date32` nullable | DOB del maestro Sackmann usado para la derivación. |
| `age_a`, `age_b` | `float64` nullable | `(D - dob) / 365,2425` años. |
| `age_diff` | `float64` nullable | `age_a - age_b`. |
| `age_distance_30_a`, `age_distance_30_b` | `float64` nullable | `abs(age - 30)`, transformación no lineal Age.30. |
| `age_distance_30_diff` | `float64` nullable | Distancia de A a 30 menos distancia de B a 30. |
| `age_missing_a`, `age_missing_b` | `boolean` | DOB ausente, jugador ausente del maestro o DOB incompatible con `D`. |
| `age_invalid_for_date_a`, `age_invalid_for_date_b` | `boolean` | Distinguen específicamente una DOB igual o posterior a `D`; en ese caso no se fabrica una edad. |

### Cuotas, mercado y salida futura

| Columna | Tipo Arrow | Descripción |
|---|---|---|
| `odds_a`, `odds_b` | `float64` nullable | Cuotas decimales de cada lado, ambas necesarias y estrictamente mayores que 1. |
| `raw_implied_probability_a`, `raw_implied_probability_b` | `float64` nullable | Probabilidades brutas `q_i = 1 / odds_i`. |
| `market_probability_a`, `market_probability_b` | `float64` nullable | Probabilidades de-vigadas `q_i / (q_a + q_b)`; suman 1. |
| `market_overround` | `float64` nullable | `q_a + q_b`. |
| `market_margin` | `float64` nullable | `market_overround - 1`. |
| `market_retrieved_at_utc` | `timestamp[us, UTC]` nullable | Instante acreditado de adquisición de las cuotas. |
| `prediction_as_of_utc` | `timestamp[us, UTC]` nullable | Corte de la predicción; debe cumplir `market_retrieved_at_utc < prediction_as_of_utc`. |
| `model_probability_a` | `float64` nullable | Probabilidad del modelo para A; queda nula hasta la fase 7. |
| `edge` | `float64` nullable | `model_probability_a - market_probability_a`; solo existe si ambas probabilidades existen. |

Los CSV Sackmann no contienen cuotas. En los dos datasets históricos de la
fase 6 todas las columnas de mercado, `model_probability_a` y `edge` son
nulas; no se completan retroactivamente con una captura web posterior.

### Manifiesto e inventario de conflictos

`data/processed/features_active/manifest.json` es el puntero atómico al run
activo. El `manifest.json` dentro de `runs/<fingerprint>/` registra el esquema,
fingerprint, fuentes, parámetros, columnas de modelo, hashes SHA-256, tamaños,
recuentos, rangos, exclusiones, balance de `y` y auditoría de rankings.

`data/processed/features_active/runs/<fingerprint>/ranking_conflicts.csv`
contiene:

| Columna | Tipo CSV | Descripción |
|---|---|---|
| `gender` | texto | Universo del conflicto. |
| `player_id` | entero | ID de la clave ambigua. |
| `ranking_date` | fecha ISO | Fecha puesta en cuarentena. |
| `source_file` | ruta POSIX | CSV relativo a `data/raw`; no contiene rutas locales absolutas. |
| `source_row` | entero | Número de fila CSV, incluida la cabecera en el cómputo. |
| `rank`, `points`, `tours` | entero nullable | Valores exactos de cada observación incompatible. |

## Artefactos de modelado

La fase 7 publica runs inmutables en
`models/phase7/runs/<fingerprint>/`. `models/phase7/manifest.json` es el
puntero activo y contiene también el inventario verificado del run.

### Predicciones temporales OOF

`evaluation/oof_M.parquet` y `evaluation/oof_F.parquet` contienen únicamente
los tests futuros acumulados de 2016–2025:

| Columna | Tipo | Descripción |
|---|---|---|
| `record_id` | texto | Identidad única heredada del dataset causal. Una fila aparece en un solo fold. |
| `gender` | texto | `M` o `F`; cada archivo contiene un universo. |
| `match_date` | fecha | Fecha del partido usada para asignar temporada. |
| `tour_level`, `surface` | texto nullable | Segmento de evaluación. |
| `y` | entero binario | Resultado observado, usado solo después de predecir. |
| `test_season` | entero | Temporada `Y` del fold. |
| `logistic_raw` | probabilidad | Salida de la logística entrenada hasta `Y-2`. |
| `logistic_platt` | probabilidad | Salida anterior calibrada exclusivamente con `Y-1`. |
| `lightgbm_raw` | probabilidad | Salida de LightGBM entrenado hasta `Y-2`. |
| `lightgbm_platt` | probabilidad | LightGBM calibrado exclusivamente con `Y-1`. |
| `ranking_probability` | probabilidad nullable | Baseline rank-only ajustado con train; nulo sin ranking. |
| `ranking_favorite_decision` | entero binario nullable | Favorito determinista; nulo con empate o ranking ausente. |
| `market_devig` | probabilidad nullable | Probabilidad de mercado observada; nula en el histórico actual. |

### `evaluation/metrics.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `gender` | texto | Universo evaluado. |
| `population` | texto | `native`, `ranking_common_support` o `market_common_support`. |
| `model` | texto | Estimador o baseline evaluado. |
| `scope` | texto | `global` o `segment`. |
| `tour_level`, `surface` | texto nullable | `ALL` para global; valores reales para segmento. |
| `n_total` | entero | Filas de la población/segmento antes de filtrar ausencias. |
| `n_evaluated` | entero | Filas con predicción disponible. |
| `coverage` | float | `n_evaluated / n_total`. |
| `positive_rate` | float nullable | Tasa de `y=1` en las filas evaluadas. |
| `accuracy` | float nullable | Accuracy con umbral 0,5. |
| `log_loss` | float nullable | Entropía cruzada binaria media. |
| `brier` | float nullable | Error cuadrático medio de la probabilidad. |
| `auc` | float nullable | ROC AUC; nula si el segmento contiene una clase. |
| `auc_defined` | boolean | Indica si AUC se pudo identificar. |

`ranking_favorite` solo tiene accuracy: no se inventa una probabilidad para
rellenar log-loss, Brier o AUC.

### Curvas, folds y auditorías

- `reliability_bins.csv`: modelo, bin cuantílico, mínimos/máximos,
  probabilidad media, tasa observada, soporte y cobertura.
- `folds.csv`: género, temporada, tamaños, rangos y tasa positiva de train,
  calibración y test.
- `market_coverage.csv`: filas totales, filas con mercado, cobertura y estado
  evaluable/no evaluable.
- `suspicious_segments.csv`: todo segmento con accuracy `>0,85`, soporte,
  flag de muestra pequeña y comprobaciones estructurales. El run actual tiene
  30 filas auditadas; todas tienen `n<200` y se clasifican como muestra pequeña,
  no como evidencia de rendimiento extraordinario.

### Bundles y manifiesto

Cada género contiene:

| Archivo | Contenido |
|---|---|
| `deployment_bundle.joblib` | LightGBM principal, calibrador Platt, género, corte y tamaños. |
| `lightgbm.joblib` | Pipeline final LightGBM con preprocesador ajustado. |
| `lightgbm_platt.joblib` | Calibrador final ajustado sobre predicciones OOF. |
| `logistic.joblib` | Pipeline final de la baseline logística. |
| `logistic_platt.joblib` | Calibrador OOF de la baseline logística. |
| `ranking_probability.joblib` | Baseline probabilístico final de ranking. |

El `manifest.json` del run registra versión de artefacto, fuentes, parámetros,
perfil, versiones del runtime, hashes del código, resúmenes, métricas globales
y, para cada archivo, ruta relativa, tamaño y SHA-256. El bundle solo se
deserializa después de verificar el inventario completo.

## CSV de predicciones diarias

Cada ejecución de fase 8 publica la cartelera completa en
`data/processed/predictions/predictions_YYYY-MM-DD_<timestamp UTC>.csv`. A es
siempre `player_1` de Tennis Explorer y B es `player_2`; no se reorienta según
favorito, ranking o resultado.

| Columna | Tipo CSV | Descripción |
|---|---|---|
| `prediction_date` | fecha ISO | Fecha civil `D` de la cartelera. |
| `prediction_as_of_utc` | timestamp UTC | Instante en que comenzó la inferencia; debe ser posterior a la captura usada y su fecha civil no puede superar `prediction_date` para ser oficial. |
| `source_retrieved_at_utc` | timestamp UTC | Instante acreditado del snapshot de Tennis Explorer; su fecha civil no puede superar `prediction_date` para ser oficial. |
| `source_snapshot_sha256` | SHA-256 | Hash de los bytes HTML validados. |
| `tournament` | texto | Torneo visible, conservado desde el scraper. |
| `tour_level` | texto | Etiqueta diaria literal `ATP`, `WTA`, `Challenger` o `ITF`. |
| `canonical_tour_level` | texto nullable | Categoría de modelado `ATP Tour`, `WTA Tour`, `Challenger` o `ITF`; nula en filas no predichas. |
| `gender` | texto | Universo `M` o `F` que determina Elo y modelo. |
| `surface` | texto nullable | Superficie material explícita del mismo HTML; nula si la fuente no la publicó o solo indicó `Indoors`. |
| `scheduled_time` | texto nullable | Hora local visible sin zona inferida. |
| `status` | texto | Estado conservador de fase 4. Solo `scheduled` es elegible. |
| `player_a_name`, `player_b_name` | texto | Nombres visibles en el orden de la fuente. |
| `player_a_slug`, `player_b_slug` | texto nullable | Slugs estables de Tennis Explorer. |
| `player_a_id`, `player_b_id` | entero nullable | IDs Sackmann del género correcto; nulos si no se resolvieron. |
| `mapping_status` | texto | `mapped` exige ambos IDs; en otro caso `unmapped`. |
| `odds_a`, `odds_b` | decimal nullable | Cuotas decimales visibles del snapshot. |
| `model_probability_raw_a` | probabilidad nullable | Salida LightGBM previa a Platt. |
| `model_probability_a` | probabilidad nullable | `P(A gana)` calibrada. |
| `model_probability_b` | probabilidad nullable | Complemento exacto `1 - model_probability_a`. |
| `market_probability_a`, `market_probability_b` | probabilidad nullable | Probabilidades de-vigadas; ambas nulas si falta una cuota. |
| `edge_a` | decimal nullable | `model_probability_a - market_probability_a`, solo si `market_comparison_status=strictly_pre_date`; nulo para `prestart_unverified`. |
| `edge_b` | decimal nullable | `model_probability_b - market_probability_b` bajo el mismo corte causal; es `-edge_a` salvo redondeo y queda nulo para cuotas del propio día. |
| `confidence` | texto | `HIGH`, `MEDIUM`, `LOW` o `UNAVAILABLE`; mide inputs, no extremidad de la probabilidad. |
| `confidence_flags` | texto | Razones únicas separadas por `|`; siempre poblado cuando `confidence=UNAVAILABLE`. |
| `prediction_status` | texto | `predicted` o `not_predicted`. |
| `model_profile` | texto nullable | Perfil del bundle; actualmente `sports_only`. |
| `model_fingerprint` | SHA-256 nullable | Run inmutable de fase 7 utilizado. |
| `model_training_max_date` | fecha nullable | Máxima fecha fuente incluida al entrenar; una fila oficial exige este corte y el corte disponible descrito abajo. |
| `feature_history_max_date` | fecha nullable | Última fecha disponible en el Parquet causal del género. |
| `ranking_source_max_date` | fecha nullable | Última fecha global de rankings disponible en la fuente compatible. |
| `feature_fingerprint` | SHA-256 nullable | Identidad del dataset de fase 6 verificado antes de inferir. |

Los modelos actuales son `sports_only`: `odds_*` y
`market_probability_*` no entran al estimador, aunque se conservan para
comparar el resultado calibrado con el mercado.

### Correcciones de contrato diario de fase 9

| Columna | Tipo | Contrato vigente |
|---|---|---|
| `result_available_date` | fecha | `match_date + 21 dias`; un resultado solo se usa si esta fecha es `< D`. |
| `market_probability_a`, `market_probability_b` | probabilidad nullable | Probabilidades de-vigadas para presentacion siempre que existan dos cuotas validas, incluso en una captura del propio dia. |
| `market_comparison_status` | texto | `missing`, `prestart_unverified` o `strictly_pre_date`. |
| `edge_a`, `edge_b` | decimal nullable | Solo se calculan con estado `strictly_pre_date`; quedan nulos para `prestart_unverified`. |
| `model_training_max_date` | fecha nullable | Maxima `tourney_date` fuente considerada al entrenar. |
| `model_training_available_max_date` | fecha nullable | Debe ser exactamente el corte fuente mas 21 dias y ser `< prediction_date` para una oficial. |

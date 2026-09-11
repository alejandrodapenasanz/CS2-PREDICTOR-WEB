# Pipeline diario de predicción

## Frescura y fallbacks

El pipeline publica un sidecar lateral `TENNIS/freshness.json`; no modifica las
tablas append-only. Los umbrales viven en `freshness.config.json`: 24 horas para
TennisRatio y Tennis Explorer, y 1080 horas (45 días) para Sackmann. En
Sackmann se mide la fecha máxima real del dato fuente, no la hora de una
descarga sin novedades; la hora del último lote descargado se conserva aparte.

El resumen y el dashboard distinguen un fallo de actualización
(`source_update_failed`) de un pipeline no ejecutado (`pipeline_not_run`). Una
cartelera de más de 24 horas añade `data_stale_<fuente>_<edad>h`; si la agenda
procede de Tennis Explorer como respaldo, añade
`agenda_fallback_tennis_explorer`. Ambos casos conservan una predicción válida,
pero fuerzan confianza `LOW`. La edad se recalcula también en el navegador.

Si un partido aparece por primera vez durante `D`, no se reutilizan como
features su superficie, nivel ni cuotas capturados ese mismo día. La única vía
alternativa para la superficie es el catálogo de la misma identidad exacta de
torneo y edición, con captura anterior a `D`. No se infiere por ciudad, nombre
histórico ni `Indoors`; ante conflicto queda nula. Cuando el catálogo aporta
también un nivel inequívoco, ambos campos son contexto causal válido y la
confianza puede subir legítimamente. Sin esa evidencia, el modelo usa solo el
histórico previo y conserva `LOW`.

## Contrato operativo

`scripts/daily_predictions.py` encadena, en este orden:

1. carga la cartelera TennisRatio publicada por el refresco diario; Tennis
   Explorer queda como respaldo visible para ejecuciones históricas sin ese
   snapshot, pero su hora local sin zona no habilita una predicción oficial;
2. separa la cartelera actual de la última observación publicada estrictamente
   antes de `D`; superficie y nivel también pueden proceder del catálogo
   exacto torneo+edición construido con capturas `< D`; las cuotas solo de la
   observación exacta pre-D;
3. resuelve identidades TennisRatio observadas, resueltas y publicadas antes
   de `D`; cualquier identidad nueva o contradictoria queda sin predicción;
4. reconstruye forma, H2H y descanso solo con resultados cuya
   `result_available_date=source_date+21 días` cumple
   `result_available_date < D`;
5. consulta Elo, ranking y edad as-of `D`;
6. aplica el LightGBM del género correcto y su calibrador Platt;
7. quita el margen de las dos cuotas y compara modelo contra mercado;
8. publica todas las filas, predichas o no, en un CSV timestamped;
9. registra partido, predicción y estadísticas prepartido en
   `BBDD/tennis.sqlite3`;
10. observa y concilia una sola jornada anterior pendiente, si existe.

Jugador A es siempre `player_1` de la fuente de agenda y B es siempre
`player_2`; la fuente nunca se reorienta según ganador, ranking o cuota.
La probabilidad complementaria se calcula como `P(B) = 1 - P(A)`. Los edges
son `P_modelo - P_mercado` para cada lado y, cuando hay dos cuotas válidas,
suman cero salvo redondeo.

Solo recibe probabilidad una fila que cumpla simultáneamente:

- `status == "scheduled"` en el snapshot;
- `scheduled_start_utc` está publicado con zona inequívoca y
  `prediction_as_of_utc < scheduled_start_utc`; la igualdad ya queda fuera;
- la observación de agenda que aporta inputs fue capturada y publicada en una
  fecha civil estrictamente anterior a `D`;
- ambos jugadores están mapeados;
- existe un modelo íntegro del mismo género;
- `training_available_max_date = training_max_date + 21 días`;
- `training_available_max_date < D`.

Los partidos terminados, en juego, cancelados, walkover, de estado desconocido
o no mapeados permanecen en el CSV con probabilidades nulas. También permanece
visible un horario TBD/nulo, pero con `scheduled_start_utc_missing` y sin
probabilidad. La restricción del corte del modelo evita una fuga que las
features por sí solas no impedirían al solicitar una fecha histórica.

## Contexto causal e integridad

Los Parquet de fase 6 se verifican por tamaño y SHA-256. Para cada género se
leen únicamente las columnas necesarias y las filas de los jugadores del día
cuyo `result_available_date < D`. `y` permite recuperar el ganador real pese a
la orientación A/B aleatoria del entrenamiento. Los resultados se aplican en
bloques completos de disponibilidad, igual que en fase 6; `match_date < D` por
sí solo nunca habilita un resultado.

Antes de predecir se exige que coincidan:

- fingerprint del dataset en modelo y manifiesto de features;
- número de filas y fecha máxima de entrenamiento;
- commit Sackmann de raw, features y Elo;
- versión y fecha máxima del run Elo.

Un cambio de fuente requiere reconstruir Elo, features y modelos. No se mezcla
silenciosamente un ranking o maestro nuevo con un modelo antiguo.

### Elo persistido y overlay vivo de TennisRatio

TennisRatio no se concatena al Parquet histórico. Sus resultados terminales
post-handoff sí se incorporan al Elo general persistido al reconstruir Elo y,
por ello, cambian de forma trazable los fingerprints de Elo, features y del
challenger. Vive en `data/processed/tennisratio.sqlite3`, una SQLite
lateral append-only con procedencia, SHA-256, URL y `first_seen_at_utc`. La
BBDD operativa sagrada `BBDD/tennis.sqlite3` no se modifica por esta ingesta.

Cada observación separa:

- `effective_date`: fecha deportiva publicada para el hecho;
- `available_date`: primera fecha UTC en la que el hecho y las dos identidades
  necesarias estaban disponibles localmente;
- `D`: fecha del partido que se va a predecir.

Agenda, identidad, ranking y resultado solo son elegibles como inputs cuando
su hecho, primera observación y publicación local cumplen el corte estricto
`< D`. La agenda actual puede mostrar un cambio de estado de `D`, pero sus
columnas de modelo se congelan desde `load_agenda_as_of(D)`. Por tanto, la
primera descarga completa no simula que el proyecto conocía el pasado: todo lo
descubierto en `D` empieza a ser visible en `D+1`. Añadir observaciones
posteriores no puede cambiar un snapshot ya emitido para una fecha anterior.
El horario de inicio es una puerta operativa distinta: se toma de la agenda
actual y conserva el instante UTC exacto que TennisRatio publica. Una captura
pre-`D` puede seguir aportando features, pero nunca autoriza inferir a la hora
de inicio o después. No se convierte la hora local de Tennis Explorer ni se
inventa zona para un TBD.

Para servir una predicción futura, el estado dirigido de forma/H2H/descanso se
continúa con resultados laterales posteriores al corte fijo. El Elo persistido
ya contiene los resultados visibles durante el rebuild; el overlay en memoria
solo procesa observaciones con `available_date` posterior al máximo persistido.
Así, ningún evento entra dos veces. Varios partidos descubiertos el mismo día
no reciben un orden intradía inventado. El ranking lateral reemplaza
al snapshot Sackmann de un jugador solo cuando es causal y más reciente; como
TennisRatio no acredita puntos ATP/WTA en ese contrato, `rank_points_*` queda
nulo en vez de mezclarlo con puntos antiguos.

La lectura dirigida del histórico usa lotes de `pyarrow.parquet.ParquetFile`
y filtra `result_available_date < D` antes de construir el estado de los
jugadores objetivo. Así no depende del módulo opcional `pyarrow.dataset` y no
cambia el corte causal ni duplica un partido cuando ambos jugadores son parte
del objetivo.

TennisRatio puede publicar el mismo partido desde los perfiles de ambos
jugadores con etiquetas distintas para el torneo o el nivel. Si esas copias
coinciden en jugadores, fecha, ganador, marcador y sets, pero discrepan solo en
metadatos de contexto, el partido se excluye del overlay: no se elige una
etiqueta arbitraria. Una discrepancia de ganador, marcador o sets sigue siendo
un conflicto deportivo y hace fallar cerrado la construcción del overlay.

Las cuotas y los agregados actuales incluidos en el HTML se conservan solo en
la evidencia cruda/lateral. No salen por la API de resultados que alimenta
forma o Elo y nunca se usan como si hubieran existido antes del partido. El
overlay en memoria es de serving: no fabrica filas históricas. El adaptador
persistido de Elo es una ruta distinta, versionada en
`config/elo_handoff.json`.

La conciliación de una jornada pendiente enlaza TennisRatio por género y par
no ordenado de IDs Sackmann. Cuando la agenda oficial también pertenece a
TennisRatio, cada coincidencia única con marcador terminal se liquida aunque
otras filas aún no estén disponibles: lo ausente o ambiguo queda pendiente y
no invalida la evidencia segura. El run registra `official_pending_count`,
`matched_count`, `unmatched_count` y `coverage_complete`. Para agendas con IDs
de otra fuente se sigue exigiendo cobertura completa antes de desplazar su
respaldo nativo de Tennis Explorer. En todos los casos, la única escritura operativa pasa por
`OperationsStore.reconcile_observations` y respeta los triggers append-only.

La caché persistente de superficies combina publicaciones exactas de
TennisRatio y las fichas exactas `tournament_href` de Tennis Explorer sin hacer
otra petición. En Tennis Explorer se exige que el href contenga el año de la
edición; TennisRatio usa la etiqueta literal, género y año publicados por la
propia fuente. Los Futures agregados, `Indoors`, conflictos y ediciones sin
identidad exacta permanecen nulos. La tabla no aporta de forma fiable ronda ni
`best_of`; ambos permanecen nulos y el preprocesador entrenado los imputa.

## Mercado y modelo actual

El modelo activo tiene perfil `sports_only`, porque Sackmann no contiene
cuotas históricas y la cobertura de fase 7 fue 0 %. Por tanto, las cuotas
diarias no entran actualmente al LightGBM. Dos cuotas válidas se transforman y
conservan como `market_probability_*` para presentación. Si fueron capturadas
el propio día, su estado es `prestart_unverified` y `edge_*` queda nulo. El edge
solo se calcula cuando `market_comparison_status=strictly_pre_date`.

Esto permite una comparación descriptiva, pero todavía no demuestra que el
modelo supere al mercado: esa afirmación exige acumular cuotas prospectivas y
resultados, y evaluarlos después sobre soporte temporal común.

## Confianza

La confianza depende de inputs, nunca de lo extrema que sea la probabilidad.
Los umbrales configurables de `src/daily_pipeline/confidence.py` son:

- mínimo 20 partidos de Elo general por jugador;
- mínimo 10 partidos en la superficie por jugador;
- máximo 14 días de lag desde el último resultado **ya disponible** y
  desde la fuente de rankings;
- máximo 14 días para el snapshot de ranking individual.

El lag del histórico se calcula como
`D - feature_history_available_max_date`, nunca como
`D - feature_history_max_date`. El CSV conserva tres cortes distintos: la
fecha deportiva del último resultado servido, su primera disponibilidad
causal y `model_training_available_max_date`, que continúa describiendo solo
el dataset con el que se entrenó el bundle. Así, el propio embargo o un overlay
lateral legítimo no se confunden con la edad del modelo; un lag posterior
superior a 14 días genera el flag explícito
`history_available_stale_<N>d`.

Superficie ausente, histórico obsoleto, ranking ausente o menos de 20 partidos
generan `LOW`. Ausencia de ronda/`best_of`, poca muestra por superficie, edad
ausente o mercado ausente generan advertencias de calidad. `UNAVAILABLE`
significa que no se calculó ninguna probabilidad y siempre lleva una razón.
`confidence_flags` contiene todas las razones separadas por `|`.

## Salida

Cada ejecución publica:

```text
data/processed/predictions/
└── predictions_YYYY-MM-DD_YYYYMMDDThhmmssffffffZ.csv
```

El CSV conserva procedencia y timestamp de captura, estado, identidades,
mapping, cuotas, probabilidades raw y calibrada, mercado de-vigado, ambos
edges, confianza, cortes de fuentes y fingerprints. La escritura es atómica y
nunca sobrescribe una publicación con el mismo timestamp.

Si el reloj de Windows devuelve el mismo tick para la captura y la inferencia
aunque la descarga ya haya terminado, el pipeline asigna a la inferencia el
microtick lógico siguiente. La BBDD mantiene la desigualdad estricta
`source_retrieved_at_utc < prediction_as_of_utc`; no relaja el corte.

La consola ordena las filas predichas por magnitud absoluta del edge y muestra
como máximo 50; el CSV siempre contiene la cartelera completa.

El contrato append-only, la selección de la primera predicción oficial, la
rotación de fechas pendientes y el reentreno automático de cada arranque se
documentan en
[`operations_database.md`](operations_database.md).

Un CSV histórico puede generarse para diagnóstico, pero la BBDD nunca lo
promueve retrospectivamente: para ser oficial, captura y predicción deben
tener fecha civil no posterior a la jornada, el snapshot debe seguir
`scheduled`, debe cumplirse
`prediction_as_of_utc < scheduled_start_utc`, el corte del modelo debe ser
estrictamente anterior y no puede existir una observación previa. Un
settlement exige además
`observed_at_utc > prediction_as_of_utc`.

## Contrato causal vigente de fase 9

Sackmann publica `tourney_date` como inicio del torneo, no la fecha exacta de
cada ronda. El pipeline aplica por ello el contrato conservador
`result_available_date = match_date + 21 dias` y solo reconstruye historia con
`result_available_date < D`. El bundle debe acreditar asimismo
`training_available_max_date < D` y que esa fecha disponible es exactamente
`training_max_date + 21 dias`.

El run Elo puede cubrir fechas disponibles posteriores al maximo incluido en
features; debe alcanzar al menos ese maximo y mantener el mismo contrato y
fingerprint. Esta asimetria evita rechazar un checkpoint valido por un ultimo
bloque raw no elegible.

Dos cuotas validas siempre se transforman y conservan para presentacion. Si el
snapshot es del propio dia, `market_comparison_status=prestart_unverified` y
los edges quedan nulos. Solo `strictly_pre_date` habilita una comparacion
causal y `edge = P_modelo - P_mercado`.

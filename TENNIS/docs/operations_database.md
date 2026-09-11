# Base operativa y ciclo diario

## Objetivo

`BBDD/tennis.sqlite3` es el registro prospectivo y auditable del sistema. En
cada ejecución conserva la cartelera observada, las features prepartido de
cada jugador, la probabilidad emitida, el nivel de confianza y, cuando existe
evidencia terminal suficiente, el resultado real conciliado.

La base no sustituye los CSV/Parquet históricos ni convierte una predicción en
verdad. Su utilidad es medir rendimiento futuro, calibración y drift sobre
predicciones publicadas antes de conocer el desenlace.

## Invariantes

- La primera predicción válida registrada para un `source_match_id` es la
  predicción oficial de evaluación. Otra ejecución puede guardar una nueva
  observación, pero no reemplazarla ni seleccionar retrospectivamente la que
  mejor salió.
- Una predicción solo es elegible como oficial si el modelo acredita ambos
  cortes: `model_training_max_date < match_date` y
  `model_training_available_max_date = model_training_max_date + 21 días`
  con `model_training_available_max_date < match_date`. Además, tanto la
  captura como la predicción tienen fecha civil `<= match_date`, la fuente
  todavía marca el partido como `scheduled` y no existe ninguna observación
  previa del partido.
  Ejecutar `-Date` después de la jornada conserva el replay para diagnóstico,
  pero lo marca no oficial.
- Cuando la cartelera procede de TennisRatio, el estado visible de hoy y los
  inputs del modelo son contratos distintos. La fila puede mostrar el estado y
  las cuotas observados hoy, pero superficie, nivel, cuotas e identidad solo
  son elegibles como features si su captura y su resolución estaban publicadas
  en una fecha civil estrictamente anterior a `match_date`. Sin una captura
  causal previa la fila permanece visible, pero no se predice.
- Predicciones, estadísticas prepartido, observaciones, settlements y
  conflictos son append-only o inmutables mediante triggers SQLite.
- El resultado se concilia comparando `winner_slug` con los slugs A/B de la
  predicción oficial. Si Tennis Explorer y la agenda usan IDs de partido
  distintos, la traducción exige fecha, género y el par no ordenado de IDs
  Sackmann. También se admite completar una pareja cuando exactamente un
  jugador ya está mapeado y ese jugador aparece en un único partido oficial de
  la fecha. La posición y el nombre visible nunca determinan por sí solos el
  label; cualquier duplicidad queda sin liquidar.
- Solo se liquida `status=finished` con sets terminales, marcador, lado ganador,
  slug ganador y evidencia explícita coherentes. Walkovers, cancelaciones,
  estados parciales o contradicciones quedan pendientes o en revisión.
- El timestamp de una observación liquidable debe ser estrictamente posterior
  a `prediction_as_of_utc`. Una observación anterior o simultánea nunca se
  convierte en label.
- Las estadísticas son inputs individuales calculados as-of la fecha del
  partido. El almacén rechaza nombres/fuentes que intenten introducir
  probabilidades, winners, labels, edges o resultados como estadísticas.

## Tablas principales

| Tabla | Contenido |
|---|---|
| `runs` | Identidad, fecha, hash y estado de cada predicción/observación. |
| `snapshots` | Procedencia, SHA-256 y timestamp de cada HTML observado. |
| `matches` | Identidad estable y contexto básico del partido. |
| `predictions` | Payload completo de cada ejecución, incluida confianza. |
| `official_predictions` | Primera predicción válida congelada por partido. |
| `player_statistics` | Elo, forma, descanso, ranking, puntos y edad prepartido. |
| `observations` | Estados/resultados vistos, incluso si aún no son liquidables. |
| `settlements` | Ganador real reconciliado contra la orientación A/B oficial. |
| `conflicts` | Contradicciones append-only que nunca se corrigen en silencio. |
| `review_queue` | Casos sin identidad, sin predicción oficial o ambiguos. |

## Metricas prospectivas publicadas en la web

La web separa dos conceptos que no son intercambiables:

- **Confianza del input**: es el nivel `HIGH`, `MEDIUM`, `LOW` o
  `UNAVAILABLE` producido antes del partido por
  `assess_vector_confidence()`. Resume cobertura y frescura de los inputs. Se
  presenta como `ALTA`, `MEDIA`, `BAJA` o `NO DISPONIBLE`, nunca como un
  porcentaje y nunca como accuracy.
- **Acierto real (partidos liquidados)**: se calcula en modo solo lectura
  sobre todas las predicciones oficiales que tienen `settlement_id`,
  `settled_at_utc` y `actual_outcome_a`. No se limita a la jornada que aparece
  en la tabla diaria.

Para la segunda metrica, el builder compara la probabilidad calibrada del
ganador previsto con el indicador de que ese ganador acerto. Publica numero de
liquidados, numero evaluable, aciertos, accuracy, Brier, log-loss y una curva
de calibracion por cuantiles de hasta diez bins. Si no hay muestra, accuracy,
Brier y log-loss son nulos y la interfaz muestra `sin muestra / no calculado`;
un conjunto inferior a 100 partidos se etiqueta como provisional. Este umbral
coincide con el criterio de muestra pequena usado en la auditoria de segmentos.

`WEB/build_web.py` abre SQLite con `mode=ro` y `PRAGMA query_only=ON`. La
publicacion no actualiza `predictions`, `observations`, `settlements` ni ninguna
otra tabla operativa.

El esquema está versionado mediante `PRAGMA user_version` y
`schema_versions`. Cada operación multi-fila se ejecuta en una transacción; un
fallo estructural revierte la operación completa.

TennisRatio publica un instante `data-utc` por partido y el adaptador lo
conserva como `scheduled_start_utc`. La selección oficial exige de nuevo, en la
capa SQLite, `prediction_as_of_utc < scheduled_start_utc`; igualdad y cualquier
instante posterior quedan no oficiales. `None`/TBD también falla cerrado. Tennis
Explorer solo aporta una hora local sin zona universal: sus filas pueden seguir
visibles como respaldo, pero no reciben probabilidad ni se oficializan salvo
que otro contrato de fuente aporte un instante UTC inequívoco. Nunca se inventa
una zona horaria.

En relojes con resolución insuficiente, el pipeline representa el orden real
captura-antes-de-inferencia mediante el microtick lógico siguiente. La
validación SQLite conserva la desigualdad estricta entre ambos instantes.

## Flujo de cada mañana

`scripts/daily_predictions.py` llama a
`src.operations.run_operational_daily_pipeline`:

1. ejecuta el pipeline causal y publica su CSV;
2. registra todas las filas, incluidas las no predichas, sin inventar valores;
3. almacena las features individuales de jugadores mapeados en formato largo;
4. reprocesa idempotentemente los snapshots Tennis Explorer ya almacenados y
   añade solo observaciones canónicas nuevas mediante `OperationsStore`;
5. selecciona una sola fecha anterior con predicciones oficiales sin liquidar;
6. intenta resolver la fecha pendiente con TennisRatio usando solo
   evidencia terminal, inequívoca y publicada antes de la fecha actual;
7. si TennisRatio no aporta una coincidencia segura, usa Tennis Explorer como
   fallback y enlaza únicamente resultados uno-a-uno por ID exacto o identidad
   Sackmann; el HTML crudo conserva las filas no enlazadas;
8. registra las observaciones por la API append-only sancionada y liquida
   únicamente las evidencias válidas;
9. si todo lo anterior termina bien, `run_tennis.ps1` ejecuta el builder web
   existente para regenerar `WEB/data.js` desde la SQLite en modo lectura.

La selección prioriza la fecha pendiente más reciente que nunca se haya
consultado. Después rota por la que lleve más tiempo sin observarse. Un
refresco vacío también queda anotado, por lo que no bloquea eternamente otras
fechas. La propia fecha que se está prediciendo nunca se consulta como
resultado en esa ejecución.

Cada invocación consulta como máximo una fecha anterior. El refresco diario de
TennisRatio se ejecuta automáticamente antes de la inferencia mediante
`run_tennis.ps1`; también existe el modo aislado `-UpdateOnly`, usado por la
tarea programada diaria. Ambos caminos llaman al mismo adaptador idempotente y
al mismo transporte Scrapling fijado. No acceden a `/api/`, no siguen
redirecciones fuera del host permitido y validan `robots.txt` con matching
wildcard completo, de modo que una regla de query como `/*?q=` no bloquea
erróneamente `/atp-matches.html`.

El replay de snapshots ya almacenados forma parte de
`scripts/daily_predictions.py`, por lo que se ejecuta automáticamente desde
ambos `start.ps1`. También puede invocarse de forma aislada y segura con:

```powershell
python scripts\reconcile_results.py --before-date 2026-08-31
```

Si ninguna fuente puede aportar evidencia inequívoca por HTTP, caché, identidad
o HTML inesperado, la predicción ya guardada se conserva. Una coincidencia
segura sí puede liquidarse aunque otras filas de la fecha sigan pendientes; el
run audita cobertura, no mapeados y ambigüedades. Un fallo de integridad SQLite
sí detiene la ejecución.

## Lanzador y reentreno automático

Desde cualquier directorio:

```powershell
.\TENNIS\run_tennis.ps1
.\TENNIS\run_tennis.ps1 -Date 2026-07-30
.\TENNIS\run_tennis.ps1 -Retrain
.\TENNIS\run_tennis.ps1 -Retrain -Date 2026-07-30
```

Todo arranque operativo completo falla rápido y respeta este orden antes del
diario; `-Retrain` se conserva como alias compatible del mismo flujo TENNIS:

1. `scripts/update_sources.py`;
2. `scripts/update_tennisratio.py`, incluido el remapeo append-only contra el
   Sackmann que acaba de actualizarse;
3. `scripts/audit_identities.py`;
4. `scripts/build_elo.py`;
5. `scripts/build_features.py`;
6. `scripts/retrain_models.py`;
7. `scripts/daily_predictions.py --retrained`.

Después del paso diario, y solo si su código de salida es cero, se ejecuta
`WEB/build_web.py`. Si el builder falla, el lanzador conserva la BBDD y el CSV,
informa del error y devuelve un código no cero; no presenta como actualizada
una web cuyo `data.js` no se haya podido regenerar.

Los constructores siguen siendo idempotentes: si las fuentes y fingerprints
no han cambiado, reutilizan los artefactos compatibles en vez de fingir un
modelo nuevo. `-UpdateOnly` queda fuera de este ciclo y solo realiza el refresco
diario idempotente de TennisRatio. Un `-Date` explícito sin `-Retrain` también
queda fuera del entrenamiento para que un replay histórico no active un modelo
con corte antiguo; `-Retrain -Date` sigue disponible cuando ese corte se pide
deliberadamente.

## Contrato v2 de fase 9

El esquema vigente es v2. Una base vacia se crea directamente en v2; una v1
y una base completa sin `PRAGMA user_version` se migran atomicamente y se
registra la version en `schema_versions`.

La selección oficial aplica el mismo contrato descrito en los invariantes:
exige el corte preinicio UTC y ambos cortes causales, y comprueba que el
disponible sea exactamente el corte fuente más 21 días. Cualquier igualdad en
la puerta temporal falla cerrada ante metadata incoherente.

Al rotar fechas pendientes, el scheduler compara el timestamp maximo del run
de observacion con el timestamp maximo de sus observaciones y usa el mas
reciente de ambos. Asi no prioriza accidentalmente una columna mediante
`COALESCE` cuando las dos contienen evidencia.

## Uso correcto para aprendizaje

La base es muy útil para evaluar el sistema de manera prospectiva, comparar el
modelo con el mercado sobre soporte común y detectar pérdida de calibración.
No debe usarse la columna predicha como feature ni como etiqueta. Para futuros
reentrenos solo entra el resultado real verificado, siguiendo de nuevo el corte
temporal; el pipeline histórico de Sackmann continúa siendo la fuente canónica
de entrenamiento. Antes de incorporar resultados operativos como una nueva
fuente histórica haría falta una fase explícita de deduplicación, procedencia y
validación, no una unión automática silenciosa.

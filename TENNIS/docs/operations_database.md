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
- Predicciones, estadísticas prepartido, observaciones, settlements y
  conflictos son append-only o inmutables mediante triggers SQLite.
- El resultado se concilia comparando `winner_slug` con los slugs A/B de la
  predicción oficial. La posición del jugador en la página y el nombre visible
  no determinan el label, porque Tennis Explorer puede reordenar al ganador.
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

El esquema está versionado mediante `PRAGMA user_version` y
`schema_versions`. Cada operación multi-fila se ejecuta en una transacción; un
fallo estructural revierte la operación completa.

Tennis Explorer no aporta una hora oficial con zona horaria que pueda
compararse de forma universal. Para permitir el uso de cada mañana sin fingir
esa precisión, el corte operativo usa fecha civil y exige además evidencia
`scheduled` en el snapshot. Esta regla permite una captura del mismo día
antes del inicio, pero falla cerrada ante estados `live`, `finished`,
`walkover`, `cancelled`, capturas posteriores o cualquier resultado ya
observado.

En relojes con resolución insuficiente, el pipeline representa el orden real
captura-antes-de-inferencia mediante el microtick lógico siguiente. La
validación SQLite conserva la desigualdad estricta entre ambos instantes.

## Flujo de cada mañana

`scripts/daily_predictions.py` llama a
`src.operations.run_operational_daily_pipeline`:

1. ejecuta el pipeline causal y publica su CSV;
2. registra todas las filas, incluidas las no predichas, sin inventar valores;
3. almacena las features individuales de jugadores mapeados en formato largo;
4. selecciona una sola fecha anterior con predicciones oficiales sin liquidar;
5. obtiene un snapshot append-only mediante `refresh_daily_results`;
6. registra las observaciones y liquida únicamente las evidencias válidas.
7. si todo lo anterior termina bien, `run_tennis.ps1` ejecuta el builder web
   existente para regenerar `WEB/data.js` desde la SQLite en modo lectura.

La selección prioriza la fecha pendiente más reciente que nunca se haya
consultado. Después rota por la que lleve más tiempo sin observarse. Un
refresco vacío también queda anotado, por lo que no bloquea eternamente otras
fechas. La propia fecha que se está prediciendo nunca se consulta como
resultado en esa ejecución.

Cada invocación consulta como máximo una página anterior, pero no hay contador
ni límite diario en el código. El usuario controla manualmente cuántas veces
ejecuta el lanzador. No hay reintentos HTTP automáticos ni bypass de WAF.

Si la observación de resultados falla por HTTP, caché, WAF o HTML inesperado,
la predicción ya guardada se conserva y el script emite una advertencia clara.
Un fallo de integridad SQLite sí detiene la ejecución.

## Lanzador y reentreno semanal

Desde cualquier directorio:

```powershell
.\TENNIS\run_tennis.ps1
.\TENNIS\run_tennis.ps1 -Date 2026-07-30
.\TENNIS\run_tennis.ps1 -Retrain
.\TENNIS\run_tennis.ps1 -Retrain -Date 2026-07-30
```

Sin `-Retrain` se ejecutan predicción, registro y conciliación. Con `-Retrain`,
el lanzador falla rápido y respeta este orden antes del diario:

1. `scripts/update_sources.py`;
2. `scripts/audit_identities.py`;
3. `scripts/build_elo.py`;
4. `scripts/build_features.py`;
5. `scripts/retrain_models.py`;
6. `scripts/daily_predictions.py --retrained`.

Después del paso diario, y solo si su código de salida es cero, se ejecuta
`WEB/build_web.py`. Si el builder falla, el lanzador conserva la BBDD y el CSV,
informa del error y devuelve un código no cero; no presenta como actualizada
una web cuyo `data.js` no se haya podido regenerar.

Los constructores siguen siendo idempotentes: si las fuentes y fingerprints
no han cambiado, reutilizan los artefactos compatibles en vez de fingir un
modelo nuevo.

## Contrato v2 de fase 9

El esquema vigente es v2. Una base vacia se crea directamente en v2; una v1
y una base completa sin `PRAGMA user_version` se migran atomicamente y se
registra la version en `schema_versions`.

La selección oficial aplica el mismo contrato descrito en los invariantes:
exige ambos cortes y comprueba que el disponible sea exactamente el corte
fuente más 21 días. Esta igualdad falla cerrada ante metadata incoherente.

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

# Recuperación automática de resultados — 2026-09-28

## Qué existía y qué faltaba

CS2 ya obtenía resultados recientes, guardaba mapas en `maps` y liquidaba el
`prediction_ledger`. Su cola de pendientes priorizaba partidos sin ganador:
tener resultado/liquidación no garantizaba disponer de todos los mapas.
TENNIS ya conciliaba una jornada anterior, usando TennisRatio y Tennis Explorer,
y conservaba marcadores como texto en observaciones inmutables. Faltaba una
representación estructurada de los sets y reintentar los liquidados incompletos.

## Contrato nuevo

- CS2: `PIPELINE/result_recovery.py` reutiliza transporte, cooldowns, parser y
  archivo HTML existentes. `BBDD/result_store.py` es dueño de las escrituras.
  La etapa se ejecuta automáticamente en `PIPELINE/start.py`; la ingesta final
  usa **la misma API sancionada** `finalize_prediction_ledger`.
- TENNIS: ambos launchers pasan por `scripts/daily_predictions.py`, que añade
  recuperación tras predecir/conciliar. `operations/result_recovery.py` reutiliza
  mapeadores, `TennisRatioClient`, `refresh_daily_results` y `OperationsStore`.
  Primero interpreta observaciones ya guardadas; después consulta resultados
  locales Ratio, perfiles con URL previamente observada y fallback Explorer.
- Un partido liquidado puede seguir pendiente de mapas/sets. Los partidos con
  detalle completo se excluyen de esta recuperación. Otros scrapers pueden
  visitar esos perfiles/partidos para estadísticas: es otro propósito.
- No se deducen identidades por nombres parecidos. Perfiles Ratio se cruzan por
  género, fecha exacta y par de IDs nativos único. Los cruces entre fuentes
  conservan el mapeo existente a IDs Sackmann. Dos partidos del mismo par en
  una fecha, sin otra identidad inequívoca, quedan pendientes.
- Tennis Abstract conserva su adquisición existente. Esta tarea no convierte
  fechas ambiguas de su histórico en identificadores de partidos por intuición.

## Persistencia y estados

En `VAULT/CS2/BBDD/cs2.db` se añaden `result_evidence`, `result_status`,
`result_maps`, `result_fetch_state` y la vista `result_settlement_v1`.
Evidencia JSON inmutable identificada por hash; mapas por `(match_id,map_number)`.
El marcador de serie se normaliza a mapas ganados, incluido BO1. La orientación
se verifica con IDs; un par exactamente invertido se reorienta conservando el
payload original. OT es nullable: solo se marca con evidencia explícita, nunca
porque un número de rondas parezca alto. No se inventa un formato ausente.
BLACKBOX exporta/restaura estas tablas junto al ledger; se prueba con filas reales
de fixture y hashes de contenido antes/después.

En `VAULT/TENNIS/BBDD/tennis.sqlite3` la migración aditiva v3 añade
`result_details`, `result_sets`, `result_recovery`, `result_detail_processed`,
`result_detail_conflicts` y `result_settlement_v1`. Los triggers del triplete
original permanecen intactos. Cada detalle conserva el `observation_id` de la
evidencia; el raw completo permanece en la observación. La clave de set es
`(source_match_id,set_number)`.

Se separa `status` (`unknown`, `live`, `finished`, `cancelled`, `postponed`,
`walkover`, `retirement`, etc.) de `detail_status` (`unknown`, `global`, `partial`,
`detailed`, `not_applicable`) y de `complete`. Conocer el ganador no basta.
Un resultado normal completo exige todos los mapas/sets y acuerdo con el global.
Un `2-1` de tenis no se convierte en tres sets inventados. `7-6(5)` solo acredita
los cinco puntos del perdedor del tie-break; los del ganador quedan nulos.
`[10-8]` se etiqueta como puntos de match tie-break, no como juegos.
Retirements sin certeza sobre el detalle total continúan pendientes con backoff;
cancelaciones y walkovers no necesitan sets. Sets de formatos cortos no
acreditados se conservan, sin asumir completitud con reglas inventadas.

Una evidencia pobre no elimina mapas/sets. Evidencia terminal contradictoria
se conserva en raw/cuarentena y se avisa; no reemplaza hechos terminales. Un
marcador **live** puede crecer al llegar el final, sin duplicar el set. La
liquidación sigue usando la validación previa del ganador y sus reglas actuales,
independientemente de que el detalle sea incompleto o requiera revisión.

## Predicciones, mercado y anti-fugas

No se reescriben probabilidades, versiones ni features históricas. Se liquidan
solo predicciones elegibles por las APIs anteriores. Cancelaciones, walkovers
y abandonos se guardan pero no se fuerzan a accuracy/ROI: decisión confirmada
por el usuario. Las vistas exponen ganador previsto/real, acierto, probabilidad,
versión, fecha de predicción y liquidación, marcador y payload congelado.
Cuando existe mercado prepartido congelado, exponen favorito, underdog y upset;
sin mercado, queda NULL. CS2 exige timestamp de opening odds anterior al kickoff
para interpretar su probabilidad de-vigada guardada. No hay odds ni ROI inventados.

`event_time`/`event_date` y `obtained_at_utc` son distintos. CS2 recupera en tablas
laterales, sin añadir estos resultados a `matches`, `maps` o snapshots usados por
el modelo; así este cambio **no amplía** el problema de orden intradía detectado
por la auditoría anterior. No se arregla aquí ese modelo ni se añaden features.
En TENNIS los sets laterales no son fuente de features. El handoff ya existente
de resultados Explorer a Elo mantiene fecha efectiva **y disponibilidad `< D`**:
un test llama al cargador real, persiste un resultado en D, verifica la invariancia
de D y que solo pueda aparecer desde D+1. Las nuevas capturas de perfiles Ratio
para liquidación no reescriben su almacén de features.

## Presupuesto, reintentos y uso

| Variable | Por ejecución | Defecto |
|---|---|---:|
| `CS2_RESULT_RECOVERY_LIMIT` | páginas de partidos CS2 | 20 |
| `TENNIS_RESULT_RECOVERY_DATES` | fechas pendientes adicionales | 3 |
| `TENNIS_RESULT_RECOVERY_PROFILES` | perfiles Ratio adicionales | 10 |

El paso previo de conciliación de tenis conserva su jornada; no se vuelve a
pedir esa misma página Explorer. Los perfiles visitados se deduplican por URL.
Las colas incluyen antiguos sin resultado y liquidados parciales; los intentos
más antiguos tienen prioridad. Backoff persistente: 6, 12, 24, 48, 96 y hasta
168 horas. No se elimina el pendiente por agotar intentos. Un WAF/429 detiene el
lote de esa fuente y conserva sus límites; un error individual no aborta las
predicciones del día. No hay Selenium, proxies ni nuevas técnicas de evasión.

Los launchers no necesitan flags nuevos. Para una recuperación manual acotada
de tenis se amplía la herramienta existente, sin crear otro pipeline:

```powershell
.\VAULT\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\reconcile_results.py --recover-details --max-result-dates 3 --max-result-profiles 10
```

Los resúmenes `[CS2 resultados]`, `[CS2 resultados persistidos]` y
`[TENNIS resultados]` informan revisados, nuevos, enriquecidos, mapas/sets,
liquidaciones, pendientes y errores. CS2 conserva el resumen en el run JSON;
tenis lo imprime en el log persistente del launcher y conserva los reintentos
por partido en SQLite. Los contadores de recuperación se refieren a su etapa;
la conciliación previa de tenis imprime sus liquidaciones por separado.

## Auditoría del histórico (solo lectura, 28-09-2026)

Reconstrucción en SQLite **en memoria**, sin aplicar backfill a producción:

| Cobertura | CS2 | TENNIS |
|---|---:|---:|
| Partidos examinados | 11.477 | 3.491 |
| Con ganador | 11.461 | 2.737 |
| Con marcador global | 11.461 | 2.737 |
| Con mapas/sets completos | 1.635 | 139 |
| Con resultado global pero detalle incompleto | 9.826 | 2.598 |
| Pendientes de resultado o detalle | 9.842 | 3.352 |

CS2 incluye registros fechados hasta hoy (la selección de red excluye kickoff
futuro); tenis cuenta fechas anteriores a hoy. No son el total de predicciones:
incluyen partidos sin pick oficial. El replay de tenis permite estructurar 327
sets ya almacenados, sin descargarlos de nuevo; detecta una contradicción de
ganador que queda en cuarentena, no se corrige el pasado. Los pendientes son
**candidatos**, no promesa de que la fuente todavía exponga todo el detalle.

No se ha realizado scraping histórico masivo. La recuperación automática avanzará
en lotes al ejecutar el launcher, manteniendo los bloqueos y límites existentes.

Comprobación adicional CS2, 29-09-2026 05:10 UTC: al incluir el master vigente,
9.844 candidatos elegibles tienen URL exacta (ninguno sin URL). El inventario
fechado hasta el 29 incluye 11.503 partidos y 9.868 pendientes; la diferencia con
los elegibles incluye los kickoff todavía futuros. Ganadores, globales y mapas
completos mantienen las cifras de la tabla anterior. No se hicieron peticiones.

## Verificación final — 29-09-2026

- CS2: puerta completa OK; 469 tests, 4 skips explícitos y 17 subtests.
  Smoke determinista OK (`rows=160`, `dataset=3a45dacdaa5d`). Además se ejecutaron
  23 tests de ingesta/BLACKBOX, incluyendo los de pipeline vivo que la puerta
  general excluye. El backup/restore conserva las tablas nuevas y el ledger.
- TENNIS: puerta completa OK; 661 tests y 264 subtests. Smoke OK, 16 imports y
  9 entrypoints. Se ejercitó el pipeline de recuperación con fixtures, fallos
  temporales, reintentos y liquidación, sin red.
- Ruff lint/formato, mypy, cobertura de imports y `pip check` OK en ambos
  entornos CPython 3.13. Sin dependencias nuevas en este cambio.
- Scraper HLTV: `smoke_boot.py` OK (red desactivada) y `PIPELINE/start.py --help`
  con el intérprete real del scraper. Parser contrastado también con dos HTML
  de resultados ya guardados, sin descargar ni escribir resultados reales.
- Arquitectura: cada dominio escribe por sus APIs; launchers solo orquestan.
  Anti-fugas: nuevas capturas no alteran las predicciones históricas ni las
  features de su propio día; no se modificaron modelos ni selección de features.
  Dependencias: no imports cruzados entre deportes ni paquetes añadidos.

Las pruebas son offline y de arranque. **No** se ejecutó un `start.ps1` completo
de producción ni se activaron modelos: las migraciones aditivas y la recuperación
se aplicarán automáticamente al usar los entrypoints normales. Los recuentos
anteriores no significan que ya se haya descargado ese histórico pendiente.

# Rankings HLTV y Valve/VRS

El ranking general ya se guardaba en `team_ranking_snapshots`: posiciones,
puntos, equipo HLTV, fecha indicada por la página, captura, run y procedencia.
Las features existentes de ranking son independientes de Elo/Glicko.

## Corrección de actualización (2026-09-12)

La auditoría encontró 4.311 filas HLTV y 6.845 Valve, pero la captura más reciente
era del 13 de julio. `update_fetch_state_from_run` renovaba la frescura incluso
cuando `collect_rankings` había omitido la petición por caché. Cada arranque
aplazaba de nuevo siete días una descarga que no se había hecho.

Ahora una omisión no altera la fecha de captura. Además, el scraper comprueba
la última captura realmente persistida para recuperarse de marcadores antiguos
incorrectos. Los cooldowns de estados bloqueados/error se conservan; no es una
excepción a Cloudflare. Una respuesta sin equipos no cuenta como actualización.

## Etiquetas de las fichas de partido

La ingesta extrae también las etiquetas explícitas `HLTV: #N` y `VRS: #N` del
HTML comprimido que ya capturaba el pipeline. No requiere peticiones extra.
Se guardan aditivamente en `match_team_ranking_observations`, incluyendo:

- ID estable del equipo y del partido, y referencias al core cuando existen.
- Fuente HLTV/Valve, posición y edición exacta obtenida de su URL fechada.
- Fecha/hora REAL de captura, URL de ranking y partido, archivo y SHA-256.

No se inventan puntos desde una posición. Links sin edición/ID, externos,
contradictorios o con publicación posterior a la captura no producen una
observación utilizable. Los archivos deben coincidir con su hash conservado.
La inserción es idempotente y BLACKBOX incluye la nueva tabla.

Una etiqueta de una ficha antigua puede referirse a una edición antigua;
por ello NO reemplaza el ranking global ni modifica Elo o Glicko.
Para investigación, `observations_asof(D)` exige edición y captura anteriores
al día D: capturar hoy una página histórica no crea conocimiento del pasado.

Es razonable probar posiciones/puntos y diferencias entre ambas fuentes como
señales adicionales de fuerza. No deben transformarse arbitrariamente en Elo:
la posición es ordinal, no una distancia calibrada entre equipos. Una futura
evaluación temporal debe medir si aportan algo frente al rating existente y pasar
por la puerta. La integración siguiente las hace candidatas; no fuerza su uso.

## Integración automática en `-Retrain`

`MODEL/cs2model/match_rankings.py` lee la tabla lateral y alimenta la familia
`match_rankings`, separada de `rankings`. No cambia las columnas ni los valores
que consume un modelo antiguo. El entrenamiento y la inferencia comparten el
mismo constructor causal, incluido el intercambio del orden de los equipos.

- Diferencias de posición HLTV y Valve/VRS, positivas a favor de team1.
- Indicadores de disponibilidad por fuente y global. Ausencia = valor neutro
  con indicador 0; no se inventan posiciones ni puntos.
- Antigüedad de la edición, no del último intento de descarga.
- Edición y captura UTC demostrable estrictamente anteriores al día D.
- Se elige la edición más reciente conocida por equipo; ambos deben tener la
  misma edición. Conflictos o ediciones diferentes dejan esa fuente ausente.
- Las recapturas no rejuvenecen la edición ni multiplican filas del training.
  La evidencia conserva las fechas, URLs y hashes seleccionados por partido.

No requiere un flag nuevo: `start.ps1 -Retrain`, `CS2/start.ps1 -Retrain` y el
reentreno automático llegan a `MODEL/train.py`. La familia participa en el
universo de candidatas y la selección temporal dentro de cada fold, no sobre
etiquetas del hold-out exterior. `feature_thresholds.match_rankings` exige
200 partidos cerrados con evidencia causal, no 200 capturas. También necesita
cobertura de validación (20 por defecto) y mejora de log-loss (0,0005 por defecto),
tanto frente al núcleo como al añadirla a las otras familias elegidas.

El entrenamiento imprime `match_rankings READY/WAIT (N/200)` y registra cobertura,
decisión y motivo en `feature_policies.match_rankings`. El contrato es
`dated_match_badges_v1`. La puerta champion/challenger final sigue decidiendo la
promoción sobre las mismas filas temporales; disponer de nuevos rankings nunca
mueve directamente `latest` ni elimina el modelo anterior.

Auditoría inicial del lote del 12 de septiembre de 2026: 687 observaciones, todas
capturadas ese día; 0 partidos históricos utilizables entre 11.054 terminados
antes de deduplicar, cuyo último resultado era del propio día 12. El lote espera
partidos posteriores a la captura y muestra suficiente; aún no demuestra una
mejora real de accuracy. Las regresiones prueban selección de señal sintética,
rechazo de variables sin señal, as-of y paridad training/serving.

## Operación

`start.ps1` lo incorpora mediante la ingesta normal. Para una auditoría sin red:

```powershell
VAULT\CS2\.venv\Scripts\python.exe CS2\BBDD\ranking_store.py --runs VAULT\CS2\PIPELINE\runs\<run>
```

Sin `--apply` es solo lectura; con él se insertan únicamente las observaciones
verificadas. La tabla sagrada `prediction_ledger` no se modifica por esta API.

Verificación local del 12 de septiembre: sobre 262 HTML ya archivados se
insertaron 687 observaciones (435 HLTV y 252 VRS), sin errores de hash ni
cuarentena. Cubren 142 y 128 equipos distintos, respectivamente; las cifras
de observaciones incluyen capturas repetidas, no son partidos independientes.
Pasaron las puertas CS2 (359 tests, 17 subtests), los 45 tests de integración
del scraper y el smoke determinista. La prueba de BLACKBOX incluye una
observación poblada y exige el mismo hash antes/después de restaurar.

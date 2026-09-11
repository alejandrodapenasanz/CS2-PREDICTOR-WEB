# Pistols y conversiones: adquisición causal y experimento

## Contrato y alcance

La hipótesis es que el historial de pistols y de las rondas siguientes aporta
información para predecir el **ganador del partido**. No se presupone una mejora
ni se transforma automáticamente una pistol ganada en tres rondas ganadas.

La tabla pública `https://www.hltv.org/stats/teams/pistols?csVersion=CS2` sirve
como referencia agregada. No se ha verificado que `rankingFilter=Top200`
signifique todos los equipos; omitir el filtro evita asumirlo. Este cambio NO
incorpora un descargador de esa tabla: aprovecha los HTML por mapa que el pipeline
ya captura, con más detalle para auditar las conversiones.

Cambiar `startDate`/`endDate` en una consulta hecha hoy limita el periodo de los
resultados, pero **no acredita que la respuesta estuviera disponible entonces**.
Nunca se retrofecha una descarga nueva. El backtest usa capturas originales cuyo
hash SHA-256 y timestamp se conservan junto al HTML comprimido. Para predecir en
el día D se exige, simultáneamente, `played_day < D` y `captured_day < D`.

## Adquisición y persistencia

- `PIPELINE/round_history.py` valida IDs nativos, equipos, formato, iconos,
  parciales, ganador de cada ronda, orientación CT/T y marcador final.
- Se admiten MR12/MR15 y múltiples prórrogas, pero solo MR12 desde CS2 entra en
  estas features. Las pistols son las primeras rondas de cada mitad reglamentaria;
  la prórroga nunca añade pistols.
- `BBDD/round_history_store.py` cruza los IDs del HTML con partido, mapa y equipos
  de la base. Ambigüedades, identidades contradictorias o hashes incorrectos se
  excluyen y aparecen en el informe; los archivos originales no se borran.
- `map_round_sources` y `map_rounds` son tablas aditivas. La primera guarda
  procedencia, captura original, fecha del mapa, hash y versión del parser.
  La segunda guarda las rondas. Repetir la ingestión es idempotente.
- La ingestión habitual incorpora las capturas del run después de insertar los
  mapas; no hace nuevas peticiones HTTP. BLACKBOX preserva ambas tablas y tiene
  prueba de backup/restauración con datos poblados.
- Ni el backfill ni el experimento escriben en `prediction_ledger`.

Desde `CS2/`:

```powershell
.\.venv\Scripts\python.exe BBDD/round_history_store.py
.\.venv\Scripts\python.exe BBDD/round_history_store.py --apply
.\.venv\Scripts\python.exe MODEL/run_pistol_ablation.py
```

El primer comando solo audita. El segundo incorpora el archivo disponible a las
tablas laterales. El tercero entrena y evalúa sin sustituir producción.

## Features y ausencia de historial

Se usa la primera captura válida por mapa, ventana previa de 90 días, y al menos
10 pistols por equipo para activar diferencias de tasas. No se descartan las
filas sin cobertura: conservan indicadores y tamaños de muestra explícitos.

Cada tasa se suaviza mediante `(éxitos + 20 × 0,5) / (N + 20)`. Es un prior
neutral fijo, no una tasa de conversión aprendida de datos futuros. Se emiten
diferencias equipo1 menos equipo2; el aumento de datos por intercambio de equipos
invierte esas diferencias, pero no los indicadores simétricos de cobertura.

| Variable | Definición y denominador |
| --- | --- |
| win / ct_win / t_win | Pistols ganadas entre pistols jugadas, total o por lado |
| convert2 / convert3 | Ronda 2 / 3 ganada, condicionada a ganar la pistol |
| sweep3 | Ganar las rondas 2 y 3 tras ganar la pistol |
| break2 / recover3 | Ronda 2 / 3 ganada tras perder la pistol |

Solo cuentan en cada denominador las rondas realmente observadas. Si un mapa
termina antes, no se inventan conversiones. El informe ofrece tasas empíricas,
denominadores y media de rondas ganadas en las tres primeras tras ganar la pistol.

No se utiliza el mapa, veto ni lado inicial del partido objetivo si no existe
evidencia previa. Estas tasas simples no añaden ajuste propio por rival o cambios
de roster: el modelo conserva sus ratings y otras señales existentes.
El cálculo es único para entrenamiento e inferencia. El modelo actual ignora
estas columnas; solo un artefacto aprobado que las incluya las consumiría.

El experimento optativo `--opponent-adjusted` añade un ajuste causal por la fuerza
del rival y compara también la predicción de pistols; ver
[Pistols ajustadas por rival](PISTOL_OPPONENTS.md).

## Comparación y promoción

Se reconstruye TODO el historial cronológicamente antes de separar entrenamiento
y hold-out. Se mantiene la arquitectura, pesos y método de calibración del vivo.
Se comparan un control reentrenado sin columnas nuevas, una familia mínima de
pistols y una familia de pistols + conversiones. Todos usan las mismas filas.

La calibración se separa por días enteros. Antes de entrenar y sin mirar resultados
de challengers, se comprueba la partición habitual; si no deja 100 partidos con
cobertura en el ajuste base, se prueban los últimos 14 y después 7 días. Se exigen
además 100 filas de calibración. Si no se cumple, el experimento para con
`insufficient_causal_training_coverage`. Cada intento queda documentado. Esta
ventana corta limita la estabilidad de la calibración y debe constar al interpretar
los resultados. La versión actual para si el vivo cambia a un router hasta
verificar el corte por días de la calibración de cada régimen.

El hold-out es estrictamente posterior al corte de entrenamiento del vivo. La
puerta compartida decide por log-loss y Brier, con los márgenes de configuración.
Accuracy y ECE se reportan, pero no reemplazan esa decisión. Se identifica el
conjunto por IDs/fechas/resultados y hash; repetir inferencia del mismo artefacto
sobre las mismas filas debe coincidir dentro de `1e-12`.

`--promote` requiere la puerta favorable, las puertas de salud del candidato y del
artefacto registrado, y verificación del modelo serializado. Despliega exactamente
el artefacto evaluado, **nunca un reentreno posterior al hold-out**. Usa la API de
promoción compartida con comprobación de hashes y del puntero vivo esperado.
Sin promoción no se crean versiones de modelos rechazados en el registro.

Los intervalos CI95 usan bootstrap emparejado por día (2.000 réplicas, semilla 42);
esto es estimación de incertidumbre, no un split aleatorio de entrenamiento.
Son exploratorios, sin corrección por comparar dos familias. Un intervalo que
incluya cero no demuestra mejora. El control distingue el efecto de reentrenar
del de añadir las estadísticas. No se promete una mejora futura por ganar una
comparación aislada.

Resultados reproducibles: `MODEL/results/pistol_rounds/ablation.json` e
`ingestion.json`. Contienen cobertura, periodos, N, métricas globales y por
cobertura, intervalos, decisiones y ejemplos de picks que cambian.

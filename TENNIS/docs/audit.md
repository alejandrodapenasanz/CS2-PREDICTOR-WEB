# Auditoría final anti-fugas y de honestidad

## Estado del informe

Este documento registra el contrato comprobable y los resultados de los
artefactos reconstruidos tras las correcciones. No se copian métricas ni
recuentos de runs superseded.

| Componente | Estado documental |
|---|---|
| Revisión de contratos y correcciones de código | Implementada; evidencia descrita abajo |
| Regeneración de cuarentena diagnóstica | Completada: esquema v3, 90 claves, commit `8373358...5a8d`; uso solo operativo actual |
| Reconstrucción Elo v5 | Completada: runs `elo-m-837d7b...` y `elo-f-ec4cff...`; SQLite íntegra |
| Reconstrucción features v2 | Completada: fingerprint `bfe639b5...b1f93`; 1.728.855 filas |
| Reentreno y backtest | Completado: modelo v3 `e827272ef3e55f36f9c048850e2e61b19241533464499fb070e28d300f7ef690`; 20 folds temporales |
| Suite completa y E2E | `329/329 tests correctos` |
| Ejecución diaria/BBDD/web | `429 filas; 130 programados; 83 predicciones calibradas oficiales; 346 degradadas sin inventar; 83 publicadas en WEB; ejecución final desde caché sin nuevas peticiones web` |

Los artefactos anteriores permanecen superseded y los cargadores fallan cerrado
si un hash, inventario de código o contrato aguas arriba no coincide.

## Resumen de hallazgos y correcciones

| ID | Hallazgo | Riesgo | Corrección |
|---|---|---|---|
| F9-01 | Sackmann usa normalmente `tourney_date` como inicio aproximado del torneo, no como día real de la ronda. | Una ronda futura del mismo torneo podía entrar demasiado pronto en Elo, forma, H2H, descanso o entrenamiento. | Política central `sackmann-tourney-start-embargo-v1`: 21 días de embargo y disponibilidad estricta desde `D+22`. |
| F9-02 | La DOB del maestro actual se estaba interpretando como si la colisión de un ID hubiera sido observable históricamente. | Selección retrospectiva de filas mediante información adquirida después. | La cuarentena DOB es solo diagnóstico para inferencia operativa actual; la exclusión histórica queda desactivada. |
| F9-03 | `RET`, `DEF`, `ABD` y `ABN` se excluían junto a partidos no iniciados. | Universo retrospectivo artificialmente limpio y sesgo de selección. | Se incluyen cuando hay ganador/perdedor distintos; solo `W/O`, `Walkover` y `BYE` se excluyen por no acreditar inicio. |
| F9-04 | La orientación de serving podía depender de cuál jugador aparecía como A. | Probabilidades distintas al intercambiar jugadores. | Predicción cruda y calibrada simétrica: se evalúan ambas orientaciones y se promedian de forma complementaria. |
| F9-05 | Una BBDD con resultados diarios podría convertirse en un bucle de realimentación no auditado. | Labels duplicados, tardíos o conciliados incorrectamente podrían contaminar reentreno. | `BBDD/tennis.sqlite3` se usa para monitorización prospectiva; no se concatena automáticamente con Sackmann ni alimenta modelos. |
| F9-06 | Los artefactos anteriores conservaban contratos incompatibles. | Mezcla silenciosa de Elo, features, modelo o código de versiones diferentes. | Versiones Elo v5/features v2/modelo v3, fingerprints, hashes, contratos aguas arriba e inventarios de código con carga fail-closed. La evidencia final exige regeneración completa. |

## 1. Fugas temporales

### Política común de fecha fuente

`src/temporal.py` es la única definición admitida:

```text
source_date = tourney_date
result_available_date = source_date + 21 días
usable(as_of_date) ⇔ result_available_date < as_of_date
```

La igualdad se rechaza. Un resultado fuente `D` no es observable en `D+21` y
puede entrar por primera vez en `D+22`. El parámetro es configurable, pero su
valor y versión forman parte de fingerprints y manifiestos; no puede cambiarse
sin producir artefactos distintos.

| Información | Fuente y corte | Evidencia ejecutable | Conclusión |
|---|---|---|---|
| Elo general/superficie | Estados efectivos con `state_date < D` | `src/elo/build.py`, `src/elo/engine.py`, `src/elo/service.py` | El resultado actual se previsualiza sin mutar; solo se aplican resultados ya disponibles. |
| Forma reciente | Solo resultados disponibles; ventanas calculadas con su fecha fuente | `src/features/dataset.py`, `src/features/state.py` | Ningún label con disponibilidad `>=D` entra en los contadores. |
| H2H global/superficie | Mismo estado causal disponible | `src/features/state.py`, `src/features/vector.py` | El partido actual no entra en su propio H2H. |
| Descanso | Última fecha fuente entre resultados ya disponibles | `src/features/state.py`, `src/features/vector.py` | No usa rondas aún embargadas; puede ser conservador por el retraso. |
| Ranking/puntos | Último snapshot limpio con `ranking_date < D` | `src/features/rankings.py` | Igualdad y futuro quedan excluidos; conflictos retroceden a snapshot limpio. |
| Edad | DOB maestra y fecha fuente `D` | `src/features/players.py` | DOB ausente/imposible queda nula y marcada; la DOB no selecciona historia. |
| Cuotas históricas | Sackmann no aporta timestamps causales | `src/features/market.py`, esquema v2 | Permanecen nulas; no se fabrican retrospectivamente. |
| Cuotas diarias | Solo son feature/edge si la captura es estrictamente anterior a la fecha del partido | `src/daily_pipeline/pipeline.py` | Las cuotas del mismo día pueden mostrarse como `prestart_unverified`, pero no entran al modelo ni generan edge causal. |
| Label de train/calibración | `result_available_date` estrictamente anterior al siguiente corte | `src/modeling/splits.py`, `src/modeling/backtest.py` | Train y calibración no consumen resultados que aún no estarían disponibles. |
| Modelo final | `result_available_date < training_as_of_date` | `src/modeling/training.py`, `src/modeling/service.py` | El bundle registra corte, máxima fecha fuente y máxima disponibilidad; serving falla cerrado si no preceden a la predicción. |

### Evidencia de tests

- Embargo, igualdad y primer día utilizable: `tests/test_temporal_embargo.py`.
- Invariancia Elo ante futuro: `tests/test_elo_engine.py` y
  `tests/test_elo_pipeline.py`.
- Invariancia de features ante futuro: `tests/test_feature_dataset.py` y
  `tests/test_feature_state.py`.
- Corte de folds y reentreno: `tests/test_modeling_splits.py`,
  `tests/test_model_backtest.py` y `tests/test_model_training.py`.

Resultado focal y suite consolidada: `329/329 tests correctos`.

### Limitación no eliminable con esta fuente

Veintiún días es un embargo conservador, no una prueba de la fecha real de cada
ronda. Un torneo excepcionalmente largo, suspendido o reprogramado puede
terminar después. La solución completa requeriría fechas reales por partido
verificadas; mientras no existan, el backtest conserva incertidumbre residual.

## 2. Fuga por orientación A/B

La orientación histórica se deriva de
`SHA256((42, gender, source_record_hash))`, no de ganador/perdedor ni del orden
de carga. `y=1` significa que ganó A. Reordenar el dataset o añadir futuro no
cambia la orientación de una fila existente.

En inferencia se calculan `P(A gana | A,B)` y la probabilidad complementaria de
la fila invertida; su promedio complementario produce la probabilidad cruda
simétrica. Platt se ajusta y aplica con el mismo contrato simétrico. Los IDs,
nombres, roles fuente, `y`, `model_probability_a` y `edge` están fuera de
`MODEL_FEATURE_COLUMNS`.

Evidencia: `src/modeling/orientation.py`,
`tests/test_model_orientation.py`, `tests/test_feature_market_orientation.py`.

- Balance regenerado: `M=0,4993555922` (478.892/959.020) y
  `F=0,5004267148` (385.246/769.835).
- Al intercambiar A/B, probabilidad cruda y calibrada suman 1 con precisión de
  15 decimales; el residuo admitido por el test es como máximo `10^-15`.
- El orden ganador/perdedor de la fuente, IDs, nombres y `y` no forman parte de
  `MODEL_FEATURE_COLUMNS`. La orientación depende de la identidad hash del
  partido, no de su posición de carga; el test sintético de 10.000 identidades
  exige que el ganador aparezca en A entre 47 % y 53 %.

## 3. Fuga de calibración o selección de modelo

Para un test de temporada `Y`:

```text
train: temporadas <= Y-2 y result_available_date < inicio de calibración
calibración: temporada Y-1 y result_available_date < inicio de test Y
test: temporada Y
```

El preprocesador y el estimador se ajustan solo con train. El calibrador Platt
se ajusta con probabilidades de calibración generadas por ese estimador ya
congelado. Para el modelo final, el calibrador usa exclusivamente predicciones
OOF temporales disponibles antes de `training_as_of_date`, nunca predicciones
in-sample.

Los hiperparámetros son constantes versionadas en
`src/modeling/parameters.py`; el pipeline no ejecuta búsqueda ni early stopping
contra calibración o test. Semilla de orientación, logística y LightGBM: `42`.
Si se incorporase tuning en el futuro, necesitaría un tramo interior adicional
y una nueva auditoría.

El run contiene 10 folds por género, con tests 2016–2025. Acumula 277.270
predicciones OOF en `M` y 270.604 en `F`; son exactamente las filas usadas por
los calibradores finales. En cada fold, la máxima disponibilidad de train
precede el inicio de calibración y la máxima disponibilidad de calibración
precede el inicio del test. El modelo final se cortó en
`training_as_of_date=2026-08-09`; las máximas disponibilidades admitidas son
2026-06-22 (`M`) y 2026-06-23 (`F`).

## 4. Honestidad del backtest

No existe split aleatorio. La ventana es expansiva, el test avanza por temporada
y cada fold registra rangos fuente y máximas fechas de disponibilidad de train,
calibración y test. Accuracy, log-loss, Brier y AUC se calculan globalmente y
por género × nivel × superficie. Los baselines obligatorios son favorito por
ranking y favorito por mercado de-vigado sobre soporte común.

El mercado histórico todavía no es evaluable: Sackmann carece de cuotas con
timestamp causal. Por tanto, no se puede afirmar que el modelo supere al
mercado. La BBDD prospectiva permitirá medirlo cuando exista muestra suficiente,
sin convertir esa observación automáticamente en entrenamiento.

### Reproducción final

| Comprobación | Resultado |
|---|---|
| Fingerprint del run de modelos | `e827272ef3e55f36f9c048850e2e61b19241533464499fb070e28d300f7ef690` |
| Métricas globales LightGBM Platt `M` | accuracy 0,690782; log-loss 0,577382; Brier 0,197600; AUC 0,763160; n=277.270 |
| Métricas globales LightGBM Platt `F` | accuracy 0,703855; log-loss 0,561391; Brier 0,191027; AUC 0,779665; n=270.604 |
| Métricas por nivel × superficie | 429 filas segmentadas reproducibles en `evaluation/metrics.csv` (220 `M`, 209 `F`) |
| Brier antes/después de Platt | `M`: 0,197668 → 0,197600; `F`: 0,191031 → 0,191027 |
| Favorito por ranking | accuracy `M` 0,648952 sobre 251.199 filas; `F` 0,643530 sobre 189.623 |
| Comparación mercado sobre soporte común | `NO EVALUABLE hasta disponer de cuotas históricas/prospectivas suficientes` |
| Segmentos con accuracy >85 % | 30 celdas auditadas; todas con n<200 y marcadas como muestra pequeña, no como evidencia de rendimiento extraordinario |

Las 30 celdas sospechosas se explican por soporte mínimo: 26 son `M/Team` en
grass o carpet (`n=2..17`, accuracy 0,8571..1,0) y 4 son `F/Team/Clay`
(`n=38`, accuracy 0,8684..0,8947). El inventario confirma cero IDs de registro
duplicados, contrato temporal válido y cero predictors prohibidos. La conclusión
es «muestra pequeña; no se detectó fuga estructural»; por tanto, esas celdas no
se consideran fiables ni se presentan como éxito.

## 5. Robustez operativa y ausencia de datos inventados

| Caso límite | Comportamiento exigido | Evidencia |
|---|---|---|
| Día sin partidos | DataFrame/CSV válido vacío; ejecución sin probabilidades inventadas | `tests/test_daily_pipeline.py`, E2E |
| Jugador sin mapear | Partido conservado, `UNAVAILABLE` y flag `player_unmapped`; cola de revisión | tests de mapping y pipeline diario |
| Jugador sin histórico | Cold start 1500 visible, muestra cero y confianza baja/no disponible según contrato | tests Elo y confianza |
| Cuota ausente o unilateral | Mercado/edge nulos; nunca completar el lado faltante | tests de mercado y pipeline diario |
| Cuota del mismo día | Probabilidad de-vigada presentable como no verificada; no es feature ni edge causal | tests del pipeline diario |
| HTML inesperado | Error de esquema claro; no devolver filas fabricadas | tests de parser Tennis Explorer |
| Walkover/cancelado | Sin predicción oficial ni settlement deportivo inventado | tests diarios/operativos |
| Resultado ambiguo o slug discordante | No liquida; registra revisión | tests de `operations` |
| Replay de fecha pasada | Puede producir diagnóstico, nunca rendimiento oficial retrospectivo | tests de BBDD operativa |

Resultado de pruebas forzadas y E2E de fixture: `329/329 tests correctos`.

## 6. Reproducibilidad e integridad de artefactos

- Semillas fijas: `42` para orientación, logística y LightGBM.
- Fórmulas, embargo, esquema y parámetros forman parte de fingerprints.
- Elo, features y modelos se publican como runs inmutables con manifiestos,
  tamaño y SHA-256; los punteros activos cambian solo tras validar el run.
- El contrato features v2 fija el fingerprint, versión, parámetros y política
  del Elo v5 consumido.
- El bundle del modelo fija fuente, corte de reentreno, máxima fecha disponible,
  calibración OOF y contratos aguas arriba.
- Los inventarios de código se recalculan al cargar; una ausencia o diferencia
  obliga a reconstruir en lugar de deserializar/servir silenciosamente.
- `run_tennis.ps1 -Retrain [-Date D]` ejecuta fuentes → diagnóstico de
  identidades → Elo → features → modelos → predicción con el mismo corte `D`.
- La BBDD es append-only para observaciones y preserva la primera predicción
  oficial; no es fuente automática de train.

La reactivación de un run completo conserva el fingerprint y valida hashes
antes de mover el puntero; una corrupción, archivo ausente o inventario de
código distinto falla cerrado. Esta idempotencia y la publicación atómica están
cubiertas por `329/329 tests correctos`.

## Valoración honesta de fiabilidad

Las métricas temporales respaldan usar el modelo como estimador comparativo en
segmentos amplios de ATP, WTA y Grand Slam cuando mapping, superficie, ranking e
historial están completos. Aun así, «HIGH» solo describe cobertura y frescura
de inputs: no garantiza acierto ni rentabilidad. Sin cuotas históricas causales
no se ha demostrado ninguna ventaja sobre el mercado.

Los segmentos candidatos a mayor confianza serán aquellos con mapping completo,
histórico suficiente, superficie conocida, rankings recientes y soporte amplio
en folds temporales. Su aceptación depende de log-loss, Brier, calibración y
tamaño por segmento regenerados, no solo de accuracy.

No conviene fiarse de:

- ITF y parte de Challenger con jugadores nuevos, IDs no resueltos, rankings
  ausentes o poco historial;
- superficies/niveles con muestra pequeña, especialmente `Carpet` y contextos
  raros;
- probabilidades extremas acompañadas de `LOW` o `UNAVAILABLE`;
- partidos con identidad DOB incompatible, superficie desconocida, HTML
  ambiguo, estado no programado o cuotas parciales;
- cualquier supuesto edge de mercado sin una observación estrictamente causal
  y una muestra prospectiva suficiente.

La posible contaminación histórica de IDs reutilizados y la aproximación de 21
días son limitaciones reales. El sistema las marca y documenta; no puede
resolverlas inventando identidades ni fechas.

## Cierre

La auditoría encontró y corrigió fugas/selecciones deshonestas reales, reconstruyó
desde cero Elo, features y modelos y dejó trazabilidad desde commit fuente hasta
serving. La valoración final sigue siendo deliberadamente conservadora: el
backtest es temporal y reproducible, pero el embargo de 21 días aproxima fechas
reales y los posibles IDs históricos reutilizados no pueden repararse sin una
fuente de identidad fechada. La ejecución operativa final queda registrada como
`429 filas; 130 programados; 83 predicciones calibradas oficiales; 346
degradadas sin inventar; 83 publicadas en WEB; ejecución final desde caché sin
nuevas peticiones web`.

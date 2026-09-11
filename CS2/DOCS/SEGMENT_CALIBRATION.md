# Calibración por segmento (CS2)

## Alcance

Este diagnóstico mide si la probabilidad media predicha coincide con la tasa
real en tres cortes prepartido:

- diferencia **absoluta** de Elo as-of: `0-50`, `50-100`, `100-200`,
  `200-300`, `300+`;
- stage anunciado;
- entorno `LAN` / `online`.

La fuente principal son las predicciones cerradas OOS del walk-forward
temporal. El Elo usado para agrupar es el emitido antes del resultado. El
seguimiento live lee exclusivamente filas `evaluated` y congeladas de
`prediction_ledger`; no modifica el ledger ni realimenta el entrenamiento.

## Métricas y criterio de lectura

Cada segmento contiene `N`, probabilidad media, tasa real, gap de calibración
(`real - predicha`), accuracy, log-loss, Brier, ECE y las celdas de la curva de
fiabilidad. El umbral normal es `N >= 100`; por debajo aparece
`insufficient_sample` y no se interpreta.

El estado `deviation_detected` es un **screening descriptivo**, no una prueba
causal ni una autorización para cambiar el modelo. Exige un gap medio material
con intervalo residual del 95% fuera de cero, o ECE material respaldado por una
celda de fiabilidad con al menos 20 casos. Hay comparaciones múltiples y ECE
tiene sesgo muestral: una señal aislada se trata como hipótesis.

La muestra live necesita además 1.000 resultados evaluados antes de poder
participar siquiera en una revisión por segmentos. Alcanzar ese N tampoco
autoriza una promoción: la decisión siempre se toma en el backtest temporal y
por la puerta champion/challenger.

Los umbrales están versionados en `MODEL/config.yaml`, bajo
`segment_calibration`.

## Escalera de decisión (manual)

1. Si los segmentos calibran de forma compatible con ruido muestral, no se
   cambia nada.
2. Si una señal material se repite con cobertura suficiente, se ejecuta el
   challenger general con `--segment-interactions`. La familia añade
   `elo_diff x stage`, `elo_diff x LAN/online` y `elo_diff x tramo absoluto`.
   Primero debe ganar el hold-out interno temporal fold-local y luego la puerta
   de promoción contra producción sobre el mismo hold-out causal.
3. Solo si un segmento es grande, la desviación persiste y las interacciones no
   la resuelven, se diseña un modelo dedicado. También es challenger y pasa por
   la misma puerta.

No hay saltos automáticos entre escalones. El artefacto vivo y `last_good`
permanecen protegidos por la puerta existente.

## Comandos

Desde `CS2/`:

```powershell
# Informe del backtest temporal ya cerrado
.\.venv\Scripts\python.exe MODEL\measure_segment_calibration.py

# Seguimiento live desde el ledger (salida completa queda en results/)
.\.venv\Scripts\python.exe MODEL\evaluate_live_ledger.py

# Experimento opcional; no fuerza selección ni promoción
.\.venv\Scripts\python.exe MODEL\train.py --segment-interactions

# Comparación reproducible: bloque completo vs interacción mínima swiss
.\.venv\Scripts\python.exe MODEL\run_segment_interaction_ablation.py --promote
```

Salidas:

- `MODEL/results/segment_calibration_backtest.json` y
  `SEGMENT_CALIBRATION_BACKTEST.md`;
- `MODEL/results/live_ledger_evaluation.json`, con desglose global y por hash de
  artefacto;
- `MODEL/results/segment_interaction_ablation/segment_interaction_ablation.json`,
  con la cohorte común, métricas dentro/fuera de la puerta y calibración swiss.

## Limitaciones actuales

El backtest contiene miles de predicciones, pero la cobertura histórica de
stage y LAN/online empieza mucho más tarde que la de ratings. Se informa el N y
la cobertura desconocida en vez de inferir contexto ausente. En particular, la
muestra live actual no decide por segmento.

## Resultado de la prueba de interacciones (2026-08-26)

La muestra larga actualizada contiene 7.978 predicciones OOS. La señal swiss
persiste (N=214, gap real-predicha `+0,0687`, CI95 `[+0,0050, +0,1325]`), pero
el hold-out independiente de promoción solo contiene 87 partidos swiss y no
repite el gap: el vivo tiene `-0,0060`, CI95 `[-0,1099, +0,0979]`. Por tanto,
ese subgrupo del hold-out no se interpreta aisladamente.

La interacción mínima mejora ligeramente las métricas globales, pero no supera
los márgenes anti-churn; el bloque completo empeora. La puerta rechaza ambas,
no cambia producción y `--segment-interactions` permanece apagado. El informe
completo está en `MODEL/results/SEGMENT_INTERACTION_ABLATION.md`.

# Auditoría OOS frente a puerta champion/challenger

Fecha de auditoría: 2026-08-26. Alcance: CS2. La tabla OOS describe un
backtest prequential con un refit por periodo; la tabla de puerta describe un
único shadow congelado en el corte del vivo. Solo la segunda decide promoción.

## Ejecución que originó la discrepancia

Bundle auditado: `20260824_125852Z`. Incumbente:
`20260810_064827Z`. Receta:
`13592ca482a1a9c2cab471e5eea73cea413184626d91ef01ff79a2466e09e90d`.

| Evaluación | Periodo | N | Con odds | Sin odds | Ajuste/calibración | Objeto | Log loss | Brier |
|---|---|---:|---:|---:|---|---|---:|---:|
| OOS prequential | 2026-07-19 → 2026-08-23 | 707 | 484 (68,46 %) | 223 (31,54 %) | Legacy: sigmoid 20 %; rama odds sin pesos de recencia | Router distinto por fold semanal | 0,631208 | 0,220360 |
| Puerta fija | 2026-08-11 → 2026-08-23 | 260 | 174 (66,92 %) | 86 (33,08 %) | Sigmoid 18 %; half-life 365 días | Un shadow ajustado hasta 2026-08-10 | 0,699512 | 0,249280 |

Los 260 partidos de la puerta están dentro de los 707 OOS: son el 36,78 % del
OOS y el 100 % de la puerta. No reciben, sin embargo, las mismas probabilidades:
en el OOS los folds posteriores pueden reajustarse con resultados anteriores a
cada fold; el shadow permanece congelado durante todo el sufijo. Por tanto no
era correcto presentar `0,6312` y `0,6995` como dos mediciones del mismo objeto.

La familia/arquitectura sí era la misma (`router_two_models` con reserva
`super_learner_cal`) y la huella de receta coincide en selección y puerta. El
pickle final del bundle tiene SHA-256
`abdce6bcb69c3777b2bb3ef02bd82389bda83bfb7ed10aa1adc75aae0b445471`,
pero fue ajustado hasta 2026-08-23. Tampoco es el shadow efímero ajustado solo
hasta el corte y, por anti-fugas, no puede sustituirlo en la evaluación.

### Reconstrucción de la puerta histórica por régimen

| Régimen | N | Vivo accuracy | Challenger accuracy | Vivo log loss | Challenger log loss | Vivo Brier | Challenger Brier | Vivo ECE | Challenger ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Con odds | 174 | 59,20 % | 59,20 % | 0,658459 | 0,710109 | 0,233771 | 0,253444 | 0,068864 | 0,171380 |
| Sin odds | 86 | 60,47 % | 59,30 % | 0,675553 | 0,678071 | 0,239893 | 0,240857 | 0,076805 | 0,082294 |

El router no cayó casi siempre a reserva. El perjuicio se concentró en la rama
con odds: discriminaba algo mejor por AUC, pero estaba mucho peor calibrada y
pagó esa sobreconfianza en log loss. La rama sin odds quedó cerca del vivo, pero
también ligeramente por detrás. El rechazo histórico no fue causado por un
filtro que eliminase la mayoría de las odds.

La reconstrucción encuentra las mismas 260 identidades y reproduce exactamente
los dos agregados guardados (`0,6995116383058115` y `0,2492803504999391`). La
BBDD viva, no obstante, recibió enriquecimientos posteriores: su fingerprint
crudo ya no coincide con el manifest del 24/08 y la huella completa de
probabilidades no es idéntica. Por ello el desglose histórico es una
reconstrucción métrica, no una prueba byte a byte. El defecto de trazabilidad se
ha cerrado: desde esta corrección cada nueva decisión persiste directamente el
periodo, los conteos y métricas por régimen.

## Defecto encontrado y corrección

La puerta en sí ya emparejaba vivo y challenger por `match_id`, fecha y etiqueta,
y les entregaba las mismas features point-in-time. Sí había una asimetría en el
diagnóstico OOS que inducía a compararlo indebidamente con la puerta:

- el primario OOS usaba la cola de calibración por defecto 20 %, sin los pesos
  productivos de recencia;
- el shadow usaba 18 % y half-life de 365 días;
- la evaluación aceptaba la política global de reserva en vez de exigir
  explícitamente la familia congelada para promoción;
- el informe no distinguía un stream con refits de un shadow fijo.

Ahora el OOS usa los parámetros productivos y la reserva congelada, declara
`prequential_refit_per_period`, y la puerta persiste contrato, periodo y
desglose. Además, el mismo shadow se puntúa dos veces sobre la misma cohorte y se
exige igualdad de probabilidades, log loss y Brier con tolerancia absoluta
`1e-12`.

La cuota se valida mediante la misma función en ambas rutas: primera apertura
guardada, dos decimales coherentes, `captured_at_utc < kickoff_utc` y de-vig a
suma uno. No se encontró un filtro causal más estricto en la puerta.

## Reejecución corregida

El entrenamiento completo del 26/08 mantuvo la receta router y produjo:

| Evaluación | Periodo | N | Con odds | Sin odds | Accuracy | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OOS prequential corregido | 2026-07-19 → 2026-08-26 | 771 | 529 | 242 | 66,67 % | 0,624275 | 0,217236 | 0,025496 |
| OOS, solo con odds | mismo | 529 | 529 | 0 | 67,67 % | 0,622775 | 0,216392 | 0,044821 |
| OOS, solo sin odds | mismo | 242 | 0 | 242 | 64,46 % | 0,627554 | 0,219082 | 0,033953 |

Puerta fija corregida, 324 partidos del 11/08 al 26/08:

| Régimen | N | Vivo accuracy | Challenger accuracy | Vivo log loss | Challenger log loss | Vivo Brier | Challenger Brier | Vivo ECE | Challenger ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Todos | 324 | 62,04 % | 62,65 % | 0,648969 | 0,673535 | 0,228952 | 0,237012 | — | — |
| Con odds | 219 | 60,73 % | 62,10 % | 0,651598 | 0,687334 | 0,230568 | 0,242260 | 0,047982 | 0,142598 |
| Sin odds | 105 | 64,76 % | 63,81 % | 0,643485 | 0,644756 | 0,225580 | 0,226065 | 0,061648 | 0,065102 |

La comprobación mismo shadow/mismas filas pasó: delta de log loss `0`, delta de
Brier `0` y máxima diferencia de probabilidad `2,22e-16`, dentro de `1e-12`.
La puerta rechazó (`delta log loss = +0,024567`), así que no hubo promoción.

Estado final validado:

- `latest`: `20260810_064827Z`, SHA-256
  `23ee5b52561e975192c873a35bcfbfdbd3293eb9ce52f134f33971fd7d439b47`;
- runtime `MODEL/artifacts/model.pkl`: mismo SHA-256;
- `last_good`: `20260803_064101Z`, SHA-256
  `15edae6fdae96334f81f36754c95cdfca1174a741cc0391635aea5f45d231844`.

## Verificación

- puerta local CS2: `cs2_quality_gate=ok`;
- pytest: 292 passed, 4 skipped, 17 subtests passed;
- cobertura de imports: 100 ficheros, 13 imports externos, `ok`;
- boots reales: `train`, auditoría, retrain, gestión de modelos, ledger,
  calibración de segmentos, BBDD y enriquecimiento, `ok`;
- smoke determinista: `smoke_pipeline=ok`, 160 filas, seed 42.

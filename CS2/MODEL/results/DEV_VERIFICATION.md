# Verificacion modelo rama dev - 2026-07-06

## Resumen ejecutivo

- Datos: 9.444 series (2025-09-11 -> 2026-07-06), 7.008 filas de test walk-forward.
- CatBoost: instalado y validado en el sweep profesional completo.
- Modelo de produccion final: `ensemble3_cal`.
- Componentes: `logistic + lightgbm + catboost`.
- Calibracion: `sigmoid` (Platt).
- Configuracion final: `--form-half-life 90 --wf-gap 0`.
- Artefacto final: `MODEL/artifacts/model.pkl`, validado con `load_artifact()`.
- Predicciones/web/BBDD: regeneradas con el artefacto `ensemble3_cal`.

La metrica de seleccion es **log loss walk-forward**, no accuracy bruta. Para un sistema
de apuestas importa que las probabilidades esten calibradas, porque EV/Kelly dependen de
la calidad probabilistica.

## Resultado final con CatBoost

| Config | Modelo produccion | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `h90_gap0` | `ensemble3_cal` | 0.629060 | 0.219697 | 0.691178 | 0.019894 | 0.644977 |

Accuracy final: **64.4977%**.

## Barrido profesional half-life / gap con CatBoost

| Config | Modelo | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `h90_gap0` | `ensemble3_cal` | 0.629060 | 0.219697 | 0.691178 | 0.019894 | 0.644977 |
| `h60_gap0` | `ensemble3_cal` | 0.629241 | 0.219785 | 0.690732 | 0.018490 | 0.644264 |
| `h45_gap0` | `ensemble3_cal` | 0.629294 | 0.219814 | 0.690574 | 0.017982 | 0.643693 |
| `h180_gap0` | `ensemble3_cal` | 0.629481 | 0.219874 | 0.690607 | 0.018541 | 0.643550 |
| `h120_gap0` | `ensemble3_cal` | 0.629681 | 0.219983 | 0.689929 | 0.018696 | 0.641695 |
| `h60_gap1` | `ensemble3_cal` | 0.630500 | 0.220364 | 0.688376 | 0.017363 | 0.644121 |
| `h45_gap1` | `ensemble_beta` | 0.630551 | 0.220432 | 0.687727 | 0.012745 | 0.643550 |
| `h180_gap1` | `ensemble3_cal` | 0.630650 | 0.220431 | 0.688078 | 0.017739 | 0.645263 |
| `h90_gap1` | `ensemble3_cal` | 0.630687 | 0.220454 | 0.687969 | 0.015533 | 0.644264 |
| `h120_gap1` | `ensemble3_cal` | 0.630755 | 0.220485 | 0.687803 | 0.017425 | 0.643836 |

Nota: `h180_gap1` tiene una accuracy ligeramente mayor (64.5263%), pero peor log loss
que `h90_gap0`. Por eso no pasa a produccion.

## Comparacion contra verificaciones anteriores

| Variante | Modelo produccion | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Baseline `main` | `ensemble_cal` | 0.630430 | 0.220316 | 0.688625 | 0.016633 | 0.648116 |
| Dev sin CatBoost h45 | `ensemble_beta` | 0.629296 | 0.219890 | 0.690081 | 0.015874 | 0.646689 |
| Dev profesional CatBoost h90 | `ensemble3_cal` | 0.629060 | 0.219697 | 0.691178 | 0.019894 | 0.644977 |

Delta final CatBoost vs baseline `main`:

- Log loss: -0.001370 (mejora; menor es mejor).
- Brier: -0.000619.
- ROC-AUC: +0.002553.
- ECE 10: +0.003261.
- Accuracy: -0.003139.

Veredicto: **MEJORA PROFESIONAL EN LOG LOSS / BRIER / AUC**. La accuracy baja frente
al baseline por 0.31 puntos porcentuales, pero la seleccion correcta para betting es la
probabilidad calibrada. La ECE empeora ligeramente, aunque queda dentro del margen
aceptable definido por el runbook (`ECE_dev <= ECE_baseline + 0.01`).

## Artefacto final

- `production_model`: `ensemble3_cal`
- `production_calibration`: `sigmoid`
- `production_components`: `['logistic', 'lightgbm', 'catboost']`
- `catboost_enabled`: `True`
- `form_half_life_days`: `90.0`
- `walk_forward_gap`: `0`
- `n_train_rows`: `9444`
- `date_min`: `2025-09-11`
- `date_max`: `2026-07-06`

## Regeneracion posterior

- `PIPELINE/enrich_predictions.py`: OK, cargo `ensemble3_cal` y genero 17 predicciones.
- `MODEL/analyze_context_calibration.py`: OK, contexto sigue en `insufficient_sample` (90 muestras, minimo 200).
- `BBDD/build_db.py`: OK, `predictions_rows=17`, `odds_rows=484`, `player_stat_snapshots_rows=3712`.
- `WEB/build_web.py`: OK, `WEB/data.js` generado con 17 partidos y modelo `ensemble3_cal`.

## Archivos de auditoria

- Output completo: `MODEL/results/professional_train_20260706_173628/professional_training_output.md`.
- Copia timestamped: `MODEL/results/professional_train_20260706_173628/professional_training_output_20260706_173628.md`.
- Reportes por configuracion: `MODEL/results/professional_train_20260706_173628/`.
- Reporte final activo: `MODEL/results/REPORT.md`.

Los logs grandes quedan ignorados por git para no versionar artefactos pesados, pero se
mantienen en disco para estudiar el entrenamiento.

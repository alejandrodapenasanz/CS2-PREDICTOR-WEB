# Verificacion modelo rama dev - 2026-07-06

## Resumen

- Datos: 9.444 series (2025-09-11 -> 2026-07-06).
- CatBoost: instalado, pero el smoke exacto con CatBoost agoto 20 min y se descarto para esta verificacion. El A/B final se hizo con `--no-catboost`.
- Smoke test sintetico sin CatBoost: OK. Mejor candidato sintetico: `ensemble_beta`.
- Scraper online: OK en run debug `2026-07-06_133146Z`, `scrapling_successes=44`, bloqueos=0, errores=0, Analytics=3/3.

## A/B principal

| Variante | Modelo produccion | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Baseline `main` | `ensemble_cal` | 0.630430 | 0.220316 | 0.688625 | 0.016633 | 0.648116 |
| Dev default h120 | `ensemble_beta` | 0.630166 | 0.220256 | 0.688767 | 0.014963 | 0.643408 |
| Dev final h45 | `ensemble_beta` | 0.629296 | 0.219890 | 0.690081 | 0.015874 | 0.646689 |

Delta final vs baseline:

- Log loss: -0.001134 (mejora; menor es mejor).
- Brier: -0.000426.
- ROC-AUC: +0.001456.
- ECE 10: -0.000759.
- Accuracy: -0.001427.

Veredicto: **MEJORA**. El objetivo principal es log loss/calibracion; la caida de accuracy es de 0.14 puntos porcentuales y queda compensada por mejor log loss, Brier, AUC y ECE.

## Barrido de half-life / gap

| Config | Modelo | Log loss | Brier | ROC-AUC | ECE 10 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `--form-half-life 45` | `ensemble_beta` | 0.629296 | 0.219890 | 0.690081 | 0.015874 | 0.646689 |
| `--form-half-life 60` | `ensemble_beta` | 0.629595 | 0.219998 | 0.689827 | 0.015684 | 0.644692 |
| `--form-half-life 90` | `ensemble_beta` | 0.629549 | 0.219977 | 0.689897 | 0.014715 | 0.644406 |
| `--form-half-life 120` | `ensemble_beta` | 0.630166 | 0.220256 | 0.688767 | 0.014963 | 0.643408 |
| `--form-half-life 180` | `ensemble_beta` | 0.629857 | 0.220115 | 0.689455 | 0.014720 | 0.644549 |
| `--wf-gap 1` | `ensemble_beta` | 0.631095 | 0.220672 | 0.686782 | 0.013939 | 0.642694 |

Seleccion final: `--form-half-life 45 --no-catboost --wf-gap 0`.

## Artefacto final

- `production_model`: `ensemble_beta`
- `production_calibration`: `beta`
- `production_components`: `['logistic', 'lightgbm']`
- `catboost_enabled`: `False`
- `form_half_life_days`: `45.0`
- `walk_forward_gap`: `0`

El artefacto se verifico con `load_artifact()` y una prediccion minima (`p_zero_row=0.525456063657838`). `DAILY_SNAPSHOTS/enrich_predictions.py` tambien cargo el modelo correctamente y genero 18 predicciones.

## Notas

- El fallo inicial de carga del artefacto con `ensemble_beta` revelo que `BetaCalibratedClassifier` estaba serializado como `__main__`. Se corrigio moviendolo a `MODEL/cs2model/calibration.py`.
- La prueba online del scraper detecto y corrigio problemas de PowerShell en `start.ps1`: quoting de `python -c`, tracebacks de import probing y contaminacion del pipeline por stdout/stderr nativo.
- El run debug con `-MaxMatches 3` no promueve master, por diseno.

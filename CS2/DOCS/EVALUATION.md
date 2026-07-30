# EVALUATION — harness de medición honesta (walk-forward anidado)

Este paso **no busca subir accuracy**: busca **medirla honestamente**. El sesgo
L1 detectado en [AUDIT.md](AUDIT.md) ya está cerrado: tanto producción como el
harness anidado separan *decidir* de *medir* y ninguna etiqueta elige el modelo
que la predice.

- Motor: [`CS2/MODEL/cs2model/evaluation.py`](../MODEL/cs2model/evaluation.py) (numpy puro, fitter enchufable).
- CLI: [`CS2/MODEL/evaluate.py`](../MODEL/evaluate.py).
- Config: sección `evaluation:` en [`CS2/MODEL/config.yaml`](../MODEL/config.yaml).
- Tests: [`CS2/TESTS/test_eval_framework.py`](../TESTS/test_eval_framework.py).
- Produccion: `train.py` usa desde 2026-07-27 la misma disciplina causal para
  seleccionar candidatos; el harness sigue siendo la certificacion independiente.

## Estado actual (2026-07-27)

- **L1 cerrado en produccion:** `nested_model_policy` elige el candidato de cada
  semana usando solo OOS de semanas anteriores. La metrica publicada pertenece a
  esa politica; `production_model` es el candidato elegido con todo el OOS ya
  cerrado para ajustar el siguiente periodo.
- **L1 cerrado en el zoo:** `nested_policy` decide modelo y assembly dentro de
  cada `outer_train`. `diagnostic_best_combo` nunca se presenta como evidencia
  promocionable sobre el mismo outer test.
- **Calibracion sin in-sample:** se ajusta sobre OOS interno y el metodo se elige
  en una cola temporal separada.
- **L2 real:** `TESTS/test_leakage_audit.py` reconstruye todas las columnas
  enriquecidas por prefijos de `cs2.db` y audita los selectores as-of.

---

## 1. Walk-forward ANIDADO

```
BUCLE EXTERNO (estimación insesgada):
  para cada bloque de test T_k (paso = outer_step semanas, tras warmup):
      outer_train = periodos < inicio(T_k) - GAP          # pasado puro
      ── BUCLE INTERNO (todas las decisiones, solo con outer_train) ──
         · activación de familias por cobertura en outer_train
         · hiperparámetro (L2) por walk-forward interno
         · método de calibración (identity/Platt/isotónica) por log loss interno
         → se congela la combinación de MENOR log loss interno
      refit en TODO outer_train con esa combinación
      predice T_k  (una sola vez; NUNCA se reutiliza para decidir)
  MÉTRICAS = agregado sobre las predicciones OOS externas
MODELO DE PRODUCCIÓN = reentrenar con todo el histórico (mismo protocolo interno)
```

| Decisión | Bucle | Datos que ve |
|---|---|---|
| Activar familia (umbral cobertura) | interno | solo `outer_train` |
| Hiperparámetro L2 | interno | walk-forward interno de `outer_train` |
| Método de calibración | interno | OOS interno |
| **Métricas reportadas** | externo | `T_k` — no influye en nada |
| Modelo de producción | reentreno final | todo el histórico |

- **Gap/embargo** (`gap_periods`, def. 1): separa fin de train y test en ambos bucles.
- **Expansiva vs deslizante**: `--window both` corre las dos y las compara.
- **Fitter enchufable**: el núcleo trae un logístico numpy (reproducible, sin deps);
  el fitter de producción (LightGBM/super-learner) se inyecta sin cambiar el protocolo.

## 2. Métricas (artefacto `eval_metrics.json` + reporte + gráfico)

- **log loss** (PRIMARIA, se mantiene como criterio de selección), **Brier**, **ECE**
  (+ curva de fiabilidad `reliability_curve.png`), **ROC-AUC**, **accuracy**.
- **Apuestas vs Model B** sobre el subconjunto con cuotas: **ROI** a stake plano
  (lado = favorito del modelo con EV+ contra la **apertura**) y **CLV** (cierre −
  apertura del lado apostado). El cierre **solo** audita CLV, nunca decide la apuesta.

## 3. Baselines (apples-to-apples, mismo OOS externo)

1. **Modelo (nested)** — el pipeline honesto.
2. **Elo** — probabilidad centrada del rating.
3. **Mercado / Model B** — probabilidad implícita de la cuota de apertura.
4. **"65%" recomputado** — accuracy del favorito del modelo en el OOS externo
   (`recomputed_favorite_accuracy`). Toda mejora futura se compara contra estos.

## 4. Cómo ejecutarlo

```powershell
cd CS2
# Smoke reproducible (sin cs2.db; valida el harness en segundos):
python MODEL\evaluate.py --synthetic --window both

# Baseline REAL sobre cs2.db (en tu máquina, con el stack completo):
python MODEL\evaluate.py --window both
python MODEL\evaluate.py --window expanding --outer-step 4 --gap 1
```
Artefactos en `MODEL/results/`: `eval_metrics.json`, `eval_calibration.json`,
`reliability_curve.png`, `EVAL_REPORT.md`, y el registro append-only
`experiments.jsonl` (un manifest por run: commit git, hash de datos, params, métricas).

## 5. Auditoría de leakage (tests que FALLAN si algo mira al futuro)

`assert_point_in_time(rows, build_features, keys)` reconstruye la feature de la
fila *i* usando **solo** `rows[:i+1]` y exige igualdad con el cálculo global: si una
feature mira al futuro, difieren → `PointInTimeError`. Tests incluidos:
- **future-feature detectada** (falla como debe) y **feature limpia** (no falso positivo).
- **odds**: el cierre nunca cambia la apuesta (mismo ROI con cierres distintos; solo cambia el CLV).
- **recencia causal**: los pesos dependen solo de fechas ≤ t y son monótonos.
- **gap respetado**: `train_period_max ≤ test_period_min − gap` en cada fold.
- **el test externo no influye**: las decisiones internas no dependen del OOS externo.
- **snapshots reales enriquecidos**: odds de apertura, Analytics, lineups y stats
  de jugador elegidos por los loaders nunca cruzan el inicio del partido.

> Límite honesto: la reconstrucción detecta features que usan partidos **futuros**.
> El uso del **propio resultado** del partido se previene por diseño (la BBDD es
> point-in-time; `enrich`/`dataio` hacen joins as-of) y con la disciplina de no
> incluir columnas de resultado en el vector.

## 6. Reproducibilidad

- Seeds fijas (`EvalConfig.seed`, def. 42 desde `config.yaml`).
- Config versionada (`evaluation:` en `config.yaml`; overrides por CLI `--gap/--outer-step`).
- **Run manifest** por evaluación (commit git, SHA-256 del dataset, params, versiones,
  métricas) → línea en `experiments.jsonl`.
- `requirements.txt` conserva rangos; `requirements.lock.txt` fija el grafo
  completo resuelto con Python 3.12 y es el fichero instalado por CI.

## 7. Números del baseline (estado)

Recomputado sobre la BBDD real el 27-07-2026. El adaptador conserva 47 periodos
semanales distintos; una prueba de regresión impide volver a colapsar las fechas
en un único periodo. PAV/isotónica también se valida con 100.000 observaciones
para evitar overflow en históricos grandes.

| Ventana | n_eval | log_loss (modelo) | ROC-AUC | fav_acc recomputada |
|---|---:|---:|---:|---:|
| expanding (real) | 7.290 | 0,6689 | 0,6179 | 59,30% |
| sliding (real) | 7.290 | 0,6689 | 0,6179 | 59,30% |

Este CLI usa el fitter logístico NumPy mínimo y actúa como control independiente
del protocolo; no es el candidato productivo. Elo obtiene log loss `0,6578`.
La evaluación del zoo con fitters reales está en `MODEL_COMPARISON.md`: su
`nested_policy` logra `0,6338`, mientras producción logra `0,6318`.

Expanding y sliding son iguales mientras los conjuntos train/test de todos los
folds coincidan (47 semanas frente a `train_width=52`). `compare_models.py`
comprueba los conjuntos exactos y reutiliza el resultado equivalente; cuando el
histórico supere la ventana, volverá a ejecutar ambos.

Los tests (`pytest TESTS/test_eval_framework.py`) pasan (10/10).

---

*La medicion independiente y la seleccion productiva comparten la misma regla:
ninguna etiqueta puede decidir el modelo que la predice.*

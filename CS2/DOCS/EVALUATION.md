# EVALUATION — harness de medición honesta (walk-forward anidado)

Este paso **no busca subir accuracy**: busca **medirla honestamente**. Arregla el
sesgo L1 de [AUDIT.md](AUDIT.md): hoy el modelo de producción se selecciona
sobre el **mismo** conjunto OOS que luego se reporta. El harness anidado separa
*decidir* de *medir*.

- Motor: [`CS2/MODEL/cs2model/evaluation.py`](../MODEL/cs2model/evaluation.py) (numpy puro, fitter enchufable).
- CLI: [`CS2/MODEL/evaluate.py`](../MODEL/evaluate.py).
- Config: sección `evaluation:` en [`CS2/MODEL/config.yaml`](../MODEL/config.yaml).
- Tests: [`CS2/TESTS/test_eval_framework.py`](../TESTS/test_eval_framework.py).
- No toca `train.py` (producción intacta). Es una capa de medición aparte.

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

> Límite honesto: la reconstrucción detecta features que usan partidos **futuros**.
> El uso del **propio resultado** del partido se previene por diseño (la BBDD es
> point-in-time; `enrich`/`dataio` hacen joins as-of) y con la disciplina de no
> incluir columnas de resultado en el vector.

## 6. Reproducibilidad

- Seeds fijas (`EvalConfig.seed`, def. 42 desde `config.yaml`).
- Config versionada (`evaluation:` en `config.yaml`; overrides por CLI `--gap/--outer-step`).
- **Run manifest** por evaluación (commit git, SHA-256 del dataset, params, versiones,
  métricas) → línea en `experiments.jsonl`.
- Deps críticas **pineadas** en `requirements.txt` (`numpy<3`, `pandas<3`,
  `scikit-learn<2`, `lightgbm<5`) para poder recargar `model.pkl` y reproducir números.

## 7. Números del baseline (estado)

**No es posible recomputar el baseline REAL en el entorno de desarrollo** (sin
`cs2.db`, sin red, sin stack ML completo). Lo que sí está **validado ejecutando** el
harness aquí es el propio motor, sobre **datos sintéticos deterministas** (seed 42):

| Ventana | n_eval | log_loss (modelo) | ROC-AUC | fav_acc recomputada |
|---|---:|---:|---:|---:|
| expandido (sintético) | 700 | ~0.624 | ~0.72 | ~0.654 |
| deslizante (sintético) | 700 | ~0.623 | ~0.72 | ~0.656 |

> Son números **sintéticos** que solo demuestran que el harness corre, reporta
> baselines y reproduce el patrón "~65% de acierto del favorito". El baseline
> **real** (recomputo honesto del 64.4% de `REPORT.md`, que se espera **≤** por
> quitar el sesgo de selección L1) se obtiene con `python MODEL\evaluate.py --window both`
> sobre tu `cs2.db`. Si me pasas la BBDD, lo recomputo y actualizo esta tabla.

Los tests (`pytest TESTS/test_eval_framework.py`) pasan (8/8) con numpy puro.

---

*Rama `feature/eval-framework`. Mide, no optimiza. Producción (`train.py`) sin cambios;
en un PR posterior se alineará su selección a este protocolo anidado.*

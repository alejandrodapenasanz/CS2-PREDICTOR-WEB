# PROMPT.md — Contexto completo para una IA que retoma el proyecto

Léeme **primero y entero**. Tras esto tendrás todo el contexto de los cambios
recientes en el dominio **CS2**. Todo el trabajo reciente toca **solo `CS2/`**;
por eso la documentación vive ahora dentro de `CS2/DOCS/` (antes había un `/docs`
en la raíz, ya eliminado y reubicado aquí).

> Regla de oro del proyecto: **la BBDD guarda la verdad con timestamp; features y
> ratings son vistas point-in-time reconstruibles.** El objetivo de modelado es
> **bajar log loss y mejorar calibración y CLV**, no perseguir accuracy bruta
> (techo realista ~65% acc / ~0.6 log loss; el 85% de algunos papers es leakage
> in-game).

---

## 0. Mapa de documentación (todo en `CS2/DOCS/`)

| Doc | Qué contiene |
|---|---|
| `PROJECT.md` (en `CS2/`) | Especificación de diseño original (teoría, decisiones). Fuente de verdad conceptual. |
| `README.md` (en `CS2/`) | Comandos y estado operativo. |
| [AUDIT.md](AUDIT.md) | Auditoría inicial: arquitectura, flujo de datos, el 65%, riesgos de leakage, deuda. |
| [RECOVERY.md](RECOVERY.md) | Cómo respaldar/reconstruir la BBDD (herramienta BLACKBOX). |
| [PIPELINE.md](PIPELINE.md) | `start.ps1` refactorizada: etapas, flags, config, logging, exit-codes. |
| [EVALUATION.md](EVALUATION.md) | Harness de evaluación honesto (walk-forward anidado). |
| [MODEL_UPGRADE.md](MODEL_UPGRADE.md) | Zoo de modelos, block-wise, calibración, veredicto "¿híbrido?". |
| `CHANGELOG.md` | Registro cronológico de decisiones. |
| `extra_features.md` | Catálogo de features futuras. |

---

## 1. El proyecto en 30 segundos
Predicción **pre-partido** de Counter-Strike 2 (series Bo3) desde datos de HLTV,
orientada a **probabilidades calibradas**. Estructura: `CS2/BBDD` (SQLite fuente de
verdad), `CS2/MODEL` (núcleo ML), `CS2/PIPELINE` (scrape + enrich diario),
`CS2/SCRAPER` (cliente HLTV), `WEB/` (dashboard compartido, en la raíz),
`CS2/start.ps1` (orquestador de una orden; wrapper compatible en la raíz).

---

## 2. Entorno y límites (IMPORTANTE, honestidad)
- **Los datos NO están en git** (`.gitignore`): `cs2.db`, seed histórico, runs,
  artefactos, backups. Un clon limpio trae **solo código**. Ver [RECOVERY.md](RECOVERY.md).
- El desarrollo de estos cambios se hizo en un entorno **sin `cs2.db`, sin red a
  HLTV y con Python 3.14** (sin todo el stack ML). Por eso: las piezas numéricas se
  validaron con **numpy/scipy/scikit-learn/optuna/LightGBM** sobre **datos
  sintéticos**; xgboost/catboost/TabPFN quedaron **import-guarded**. **Los números
  reales (baseline, comparativa de modelos) se recomputan en la máquina del usuario**
  con los comandos indicados. No confundir números sintéticos con reales.

---

## 3. Qué se ha hecho (5 líneas de trabajo)

### 3.1. Auditoría → [AUDIT.md](AUDIT.md)
Radiografía completa del proyecto. Hallazgos clave que condicionan todo lo demás:
- El **"65% de accuracy"** = **0.6443** walk-forward OOS (`MODEL/results/REPORT.md`),
  **casi igual al baseline** Elo/Glicko. El valor del modelo está en log loss/
  calibración/AUC, no en accuracy.
- **L1 (sesgo de selección):** producción (`train.py`) selecciona el modelo sobre el
  **mismo** OOS que reporta → optimista. Lo arregla el harness anidado (3.4).
- **L2:** el test de fuga original solo cubría el núcleo rolling. Ampliado (3.4).
- Umbrales de activación de familias eran **fijos** (154/209/200/50-LAN); se
  sustituyen por decisión **data-driven** (3.5).
- Deps sin pin (parcialmente resuelto), datos fuera de git (resuelto con BLACKBOX).

### 3.2. BLACKBOX — respaldo/reconstrucción → [RECOVERY.md](RECOVERY.md)
- `CS2/BBDD/blackbox.py` (stdlib): `export`/`verify`/`restore`/`autoheal`. Empaqueta
  **solo la fuente de verdad** de `cs2.db` en `CS2/BBDD/BLACKBOX/` (SQLite podado +
  volcado SQL, gzip, manifest con SHA-256). Guardián: no pisa una BBDD sana.
- Cableado en `start.ps1`: **auto-heal al inicio** (restaura si `cs2.db` falta/vacía/
  corrupta) + flags `-BackupBlackbox`, `-RestoreBlackbox`, `-SkipAutoHeal`.
- `CS2/BBDD/BLACKBOX/README.md` + `restore_standalone.py`: restaurar a mano sin el
  proyecto. Test: `CS2/TESTS/test_blackbox_disaster.py`.

### 3.3. Refactor de `start.ps1` → [PIPELINE.md](PIPELINE.md)
- Mismas etapas/orden/gates que antes (**comportamiento preservado**, con y sin
  `-Retrain`), pero profesionalizado: **modelo de etapas** con progreso/cronometraje/
  `timing.json`, **logging con niveles** (`-LogLevel`, JSONL), **`-DryRun`/`-WhatIf`**,
  **validación de parámetros**, **exit-codes por clase de fallo**, y **config-driven**
  (`CS2/PIPELINE/pipeline.config.psd1` + `pipeline.helpers.ps1`).
- Único borrado: la variable muerta `$baseOk`. El **doble ingest se conservó** (no es
  redundante: `enrich` lee `cs2.db`).

### 3.4. Harness de evaluación honesto → [EVALUATION.md](EVALUATION.md)
- `CS2/MODEL/cs2model/evaluation.py` + `CS2/MODEL/evaluate.py`. **Walk-forward
  ANIDADO**: bucle externo = estimación insesgada; bucle interno = TODAS las
  decisiones (con datos pasados) + gap/embargo. Arregla **L1**.
- Métricas (log loss primaria, Brier, ECE + curva de fiabilidad, AUC, accuracy),
  baselines **Elo** y **mercado/Model B**, **recomputo honesto del 65%**, y
  **auditoría de leakage** ampliada (`assert_point_in_time` que FALLA si una feature
  mira al futuro). Reproducibilidad: seeds, sección `evaluation` en `config.yaml`,
  registro `experiments.jsonl`. Test: `CS2/TESTS/test_eval_framework.py`.

### 3.5. Upgrade de modelo → [MODEL_UPGRADE.md](MODEL_UPGRADE.md)
- **Zoo** (`cs2model/model_zoo.py`): elasticnet, RF, ExtraTrees, HistGB, LightGBM;
  xgboost/catboost/TabPFN import-guarded. **Política de missing**: árboles NaN-native;
  lineales/bosques impute+flag. Optuna TPE en el inner; **half-life de recencia como
  hiperparámetro**.
- **Block-wise** (`cs2model/blockwise.py`): 3 estrategias (indicators/profiles/
  two_stage) + **activación de familias data-driven** por CV (entra si baja el log
  loss OOS; sustituye umbrales fijos).
- **Calibración** (`cs2model/calibration_suite.py`): Platt/isotónica/Beta/Venn-Abers,
  en split temporal, antes de la capa de EV.
- **CLV** vs línea de cierre en el reporte de apuestas. Model A **sin odds**.
- CLI `CS2/MODEL/compare_models.py` → `MODEL_COMPARISON.md`. Test: `test_model_zoo.py`.
- **Veredicto empírico "¿híbrido lo mejor?": NO claramente.** El modelo único con
  indicadores iguala/bate a profiles/two_stage en log loss (los híbridos solo mejoran
  algo la calibración) porque las familias enriquecidas están poco cubiertas
  (~87–209 de 9.703). Detalle y recomputo real en [MODEL_UPGRADE.md](MODEL_UPGRADE.md).

---

## 4. Cómo asegurar "el mejor modelo" al hacer `start.ps1 -Retrain`

**Qué hace hoy `-Retrain`:** ejecuta `MODEL/train.py`, que **ya** hace walk-forward
temporal, **selecciona por menor log loss**, entrena `super_learner_cal` con
calibración y Optuna purgado, y versiona el artefacto. Es un modelo **sólido y
seleccionado por log loss** — el "mejor" según el pipeline de producción **actual**.
Además ahora arranca con **auto-heal de BLACKBOX** (restaura `cs2.db` si falta) y deja
`timing.json`.

**Matiz honesto (no lo ocultes):** la selección de `train.py` mira el mismo OOS que
reporta (sesgo **L1**). El harness nuevo (`MODEL/evaluate.py` anidado y
`MODEL/compare_models.py` con el zoo) mide **sin** ese sesgo y puede elegir un Model A
mejor calibrado, pero **todavía NO está cableado a `-Retrain`** (decisión deferida a
propósito: no se recablea producción sobre evidencia sintética).

**Receta recomendada (empírica) para tener de verdad el mejor modelo:**
1. `.\start.ps1 -Retrain` → artefacto de producción actual (super_learner por log
   loss). Suficiente para operar la web.
2. `python MODEL\compare_models.py --window both --n-trials 40` sobre tu `cs2.db` →
   comparativa honesta (zoo × block-wise, nested). Mira `MODEL_COMPARISON.md`.
3. Si el `best_combo` del reporte **bate a `train.py` en log loss nested**, ese es el
   candidato a promocionar → *follow-up*: cablear ese ganador en `train.py`
   (pendiente; ver §6).
4. Decide por **log loss/calibración/CLV**, no por accuracy.

> Resumen: `-Retrain` ya te da un modelo bien seleccionado por log loss; para
> **certificar** que es el mejor, contrástalo con `compare_models.py` sobre datos
> reales. No promociones nada a ciegas por resultados sintéticos.

---

## 5. Ramas e integración
Trabajo hecho en ramas apiladas: `feature/blackbox` → `feature/pipeline-refactor`
(fusionadas a `pre-dev`) y luego `feature/eval-framework` → `feature/model-upgrade`.
**Tras este handoff, toda la cadena se integra en `pre-dev`** (fast-forward), de modo
que `pre-dev` contiene: BLACKBOX + refactor de pipeline + harness de evaluación +
upgrade de modelo + esta documentación. Trabaja sobre `pre-dev`.

---

## 6. Decisiones abiertas / próximos pasos
- **Reconciliar producción con el harness anidado** (arreglar L1 en `train.py`):
  cablear el `best_combo` de `compare_models.py` como Model A de producción una vez el
  CV real lo confirme. Es el paso pendiente más importante para "el mejor modelo".
- **Ampliar tests de fuga (L2)** a las familias enriquecidas sobre `cs2.db` real.
- **Pins de deps**: hechas las críticas (numpy/pandas/scikit-learn/lightgbm en
  `requirements.txt`); falta lockfile completo si se quiere reproducibilidad total.
- **Cifras únicas**: `MODEL/results/REPORT.md` es la fuente viva; PROJECT/README
  tienen tablas ilustrativas con fecha.
- Familias `two_stage`/GBDT/nivel-de-mapa: reevaluar cuando crezca la cobertura.

---

## 7. Cómo validar rápido
```powershell
cd CS2
python -m pytest TESTS/test_blackbox_disaster.py TESTS/test_eval_framework.py TESTS/test_model_zoo.py -q
.\start.ps1 -DryRun -SkipScrape            # recorre las etapas sin ejecutar
python MODEL\evaluate.py --synthetic --window both     # smoke del harness anidado
python MODEL\compare_models.py --synthetic --n-trials 8 # smoke del zoo/block-wise
```
El resto de la suite (`pytest TESTS/`) y `ruff`/`mypy`/smoke corren en el CI
(Python 3.12 con todas las deps).

---

## 8. Gotchas
- **No versiones datos ni secretos** (el `.gitignore` ya lo impide; respétalo).
  Los artefactos generados por evaluación (EVAL_REPORT/MODEL_COMPARISON/png/jsonl)
  están gitignored.
- Deps opcionales (xgboost/catboost/TabPFN): el código las salta si no están
  instaladas (`available_models()`); instálalas para usarlas.
- Git avisa de CRLF/LF en Windows; es inofensivo.
- El modelo de producción (`model.pkl`) es sensible a versiones: usa las deps
  pineadas para recargarlo.

*Fin del handoff. Con esto tienes el contexto completo; profundiza en el doc concreto
de `CS2/DOCS/` según la tarea.*

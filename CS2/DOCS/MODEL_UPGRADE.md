# MODEL UPGRADE — zoo de algoritmos + block-wise + calibración (Model A)

Construido sobre el harness anidado de [EVALUATION.md](EVALUATION.md)
(walk-forward anidado, selección por **log loss**). Objetivo (acordado): **bajar
log loss, mejorar calibración y CLV**, no perseguir accuracy bruta. Model A sigue
**sin odds** (el mercado es baseline).

- Zoo: [`cs2model/model_zoo.py`](../MODEL/cs2model/model_zoo.py)
- Calibración: [`cs2model/calibration_suite.py`](../MODEL/cs2model/calibration_suite.py)
- Block-wise + activación + driver: [`cs2model/blockwise.py`](../MODEL/cs2model/blockwise.py)
- CLI/reporte: [`MODEL/compare_models.py`](../MODEL/compare_models.py)

## Estado metodologico actual (2026-07-27)

El reporte separa dos conceptos:

- `nested_policy`: metrica primaria. En cada outer fold elige modelo y assembly
  con una cola temporal de `outer_train`, antes de observar el test.
- `diagnostic_best_combo`: mejor combo fijo al mirar todo el outer OOS. Sirve
  para investigar, pero no se promociona usando esa misma cifra.

El adaptador real expone todas las familias auto-gated de produccion. Una familia
solo entra en el forward selection cuando alcanza su cobertura dentro del train
del fold; despues debe demostrar mejora de log loss OOS. El modo `profiles`
agrupa por patron y ajusta una vez por grupo, no una vez por fila.
Optuna y el forward selection se recalculan cada seis folds externos
(`--retune-folds`, unas 24 semanas con `outer_step=4`); entre medias se reutiliza
exclusivamente la ultima decision causal.

## A. Zoo de algoritmos
Interfaz común `ZooModel` (`.fit/.predict_proba`) con **política de missing** por
modelo. Disponibles según entorno (`available_models()`):
- **logistic_en** — logística elasticnet (saga; `C`, `l1_ratio` tuneados) — baseline calibrado.
- **lightgbm, hist_gb** — GBDT NaN-native. **xgboost, catboost** — import-guarded.
- **random_forest, extra_trees** — bosques sklearn.
- Optuna TPE dentro del bucle interno; **el half-life de recencia es un hiperparámetro más**.
- Stacking: el harness ya combina candidatos por log loss (super-learner); los base
  entran como miembros y el meta se decide con datos pasados.
- Elo/Glicko/TrueSkill/BT/Kalman se quedan como **features**, no como modelos finales.

## B. Missing handling
- **Árboles NaN-native (lightgbm/hist_gb/xgboost/catboost): reciben NaN** — el árbol
  aprende la dirección del hueco; el 0 no ensucia los splits.
- **Lineales / RF / ExtraTrees: imputan (mediana de train) + FLAGS de disponibilidad**.
El flag de cobertura se conserva siempre como columna.

## C. Block-wise missing y activación data-driven
Tres estrategias comparadas por log loss WF (`assembly`):
- **(i) indicators** — un modelo sobre todo con flags.
- **(ii) profiles** — submodelo por patrón de disponibilidad (routing; fallback a base).
- **(iii) two_stage** — base con features siempre disponibles + enriquecido que solo
  actúa donde esas features existen (gating por cobertura).

**Activación de familias DATA-DRIVEN**: primero aplica un minimo de cobertura
causal dentro de `outer_train`; luego un forward-greedy añade una familia **solo
si baja el log loss OOS** en la CV interna (con embargo). El umbral evita decidir
con unas pocas filas; no sustituye la prueba empirica de mejora.

## D. Calibración
Suite en split **temporal** separado: **Platt, isotónica, Beta (Kull), Venn-Abers
(IVAP)**; se elige por log loss en un holdout (evita el optimismo de calibrar y medir
en el mismo set). Salida calibrada **antes de la capa de EV**. Aviso: la isotónica
sobreajusta con n pequeño; por eso se mide en holdout separado.

## E. Feature engineering (reutilizar + añadir)
La mayoría de palancas **ya existen** como familias en `features.py` y ahora entran
por activación data-driven (forma en ventanas, H2H con recencia, LAN/online, tier/
stakes, fatiga, SoS ajustada por rival, player ratings, roster stability). Añadido
genuinamente nuevo (cada uno tras su flag; el CV decide): **Elo por mapa → agregación
Bo3** (sobre `compositional_bo3` existente) y **reset de rating por cambio de roster**.
El modelo por-mapa completo queda como track siguiente si el CV muestra lift del
Elo-de-mapa como feature.

## F. Model A vs Model B + CLV
Model A sin odds. El reporte de apuestas mide ROI (lado = favorito del modelo con
EV+ **vs apertura**) y **CLV vs la línea de CIERRE** (`clv_vs_closing_mean`), que es
el estándar real de habilidad. EV/stake sigue como capa operativa aparte.

## G. Expectativas
Techo ~65% acc / ~0.6 log loss es lo que logran los buenos modelos públicos y las
casas; los 85% de papers usan features in-game (leakage pre-partido). Éxito =
log loss ↓, calibración ↑, CLV ≥ 0.

## Cómo ejecutarlo
```powershell
cd CS2
python MODEL\compare_models.py --synthetic --n-trials 8            # smoke reproducible
python MODEL\compare_models.py --window both --n-trials 40         # REAL (cs2.db)
python MODEL\compare_models.py --models logistic_en,lightgbm,hist_gb,random_forest,extra_trees --assemblies indicators,profiles,two_stage
```
Salida: `MODEL/results/MODEL_COMPARISON.md` + `model_comparison.json` + línea en
`experiments.jsonl`. `best_combo=nested_policy` es la estimacion insesgada de la
politica; el mejor combo fijo se registra aparte como diagnostico.

## Validación real (27-07-2026)

El comando exacto `python MODEL\compare_models.py --window both --n-trials 40
--verbose` se ejecutó sobre 9.726 filas (`n_eval=7.290`). Esta certificación
comparó `logistic_en` y `lightgbm` en los tres assemblies; el artefacto productivo
ya evalúa además CatBoost, XGBoost y Random Forest como miembros del Super Learner.

| Estimación causal | Accuracy | Log loss | Brier | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Zoo `nested_policy` | 0,6379 | 0,6338 | 0,2215 | 0,6837 | 0,0223 |
| Producción `nested_model_policy` | **0,6412** | **0,6318** | **0,2209** | **0,6858** | **0,0099** |

El mejor combo fijo retrospectivo fue `logistic_en|indicators` (`log loss
0,6325`), pero no es promocionable con el mismo OOS usado para identificarlo.
Conclusión: **se mantiene `super_learner_cal` como ajuste de producción**; la
política causal productiva supera al zoo en log loss y calibración.

Los modelos instalados y detectados son logistic_en, Random Forest, ExtraTrees,
HistGB, LightGBM, XGBoost y CatBoost. Las dependencias opcionales siguen
import-guarded para otros entornos.

---

## Veredicto: ¿es el enfoque híbrido lo mejor?

**Respuesta corta: NO de forma clara. El modelo único con indicadores (i) iguala o
bate a los híbridos (ii profiles / iii two-stage) en log loss; la complejidad extra
no compensa porque las familias enriquecidas están demasiado poco cubiertas para
entrenar submodelos fiables.** Fundamentado en tres patas convergentes:

### 1. Empírico (CV sintético con block-missing, 5 modelos, Optuna 6 trials, 3 folds)
Resultado ejecutado (`n_eval=396`; la familia enriquecida cubierta ~parcial, imitando la realidad):

| combo | log_loss | Brier | ECE | ROC-AUC | Acc |
|---|---:|---:|---:|---:|---:|
| market _(baseline)_ | **0.548** | 0.187 | 0.054 | 0.788 | 0.700 |
| elo _(baseline)_ | 0.613 | 0.211 | 0.065 | 0.730 | 0.669 |
| **logistic_en \| indicators (i)** | **0.6431** | 0.226 | 0.095 | 0.696 | 0.649 |
| logistic_en \| two_stage (iii) | 0.6489 | 0.228 | **0.082** | 0.686 | 0.644 |
| logistic_en \| profiles (ii) | 0.6499 | 0.228 | **0.073** | 0.685 | 0.644 |
| lightgbm \| (i/ii/iii) | 0.6844 | 0.244 | 0.110 | 0.607 | 0.599 |

Lecturas: (a) **(i) gana en log loss**; (ii)/(iii) quedan a 3ª decimal por detrás.
(b) Matiz honesto: (ii)/(iii) **sí mejoran la calibración (ECE 0.073/0.082 vs 0.095)**
— tienen valor si el objetivo es puramente calibración. (c) **logistic_en > lightgbm**
a esta escala (n pequeño): el GBDT sobreajusta; coincide con el hallazgo del propio
`REPORT.md` (la logística casi iguala al ensemble). (d) Ningún modelo bate al
**mercado** — exactamente el techo del punto G.

### 2. Coberturas REALES (de `MODEL/results/REPORT.md`)
analytics 154, rankings 209, snapshots 193, box scores 87, roster 140 — todas
~**87–209 de 9.703**. Con tan pocas filas cubiertas, (ii) parte los datos en
submodelos aún más pequeños y (iii) entrena el enriquecimiento con ~200 casos:
**data-starved**. En cambio (i) usa las 9.703 filas y deja que el árbol/lineal
aprenda la dirección del hueco con el flag de disponibilidad.

### 3. Teoría
Submodelos por partición (ii) y modelo de enriquecimiento separado (iii) necesitan
muestra suficiente por bloque; con block-missing severo el estimador único con
indicadores es el uso más eficiente de la información (y evita la varianza de
enrutar/mezclar submodelos ruidosos).

### Recomendación (data-driven, no dogma)
- **Producción por defecto: `logistic_en | indicators` + activación data-driven**
  (una familia entra solo si baja el log loss OOS; en el experimento el greedy
  aceptó `style` solo en 1 de 3 folds — coherente con su cobertura escasa).
- **Conservar `two_stage` como opción que el CV puede elegir** cuando una familia
  se densifique (mejora ECE). Reevaluar (ii)/(iii) cuando analytics/jugador crezcan
  a varios miles de casos.
- **GBDT (lightgbm/hist_gb)**: reservarlos para cuando haya más datos; hoy el
  lineal regularizado es el Model A candidato.
- El **half-life de recencia** salió tuneado a 180–730 días según fold (no 0): la
  ponderación por recencia ayuda y debe seguir siendo hiperparámetro.

> Veredicto **provisional-empírico** (sintético + coberturas reales + teoría, las
> tres apuntan igual). El definitivo se recomputa en tu máquina:
> `python MODEL\compare_models.py --window both --n-trials 40` sobre `cs2.db`.
> Compara `nested_policy` con produccion; `diagnostic_best_combo` no se promociona
> por si solo porque fue elegido mirando el mismo outer OOS.

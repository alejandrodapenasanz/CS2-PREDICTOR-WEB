# LAST change — runbook de verificación (scraper Scrapling + mejora del modelo)

> **Si eres un agente y te acaban de decir "Lee last change.md": este fichero es tu
> encargo completo.** Contiene (A) qué se cambió y por qué, (B) el protocolo exacto
> para verificar que el modelo ha mejorado, y (C) qué reportar. No necesitas más
> contexto. Ejecuta la sección **«AGENTE: EMPIEZA AQUÍ»** paso a paso.

Rama: `dev`. Origen de los cambios: iteración del 2026-07-06.
Repo: predictor pre-partido de CS2 (HLTV → Glicko-2 + GBDT calibrado → web). Detalle
de diseño en `../../PROJECT.md`; features latentes en `../extra_features.md`.

---

## AGENTE: EMPIEZA AQUÍ

### Tu rol y objetivo
Estás en un **PC sin restricciones de red** (a diferencia de la máquina donde se
escribió el código, que tenía inspección TLS corporativa y **no** tenía los datos).
Tu objetivo es **comprobar, de forma rigurosa y sin fuga temporal, que las mejoras del
modelo implementadas en la rama `dev` mejoran (o al menos no empeoran) la calidad
predictiva** respecto al modelo anterior, y dejar constancia de qué se hizo. Todo lo
que se cambió está resumido en las secciones §1 (scraper) y §2 (modelo) más abajo.

La métrica que manda es **log loss walk-forward** (calibración), no la accuracy.
Objetivos secundarios: Brier, ROC-AUC, ECE. El trainer **ya elige el modelo de
producción por menor log loss**, así que las mejoras son *candidatos seguros por
construcción*: solo pasan a producción si ganan.

### Regla de oro de la verificación
La comparación limpia y sin fuga es **código antiguo vs código nuevo sobre LOS MISMOS
DATOS**. No compares contra los números históricos de §3 salvo como referencia
direccional (el dataset ha crecido desde entonces). Haz el **A/B sobre datos idénticos**
del Paso 4.

### Requisitos previos (verifícalos antes de nada)
1. **Datos presentes** (no están en git; deben existir en este PC):
   - `SCRAPER/hltv-scraper-api/hltv_scraper/data/raw/history_10000_*/results_all.json`
   - `PIPELINE/master/matches.json` (opcional; extiende el histórico hasta hoy)
   Si faltan, primero consigue los datos (o ejecuta un scrape completo, ver §1) —
   **sin datos no se puede entrenar ni verificar**.
2. **Python del modelo** con: numpy, pandas, scikit-learn, scipy, matplotlib,
   lightgbm, shap. Instala si falta: `python -m pip install -r requirements.txt`
3. **CatBoost (recomendado)** para activar los candidatos `catboost_cal` /
   `ensemble3_cal`: `python -m pip install catboost`
   (si no instala en tu versión de Python, la verificación sigue siendo válida sin él;
   se omiten esos dos candidatos).
4. **Python 3.13** SOLO si además vas a probar el scraper en vivo (Scrapling no soporta
   3.14). Para verificar el MODELO no hace falta el scraper.

### Protocolo de verificación (ejecuta en orden)

**Paso 0 — Sitúate y confirma rama.**
```powershell
git branch --show-current        # debe decir: dev
git log --oneline -3
```

**Paso 1 — Sanity check del código nuevo (sin datos, ~1 min).**
Ejecuta el smoke test sintético para confirmar que todo el pipeline nuevo corre
(candidatos, calibración beta/isotónica, early stopping, monótonas, artefacto):
```powershell
python - << 'PY'
import sys, random, math
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path("MODEL").resolve()))
import numpy as np; random.seed(0); np.random.seed(0)
import train
from cs2model.features import build_training_frame, FEATURE_COLUMNS, _period_index
skills={f"t{i}":np.random.normal(0,1) for i in range(40)}; teams=list(skills)
rows=[]; start=datetime(2024,1,1)
for k in range(3000):
    a,b=random.sample(teams,2); d=start+timedelta(hours=k*6)
    aw=1 if random.random()<1/(1+math.exp(-(skills[a]-skills[b]))) else 0
    fmt=random.choice(["bo1","bo3","bo3","bo5"]); mx={"bo1":1,"bo3":2,"bo5":3}[fmt]
    s1,s2=(mx,random.randint(0,mx-1)) if aw else (random.randint(0,mx-1),mx)
    rows.append(dict(id=str(k),date=d.strftime("%Y-%m-%d"),date_obj=d,event=f"e{k%12}",
        format=fmt,team1=a,team2=b,team1_key=a,team2_key=b,score1=s1,score2=s2,team1_win=aw))
Xd,y,meta,_=build_training_frame(rows,form_half_life=90.0); cols=list(FEATURE_COLUMNS)
X=train._matrix(Xd,cols); y=np.array(y); per=np.array([_period_index(r["date_obj"]) for r in rows])
hc=train._catboost_available()
preds=train.walk_forward(X,y,per,meta,cols,6,300,gap=1,has_catboost=hc)
m=train.summarize(preds); specs=train.candidate_specs(hc)
for n in specs:
    if n in m: print(f"  {n:16s} logloss={m[n]['log_loss']:.3f} acc={m[n]['accuracy']:.3f}")
print("SMOKE OK · catboost:", hc)
PY
```
Espera ver varias filas de candidatos y `SMOKE OK`. Si falla, para y reporta el error.

**Paso 2 — Captura el BASELINE (código ANTIGUO sobre los datos actuales).**
Guarda el REPORT actual si existe, luego entrena con el código de `main`:
```powershell
if (Test-Path MODEL\results\REPORT.md) { Copy-Item MODEL\results\REPORT.md MODEL\results\REPORT_baseline_prev.md -Force }
git stash push -- MODEL/train.py MODEL/cs2model/features.py 2>$null   # por si hay algo sin commitear
git checkout main -- MODEL/train.py MODEL/cs2model/features.py
python MODEL\train.py
Copy-Item MODEL\results\REPORT.md MODEL\results\REPORT_baseline.md -Force
Copy-Item MODEL\results\metrics.json MODEL\results\metrics_baseline.json -Force
```
`REPORT_baseline.md` = modelo ANTIGUO sobre los datos de hoy. Anota su
**production_model** y su log loss/Brier/AUC/ECE.

**Paso 3 — Entrena el modelo NUEVO (rama dev) sobre los MISMOS datos.**
```powershell
git checkout dev -- MODEL/train.py MODEL/cs2model/features.py
python MODEL\train.py
Copy-Item MODEL\results\REPORT.md MODEL\results\REPORT_dev.md -Force
Copy-Item MODEL\results\metrics.json MODEL\results\metrics_dev.json -Force
```

**Paso 4 — Compara baseline vs dev (A/B sobre datos idénticos).**
Abre `MODEL/results/REPORT_baseline.md` y `MODEL/results/REPORT_dev.md`. Compara la
tabla walk-forward. Fíjate en:
- El **production_model** de cada uno y su **log loss** (menor = mejor) y **Brier**, **ECE**.
- En `REPORT_dev.md`, si aparecen y ganan los candidatos nuevos: `catboost_cal`,
  `ensemble3_cal`, `ensemble_beta`, `ensemble_iso`.
- Que la **ECE** no empeore de forma relevante (calibración).

Puedes cuantificar la delta con:
```powershell
python - << 'PY'
import json
b=json.load(open("MODEL/results/metrics_baseline.json"))
d=json.load(open("MODEL/results/metrics_dev.json"))
def best(m):
    cands={k:v for k,v in m.items() if k not in ("base_rate","elo","glicko")}
    return min(cands.items(), key=lambda kv: kv[1]["log_loss"])
nb,vb=best(b); nd,vd=best(d)
print(f"BASELINE prod={nb:16s} logloss={vb['log_loss']:.4f} brier={vb['brier']:.4f} auc={vb['roc_auc']:.4f} ece={vb['ece_10']:.4f} acc={vb['accuracy']:.4f}")
print(f"DEV      prod={nd:16s} logloss={vd['log_loss']:.4f} brier={vd['brier']:.4f} auc={vd['roc_auc']:.4f} ece={vd['ece_10']:.4f} acc={vd['accuracy']:.4f}")
print(f"DELTA logloss={vd['log_loss']-vb['log_loss']:+.4f} (negativo = mejora)  brier={vd['brier']-vb['brier']:+.4f}  auc={vd['roc_auc']-vb['roc_auc']:+.4f}  ece={vd['ece_10']-vb['ece_10']:+.4f}")
PY
```

**Paso 5 (opcional) — Explora hiperparámetros nuevos** (pueden mejorar más):
```powershell
# half-life del decay de forma (prueba varios y quédate con el de menor log loss):
foreach ($h in 45,60,90,120,180) { python MODEL\train.py --form-half-life $h }
# gap temporal más estricto:
python MODEL\train.py --wf-gap 1
```
(Cada corrida sobrescribe `MODEL/results/REPORT.md`; cópialo si quieres conservarlo.)

**Paso 6 — Deja el artefacto de producción definitivo.**
Vuelve a entrenar con la mejor configuración encontrada (por defecto sin flags ya usa
el código nuevo completo) para regenerar `MODEL/artifacts/model.pkl`, y confirma que
carga y puntúa:
```powershell
python MODEL\train.py
python - << 'PY'
import sys; from pathlib import Path; sys.path.insert(0,str(Path("MODEL").resolve()))
from cs2model.artifacts import load_artifact
a=load_artifact()
print("production_model:", a.metadata.get("production_model"),
      "| calibration:", a.metadata.get("production_calibration"),
      "| components:", a.metadata.get("production_components"),
      "| catboost:", a.metadata.get("catboost_enabled"))
PY
```

**Paso 7 (opcional) — Verifica el scraper anti-bloqueo en vivo** (requiere Python 3.13):
```powershell
.\start.ps1 -MaxMatches 3 -SkipPlayerStats -SkipTeamProfiles
```
Debe crear el venv con 3.13, instalar Scrapling + navegadores y scrapear sin quedar
bloqueado. Revisa `PIPELINE/runs/<run>/fetch_diagnostics.json`: busca
`scrapling_successes` > 0. (Este paso NO es necesario para verificar el modelo.)

### Criterio de aceptación
- ✅ **El modelo ha mejorado** si en el A/B (Paso 4) el `production_model` de `dev`
  tiene **log loss ≤ baseline** (idealmente menor) **sin empeorar la ECE** de forma
  relevante (regla práctica: ECE_dev ≤ ECE_baseline + 0.01).
- ✅ **Aceptable/neutro** si el log loss queda igual (±0.001): las mejoras no dañan y
  añaden robustez (monótonas, early stopping, candidatos extra listos para cuando haya
  más datos).
- ⚠️ **Regresión** si el log loss de `dev` es claramente mayor: repórtalo con los dos
  REPORT y no promuevas; revisa si CatBoost/beta están introduciendo ruido (prueba
  `--no-catboost`).

### Qué reportar (plantilla)
```
VERIFICACIÓN MODELO (rama dev) — <fecha>
- Datos: <n series>  (<fecha_min> -> <fecha_max>)  · CatBoost: <sí/no>
- Baseline (main):  prod=<modelo>  logloss=<x>  brier=<x>  auc=<x>  ece=<x>  acc=<x>
- Dev (mejoras):    prod=<modelo>  logloss=<x>  brier=<x>  auc=<x>  ece=<x>  acc=<x>
- Delta log loss: <±x>   (negativo = mejora)   Veredicto: <MEJORA/NEUTRO/REGRESIÓN>
- Candidatos nuevos que ganaron: <catboost_cal/ensemble3_cal/ensemble_beta/ensemble_iso/ninguno>
- Half-life óptimo probado: <valor>   wf-gap: <0/1>
- Scraper (si se probó): scrapling_successes=<n>, bloqueos=<n>
- Notas / anomalías: <...>
```

---

## 1. QUÉ SE CAMBIÓ — Scraper (robustez + evasión de Cloudflare)

Arquitectura por **tiers** sobre el único choke point `fetch_html` en
`PIPELINE/start.py` (se reutilizan todas las guardas previas: presupuesto,
cooldown, cuarentena, backoff con jitter, caché, circuit breaker):

| Tier | Qué | Cuándo |
|---|---|---|
| 0 | **Caché** en memoria por URL | siempre |
| 1 | **Scrapling `Fetcher(impersonate="chrome")`** — fingerprint TLS/JA3 real (curl_cffi), HTTP/2, orden de cabeceras; reutiliza `cf_clearance` | primario |
| 2 | **Scrapling `StealthyFetcher(solve_cloudflare=True)`** — navegador patchright que resuelve el challenge y **acuña** `cf_clearance` fresca (persistida en `cf_session.json`, reutilizada por Tier 1) | si Tier 1 se bloquea |
| 3-5 | **`requests` → `cloudscraper` → refresh `grab_cf.py`** | red de seguridad |

- **Por qué funciona**: `requests` manda un handshake OpenSSL estático que Cloudflare
  marca (→ 403 "Just a moment"); `curl_cffi` replica el ClientHello de Chrome real.
  *Validado en vivo*: `requests` → **403**; Scrapling impersonate → **200 (360 KB)** en
  `https://www.hltv.org/matches`.
- **Binding cf_clearance ↔ IP + UA + JA3**: el Tier 2 acuña la cookie y guarda la UA;
  los tiers HTTP la reusan con la misma UA (evita el bucle de re-challenge).
- **Detección de bloqueo**: header `cf-mitigated: challenge` + marcadores ampliados
  (`challenge-platform`, `turnstile`, `__cf_chl`, …); UA por defecto Chrome 140.
- **CA corporativa (redes con inspección TLS)**: `start.ps1::Ensure-CaBundle` hace un
  probe TLS y, si detecta MITM, exporta el trust store de Windows a
  `SCRAPER/hltv-scraper-api/corp_ca_bundle.pem` y lo publica por env. **En un PC sin
  restricciones el probe pasa y no genera nada** (usa certifi).
- **Proxy** opcional vía `HLTV_PROXY` (off por defecto).
- **Orquestación** (`start.ps1`): crea el venv del scraper con **Python 3.13** (Scrapling
  no soporta 3.14), instala `scrapling[fetchers]` + navegadores, y **degrada** a
  requests/cloudscraper con aviso si no hay Python 3.10-3.13.

Variables de entorno nuevas (defaults en `start.ps1`): `HLTV_USE_SCRAPLING=1`,
`HLTV_SOLVE_CLOUDFLARE=1`, `HLTV_IMPERSONATE=chrome`, `HLTV_STEALTH_HEADLESS=1`,
`HLTV_STEALTH_TIMEOUT_MS=90000`, `HLTV_STEALTH_MAX_SOLVES_PER_RUN=6`,
`HLTV_SCRAPLING_TIER1_ATTEMPTS=3`, `HLTV_PROXY=` (vacío), `HLTV_CA_BUNDLE=` (auto).

Archivos tocados (scraper): `PIPELINE/start.py`, `start.ps1`,
`SCRAPER/hltv-scraper-api/requirements.txt` (`+ scrapling[fetchers]`).

## 2. QUÉ SE CAMBIÓ — Modelo (más accuracy/calibración, incremental y seguro)

Principio: el trainer **elige producción por menor log loss walk-forward**, así que
todo se añade como **candidato** (solo gana si mejora). Nada requiere datos nuevos ni
relaja la disciplina anti-fuga.

1. **CatBoost** (`catboost_cal`) y **ensemble de 3** (`ensemble3_cal` = Logística ⊕
   LightGBM ⊕ CatBoost, calibrados). *Opcional*: si no está instalado, se omiten.
2. **Restricciones monótonas** en GBDT para features direccionales ("ventaja de team1"
   → P(team1)↑): reducen overfitting con ~7k series. Lista en `MONOTONE_INCREASING`.
3. **Early stopping** en LightGBM/CatBoost sobre holdout interno (log loss); techo
   n_estimators 2000 (antes 350 fijos), LR 0.02, num_leaves 31, min_child_samples 80,
   reg_lambda 5, subsample/colsample 0.8.
4. **Candidatos de calibración**: sigmoid (Platt), **isotónica** y **beta**
   (Kull & Flach 2017; incluye la identidad, no descalibra). Comparten base (barato).
5. **Half-life de decay tunable** (`--form-half-life`, default 120 d, antes fija).
6. **Gap temporal** en walk-forward (`--wf-gap`, default 0) para endurecer anti-fuga.

Descartado a propósito: **stacking meta-aprendido** (el artefacto reproduce medias
ponderadas de componentes, no un meta-modelo); `ensemble3_cal` captura casi todo el
beneficio de forma reproducible. Los ítems estructurales de mayor impacto de la
literatura (composicional Bo3 por mapa/veto; ratings por jugador TrueSkill/WHR) **no**
se incluyen: build mayor y *gated* por muestra point-in-time (ver `../extra_features.md`).

Selección/guardado: `candidate_specs(has_catboost)` define `name -> (kinds, método)`;
`walk_forward` los evalúa; `main` elige el de menor log loss y reconstruye el candidato
como **Components** (estimadores calibrados, peso igual) → `model.pkl` lo reproduce en
`enrich_predictions.py`. Metadatos nuevos: `production_calibration`,
`production_components`, `form_half_life_days`, `walk_forward_gap`, `catboost_enabled`,
`monotone_features`.

Archivos tocados (modelo): `MODEL/train.py`, `MODEL/cs2model/features.py`.

### Verificación ya hecha en la máquina de origen (sin datos reales)
- `py_compile` OK en `start.py`, `train.py`, `features.py`.
- **Smoke test sintético** completo (el del Paso 1): build → walk-forward (con gap) →
  todos los candidatos → selección por log loss → ajuste final → guardar/cargar
  artefacto → predecir → SHAP. Eligió `ensemble_beta` sobre datos sintéticos.
- Scraper Tier 1 validado en vivo (200 en HLTV) con pocas llamadas.

## 3. BASELINE de referencia (histórico, NO usar como A/B — el dataset ha crecido)

Walk-forward previo (era CS2, 9.444 series, 7.008 test) del `REPORT.md` anterior:

| Modelo | Accuracy | Log loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Glicko-2 (baseline) | 0.6092 | 0.6596 | 0.2330 | 0.6493 | 0.0498 |
| Logística (Platt) | 0.6404 | 0.6329 | 0.2214 | 0.6838 | 0.0162 |
| LightGBM (Platt) | 0.6384 | 0.6346 | 0.2222 | 0.6820 | 0.0167 |
| **Ensemble (Platt) — producción anterior** | **0.6481** | **0.6304** | **0.2203** | **0.6886** | **0.0166** |

Úsalo solo como referencia direccional. El A/B válido es baseline (main) vs dev sobre
los **mismos** datos (Pasos 2-4).

## 4. Riesgos / notas
- **Python 3.14**: Scrapling stealth no instala ahí (el scraper degrada a HTTP/requests
  con aviso). Usa 3.13 para el modo completo. CatBoost también puede no tener wheels en
  3.14 → instálalo en un Python soportado o corre sin él (`--no-catboost`).
- **`corp_ca_bundle.pem`** y `SCRAPER/Scrapling/` (clon local) están gitignored:
  Scrapling se instala desde PyPI vía `requirements.txt`; el clon no es necesario.
- **Coste del Tier 2**: el navegador es lento (5-11 s/solve), con tope
  `HLTV_STEALTH_MAX_SOLVES_PER_RUN=6`. Es fallback; el camino normal es Tier 1.
- Reentrenar tarda según el histórico; el walk-forward es la parte lenta.

## 5. Fuentes (resumen)
- Scrapling docs (StealthyFetcher / Fetchers) — scrapling.readthedocs.io
- curl_cffi (impersonation TLS/JA3, HTTP/2/3) — github.com/lexiforest/curl_cffi
- Cloudflare `cf-mitigated` / clearance cookie — developers.cloudflare.com
- CS:GO rating systems (per-player TrueSkill) — arxiv.org/abs/2410.02831
- Whole-History Rating — remi-coulom.fr/WHR
- Map-veto bandit (Bo3 composicional) — arxiv.org/abs/2106.08888
- Beta calibration (Kull & Flach 2017) — proceedings.mlr.press/v54/kull17a
- Dixon–Coles time-weighting (half-life) — dashee87.github.io
- Rethinking evaluation metrics (log loss/ECE) — arxiv.org/abs/2309.06248

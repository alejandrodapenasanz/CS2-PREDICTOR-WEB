# CS2 Predictor

Sistema de **predicción pre-partido** de Counter-Strike 2 a partir de datos de
HLTV, orientado a **probabilidades bien calibradas** (no solo "quién gana"):
cuando el modelo dice 70%, el favorito debe ganar ~70% de las veces. Proyecto
académico, sin fines de apuesta real.

- **Especificación de diseño (teoría y decisiones):** [`PROJECT.md`](PROJECT.md)
- **Registro de cambios y decisiones:** [`DOCS/CHANGELOG.md`](DOCS/CHANGELOG.md)
- **Features pendientes / futuras:** resumidas en `PROJECT.md`; detalle extendido en [`DOCS/extra_features.md`](DOCS/extra_features.md)

---

## 1. Qué hace, en una frase

Dado un partido futuro, reconstruye el estado de fuerza de ambos equipos **tal
como era justo antes del partido** (rating Glicko-2, forma, head-to-head…) y
produce una probabilidad calibrada de victoria, que la web muestra con su nivel
de fiabilidad y, donde hay, el contraste con el mercado de apuestas.

## 2. Arquitectura y flujo de datos

```
   HLTV (scraping online)                  [SCRAPER/  — cf_session + backoff + caché]
        │  results_all.json (10k series, era CS2)  +  snapshots diarios con odds
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ MODEL/cs2model   Glicko-2 cronológico + features POINT-IN-TIME         │
  │                  (mismo motor en backtest y en vivo → cero fuga)       │
  └──────────────────────────────────────────────────────────────────────┘
        │                          │                          │
        ▼                          ▼                          ▼
  MODEL/train.py            BBDD/build_db.py          PIPELINE/
  → artifacts/registry/     → cs2.db (3 capas)        enrich_predictions.py
    (versiones, métricas,     staging/core/mart        (carga la copia runtime
     SHAP y punteros)         + ratings point-in-time   artifacts/model.pkl +
                                                        odds, roster, fatiga…)
        │                                                       │
        └──────────────────────────► ../WEB/build_web.py ◄───────┘
                                      → ../WEB/data.js → ../WEB/index.html
                                         (dashboard cyberpunk:
                                          Partidos · Best Opportunity · Modelo)
```

La **fuente de verdad** es la base de datos / los hechos inmutables; las
features y los ratings son **vistas calculadas** reconstruibles a cualquier
instante del pasado. Por eso el "cero fuga temporal" está garantizado por el
diseño, no por vigilancia manual.

Desde 2026-07-07 `BBDD/cs2.db` funciona como BBDD viva: `BBDD/build_db.py`
solo inicializa/migra y siembra si la base está vacía; cada `start.ps1` aplica
`BBDD/ingest.py` sobre el run nuevo y luego `BBDD/export_master_json.py`
regenera el JSON de compatibilidad. El entrenamiento (`MODEL/train.py`) lee de
SQLite por defecto; `--raw` queda para debug/legacy.

## 3. Carpetas

Esta documentación asume que la terminal está situada en `CS2/`:

```powershell
cd CS2
```

| Carpeta | Contenido | Doc |
|---|---|---|
| [`MODEL/`](MODEL/) | Núcleo de ML: Glicko-2, features point-in-time, entrenamiento, calibración, walk-forward, SHAP, artefacto. | Documentado aquí y en `PROJECT.md` |
| [`BBDD/`](BBDD/) | Esquema SQLite + `build_db.py` (init/semilla), `ingest.py` (upsert incremental), `export_master_json.py`. La base como fuente de verdad. | Documentado aquí y en `PROJECT.md` |
| [`PIPELINE/`](PIPELINE/) | Pipeline diario: snapshot pre-partido, odds por casa, predicción del modelo, flags de fiabilidad, calibración real rolling. | Documentado aquí y en `PROJECT.md` |
| [`../WEB/`](../WEB/) | Dashboard compartido entre deportes. La build actual publica CS2. | — |
| [`SCRAPER/`](SCRAPER/) | Scraper online de HLTV con sesión Cloudflare, backoff, caché y entorno bloqueado fail-closed. | Documentado aquí y en `PROJECT.md` |
| [`TESTS/`](TESTS/) | Tests transversales, simulador de cartera, gráficos y cachés locales de tooling. | `python -m pytest` |
| [`DOCS/`](DOCS/) | Changelog, runbooks y catálogo de features futuras. | Documentación auxiliar |

## 4. El modelo

### 4.1. Por qué Glicko-2

Sobre Elo plano: modela la **incertidumbre** del rating (RD) y la volatilidad.
Distingue "1500 con 200 partidos" de "1500 con 3"; la RD crece con la
inactividad (red flag por datos viejos). En la evaluación, el baseline Glicko-2
supera al de Elo (AUC 0.649 vs 0.642).

### 4.2. Features

El modelo estable usa un núcleo point-in-time reconstruible: Glicko-2/RD, Elo,
forma global y con decay, forma específica por formato (BO1/BO3/BO5),
actividad/recencia, fuerza de rivales recientes, head-to-head y señales de
score. Las fuentes de cobertura parcial no se mezclan silenciosamente con ese
núcleo.

La dureza del calendario no se reduce al último rival: `strength_of_schedule`
resume el Elo de los oponentes recientes con decay, la variabilidad de esa
oposición y el rendimiento real menos el esperado por Elo. Se calcula antes de
cada partido y entra automáticamente al llegar a 800 filas válidas.

Hay bloques opcionales preparados pero protegidos por muestra mínima:

- **HLTV Analytics:** map pool, pick/ban y señales de la pestaña Analytics. Se
  activa automáticamente cuando haya suficientes partidos cerrados con Analytics
  point-in-time.
- **Alineación anunciada y metadatos del evento:** los cinco anunciados se
  guardan separados de la alineación real y se usan solo si la foto precede al
  inicio. Sus diferencias de Rating/KPR/KAST/ADR, profundidad, Round Swing y
  stand-ins se activan automáticamente con 200 casos cerrados; prize pool y
  número de equipos, con 300. No hay switch manual.
- **Contexto de torneo:** LAN/online, fase (`opening`, `group`, `swiss`,
  `quarter`, `semi`, `final`, playoffs/bracket), winner advances y elimination.
  Se guarda ya en BBDD y se entrena solo cuando haya muestra suficiente y no
  desbalanceada entre LAN/online.
- **Stats individuales de jugador:** se capturan como snapshot pre-partido
  adaptativo (`past3months` si hay muestra, si no `past6months`, si no
  `past12months`). Entran automaticamente al llegar a 200 partidos cerrados con
  cobertura point-in-time.
- **Comparacion de roster:** los cinco jugadores de un equipo se comparan en una
  misma ventana temporal representativa (3, 6 o 12 meses). El vector conserva
  media, mejor jugador, media de los dos mejores, mediana, media de los dos
  peores y dispersion; evita derivadas redundantes como `spread`, `star_gap` y
  `weak_link_gap`. No presupone que una estrella pueda siempre ganar sola: ese
  efecto se aprende y se valida en walk-forward cuando haya 200 partidos con
  snapshots pre-partido completos (minimo cuatro jugadores y cinco mapas por
  jugador en ambos equipos).
- **Cambio de alineacion 90d:** para cada partido futuro se compara la
  alineacion anunciada 5v5 con la alineacion real 5v5 del ultimo partido previo
  del equipo dentro de 90 dias. Un cambio confirmado aparece como
  `ROSTER_CHANGE_90D` rojo con altas, bajas y partido de referencia. Si falta
  cualquiera de las dos alineaciones completas, el estado se muestra como no
  verificable y no se genera una red flag.
- **Rating sensible al roster (challenger):** usa esa misma evidencia causal para
  descontar de forma proporcional el crédito Elo/Glicko cuando cambian al menos
  2 de 5 jugadores y aumenta la RD. No sustituye al rating actual: ambos se miden
  sobre el mismo hold-out temporal y `roster_glicko_cal` solo puede promocionar
  pasando la selección walk-forward y la puerta champion/challenger. Sin 5v5
  completo coincide con el rating normal y la predicción sigue disponible.
- **Box score/mapas, historial de evento, rankings y roster:** cada familia tiene
  su columna de disponibilidad y umbral propio. `MODEL/train.py` decide `ON/OFF`
  en cada reentrenamiento; no hay switches manuales.

- **Ratings alternativos:** MOV y TrueSkill de equipo se activan con 800 casos;
  TrueSkill por jugador, con 200. Bradley-Terry bayesiano y Kalman se calculan
  causalmente y se evalúan como miembros independientes del super-learner. No se
  añaden a los learners genéricos: el ensemble aprende un peso no negativo y puede
  descartarlos asignándoles cero.
- **BO3 composicional:** estima los picks y el decider desde Analytics capturado
  pre-match, contrae el winrate de mapas con poca muestra y combina tres
  probabilidades. Se activa con 500 series elegibles y al menos 10 mapas por lado;
  el veto real obtenido después del partido nunca entra como predictor.

El entrenamiento también audita la calibración por tramos absolutos de Elo,
LAN/online, fase, formato y tier de evento. Los segmentos con menos de 100
predicciones walk-forward se marcan como no concluyentes; el ledger live se
monitoriza por separado y su muestra actual no decide. La vía opcional de
interacciones y la escalera manual están documentadas en
[`DOCS/SEGMENT_CALIBRATION.md`](DOCS/SEGMENT_CALIBRATION.md). La poda automática es deliberadamente conservadora:
solo retira columnas constantes o exactamente redundantes sin usar el resultado;
VIF, permutation importance cronológica, RFE y SHAP quedan en un informe para no
introducir selección supervisada fuera de una validación temporal anidada.

Para BO3 se entrena un target auxiliar `0-2/1-2/2-1/2-0`. Se activa sin switch
manual cuando hay 2.000 series, al menos 300 por clase y mejora el baseline
empírico en walk-forward. El predictor principal de ganador permanece separado;
el sidecar añade `series_score_distribution` y `predicted_series_score`. El
modelo por mapa todavía espera una muestra point-in-time suficiente.

### 4.3. Modelos, calibración y selección

Walk-forward semanal (ventana expansiva, nunca k-fold aleatorio). Se evalúan
baselines (base-rate, Elo, Glicko-2), logística, LightGBM, CatBoost y ensembles
calibrados con Platt/isotónica/beta; por **menor log loss** se elige el
challenger interno, que aún debe superar la puerta contra el vivo.

### 4.4. Resultados vigentes (27-07-2026, 9.726 series, 7.290 OOS)

| Modelo | Accuracy | Log loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Base rate | 0.5636 | 0.6851 | 0.2460 | 0.4881 | 0.0020 |
| Elo | 0.6008 | 0.6578 | 0.2329 | 0.6399 | 0.0429 |
| Glicko-2 | 0.6078 | 0.6598 | 0.2332 | 0.6479 | 0.0495 |
| Logística calibrada | 0.6413 | 0.6333 | 0.2214 | 0.6832 | 0.0092 |
| LightGBM calibrado | 0.6390 | 0.6354 | 0.2225 | 0.6788 | 0.0114 |
| Random Forest calibrado | 0.6387 | 0.6335 | 0.2216 | 0.6834 | 0.0138 |
| Super Learner fijo (diagnóstico) | 0.6451 | 0.6304 | 0.2202 | 0.6885 | 0.0128 |
| **Política causal de selección (primaria)** | **0.6412** | **0.6318** | **0.2209** | **0.6858** | **0.0099** |

> La métrica primaria es `nested_model_policy`: cada semana decide con OOS
> estrictamente anterior. `super_learner_cal` es el candidato ajustado sobre todo
> el histórico para el siguiente periodo; su fila retrospectiva es diagnóstica,
> no una estimación sin sesgo. La certificación independiente del zoo quedó en
> log loss `0.6338`, por lo que no sustituye al modelo productivo actual.

### 4.5. Accuracy por probabilidad predicha

Desglose histórico walk-forward del favorito del Modelo A sin odds. La
probabilidad de esta tabla es `max(p_team1, 1-p_team1)`; no representa todavía
el ranking de arquitecturas con odds incorporadas al artefacto:

| Banda predicha | Partidos | Aciertos | Accuracy observada | Prob. media predicha |
|---|---:|---:|---:|---:|
| 50-60% | 2.971 | 1.644 | 55,33% | 54,86% |
| 60-70% | 2.358 | 1.524 | 64,63% | 64,82% |
| 70-80% | 1.522 | 1.133 | 74,44% | 74,38% |
| 80-90% | 432 | 366 | 84,72% | 83,13% |
| 90-100% | 7 | 7 | 100,00% | 90,73% |
| **Total** | **7.290** | **4.674** | **64,12%** | — |

La franja 90-100% no es interpretable todavia (`n=7`). Las demas franjas son
monotonas y las tres primeras estan bien calibradas; el mayor margen de mejora
sigue en los partidos cercanos al 50%.

### 4.6. Benchmark de mercado (odds)

Con odds de apertura guardadas (hoy `n=237`, todavía ilustrativo), el **mercado**
queda en log loss `0.6107`; el benchmark solo se compara en ese subconjunto, no
contra las 7.290 predicciones completas del modelo. Confirma que las odds son muy
informativas (PROJECT.md §6.5). Producción compara ahora un router de dos modelos
contra un LightGBM mixto con `NaN` nativo, siempre sobre el mismo hold-out
temporal; la puerta champion/challenger decide si la arquitectura ganadora puede
reemplazar al vivo. Contrato completo en `DOCS/ODDS_ARCHITECTURES.md`.

## 5. La web (dashboard)

Estática (HTML+JS sobre `data.js`), **sin servidor**, estilo cyberpunk
(negro/rojo neón). Tres vistas:

La build compartida obtiene la etiqueta del modelo CS2 desde el puntero vivo
`MODEL/artifacts/registry/latest.json` y su `metadata.json`; no deserializa el
`model.pkl` desde WEB ni desde el entorno de TENNIS. Así, un desajuste de
dependencias de presentación no puede ocultar en el dashboard un modelo vivo.

- **Partidos** — cartelera con predicción del modelo, mercado, **fiabilidad** y
  señales; panel de detalle con ratings Glicko, mercado, pool de mapas, roster,
  forma, fatiga, contexto y rosters.
- **★ Best Opportunity** — ranking por **confianza del modelo × fiabilidad de
  los datos** (penaliza RD alta, poca historia, BO1, rival por definir). Muestra
  dónde el modelo está seguro **y** respaldado por datos. La política por
  defecto conserva elegibilidad desde confianza `>=60%` y fiabilidad `>=45%`;
  la pestaña web aplica además un mínimo visible de `>=65%`. Cada run congela en
  SQLite el score, la elegibilidad y los rankings global/diario;
  `is_best_opportunity=1` identifica el número 1 de su día. Para cada partido,
  la alineación anunciada 5v5 tiene prioridad sobre el perfil general del equipo;
  la cobertura de player stats se divide entre esos cinco y reduce fiabilidad si
  faltan snapshots. La política `best_opportunity_v2_confirmed_lineups` exige
  además dos alineaciones anunciadas de cinco IDs únicos que coincidan exactamente
  con los rosters mostrados. Un lineup incompleto puede seguir apareciendo en
  Partidos con el flag `PREMATCH_LINEUP_INCOMPLETE`, pero nunca en Best Opportunity.
  Cada enrich escribe `roster_integrity_report.json` y aborta antes de publicar si
  detecta un roster completo incoherente.
  También muestra la banda heurística sobre la probabilidad pura del modelo
  (`±` puntos porcentuales y nivel alto/medio/bajo). Mide incertidumbre sobre
  el estimado `p`, no la varianza del resultado, y no altera por sí sola el
  ranking ni se conecta al bankroll. La penalización de staking heredada sigue
  aislada en `model_epistemic_std`; contrato y límites en
  `DOCS/ESTIMATE_UNCERTAINTY.md`.
  Cada tarjeta muestra también la tasa histórica de derrotas del equipo cuando
  una predicción oficial congelada en `prediction_ledger` lo marcó favorito con
  probabilidad `>=51%`. Usa todo el historial causal anterior disponible y no
  exige que el partido tuviera odds; filas sin predicción prospectiva no se
  reconstruyen retrospectivamente.
- **Modelo** — métricas walk-forward, importancia SHAP, curva de calibración y
  evolución temporal de la accuracy real. El selector ofrece últimos 7 días,
  1 mes, 3 meses, 6 meses y 1 año; usa agregación diaria, semanal o mensual
  según el horizonte. Cada punto se calcula con predicciones fuera de muestra y
  ganadores reales, y se regenera automáticamente después de cada entrenamiento.

## 6. Cómo ejecutarlo

El componente de modelo exige **CPython 3.13 exacto** y vive en `CS2/.venv`.
`start.ps1` crea o repara ese entorno sin modificar el Python global, instala
exclusivamente el lock con hashes y valida tanto los pins como `pip check`.
Las reparaciones se preparan primero en `.venv.build`; solo tras validarlas se
intercambian por el entorno activo, conservando el anterior en `.venv.previous`
para restaurarlo automáticamente si el swap falla.
Para aprovisionarlo manualmente desde `CS2/`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-deps --only-binary=:all: `
  --require-hashes -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` declara solo dependencias del modelo, enriquecimiento y
tooling. El scraper mantiene su manifiesto, lock, wheels y venv dentro de
`SCRAPER/hltv-scraper-api/`. `requirements.lock.txt` es el resultado completo
para CPython 3.13 que usan producción y CI. Se regenera deliberadamente con
hashes (nunca durante `start.ps1`) mediante:

```powershell
py -3.13 -m pip install pip==25.3 pip-tools==7.5.2
py -3.13 -m piptools compile --resolver=backtracking --generate-hashes `
  --allow-unsafe --strip-extras --no-emit-index-url `
  --pip-args="--only-binary=:all:" `
  --output-file=requirements.lock.txt requirements.txt
```

Ese pin de pip es exclusivo de la herramienta de generación: pip-tools 7.5.2
no funciona con las APIs internas de pip 26. Es preferible usar un entorno
temporal separado para estas herramientas, no instalarlas en el venv de producción.

El constraint de `numba` se mantiene en `0.65.1`: su wheel CPython 3.13 pasa
import, JIT y SHAP `TreeExplainer` bajo Windows Application Control. La versión
`0.66.0` fue rechazada por esa política; cualquier subida debe superar primero
el smoke binario que ejecutan los `Ensure-*Python`.

`scikit-learn==1.8.0` queda fijado en el manifiesto y en el lock: es la versión
con la que se serializaron el vivo y `last_good`. Un cambio de versión requiere
validar los artefactos por la puerta, no solo que `pip check` pase. El
[contrato de persistencia de scikit-learn](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations)
no garantiza cargar modelos entre versiones. No se ocultan esos avisos.

Si una carpeta residual del registro deniega acceso incluso a `stat/lstat`,
`check_retrain.py` omite únicamente esa entrada y publica `registry_warnings`
en su JSON. `start.ps1` muestra esos avisos en consola/log y continúa usando
los cortes de los intentos legibles y del vivo. No toma propiedad, no borra el
residuo, no activa modelos y no modifica los punteros ni el ledger.

El aprovisionamiento del scraper tambien exige CPython 3.13 exacto: instala solo
`SCRAPER/hltv-scraper-api/requirements.lock.txt` con hashes y usa
`SCRAPER/hltv-scraper-api/wheels/` para artefactos locales verificados. No existe
un fallback que instale paquetes sueltos; cualquier divergencia falla antes del
scraping.

SQLite va incluido en Python.

**Scraper por tiers (anti-bloqueo Cloudflare).** El scraper usa
[Scrapling](https://scrapling.readthedocs.io): Tier 1 HTTP con impersonation TLS/JA3
(`curl_cffi`) y Tier 2 navegador stealth que resuelve el challenge, con
`requests`/`cloudscraper` como transporte HTTP incluido en el lock. `start.ps1`
exige **CPython 3.13 exacto**, instala `scrapling[fetchers]` desde el lock y
aprovisiona sus navegadores. Si el entorno no se puede validar, falla de forma
explícita: no degrada a otra versión de Python ni instala dependencias sueltas.
En redes con inspección TLS (proxy corporativo),
`start.ps1` genera un CA bundle automáticamente. Todos los cambios de esta iteración
están documentados en [`DOCS/runbooks/LAST_CHANGE_2026-07-06.md`](DOCS/runbooks/LAST_CHANGE_2026-07-06.md).

**Una sola orden hace TODO** (scrape → update → modelo → enrich → BBDD → web):

```powershell
.\start.ps1                  # pipeline completa (con acceso a HLTV)
.\start.ps1 -MaxMatches 3 -SkipPlayerStats -SkipTeamProfiles  # prueba online rapida; no publica master
.\start.ps1 -RecoveryWindowDays 3  # reintenta huecos recientes de odds/detalle/Analytics
.\start.ps1 -SkipScrape      # modo debug/offline explícito: usa el último run
start ..\WEB\index.html      # abrir el dashboard compartido
```

Si el scraping falla o devuelve cero datos, `start.ps1` falla por defecto: la
pipeline principal es online y no acepta silenciosamente un run viejo. Para un
debug puntual se puede pasar `-AllowOfflineFallback`. Si existe un modelo vivo,
su `metadata.date_max` define `live_cutoff`. Para programar el siguiente intento,
el pipeline usa
`attempt_cutoff = max(live_cutoff, último intento terminal válido registrado)`
y cuenta únicamente etiquetas estrictamente posteriores a `attempt_cutoff`; al
acumular **100** lanza el reentreno automático. Así un rechazo o aplazamiento no
repite el mismo entrenamiento en cada run. `-Retrain` fuerza el intento manual,
pero ambos caminos atraviesan exactamente la misma puerta de promoción, cuyo
hold-out sigue empezando después de `live_cutoff`; entrenar no implica sustituir
producción. `-RollbackModel` restaura el
`last_good` validado. Flags: `-SkipScrape`, `-AllowOfflineFallback`, `-Retrain`,
`-RollbackModel`, `-NoDb`, `-MaxMatches N`,
`-SkipPlayerStats`, `-SkipTeamProfiles`, `-SkipSameDayRecovery`,
`-RecoveryWindowDays N`, `-RecoveryDelay S`, `-RecreateScraperVenv`.

Cada ejecución de `start.ps1` crea un transcript completo en `PIPELINE/logs/start_*.log`
y muestra cada comando con hora, exit code y duración. Al cerrarse, rota como
una unidad el transcript, JSONL, tiempos y decisiones, y conserva solo la
ejecución actual y la anterior. Si Cloudflare bloquea el
navegador stealth durante demasiado tiempo, el pipeline lanza automáticamente
`grab_cf.py` para abrir una ventana visible y renovar `cf_clearance`.
Con `-Retrain`, el trainer también recibe `--verbose` y muestra folds, estudios
Optuna y pesos; `-Quiet` conserva la salida resumida.

El scrape diario consulta primero `BBDD/cs2.db`: `/results` solo se pide para
resolver IDs `pending_result` conocidos, y solo pagina a `offset=100/200` si los
pendientes no aparecieron en las páginas anteriores. Si SQLite no tiene
pendientes, esa fase se salta.

Las stats de jugador tambien son incrementales por ID. El pipeline reutiliza
snapshots completos dentro del TTL de 3 dias y genera un fichero temporal con
solo los jugadores nuevos, vencidos o incompletos. La ventana se elige de forma
adaptativa (`past3months` -> `past6months` -> `past12months`) hasta alcanzar 10
mapas. El parser de `/stats/players/compare` lee cada columna por separado y el
perfil individual confirma Rating, DPR, ADR y el numero de mapas. Si HLTV
devuelve cero mapas se registra `not_found` (sin muestra), no como bloqueo. Ante
el primer challenge de esta fase se guarda el progreso, se renueva
`cf_session.json` automaticamente y se reintenta sin esperar varios backoffs
largos.

Los runs con `-MaxMatches` son solo pruebas parciales: scrapean HLTV online,
pero no actualizan `PIPELINE/master/manifest.json` ni
`PIPELINE/master/matches.json`. La BBDD y la web se siguen regenerando
desde el ultimo run completo publicado.

Pasos sueltos (si los necesitas):

```powershell
.\.venv\Scripts\Activate.ps1             # comandos de modelo/BBDD/tooling
python MODEL\train.py                          # entrena desde BBDD\cs2.db → MODEL\results\REPORT.md
python MODEL\train.py --warmup-weeks 10 --min-train 800
python MODEL\train.py --raw <results_all.json> # modo legacy/debug si necesitas saltarte SQLite
python MODEL\run_professional_training.py --install-deps  # sweep CatBoost/half-life/gap + output .md
& .\SCRAPER\hltv-scraper-api\.venv\Scripts\python.exe PIPELINE\start.py  # scraper aislado
python PIPELINE\enrich_predictions.py   # puntúa el último run con el modelo
python BBDD\build_db.py                        # crea/migra/siembra BBDD\cs2.db solo si hace falta
python BBDD\ingest.py --run-dir <PIPELINE\runs\RUN_ID>  # upsert incremental del run
python BBDD\export_master_json.py              # export compat desde SQLite
python ..\WEB\build_web.py --sport-root .      # genera ..\WEB\data.js para CS2
python TESTS\tests_wallet_simulator.py --initial-wallet 100  # cartera + accuracy por franjas
python TESTS\tests_best_opportunity_simulator.py  # accuracy de Best Opportunity sin duplicados
python MODEL\analyze_walkforward_errors.py       # auditoria OOS de fallos, CSV y graficos
python MODEL\train.py --verbose                  # entrenamiento productivo: core + todos los algoritmos disponibles
python MODEL\train.py --optuna-trials 8 --verbose # Optuna purgado; ya son los valores automaticos por defecto
python MODEL\train.py --config MODEL\config.yaml --verbose # configuracion versionada; CLI tiene prioridad
python MODEL\monitor_drift.py                  # log loss rodante, Page-Hinkley y CLV
python MODEL\manage_models.py rollback         # restaura last_good sin reentrenar
python MODEL\audit_promotion_consistency.py <VERSION>  # audita una puerta historica sin mutar produccion
python MODEL\smoke_pipeline.py                 # smoke completo aislado, sin promover
python -m ruff check MODEL BBDD PIPELINE TESTS
python -m mypy
python MODEL\run_professional_training.py --algorithms all --feature-profile core --half-lives 45,60,90,120,180 --wf-gaps 0,1
```

Los tests respetan la misma frontera de entornos que producción. La suite de
modelo/BBDD usa `CS2/.venv`; los tests que importan `PIPELINE/start.py` usan el
venv del scraper, porque ese es su intérprete productivo. CI ejecuta exactamente
esta separación:

```powershell
& .\.venv\Scripts\python.exe -m pytest TESTS -q `
  --ignore=TESTS/test_hltv_parsers.py `
  --ignore=TESTS/test_bbdd_live_pipeline.py
& .\SCRAPER\hltv-scraper-api\.venv\Scripts\python.exe -m pytest -q `
  SCRAPER/hltv-scraper-api/tests/test_dependency_contract.py `
  TESTS/test_hltv_parsers.py TESTS/test_bbdd_live_pipeline.py
```

El simulador apuesta siempre al favorito puro del modelo. Las cuotas solo
permiten calcular beneficio, EV y stake; nunca hacen que el backtest elija el
equipo con probabilidad inferior al 50%. Genera los PNG, registros CSV y la
tabla `accuracy_by_confidence_band.csv` en `TESTS/graphs/`.

Para entrenar con la mejor calidad posible: ejecuta primero un `start.ps1`
completo para tener snapshots, odds, Analytics, contexto de torneo, rosters y
stats de jugador actualizados e ingeridos en SQLite; luego usa
`.\.venv\Scripts\python.exe MODEL\run_professional_training.py --install-deps`
para probar CatBoost, ensembles, half-life y gap con salida verbosa en
`MODEL\results\professional_training_output.md`. El entrenamiento usa validacion walk-forward temporal.
`nested_model_policy` elige cada semana solo con OOS de semanas anteriores y es
la metrica primaria sin sesgo L1. Si hay incumbente, la receta productiva se
congela al final de `live_cutoff` y se ajusta una única vez sobre ese prefijo. Ese
shadow fijo predice el sufijo completo sin aprender de él. La misma receta puede
refitearse sobre el histórico completo antes de materializar la decisión; ese
trabajo no alimenta al shadow. El resultado puede registrarse para auditoría,
pero **solo se publica si gana**. El commit de punteros decide si actualiza la copia runtime
`MODEL\artifacts\model.pkl`.
Analytics, contexto, stats de jugador, box score/mapas, historial de evento,
rankings y roster se calculan y guardan siempre que exista evidencia previa al
partido. Cada familia entra sola unicamente cuando supera el umbral de
`MODEL\config.yaml` y reduce el log loss en el holdout temporal interno del
fold. No hay switch manual. La decision y su evidencia quedan en
`MODEL\results\fold_local_feature_selection.json`, `MODEL\results\REPORT.md` y
`artifact.metadata["feature_policies"]`.

Además se generan `segment_calibration.json`, `feature_pruning.json`,
`rich_target_bo3.json` y `rich_target_bo3_predictions.json`. Estos archivos
permiten comprobar cobertura, calibración por contexto, multicolinealidad y si
el target de marcador superó realmente su baseline temporal.
`optuna_tuning.json` registra cada estudio interno, sus periodos purgados, el
valor de `C` y el log loss objetivo. Optuna se ejecuta automáticamente durante
`start.ps1 -Retrain`; `--optuna-trials 0` existe solo para smoke/debug.

La configuracion operativa vive en `MODEL/config.yaml`. Sus dataclasses
validadas centralizan semillas, defaults de entrenamiento, umbrales de
auto-activacion, hiperparametros principales, drift, promoción, reentreno
automático, retención y limites del backtest.
Cada entreno conserva `config.effective.yaml` y `experiment_manifest.json`
con SHA-256 de datos/config, commit Git, estado dirty, argumentos y versiones.

`start.ps1` ejecuta `MODEL/monitor_drift.py` despues del ingest final y actualiza
`MODEL/results/drift_live.json`. Solo evalua predicciones congeladas antes del
inicio que ya tienen resultado. Page-Hinkley vigila aumentos de log loss y
empeoramiento de CLV; el map pool se reconstruye con mapas anteriores y el
parche solo se marca si una fuente pre-match lo proporciona explicitamente.

Antes de entrenar, `start.ps1` ejecuta `BBDD/repair_integrity.py` y el health
gate de datos. El challenger se bloquea si hay participantes provisionales,
partidos online sin dos IDs HLTV confirmados, FK rotas, duplicados inequivocos,
cobertura reciente degradada, metricas no reproducibles desde el CSV o peor log
loss que Glicko. Superados esos controles, todavía debe batir al modelo vivo en
la puerta común de promoción. `prediction_ledger` congela una unica prediccion
trazable por partido y `MODEL/evaluate_live_ledger.py` evalua exactamente el hash
desplegado.

### 6.1. Promoción, rollback y retención

`MODEL/artifacts/model.pkl` es la **copia runtime** que carga el pipeline. La
fuente versionada está en `MODEL/artifacts/registry/<timestamp>/`:
`registry/latest.json` identifica lógicamente al vivo validado y
`registry/last_good.json` conserva al incumbente saliente de la última promoción.
Los punteros incluyen versión, ruta canónica y SHA-256; publicar o restaurar
`model.pkl` se hace mediante reemplazo atómico.

Antes de abrir el bloque de promoción, `recipe_mask` limita todos los ajustes a
filas con fecha `<= live_cutoff`. Con ese prefijo se congelan las features, la
familia `best_name`, el ajuste Optuna purgado y los pesos; después se ajusta **una
sola vez** el shadow. El mismo objeto predice todo el sufijo `> live_cutoff` sin
refits, actualizaciones ni acceso a sus etiquetas. Solo después se revelan las
etiquetas para comparar shadow e incumbente sobre exactamente las mismas filas
point-in-time e IDs.

Se requieren al menos **100** partidos comunes. La métrica primaria es log loss:
una mejora mínima de `0.001` promociona y un empeoramiento de esa magnitud
rechaza. Dentro de esa banda de empate solo promociona una mejora Brier mínima de
`0.0005`. Una muestra insuficiente aplaza la decisión. Rechazo o aplazamiento
dejan sin cambios `latest.json`, `last_good.json` y `model.pkl`, y el informe
registra `n`, cutoff, ambas métricas, deltas y razón.

`promotion_decision.json` sella además la evidencia: `holdout_sha256` cubre la
identidad emparejada (ID, fecha y etiqueta), y `prediction_sha256` cubre esos
mismos campos junto con las probabilidades de incumbente y shadow. Esta segunda
huella queda vacía cuando la muestra no alcanza siquiera la evaluación.

Se prepara un **refit** de la misma receta congelada sobre todo el histórico
disponible; puede calcularse antes de cerrar la comparación porque no interviene
en ella, pero solo se publica si el shadow gana. La evidencia probabilística de
la puerta valida el shadow fijo y su receta as-of; no es una medición independiente
de los bytes exactos del pickle full-history. El health gate final abre
directamente `registry/<version>/model.pkl`. Su `candidate_reference` verifica el
SHA-256 de esos bytes y ese mismo valor es el
`expected_candidate_sha256` obligatorio del CAS.

El bundle núcleo (`model.pkl`, metadata, SHAP, manifest y config) se construye en
un directorio de staging oculto y se mueve una sola vez a
`registry/<timestamp>/`; sus archivos no se reescriben. Solo se permiten los
sidecars auditables aditivos sancionados, como `promotion_decision.json` y
`deployment.json`, que no alteran esos archivos ni el SHA del modelo. La metadata
del núcleo es deliberadamente pre-commit: puede indicar
`promotion_approved` y `deployment_state=pending_pointer_commit`, pero eso **no**
demuestra que esté publicada. `latest.json` junto con el hash runtime y el recibo
`deployment.json` confirman el despliegue; los punteros son la autoridad si el
recibo no pudo escribirse. Cuando existe, el recibo replica ambas huellas como
`decision_holdout_sha256` y `decision_prediction_sha256`.

La publicación toma `.deployment.lock` y aplica compare-and-swap (CAS): vuelve a
comprobar la versión y el hash esperados del incumbente, además del hash del
challenger. Esos parámetros CAS son obligatorios (`expected_incumbent` y
`expected_candidate_sha256`; bootstrap exige el SHA esperado del candidato) y no
existe una vía opcional que los omita. Si algo cambió desde la evaluación, falla
cerrada sin mover punteros.

Rollback operativo:

```powershell
.\start.ps1 -RollbackModel
python MODEL\manage_models.py rollback
```

El rollback restaura `last_good` como `latest`/`model.pkl` y rota el vivo saliente
a `last_good`, por lo que la operación puede deshacerse de nuevo sin reentrenar.

Si una migración antigua dejó ambos punteros en la misma versión, la única vía
sancionada para fijar un rollback distinto es:

```powershell
python MODEL\manage_models.py set-last-good --version <YYYYMMDD_HHMMSSZ> --expected-sha256 <SHA256>
```

El comando carga y puntúa el artefacto, exige semilla 42 y el SHA inspeccionado,
y bajo `.deployment.lock` aplica CAS sobre `latest`, `last_good` y el hash runtime.
Solo sustituye atómicamente `last_good.json`; `latest.json` y `model.pkl` quedan
byte a byte intactos. No uses este comando para saltarte la puerta de promoción:
el operador debe elegir una versión previamente validada.

La retención configurada conserva **0 versiones adicionales** del registry:
únicamente `latest` y `last_good`, que siempre están protegidos.
`PIPELINE/runs/` conserva los **2** últimos no referenciados, el run publicado por
`master/manifest.json`, cualquier run todavía referenciado y todo directorio de
nombre desconocido/no canónico. La primera poda está bloqueada: primero se genera
y revisa una keep-list/delete-list con token ligado al estado del filesystem y
solo una confirmación manual explícita habilita esa política. Si el plan cambia,
el token caduca y se exige un preview nuevo.

Desde `CS2/`, el preview y su confirmación explícita son:

```powershell
python MODEL\manage_models.py prune --scope all
python MODEL\manage_models.py prune --scope all --confirm "registry=<TOKEN_REGISTRY>,runs=<TOKEN_RUNS>"
```

El primer comando no borra nada y devuelve los tokens exactos que deben sustituir
los placeholders del segundo. Después de aprobar esa política, `start.ps1` puede
aplicar automáticamente planes compatibles; un cambio de política o de N exige
otra revisión manual.

La aplicación mueve primero todos los candidatos a
`.retention-quarantine/<TOKEN>/` mediante renames atómicos. Si falla el staging,
restaura todos los nombres; si falla el borrado, deja manifest y pendientes fuera
de los nombres canónicos. Un bloqueo de ACL/handle/antivirus durante el preflight,
staging o borrado es **no fatal para la pipeline**: queda registrado de forma
persistente en `MODEL/artifacts/pending_cleanup.json`, el resumen y el dashboard
muestran la advertencia, y `latest`/`last_good` siguen protegidos. Tras corregir la
causa, la vía sancionada de reintento para una quarantine ya creada es
`python MODEL\manage_models.py prune --scope registry --resume <TOKEN>`.
Un candidato bloqueado antes del staging queda protegido con la razón
`cleanup_pending`; así las podas posteriores pueden retirar el resto de
versiones vencidas sin volver a tocar el residuo ni ocultar la advertencia.
Si el corte ocurrió antes del primer `rmtree`, ese comando valida la identidad de
cada bundle y revierte el staging completo; si ocurrió después, continúa el
borrado solo dentro de quarantine. Cualquier quarantine pendiente bloquea un
preview confirmable; la poda automática avisa y continúa sin intentar una nueva
mutación hasta completar su recuperación. Los errores de seguridad estructural
(punteros inválidos, symlinks/reparse points, token obsoleto o health inconsistente)
siguen siendo fatales y nunca se degradan a una simple advertencia.
En Windows, los atributos `ReadOnly` se limpian solo en la copia ya situada en
quarantine; los bundles canónicos no se modifican durante el preflight.

Las paginas con participantes provisionales se archivan igualmente en
`raw_snapshots`, de modo que el siguiente `start.ps1` puede recuperarlas sin
perder evidencia. No se normalizan ni se predicen hasta que ambos equipos
tengan nombre real e ID HLTV. Odds de apertura, player/ranking snapshots,
Analytics, contexto y alineacion pre-match se conservan por observacion con
`captured_at_utc`; el resultado nunca sobrescribe esas fotos.

La reparacion de integridad tambien puede ejecutarse de forma explicita:

```powershell
python BBDD\repair_integrity.py
```

Antes de modificar la BBDD crea un backup, recupera IDs historicos desde la
evidencia raw, corrige referencias, purga placeholders sin valor normalizado y
consolida solo duplicados de identidad inequivoca. Es idempotente: repetirla
sobre una base sana no cambia filas. Encuentros parecidos con IDs HLTV
distintos se conservan como partidos reales separados.

El backtest de `MODEL/results/economic_backtest.json` parte de 1.000 EUR,
desglosa vig, Kelly fraccional, limites de stake/payout, drawdown y CLV contra
la ultima cuota valida antes del inicio. Las cuotas de cierre son auditoria:
modificarlas no puede cambiar el lado, EV ni stake decidido en apertura.

Cada entrenamiento genera tambien `MODEL\results\favorite_accuracy_bands.json`
con el acierto walk-forward del favorito en las franjas 50-60, 60-70, 70-80,
80-90 y 90-100%. `start.ps1 -Retrain` reconstruye despues `..\WEB\data.js`; si
la puerta rechaza o aplaza el challenger, la web sigue asociada al hash del vivo
y no presenta el reentreno descartado como producción.

Rutina manual cuando quieras evaluar un challenger sin esperar a las 100
etiquetas posteriores a `attempt_cutoff` (la promoción puede rechazarse o
aplazarse):

```powershell
.\start.ps1 -Retrain
```

En la operativa normal basta `.\start.ps1`: al llegar a 100 partidos etiquetados
estrictamente posteriores a `attempt_cutoff` activa el mismo reentreno y la misma
puerta automáticamente.

Rutina profesional profunda (cuando quieras repetir el sweep CatBoost/half-life/gap):

```powershell
.\start.ps1
python MODEL\run_professional_training.py --install-deps
python PIPELINE\enrich_predictions.py
python BBDD\ingest.py
python BBDD\export_master_json.py
python ..\WEB\build_web.py --sport-root .
```

Cada etapa de la pipeline está documentada en la cabecera de `start.ps1`. Las
features contempladas pero aún no activas están resumidas en `PROJECT.md` y
detalladas en `DOCS\extra_features.md`.

## 7. Garantías y ética

- **Cero fuga temporal** por diseño (features point-in-time, validación
  cronológica). Síntoma de fuga que se evita: 80% en validación, 55% en real.
- **Calibración** validada con curvas, no solo accuracy.
- Proyecto **académico, sin apuestas reales**. Integridad competitiva solo con
  sanciones oficiales (PROJECT.md §6.7). Scraping de bajo volumen, respetuoso con
  los ToS de HLTV (caché para no re-descargar, backoff, ritmo humano).

## 8. Estado y siguientes pasos

Funciona end-to-end en modo online. A 2026-07-06 el run publicado es
`2026-07-06_065318Z` (17 partidos publicables tras filtrar empezados/pasados). La web muestra stake recomendado por partido
con Kelly fraccional ajustado por fiabilidad, consenso de mercado y calibracion.
La BBDD fisica `BBDD/cs2.db` guarda odds, predicciones, snapshots crudos,
assets normalizados, contexto, rankings y stats agregadas/por mapa cuando
existen. Cada ingest crea backup en `BBDD/backups/`. Para sacar una copia SQL
legible:

```powershell
sqlite3 BBDD\cs2.db ".backup 'BBDD\backups\cs2_manual_backup.db'"
sqlite3 BBDD\cs2.db ".dump" > BBDD\cs2_dump.sql
python -c "import sqlite3; c=sqlite3.connect('BBDD/cs2.db'); b=sqlite3.connect('BBDD/backups/cs2_manual_backup.db'); c.backup(b); b.close(); c.close()"
```

El mayor margen de mejora (banda de partidos parejos) sigue estando en box score
por mapa, stats de jugador point-in-time, veto real y odds historicas. Plan
detallado en `DOCS\extra_features.md`.

El benchmark Model B (stats + odds de apertura) se conserva. Además, el entreno
compara automáticamente las arquitecturas productivas router y mixta cuando hay
al menos `feature_thresholds.opening_odds` filas causales (120 por defecto), con
métricas separadas con/sin odds y promoción por la puerta común.

Estado de cobertura medido el 2026-07-10 tras deduplicar semilla+HLTV: evento
6017/200 (ON); box score/mapas 20/200, Analytics 100/120, jugadores 88/200,
rankings 86/200 y roster 61/200 (OFF). Contexto tiene 65/200, con 7 LAN y 58 online, por lo que sigue OFF
hasta alcanzar tambien 50 LAN. Estos estados se recalculan automaticamente en
cada entrenamiento.

### Pistols y conversiones (candidatas)

Las capturas archivadas de rondas por mapa se incorporan a tablas laterales con
disponibilidad original verificable. El experimento `MODEL/run_pistol_ablation.py`
compara control, pistols y conversiones sin activar nuevas variables por defecto.
Ver [contrato, comandos y límites causales](DOCS/PISTOL_ROUNDS.md).

La opción `--opponent-adjusted` contrasta el rendimiento de pistols por encima de
lo esperado según el Elo del rival. Mantiene la puerta de promoción y exige
además muestra nueva tras el periodo ya examinado. Ver
[fórmula, prueba temporal y ejecución](DOCS/PISTOL_OPPONENTS.md).

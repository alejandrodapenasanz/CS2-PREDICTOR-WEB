# CS2 Predictor

Sistema de **predicción pre-partido** de Counter-Strike 2 a partir de datos de
HLTV, orientado a **probabilidades bien calibradas** (no solo "quién gana"):
cuando el modelo dice 70%, el favorito debe ganar ~70% de las veces. Proyecto
académico, sin fines de apuesta real.

- **Especificación de diseño (teoría y decisiones):** [`PROJECT.md`](PROJECT.md)
- **Registro de cambios y decisiones:** [`PROJECT_DOCS/CHANGELOG.md`](PROJECT_DOCS/CHANGELOG.md)
- **Features pendientes / futuras:** resumidas en `PROJECT.md`; detalle extendido en [`PROJECT_DOCS/extra_features.md`](PROJECT_DOCS/extra_features.md)

---

## 1. Qué hace, en una frase

Dado un partido futuro, reconstruye el estado de fuerza de ambos equipos **tal
como era justo antes del partido** (rating Glicko-2, forma, head-to-head…) y
produce una probabilidad calibrada de victoria, que la web muestra con su nivel
de fiabilidad y, donde hay, el contraste con el mercado de apuestas.

## 2. Arquitectura y flujo de datos

```
   HLTV (scraping online)                  [SCRAPPER/  — cf_session + backoff + caché]
        │  results_all.json (10k series, era CS2)  +  snapshots diarios con odds
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ MODEL/cs2model   Glicko-2 cronológico + features POINT-IN-TIME         │
  │                  (mismo motor en backtest y en vivo → cero fuga)       │
  └──────────────────────────────────────────────────────────────────────┘
        │                          │                          │
        ▼                          ▼                          ▼
  MODEL/train.py            BBDD/build_db.py          DAILY_SNAPSHOTS/
  → artifacts/model.pkl     → cs2.db (3 capas)        enrich_predictions.py
    (modelo calibrado         staging/core/mart        (carga el modelo, puntúa
     + métricas + SHAP)       + ratings point-in-time   los partidos del día +
                                                        odds, roster, fatiga…)
        │                                                       │
        └───────────────────────────► WEB/build_web.py ◄────────┘
                                       → WEB/data.js → WEB/index.html
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

| Carpeta | Contenido | Doc |
|---|---|---|
| [`MODEL/`](MODEL/) | Núcleo de ML: Glicko-2, features point-in-time, entrenamiento, calibración, walk-forward, SHAP, artefacto. | Documentado aquí y en `PROJECT.md` |
| [`BBDD/`](BBDD/) | Esquema SQLite + `build_db.py` (init/semilla), `ingest.py` (upsert incremental), `export_master_json.py`. La base como fuente de verdad. | Documentado aquí y en `PROJECT.md` |
| [`DAILY_SNAPSHOTS/`](DAILY_SNAPSHOTS/) | Pipeline diario: snapshot pre-partido, odds por casa, predicción del modelo, flags de fiabilidad, calibración real rolling. | Documentado aquí y en `PROJECT.md` |
| [`WEB/`](WEB/) | Dashboard estático cyberpunk (negro/rojo). 3 vistas. Sin servidor. | — |
| [`SCRAPPER/`](SCRAPPER/) | Scraper online de HLTV con sesión Cloudflare, backoff, caché y fallback HTTP directo para listados. | Documentado aquí y en `PROJECT.md` |

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

Hay bloques opcionales preparados pero protegidos por muestra mínima:

- **HLTV Analytics:** map pool, pick/ban y señales de la pestaña Analytics. Se
  activa automáticamente cuando haya suficientes partidos cerrados con Analytics
  point-in-time.
- **Contexto de torneo:** LAN/online, fase (`opening`, `group`, `swiss`,
  `quarter`, `semi`, `final`, playoffs/bracket), winner advances y elimination.
  Se guarda ya en BBDD y se entrena solo cuando haya muestra suficiente y no
  desbalanceada entre LAN/online.
- **Stats individuales de jugador:** se capturan como snapshot pre-partido
  adaptativo (`past3months` si hay muestra, si no `past6months`, si no
  `past12months`). Entran automaticamente al llegar a 200 partidos cerrados con
  cobertura point-in-time.
- **Box score/mapas, historial de evento, rankings y roster:** cada familia tiene
  su columna de disponibilidad y umbral propio. `MODEL/train.py` decide `ON/OFF`
  en cada reentrenamiento; no hay switches manuales.

### 4.3. Modelos, calibración y selección

Walk-forward semanal (ventana expansiva, nunca k-fold aleatorio). Se evalúan
baselines (base-rate, Elo, Glicko-2), logística, LightGBM, CatBoost y ensembles
calibrados con Platt/isotónica/beta; se elige el de producción por **menor log loss**.

### 4.4. Resultados (era CS2, 9.492 series unicas, 7.056 de test)

| Modelo | Accuracy | Log loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Base rate | 0.563 | 0.685 | 0.246 | 0.487 | 0.002 |
| Elo | 0.603 | 0.656 | 0.232 | 0.643 | 0.043 |
| Glicko-2 | 0.609 | 0.659 | 0.233 | 0.650 | 0.049 |
| Logística calibrada | 0.640 | 0.632 | 0.221 | 0.685 | 0.014 |
| LightGBM calibrado | 0.640 | 0.631 | 0.221 | 0.687 | 0.017 |
| CatBoost calibrado | 0.640 | 0.630 | 0.220 | 0.689 | 0.023 |
| Ensemble LGBM+Log (Platt) | 0.644 | 0.629 | 0.220 | 0.691 | 0.014 |
| Ensemble beta | 0.645 | 0.629 | 0.220 | 0.691 | 0.015 |
| **Ensemble + CatBoost (producción)** | **0.644** | **0.628** | **0.219** | **0.692** | **0.017** |

> Hallazgo honesto: con features de resultados (relaciones monótonas tipo
> "diferencia de rating"), la logística calibrada casi iguala al ensemble. Tras
> ejecutar el sweep profesional con CatBoost, la mejor configuración por log loss
> es `ensemble3_cal` con `--form-half-life 90 --wf-gap 0`. La elección sigue siendo
> data-driven: se prioriza log loss/calibración frente a maximizar accuracy bruta.

### 4.5. Accuracy por probabilidad predicha

Desglose walk-forward del favorito puro del modelo. La probabilidad de esta tabla
es siempre `max(p_team1, 1-p_team1)`; las odds no cambian el equipo elegido:

| Banda predicha | Partidos | Aciertos | Accuracy observada | Prob. media predicha |
|---|---:|---:|---:|---:|
| 50-60% | 2.823 | 1.545 | 54,73% | 54,86% |
| 60-70% | 2.329 | 1.516 | 65,09% | 64,88% |
| 70-80% | 1.496 | 1.127 | 75,33% | 74,34% |
| 80-90% | 404 | 354 | 87,62% | 83,12% |
| 90-100% | 4 | 3 | 75,00% | 91,12% |
| **Total** | **7.056** | **4.545** | **64,41%** | — |

La franja 90-100% no es interpretable todavia (`n=4`). Las demas franjas son
monotonas y las tres primeras estan bien calibradas; el mayor margen de mejora
sigue en los partidos cercanos al 50%.

### 4.6. Benchmark de mercado (odds)

Con odds de apertura guardadas (hoy n=98 partidos unicos, todavía ilustrativo): el **mercado** queda en
log loss 0.610 vs 0.628 del modelo en esos mismos partidos. Confirma que las odds son muy
informativas (PROJECT.md §6.5): se usan como **benchmark a batir** y como blend
de mercado en la web, no como feature única (canibalizaría el interés
académico). El modelo entrenado es "Model A" (solo stats) por diseño.

## 5. La web (dashboard)

Estática (HTML+JS sobre `data.js`), **sin servidor**, estilo cyberpunk
(negro/rojo neón). Tres vistas:

- **Partidos** — cartelera con predicción del modelo, mercado, **fiabilidad** y
  señales; panel de detalle con ratings Glicko, mercado, pool de mapas, roster,
  forma, fatiga, contexto y rosters.
- **★ Best Opportunity** — ranking por **confianza del modelo × fiabilidad de
  los datos** (penaliza RD alta, poca historia, BO1, rival por definir). Muestra
  dónde el modelo está seguro **y** respaldado por datos.
- **Modelo** — métricas walk-forward, importancia SHAP y curva de calibración
  con resultados reales.

## 6. Cómo ejecutarlo

Requiere `python` en PATH. Instala dependencias desde la raíz:

```powershell
python -m pip install -r requirements.txt
```

SQLite va incluido en Python.

**Scraper por tiers (anti-bloqueo Cloudflare).** El scraper usa
[Scrapling](https://scrapling.readthedocs.io): Tier 1 HTTP con impersonation TLS/JA3
(`curl_cffi`) y Tier 2 navegador stealth que resuelve el challenge, con
`requests`/`cloudscraper` como red de seguridad. `start.ps1` crea el venv del scraper
con **Python 3.13** (Scrapling **no** soporta 3.14) e instala `scrapling[fetchers]` +
navegadores automáticamente. Si no hay Python 3.10-3.13, el scraper degrada a
`requests`/`cloudscraper` con aviso. En redes con inspección TLS (proxy corporativo),
`start.ps1` genera un CA bundle automáticamente. Todos los cambios de esta iteración
están documentados en [`PROJECT_DOCS/runbooks/LAST_CHANGE_2026-07-06.md`](PROJECT_DOCS/runbooks/LAST_CHANGE_2026-07-06.md).

**Una sola orden hace TODO** (scrape → update → modelo → enrich → BBDD → web):

```powershell
.\start.ps1                  # pipeline completa (con acceso a HLTV)
.\start.ps1 -MaxMatches 3 -SkipPlayerStats -SkipTeamProfiles  # prueba online rapida; no publica master
.\start.ps1 -RecoveryWindowDays 3  # reintenta huecos recientes de odds/detalle/Analytics
.\start.ps1 -SkipScrape      # modo debug/offline explícito: usa el último run
start .\WEB\index.html       # abrir el dashboard
```

Si el scraping falla o devuelve cero datos, `start.ps1` falla por defecto: la
pipeline principal es online y no acepta silenciosamente un run viejo. Para un
debug puntual se puede pasar `-AllowOfflineFallback`. Entrena el modelo solo si
falta el artefacto; fuerza reentrenamiento con `-Retrain` cuando incorpores
features nuevas, mucho histórico nuevo o quieras recalibrar. Flags: `-SkipScrape`,
`-AllowOfflineFallback`, `-Retrain`, `-NoDb`, `-MaxMatches N`,
`-SkipPlayerStats`, `-SkipTeamProfiles`, `-SkipSameDayRecovery`,
`-RecoveryWindowDays N`, `-RecoveryDelay S`, `-RecreateScraperVenv`.

Cada ejecución de `start.ps1` crea un transcript completo en `logs/start_*.log`
y muestra cada comando con hora, exit code y duración. Si Cloudflare bloquea el
navegador stealth durante demasiado tiempo, el pipeline lanza automáticamente
`grab_cf.py` para abrir una ventana visible y renovar `cf_clearance`.

El scrape diario consulta primero `BBDD/cs2.db`: `/results` solo se pide para
resolver IDs `pending_result` conocidos, y solo pagina a `offset=100/200` si los
pendientes no aparecieron en las páginas anteriores. Si SQLite no tiene
pendientes, esa fase se salta.

Los runs con `-MaxMatches` son solo pruebas parciales: scrapean HLTV online,
pero no actualizan `DAILY_SNAPSHOTS/master/manifest.json` ni
`DAILY_SNAPSHOTS/master/matches.json`. La BBDD y la web se siguen regenerando
desde el ultimo run completo publicado.

Pasos sueltos (si los necesitas):

```powershell
python MODEL\train.py                          # entrena desde BBDD\cs2.db → MODEL\results\REPORT.md
python MODEL\train.py --warmup-weeks 10 --min-train 800
python MODEL\train.py --raw <results_all.json> # modo legacy/debug si necesitas saltarte SQLite
python MODEL\run_professional_training.py --install-deps  # sweep CatBoost/half-life/gap + output .md
python DAILY_SNAPSHOTS\start.py                # scrape diario + update pendientes (necesita HLTV)
python DAILY_SNAPSHOTS\enrich_predictions.py   # puntúa el último run con el modelo
python BBDD\build_db.py                        # crea/migra/siembra BBDD\cs2.db solo si hace falta
python BBDD\ingest.py --run-dir <DAILY_SNAPSHOTS\runs\RUN_ID>  # upsert incremental del run
python BBDD\export_master_json.py              # export compat desde SQLite
python WEB\build_web.py                        # genera WEB\data.js
python tests\tests_wallet_simulator.py --initial-wallet 100  # cartera + accuracy por franjas
python MODEL\analyze_walkforward_errors.py       # auditoria OOS de fallos, CSV y graficos
python MODEL\train.py --verbose                  # entrenamiento productivo: core + todos los algoritmos disponibles
python MODEL\run_professional_training.py --algorithms all --feature-profile core --half-lives 45,60,90,120,180 --wf-gaps 0,1
```

El simulador apuesta siempre al favorito puro del modelo. Las cuotas solo
permiten calcular beneficio, EV y stake; nunca hacen que el backtest elija el
equipo con probabilidad inferior al 50%. Genera los PNG, registros CSV y la
tabla `accuracy_by_confidence_band.csv` en `tests/graphs/`.

Para entrenar con la mejor calidad posible: ejecuta primero un `start.ps1`
completo para tener snapshots, odds, Analytics, contexto de torneo, rosters y
stats de jugador actualizados e ingeridos en SQLite; luego usa
`python MODEL\run_professional_training.py --install-deps`
para probar CatBoost, ensembles, half-life y gap con salida verbosa en
`MODEL\results\professional_training_output.md`. El entrenamiento usa validacion walk-forward temporal,
elige produccion por menor log loss, guarda el artefacto final en
`MODEL\artifacts\model.pkl` y versiona una copia en `MODEL\artifacts\registry`.
Analytics, contexto, stats de jugador, box score/mapas, historial de evento,
rankings y roster se calculan y guardan siempre que exista evidencia previa al
partido. Cada familia entra sola cuando supera el umbral de `MODEL\train.py`.
El estado exacto se imprime durante el entreno y queda en
`MODEL\results\REPORT.md` y `artifact.metadata["feature_policies"]`.

Cada entrenamiento genera tambien `MODEL\results\favorite_accuracy_bands.json`
con el acierto walk-forward del favorito en las franjas 50-60, 60-70, 70-80,
80-90 y 90-100%. `start.ps1 -Retrain` reconstruye despues `WEB\data.js`, por lo
que la tabla de la pestaña **BBDD** queda asociada automaticamente al modelo y
timestamp del ultimo entreno.

Rutina recomendada de lunes (rapida y suficiente para operativa normal):

```powershell
.\start.ps1 -Retrain
```

Rutina profesional profunda (cuando quieras repetir el sweep CatBoost/half-life/gap):

```powershell
.\start.ps1
python MODEL\run_professional_training.py --install-deps
python DAILY_SNAPSHOTS\enrich_predictions.py
python BBDD\ingest.py
python BBDD\export_master_json.py
python WEB\build_web.py
```

Cada etapa de la pipeline está documentada en la cabecera de `start.ps1`. Las
features contempladas pero aún no activas están resumidas en `PROJECT.md` y
detalladas en `PROJECT_DOCS\extra_features.md`.

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
detallado en `PROJECT_DOCS\extra_features.md`.

El Model B (stats + odds de apertura) ya está implementado como evaluación
walk-forward separada. A 2026-07-10 hay 98 partidos unicos completados con odds de
apertura, por debajo del mínimo configurado para concluir; se activará
automáticamente cuando el pipeline diario acumule muestra suficiente.

Estado de cobertura medido el 2026-07-10 tras deduplicar semilla+HLTV: evento
6017/200 (ON); box score/mapas 20/200, Analytics 100/120, jugadores 88/200,
rankings 86/200 y roster 61/200 (OFF). Contexto tiene 65/200, con 7 LAN y 58 online, por lo que sigue OFF
hasta alcanzar tambien 50 LAN. Estos estados se recalculan automaticamente en
cada entrenamiento.

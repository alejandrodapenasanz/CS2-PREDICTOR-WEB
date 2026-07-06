# CS2 Predictor

Sistema de **predicción pre-partido** de Counter-Strike 2 a partir de datos de
HLTV, orientado a **probabilidades bien calibradas** (no solo "quién gana"):
cuando el modelo dice 70%, el favorito debe ganar ~70% de las veces. Proyecto
académico, sin fines de apuesta real.

- **Especificación de diseño (teoría y decisiones):** [`PROJECT.md`](PROJECT.md)
- **Registro de cambios y decisiones:** [`CHANGELOG.md`](CHANGELOG.md)
- **Features pendientes / futuras:** [`extra_features.md`](extra_features.md)

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

## 3. Carpetas

| Carpeta | Contenido | Doc |
|---|---|---|
| [`MODEL/`](MODEL/) | Núcleo de ML: Glicko-2, features point-in-time, entrenamiento, calibración, walk-forward, SHAP, artefacto. | Documentado aquí y en `PROJECT.md` |
| [`BBDD/`](BBDD/) | Esquema SQLite + `build_db.py` (ingesta a 3 capas). La base como fuente de verdad. | Documentado aquí y en `PROJECT.md` |
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

El modelo estable usa features point-in-time derivadas de resultados y assets
históricos: Glicko-2/RD, Elo, forma global y con decay, forma específica por
formato (BO1/BO3/BO5), actividad/recencia, fuerza de rivales recientes,
head-to-head, experiencia en evento, mapas/rounds cuando hay box score y señales
de score.

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
  `past12months`). De momento se muestran y se guardan, pero no sustituyen al
  modelo estable hasta validar que no introducen fuga ni overfitting.

### 4.3. Modelos, calibración y selección

Walk-forward semanal (ventana expansiva, nunca k-fold aleatorio). Se evalúan
baselines (base-rate, Elo, Glicko-2), logística y LightGBM (ambos calibrados con
Platt) y su ensemble; se elige el de producción por **menor log loss**.

### 4.4. Resultados (era CS2, 9.444 series, 7.008 de test)

| Modelo | Accuracy | Log loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Base rate | 0.562 | 0.686 | 0.246 | 0.486 | — |
| Elo | 0.603 | 0.657 | 0.233 | 0.642 | 0.042 |
| Glicko-2 | 0.609 | 0.660 | 0.233 | 0.649 | 0.050 |
| Logística calibrada | 0.640 | 0.633 | 0.221 | 0.684 | 0.016 |
| LightGBM calibrado | 0.638 | 0.635 | 0.222 | 0.682 | 0.017 |
| **Ensemble calibrado (producción)** | **0.648** | **0.630** | **0.220** | **0.689** | **0.017** |

> Hallazgo honesto: con features de resultados (relaciones monótonas tipo
> "diferencia de rating"), la logística calibrada casi iguala al ensemble. Tras
> añadir forma específica por formato, el ensemble calibrado queda ligeramente
> por delante en log loss y pasa a producción. La elección sigue siendo
> data-driven.

### 4.5. ¿Tiene mérito el 64.8%? (acierto por competitividad)

Desglose del acierto por confianza del modelo — el 64.8% **no** está inflado por
palizas (solo el 5.8% lo son):

| Banda (confianza) | % partidos | Accuracy | Log loss |
|---|---:|---:|---:|
| coinflip (<55%) | 21.2% | 55.0% | 0.691 (≈ azar, ln2≈0.693) |
| parejos (55-65%) | 37.1% | 60.7% | 0.668 |
| claros (65-80%) | 36.0% | 71.5% | 0.591 |
| palizas (≥80%) | 5.8% | 85.2% | 0.415 |

El valor está en la banda 55-80%; en los coinflips genuinos la incertidumbre es
irreducible. **El techo de mejora está en la banda 55-65%** (37% de los
partidos), y para subir ahí hacen falta features de jugador/mapa que requieren
más scraping.

### 4.6. Benchmark de mercado (odds)

Con odds de apertura guardadas (hoy n=69, ilustrativo): el **mercado** queda en
log loss 0.645 vs 0.665 del modelo en esos mismos partidos. Confirma que las odds son muy
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

Los runs con `-MaxMatches` son solo pruebas parciales: scrapean HLTV online,
pero no actualizan `DAILY_SNAPSHOTS/master/manifest.json` ni
`DAILY_SNAPSHOTS/master/matches.json`. La BBDD y la web se siguen regenerando
desde el ultimo run completo publicado.

Pasos sueltos (si los necesitas):

```powershell
python MODEL\train.py                          # entrena + evalúa → MODEL\results\REPORT.md
python MODEL\train.py --warmup-weeks 10 --min-train 800
python DAILY_SNAPSHOTS\start.py                # scrape diario + update pendientes (necesita HLTV)
python DAILY_SNAPSHOTS\enrich_predictions.py   # puntúa el último run con el modelo
python BBDD\build_db.py                        # construye BBDD\cs2.db
python WEB\build_web.py                        # genera WEB\data.js
```

Para entrenar con la mejor calidad posible: ejecuta primero un `start.ps1`
completo para tener snapshots, odds, Analytics, contexto de torneo, rosters y
stats de jugador actualizados; luego usa `.\start.ps1 -Retrain` o
`python MODEL\train.py`. El entrenamiento usa validacion walk-forward temporal,
elige produccion por menor log loss, guarda el artefacto en
`MODEL\artifacts\model.pkl` y versiona una copia en `MODEL\artifacts\registry`.
Los bloques Analytics/contexto/stats de jugador se calculan y guardan siempre,
pero solo entran en el modelo cuando superan los umbrales de muestra definidos
en `MODEL\train.py`.

Cada etapa de la pipeline está documentada en la cabecera de `start.ps1`. Las
features contempladas pero aún no activas están en `extra_features.md`.

## 7. Garantías y ética

- **Cero fuga temporal** por diseño (features point-in-time, validación
  cronológica). Síntoma de fuga que se evita: 80% en validación, 55% en real.
- **Calibración** validada con curvas, no solo accuracy.
- Proyecto **académico, sin apuestas reales**. Integridad competitiva solo con
  sanciones oficiales (PROJECT.md §6.7). Scraping de bajo volumen, respetuoso con
  los ToS de HLTV (caché para no re-descargar, backoff, ritmo humano).

## 8. Estado y siguientes pasos

Funciona end-to-end en modo online. A 2026-07-06 el run publicado es
`2026-07-06_065318Z` (20 partidos). La web muestra stake recomendado por partido
con Kelly fraccional ajustado por fiabilidad, consenso de mercado y calibracion.
La BBDD fisica `BBDD/cs2.db` guarda odds, predicciones, snapshots crudos y stats
agregadas de jugador por run, y cada reconstruccion crea backup en
`BBDD/backups/`.

El mayor margen de mejora (banda de partidos parejos) sigue estando en box score
por mapa, stats de jugador point-in-time, veto real y odds historicas. Plan
detallado en `extra_features.md`.

El Model B (stats + odds de apertura) ya está implementado como evaluación
walk-forward separada. A 2026-07-06 hay 69 partidos completados con odds de
apertura, por debajo del mínimo configurado para concluir; se activará
automáticamente cuando el pipeline diario acumule muestra suficiente.

Los bloques opcionales quedan preparados pero apagados en el artefacto actual:
HLTV Analytics tiene 53/120 filas cerradas y contexto de torneo tiene 118/200
filas cerradas, con 37 LAN y 81 online (umbral prudente: 50 por entorno).

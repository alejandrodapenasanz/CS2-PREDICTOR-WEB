# Changelog / registro de decisiones

## 2026-07-27 - BLACKBOX: caja negra portatil de la fuente de verdad

- Nueva herramienta `BBDD/blackbox.py` (solo stdlib) con cuatro mecanismos:
  `export`, `verify`, `restore` y `autoheal`. Permite reconstruir `cs2.db`
  completa a partir de una unica carpeta portatil `BBDD/BLACKBOX/` (USB/nube).
- Principio: en BLACKBOX solo se guarda la FUENTE DE VERDAD no recomputable
  (identidad, hechos, staging raw, odds, sanciones, predicciones congeladas).
  Lo derivado (`ratings_history`, `match_features`) y lo operativo
  (`fetch_state`, `ingest_runs`) se EXCLUYE: se regenera.
- Formato duradero y no propietario: SQLite podado + volcado SQL de texto,
  gzip, con `manifest.json` (SHA-256 por fichero y hash de contenido por tabla,
  version y timestamp). Export atomico con auto-verificacion y generacion previa
  en `.prev/`.
- `restore` es idempotente y tiene guardian: nunca pisa una `cs2.db` sana sin
  `--force`, y respalda en `BBDD/backups/` antes de cualquier reemplazo.
- `start.ps1`: auto-heal al inicio (restaura si la BBDD falta/vacia/corrupta)
  y flags `-BackupBlackbox`, `-RestoreBlackbox`, `-SkipAutoHeal`.
- `BBDD/BLACKBOX/README.md` + `restore_standalone.py` permiten restaurar a mano
  sin el proyecto. Test `TESTS/test_blackbox_disaster.py` simula el desastre y
  verifica identidad por checksums + dry-run del lector de entrenamiento.

## 2026-07-26 - Dominio CS2 aislado para arquitectura multideporte

- Todo el dominio (`BBDD`, `MODEL`, `PIPELINE`, `SCRAPER`, `TESTS`, `DOCS`,
  configuración, dependencias y documentación) vive bajo `CS2/`.
- `WEB/` permanece en la raíz como presentación compartida. Su builder recibe
  `--sport-root` y publica actualmente el dominio `CS2`.
- El `start.ps1` raíz es un wrapper compatible; la orquestación real está en
  `CS2/start.ps1`.
- BBDD, 73 runs, 90 backups, 47 logs, artefactos y venv se movieron sin
  reconstrucción ni pérdida.
- La reingesta deja de borrar predicciones del mismo run que ya no aparecen en
  la cartelera actual. Se restauraron 9 snapshots desde backup y un test exige
  que el histórico permanezca append-only.

## 2026-07-26 - Best Opportunity point-in-time

- La politica `best_opportunity_v1` centraliza el score
  `decision_confidence * reliability_score` y los gates 60%/45%.
- Cada prediccion persiste score, elegibilidad, ranking global/diario,
  `is_best_opportunity` y version de politica en SQLite.
- La ingesta tambien completa favorito, politica de decision, contexto y cuotas
  minimas que ya formaban parte del esquema profesional.
- `TESTS/tests_best_opportunity_simulator.py` usa la ultima foto real anterior
  al inicio, deduplica por partido y genera el informe bajo `MODEL/results/`.
- Backtest inicial: pool elegible 69/105 (65.71%); numero 1 diario 11/17
  (64.71%); top 3 diario 37/50 (74.00%). La muestra diaria aun es pequena.

## 2026-07-26 - Raiz minima y carpetas core

- `DAILY_SNAPSHOTS` pasa a `PIPELINE`, `SCRAPPER` a `SCRAPER`,
  `PROJECT_DOCS` a `DOCS` y `tests` a `TESTS`.
- Los logs operativos viven en `PIPELINE/logs`; los caches de Pytest, Ruff y
  Mypy se consolidan en `TESTS/.cache`.
- `pyproject.toml` reemplaza los ficheros separados `ruff.toml` y `mypy.ini`.
- Cada carpeta superior tiene ownership unico; la raiz conserva solo docs,
  dependencias, configuracion compartida y `start.ps1`.

## 2026-07-26 - Red flag de cambio de roster point-in-time

- La alineacion anunciada del partido se compara por IDs HLTV con la alineacion
  real del ultimo partido anterior del equipo dentro de una ventana de 90 dias.
- Solo se aceptan lineups exactas de cinco jugadores y snapshots capturados
  antes del inicio. Se ignoran futuros, historicos de mas de 90 dias y
  alineaciones parciales.
- `ROSTER_CHANGE_90D` usa nivel `danger` y muestra altas, bajas, partido anterior
  y cantidad de partidos comparados. Los stand-ins anunciados conservan ademas
  su warning independiente.
- La evidencia se persiste en `predictions.controls_json`; la alerta no parchea
  la probabilidad del modelo.
- Validacion viva del run `2026-07-26_131713Z`: 40 alineaciones sin cambio, 6
  cambios, 17 sin historico completo y 7 sin lineup anunciada completa.

## 2026-07-16 - Player stats selectivas y parser por columna

- **Causa corregida**: HLTV renderiza `/stats/players/compare` como dos
  columnas independientes. El parser anterior podia copiar valores de la
  columna con datos a un jugador con `Based on 0 maps`.
- **Parser estructural**: cada columna conserva sus propios mapas y metricas;
  el perfil individual valida Rating, DPR, ADR y el resto de campos core. Cero
  mapas elimina valores de compare no fiables y se clasifica como `not_found`.
- **Refresco por jugador**: el TTL se evalua por ID. El scraper recibe solo
  jugadores nuevos, vencidos o incompletos; los frescos se materializan desde
  SQLite sin crear snapshots ni checkpoints ficticios.
- **Cloudflare**: player stats corta y guarda parcial ante el primer challenge,
  renueva `cf_session.json` automaticamente y reintenta sin acumular esperas
  exponenciales largas.
- **Estado idempotente**: las ventanas 3/6/12 meses se agregan antes de escribir
  `fetch_state`; una ventana completa prevalece sobre otra parcial del mismo
  run y reingerir la misma evidencia no incrementa `fetch_count`.
- **Panel BBDD**: `not_found` aparece como "Sin muestra", separado de
  `partial`/`blocked`/`error`. Validacion viva: 565 OK, 4 sin muestra y 0 issues.
- **Tests**: 96 tests + 17 subtests pasan; parser validado tambien contra HTML
  real de HLTV con un jugador de 31 mapas frente a otro de 0 mapas.

## 2026-07-13 - Rigor de produccion C12-C17

- **Drift causal**: `MODEL/monitor_drift.py` y `cs2model/drift.py` calculan log
  loss/CLV rodantes y Page-Hinkley solo sobre predicciones pre-match cerradas.
- **Regimen**: map pool reconstruido exclusivamente con mapstats anteriores;
  parche desconocido salvo metadata pre-match explicita. Familia auto-gated.
- **Configuracion versionada**: `MODEL/config.yaml` + dataclasses validadas para
  defaults, umbrales, estimadores, drift y backtest.
- **Reproducibilidad**: semilla global y manifiesto con hashes de dataset/config,
  Git, CLI y dependencias; YAML efectivo archivado con cada experimento.
- **Backtest realista**: vig/de-vig, Kelly fraccional, limites de stake/payout,
  ejecucion configurable y CLV contra la ultima cuota pre-match.
- **CI bloqueante**: ruff, mypy de infraestructura, 89 tests + 17 subtests y
  smoke determinista end-to-end en push/PR a main/dev/pre-dev.

## 2026-07-07 - BBDD viva e ingest incremental

- **SQLite como fuente de verdad viva**: `BBDD/build_db.py` pasa a crear/migrar
  esquema y sembrar solo si `matches` esta vacia. Ya no se reconstruye la base
  desde cero en cada ejecucion normal.
- **Ingest incremental**: nuevo `BBDD/ingest.py` aplica cada run con upserts:
  partidos, odds, predicciones, raw snapshots, rankings, rosters y assets HLTV
  normalizados (`maps`, `veto`, `match_lineups`, `map_player_stats`,
  `map_player_side_stats`).
- **Freshness compartida**: nueva tabla `fetch_state` y lectura desde
  `PIPELINE/start.py` para saltar perfiles/stats/rankings/assets frescos
  y reintentar pronto bloqueos o errores sin borrar informacion buena previa.
- **Auditoria**: nueva tabla `ingest_runs` y flags en `matches` para cobertura
  (`has_prematch_odds`, `has_player_snapshot`, `has_ranking_snapshot`,
  `has_analytics`, `has_context`, `has_box_score`, `has_veto`).
- **Compatibilidad**: nuevo `BBDD/export_master_json.py` regenera
  `PIPELINE/master/matches.json` desde SQLite mientras la web/scraper
  conservan consumidores JSON.
- **Entrenamiento**: `MODEL/train.py` lee de `BBDD/cs2.db` por defecto; `--raw`
  queda como modo legacy/debug. Tests nuevos cubren esquema, TTL, odds de
  apertura y auditoria de fuga temporal.
- **Orquestacion**: `start.ps1` ingiere el run antes de entrenar cuando se usa
  `-Retrain`, y vuelve a ingerir al final para congelar predicciones/staking y
  exportar el master JSON de compatibilidad.
- **Resultados dirigidos por BBDD**: la fase `/results` de `start.py` ya no pagina
  offsets antiguos a ciegas; primero lee los `hltv_match_id` pendientes de
  SQLite, salta la fase si no hay pendientes y solo llega a `offset=100/200` si
  esos IDs no aparecieron antes.
- **Diagnostico y bloqueo**: `start.ps1` deja transcript en `PIPELINE/logs/start_*.log`
  con comandos, exit code y duracion. Si el navegador stealth queda bloqueado,
  se lanza `grab_cf.py` para renovar `cf_clearance` con una ventana visible y se
  marca `fetch_state` para no repetir inmediatamente assets bloqueados.

## 2026-07-06 - Scraper por tiers con Scrapling + candidatos de modelo

Detalle completo en `DOCS/runbooks/LAST_CHANGE_2026-07-06.md`. Resumen:

- **Scraper anti-bloqueo (Scrapling)**: `PIPELINE/start.py::fetch_html` pasa
  a arquitectura por tiers sobre el choke point único: Tier 1 `Fetcher(impersonate=
  "chrome")` (curl_cffi, fingerprint TLS/JA3 real) reutilizando `cf_clearance`;
  Tier 2 `StealthyFetcher(solve_cloudflare=True)` que resuelve el challenge y acuña
  cookie fresca (persistida en `cf_session.json`); Tiers 3-5 `requests`/`cloudscraper`/
  `grab_cf` como red de seguridad. Validado en vivo: `requests` → 403, Scrapling
  impersonate → 200 en HLTV.
- **Detección de bloqueo** ampliada: header `cf-mitigated: challenge` + marcadores
  (`challenge-platform`, `turnstile`, `__cf_chl`, …); UA por defecto Chrome 140.
- **CA corporativa (redes con inspección TLS)**: `start.ps1::Ensure-CaBundle` hace
  probe TLS y, si detecta MITM, exporta el trust store de Windows a
  `corp_ca_bundle.pem` y lo publica por env. En red normal no genera nada (certifi).
- **Orquestación**: `start.ps1` crea el venv del scraper con **Python 3.13**
  (Scrapling no soporta 3.14), instala `scrapling[fetchers]` + navegadores, y degrada
  a requests/cloudscraper con aviso si no hay Python soportado. Nuevas variables
  `HLTV_USE_SCRAPLING`, `HLTV_SOLVE_CLOUDFLARE`, `HLTV_IMPERSONATE`, `HLTV_PROXY`, etc.
- **Modelo (candidatos seguros por construcción)**: el trainer elige por menor log
  loss walk-forward, así que se añaden candidatos que solo ganan si mejoran:
  **CatBoost** (`catboost_cal`) y **ensemble de 3** (`ensemble3_cal`, opcional),
  **restricciones monótonas** en GBDT, **early stopping**, calibración **beta**
  (Kull & Flach) e **isotónica** además de Platt, **half-life de decay tunable**
  (`--form-half-life`) y **gap temporal** en walk-forward (`--wf-gap`). Verificado con
  smoke test sintético; falta reentrenar con datos reales en un PC sin restricciones.

## 2026-07-02 - HLTV betting analytics por defecto

- **Analytics future-proof**: `PIPELINE/start.py` captura
  `/betting/analytics/<match_id>/<slug>` por defecto para partidos upcoming y
  para pendientes que aun no tengan analytics disponible.
- **Persistencia auditable**: cada captura queda en `analytics/<match_id>.json`
  y su HTML crudo en `raw_html/analytics_page/` con checksum SHA256.
- **Master + BBDD**: `master/matches.json` guarda la ultima analytics por
  partido; `BBDD/build_db.py` archiva los JSON como
  `raw_snapshots(kind='match_analytics')`.
- **Control anti-leakage**: estos datos solo deben usarse como features si la
  captura fue pre-partido; si llegan despues del resultado quedan como contexto
  o auditoria.

## 2026-07-01 - HLTV mapstats/veto/rankings online

- **Assets HLTV completados**: `PIPELINE/start.py` captura veto real,
  links `mapstatsid` y box score por mapa desde HLTV, con tablas `total`, `ct`
  y `t` por jugador.
- **Persistencia normalizada**: `BBDD/cs2_prediction_schema.sql` y
  `BBDD/build_db.py` rellenan `maps`, `veto`, `match_lineups`,
  `map_player_stats`, `map_player_side_stats` y `team_ranking_snapshots`, ademas
  de archivar los JSON crudos como `raw_snapshots(kind='match_assets')`.
- **Backfill real ejecutado**: 40 partidos completados del master tienen assets;
  la BBDD reconstruida contiene 80 mapas, 273 pasos de veto, 383 lineups, 800
  filas total jugador/mapa y 2.400 filas jugador/mapa/lado.
- **Rankings actuales**: capturados 240 equipos en ranking HLTV y 380 en ranking
  Valve, con timestamp y payload original.
- **Reentrenamiento**: el loader usa assets como fallback cuando falta `detail`;
  modelo reentrenado con 9.366 series. Produccion sigue en `ensemble_cal`
  (log loss walk-forward 0.6303, accuracy 0.6455).

## 2026-07-01 — Formato BO1/BO3, stake y backups fisicos

- **Modelo format-aware**: `MODEL/cs2model/features.py` pasa de 28 a 37
  features; mantiene rating/forma global, pero añade forma, experiencia,
  diferencial de marcador, fuerza de rivales y H2H especificos del mismo
  best-of (BO1/BO3/BO5). Nuevo test `TESTS/test_format_features.py`.
- **Reentrenamiento real**: `MODEL/train.py` reentrenado con 9.365 series; el
  modelo de produccion pasa a `ensemble_cal` por menor log loss walk-forward
  (0.6302).
- **Stake por partido**: `PIPELINE/enrich_predictions.py` añade
  recomendacion de stake basada en Kelly fraccional (25%) ajustado por
  fiabilidad, consenso de mercado y ECE walk-forward. La web muestra stake en
  cartelera, detalle y Best Opportunity.
- **BBDD como backup fisico**: `BBDD/cs2_prediction_schema.sql` añade
  `raw_snapshots` y `player_stat_snapshots`; `BBDD/build_db.py` las rellena desde
  `PIPELINE/runs` y crea copias timestamp en `BBDD/backups/`.
- **Dependencias raiz**: nuevo `requirements.txt` con dependencias de modelo,
  scraper, API auxiliar y tests.

## 2026-06-30 (iteración 5) — Protección contra runs parciales

- **Incidente corregido**: una validación con `-MaxMatches 1` quedó promovida
  como último run y regeneró la web con 1 solo partido. No se habían borrado los
  partidos: el run completo `2026-06-30_053753Z` seguía intacto.
- **Restauración**: `PIPELINE/master/manifest.json` vuelve a apuntar al
  run completo, `WEB/data.js` se regeneró con 34 partidos y 14 programados para
  2026-07-01.
- **Blindaje**: `PIPELINE/start.py` añade `--no-promote`; además, todo
  run con `--max-matches` se considera debug y no actualiza el manifest maestro
  ni `master/matches.json`. `start.ps1` pasa `--no-promote` automáticamente
  cuando se usa `-MaxMatches`.
- **Limpieza del incidente**: retiradas del master 92 referencias a runs
  parciales de validación y eliminadas sus carpetas de `PIPELINE/runs`.
- **BBDD robusta ante debug runs**: `BBDD/build_db.py` elige
  `predictions_enriched.json` desde el run publicado en el manifest, no desde la
  carpeta de run más reciente.

## 2026-06-30 (iteración 4) — Pipeline online estricta

- **`start.ps1` pasa a online estricto**: separa Python de modelo y venv del
  scraper, recrea el venv si hace falta, scrapea HLTV por defecto y falla si el
  scrape no produce datos. El fallback offline queda solo bajo flags explícitas:
  `-SkipScrape` o `-AllowOfflineFallback`.
- **Cloudflare session en Scrapy**: nuevo `CloudflareSessionMiddleware` inyecta
  `cf_clearance` y `User-Agent` desde `cf_session.json` en las peticiones HLTV.
- **Listados online robustos**: `PIPELINE/start.py` descarga `/results` y
  `/matches` con `requests + parsel` autenticado por `cf_session.json`, usando
  los parsers existentes. Scrapy queda como fallback.
- **Fallback pre-partido sin leakage**: si una página de detalle de partido está
  rate-limited, el snapshot conserva los datos del listado (`upcoming_row_fallback`)
  para poder puntuar el partido sin meter información futura.
- **Validación online real**: `.\start.ps1 -MaxMatches 1 -SkipPlayerStats
  -SkipTeamProfiles` scrapeó 300 resultados recientes y 1 partido próximo. La
  iteración 5 corrige que este tipo de run parcial no debe promocionarse ni
  sustituir el dashboard publicado.

## 2026-06-30 (iteración 3) — Odds en BBDD y Model B preparado

- **Odds volcadas a SQLite**: `BBDD/build_db.py` fusiona el histórico con
  `PIPELINE/master/matches.json` y rellena `odds` con apertura, cierre e
  historial live. La tabla guarda cuota decimal, probabilidad implícita
  normalizada, overround, timestamp y provenance. Resultado actual:
  9.352 partidos y 69 filas de odds.
- **Bookmakers preservados hacia delante**: `PIPELINE/start.py` conserva
  `providers` dentro de cada punto de odds. Los snapshots antiguos siguen siendo
  válidos como `bookmaker=average`.
- **Model B implementado, no sobrevendido**: `MODEL/train.py` construye features
  de odds de apertura y evalúa stats+odds walk-forward solo si hay muestra
  suficiente. Hoy hay 16 partidos con opening odds, menos del mínimo de 40, así
  que `MODEL/results/model_b_eval.json` queda como no concluyente. El benchmark
  de mercado sigue siendo ilustrativo: log loss 0.5437 vs 0.6344 del modelo en
  n=16.
- **Test anti-fuga para odds**: `TESTS/test_odds_point_in_time.py` verifica que
  el entrenamiento usa apertura y no cierre; si falta apertura, usa solo el
  primer punto de `odds_history`.
- **Dependencias ML restauradas**: instalados `lightgbm` y `shap` en el Python
  activo para regenerar SHAP real en `MODEL/results/REPORT.md`.
- **Limpieza de encargo temporal**: el encargo puntual se integra en código/docs
  y se elimina; el plan permanente queda en `extra_features.md`.

## 2026-06-30 (iteración 2) — Minimalismo, pipeline única y odds

- **Evaluación honesta del acierto**: desglose por competitividad (el 64.5% no
  es por palizas: solo el 8% lo son; el techo está en la banda 55-65%) y
  **benchmark de mercado** (con n=16, el mercado gana: log loss 0.54 vs 0.63).
  Ambos visibles en la web (vista Modelo) y en `MODEL/results/REPORT.md`.
- **Datos extendidos a hoy**: el entrenamiento fusiona los partidos completados
  del pipeline diario (`load_training_rows`) → llega a 30/6, no solo al backfill.
- **Odds**: confirmado que el scraper ya las captura (apertura/cierre por casa).
  Quedaba pendiente guardarlas y volcarlas a la tabla `odds` (es el dato más
  predictivo que falta a escala).
- **Pipeline única**: `.\start.ps1` (sin flags) hace scrape → update → modelo →
  enrich → BBDD → web. En la iteración 4 pasó a fallar fuerte si el scrape
  online no produce datos. Entrena solo si falta el artefacto. Definición al
  detalle en `start.ps1`.
- **Minimalismo**: eliminada la carpeta `BACKTEST/` (legacy, superada por
  `MODEL/`), los raw de prueba (`smoke`, `smoke2`, `model_seed`) y los
  `__pycache__`. Solo queda lo imprescindible para ejecutar.
- **Rename** `Scrapper/` → `SCRAPER/` (coherencia de mayúsculas); referencias
  en código y docs actualizadas.

## 2026-06-30 — Reescritura del núcleo predictivo y la web

Revisión mayor que cierra varios `[ABIERTO]` de `PROJECT.md` y profesionaliza el
proyecto. Resumen y por qué de cada decisión.

### Modelo (nuevo `MODEL/`)
- **Glicko-2 sobre Elo plano** (PROJECT.md §5). Periodos de rating semanales; la
  RD modela incertidumbre y crece con la inactividad. *Por qué:* distingue
  ratings maduros de inestables y da una "red flag" principiada por datos viejos.
  *Resultado:* Glicko-2 baseline supera a Elo en AUC (0.649 vs 0.642).
- **Features point-in-time compartidas** entre backtest y producción
  (`ChronologicalState`). *Por qué:* el "cero fuga" pasa a estar garantizado por
  el flujo (emitir features antes de observar el resultado), no por vigilancia.
- **Gradient boosting (LightGBM) + logística + ensemble**, todos calibrados, con
  **selección por log loss walk-forward**. *Por qué:* PROJECT.md pide GBM, pero
  honestamente con features de resultados la logística calibrada iguala/supera al
  GBM (literatura §3); la elección es data-driven, no por preferencia.
- **Calibración Platt** elegida sobre isotónica. *Por qué:* menor varianza con
  holdouts pequeños; la isotónica degradaba un modelo ya bien calibrado.
- **Stats de jugador NO entran al modelo.** *Por qué:* el histórico de HLTV no
  las trae fechadas; usar el snapshot actual sería leakage (§9). Se mantienen
  como contexto en la web y para flags.
- **SHAP** para interpretabilidad (§6.5): confirma `glicko_prob` como feature
  dominante.
- Artefacto persistido (`MODEL/artifacts/model.pkl`) que carga producción.

### Producción (`PIPELINE/enrich_predictions.py`)
- Sustituido el scorer logístico **hardcodeado** por el artefacto entrenado y
  calibrado. Fallback seguro al scorer ligero si no hay artefacto.
- La predicción **primaria** es el modelo de stats (Model A, interpretable, sin
  canibalizar por el mercado §6.5); el blend con odds queda como vista de mercado
  secundaria. La **confianza** (y "Best Opportunity") se basan en el modelo.
- Predicción **simétrica** (se promedian ambos órdenes de equipos).
- La calibración diaria real valida ahora el modelo (`model_prob_team1`).

### BBDD (`BBDD/`)
- Esquema ampliado (no destructivo): `raw_results` (staging con provenance),
  `predictions` (auditoría), `match_features` con columnas reales, índices.
- `build_db.py`: ingesta `results_all.json` → SQLite materializando las tres
  capas (staging / core / feature mart) con ratings_history point-in-time.
  Poblada: 9.326 partidos, 18.652 filas de ratings/features. SQLite, sin instalar.

### WEB (`WEB/`)
- Rediseño **cyberpunk** (negro/rojo neón, scanlines, mono): tres vistas —
  Partidos, **Best Opportunity** (ranking por confianza × fiabilidad de datos),
  y Modelo (métricas walk-forward + SHAP + curva de calibración real).
- `build_web.py` embebe métricas del modelo y SHAP en `data.js`.

### Orquestación / scraping
- `start.ps1` robusto al **venv roto** (caía siempre): detecta venv inválido y
  usa el `python` del sistema. Nuevas flags `-Retrain`, `-BuildDb`, `-SkipScrape`.
- La documentación de scraping queda centralizada en `README.md`, `PROJECT.md`
  y la cabecera de `start.ps1`.

### Pendiente (necesita acceso a HLTV)
- Box score por mapa → `map_player_stats`, veto real, arquitectura composicional
  Bo3. Stats de jugador point-in-time. Odds históricas. Tabla `sanctions`.

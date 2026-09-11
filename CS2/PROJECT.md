# CS2 Match Prediction — Especificación Técnica del Proyecto

**Proyecto académico de predicción de resultados de partidos profesionales de Counter-Strike 2 a partir de datos de HLTV.**

*Última validación del documento: julio de 2026 · Era de datos objetivo: CS2 (desde octubre de 2023).*

---

## 0. Cómo leer este documento

Este es el documento de diseño de referencia del proyecto. Define **qué** se construye, **por qué** se toma cada decisión y **cómo** se garantiza que el modelo sea fiable y reproducible. Está pensado para que cualquiera (o una herramienta como Claude Code) pueda retomar el trabajo con el contexto completo.

Las decisiones marcadas con `[ABIERTO: …]` son las que aún dependen de tu criterio y conviene cerrar explícitamente antes de avanzar.

El principio que gobierna todo el proyecto, y que conviene tener presente en cada sección:

> La base de datos guarda **la verdad de lo que pasó, con su sello de tiempo**. Las features y los ratings son **vistas calculadas** sobre esa verdad, reconstruibles a cualquier instante del pasado. Si se respeta esa separación, el "cero fuga temporal" deja de ser algo que se vigila a mano y pasa a estar garantizado por el diseño.

---

## 0.1. Glosario rápido

- **Snapshot:** captura congelada de los datos disponibles en un momento concreto: partido, cuotas, estadísticas de jugadores, rankings, Analytics, etc. Sirve para guardar lo que se sabía entonces, aunque HLTV o las casas cambien o borren información después. Es la pieza clave para backups, auditoría y entrenamiento sin fuga temporal.
- **Glicko / Glicko-2:** sistema de rating parecido a Elo, pero más profesional para este caso porque no solo estima la fuerza de un equipo, sino también la incertidumbre de esa estimación mediante RD (*rating deviation*) y volatilidad. Un equipo con pocos partidos puede tener buen rating, pero si su RD es alta el sistema reconoce que hay poca confianza.
- **Tener Glicko a favor:** significa que, antes del partido, el rating Glicko del equipo es superior al del rival. En la web se interpreta como una ventaja estadística histórica: cuanto mayor sea la diferencia y menor la incertidumbre, más fuerte es la señal. No significa victoria garantizada.
- **Log loss:** métrica para evaluar probabilidades, no solo aciertos/fallos. Premia probabilidades bien calibradas y penaliza mucho las predicciones confiadas que fallan. Por ejemplo, fallar diciendo 90 % es bastante peor que fallar diciendo 55 %. En betting es más útil que mirar solo accuracy, porque el valor esperado depende de la calidad de la probabilidad.

---

## 1. Objetivo y alcance

### 1.1. Objetivo

Estimar, **antes de que empiece un partido**, la probabilidad de que cada equipo gane la **serie** (Best-of-3), a partir de información histórica disponible en HLTV.

El entregable no es una etiqueta binaria sino una **probabilidad bien calibrada**: cuando el modelo dice 70 %, el favorito debe ganar aproximadamente el 70 % de las veces.

### 1.2. Prioridades del modelo (en orden, según decisión del autor)

1. Máxima exactitud (accuracy).
2. Probabilidades bien calibradas.
3. Interpretabilidad (entender qué pesa).
4. Capacidad de batir las cuotas de las casas.

> **Nota metodológica importante.** Las prioridades 1 y 2 no son independientes en este problema. Un modelo se entrena minimizando *log loss*, lo que produce buena calibración **y** buena exactitud de forma conjunta. Optimizar la exactitud de forma directa empuja al modelo a elegir siempre al favorito y a tirar la información probabilística. Por tanto se tratan como un objetivo conjunto, no como una jerarquía estricta.

### 1.3. Alcance de datos

- **Formato objetivo:** serie Bo3 (formato dominante en el circuito).
- **Cobertura:** todos los tiers, incluido online. Esta elección amplía la cobertura y hace más interesante la evaluación (el baseline "elige al favorito" es más débil), pero introduce ruido y exige tratar explícitamente la incertidumbre y la integridad competitiva (ver §6 y §7).

### 1.4. Fuera de alcance (por ahora)

- Predicción de win probability **en vivo, por ronda** (requiere parsear demos, no datos de HLTV; ver §11).
- Predicción de mercados de propuesta (props), kills individuales, etc.
- Cualquier uso con fines de apuesta real; el proyecto es académico.

---

## 2. Formulación del problema

### 2.1. Es clasificación supervisada, no aprendizaje por refuerzo

El problema es: *dado un vector de features conocidas antes del partido, predecir una etiqueta probabilística de victoria*. Esto es **clasificación binaria supervisada** sobre datos tabulares.

**No se usa Deep Reinforcement Learning (DRL).** DRL resuelve problemas de **decisión secuencial**: un agente toma acciones en un entorno para maximizar una recompensa acumulada. Predecir un resultado no tiene agente, ni espacio de acciones, ni entorno con el que interactuar. Forzar DRL aquí añade complejidad, multiplica la necesidad de datos, hace la validación casi imposible y rinde peor que un modelo tabular bien ajustado.

> **Dónde sí cabría RL (fuera del alcance principal):** modelar la **decisión de veto** (qué mapa pickear/banear) como un problema de bandits o decisión secuencial. Eso es optimización de decisiones, no predicción de resultados, y se deja como posible extensión.

### 2.2. Dos problemas que no hay que confundir

| | **Pre-partido (este proyecto)** | **Win probability en vivo** |
|---|---|---|
| Entrada | Datos agregados de HLTV | Estado de ronda (HP, equipamiento, jugadores vivos, tiempo) |
| Fuente | Web de HLTV | Demos parseados |
| Tarea | ¿Quién gana la serie? | ¿Quién gana la ronda? |
| Modelo de referencia | Sistemas de rating + GBM | XGBoost sobre eventos de ronda |

La literatura fundacional de win probability (Xenopoulos et al.) resuelve la columna derecha; sus features clave (valor de equipamiento, HP) **no existen** en datos agregados de HLTV. No se debe copiar ese enfoque para este problema.

---

## 3. Trabajo relacionado

Resumen de la literatura relevante (referencias completas en §13):

- **Xenopoulos et al. (2020), win probability con XGBoost.** Modelo de probabilidad de victoria por ronda entrenado sobre decenas de millones de eventos de juego. Hallazgo central: el valor del equipamiento del equipo es el principal determinante de ganar una ronda, por encima incluso del número de jugadores vivos. Relevante como referencia conceptual, pero resuelve el problema *en vivo*, no el pre-partido.
- **Björklund et al. (2018).** Predicción a partir de composición de equipo y clustering de roles + red neuronal sobre demos.
- **Makarov et al. (2017).** Enfoque clásico con regresión logística.
- **Estudio comparativo con TrueSkill (IEEE).** Comparó árboles de decisión, gradient boosting, XGBoost, regresión logística y redes neuronales; XGBoost y la red neuronal rindieron mejor, sin diferencia significativa entre ambos, y la introducción de valores TrueSkill mejoró ligeramente todos los algoritmos. Conclusión práctica: **la red neuronal no supera a XGBoost en datos tabulares**, y el sistema de rating es la pieza más valiosa.
- **Interpretabilidad con SHAP (2025).** Uso de machine learning interpretable con valores SHAP para analizar el rendimiento de jugadores profesionales de Counter-Strike.

**Síntesis:** el estado del arte para predicción tabular pre-partido es **gradient boosting + un buen sistema de rating como feature**, con calibración y validación temporal rigurosa.

---

## 4. Datos

### 4.1. Fuente: HLTV

HLTV es la fuente canónica de resultados, ratings y rankings. Consideraciones de obtención:

- La web renderiza contenido en servidor y está protegida con Cloudflare, que bloquea peticiones automatizadas. El scraping directo ingenuo no funciona.
- Herramientas viables: el paquete `hltv` de npm; scrapers en Python basados en `undetected-chromedriver` o Camoufox (sortean Cloudflare con fingerprinting TLS/JA3); actores de Apify; el endpoint de la API móvil (más estable que el HTML).
- **Disciplina obligatoria:** respetar rate limits, cachear de forma agresiva (no re-scrapear nunca lo ya descargado) y operar a bajo volumen. El scraping va contra los ToS de HLTV; para un proyecto académico personal de bajo volumen suele tolerarse, pero se opera con consideración hacia sus servidores.
- Evaluar datasets ya publicados (p. ej. Kaggle) para arrancar sin scraping inicial.

`[ABIERTO: elegir la herramienta concreta de scraping y dejarla fijada para reproducibilidad.]`

### 4.2. Alcance temporal y rupturas de régimen

El corte natural de inicio es el **lanzamiento de CS2 (octubre de 2023)**, por una ruptura de régimen dura: el cambio a **MR12** alteró la economía, el número de rondas y el valor de "salvar". Mezclar CS:GO con CS2 sin marcarlo contamina el dataset.

Rupturas de régimen a tener en cuenta:

| Evento | Fecha | Efecto |
|---|---|---|
| CS:GO → CS2 (MR12) | oct. 2023 | Cambia la dinámica del juego entero |
| HLTV Rating 2.1 | oct. 2024 | Recalibra medias a 1.00; ajusta CS2 (asistencias 41→26 de daño) |
| HLTV Rating 3.0 | ago. 2025 | Mayor revisión desde 2017; añade ajuste económico y *Round Swing* |
| Reajuste Rating 3.0 | oct. 2025 | Devuelve peso a kills y daño |
| Rotaciones del pool de mapas | periódicas (Valve) | Cambian qué mapas se juegan |

**Decisión recomendada:** entrenar **solo con la era CS2** para el modelo principal. Incluir CS:GO solo como estudio de ablación, siempre con un flag de era explícito.

### 4.3. Consistencia de los ratings

La fórmula de rating de HLTV ha cambiado (2.0 → 2.1 → 3.0). La buena noticia: HLTV **recalculó todos los partidos históricos** bajo la fórmula 3.0, así que scrapear *ahora* devuelve ratings homogéneos. Aun así, por robustez:

- Apoyarse también en **stats crudas** (ADR, KAST, KPR, DPR), cuyas definiciones son estables.
- **Nunca** mezclar ratings extraídos en fechas distintas bajo fórmulas distintas.

### 4.4. Modelo de datos (estado implementado en SQLite)

**Estado actual (2026-07-07):** `BBDD/cs2.db` es una fuente de verdad viva. `BBDD/build_db.py` crea/migra el esquema y siembra el historico solo si `matches` esta vacia; `BBDD/ingest.py` aplica cada run con upserts incrementales; `BBDD/export_master_json.py` genera el `master/matches.json` de compatibilidad desde SQLite. La ejecucion normal no borra ni reconstruye desde cero, para no perder odds, stats individuales ni snapshots que puedan cambiar en HLTV o en las casas.

Definición: `BBDD/cs2_prediction_schema.sql`. Constructor reproducible:
`BBDD/build_db.py`. BBDD física: `BBDD/cs2.db`, con backups locales en
`BBDD/backups/`. El espejo adicional está desactivado por defecto para no
duplicar gigabytes en el mismo disco; se habilita solo apuntando
`CS2_BACKUP_MIRROR_DIR` a otro disco o almacenamiento sincronizado.

La BBDD separa tres cosas que no deben mezclarse:

1. **Archivo raw append-only:** lo que HLTV/odds devolvió exactamente en cada run, con timestamp y fichero fuente.
2. **Hechos normalizados:** partidos, mapas, veto, alineaciones, odds, rosters y box scores.
3. **Features/predicciones point-in-time:** lo calculado antes del partido, auditable y reconstruible.

```mermaid
flowchart TD
    subgraph SCRAPE["Scraping online / PIPELINE"]
        RUN["run_manifest + upcoming + match_snapshot"]
        HTML["raw_html"]
        ANA["match_analytics"]
        PROF["team_profile + player_compare_stats"]
        RANK["team_ranking"]
        ASSET["match_assets: veto, lineups, mapstats"]
        CTX["match_context: stage, LAN/online, incentivo"]
    end

    subgraph RAW["Archivo raw append-only"]
        RR["raw_results"]
        RS["raw_snapshots<br/>JSON/HTML exacto por run"]
        PSS["player_stat_snapshots<br/>stats individuales capturadas"]
        TRS["team_ranking_snapshots"]
    end

    subgraph CORE["Core normalizado"]
        TE["teams"]
        PL["players"]
        EV["events"]
        RO["team_rosters<br/>valid_from / valid_to"]
        MA["matches<br/>status + data_tier"]
        FS["fetch_state<br/>TTL por entidad"]
        IR["ingest_runs<br/>auditoria"]
        OD["odds<br/>opening / live / closing"]
        MP["maps"]
        VE["veto"]
        LU["match_lineups"]
        MPS["map_player_stats"]
        SIDE["map_player_side_stats<br/>total / CT / T"]
    end

    subgraph MART["Feature mart y auditoría"]
        RH["ratings_history<br/>Glicko antes del partido"]
        MF["match_features<br/>matriz ML"]
        PR["predictions<br/>modelo, decision, stake, flags, JSON"]
    end

    RUN --> RS
    HTML --> RS
    ANA --> RS
    PROF --> RS
    PROF --> PSS
    RANK --> RS
    RANK --> TRS
    ASSET --> RS
    RR --> MA
    RS --> TE
    RS --> PL
    RS --> RO
    RS --> OD
    RS --> FS
    RUN --> IR
    ASSET --> VE
    ASSET --> CTX
    CTX --> MA
    ASSET --> LU
    ASSET --> MP
    MP --> MPS
    MPS --> SIDE
    TE --> MA
    EV --> MA
    MA --> RH
    RO --> MF
    RH --> MF
    MF --> PR
    OD --> PR
```

Tablas clave y qué preservan:

- `matches`: separa `status` (`scheduled`, `pending_result`, `completed`) y `data_tier` (`historical_seed`, `prematch_captured`, `completed`). Los campos PRE-MATCH (`prematch_captured_at_utc`, odds, contexto, snapshots y flags de cobertura) no se pisan con consultas futuras; cuando llega el resultado solo se rellena el bloque RESULT (`winner_team_id`, `score_t1`, `score_t2`, `result_filled_at_utc`).
- `fetch_state`: TTL compartido por entidad para evitar rescrapear lo fresco: perfiles de equipo 7 dias, stats de jugador 3 dias, rankings 7 dias y assets/Analytics casi permanentes si ya se capturaron correctamente. Para `player_stats` la decision es individual: un jugador fresco no se vuelve a pedir porque otro miembro del roster sea nuevo o este vencido. Estados `blocked`/`error` se reintentan pronto y `not_found` significa que HLTV respondio correctamente pero no habia una muestra util. `PIPELINE/start.py` consulta esta tabla antes de pedir perfiles, stats, rankings, assets o Analytics.
- `ingest_runs`: auditoria de cada run aplicado a SQLite: `run_id`, inicio/fin, estado, filas upserted y requests saltadas por frescura.
- `raw_snapshots`: snapshots exactos de `run_manifest`, `upcoming_matches`, `match_snapshot`, `team_profile`, `player_compare_stats`, `predictions_enriched`, `match_assets`, `match_analytics`, `team_ranking`, `raw_html` y `data_quality_report`. Es el seguro contra que HLTV cambie o borre información después.
- `matches.stage`, `matches.environment`, `matches.stage_detail`, `matches.incentive_label`, `matches.high_stakes`, `matches.opening_match`, `matches.winner_advances`, `matches.loser_eliminated`, `matches.bracket` y `matches.context_json`: contexto parseado del bloque `Maps` de HLTV. Guarda LAN/online, fase (group/swiss/playoff/etc.), detalle textual ("Winner advances...", "elimination match", Swiss record), flags consultables y el JSON completo para auditoría.
- `player_stat_snapshots`: stats individuales capturadas en el run pre-partido desde `/stats/players/compare` y el perfil `/stats/players/{id}/{slug}`: jugador HLTV, `time_filter` elegido de forma adaptativa (`past3months` si tiene muestra suficiente; si no `past6months`; si no `past12months`), rating, KPR, DPR, APR, KAST, Impact, ADR, Round Swing, multi-kill rating, AWP KPR, HS %, opening KPR/DPR, flash assists y `payload_json` completo. Esto conserva las stats **tal como estaban disponibles en ese momento**.
- `map_player_stats` y `map_player_side_stats`: box score real por jugador/mapa y por lado `total`/`ct`/`t` cuando el partido ya tiene assets/mapstats. Es la fuente granular para recalcular forma L5/L10/L20 sin consultar páginas históricas cambiantes.
- `team_rosters`: pertenencia temporal real por jugador/equipo (`valid_from`, `valid_to`, `source_run_id`, `source_signature`). Un jugador no "es" de un equipo: estuvo en él durante un intervalo observado.
- `prematch_lineup_snapshots`: announced pre-match lineup with capture time, player, team, and stand-in marker. It is intentionally separate from `match_lineups`, the actual lineup after mapstats.
- **Red flag de cambio de roster (90 dias):** `enrich_predictions.py` exige cinco
  IDs de jugador en la alineacion anunciada antes del inicio y los compara con
  los cinco IDs de la alineacion real del ultimo partido anterior del mismo
  equipo dentro de 90 dias. Si difieren, genera `ROSTER_CHANGE_90D` en nivel
  `danger` y conserva en `controls_json.roster_change_90d` el partido usado,
  jugadores que entran/salen, muestra comparada y alineacion modal. Historicos
  incompletos, partidos futuros, snapshots post-inicio y equipos sin ID se
  descartan: quedan como cobertura desconocida y nunca como cambio inventado.
- `match_analytics_snapshots`, `match_analytics_map_stats`, and `match_analytics_map_handicap`: point-in-time Analytics Center data: map first pick/ban, win rate and sample; BO3 distribution, overtime, round margins, core/stand-in signals, and event metadata. Raw HTML and `payload_json` remain available for audit.
- `events`: stores HLTV event ID, prize pool, teams competing, and source provenance. A scheduled match can update corrected participants, time, or stage without rewriting any historical snapshot.
- `odds`: cuotas por bookmaker y timestamp, separando `opening`, `live` y `closing`. La apertura sirve para benchmark y EV; el cierre se guarda para auditoría, no como feature pre-partido.
- `predictions`: congela la probabilidad del artefacto (`prob_team1`), la probabilidad operativa (`decision_prob_team1`), régimen/arquitectura (`prediction_regime`, `prediction_architecture`), trazabilidad de apertura recuperada (`opening_odds_recovered`, `opening_odds_captured_at_utc`), fiabilidad, peso de mercado, política, favorito, cuota mínima de value (`decision_min_value_odds`, `team1_min_value_odds`, `team2_min_value_odds`), incertidumbre del estimado (`ensemble_disagreement`, `estimate_band_half_width`, `estimate_confidence_level`, `estimate_history_coverage`), Best Opportunity point-in-time (`opportunity_score`, elegibilidad, ranking global/diario, `is_best_opportunity` y versión de política), contexto normalizado (`context_environment`, `context_stage`, `context_stage_detail`, `context_incentive_label`, `context_high_stakes`, `context_winner_advances`, `context_loser_eliminated`, `context_opening_match`, `context_bracket`, `context_json`) y los JSON completos de `prediction`, `features`, `odds`, `staking`, `controls`, `flags`, `data_quality` y `rosters`.

Best Opportunity usa la política `best_opportunity_v2_confirmed_lineups`: además
de los umbrales de confianza y fiabilidad, ambos lados deben tener una alineación
pre-match de cinco IDs únicos confirmada por la página del partido. La cobertura
individual siempre divide entre cinco, aunque el perfil de equipo o HLTV solo
devuelva una parte. `PIPELINE/enrich_predictions.py` genera por run
`roster_integrity_report.json` y detiene la publicación ante cualquier diferencia
entre un 5v5 anunciado y el roster renderizado.

La pestaña Modelo consume `MODEL/results/predictions_walkforward.csv`, generado
en cada entrenamiento con una predicción causal por modelo y partido. La web
filtra `nested_model_policy`, deduplica por `match_id` y construye la evolución
temporal de la accuracy real: diaria para 7 días/1 mes, semanal para 3/6 meses y
mensual para 1 año. Nunca usa el refit final para predecir partidos con los que
ya fue entrenado. `favorite_accuracy_bands.json` se conserva para la auditoría
de calibración por confianza. En la BBDD viva, la misma comprobación es posible
uniendo `predictions.prob_team1` con `matches.winner_team_id`, conservando una
sola versión point-in-time por partido cuando se evalúe producción.

Flujo operativo de la BBDD viva:

1. `start.ps1` inicializa/migra `BBDD/cs2.db` con `BBDD/build_db.py` antes del scrapeo.
2. `PIPELINE/start.py` consulta `matches`/`fetch_state` y salta lo ya resuelto: `/results` se descarga solo para IDs `pending_result` conocidos por SQLite y solo pagina a offsets antiguos si esos IDs no aparecen en las páginas previas; perfiles, stats, rankings, Analytics y assets se gatean por frescura/cobertura. En stats de jugador se materializa la cache fresca para consumo del predictor y se crea una lista online solo con IDs nuevos, vencidos o incompletos.
3. Si una captura falla por bloqueo, el scraper marca la entidad en `fetch_state` como `blocked`/`error` con reintento corto, conserva lo bueno ya guardado y la siguiente ejecución vuelve a intentar solo los huecos.
4. Justo después del scrape, `BBDD/ingest.py --run-dir <run>` upserta hechos, odds, snapshots raw, rankings y assets normalizados (`maps`, `veto`, `match_lineups`, `map_player_stats`, `map_player_side_stats`). Así `.\start.ps1 -Retrain` entrena ya con lo recién capturado.
5. Tras `enrich_predictions.py`, `BBDD/ingest.py` se ejecuta otra vez para congelar predicciones/staking y `BBDD/export_master_json.py` genera `PIPELINE/master/matches.json` desde SQLite para compatibilidad con componentes que todavía consumen JSON.
6. `MODEL/train.py` entrena desde `BBDD/cs2.db` por defecto. `--raw <results_all.json>` queda como modo legacy/debug.

Última BBDD viva validada (`2026-07-10`, tras consolidacion fisica): 9.527 partidos (`9.492` completados, `35` programados), 958 equipos, 753 jugadores, 647 filas de odds, 569 ventanas de roster, 5.127 snapshots de stats de jugador, 3.561 filas `map_player_stats`, 10.683 filas `map_player_side_stats`, 10.535 snapshots de ranking, 4.098 snapshots raw, 98 partidos con contexto HLTV, 153 con box score, 157 con veto, 90 predicciones persistentes y `0` filas con fuga temporal detectable (`match_features.data_up_to_utc > matches.datetime_utc`). La auditoria `BBDD/deduplicate_matches.py` devuelve `duplicate_pairs=0` y `PRAGMA foreign_key_check` devuelve cero filas.

Estado de integridad validado el `2026-07-27`: 9.799 partidos, todos con
`hltv_match_id` estable y sin duplicados fisicos. La reparacion recupero 9.327
IDs historicos, arreglo 9 enlaces y 6 favoritos de predicciones, elimino 2
equipos provisionales y consolido 1 partido duplicado tras corregir su formato
imposible BO3 13-11 a BO1. El mart resultante contiene 19.450 estados
point-in-time y el loader obtiene 9.725 series entrenables. El reentreno limpio
evalua 7.289 predicciones OOS con `nested_model_policy`: 64,38% accuracy y log
loss 0,6309. El artefacto promovido tiene SHA-256 `92772a32a58ddf1f`.

`BBDD/repair_integrity.py` aplica esta politica de forma transaccional e
idempotente y crea backup antes de cambiar la BBDD. Backfillea identidad desde
la evidencia raw/master, repara referencias y purga filas provisionales que no
representan un partido confirmado. No considera duplicado un encuentro solo
por compartir fecha/equipos/marcador: si la fuente conserva IDs HLTV distintos,
se preservan ambos como doble enfrentamiento legitimo. El health gate exige
`duplicate_pairs=0`, `PRAGMA foreign_key_check=0`, cero IDs ausentes, cero
participantes provisionales y cero predicciones huerfanas o incoherentes.

Regla de oro: si HLTV no exponía un dato en el momento del scrapeo, la BBDD no lo inventa. Pero si lo exponía y el pipeline lo capturó, queda guardado físicamente en SQLite y/o en `payload_json` raw para poder reparsearlo más adelante sin volver a depender de la página actual.

Política de campos faltantes:

- Si falta por fallo técnico recuperable (`403`, timeout, HTML parcial, analytics/assets no descargados), `start.ps1` debe reintentarlo en siguientes ejecuciones y conservar el estado previo. No se borra un dato bueno por una captura incompleta posterior.
- Si falta porque HLTV no lo exponía antes del partido (por ejemplo odds no publicadas, Analytics inexistente, veto real no disponible hasta jugarse), se guarda como `NULL`/`unknown` y se mantiene un indicador de cobertura. No se inventa ni se copia desde una consulta futura para entrenar.
- Para entrenamiento general no se eliminan filas completas solo por tener campos secundarios ausentes: se usan modelos tolerantes a missingness, valores neutrales solo cuando tienen sentido y flags de cobertura. Eliminar filas se reserva para campos core imposibles (`team`, fecha, resultado, formato) o experimentos de muestra completa.
- Antes de activar una nueva familia de features en producción se exige muestra point-in-time suficiente y validación walk-forward. Mientras tanto se guarda y se muestra para auditoría.

### 4.5. Scraping avanzado en HLTV

HLTV no ofrece API pública y protege la web con Cloudflare, que devuelve `403`/captcha ante peticiones automatizadas. El scraping serio se organiza en cuatro decisiones: **qué herramienta**, **cómo evadir el bloqueo**, **cómo no abusar** y **cómo cachear**.

#### 4.5.1. Herramientas (de menos a más coste de infraestructura)

| Herramienta | Lenguaje | Notas |
|---|---|---|
| `gigobyte/hltv` | Node.js | API no oficial más madura. Métodos documentados con su nº de requests: `getMatches`, `getResults` (filtros `eventIds`, `bestOfX`), `getMatchStats`, `getMatchMapStats`, `getTeamRanking` (por fecha), `getTeamStats`, `getPlayer`, `getPlayerStats`, `getEvent`, `getPastEvents`. Advierte expresamente de ban por Cloudflare si se abusa. |
| `hltv-async-api` (akimerslys) | Python (async) | Rotación de proxies y de user-agent, `max_delay`/`max_retries`, backoff ante 403. Cubre `team_map` stats, map stats, `get_match_stats`, `get_demo_id`, `get_best_players`. ~5× más rápido en su versión async. |
| `jparedesDS/hltv-scraper` | Python | Basado en `undetected-chromedriver`. Extrae stats de jugador (Impact, KAST, opening kills, tipos de kill por arma) y stats de equipo por mapa a CSV. Buen punto de partida para ver qué campos hay. |
| `fanden/hltv-match-api` | Java (Playwright) | Navegador headless + resolución de captcha (2Captcha) + sesiones de ~30 min para datos live. Más infraestructura. |
| Actores de Apify | (gestionado) | `paco_nassa~hltv-org-*` (partidos, ranking histórico con rosters), `getdataforme~hltv-teaminfo-scraper`. Devuelven JSON sin montar infra propia; coste por resultado. |

**Recomendación:** empezar con `gigobyte/hltv` (Node) o `hltv-async-api` (Python) según tu stack. Reservar el navegador headless + captcha (Playwright/2Captcha) solo para lo que las librerías no cubran. Considerar Apify si no quieres mantener la evasión tú mismo.

`[ABIERTO: fijar la herramienta principal y dejarla versionada para reproducibilidad.]`

#### 4.5.2. Evasión de bloqueo (en orden de agresividad)

1. **Backoff exponencial ante 403.** Reintentar aumentando el retardo (p. ej. 2 s → 4 s → 5 s) en lugar de martillear.
2. **User-agent realista y rotativo.** Un UA de Chrome de escritorio actual; rotar entre varios.
3. **Rate limiting estricto.** Retardo aleatorizado entre peticiones (p. ej. 3–10 s) para no parecer un bot.
4. **Rotación de proxies** si haces backfill histórico voluminoso (muchas páginas de `/stats`).
5. **Navegador headless indetectable** (`undetected-chromedriver`, Camoufox, Playwright con fingerprint TLS/JA3) cuando lo anterior no baste.
6. **Resolución de captcha** (2Captcha y similares) solo como último recurso, sobre todo para datos en vivo.

Implementación actual: `PIPELINE/start.py` usa sesión HTTP persistente, `cf_session.json`, pausa mínima entre peticiones, `Retry-After`, backoff largo, caché en memoria por URL durante el run, warm-up inicial de sesión, cuarentena temporal de URLs fallidas, presupuesto máximo de peticiones por run, circuit breaker global si aparecen bloqueos repetidos y fallback Scrapy. El navegador stealth tiene timeout acotado y, si queda bloqueado, `start.ps1` permite refrescar automáticamente la cookie con `SCRAPER/hltv-scraper-api/hltv_scraper/hltv_scraper/grab_cf.py`, que abre una ventana visible para obtener una nueva `cf_clearance`. La fase de stats de jugador corta ante el primer challenge, guarda su checkpoint, renueva la sesion y reintenta; no espera una cadena de backoffs antes de pedir una cookie nueva. Ademas, `/stats/players/compare` se parsea por columna de jugador y el perfil individual valida los campos core, evitando copiar valores del rival cuando HLTV renderiza una columna sin muestra. `start.ps1` favorece completitud sobre velocidad: delays por defecto más altos, timeouts amplios, variables `HLTV_*` conservadoras y transcript completo en `PIPELINE/logs/start_*.log`. Al final de cada run se guarda `fetch_diagnostics.json` con intentos HTTP, cache hits, bloqueos, errores, URLs problemáticas y segundos dormidos por cooldown.

#### 4.5.3. No abusar y cachear (obligatorio)

- **Calcula el presupuesto de requests antes de lanzar un backfill.** Métodos como `getResults` o `getMatchesStats` paginan y pueden disparar cientos de peticiones; las librerías documentan el coste por método justo para que puedas throttlear.
- **Cachea todo lo descargado en la capa raw** (§4.4) y no re-scrapees nunca lo ya obtenido. El histórico es inmutable: una vez bajado un partido terminado, no cambia.
- **Cachea también dentro del run.** Si odds, detalle, contexto y assets necesitan la misma URL en pocos minutos, el scraper reutiliza el HTML ya descargado en memoria (`HLTV_FETCH_CACHE_TTL`) para reducir peticiones repetidas.
- **Circuit breaker ante WAF/Cloudflare.** Si aparecen `403`, `429`, challenge HTML o errores equivalentes de forma consecutiva, el scraper aplica cooldown global (`HLTV_BLOCK_COOLDOWN_*`, `HLTV_CIRCUIT_BREAKER_SLEEP`) antes de seguir. En rutas `/stats/`, un `403` o challenge (excepto `429`) pasa inmediatamente al navegador stealth, visible por defecto, y, si sigue el bloqueo, permite una renovación interactiva de `cf_session`; si esta falla, la URL queda en cuarentena y el run continúa parcial, sin repetir diez esperas de veinte minutos. Los `429` y errores de servidor sin challenge mantienen el backoff y `Retry-After`.
- **Cuarentena por URL y presupuesto de requests.** Si una URL falla tras sus reintentos, se marca en cuarentena (`HLTV_URL_QUARANTINE_SECONDS`) para que otras fases no vuelvan a insistir dentro del mismo run. Además hay un techo de peticiones (`HLTV_MAX_HTTP_REQUESTS_PER_RUN`) y el scraper de `/stats/players/compare` corta en parcial si alcanza su presupuesto.
- **Warm-up y diagnóstico.** El run hace una petición inicial suave a HLTV para validar sesión y escribe diagnósticos de red en `fetch_diagnostics.json`. Si tarda mucho, la consola verbose muestra exactamente URL, intento, bloqueo y espera.
- **Backfill una vez, incremental después.** Una pasada inicial para el histórico de la era CS2; luego solo partidos nuevos (incremental) y refresco del ranking semanal.
- **Recuperación de huecos recientes.** Cada run reintenta partidos activos o de los últimos días que quedaron sin odds, detalle real o Betting Analytics por bloqueo/403. El dato recuperado se guarda con el timestamp del nuevo scrape; no se inventa ni se retrofecha.
- **Respeto a los ToS:** bajo volumen, ritmo humano, uso académico personal. El scraping va contra los términos de HLTV; se opera con consideración hacia sus servidores.

### 4.6. Dónde extraer las métricas especialmente útiles

HLTV organiza las estadísticas en páginas con **filtros de rango de fechas** en la URL. Ese filtro es la pieza clave para la recogida **point-in-time**: al construir features de un partido, consulta siempre con `endDate` = el día anterior al partido, **nunca** las stats "actuales" (eso sería fuga; §9).

#### 4.6.1. Mapa de páginas → datos

| Página / método | URL o método | Qué saca (especialmente útil) |
|---|---|---|
| Stats de equipo (rango) | `/stats/teams?startDate=&endDate=&rankingFilter=` | Rating de equipo, winrate, K/D agregados en una ventana temporal definida por ti |
| Stats de equipo por mapa | `/stats/teams/maps/{id}/{name}` | **Winrate por mapa** y por lado (CT/T): base de la arquitectura composicional del Bo3 (§7.3) |
| Stats de jugador (rango) | `/stats/players?startDate=&endDate=&rankingFilter=` | Rating, KAST, ADR, KPR, DPR, Impact/Round Swing, opening kills por jugador en ventana |
| Perfil individual avanzado | `/stats/players/individual/{id}/{name}` | Desglose fino: tipos de kill por arma, openers, multis |
| Clutch | `/stats/players/clutches/{id}/1on1/{name}` | 1vX, clutch points, 1v1 win% (rareza alta → ver caveat) |
| Página de partido | `/matches/{id}/...` | **Veto completo**, alineaciones reales (stand-ins), formato, evento |
| Box score por mapa | `getMatchMapStats` (lib) | **Stats por (jugador, mapa)**: la fuente granular para `map_player_stats` y para calcular tú mismo todos los agregados de forma consistente |
| Ranking de equipos | `/ranking/teams/...` (actual e histórico por fecha) | Puntos y posición; histórico por fecha = point-in-time del ranking |
| Evento | `getEvent` / página de evento | **Tier, prize pool**, equipos participantes, LAN/online |

> **Recomendación de robustez:** en lugar de fiarte de las ventanas pre-agregadas de HLTV (que dependen de su definición de rating del momento), descarga el **box score por mapa** (`getMatchMapStats`) a `map_player_stats` y **computa tú** las medias, la forma y el decay. Así controlas la consistencia temporal (§4.3) y el point-in-time de extremo a extremo.

#### 4.6.2. Métricas más predictivas y cómo tratarlas

Con Rating 3.0, las stats de jugador vienen **ajustadas por economía** y desglosadas en seis sub-ratings: **Kills, Damage, Survival, KAST, Multi-Kills y Round Swing**. El *Round Swing* es la métrica estrella: mide cuánto cambió un jugador la probabilidad de ganar la ronda, con conocimiento de mapa, lado y economía, repartiendo el crédito entre el killer, los que hacen daño, los flashes y los trade kills. Es la mejor señal individual de impacto disponible hoy.

Jerarquía de utilidad para predicción:

- **Alta señal, baja varianza (úsalas siempre):** Rating 3.0 y sus sub-ratings (sobre todo Round Swing), KAST, ADR, winrate por mapa y por lado, opening kills/deaths ratio. Agrégalas por equipo con media + máximo + mínimo + varianza (§6.3).
- **Señal de rol/estilo (útiles, pero describen tendencia, no calidad):** los **atributos** de HLTV (Firepower, Opening, Entrying, Trading, Clutching, Sniping, Utility). Sirven para caracterizar composición (¿hay AWP principal?, ¿dependencia de una estrella?), no como medida directa de "es mejor".
- **Alta varianza (trátalas con cuidado):** clutch / 1vX. Son raras y necesitan muestras grandes; HLTV mismo advierte que miden más la *oportunidad* de clutchear (rol) que la *habilidad*. Úsalas solo agregadas, con ventanas amplias y *shrinkage* (§6.6), nunca como feature dominante.

> **Trampa de muestra pequeña.** Muchas de estas métricas (clutch, atributos, eco-frags) son inestables con pocos partidos, justo lo habitual en tier bajo. Reglas: ventanas amplias con decay, *shrinkage* bayesiano hacia la media del tier/región, y el tamaño de muestra como feature explícita para que el modelo sepa de cuánta evidencia dispone.

---

## 5. Sistema de rating (la feature más importante)

### 5.1. Elección: Glicko-2 sobre Elo plano

Dado que se cubren todos los tiers (muchos equipos con pocos partidos y rosters volátiles), se usa **Glicko-2**, que modela la **incertidumbre del rating** mediante la *rating deviation* (RD) y la volatilidad (sigma). Ventajas frente a Elo plano:

- Distingue "1500 con 200 partidos" de "1500 con 3 partidos".
- La **RD crece automáticamente con la inactividad**: un equipo parado 6 meses obtiene mayor incertidumbre sin necesidad de descartarlo. Esto es, de hecho, la versión principiada de una "red flag" por datos viejos.

### 5.2. Cómo se usa

- Se calcula **cronológicamente** sobre el histórico, en **periodos de rating semanales** (alineado con la actualización del ranking de HLTV).
- Se guarda en `ratings_history` el estado *antes de cada partido*.
- Como features entran tanto el **rating** como su **RD** (incertidumbre) y la **diferencia A−B** de ratings.
- Se mantiene además como features los **puntos del ranking oficial de HLTV** (basados en resultados, fuerza del rival, tier del evento y decaimiento; actualizados semanalmente con ventana de 3 meses y más peso a lo reciente; LAN pesa más que online; los cambios de roster resetean parte de los puntos).

### 5.3. Challenger sensible al roster

El rating de organización anterior se conserva como campeón. En paralelo se
reconstruye un Glicko/Elo causal sensible al quinteto y se evalúa como challenger:

- Antes de cada partido compara el 5v5 anunciado (capturado estrictamente antes
  del inicio) con el último 5v5 real ya observado del equipo, dentro de 90 días.
- Un cambio de núcleo es, por defecto, sustituir al menos 2 de 5 jugadores. Con
  `k` jugadores mantenidos, conserva la fracción
  `f = 0,20 + 0,80 × k/5` del crédito respecto a 1500:
  `rating' = 1500 + f × (rating - 1500)`.
- La incertidumbre aumenta sin un reset brusco:
  `RD'² = f × RD² + (1-f) × 350²`. El suelo de crédito del 20 % conserva parte
  del historial de la organización incluso con una reconstrucción total.
- Si falta cualquiera de los dos quintetos completos, no hay descuento. La
  predicción siempre existe y el challenger coincide con el rating causal normal.
- Los parámetros están versionados en `MODEL/config.yaml`. El estado se emite
  antes de observar el resultado; la alineación real actual solo se incorpora en
  `observe()` para partidos posteriores.

`roster_glicko_cal` se calibra y mide en el mismo walk-forward temporal que los
demás candidatos. Solo puede llegar a producción si gana la selección causal y,
después, la puerta champion/challenger por log loss (Brier como desempate). El
beneficio esperado son menos palos gordos y mejor calibración tras reconstrucciones,
no un salto grande de accuracy media.

---

## 6. Features / métricas

Para cada partido se construye un vector que compara A vs B, usando **diferencias A−B** donde tenga sentido. Todas las features se calculan **point-in-time** (solo con datos anteriores al partido) y con **decaimiento temporal** (los partidos viejos pesan menos, sin corte duro).

### 6.1. Nivel equipo
- Rating Glicko-2 y su RD (incertidumbre).
- Puntos y ranking de HLTV.
- Forma reciente: winrate móvil de los últimos N mapas (p. ej. 10–20), con decay.
- Winrate **por mapa** del pool activo.
- Estabilidad de roster: días desde el último cambio; mapas jugados juntos por los 5 actuales.
- Flags de contexto: LAN vs online, tier del evento, fase (grupos/semi/final).

### 6.2. Head-to-head
- Récord H2H global y reciente; H2H por mapa.

### 6.3. Nivel jugador, agregado a equipo
Componentes de HLTV: ADR (daño medio por ronda), KAST (% de rondas con kill, asistencia, supervivencia o intercambio), Impact, y en Rating 3.0 el *Round Swing* (mira la win probability del equipo antes y después de cada kill). Aproximación pública de la fórmula clásica del rating:

```
Rating ≈ 0.0073·KAST + 0.3591·KPR − 0.5329·DPR + 0.2372·Impact + 0.0032·ADR + 0.1587
```

Features por equipo: Rating, KAST, ADR, KPR, DPR, Impact/Round Swing, opening kill ratio, dependencia del AWP, clutches. **Agregar con varias estadísticas, no solo la media:** media + máximo (la estrella) + mínimo + varianza. La dispersión del talento importa.

> **Por qué trabajar a nivel jugador.** Como los rosters rotan, construir features de jugador y agregarlas a los 5 actuales es más robusto que el historial de equipo. Un superequipo recién formado no tiene historial *de equipo*, pero sus 5 jugadores arrastran historiales individuales ricos. La forma del jugador viaja con él; la del equipo se evapora con cada cambio.

Estado implementado:

- Cada `start.ps1` captura `/stats/players/compare/{p1}/{slug1}/{p2}/{slug2}` para los jugadores detectados en rosters con **ventana adaptativa**: primero `past3months`; si algún jugador de la pareja no alcanza la muestra mínima (`10` mapas), prueba `past6months`; si sigue sin muestra, `past12months`. Se guarda por jugador la ventana más reciente suficientemente representativa; si ninguna llega al mínimo, se usa la ventana disponible con más mapas. El filtro de 9 meses no se usa porque no es un `timeFilter` estándar visible de HLTV; se prefiere no inventarlo.
- La página **Full comparison** aporta Round Swing, multi-kill, AWP, opening y flashes, pero no expone de manera fiable ADR, DPR e Impact para ambos jugadores. Por eso el mismo ciclo completa el snapshot seleccionado con `/stats/players/{id}/{slug}` en la ventana equivalente. Un snapshot solo queda `ok` si contiene Rating 3.0, KPR, DPR, APR, KAST, Impact y ADR; el caché de tres días no oculta un snapshot incompleto y lo repara en el siguiente ciclo.
- Después el proyecto agrega por equipo: media, máximo/estrella, mínimo/weak link, desviación, spread, star gap y weak-link gap. Eso mide el impacto de tener una estrella o un jugador muy por debajo sin diluirlo en una media simple.
- Estas métricas quedan en `features`, `rosters` y `player_stat_snapshots`. El join cronologico usa solo snapshots con `captured_at_utc <= datetime_utc`; `MODEL/train.py` activa automaticamente sus columnas al llegar a 200 partidos cerrados con cobertura de ambos equipos.
- Si HLTV no devuelve una métrica de forma inequívoca para ambos lados en el HTML parseable, se guarda el raw/payload y no se inventa el valor. Es preferible una cobertura parcial honesta a contaminar el modelo con números mal asignados.

### 6.3.1. Politica de comparacion de jugadores (actualizacion 2026-07-13)

La comparacion no reduce el roster a una media simple ni impone un coeficiente
fijo de "carry". Para cada equipo se escoge una sola ventana comun:
`past3months` si hay al menos cuatro jugadores y cinco mapas por jugador; si no,
`past6months`, despues `past12months`, y solo al final la ventana con mejor
cobertura disponible. Asi no se compara la forma de tres meses de un jugador con
un ano de otro compañero.

El vector entrenable por enfrentamiento usa diferencias A-B de:

- Rating medio, mejor jugador, media de los dos mejores, mediana, media de los
  dos peores y desviacion estandar. Esto separa estrella aislada, duo fuerte y
  profundidad/weak core.
- KPR, KAST, ADR e Impact medios, mas Round Swing y Opening KPR medios y de los
  dos mejores. `Round Swing` aporta valor contextual de ronda; no sustituye al
  resto de dimensiones de rendimiento.
- Cobertura, mapas totales y minimo de mapas por jugador. Solo se activa la
  familia cuando ambos lados tienen cobertura >=80% y al menos cinco mapas por
  jugador.

`min`, `spread`, `star_gap` y `weak_link_gap` se mantienen en el snapshot y la
web como diagnostico, pero no entran al vector: son combinaciones algebraicas de
media/maximo/minimo y volverian inestable la estimacion lineal sin aportar
informacion nueva. La literatura de agregacion de habilidades no sostiene que
el maximo sea siempre la mejor regla en CS; por eso se conserva toda la forma de
la distribucion y se deja al modelo aprender el peso de una estrella frente a la
profundidad en validacion walk-forward.

Los snapshots de jugador solo pasan de observabilidad a modelo al alcanzar 200
partidos cerrados point-in-time. Antes se guardan, se muestran y se auditan, pero
no se reentrena produccion con una muestra insuficiente.

Referencias de diseno: Dehpanah et al. (2021), *Evaluating Team Skill
Aggregation in Online Competitive Games*; PNX et al. (2020), *Valuing Player
Actions in Counter-Strike: Global Offensive*; y la documentacion de HLTV Rating
3.0 sobre sus subratings y Round Swing.

### 6.4. Contexto de mapas / veto
- Pool activo, mapas que cada equipo suele pickear/banear, ventaja esperada por mapa.

### 6.4.1. HLTV Betting Analytics

El scraper captura Analytics Center como JSON/HTML raw y filas SQLite normalizadas. La primera foto pre-match y la cuota de apertura son inmutables; mientras un partido esta programado, cuotas, Analytics, contexto y alineacion anunciada se refrescan cada 6 horas (cada 2 horas en las ultimas 24 horas), anexando cada observacion. El entrenamiento lee exclusivamente la ultima foto completa cuya hora sea anterior al inicio: nunca el roster real ni un Analytics posterior. Analytics basico entra automaticamente con 120 resultados cerrados point-in-time; su familia extendida (core/stand-ins, distribucion BO3 y margenes de rondas) y las alineaciones anunciadas 5v5 entran con 200. Prize pool y numero de equipos del evento quedan guardados desde ya como contexto de calibracion y se activan automaticamente con 300 casos point-in-time.

La pestaña **Betting Analytics** de cada partido se captura en todos los scrapeos futuros cuando HLTV la expone. Se guarda como snapshot bruto (`analytics/*.json`) y se archiva en SQLite como `raw_snapshots(kind='match_analytics')`.

Política de uso:

- **Captura/persistencia:** siempre que exista la página, se guarda con `captured_at` y queda asociada al partido. No se recalcula desde cero perdiendo el estado anterior.
- **Entrenamiento:** sus columnas `analytics_*` no entran al Modelo A hasta tener muestra suficiente de partidos cerrados con Analytics capturado **antes o el mismo día del partido**. Umbral actual: **120 partidos cerrados point-in-time**.
- **Activación automática:** cuando el dataset supera ese umbral, `MODEL/train.py` añade automáticamente las features `analytics_*` al vector de entrenamiento y lo deja registrado en `artifact.metadata["feature_policies"]["analytics"]`.
- **Mientras no hay muestra:** se siguen guardando y mostrando para auditoría, pero quedan excluidas del entrenamiento para evitar overfitting por muestra pequeña.
- **Anti-fuga:** si un snapshot de Analytics aparece con `captured_at` posterior a la fecha del partido, se descarta para entrenamiento aunque siga archivado como dato bruto.

### 6.4.2. Contexto competitivo desde la caja `Maps`

HLTV muestra en la caja `Maps` datos como `Best of 3 (LAN)`, `Swiss round 4 (teams with a 2-1 record). Winner advances to playoffs.`, `elimination match`, `3rd place decider` o notas de sustitucion. El pipeline parsea ese texto en `match_context` y lo guarda en snapshots, assets, SQLite y web.

Uso actual:

- **Persistencia:** `environment` (`lan`/`online`), `stage`, `stage_detail`, `winner_advances`, `loser_eliminated`, `swiss_record`, `incentive_label` y `substitution_notes` quedan archivados con el raw original.
- **Web/fiabilidad:** se muestran como contexto y flags (`LAN_MATCH`, `WINNER_ADVANCES`, `ELIMINATION_MATCH`, `SUBSTITUTION_NOTE`). Ayudan a leer el riesgo sin modificar a mano la probabilidad pura.
- **Diagnostico automatico:** `MODEL/analyze_context_calibration.py` genera `MODEL/results/CONTEXT_CALIBRATION.md` y `MODEL/results/context_calibration.json` con log loss, Brier, ECE y accuracy por `environment`, `stage`, `high_stakes`, `winner_advances`, `loser_eliminated` e `incentive_label`. `start.ps1` lo ejecuta despues de enriquecer predicciones. No se escriben copias en la raiz para mantenerla limpia.
- **Entrenamiento/calibracion:** las columnas `context_*` ya se calculan y quedan listas para inferencia. `MODEL/train.py` las activa en el Modelo A cuando hay al menos **200 partidos cerrados con contexto point-in-time** y cobertura minima de **50 LAN + 50 online**. El resultado global se mide por log loss/Brier walk-forward y el estado queda en `artifact.metadata["feature_policies"]["context"]`.
- **Limite epistemologico:** "winner advances" o "elimination match" si describe incentivo competitivo observable; no permite inferir motivacion interna, scrims, problemas privados o liquidez real del mercado.

### 6.5. Cuotas (odds)

**Cuota de apertura** = la primera cuota pre-partido que el pipeline consigue capturar para una casa/mercado. Es importante porque representa una estimación temprana del mercado antes de que entre información tardía: cambios de roster, stand-ins, mapas filtrados, lesiones, volumen de apuestas o movimientos de línea. **Cuota de cierre** = la última cuota antes de empezar; suele ser más eficiente, pero puede contener información que no estaba disponible cuando se habría hecho la predicción. **Cuota live** = cuota durante el partido; se guarda para auditoría, no para el modelo pre-partido.

En la BBDD se guardan todas con `captured_at_utc`, `bookmaker`, `market_type`, odds decimales, probabilidad implícita normalizada y overround. Para entrenamiento/evaluación pre-partido, la apertura es el benchmark legítimo; para staking se usa la cuota disponible capturada en el snapshot operativo, siempre con timestamp.

Se tratan como **dato predictivo y benchmark de decisión**, manteniendo una rama
estadística dedicada que no depende de ellas. Razones:

- Es probablemente la feature individual más predictiva, pero **canibaliza** el resto (el modelo se convierte en una copia con ruido de la casa) y vacía el interés académico ("¿qué stats importan?").
- Hay riesgo de fuga encubierta: la cuota de **cierre** incorpora información de último minuto (alineaciones, bajas) y refleja el consenso final del mercado. **Usar solo cuotas de apertura**, con timestamp estricto.
- Las casas no son verdad absoluta: incorporan margen (*overround*), sesgos de mercado, límites de liquidez y comportamiento de apostadores. La literatura de betting market efficiency recomienda usar probabilidades implícitas **normalizadas** como benchmark, no copiar odds sin crítica.

**Decisión de diseño:** mantener Model A (solo stats) y el benchmark Model B,
pero dejar que la puerta compare dos sistemas productivos: (A) router con
primario stats+opening odds y reserva dedicada sin odds; (B) LightGBM único con
odds `NaN` nativo e indicador `odds_available`. Umbral actual configurable:
**mínimo 120 partidos cerrados con odds causales**. Véase
`DOCS/ODDS_ARCHITECTURES.md`.

El peso de cada feature (incluida la cuota) no se decide a mano: se cuantifica con **SHAP** (cubre la prioridad de interpretabilidad).

#### 6.5.1. Separación modelo vs decisión operativa

El proyecto separa tres conceptos que no deben mezclarse:

| Capa | Campo | Usa odds | Propósito |
|---|---|---:|---|
| Modelo de producción | `model_prob_team1` / `prob_team1` | Según `prediction_regime` | Mejor arquitectura validada; siempre predice |
| Mercado normalizado | `odds_prob_team1` | Sí | Benchmark, cálculo de EV y referencia externa |
| Decisión operativa | `decision_prob_team1` | A veces | Ranking, stake y gestión de riesgo |

Reglas:

- El artefacto productivo puede incorporar opening odds si la puerta lo aprueba;
  `prediction_regime` declara la ruta real. Model A sigue midiéndose por separado
  para saber qué aprende la rama puramente estadística.
- Para apostar o simular staking, las odds son obligatorias: sin cuota no existe EV ni Kelly bien definido.
- En todos los backtests de cartera, el lado queda fijado por el favorito puro del modelo (`p_team1 >= 0,5` elige team1; en caso contrario team2). Las odds pueden descartar ese favorito si no tiene EV positivo y dimensionar el stake, pero nunca invertir la selección para apostar al equipo al que el modelo asigna menos del 50%.
- La decisión operativa no vuelve a mezclar el mercado si ya está embebido en el
  modelo; evita contar la misma señal dos veces.
- Mientras haya poca muestra con odds, el peso del mercado es dinámico y conservador: mayor si la fiabilidad interna es baja; menor si hay buena historia, baja RD y datos completos.
- Cuando haya suficiente histórico de predicciones cerradas con odds, el peso modelo/mercado se aprende con validación **walk-forward expansiva**, eligiendo por log loss/Brier, no por una regla fija.

Estado implementado (julio 2026):

- Si no hay odds: `decision_prob_team1` = probabilidad del modelo encogida hacia 50/50 según `reliability_score`.
- Si hay odds y aún no hay muestra suficiente: `decision_prob_team1` mezcla modelo y mercado con un prior dinámico conservador. Rango actual de peso mercado: **5 %–30 %**, penalizado si hay pocas casas, bajo consenso o drift fuerte.
- Si hay al menos **120 partidos cerrados con odds point-in-time**: se comparan
  router y modelo mixto por log loss/Brier sobre el mismo soporte temporal; el
  blend histórico se conserva como benchmark.
- Se persisten aditivamente `prediction_regime`, `prediction_architecture`,
  `opening_odds_recovered` y `opening_odds_captured_at_utc`, además de los JSON y
  controles existentes, tanto en `predictions` como en el ledger sancionado.
- La cuota mínima de value se guarda como `team1_min_value_odds`, `team2_min_value_odds` y `decision_min_value_odds`. Fórmula: si la probabilidad operativa de un lado es `p`, la cuota decimal mínima para EV positivo empieza por encima de `1 / p`. Por eso un equipo puede tener 80 % de probabilidad de ganar y aun así no ser apuesta si la cuota actual está por debajo de ese umbral.

### 6.5.2. Activacion automatica de extended features

`MODEL/train.py` reconstruye siempre cada familia point-in-time. El umbral de
cobertura es solo la primera condicion: dentro de cada fold externo se reserva
un holdout temporal usando exclusivamente el pasado de ese fold, y la familia
entra unicamente si reduce el log loss al menos lo configurado en
`feature_selection.min_log_loss_gain`. No existe switch manual. El detalle por
fold, ganancias y motivos de exclusion queda en
`MODEL/results/fold_local_feature_selection.json`,
`artifact.metadata["feature_policies"]` y `MODEL/results/REPORT.md`.

Estado medido en `BBDD/cs2.db` el 2026-07-10 tras deduplicar semilla+HLTV (9.492 series entrenables):

| Familia | Cobertura | Umbral | Estado al proximo reentreno |
|---|---:|---:|---|
| Historial dentro del evento | 6.017 | 200 | ON automatico |
| Box score/mapas/lados/rating L5-L20 | 20 | 200 | OFF, acumulando |
| HLTV Betting Analytics | 100 | 120 | OFF, acumulando |
| Contexto LAN/online/fase | 65 (7 LAN, 58 online) | 200 y 50 por entorno | OFF, falta LAN y total |
| Stats individuales point-in-time | 88 | 200 | OFF, acumulando |
| Ranking HLTV/Valve point-in-time | 86 | 200 | OFF, acumulando |
| Estabilidad de roster point-in-time | 61 | 200 | OFF, acumulando |
| Model B con odds de apertura | 98 partidos unicos | 120 | OFF; evaluacion separada, nunca contamina el Modelo A puro |

La reparacion de integridad del 2026-07-27 corrigio 44 resultados con
participantes provisionales, fusiono 182 identidades inequivocas y reconstruyo
19.452 filas de ratings/features. Los historicos sin hora se marcan
`datetime_precision='date_only'` y todos los partidos del mismo dia emiten sus
features antes de actualizar el estado, evitando inventar un orden intradia.
`BBDD/repair_integrity.py` es idempotente y `start.ps1` lo ejecuta antes del
health gate de datos.

Un partido online solo entra en las tablas normalizadas cuando los dos
participantes tienen nombre no provisional e ID HLTV estable. Si la pagina aun
contiene `TBD`, `TBA`, `winner/loser` o falta un ID, la respuesta se conserva en
`raw_snapshots` para recuperarla en el siguiente run, pero no crea partido,
rating, feature ni prediccion. `BBDD/repair_integrity.py` purga cualquier caso
normalizado heredado sin borrar ese raw, y el gate
`live_matches_have_two_confirmed_teams` bloquea la promocion si reaparece uno.
Las fotos PRE-MATCH son append-only y fechadas: apertura, snapshots de jugador,
ranking, Analytics, contexto y alineacion anunciada se seleccionan para entrenar
solo cuando `captured_at_utc <= datetime_utc`; el RESULT posterior rellena
ganador/marcador sin reescribir la evidencia previa.

Formato BO1/BO3/BO5 y fatiga de calendario (`activity_2/7/14/30`, recencia e inactividad) son features nucleares reconstruibles para todo el historico y ya estan activas. Travel geografico, veto futuro/composicional, parches, liquidez de mercado y sanciones no se convierten en ceros falsos: permanecen fuera hasta disponer de una fuente point-in-time y un constructor verificable.

El detalle extendido vive en `DOCS/extra_features.md`; este `PROJECT.md` es la fuente de verdad resumida.

### 6.6. Incertidumbre y volatilidad (la "red flag" bien hecha)
La intuición de "el modelo dice 70 % pero hay señales de upset" **no se implementa como override manual** sobre la probabilidad: eso rompe la calibración. La forma correcta es meter la señal como **feature** para que el modelo produzca directamente un 58 % calibrado:

- Varianza de resultados recientes.
- RD del rating Glicko-2.
- Dispersión de ratings de jugadores.
- Flags de tier/online.

Una **capa de flags** sí es legítima como **anotación de confianza dirigida al humano** (no modifica la probabilidad): "predicción de baja confianza" (datos escasos / RD alta), "alto riesgo de upset" (volatilidad extrema), "contexto de integridad". Estos flags deben **validarse** mirando la calibración por subgrupos; si un subgrupo está descalibrado, la solución es añadir la feature y reentrenar, no aplicar un parche manual.

Matiz importante: los flags y la fiabilidad **no cambian `model_prob_team1`**. Sí pueden cambiar `decision_prob_team1`, que no es la probabilidad académica del modelo sino una capa operativa para ranking/stake. Esa capa hace *shrinkage* hacia 50/50 cuando los datos son pobres, de forma explícita y persistida. Así se evita presentar un 80 % con baja historia/map pool/roster como si tuviera la misma calidad que un 80 % con datos abundantes.

### 6.7. Integridad competitiva (etiquetas envenenadas)
El riesgo de amaño se concentra en tiers bajos (la industria estima que la práctica totalidad del riesgo de integridad viene de eventos de bajo tier). El patrón típico —perder mal el mapa 1 a propósito para inflar cuotas y ganar 2 y 3— está **diseñado para parecer un upset normal**, por lo que **no es predecible pre-partido desde stats**; se detecta a posteriori por organismos de integridad (ESIC, IBIA) cruzando patrones de apuestas con revisión de gameplay.

Tratamiento en el proyecto:
- Mantener la tabla `sanctions` con **sanciones oficiales publicadas** (ESIC/IBIA).
- **Limpiar del entrenamiento** los partidos de equipos/jugadores sancionados por amaño en el periodo afectado (son etiquetas envenenadas).
- Usar la lista solo como **flag de contexto**, nunca como ajuste de probabilidad.
- **Restricción ética:** no etiquetar a nadie como amañador por inferencia propia; ceñirse a sanciones oficiales (evita difamación).

---

## 7. Modelado

### 7.1. Pipeline de modelos

```mermaid
flowchart TD
    H["Histórico de partidos<br/>(hechos inmutables)"] --> R["Glicko-2 cronológico<br/>rating + RD + sigma"]
    R --> P["Probabilidad a priori<br/>(curva logística)"]
    H --> FE["Feature engineering<br/>point-in-time + decay"]
    P --> FE
    FE --> M["LightGBM / XGBoost<br/>objetivo: log loss"]
    M --> CAL["Calibración isotónica"]
    CAL --> OUT["Probabilidad de mapa"]
    OUT --> BO3["Agregación Bo3<br/>P(serie) = P(2-0)+P(2-1)"]
    BO3 --> FINAL["Probabilidad de serie"]
```

### 7.2. Modelos y baselines

- **Modelo principal:** gradient boosting — LightGBM o XGBoost; CatBoost si se quiere meter categóricas (equipo/mapa/evento) sin codificar a mano. Maneja NaN de forma nativa (clave con tiers bajos y datos incompletos).
- **Baseline de rating:** Glicko-2 + regresión logística (interpretable, calibrado).
- **Baselines obligatorios para comparar:**
  1. "Gana el de mayor ranking HLTV".
  2. Glicko-2 + logística.
  3. Probabilidad implícita de la **cuota de apertura** (el benchmark exigente).
- **Extensión avanzada (segunda iteración):** modelo **Bradley-Terry / jerárquico bayesiano** de fuerza latente de equipos; más principiado para comparaciones por pares y da probabilidades calibradas de forma natural.

> Las redes neuronales no se priorizan: en datos tabulares de este dominio no superan a XGBoost y son menos interpretables y más caras.

### 7.3. Arquitectura del Bo3

Un Bo3 no es una moneda: es el primero a 2 mapas, y los mapas dependen del veto. Dos arquitecturas:

- **Directa:** features de equipo → ganador de la serie. Buen baseline.
- **Composicional (recomendada para exactitud):** modelar P(ganar **cada mapa**) y combinar con la aritmética del Bo3. Suponiendo independencia entre mapas (aproximación razonable de primer orden) con probabilidades p₁, p₂, p₃ de los tres mapas más probables del veto:

  ```
  P(serie) = P(2-0) + P(2-1)
  ```

  Captura las asimetrías de pool, donde se decide la mayoría de los Bo3. El coste es modelar (o aproximar) el veto: como primer paso, asumir que se juegan los mapas con mayor probabilidad combinada de pick según el histórico; como segundo paso, un modelo aparte que prediga el veto. Más adelante se puede relajar la independencia (hay correlación: momentum, lado del mapa 3).

### 7.4. Calibración

La salida se recalibra (isotónica o Platt). La calibración es central para la fiabilidad y se evalúa con curvas de calibración además de las métricas escalares.

---

## 8. Evaluación

### 8.1. Métricas

| Métrica | Para qué |
|---|---|
| **Log loss** | Penaliza probabilidades mal calibradas; objetivo de entrenamiento |
| **Brier score** | Error cuadrático de la probabilidad |
| **AUC** | Capacidad de discriminación |
| **Accuracy** | Referencia secundaria (engaña por sí sola) |
| **Curva de calibración** | Que 70 % signifique 70 % |

> **Por qué la accuracy engaña:** el favorito gana la mayoría de las veces (~65–70 %), así que "elegir siempre al mejor rankeado" ya da ~65 %. Un modelo solo aporta si supera ese baseline.

### 8.2. Validación temporal

- **Split cronológico, nunca k-fold aleatorio.** Validación walk-forward / ventana expansiva: se entrena con el pasado y se valida con el futuro inmediato.
- El k-fold aleatorio infla las métricas de forma engañosa en datos temporales.

### 8.3. Benchmark de mercado

Comparar el log loss/Brier del modelo contra la probabilidad implícita **normalizada** de la **cuota de apertura**. Batir la cuota de cierre es muy difícil; batir la de apertura ya es señal de valor.

Reglas de evaluación:

- El mercado se evalúa como benchmark separado: `model` vs `market` vs `decision`.
- La métrica principal para comparar probabilidades es **log loss/Brier**, no accuracy. La literatura de modelos para betting muestra que la calibración es más importante que la exactitud bruta cuando las probabilidades alimentan EV/Kelly.
- Las odds de apertura pueden entrar en producción solo con validación
  walk-forward, muestra suficiente y aprobación de la puerta. El umbral
  operativo actual es **120 partidos cerrados con odds**.
- La capa `decision_prob_team1` se evalúa aparte porque mezcla calibración,
  fiabilidad y gestión de riesgo. No debe confundirse con el desglose causal de
  cada arquitectura ni con el benchmark Model A/Model B.

### 8.4. Análisis de fallos

Hay dos auditorias separadas para no mezclar muestras:

- `MODEL/analyze_failures.py` genera `MODEL/results/FAILURE_ANALYSIS.md` sobre snapshots live cerrados: model/risk-adjusted/decision, odds, fiabilidad y flags operativos.
- `MODEL/analyze_walkforward_errors.py` genera `MODEL/results/WALKFORWARD_FAILURE_AUDIT.md` sobre todas las predicciones historicas fuera de muestra. Une cada pronostico con sus features point-in-time, orienta las diferencias hacia el favorito y exporta todos los fallos localizados.

La auditoria walk-forward aplica:

- tasa de error, lift, risk ratio e intervalo Wilson por segmento;
- Fisher/Mann-Whitney con correccion Benjamini-Hochberg FDR;
- correlacion point-biserial, Cohen d, mutual information y matriz Spearman;
- detector auxiliar de error con cinco splits cronologicos (ridge y gradient boosting) comparado contra `1 - confianza`;
- CSV completo de fallos y lista enlazada de los errores de mayor confianza.

Resultado medido el 2026-07-10 (`n=7.056`): 2.511 fallos; el 50,9% esta por debajo de 60% de confianza y solo el 2,0% por encima de 80%. El mejor detector auxiliar no mejora usar solo la confianza (delta AUC `-0,006`), por lo que no hay evidencia de una regla oculta estable con las features actuales. Glicko/Elo/winrate/score estan fuertemente correlacionados y representan sobre todo el mismo eje de distancia de fuerza.

### 8.5. Intervenciones tras la auditoria de errores

Los fallos no se reponderan manualmente ni reciben pesos negativos: log loss ya penaliza con mayor severidad las derrotas predichas con mucha seguridad y pesos por clase/error alterarian la interpretacion de probabilidad. La literatura sobre class weighting advierte precisamente esa incompatibilidad con calibracion; focal loss tampoco es estrictamente proper para estimar probabilidades. Referencias: Caplin et al. (2022), Charoenphakdee et al. (2021), Gneiting y Raftery (2007).

Se implementan dos perfiles de features reproducibles:

- `core` (produccion): features originales point-in-time.
- `error-aware` (ablacion): consenso de Glicko/Elo/forma/formato, magnitud del margen, desacuerdo y acuerdo entre senales. Son simetricas al intercambiar equipos y no usan resultados futuros.

En el A/B temporal con 9.499 series y 7.063 predicciones OOS, `core` obtuvo Super Learner log loss `0,632053`, Brier `0,221023`, ECE `0,011748`; `error-aware` obtuvo `0,632335`, `0,221136`, `0,011945`. Por tanto el perfil enriquecido queda disponible para futuras re-evaluaciones, pero no se activa por defecto: la diferencia no supera el umbral practico de promocion y empeora levemente las metricas probabilisticas.

El entrenamiento ahora usa holdouts internos cronologicos para early stopping y calibracion. Compara logistica, LightGBM, CatBoost, XGBoost y Random Forest; ademas construye `super_learner_cal`, un ensemble con pesos no negativos que suman 1 y se aprenden solo de predicciones OOS anteriores. En el barrido completo del mismo protocolo, el Super Learner obtuvo log loss `0,6308`, Brier `0,2204`, ECE `0,0136`, accuracy `64,11%`; los pesos finales fueron logistica `0,401`, CatBoost `0,173`, Random Forest `0,426`, LightGBM `0`, XGBoost `0`.

Referencias metodologicas: [Super Learner](https://doi.org/10.2202/1544-6115.1309), [Cross-validation temporal](https://www.sciencedirect.com/science/article/pii/S0020025511006773), [proper scoring rules](https://doi.org/10.1198/016214506000001437), [Counter-Strike ML](https://dspace.cvut.cz/handle/10467/99181), [XGBoost](https://arxiv.org/abs/1603.02754), [LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) y [Random Forests](https://doi.org/10.1023/A:1010933404324).

Principio metodológico: que un flag aparezca en un fallo **no prueba causalidad**. Solo identifica un mecanismo plausible a testear con más muestra: map/veto incierto, roster reciente, baja cobertura de jugadores, BO1, schedule/fatiga, desacuerdo con mercado, etc.

---

## 9. Disciplina anti-fuga temporal (sección crítica de fiabilidad)

Es el error que más proyectos arruina. Reglas, en orden de impacto:

1. **Cero fuga temporal.** Cada feature se calcula usando *solo* datos anteriores al partido (rolling "as-of" la fecha). El Elo/Glicko se actualiza cronológicamente. Un winrate que incluya el propio partido a predecir invalida el modelo.
2. **Las stats de un partido histórico son las de ese mapa concreto**, nunca un agregado descargado después. Pegar el "rating actual" de un jugador a un partido de 2024 mete el futuro en el pasado.
3. **`ratings_history.before_match_id`** garantiza la reconstrucción del rating tal como era antes de cada partido. No guardar nunca un "rating actual" mutable en la fila del equipo.
4. **`match_features.data_up_to_utc`** registra hasta qué fecha se usaron datos: auditoría de no-fuga.
5. **Split cronológico** (ver §8.2).

Síntoma típico de fuga: 80 % de accuracy en validación y 55 % en producción.

---

## 10. Mantenimiento y cadencia de actualización (MLOps)

Tres relojes distintos, no uno:

### 10.1. Ratings Glicko-2 → continuo (semanal)
Actualización recursiva del estado por periodo de rating semanal. Barato y constante. Mantiene "fresco" el sistema sin reentrenar: un equipo en racha ve subir su rating y el modelo lo nota aunque sus pesos no cambien.

### 10.2. Features de un partido nuevo → en el momento de predecir
Se calculan point-in-time desde el estado actual de la base. No es reentrenar.

### 10.3. Modelo GBM/CatBoost → por nueva evidencia o bajo demanda
- **Reajuste automático:** `live_cutoff` es el `date_max` del vivo, pero el
  disparador usa `attempt_cutoff`, el máximo entre `live_cutoff` y el último
  intento terminal válido registrado. Al reunir **100** partidos etiquetados con
  fecha estrictamente posterior a `attempt_cutoff` crea un challenger. Esto impide
  repetir determinísticamente el mismo intento tras rechazo o aplazamiento.
  `-Retrain` fuerza el intento manualmente; ambos pasan por la misma puerta y
  pueden terminar en promoción, rechazo o aplazamiento.
- **Sweep profundo opcional:** tras actualizar snapshots, odds, Analytics,
  contexto, rosters y stats, `python MODEL\run_professional_training.py
  --install-deps` reevalúa algoritmos/half-life/gap; tampoco autoriza saltarse la
  puerta de producción.
- **Extended features sin switches:** una familia debe superar cobertura y ganar
  el holdout temporal fold-local por log loss. Cruzar el umbral ya no basta.
  Player snapshots, rankings y Analytics quedan automaticamente `OFF` cuando
  no demuestran mejora, aunque ya tengan la muestra minima.
- **Accuracy por franjas automatica:** cada entreno calcula sobre las predicciones
  walk-forward del challenger las bandas 50-60/60-70/70-80/80-90/90-100 y guarda
  aciertos, muestra, accuracy, probabilidad media y gap. La web solo presenta ese
  artefacto como vivo si supera la puerta; un rechazo no cambia el hash productivo.
- **Que prueba el entrenamiento profesional:** Logistica calibrada, LightGBM, CatBoost, ensembles, calibracion Platt/isotonica/beta, `form_half_life` en `45,60,90,120,180` y `wf_gap` en `0,1`.
- **Criterio de seleccion:** menor **log loss walk-forward**. La accuracy se reporta, pero no decide produccion si empeora la calidad probabilistica.
- **Seleccion sin sesgo L1 (desde 2026-07-27):** la metrica primaria es
  `nested_model_policy`. Antes de cada semana elige candidato usando solo el OOS
  acumulado de semanas anteriores. Con incumbente, `production_model` y el resto
  de la receta se congelan con `recipe_mask <= live_cutoff`; el bloque posterior
  no puede cambiarlos ni reajustar el shadow. Su score retrospectivo sobre todo el
  histórico queda como diagnóstico, no como estimación insesgada de la política
  completa.
- **Artefacto final actual (2026-07-27):** refit `super_learner_cal` sobre 9.726
  series. La estimación primaria `nested_model_policy` usa `n_eval=7.290`:
  accuracy `64,31%`, log loss `0,6308`, Brier `0,2203`, ROC-AUC `0,6886` y
  ECE `0,0127`. Frente al artefacto anterior mejora `+0,20` puntos de accuracy y
  `-0,00105` de log loss. Las familias finales aprobadas son
  `team_trueskill` y `event_history`.
- **Reajuste por evento (bajo demanda):** un cambio del pool de mapas de Valve,
  un parche grande de jugabilidad o un cambio de fórmula de rating de HLTV son
  motivos para lanzar `-Retrain` sin esperar al umbral automático. El challenger
  resultante no evita la puerta de promoción.
- **Reajuste por deriva (bajo demanda):** `MODEL/monitor_drift.py` se ejecuta
  automáticamente al final de `start.ps1` y alerta si Page-Hinkley o las ventanas
  de log loss/CLV superan `MODEL/config.yaml`; no publica ni reentrena por sí solo.
  El operador puede lanzar `-Retrain`, mientras el único disparador automático
  sigue siendo acumular al menos 100 etiquetas posteriores a `attempt_cutoff`.

### 10.4. Recalibración → incluida en cada reentreno/sweep
La calibración se desajusta antes que la capacidad de ranking. Cada reentreno y
el sweep profesional comparan Platt, isotónica y beta dentro de la validacion
walk-forward; si en el futuro se separa una capa de recalibracion ligera, debe
validarse con el mismo criterio de log loss/Brier y la misma puerta de promoción.

### 10.5. Busqueda de hiperparametros
Optuna purgado ajusta la regularizacion logistica de forma causal durante el walk-forward (8 trials, retune cada 26 semanas). El sweep amplio de algoritmos/half-life/gap sigue siendo trimestral o ante drift grande.

### 10.6. Monitorizacion
Cada entrenamiento guarda `experiment_manifest.json` con SHA-256 del dataset y
configuracion, semilla, commit/dirty state, argumentos y versiones. La promocion
separa selección, evaluación y despliegue. Con incumbente fija
`live_cutoff = incumbent.metadata.date_max`; `recipe_mask` restringe al prefijo
`<= live_cutoff` la selección de features, la familia `best_name`, Optuna purgado
y los pesos. Con esa receta se ajusta una única instancia shadow. El mismo objeto
predice todo el sufijo `> live_cutoff` de una vez, sin refits intermedios ni
etiquetas del sufijo; estas solo se revelan después para compararlo con el
incumbente sobre los mismos IDs y filas point-in-time. Las predicciones
walk-forward post-corte del diagnóstico no deciden la promoción.

Se exigen al menos 100 filas comunes. Log loss es primaria (`epsilon=0,001`);
dentro de esa banda solo promociona una mejora Brier de al menos `0,0005`.
Rechazo o muestra insuficiente conservan producción intacta. La receta congelada
puede refitearse sobre todo el histórico antes de materializar la decisión, ya que
ese cálculo no alimenta al shadow; el pickle resultante solo se publica si este
gana. Sus métricas validan el shadow as-of, no esos parámetros refiteados.
`promotion_decision.json` conserva `holdout_sha256` para IDs/fechas/etiquetas y
`prediction_sha256` para esa identidad más probabilidades de incumbente/shadow.
El health gate final carga directamente `registry/<version>/model.pkl`; su
`candidate_reference` verifica el SHA-256 que se pasa obligatoriamente al CAS.

El bundle núcleo (`model.pkl`, metadata, SHAP, manifest y config) se prepara en un
directorio staging oculto y se mueve una vez a una versión nueva. Esos archivos
son inmutables. `promotion_decision.json` y `deployment.json` son sidecars
auditables aditivos sancionados y no alteran el bundle ni el SHA del modelo. Por
eso `metadata.json` representa el estado pre-commit: puede contener
`promotion_approved=true` y `deployment_state=pending_pointer_commit` incluso
después de publicar. Solo `latest.json`, el hash runtime y `deployment.json`
confirman el publish.

`MODEL/artifacts/registry/latest.json` es el puntero lógico al vivo;
`last_good.json` conserva el incumbente saliente. Solo una promoción validada
reemplaza atómicamente la copia runtime `MODEL/artifacts/model.pkl`. La promoción
toma `registry/.deployment.lock` y ejecuta CAS de versión+SHA del incumbente y
SHA del challenger; `expected_incumbent` y `expected_candidate_sha256` son
obligatorios, y bootstrap exige el SHA esperado. Si difieren de las referencias
evaluadas, aborta sin mutar. Tras el commit, `deployment.json` registra
referencias, hashes, resultado, `decision_holdout_sha256` y
`decision_prediction_sha256`; un error al emitir el recibo se
advierte sin revertir el cambio. Los punteros y el hash runtime son la autoridad
operativa. El rollback (`start.ps1 -RollbackModel` o
`MODEL/manage_models.py rollback`)
restaura `last_good` y rota el vivo anterior a ese puntero bajo el mismo lock.

`MODEL/manage_models.py set-last-good --version <ID> --expected-sha256 <SHA>` es
la vía administrativa para corregir un `last_good` legado: valida carga, semilla,
predicción y hash del artefacto y hace CAS bajo el mismo lock, sin tocar
`latest.json` ni el runtime.

La retención conserva 0 versiones adicionales: exactamente
`latest`/`last_good`, y los 2 runs no referenciados más recientes, además del
publicado por master, los referenciados y los de nombre desconocido. Estos runs
referenciados son evidencia de procedencia, no generaciones de modelo. La primera
poda exige enseñar keep/delete-list, confirmar su token exacto y registrar la
política; un cambio de estado invalida el token antes de borrar.
La aplicación renombra atómicamente todos los candidatos a
`.retention-quarantine/<TOKEN>/` antes del primer borrado. Un fallo de staging
revierte todos los renames; un fallo de `rmtree` queda descrito por un manifest
estructurado y se reanuda mediante
`manage_models.py prune --scope <tipo> --resume <TOKEN>`, sin devolver bundles
parciales al namespace canónico.

`prediction_ledger` conserva una unica prediccion operativa por partido con
participantes, kickoff, timestamp, SHA-256 del artefacto/config/politica y
probabilidad. Sus cuatro campos escalares de incertidumbre son nullable y se
añaden sin backfill: una fila histórica conserva `NULL`. Se puede actualizar
solo antes del inicio con otra predicción válida; una observación inválida más
reciente no reemplaza evidencia abierta válida. Después queda
`frozen` y al llegar el resultado se calculan accuracy, log loss y Brier sobre
esa misma fila. `MODEL/evaluate_live_ledger.py` agrupa los resultados por hash
exacto. La reparación de integridad nunca borra el ledger: si un partido que
sería purgable contiene evidencia, conserva el partido completo y reporta su ID
como protegido. Las odds de cierre solo son `closing_observed` si fueron capturadas
antes del kickoff y como maximo seis horas antes; el resto es `legacy_proxy` y
no entra en CLV.

### 10.7. Política de mercado / odds

La ruta con odds forma parte de la arquitectura si gana la comparación temporal:

- **Antes de 120 partidos cerrados con odds:** no se ajusta ni promociona una
  ruta aprendida con mercado; producción conserva la rama validada sin odds.
- **A partir de 120 partidos:** router y modelo mixto se comparan en el mismo
  hold-out por log loss y Brier, con calibración y accuracy separadas para
  `odds`/`no_odds`. La arquitectura solo llega al vivo si gana además la puerta
  champion/challenger completa.
- **Nunca** se ajusta arquitectura, peso ni calibrador mirando resultados del
  mismo día o posteriores al cutoff congelado.
- Opening odds embebidas no se mezclan de nuevo en `decision_prob_team1`. Model B
  y mercado normalizado siguen publicados como referencias independientes.

---

## 11. Extensión futura: predicción por ronda (demos)

Si se quiere abordar el problema *en vivo* (win probability por ronda), la ruta es parsear demos con **awpy** (la librería del grupo de Xenopoulos), no HLTV. Es un proyecto distinto, más rico en features (estado de ronda, economía, posiciones), y permite reproducir el enfoque fundacional de win probability.

---

## 12. Estructura de repositorio

Estructura operativa actual:

```
CS2-Predictor/
|-- CS2/
|   |-- BBDD/           # SQLite, esquema, ingest, export y backups
|   |-- DOCS/           # changelog, runbooks y features futuras
|   |-- MODEL/          # entrenamiento, features, artefactos y análisis
|   |-- PIPELINE/       # scrape diario, master, runs y logs
|   |-- SCRAPER/        # cliente HLTV, sesión, caché y venv propio
|   |-- TESTS/          # validación, gráficos y cachés de tooling
|   |-- PROJECT.md      # diseño y decisiones exclusivas de CS2
|   |-- README.md       # comandos del dominio
|   |-- pyproject.toml  # Pytest, Ruff y Mypy de CS2
|   |-- requirements.txt
|   `-- start.ps1       # entrypoint real de CS2
|-- WEB/                # dashboard compartido entre deportes
|-- README.md           # índice multideporte
|-- pyproject.toml      # agregador de calidad multideporte
`-- start.ps1           # wrapper compatible que delega en CS2
```

Principio: cada deporte es un dominio autocontenido. La **fuente de verdad de
CS2** es `CS2/BBDD/cs2.db`; la matriz de entrenamiento se genera desde ella y
no se edita a mano. Ningún dato, modelo, scraper o test de CS2 vive mezclado con
otro deporte. Solo `WEB/`, CI y los entrypoints de repositorio son compartidos.

---

## 13. Reproducibilidad y rigor académico

- **Versionar el código y la configuración**, no los datos crudos (que se reconstruyen desde la cache de scraping).
- **Snapshots de datos con fecha** para cada experimento; sin ellos no hay reproducibilidad ni validación honesta.
- **Semillas fijas** en todo proceso aleatorio.
- **Documentar cada supuesto** (independencia entre mapas, ventana de decay, periodo de rating, umbrales de drift).
- **Registro de decisiones** (un ADR o changelog): qué se decidió, cuándo y por qué. A seis meses vista es lo más valioso.
- **Nota de ética e integridad:** scraping a bajo volumen y respetuoso con los ToS; uso de datos de integridad limitado a sanciones oficiales publicadas; proyecto sin fines de apuesta real.

---

## 14. Riesgos y limitaciones

| Riesgo | Mitigación |
|---|---|
| Fuga temporal | Diseño point-in-time (§9), tests dedicados |
| Amaños / etiquetas envenenadas | Tabla `sanctions`, limpieza de entrenamiento (§6.7) |
| Ruido de tiers bajos | Glicko-2 con RD, features de incertidumbre, manejo nativo de NaN |
| Deriva de meta (parches, pool) | Reentrenamiento por evento y por drift (§10) |
| Cambios de fórmula de rating | Era CS2, stats crudas estables, no mezclar fórmulas (§4.3) |
| Bloqueo de scraping (Cloudflare) | Herramientas adecuadas, cache, rate limiting (§4.1) |
| Canibalización por las cuotas | Modelo A sin cuotas como contribución central (§6.5) |
| Apoyarse demasiado en casas | Mercado solo como benchmark/prior operativo; peso aprendido solo con walk-forward y muestra suficiente (§6.5.1, §10.7) |
| Staking sobre probabilidades mal calibradas | Usar log loss/Brier, fiabilidad, Kelly fraccional y no apostar sin odds (§8.3, §10.7) |

---

## 15. Referencias

- Xenopoulos, P., et al. (2020). *Valuing Player Actions in Counter-Strike: Global Offensive.* IEEE BigData. (Win probability por ronda con XGBoost.)
- Dehpanah, S., et al. (2021). *Evaluating Team Skill Aggregation in Online Competitive Games.* Analiza agregadores `SUM`, `MAX` y `MIN`, incluido CS:GO; se usa para no imponer una regla de estrella unica.
- Björklund, A., et al. (2018). *Predicting the outcome of CS:GO games using machine learning.* Chalmers University of Technology.
- Makarov, I., et al. (2017). Trabajo sobre predicción de resultados de e-sports con métodos clásicos.
- Estudio comparativo con TrueSkill aplicado a predicción de CS:GO (IEEE).
- Trabajo de interpretabilidad con SHAP sobre rendimiento de jugadores profesionales de Counter-Strike (2025).
- HLTV.org — documentación del sistema de ranking y de la métrica Rating (2.0 / 2.1 / 3.0). Artículo "Introducing Rating 3.0" (ago. 2025) para el detalle de Round Swing y el ajuste por economía.
- ESIC (Esports Integrity Commission) e IBIA (International Betting Integrity Association) — informes y sanciones de integridad.
- `awpy` — librería de parsing de demos de Counter-Strike (para la extensión por ronda).
- Herramientas de scraping de HLTV: `gigobyte/hltv` (Node), `hltv-async-api` de akimerslys (Python async), `jparedesDS/hltv-scraper` (Python + undetected-chromedriver), `fanden/hltv-match-api` (Java + Playwright), y actores de Apify para HLTV (partidos, ranking, team info).
- Walsh, C. & Joshi, A. (2024). *Machine learning for sports betting: should model selection be based on accuracy or calibration?* Conclusión usada: para betting, la calibración/log loss es más importante que accuracy para selección de modelos y Kelly.
- Hegarty, T. & Whelan, K. (2024). *Comparing Two Methods for Testing the Efficiency of Sports Betting Markets.* Conclusión usada: las probabilidades implícitas deben normalizarse para quitar overround antes de compararlas con resultados/modelos.
- Snowberg, E. & Wolfers, J. (2010). *Explaining the Favorite-Longshot Bias.* Conclusión usada: las odds pueden contener sesgos sistemáticos; no son verdad absoluta.

---

## 16. Next steps — roadmap técnico (código, stats, algoritmos)

Mejoras identificadas tras auditoría + literatura (2024-2026). **Principio:** lo que
dependa de más muestra se deja *ready-to-use* con **auto-activación por umbral**
(patrón `AUTO_FEATURE_FAMILIES` en `MODEL/train.py`): se calcula/guarda siempre y
entra al modelo solo. **Nada se activa a mano.** Estado: ✅ hecho · 🟡 parcial · ⬜ pendiente.

### A. Estadística / metodología
- ✅ **A1. Ponderación por recencia en el learner** (`sample_weight` con decaimiento
  exponencial por fecha, Dixon-Coles). `--recency-half-life` (default 365d, activo).
- ✅ **A2. Incertidumbre del estimado** — `artifact.predict_symmetric_proba_team1_with_uncertainty`
  calcula la desviación ponderada entre miembros calibrados. `enrich` la combina
  con cobertura de historia estrictamente previa y publica una semibanda numérica,
  extremos y nivel alto/medio/bajo (`ensemble_dispersion_history_v1`). No es un
  intervalo riguroso: para un partido, la varianza Bernoulli `p(1-p)` ya está en
  el porcentaje; esta banda describe incertidumbre sobre el propio `p`. Se muestra
  en Partidos y Best Opportunity sin cambiar el ranking ni alimentar bankroll;
  el staking heredado conserva por separado su señal histórica. Véase
  `DOCS/ESTIMATE_UNCERTAINTY.md`.
- ✅ **A3. Calibración/auditoría por segmento** — `segment_calibration_suite` calcula
  curva de fiabilidad, gap, ECE, accuracy, log loss y Brier por **tramo absoluto de
  Elo as-of, LAN/online y fase**; conserva además formato y tier. Los desconocidos
  se excluyen y `N<100` no se interpreta. El mismo contrato monitoriza las filas
  congeladas del `prediction_ledger`, pero el live no decide hasta 1.000 resultados.
  Las interacciones Elo×segmento están apagadas por defecto y solo se prueban con
  `--segment-interactions`, primero en gate temporal fold-local y después contra el
  champion. Véase `DOCS/SEGMENT_CALIBRATION.md`.
- ✅ **A4. Purga/embargo (`--wf-gap`) + tests de significancia** (bootstrap+Wilcoxon
  pareado sobre log loss por-partido, CI95 + MDE) vs Glicko y vs 2º mejor →
  `significance.json` y metadatos del artefacto.
- ✅ **A5. Poda de features / multicolinealidad** — antes de entrenar se eliminan
  automáticamente solo constantes y duplicados matemáticos (incluido signo opuesto),
  sin mirar el target. VIF, permutation importance en holdout cronológico, RFE y SHAP
  generan `MODEL/results/feature_pruning.json`; sus candidatos son diagnósticos y no se
  eliminan sin demostrar mejora en validación temporal anidada.
- ✅ **A6. Target más rico BO3** — sidecar multiclase `0-2/1-2/2-1/2-0`, simétrico
  A↔B y con evaluación walk-forward. Se auto-activa con al menos 2.000 BO3, 300 por
  clase y solo si mejora el baseline empírico multiclase. El detalle se guarda en
  `rich_target_bo3*.json`, el artefacto expone la distribución y `enrich` publica
  `series_score_distribution`/`predicted_series_score`. No sustituye al Modelo A de
  ganador. El target por mapa sigue pendiente: solo hay 212 series con mapstats en la
  auditoría de 2026-07-13, insuficientes para validarlo profesionalmente.
- ✅ **A7. Strength-of-schedule explícito** — por equipo y estrictamente point-in-time:
  Elo medio ponderado por recencia de los rivales, residual real-menos-esperado L10 y
  ponderado, dispersión de fuerza rival y muestra mínima. Familia `strength_of_schedule`
  auto-activada desde 800 filas válidas mediante `AUTO_FEATURE_FAMILIES`.

**Validación 2026-07-13 sobre `BBDD/cs2.db`:** A5 redujo 69→68 columnas al detectar
una única redundancia exacta (`trueskill_available == mov_available`) y ejecutó el
diagnóstico en split cronológico 7.640/1.910. A6 quedó `ON`: 8.533 BO3 elegibles,
6.290 predicciones walk-forward, log loss multiclase 1,2949 frente a 1,3593 del
baseline, accuracy de marcador 41,30% y de ganador derivada 64,05%. A7 quedó `ON`
con 7.076/800 filas. A3 está operativo, pero su contexto aún no permite conclusiones
globales: 123 partidos con LAN/online y fase conocida (11 LAN, 112 online) y 0 con
tier de evento; esos segmentos permanecen correctamente como no concluyentes.

### B. Algoritmos / rating
- ✅ **B0. Ratings MOV, TrueSkill de equipo y por-jugador** — familias auto-gated
  (`mov_rating` 800, `team_trueskill` 800, `player_rating` 200).
- ✅ **B8. Bradley-Terry bayesiano jerárquico** — actualización online causal con
  prior gaussiano compartido, aproximación Laplace diagonal, partial pooling para
  equipos con poca muestra e incertidumbre predictiva. Es candidato calibrado propio
  y miembro explícito del super-learner; no entra en la matriz genérica, por lo que
  sus pesos convexos pueden quedar en cero si no aporta.
- ✅ **B9. Rating en espacio de estados Kalman** — media/varianza por equipo,
  ruido de proceso y observación pareada. Es candidato calibrado independiente y
  miembro del super-learner con el mismo aislamiento que B8.
- ✅ **B10. Optuna dentro de CV purgada** — activo por defecto: TPE determinista,
  objetivo log loss, folds internos expansivos con embargo mínimo de una semana,
  8 trials y reoptimización causal cada 26 semanas. El ajuste final vuelve a usar
  solo el histórico disponible. Auditoría en `MODEL/results/optuna_tuning.json`.
- ✅ **B11. Modelo composicional Bo3 por mapa + veto estimado** — selecciona pick de
  cada equipo y decider exclusivamente desde Analytics pre-match, contrae winrates
  por muestra y calcula `P(serie)=p1p2+p1(1-p2)p3+(1-p1)p2p3`. Auto-gate de 500
  series válidas y mínimo 10 mapas por equipo/mapa; jamás usa el veto post-partido
  como feature. La implementación está terminada, pero permanece `OFF` por cobertura.

**Validación B0/B8-B11 (2026-07-13):** B0: MOV y TrueSkill de equipo `ON`
con 8.846/800 filas; player rating `OFF` con 119/200. B8 y B9 `ON` con
7.076/800. Como candidatos aislados sobre 7.114 predicciones OOS, Bradley-Terry
obtuvo accuracy 60,94% / log loss 0,6517 y Kalman 61,41% / 0,6473: aportan
diversidad, pero no justifican sustituir al modelo principal y por eso el
super-learner puede asignarles peso cero. B10 superó el smoke purgado con Optuna
4.9 (`best_C=0,1582`, 2 trials de prueba, 3 folds). B11 tiene 0/500 casos
completamente elegibles; queda preparado para activarse sin intervención manual.

### C. Codigo / ingenieria (rigor, no accuracy)
- ✅ **C12. Deteccion de drift** — log loss y CLV rodantes + Page-Hinkley causal en
  `cs2model/drift.py`. `monitor_drift.py` lee predicciones congeladas ya cerradas y
  `start.ps1` lo ejecuta tras el ingest. El regimen de map pool se reconstruye solo
  con mapstats anteriores; el parche permanece desconocido salvo metadata pre-match
  explicita, sin inferirlo ni inventarlo.
- ✅ **C13. Config centralizada** — `MODEL/config.yaml` + dataclasses validadas en
  `cs2model/config.py`: defaults de entrenamiento, umbrales auto-gate, estimadores,
  drift y backtest. La CLI conserva prioridad y `--config` permite experimentos.
- ✅ **C14. CI + `ruff` + `mypy` + smoke** — GitHub Actions bloquea ante lint,
  tipos de la infraestructura, tests o smoke en push/PR a main/dev/pre-dev.
  `MODEL/smoke_pipeline.py` recorre carga, features, walk-forward, ajuste, artefacto,
  drift y backtest en unos segundos sin tocar produccion.
- ✅ **C15. Reproducibilidad determinista** — semillas centralizadas para Python,
  NumPy y estimadores; cada experimento registra hashes de datos/config, Git, CLI y
  dependencias y archiva el YAML efectivo junto al modelo. La carga para inferencia
  fuerza un solo hilo en los estimadores y el test de contrato repite la misma fila
  en el mismo proceso y en procesos separados con tolerancia absoluta `1e-12`.
- ✅ **C16. Auditoría de fuga exhaustiva** — `TESTS/test_leakage_audit.py`: verifica que
  las features rolling de cada partido son idénticas al reconstruir el estado solo con
  partidos anteriores (garantía point-in-time).
- ✅ **C17. Backtest economico realista** — `cs2model/economic.py` elimina el vig
  para auditar probabilidades de mercado, simula Kelly fraccional con limites de
  stake/payout y ejecucion configurable, y calcula CLV de precio contra la ultima
  cuota pre-match. El cierre es auditoria: nunca decide lado, EV ni stake.

**Validacion C12-C17 (2026-07-13):** `pytest TESTS/ -q` = 89 tests + 17 subtests;
`ruff check MODEL BBDD PIPELINE TESTS` = OK; `python -m mypy` = 6 modulos
tipados sin errores; smoke ejecutado dos veces = 160 filas, misma huella
`5c60402fddfd` y misma accuracy. Monitor real: 79 predicciones cerradas, 55 con
CLV (69,62% de cobertura), estado `ok` y CLV medio `-0,249%`.

### Prioridad (impacto/coste)
1. A1 · 2. A4 · 3. A2 · 4. B11 + A6 (props, donde esta el edge) · 5. aumentar cobertura point-in-time.

> Honestidad: el modelo ya está en el techo predictivo publicado; A1/B10 son marginales
> en accuracy. El valor grande está en **incertidumbre (A2), rigor (A4/C16), mercados
> nuevos (A6/B11) y medir CLV** — no en exprimir mas el moneyline.

---

*Documento de diseño vivo. Cerrar los puntos `[ABIERTO: …]` y actualizar la fecha de validación tras cada revisión mayor.*

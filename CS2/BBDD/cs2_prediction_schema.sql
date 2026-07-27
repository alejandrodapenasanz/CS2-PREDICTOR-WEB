-- ============================================================================
-- CS2 MATCH PREDICTION — ESQUEMA DE BASE DE DATOS
-- ----------------------------------------------------------------------------
-- Dialecto: SQLite (3.25+, por las window functions). Notas para PostgreSQL
-- incluidas al final. Principio rector: separar HECHOS INMUTABLES (lo que pasó,
-- con timestamp, append-only) de FEATURES DERIVADAS (lo que calculas tú,
-- point-in-time). Las stats cuelgan de (jugador, mapa), NUNCA agregados
-- "actuales" pegados a partidos pasados -> eso sería leakage temporal.
--
-- Fechas/horas se guardan como TEXT en formato ISO-8601 UTC ('YYYY-MM-DD' o
-- 'YYYY-MM-DDTHH:MM:SSZ'), que es la convención recomendada en SQLite.
-- ============================================================================

PRAGMA foreign_keys = ON;   -- imprescindible en SQLite, no está activo por defecto
PRAGMA journal_mode = WAL;  -- BBDD viva: mejor concurrencia lectura/escritura

-- ============================================================================
-- 1. ENTIDADES  (identidad estable en el tiempo)
-- ============================================================================

CREATE TABLE teams (
    team_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT,
    hltv_id     INTEGER UNIQUE          -- id en HLTV, para deduplicar al scrapear
);

CREATE TABLE players (
    player_id   INTEGER PRIMARY KEY,
    nick        TEXT NOT NULL,
    real_name   TEXT,
    country     TEXT,
    hltv_id     INTEGER UNIQUE
);

CREATE TABLE events (
    event_id    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    hltv_event_id TEXT UNIQUE,
    tier        TEXT,                   -- 'S','A','B','C' o como prefieras tipificar
    is_lan      INTEGER NOT NULL DEFAULT 0 CHECK (is_lan IN (0,1)),
    region      TEXT,
    prize_pool  INTEGER,
    teams_competing INTEGER,
    start_date  TEXT,                   -- ISO-8601
    end_date    TEXT,
    source_captured_at_utc TEXT,
    source_file TEXT
);

-- ============================================================================
-- 2. PERTENENCIA TEMPORAL  (rosters que cambian: un jugador NO "pertenece" a
--    un equipo, sino que estuvo en él durante un intervalo)
-- ============================================================================

CREATE TABLE team_rosters (
    roster_id   INTEGER PRIMARY KEY,
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    player_id   INTEGER NOT NULL REFERENCES players(player_id),
    valid_from  TEXT NOT NULL,          -- fecha de alta en el roster (ISO-8601)
    valid_to    TEXT,                   -- NULL = sigue activo
    source_run_id TEXT,
    source_signature TEXT,
    source_file TEXT,
    UNIQUE (team_id, player_id, valid_from)
);

-- ============================================================================
-- 3. HECHOS INMUTABLES  (lo que pasó; append-only)
-- ----------------------------------------------------------------------------
-- Átomo: una SERIE (match) -> contiene MAPAS -> cada mapa tiene STATS por
-- jugador. Un Bo3 es una serie de mapas, no un único registro.
-- ============================================================================

CREATE TABLE matches (
    match_id        INTEGER PRIMARY KEY,
    hltv_match_id   TEXT UNIQUE,                         -- id estable de HLTV
    event_id        INTEGER NOT NULL REFERENCES events(event_id),
    datetime_utc    TEXT NOT NULL,                       -- inicio de la serie, UTC
    team1_id        INTEGER NOT NULL REFERENCES teams(team_id),
    team2_id        INTEGER NOT NULL REFERENCES teams(team_id),
    best_of         INTEGER NOT NULL CHECK (best_of IN (1,2,3,5)),
    stage           TEXT CHECK (stage IN
                       ('group','swiss','ro16','quarter','semi','final','other')),
    environment     TEXT NOT NULL DEFAULT 'unknown'
                       CHECK (environment IN ('lan','online','unknown')),
    stage_detail    TEXT,
    incentive_label TEXT,
    high_stakes     INTEGER CHECK (high_stakes IN (0,1)),
    opening_match   INTEGER CHECK (opening_match IN (0,1)),
    winner_advances INTEGER CHECK (winner_advances IN (0,1)),
    loser_eliminated INTEGER CHECK (loser_eliminated IN (0,1)),
    bracket         TEXT CHECK (bracket IN ('upper','lower')),
    context_json    TEXT,
    status          TEXT NOT NULL DEFAULT 'scheduled'
                       CHECK (status IN ('scheduled','pending_result','completed')),
    data_tier       TEXT NOT NULL DEFAULT 'prematch_captured'
                       CHECK (data_tier IN ('historical_seed','prematch_captured','completed')),
    prematch_captured_at_utc TEXT,
    result_filled_at_utc     TEXT,
    has_prematch_odds    INTEGER NOT NULL DEFAULT 0 CHECK (has_prematch_odds IN (0,1)),
    has_player_snapshot  INTEGER NOT NULL DEFAULT 0 CHECK (has_player_snapshot IN (0,1)),
    has_ranking_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (has_ranking_snapshot IN (0,1)),
    has_analytics        INTEGER NOT NULL DEFAULT 0 CHECK (has_analytics IN (0,1)),
    has_context          INTEGER NOT NULL DEFAULT 0 CHECK (has_context IN (0,1)),
    has_box_score        INTEGER NOT NULL DEFAULT 0 CHECK (has_box_score IN (0,1)),
    has_veto             INTEGER NOT NULL DEFAULT 0 CHECK (has_veto IN (0,1)),
    winner_team_id  INTEGER REFERENCES teams(team_id),   -- NULL si aún no jugado
    score_t1        INTEGER,            -- mapas ganados por team1 (serie)
    score_t2        INTEGER,
    CHECK (team1_id <> team2_id)
);

CREATE TABLE maps (
    map_id          INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    map_number      INTEGER NOT NULL,                    -- 1,2,3 dentro de la serie
    map_name        TEXT NOT NULL,                       -- 'Mirage','Inferno',...
    picked_by_team_id INTEGER REFERENCES teams(team_id), -- NULL = decider
    winner_team_id  INTEGER REFERENCES teams(team_id),
    rounds_t1       INTEGER,            -- rondas ganadas por team1 en ese mapa
    rounds_t2       INTEGER,
    team1_ct_rounds INTEGER,
    team1_t_rounds  INTEGER,
    team2_ct_rounds INTEGER,
    team2_t_rounds  INTEGER,
    overtime_t1     INTEGER,
    overtime_t2     INTEGER,
    team1_rating    REAL,
    team2_rating    REAL,
    team1_first_kills INTEGER,
    team2_first_kills INTEGER,
    team1_clutches  INTEGER,
    team2_clutches  INTEGER,
    hltv_mapstats_id TEXT,
    source_file     TEXT,
    UNIQUE (match_id, map_number)
);

CREATE TABLE veto (
    veto_id         INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    step_order      INTEGER NOT NULL,                    -- orden del veto (1,2,3...)
    team_id         INTEGER REFERENCES teams(team_id),
    action          TEXT NOT NULL CHECK (action IN ('pick','ban','decider')),
    map_name        TEXT NOT NULL,
    UNIQUE (match_id, step_order)
);

-- Alineación REAL que jugó la serie (resuelve stand-ins, suplentes, etc.)
CREATE TABLE match_lineups (
    match_id    INTEGER NOT NULL REFERENCES matches(match_id),
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    player_id   INTEGER NOT NULL REFERENCES players(player_id),
    is_standin  INTEGER NOT NULL DEFAULT 0 CHECK (is_standin IN (0,1)),
    PRIMARY KEY (match_id, team_id, player_id)
);

-- AlineaciÃ³n anunciada ANTES de jugar. Es independiente de `match_lineups`
-- (alineaciÃ³n real posterior) para no introducir leakage ni ocultar stand-ins.
CREATE TABLE prematch_lineup_snapshots (
    prematch_lineup_snapshot_id INTEGER PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES matches(match_id),
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    player_id   INTEGER NOT NULL REFERENCES players(player_id),
    captured_at_utc TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    is_standin  INTEGER NOT NULL DEFAULT 0 CHECK (is_standin IN (0,1)),
    source_file TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (match_id, team_id, player_id, captured_at_utc, source_file)
);

-- Registro atómico de rendimiento: lo que un jugador hizo en UN mapa concreto.
-- De aquí derivas TODO lo demás (medias, forma, ratings). Nunca al revés.
CREATE TABLE map_player_stats (
    map_id          INTEGER NOT NULL REFERENCES maps(map_id),
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    kills           INTEGER,
    deaths          INTEGER,
    assists         INTEGER,
    adr             REAL,               -- average damage per round
    kast            REAL,               -- % (0-100)
    kpr             REAL,               -- kills per round
    dpr             REAL,               -- deaths per round
    rating          REAL,               -- rating HLTV del mapa (3.0 recalculado)
    round_swing     REAL,               -- métrica de Rating 3.0
    opening_kills   INTEGER,
    opening_deaths  INTEGER,
    headshots       INTEGER,
    flash_assists   INTEGER,
    multi_kill_rounds INTEGER,
    clutches_won    INTEGER,
    traded_deaths   INTEGER,
    fk_diff         INTEGER,
    source_file     TEXT,
    PRIMARY KEY (map_id, player_id)
);

-- Mismas tablas HLTV, conservadas por lado. `total` replica el box score
-- agregado; `ct` y `t` permiten features por lado sin reconsultar HLTV.
CREATE TABLE map_player_side_stats (
    map_id          INTEGER NOT NULL REFERENCES maps(map_id),
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    side            TEXT NOT NULL CHECK (side IN ('total','ct','t')),
    kills           INTEGER,
    deaths          INTEGER,
    assists         INTEGER,
    adr             REAL,
    kast            REAL,
    rating          REAL,
    round_swing     REAL,
    opening_kills   INTEGER,
    opening_deaths  INTEGER,
    headshots       INTEGER,
    flash_assists   INTEGER,
    multi_kill_rounds INTEGER,
    clutches_won    INTEGER,
    traded_deaths   INTEGER,
    eco_kills       INTEGER,
    eco_deaths      INTEGER,
    eco_adr         REAL,
    eco_kast        REAL,
    source_file     TEXT,
    payload_json    TEXT,
    PRIMARY KEY (map_id, player_id, side)
);

-- ============================================================================
-- 4. DERIVADO  (lo calculas tú; SIEMPRE point-in-time)
-- ============================================================================

-- Estado del rating Glicko-2 TAL COMO ERA JUSTO ANTES de cada partido.
-- 'before_match_id' garantiza la reconstrucción sin leakage; la 'rd'
-- (rating deviation) es tu medida de incertidumbre / "datos viejos".
CREATE TABLE ratings_history (
    rating_row_id   INTEGER PRIMARY KEY,
    entity_type     TEXT NOT NULL CHECK (entity_type IN ('team','player')),
    entity_id       INTEGER NOT NULL,   -- team_id o player_id según entity_type
    before_match_id INTEGER NOT NULL REFERENCES matches(match_id),
    as_of_date      TEXT NOT NULL,      -- = datetime_utc del partido
    rating          REAL NOT NULL,
    rd              REAL NOT NULL,      -- rating deviation (incertidumbre)
    sigma           REAL,               -- volatilidad (Glicko-2)
    UNIQUE (entity_type, entity_id, before_match_id)
);

-- "Feature mart": tabla ancha lista para exportar a pandas/parquet y entrenar.
-- Una fila por (partido, equipo). 'data_up_to_utc' documenta hasta qué momento
-- se usaron datos -> prueba auditable de ausencia de fuga temporal.
-- Añade aquí tantas columnas de features como necesites.
CREATE TABLE match_features (
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    data_up_to_utc  TEXT NOT NULL,
    glicko_rating   REAL,
    glicko_rd       REAL,
    glicko_sigma    REAL,
    elo             REAL,
    form_winrate_10 REAL,               -- winrate últimos 10 mapas (con decay)
    form_winrate_20 REAL,
    winrate_overall REAL,
    avg_score_diff  REAL,
    recent_opp_elo  REAL,               -- fuerza media de rivales recientes
    streak          INTEGER,
    matches_played  INTEGER,
    days_since_last INTEGER,            -- días desde el último partido
    roster_age_days INTEGER,            -- días desde último cambio de roster
    maps_together   INTEGER,            -- mapas jugados por los 5 actuales
    PRIMARY KEY (match_id, team_id)
);

-- Cuotas: separa apertura (legítima pre-partido) de cierre (cuasi-omnisciente).
CREATE TABLE odds (
    odds_id         INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    bookmaker       TEXT NOT NULL,
    captured_at_utc TEXT NOT NULL,
    market_type     TEXT NOT NULL CHECK (market_type IN ('opening','closing','live')),
    odds_t1         REAL,               -- cuota decimal team1
    odds_t2         REAL,               -- cuota decimal team2
    prob_t1         REAL,               -- prob. implícita normalizada team1
    prob_t2         REAL,               -- prob. implícita normalizada team2
    overround       REAL,
    run_id          TEXT,
    source_file     TEXT,
    UNIQUE (match_id, bookmaker, captured_at_utc, market_type)
);

-- Lista de integridad: para limpiar etiquetas envenenadas (amaños) del
-- entrenamiento y/o flaggear contexto. SOLO sanciones oficiales publicadas.
CREATE TABLE sanctions (
    sanction_id     INTEGER PRIMARY KEY,
    entity_type     TEXT NOT NULL CHECK (entity_type IN ('team','player')),
    entity_id       INTEGER NOT NULL,
    body            TEXT NOT NULL,      -- 'ESIC','IBIA',...
    ban_start       TEXT,
    ban_end         TEXT,
    reason          TEXT,
    source_url      TEXT
);

-- ============================================================================
-- 4b. STAGING RAW  (capa 1 de la arquitectura: lo scrapeado tal cual)
--     Conserva el JSON original con su provenance; la capa core se deriva de
--     aquí mediante limpieza. Permite re-ingestar sin re-scrapear.
-- ============================================================================

CREATE TABLE raw_results (
    raw_id          INTEGER PRIMARY KEY,
    hltv_match_id   TEXT,                -- id de HLTV (string)
    source_file     TEXT NOT NULL,       -- fichero de origen (provenance)
    ingested_at_utc TEXT NOT NULL,
    payload_json    TEXT NOT NULL,       -- fila original sin transformar
    UNIQUE (hltv_match_id, source_file)
);

-- Archivo append-only de snapshots diarios. Guarda el JSON original capturado
-- por el pipeline para que un backup de cs2.db preserve exactamente lo visto ese
-- dia, aunque HLTV cambie perfiles, stats u odds mas tarde.
CREATE TABLE raw_snapshots (
    raw_snapshot_id INTEGER PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN
                       ('run_manifest','upcoming_matches','match_snapshot',
                        'team_profile','player_compare_stats','predictions_enriched',
                        'match_assets','match_analytics','team_ranking','raw_html',
                        'data_quality_report')),
    hltv_match_id   TEXT,
    hltv_team_id    TEXT,
    hltv_player_id  TEXT,
    run_id          TEXT NOT NULL,
    captured_at_utc TEXT,
    source_file     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    UNIQUE (kind, run_id, source_file)
);

-- Stats agregadas de jugador tal como las devolvio HLTV en cada run. Son
-- snapshots historicos: no sustituyen a map_player_stats, pero evitan depender
-- de consultar hoy una pagina historica que puede haber cambiado.
CREATE TABLE player_stat_snapshots (
    player_stat_snapshot_id INTEGER PRIMARY KEY,
    hltv_player_id  TEXT NOT NULL,
    player_name     TEXT,
    player_slug     TEXT,
    player_link     TEXT,
    run_id          TEXT NOT NULL,
    captured_at_utc TEXT,
    season_year     INTEGER,
    time_filter     TEXT,
    match_filter    TEXT,
    map_filter      TEXT,
    maps            INTEGER,
    rating          REAL,
    kpr             REAL,
    dpr             REAL,
    apr             REAL,
    kast            REAL,
    impact          REAL,
    adr             REAL,
    round_swing     REAL,
    multi_kill_rating REAL,
    awp_kpr         REAL,
    hs_pct          REAL,
    opening_kpr     REAL,
    opening_dpr     REAL,
    flash_assists   REAL,
    source_file     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    UNIQUE (hltv_player_id, run_id, source_file, time_filter)
);

CREATE TABLE team_ranking_snapshots (
    team_ranking_snapshot_id INTEGER PRIMARY KEY,
    team_id         INTEGER REFERENCES teams(team_id),
    hltv_team_id    TEXT,
    ranking_type    TEXT NOT NULL CHECK (ranking_type IN ('hltv','valve')),
    ranking_date_text TEXT,
    position        INTEGER,
    points          INTEGER,
    run_id          TEXT NOT NULL,
    captured_at_utc TEXT,
    source_file     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    UNIQUE (ranking_type, run_id, hltv_team_id, position)
);

-- Analytics Center de HLTV, capturado como una foto temporal y también en
-- tablas consultables. `payload_json` conserva todo el response parseado.
CREATE TABLE match_analytics_snapshots (
    analytics_snapshot_id INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    captured_at_utc TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    event_name      TEXT,
    hltv_event_id   TEXT,
    prize_pool      INTEGER,
    teams_competing INTEGER,
    team1_core_matches_lt INTEGER,
    team2_core_matches_lt INTEGER,
    team1_matches_sample INTEGER,
    team1_maps_sample INTEGER,
    team2_matches_sample INTEGER,
    team2_maps_sample INTEGER,
    team1_overtime_pct REAL,
    team2_overtime_pct REAL,
    payload_json    TEXT NOT NULL,
    UNIQUE (match_id, captured_at_utc, source_file)
);

CREATE TABLE match_analytics_map_stats (
    analytics_snapshot_id INTEGER NOT NULL REFERENCES match_analytics_snapshots(analytics_snapshot_id),
    team_id         INTEGER REFERENCES teams(team_id),
    team_name       TEXT,
    map_name        TEXT NOT NULL,
    first_pick_pct  REAL,
    first_ban_pct   REAL,
    win_pct         REAL,
    played          INTEGER,
    comment         TEXT,
    PRIMARY KEY (analytics_snapshot_id, team_name, map_name)
);

CREATE TABLE match_analytics_map_handicap (
    analytics_snapshot_id INTEGER NOT NULL REFERENCES match_analytics_snapshots(analytics_snapshot_id),
    team_id         INTEGER REFERENCES teams(team_id),
    team_name       TEXT,
    map_name        TEXT NOT NULL,
    avg_rounds_lost_in_wins REAL,
    avg_rounds_won_in_losses REAL,
    PRIMARY KEY (analytics_snapshot_id, team_name, map_name)
);

-- ============================================================================
-- 4c. PREDICCIONES DEL MODELO  (auditoría: qué predijo el modelo y cuándo)
--     Una fila por (partido, versión de modelo). Guarda la probabilidad
--     calibrada PRE-partido y, cuando llega el resultado, permite evaluar
--     calibración real sin re-derivar nada (la prob se congela en su momento).
-- ============================================================================

CREATE TABLE predictions (
    prediction_id   INTEGER PRIMARY KEY,
    match_id        INTEGER REFERENCES matches(match_id),
    hltv_match_id   TEXT,
    model_version   TEXT NOT NULL,       -- p.ej. 'glicko2+logistic_cal@2026-06-30'
    predicted_at_utc TEXT NOT NULL,
    prob_team1      REAL NOT NULL,       -- prob. calibrada de victoria de team1 (Model A)
    odds_prob_team1 REAL,                -- prob. implícita del mercado (apertura)
    blended_prob_team1 REAL,
    risk_adjusted_prob_team1 REAL,       -- Model A encogido hacia 50/50 por fiabilidad
    decision_prob_team1 REAL,            -- prob. operativa usada para ranking/staking
    decision_confidence REAL,
    decision_probability_source TEXT,
    decision_market_weight REAL,
    decision_market_weight_reasons TEXT,
    decision_policy_json TEXT,
    reliability_score REAL,
    opportunity_score REAL,
    opportunity_eligible INTEGER CHECK (opportunity_eligible IN (0,1)),
    opportunity_rank INTEGER,
    opportunity_rank_day INTEGER,
    is_best_opportunity INTEGER CHECK (is_best_opportunity IN (0,1)),
    opportunity_policy_version TEXT,
    opportunity_min_confidence REAL,
    opportunity_min_reliability REAL,
    decision_edge_team1_vs_market REAL,
    favorite_team_id INTEGER REFERENCES teams(team_id),
    favorite_name TEXT,
    decision_favorite_side TEXT CHECK (decision_favorite_side IN ('team1','team2')),
    decision_min_value_odds REAL,       -- cuota decimal minima EV=0 para el favorito operativo
    team1_min_value_odds REAL,          -- 1 / decision_prob_team1
    team2_min_value_odds REAL,          -- 1 / (1 - decision_prob_team1)
    confidence      REAL,
    context_environment TEXT CHECK (context_environment IN ('lan','online','unknown')),
    context_stage   TEXT,
    context_stage_detail TEXT,
    context_incentive_label TEXT,
    context_high_stakes INTEGER CHECK (context_high_stakes IN (0,1)),
    context_winner_advances INTEGER CHECK (context_winner_advances IN (0,1)),
    context_loser_eliminated INTEGER CHECK (context_loser_eliminated IN (0,1)),
    context_opening_match INTEGER CHECK (context_opening_match IN (0,1)),
    context_bracket TEXT CHECK (context_bracket IN ('upper','lower')),
    context_json    TEXT,
    prediction_json TEXT,               -- objeto prediction completo congelado
    features_json   TEXT,               -- features usadas/mostradas por la prediccion
    odds_json       TEXT,               -- odds agregadas del snapshot pre-partido
    staking_json    TEXT,               -- recomendacion Kelly/fraccional y motivo
    controls_json   TEXT,               -- controles profesionales: mercado, mapas, fatiga...
    flags_json      TEXT,               -- senales visibles en web
    data_quality_json TEXT,             -- provenance del snapshot y calidad
    rosters_json    TEXT,               -- rosters/players visibles en detalle
    UNIQUE (hltv_match_id, model_version)
);

-- Estado de frescura por entidad compartida. El scraper consulta esta tabla
-- antes de pedir perfiles, stats, rankings, assets o analytics a HLTV.
CREATE TABLE fetch_state (
    entity_type   TEXT NOT NULL CHECK (entity_type IN
                    ('team_profile','player_stats','ranking_hltv','ranking_valve',
                     'match_detail','match_assets','match_analytics')),
    entity_key    TEXT NOT NULL,
    last_fetched_at_utc  TEXT,
    last_status   TEXT CHECK (last_status IN ('ok','partial','blocked','not_found','error')),
    fetch_count   INTEGER NOT NULL DEFAULT 0,
    next_eligible_at_utc TEXT,
    note          TEXT,
    PRIMARY KEY (entity_type, entity_key)
);

-- Auditoría de cada ingest incremental run -> cs2.db.
CREATE TABLE ingest_runs (
    ingest_id     INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL,
    started_at_utc  TEXT NOT NULL,
    finished_at_utc TEXT,
    status        TEXT CHECK (status IN ('ok','partial','failed')),
    rows_upserted_json TEXT,
    requests_made INTEGER,
    requests_skipped_by_freshness INTEGER,
    note          TEXT
);

-- ============================================================================
-- 5. ÍNDICES  (orientados a las queries point-in-time y a los joins frecuentes)
-- ============================================================================

CREATE INDEX idx_matches_datetime     ON matches(datetime_utc);
CREATE UNIQUE INDEX idx_matches_hltv  ON matches(hltv_match_id);
CREATE INDEX idx_matches_team1        ON matches(team1_id);
CREATE INDEX idx_matches_team2        ON matches(team2_id);
CREATE INDEX idx_matches_event        ON matches(event_id);
CREATE INDEX idx_matches_context      ON matches(environment, stage, high_stakes);
CREATE INDEX idx_matches_status       ON matches(status);
CREATE INDEX idx_matches_tier         ON matches(data_tier);

CREATE INDEX idx_maps_match           ON maps(match_id);
CREATE INDEX idx_maps_mapstats        ON maps(hltv_mapstats_id);
CREATE INDEX idx_mps_player           ON map_player_stats(player_id);
CREATE INDEX idx_mps_team             ON map_player_stats(team_id);
CREATE INDEX idx_mps_side_player      ON map_player_side_stats(player_id, side);
CREATE INDEX idx_mps_side_team        ON map_player_side_stats(team_id, side);

CREATE INDEX idx_rosters_player       ON team_rosters(player_id);
CREATE INDEX idx_rosters_team         ON team_rosters(team_id);
CREATE INDEX idx_rosters_validity     ON team_rosters(team_id, valid_from, valid_to);
CREATE INDEX idx_prematch_lineups_match ON prematch_lineup_snapshots(match_id, captured_at_utc);

CREATE INDEX idx_ratings_lookup       ON ratings_history(entity_type, entity_id, as_of_date);
CREATE INDEX idx_odds_match           ON odds(match_id, market_type);
CREATE INDEX idx_sanctions_entity     ON sanctions(entity_type, entity_id);
CREATE INDEX idx_predictions_match    ON predictions(match_id);
CREATE INDEX idx_predictions_version  ON predictions(model_version);
CREATE INDEX idx_predictions_context  ON predictions(context_environment, context_stage);
CREATE INDEX idx_predictions_opportunity
    ON predictions(model_version, opportunity_eligible, opportunity_rank);
CREATE INDEX idx_teams_name           ON teams(name);
CREATE INDEX idx_events_name          ON events(name);
CREATE INDEX idx_events_hltv          ON events(hltv_event_id);
CREATE INDEX idx_raw_snapshots_run    ON raw_snapshots(run_id, kind);
CREATE INDEX idx_raw_snapshots_match  ON raw_snapshots(hltv_match_id);
CREATE INDEX idx_player_stats_player  ON player_stat_snapshots(hltv_player_id, captured_at_utc);
CREATE INDEX idx_team_rankings_team   ON team_ranking_snapshots(hltv_team_id, ranking_type, captured_at_utc);
CREATE INDEX idx_fetch_state_eligible ON fetch_state(entity_type, next_eligible_at_utc);
CREATE INDEX idx_analytics_snapshot_match ON match_analytics_snapshots(match_id, captured_at_utc);
CREATE INDEX idx_analytics_map_stats_snapshot ON match_analytics_map_stats(analytics_snapshot_id, map_name);

-- ============================================================================
-- 6. VISTA DE EJEMPLO  (roster activo de cada equipo a día de hoy)
--    Demuestra el patrón temporal: filtrar por valid_from/valid_to.
-- ============================================================================

CREATE VIEW current_rosters AS
SELECT r.team_id, t.name AS team_name, r.player_id, p.nick
FROM team_rosters r
JOIN teams   t ON t.team_id   = r.team_id
JOIN players p ON p.player_id = r.player_id
WHERE r.valid_to IS NULL;

-- ============================================================================
-- NOTAS PARA POSTGRESQL (si migras cuando el scraper escriba en concurrencia):
--   * INTEGER PRIMARY KEY  ->  GENERATED ALWAYS AS IDENTITY (o SERIAL/BIGSERIAL)
--   * TEXT para fechas      ->  usa tipos nativos DATE / TIMESTAMPTZ
--   * INTEGER 0/1 booleans  ->  usa BOOLEAN nativo (elimina los CHECK IN (0,1))
--   * Window functions y CTEs funcionan igual o mejor.
-- ============================================================================

"""Esquema SQLite versionado de la base de datos operativa.

La migración inicial se ejecuta dentro de ``BEGIN IMMEDIATE`` y fija tanto
``PRAGMA user_version`` como una fila auditable en ``schema_versions``.
Predicciones, estadísticas, observaciones, selección oficial, settlements y
conflictos están protegidos contra ``UPDATE`` y ``DELETE`` mediante triggers.
"""

from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[int] = 2

SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    run_type TEXT NOT NULL CHECK (
        run_type IN ('prediction', 'observation')
    ),
    source_system TEXT NOT NULL,
    target_date TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('building', 'complete', 'failed')
    ),
    input_rows INTEGER NOT NULL CHECK (input_rows >= 0),
    accepted_rows INTEGER NOT NULL DEFAULT 0 CHECK (accepted_rows >= 0),
    queued_rows INTEGER NOT NULL DEFAULT 0 CHECK (queued_rows >= 0),
    payload_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    source_system TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL,
    snapshot_date TEXT,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (source_system, snapshot_sha256, retrieved_at_utc)
);

CREATE TABLE IF NOT EXISTS run_snapshots (
    run_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    PRIMARY KEY (run_id, snapshot_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS matches (
    source_match_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    source_system TEXT NOT NULL,
    match_date TEXT,
    tournament TEXT,
    tournament_key TEXT,
    tour_level TEXT,
    gender TEXT CHECK (gender IS NULL OR gender IN ('M', 'F')),
    player_1_slug TEXT,
    player_2_slug TEXT,
    identity_payload_sha256 TEXT NOT NULL,
    first_snapshot_id TEXT,
    created_at_utc TEXT NOT NULL,
    CHECK (
        player_1_slug IS NULL
        OR player_2_slug IS NULL
        OR player_1_slug <> player_2_slug
    ),
    FOREIGN KEY (first_snapshot_id) REFERENCES snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    run_id TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    snapshot_id TEXT,
    prediction_as_of_utc TEXT,
    source_retrieved_at_utc TEXT,
    source_status TEXT,
    player_a_name TEXT,
    player_b_name TEXT,
    player_a_slug TEXT,
    player_b_slug TEXT,
    player_a_id INTEGER,
    player_b_id INTEGER,
    mapping_status TEXT,
    model_probability_raw_a REAL,
    model_probability_a REAL,
    model_probability_b REAL,
    market_probability_a REAL,
    market_probability_b REAL,
    edge_a REAL,
    edge_b REAL,
    confidence TEXT,
    confidence_flags TEXT,
    prediction_status TEXT,
    model_profile TEXT,
    model_fingerprint TEXT,
    model_training_max_date TEXT,
    model_training_available_max_date TEXT,
    feature_fingerprint TEXT,
    is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
    invalid_reason TEXT,
    payload_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (run_id, source_match_id),
    CHECK (
        model_probability_a IS NULL
        OR (model_probability_a >= 0.0 AND model_probability_a <= 1.0)
    ),
    CHECK (
        model_probability_b IS NULL
        OR (model_probability_b >= 0.0 AND model_probability_b <= 1.0)
    ),
    CHECK (
        model_probability_raw_a IS NULL
        OR (
            model_probability_raw_a >= 0.0
            AND model_probability_raw_a <= 1.0
        )
    ),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY (source_match_id) REFERENCES matches(source_match_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS official_predictions (
    source_match_id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL UNIQUE,
    selection_rule TEXT NOT NULL CHECK (
        selection_rule = 'first_registered_valid'
    ),
    selected_at_utc TEXT NOT NULL,
    FOREIGN KEY (source_match_id) REFERENCES matches(source_match_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS player_statistics (
    statistic_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    run_id TEXT NOT NULL,
    source_match_id TEXT,
    player_slug TEXT NOT NULL,
    player_id INTEGER,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    as_of_date TEXT NOT NULL,
    statistic_name TEXT NOT NULL,
    statistic_value REAL,
    source_kind TEXT NOT NULL CHECK (
        source_kind NOT IN ('prediction', 'settlement', 'result', 'label')
    ),
    source_snapshot_id TEXT,
    payload_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY (source_match_id) REFERENCES matches(source_match_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (source_snapshot_id) REFERENCES snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    source_match_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    snapshot_id TEXT,
    observed_at_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    player_1_sets_won INTEGER,
    player_2_sets_won INTEGER,
    sets_score TEXT,
    winner_side TEXT CHECK (
        winner_side IS NULL
        OR winner_side IN ('player_1', 'player_2')
    ),
    winner_slug TEXT,
    result_evidence TEXT,
    is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
    invalid_reason TEXT,
    payload_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS run_observations (
    run_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    PRIMARY KEY (run_id, observation_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS settlements (
    settlement_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    source_match_id TEXT NOT NULL UNIQUE,
    official_prediction_id TEXT NOT NULL UNIQUE,
    observation_id TEXT NOT NULL,
    winner_slug TEXT NOT NULL,
    actual_outcome_a INTEGER NOT NULL CHECK (actual_outcome_a IN (0, 1)),
    settled_at_utc TEXT NOT NULL,
    FOREIGN KEY (source_match_id) REFERENCES matches(source_match_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (official_prediction_id)
        REFERENCES predictions(prediction_id) ON DELETE RESTRICT,
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    conflict_type TEXT NOT NULL,
    source_match_id TEXT,
    run_id TEXT,
    existing_reference TEXT,
    incoming_reference TEXT,
    details_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS review_queue (
    queue_id TEXT PRIMARY KEY,
    record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version = 1),
    issue_type TEXT NOT NULL,
    source_match_id TEXT,
    run_id TEXT,
    observation_id TEXT,
    conflict_id TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (
        status IN ('open', 'resolved')
    ),
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    resolved_at_utc TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (conflict_id) REFERENCES conflicts(conflict_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS predictions_match_time_idx
    ON predictions(source_match_id, created_at_utc, prediction_id);
CREATE INDEX IF NOT EXISTS observations_match_time_idx
    ON observations(source_match_id, observed_at_utc, observation_id);
CREATE INDEX IF NOT EXISTS queue_status_idx
    ON review_queue(status, issue_type, created_at_utc);
CREATE INDEX IF NOT EXISTS statistics_player_date_idx
    ON player_statistics(gender, player_slug, as_of_date);

CREATE TRIGGER IF NOT EXISTS predictions_no_update
BEFORE UPDATE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS predictions_no_delete
BEFORE DELETE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS official_predictions_no_update
BEFORE UPDATE ON official_predictions
BEGIN
    SELECT RAISE(ABORT, 'official_predictions son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS official_predictions_no_delete
BEFORE DELETE ON official_predictions
BEGIN
    SELECT RAISE(ABORT, 'official_predictions son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS player_statistics_no_update
BEFORE UPDATE ON player_statistics
BEGIN
    SELECT RAISE(ABORT, 'player_statistics son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS player_statistics_no_delete
BEFORE DELETE ON player_statistics
BEGIN
    SELECT RAISE(ABORT, 'player_statistics son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS observations_no_update
BEFORE UPDATE ON observations
BEGIN
    SELECT RAISE(ABORT, 'observations son append-only');
END;

CREATE TRIGGER IF NOT EXISTS observations_no_delete
BEFORE DELETE ON observations
BEGIN
    SELECT RAISE(ABORT, 'observations son append-only');
END;

CREATE TRIGGER IF NOT EXISTS settlements_no_update
BEFORE UPDATE ON settlements
BEGIN
    SELECT RAISE(ABORT, 'settlements son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS settlements_no_delete
BEFORE DELETE ON settlements
BEGIN
    SELECT RAISE(ABORT, 'settlements son inmutables');
END;

CREATE TRIGGER IF NOT EXISTS conflicts_no_update
BEFORE UPDATE ON conflicts
BEGIN
    SELECT RAISE(ABORT, 'conflicts son append-only');
END;

CREATE TRIGGER IF NOT EXISTS conflicts_no_delete
BEFORE DELETE ON conflicts
BEGIN
    SELECT RAISE(ABORT, 'conflicts son append-only');
END;
"""

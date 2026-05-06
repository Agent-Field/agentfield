-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS triggers (
    id              TEXT PRIMARY KEY,
    source_name     TEXT NOT NULL,
    config_json     TEXT NOT NULL DEFAULT '{}',
    secret_env_var  TEXT NOT NULL DEFAULT '',
    target_node_id  TEXT NOT NULL,
    target_reasoner TEXT NOT NULL,
    event_types     TEXT NOT NULL DEFAULT '[]',
    managed_by      TEXT NOT NULL DEFAULT 'ui' CHECK (managed_by IN ('code','ui')),
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triggers_target ON triggers(target_node_id, target_reasoner);
CREATE INDEX IF NOT EXISTS idx_triggers_source ON triggers(source_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_triggers_code_managed
    ON triggers(target_node_id, target_reasoner, source_name)
    WHERE managed_by = 'code';

CREATE TABLE IF NOT EXISTS inbound_events (
    id                 TEXT PRIMARY KEY,
    trigger_id         TEXT NOT NULL,
    source_name        TEXT NOT NULL,
    event_type         TEXT NOT NULL DEFAULT '',
    raw_payload        TEXT NOT NULL DEFAULT '',
    normalized_payload TEXT NOT NULL DEFAULT '',
    idempotency_key    TEXT NOT NULL DEFAULT '',
    vc_id              TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'received',
    error_message      TEXT NOT NULL DEFAULT '',
    received_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed_at       TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_inbound_events_trigger ON inbound_events(trigger_id, received_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inbound_events_idempotency
    ON inbound_events(source_name, idempotency_key)
    WHERE idempotency_key <> '';

ALTER TABLE observability_dead_letter_queue ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'observability';
CREATE INDEX IF NOT EXISTS idx_observability_dlq_kind ON observability_dead_letter_queue(kind);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_observability_dlq_kind;
ALTER TABLE observability_dead_letter_queue DROP COLUMN IF EXISTS kind;
DROP INDEX IF EXISTS idx_inbound_events_idempotency;
DROP INDEX IF EXISTS idx_inbound_events_trigger;
DROP TABLE IF EXISTS inbound_events;
DROP INDEX IF EXISTS idx_triggers_code_managed;
DROP INDEX IF EXISTS idx_triggers_source;
DROP INDEX IF EXISTS idx_triggers_target;
DROP TABLE IF EXISTS triggers;
-- +goose StatementEnd

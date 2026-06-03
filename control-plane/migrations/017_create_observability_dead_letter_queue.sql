-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS observability_dead_letter_queue (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    error_message TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient querying by creation time (oldest first for redrive)
CREATE INDEX IF NOT EXISTS idx_observability_dlq_created_at ON observability_dead_letter_queue(created_at);

-- Index for filtering by event type/source
CREATE INDEX IF NOT EXISTS idx_observability_dlq_event_type ON observability_dead_letter_queue(event_type, event_source);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_observability_dlq_event_type;
DROP INDEX IF EXISTS idx_observability_dlq_created_at;
DROP TABLE IF EXISTS observability_dead_letter_queue;
-- +goose StatementEnd

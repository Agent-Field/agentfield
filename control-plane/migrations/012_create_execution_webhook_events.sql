-- +migrate Up
CREATE TABLE IF NOT EXISTS execution_webhook_events (
    id BIGSERIAL PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES workflow_executions(execution_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    http_status INTEGER,
    payload JSONB,
    response_body TEXT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_webhook_events_execution_id ON execution_webhook_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_webhook_events_created_at ON execution_webhook_events(created_at DESC);

-- +migrate Down
DROP TABLE IF EXISTS execution_webhook_events;

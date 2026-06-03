-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS observability_webhooks (
    id TEXT PRIMARY KEY DEFAULT 'global',
    url TEXT NOT NULL,
    secret TEXT,
    headers JSONB DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Ensure only one row exists (singleton pattern with id='global')
CREATE UNIQUE INDEX IF NOT EXISTS idx_observability_webhooks_singleton ON observability_webhooks((id IS NOT NULL));
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_observability_webhooks_singleton;
DROP TABLE IF EXISTS observability_webhooks;
-- +goose StatementEnd

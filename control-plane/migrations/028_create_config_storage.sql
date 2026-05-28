-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS config_storage (
    id          BIGSERIAL PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,
    value       TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    created_by  TEXT,
    updated_by  TEXT,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_storage_key ON config_storage(key);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_config_storage_key;
DROP TABLE IF EXISTS config_storage;
-- +goose StatementEnd

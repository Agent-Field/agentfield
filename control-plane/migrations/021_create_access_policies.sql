-- +goose Up
-- +goose StatementBegin

-- Access policies table
-- Defines tag-based authorization policies for cross-agent calls
CREATE TABLE IF NOT EXISTS access_policies (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    caller_tags     JSONB NOT NULL DEFAULT '[]',
    target_tags     JSONB NOT NULL DEFAULT '[]',
    allow_functions JSONB DEFAULT '[]',
    deny_functions  JSONB DEFAULT '[]',
    constraints     JSONB DEFAULT '{}',
    action          TEXT NOT NULL DEFAULT 'allow',  -- 'allow' or 'deny'
    priority        INTEGER NOT NULL DEFAULT 0,     -- higher = evaluated first
    enabled         BOOLEAN NOT NULL DEFAULT true,
    description     TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for finding enabled policies only
CREATE INDEX IF NOT EXISTS idx_access_policies_enabled ON access_policies(enabled)
    WHERE enabled = true;

-- Index for priority-ordered evaluation
CREATE INDEX IF NOT EXISTS idx_access_policies_priority ON access_policies(priority DESC);

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

DROP INDEX IF EXISTS idx_access_policies_priority;
DROP INDEX IF EXISTS idx_access_policies_enabled;
DROP TABLE IF EXISTS access_policies;

-- +goose StatementEnd

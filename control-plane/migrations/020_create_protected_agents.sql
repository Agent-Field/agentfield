-- +goose Up
-- +goose StatementBegin

-- Protected agents configuration table
-- Defines which agents require permission to call based on patterns
CREATE TABLE IF NOT EXISTS protected_agents_config (
    id              BIGSERIAL PRIMARY KEY,
    pattern_type    TEXT NOT NULL,            -- 'tag', 'tag_pattern', 'agent_id'
    pattern         TEXT NOT NULL,            -- e.g., 'admin', 'finance*', 'payment-gateway'
    description     TEXT,                     -- Human-readable description of the rule
    enabled         BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Each pattern type + pattern combination must be unique
    CONSTRAINT unique_pattern UNIQUE (pattern_type, pattern)
);

-- Index for efficient lookup by pattern type
CREATE INDEX IF NOT EXISTS idx_protected_agents_type ON protected_agents_config(pattern_type);

-- Index for finding enabled rules only
CREATE INDEX IF NOT EXISTS idx_protected_agents_enabled ON protected_agents_config(enabled)
    WHERE enabled = true;

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

DROP INDEX IF EXISTS idx_protected_agents_enabled;
DROP INDEX IF EXISTS idx_protected_agents_type;
DROP TABLE IF EXISTS protected_agents_config;

-- +goose StatementEnd

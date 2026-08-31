-- Migration: Per-execution token/cost usage ingestion
-- Description: Stores token and cost usage entries reported by agent SDKs in the
--              execution result envelope's "usage" object. One execution may
--              produce many rows (one per LLM/harness entry). Costs are nullable
--              because an entry may report tokens without a resolvable cost.
--              Powers the /api/ui/v1/usage/stats aggregation endpoint.
-- Created: 2026-07-17

-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS execution_usage (
    id                    BIGSERIAL PRIMARY KEY,
    execution_id          TEXT NOT NULL,
    workflow_id           TEXT NOT NULL,
    agent_node_id         TEXT NOT NULL,
    reasoner              TEXT NOT NULL DEFAULT '',
    source                TEXT NOT NULL DEFAULT '',
    provider              TEXT NOT NULL DEFAULT '',
    model                 TEXT NOT NULL DEFAULT '',
    harness               TEXT NOT NULL DEFAULT '',
    input_tokens          BIGINT NOT NULL DEFAULT 0,
    output_tokens         BIGINT NOT NULL DEFAULT 0,
    cache_read_tokens     BIGINT NOT NULL DEFAULT 0,
    cache_creation_tokens BIGINT NOT NULL DEFAULT 0,
    total_tokens          BIGINT NOT NULL DEFAULT 0,
    cost_usd              DOUBLE PRECISION,
    cost_source           TEXT NOT NULL DEFAULT '',
    created_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_usage_execution ON execution_usage(execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_usage_workflow ON execution_usage(workflow_id);
CREATE INDEX IF NOT EXISTS idx_execution_usage_agent_node_id ON execution_usage(agent_node_id);
CREATE INDEX IF NOT EXISTS idx_execution_usage_model ON execution_usage(model);
CREATE INDEX IF NOT EXISTS idx_execution_usage_created_at ON execution_usage(created_at);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_execution_usage_created_at;
DROP INDEX IF EXISTS idx_execution_usage_model;
DROP INDEX IF EXISTS idx_execution_usage_agent_node_id;
DROP INDEX IF EXISTS idx_execution_usage_workflow;
DROP INDEX IF EXISTS idx_execution_usage_execution;
DROP TABLE IF EXISTS execution_usage;
-- +goose StatementEnd

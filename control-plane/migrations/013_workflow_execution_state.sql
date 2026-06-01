-- 013_workflow_execution_state.sql
-- Introduces append-only workflow execution/run events and execution state columns.

ALTER TABLE workflow_executions
    ADD COLUMN IF NOT EXISTS run_id TEXT REFERENCES workflow_runs(run_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS state_version BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_event_sequence BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS active_children INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS pending_children INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS pending_terminal_status TEXT,
    ADD COLUMN IF NOT EXISTS status_reason TEXT,
    ADD COLUMN IF NOT EXISTS lease_owner TEXT,
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;

ALTER TABLE workflow_runs
    ADD COLUMN IF NOT EXISTS state_version BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_event_sequence BIGINT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS workflow_execution_events (
    event_id BIGSERIAL PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES workflow_executions(execution_id) ON DELETE CASCADE,
    workflow_id TEXT NOT NULL,
    run_id TEXT REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    parent_execution_id TEXT,
    sequence BIGINT NOT NULL,
    previous_sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT,
    status_reason TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    emitted_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (execution_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_workflow_execution_events_execution_sequence
    ON workflow_execution_events (execution_id, sequence);
CREATE INDEX IF NOT EXISTS idx_workflow_execution_events_run_sequence
    ON workflow_execution_events (run_id, sequence);

CREATE TABLE IF NOT EXISTS workflow_run_events (
    event_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL,
    previous_sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT,
    status_reason TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    emitted_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_workflow_run_events_sequence
    ON workflow_run_events (run_id, sequence);

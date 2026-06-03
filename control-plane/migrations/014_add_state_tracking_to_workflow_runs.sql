-- 014_add_state_tracking_to_workflow_runs.sql
-- Adds state version and event sequence tracking columns to workflow_runs table.
-- These columns are required for workflow state management and event sourcing.

ALTER TABLE workflow_runs
ADD COLUMN IF NOT EXISTS state_version INTEGER NOT NULL DEFAULT 0;

ALTER TABLE workflow_runs
ADD COLUMN IF NOT EXISTS last_event_sequence INTEGER NOT NULL DEFAULT 0;

-- Add index for efficient state queries
CREATE INDEX IF NOT EXISTS idx_workflow_runs_state_version
ON workflow_runs(state_version);

-- Add index for event sequence queries
CREATE INDEX IF NOT EXISTS idx_workflow_runs_event_sequence
ON workflow_runs(last_event_sequence);

-- +goose Up
-- Add 'paused' status for external cancel/pause execution support (Epic #238).
-- Also adds 'waiting' which was missing from constraints but is a valid canonical status
-- used by the HITL approval flow (running -> waiting -> running/cancelled/failed).
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout'));

ALTER TABLE executions DROP CONSTRAINT IF EXISTS executions_status_check;
ALTER TABLE executions ADD CONSTRAINT executions_status_check
  CHECK (status IN ('unknown', 'pending', 'queued', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked'));

-- +goose Down
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'waiting', 'succeeded', 'failed', 'cancelled', 'timeout'));

ALTER TABLE executions DROP CONSTRAINT IF EXISTS executions_status_check;
ALTER TABLE executions ADD CONSTRAINT executions_status_check
  CHECK (status IN ('unknown', 'pending', 'queued', 'running', 'waiting', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked'));

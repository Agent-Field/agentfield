-- +goose Up
-- Async executions remain queued until a pool worker dispatches them.
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'queued', 'in_progress', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout'));

-- +goose Down
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
-- Queued is unavailable after rollback, so preserve admitted work as running.
UPDATE workflow_executions SET status = 'running' WHERE status = 'queued';
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout'));

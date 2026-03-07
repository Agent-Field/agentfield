-- +goose Up
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout'));

ALTER TABLE executions DROP CONSTRAINT IF EXISTS executions_status_check;
ALTER TABLE executions ADD CONSTRAINT executions_status_check
  CHECK (status IN ('unknown', 'pending', 'queued', 'running', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked'));

-- +goose Down
ALTER TABLE workflow_executions DROP CONSTRAINT IF EXISTS workflow_executions_status_check;
ALTER TABLE workflow_executions ADD CONSTRAINT workflow_executions_status_check
  CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'succeeded', 'failed', 'cancelled', 'timeout'));

ALTER TABLE executions DROP CONSTRAINT IF EXISTS executions_status_check;
ALTER TABLE executions ADD CONSTRAINT executions_status_check
  CHECK (status IN ('unknown', 'pending', 'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked'));

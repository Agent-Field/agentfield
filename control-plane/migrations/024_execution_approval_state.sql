-- 024_execution_approval_state.sql
-- Adds approval tracking columns to workflow_executions for human-in-the-loop approval flows.
-- Also adds status_reason to the lightweight executions table so polling APIs can surface it.
-- All columns are nullable — existing rows are unaffected.

-- Approval tracking on workflow_executions
ALTER TABLE workflow_executions ADD COLUMN approval_request_id TEXT;
ALTER TABLE workflow_executions ADD COLUMN approval_request_url TEXT;
ALTER TABLE workflow_executions ADD COLUMN approval_status TEXT;
ALTER TABLE workflow_executions ADD COLUMN approval_response TEXT;
ALTER TABLE workflow_executions ADD COLUMN approval_requested_at TIMESTAMP;
ALTER TABLE workflow_executions ADD COLUMN approval_responded_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_workflow_executions_approval_request_id ON workflow_executions(approval_request_id);

-- status_reason on lightweight executions table (mirrors workflow_executions.status_reason)
ALTER TABLE executions ADD COLUMN status_reason TEXT;

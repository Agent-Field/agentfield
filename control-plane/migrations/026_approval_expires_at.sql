-- 026_approval_expires_at.sql
-- Adds approval_expires_at to workflow_executions so consumers (UI, agents)
-- can derive "expired" from status=waiting + now > approval_expires_at
-- without needing a separate execution status.

ALTER TABLE workflow_executions ADD COLUMN approval_expires_at TIMESTAMP;

-- 025_approval_callback_url.sql
-- Adds callback URL column so the control plane can push approval results
-- directly to the waiting agent instead of requiring polling.

ALTER TABLE workflow_executions ADD COLUMN approval_callback_url TEXT;

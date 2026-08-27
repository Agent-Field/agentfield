-- +goose Up
-- +goose StatementBegin
ALTER TABLE executions ADD COLUMN IF NOT EXISTS instance_id TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS instance_id TEXT;
CREATE INDEX IF NOT EXISTS idx_executions_agent_instance ON executions(agent_node_id, instance_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_instance ON workflow_executions(agent_node_id, instance_id);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_workflow_executions_agent_instance;
DROP INDEX IF EXISTS idx_executions_agent_instance;
ALTER TABLE workflow_executions DROP COLUMN IF EXISTS instance_id;
ALTER TABLE executions DROP COLUMN IF EXISTS instance_id;
-- +goose StatementEnd

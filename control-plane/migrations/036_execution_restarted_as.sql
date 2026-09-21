-- +goose Up
-- +goose StatementBegin
ALTER TABLE executions ADD COLUMN IF NOT EXISTS restarted_as_execution_id TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS restarted_as_execution_id TEXT;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
ALTER TABLE workflow_executions DROP COLUMN IF EXISTS restarted_as_execution_id;
ALTER TABLE executions DROP COLUMN IF EXISTS restarted_as_execution_id;
-- +goose StatementEnd

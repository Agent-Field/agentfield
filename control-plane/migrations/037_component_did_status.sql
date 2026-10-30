-- +goose Up
-- +goose StatementBegin
ALTER TABLE component_dids ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
-- +goose StatementEnd

-- +goose StatementBegin
CREATE INDEX IF NOT EXISTS idx_component_dids_status ON component_dids(status);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP INDEX IF EXISTS idx_component_dids_status;
-- +goose StatementEnd

-- +goose StatementBegin
ALTER TABLE component_dids DROP COLUMN IF EXISTS status;
-- +goose StatementEnd

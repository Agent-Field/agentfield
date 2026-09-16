-- +goose Up
-- +goose StatementBegin
ALTER TABLE agent_dids ADD COLUMN IF NOT EXISTS x25519_epoch INTEGER NOT NULL DEFAULT 0;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
ALTER TABLE agent_dids DROP COLUMN IF EXISTS x25519_epoch;
-- +goose StatementEnd

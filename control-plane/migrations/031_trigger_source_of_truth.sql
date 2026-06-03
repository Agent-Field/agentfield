-- Migration: Source-of-truth columns on triggers
-- Description: Adds the columns that distinguish "code is canonical" from
--              "operator may pause without a code deploy" — enables the
--              sticky-pause override (manual_override_enabled), records where
--              in the user's source code each code-managed trigger was
--              declared (code_origin), tracks when the agent last re-declared
--              the binding (last_registered_at), and flags rows whose
--              decorator was removed in code (orphaned).
-- Created: 2026-04-28

-- +goose Up
ALTER TABLE triggers
    ADD COLUMN manual_override_enabled BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE triggers
    ADD COLUMN manual_override_at TIMESTAMP;

ALTER TABLE triggers
    ADD COLUMN code_origin TEXT;

ALTER TABLE triggers
    ADD COLUMN last_registered_at TIMESTAMP;

ALTER TABLE triggers
    ADD COLUMN orphaned BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_triggers_orphaned ON triggers(orphaned);
CREATE INDEX IF NOT EXISTS idx_triggers_manual_override ON triggers(manual_override_enabled);

-- +goose Down
DROP INDEX IF EXISTS idx_triggers_manual_override;
DROP INDEX IF EXISTS idx_triggers_orphaned;
ALTER TABLE triggers DROP COLUMN orphaned;
ALTER TABLE triggers DROP COLUMN last_registered_at;
ALTER TABLE triggers DROP COLUMN code_origin;
ALTER TABLE triggers DROP COLUMN manual_override_at;
ALTER TABLE triggers DROP COLUMN manual_override_enabled;

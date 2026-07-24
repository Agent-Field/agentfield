-- Migration: Discriminate trigger event VCs from execution VCs
-- Description: Adds a `kind` column plus trigger-event-specific metadata to
--              execution_vcs so trigger event VCs (rooted at the control
--              plane's own DID, attesting that an external signed payload
--              arrived) can share the existing storage and chain-walking
--              helpers with execution VCs. Execution VCs continue to default
--              to kind='execution' so existing rows and clients are unaffected.
-- Created: 2026-04-28

-- +goose Up
ALTER TABLE execution_vcs
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'execution';

ALTER TABLE execution_vcs
    ADD COLUMN trigger_id TEXT;

ALTER TABLE execution_vcs
    ADD COLUMN source_name TEXT;

ALTER TABLE execution_vcs
    ADD COLUMN event_type TEXT;

ALTER TABLE execution_vcs
    ADD COLUMN event_id TEXT;

CREATE INDEX IF NOT EXISTS idx_execution_vcs_kind ON execution_vcs(kind);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_trigger_id ON execution_vcs(trigger_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_event_id ON execution_vcs(event_id);

-- +goose Down
DROP INDEX IF EXISTS idx_execution_vcs_event_id;
DROP INDEX IF EXISTS idx_execution_vcs_trigger_id;
DROP INDEX IF EXISTS idx_execution_vcs_kind;

ALTER TABLE execution_vcs DROP COLUMN event_id;
ALTER TABLE execution_vcs DROP COLUMN event_type;
ALTER TABLE execution_vcs DROP COLUMN source_name;
ALTER TABLE execution_vcs DROP COLUMN trigger_id;
ALTER TABLE execution_vcs DROP COLUMN kind;

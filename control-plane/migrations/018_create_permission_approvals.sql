-- +goose Up
-- +goose StatementBegin

-- Permission approvals table for tracking caller -> target permission requests
CREATE TABLE IF NOT EXISTS permission_approvals (
    id              BIGSERIAL PRIMARY KEY,
    caller_did      TEXT NOT NULL,
    target_did      TEXT NOT NULL,
    caller_agent_id TEXT NOT NULL,
    target_agent_id TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, approved, rejected, revoked
    approved_by     TEXT,
    approved_at     TIMESTAMP WITH TIME ZONE,
    rejected_by     TEXT,
    rejected_at     TIMESTAMP WITH TIME ZONE,
    revoked_by      TEXT,
    revoked_at      TIMESTAMP WITH TIME ZONE,
    expires_at      TIMESTAMP WITH TIME ZONE,
    reason          TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Each caller-target pair can only have one active record
    CONSTRAINT unique_caller_target UNIQUE (caller_did, target_did)
);

-- Index for looking up permissions by caller
CREATE INDEX IF NOT EXISTS idx_perm_approvals_caller ON permission_approvals(caller_did);

-- Index for looking up permissions by target
CREATE INDEX IF NOT EXISTS idx_perm_approvals_target ON permission_approvals(target_did);

-- Index for filtering by status (pending requests, active approvals)
CREATE INDEX IF NOT EXISTS idx_perm_approvals_status ON permission_approvals(status);

-- Index for finding expired approvals
CREATE INDEX IF NOT EXISTS idx_perm_approvals_expires ON permission_approvals(expires_at)
    WHERE expires_at IS NOT NULL;

-- Index for efficient lookup by DID pair
CREATE INDEX IF NOT EXISTS idx_perm_approvals_did_pair ON permission_approvals(caller_did, target_did);

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

DROP INDEX IF EXISTS idx_perm_approvals_did_pair;
DROP INDEX IF EXISTS idx_perm_approvals_expires;
DROP INDEX IF EXISTS idx_perm_approvals_status;
DROP INDEX IF EXISTS idx_perm_approvals_target;
DROP INDEX IF EXISTS idx_perm_approvals_caller;
DROP TABLE IF EXISTS permission_approvals;

-- +goose StatementEnd

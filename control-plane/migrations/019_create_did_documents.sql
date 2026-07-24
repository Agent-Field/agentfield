-- +goose Up
-- +goose StatementBegin

-- DID documents table for did:web resolution
-- Stores the DID document that is served when resolving a did:web identifier
CREATE TABLE IF NOT EXISTS did_documents (
    did             TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    did_document    JSONB NOT NULL,           -- Full W3C DID Document
    public_key_jwk  TEXT NOT NULL,            -- Public key in JWK format
    revoked_at      TIMESTAMP WITH TIME ZONE, -- NULL = active, set = revoked
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for looking up DID documents by agent ID
CREATE INDEX IF NOT EXISTS idx_did_docs_agent ON did_documents(agent_id);

-- Index for finding revoked DIDs
CREATE INDEX IF NOT EXISTS idx_did_docs_revoked ON did_documents(revoked_at)
    WHERE revoked_at IS NOT NULL;

-- Index for finding active (non-revoked) DIDs
CREATE INDEX IF NOT EXISTS idx_did_docs_active ON did_documents(did)
    WHERE revoked_at IS NULL;

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

DROP INDEX IF EXISTS idx_did_docs_active;
DROP INDEX IF EXISTS idx_did_docs_revoked;
DROP INDEX IF EXISTS idx_did_docs_agent;
DROP TABLE IF EXISTS did_documents;

-- +goose StatementEnd

-- Migration: Create Execution VCs table
-- Description: Execution verifiable credentials for individual component executions
-- Created: 2025-01-08

CREATE TABLE IF NOT EXISTS execution_vcs (
    vc_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    issuer_did TEXT NOT NULL,
    target_did TEXT,
    caller_did TEXT NOT NULL,
    vc_document TEXT NOT NULL, -- JSON document containing the full VC
    signature TEXT NOT NULL,
    storage_uri TEXT DEFAULT '',
    document_size_bytes INTEGER DEFAULT 0,
    input_hash TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'revoked')),
    parent_vc_id TEXT, -- For VC chains
    child_vc_ids TEXT DEFAULT '[]', -- JSON array of child VC IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (parent_vc_id) REFERENCES execution_vcs(vc_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_execution_vcs_execution_id ON execution_vcs(execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_workflow_id ON execution_vcs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_session_id ON execution_vcs(session_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_issuer_did ON execution_vcs(issuer_did);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_target_did ON execution_vcs(target_did);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_caller_did ON execution_vcs(caller_did);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_status ON execution_vcs(status);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_storage_uri ON execution_vcs(storage_uri);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_parent_vc_id ON execution_vcs(parent_vc_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_created_at ON execution_vcs(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_vcs_execution_unique ON execution_vcs(execution_id, issuer_did, target_did);

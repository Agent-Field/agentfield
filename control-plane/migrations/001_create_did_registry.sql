-- Migration: Create DID Registry table
-- Description: Organization-level DID registry for managing master DIDs and agent nodes
-- Created: 2025-01-08

CREATE TABLE IF NOT EXISTS did_registry (
    organization_id TEXT PRIMARY KEY,
    master_seed_encrypted BLOB NOT NULL,
    root_did TEXT NOT NULL UNIQUE,
    agent_nodes TEXT DEFAULT '{}', -- JSON map of agent node DIDs
    total_dids INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_key_rotation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_did_registry_root_did ON did_registry(root_did);
CREATE INDEX IF NOT EXISTS idx_did_registry_created_at ON did_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_did_registry_last_key_rotation ON did_registry(last_key_rotation);

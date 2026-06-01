-- Migration: Create Agent DIDs table
-- Description: Agent node DID information with relationships to reasoners and skills
-- Created: 2025-01-08

CREATE TABLE IF NOT EXISTS agent_dids (
    did TEXT PRIMARY KEY,
    agent_node_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    public_key_jwk TEXT NOT NULL, -- JSON Web Key format
    derivation_path TEXT NOT NULL,
    reasoners TEXT DEFAULT '{}', -- JSON map of reasoner DIDs
    skills TEXT DEFAULT '{}', -- JSON map of skill DIDs
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'revoked')),
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key constraints
    FOREIGN KEY (organization_id) REFERENCES did_registry(organization_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_dids_agent_node_id ON agent_dids(agent_node_id);
CREATE INDEX IF NOT EXISTS idx_agent_dids_organization_id ON agent_dids(organization_id);
CREATE INDEX IF NOT EXISTS idx_agent_dids_status ON agent_dids(status);
CREATE INDEX IF NOT EXISTS idx_agent_dids_registered_at ON agent_dids(registered_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_dids_agent_node_org ON agent_dids(agent_node_id, organization_id);

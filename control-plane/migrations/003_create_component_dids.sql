-- Migration: Create Component DIDs table
-- Description: Reasoner and skill DID information with component-specific metadata
-- Created: 2025-01-08

CREATE TABLE IF NOT EXISTS component_dids (
    did TEXT PRIMARY KEY,
    agent_did TEXT NOT NULL,
    component_type TEXT NOT NULL CHECK (component_type IN ('reasoner', 'skill')),
    function_name TEXT NOT NULL,
    public_key_jwk TEXT NOT NULL, -- JSON Web Key format
    derivation_path TEXT NOT NULL,
    capabilities TEXT DEFAULT '[]', -- JSON array for reasoners
    tags TEXT DEFAULT '[]', -- JSON array for skills
    exposure_level TEXT NOT NULL DEFAULT 'private' CHECK (exposure_level IN ('private', 'public', 'restricted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key constraints
    FOREIGN KEY (agent_did) REFERENCES agent_dids(did) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_component_dids_agent_did ON component_dids(agent_did);
CREATE INDEX IF NOT EXISTS idx_component_dids_component_type ON component_dids(component_type);
CREATE INDEX IF NOT EXISTS idx_component_dids_function_name ON component_dids(function_name);
CREATE INDEX IF NOT EXISTS idx_component_dids_exposure_level ON component_dids(exposure_level);
CREATE INDEX IF NOT EXISTS idx_component_dids_created_at ON component_dids(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_component_dids_agent_function ON component_dids(agent_did, function_name, component_type);

-- Migration: Fix Haxen Server ID Schema Inconsistency
-- Description: Update schema to use haxen_server_id consistently instead of organization_id
-- Created: 2025-01-21

-- Step 1: Add haxen_server_id column to did_registry table
ALTER TABLE did_registry ADD COLUMN haxen_server_id TEXT;

-- Step 2: Copy organization_id values to haxen_server_id for existing records
UPDATE did_registry SET haxen_server_id = organization_id WHERE haxen_server_id IS NULL;

-- Step 3: Make haxen_server_id NOT NULL and add unique constraint
-- First, ensure all records have haxen_server_id populated
UPDATE did_registry SET haxen_server_id = 'default' WHERE haxen_server_id IS NULL OR haxen_server_id = '';

-- Now make it NOT NULL
-- Note: SQLite doesn't support ALTER COLUMN, so we need to recreate the table
CREATE TABLE did_registry_new (
    haxen_server_id TEXT PRIMARY KEY,
    organization_id TEXT, -- Keep for backward compatibility during transition
    master_seed_encrypted BLOB NOT NULL,
    root_did TEXT NOT NULL UNIQUE,
    agent_nodes TEXT DEFAULT '{}', -- JSON map of agent node DIDs
    total_dids INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_key_rotation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Copy data from old table to new table
INSERT INTO did_registry_new (haxen_server_id, organization_id, master_seed_encrypted, root_did, agent_nodes, total_dids, created_at, last_key_rotation)
SELECT haxen_server_id, organization_id, master_seed_encrypted, root_did, agent_nodes, total_dids, created_at, last_key_rotation
FROM did_registry;

-- Drop old table and rename new table
DROP TABLE did_registry;
ALTER TABLE did_registry_new RENAME TO did_registry;

-- Recreate indexes for performance
CREATE INDEX IF NOT EXISTS idx_did_registry_root_did ON did_registry(root_did);
CREATE INDEX IF NOT EXISTS idx_did_registry_created_at ON did_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_did_registry_last_key_rotation ON did_registry(last_key_rotation);
CREATE INDEX IF NOT EXISTS idx_did_registry_organization_id ON did_registry(organization_id); -- For backward compatibility

-- Step 4: Update agent_dids table to use haxen_server_id
ALTER TABLE agent_dids ADD COLUMN haxen_server_id TEXT;

-- Copy organization_id values to haxen_server_id for existing records
UPDATE agent_dids SET haxen_server_id = organization_id WHERE haxen_server_id IS NULL;

-- Ensure all records have haxen_server_id populated
UPDATE agent_dids SET haxen_server_id = 'default' WHERE haxen_server_id IS NULL OR haxen_server_id = '';

-- Recreate agent_dids table with haxen_server_id as the foreign key
CREATE TABLE agent_dids_new (
    did TEXT PRIMARY KEY,
    agent_node_id TEXT NOT NULL,
    haxen_server_id TEXT NOT NULL,
    organization_id TEXT, -- Keep for backward compatibility during transition
    public_key_jwk TEXT NOT NULL, -- JSON Web Key format
    derivation_path TEXT NOT NULL,
    reasoners TEXT DEFAULT '{}', -- JSON map of reasoner DIDs
    skills TEXT DEFAULT '{}', -- JSON map of skill DIDs
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'revoked')),
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (haxen_server_id) REFERENCES did_registry(haxen_server_id) ON DELETE CASCADE
);

-- Copy data from old table to new table
INSERT INTO agent_dids_new (did, agent_node_id, haxen_server_id, organization_id, public_key_jwk, derivation_path, reasoners, skills, status, registered_at, created_at, updated_at)
SELECT did, agent_node_id, haxen_server_id, organization_id, public_key_jwk, derivation_path, reasoners, skills, status, registered_at, created_at, updated_at
FROM agent_dids;

-- Drop old table and rename new table
DROP TABLE agent_dids;
ALTER TABLE agent_dids_new RENAME TO agent_dids;

-- Recreate indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_dids_agent_node_id ON agent_dids(agent_node_id);
CREATE INDEX IF NOT EXISTS idx_agent_dids_haxen_server_id ON agent_dids(haxen_server_id);
CREATE INDEX IF NOT EXISTS idx_agent_dids_organization_id ON agent_dids(organization_id); -- For backward compatibility
CREATE INDEX IF NOT EXISTS idx_agent_dids_status ON agent_dids(status);
CREATE INDEX IF NOT EXISTS idx_agent_dids_registered_at ON agent_dids(registered_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_dids_agent_node_haxen_server ON agent_dids(agent_node_id, haxen_server_id);
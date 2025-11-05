-- Migration Runner: DID Schema Migration Script
-- Description: Complete DID database schema setup for the Haxen platform
-- Created: 2025-01-08
-- 
-- This script creates all necessary tables for the DID (Decentralized Identity) implementation
-- in the Haxen platform, enabling the transition from file-based to database-backed storage.

-- Create migrations tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Insert migration records
INSERT OR IGNORE INTO schema_migrations (version, description) VALUES 
    ('001', 'Create DID Registry table'),
    ('002', 'Create Agent DIDs table'),
    ('003', 'Create Component DIDs table'),
    ('004', 'Create Execution VCs table'),
    ('005', 'Create Workflow VCs table');

-- Performance optimization: Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_execution_vcs_workflow_session ON execution_vcs(workflow_id, session_id);
CREATE INDEX IF NOT EXISTS idx_execution_vcs_issuer_target ON execution_vcs(issuer_did, target_did);
CREATE INDEX IF NOT EXISTS idx_component_dids_type_exposure ON component_dids(component_type, exposure_level);
CREATE INDEX IF NOT EXISTS idx_agent_dids_org_status ON agent_dids(organization_id, status);

-- Create triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_agent_dids_timestamp 
    AFTER UPDATE ON agent_dids
    FOR EACH ROW
    BEGIN
        UPDATE agent_dids SET updated_at = CURRENT_TIMESTAMP WHERE did = NEW.did;
    END;

CREATE TRIGGER IF NOT EXISTS update_component_dids_timestamp 
    AFTER UPDATE ON component_dids
    FOR EACH ROW
    BEGIN
        UPDATE component_dids SET updated_at = CURRENT_TIMESTAMP WHERE did = NEW.did;
    END;

CREATE TRIGGER IF NOT EXISTS update_execution_vcs_timestamp 
    AFTER UPDATE ON execution_vcs
    FOR EACH ROW
    BEGIN
        UPDATE execution_vcs SET updated_at = CURRENT_TIMESTAMP WHERE vc_id = NEW.vc_id;
    END;

CREATE TRIGGER IF NOT EXISTS update_workflow_vcs_timestamp 
    AFTER UPDATE ON workflow_vcs
    FOR EACH ROW
    BEGIN
        UPDATE workflow_vcs SET updated_at = CURRENT_TIMESTAMP WHERE workflow_vc_id = NEW.workflow_vc_id;
    END;

-- Create helpful views for DID management
CREATE VIEW IF NOT EXISTS did_hierarchy_view AS
SELECT 
    dr.organization_id,
    dr.root_did,
    ad.did as agent_did,
    ad.agent_node_id,
    ad.status as agent_status,
    cd.did as component_did,
    cd.component_type,
    cd.function_name,
    cd.exposure_level
FROM did_registry dr
LEFT JOIN agent_dids ad ON dr.organization_id = ad.organization_id
LEFT JOIN component_dids cd ON ad.did = cd.agent_did
ORDER BY dr.organization_id, ad.agent_node_id, cd.component_type, cd.function_name;

-- Create view for VC audit trail
CREATE VIEW IF NOT EXISTS vc_audit_trail AS
SELECT 
    evc.vc_id,
    evc.execution_id,
    evc.workflow_id,
    evc.session_id,
    evc.status,
    evc.created_at,
    issuer.function_name as issuer_function,
    target.function_name as target_function,
    caller.function_name as caller_function,
    wvc.workflow_vc_id,
    wvc.status as workflow_status
FROM execution_vcs evc
LEFT JOIN component_dids issuer ON evc.issuer_did = issuer.did
LEFT JOIN component_dids target ON evc.target_did = target.did
LEFT JOIN component_dids caller ON evc.caller_did = caller.did
LEFT JOIN workflow_vcs wvc ON evc.workflow_id = wvc.workflow_id AND evc.session_id = wvc.session_id
ORDER BY evc.created_at DESC;
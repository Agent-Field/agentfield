-- Migration: Create Workflow VCs table
-- Description: Workflow-level verifiable credentials for complete workflow execution chains
-- Created: 2025-01-08

CREATE TABLE IF NOT EXISTS workflow_vcs (
    workflow_vc_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    component_vc_ids TEXT DEFAULT '[]', -- JSON array of execution VC IDs
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'succeeded', 'failed', 'cancelled', 'timeout')),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_workflow_id ON workflow_vcs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_session_id ON workflow_vcs(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_status ON workflow_vcs(status);
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_start_time ON workflow_vcs(start_time);
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_end_time ON workflow_vcs(end_time);
CREATE INDEX IF NOT EXISTS idx_workflow_vcs_created_at ON workflow_vcs(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_vcs_workflow_session ON workflow_vcs(workflow_id, session_id);

-- Create a view for workflow VC chain analysis
CREATE VIEW IF NOT EXISTS workflow_vc_chain_view AS
SELECT
    wvc.workflow_vc_id,
    wvc.workflow_id,
    wvc.session_id,
    wvc.status as workflow_status,
    wvc.start_time,
    wvc.end_time,
    wvc.total_steps,
    wvc.completed_steps,
    COUNT(evc.vc_id) as actual_execution_vcs,
    GROUP_CONCAT(evc.vc_id) as execution_vc_list,
    AVG(CASE WHEN evc.status = 'completed' THEN 1.0 ELSE 0.0 END) as completion_rate
FROM workflow_vcs wvc
LEFT JOIN execution_vcs evc ON JSON_EXTRACT(wvc.component_vc_ids, '$[*]') LIKE '%' || evc.vc_id || '%'
GROUP BY wvc.workflow_vc_id, wvc.workflow_id, wvc.session_id, wvc.status,
         wvc.start_time, wvc.end_time, wvc.total_steps, wvc.completed_steps;

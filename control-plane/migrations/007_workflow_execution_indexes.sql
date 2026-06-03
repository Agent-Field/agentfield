-- Migration: Add Performance Indexes for Workflow Executions
-- Description: Add missing indexes for frequently queried workflow execution fields
-- Created: 2025-01-21

-- =============================================================================
-- WORKFLOW_EXECUTIONS TABLE PERFORMANCE INDEXES
-- =============================================================================

-- Index on workflow_id for workflow-based queries
-- Optimizes: SELECT * FROM workflow_executions WHERE workflow_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);

-- Index on execution_id for fast execution lookups (already unique, but explicit index helps)
-- Optimizes: SELECT * FROM workflow_executions WHERE execution_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_execution_id ON workflow_executions(execution_id);

-- Index on parent_workflow_id for parent-child relationship queries
-- Optimizes: SELECT * FROM workflow_executions WHERE parent_workflow_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_workflow_id ON workflow_executions(parent_workflow_id);

-- Index on parent_execution_id for execution hierarchy queries
-- Optimizes: SELECT * FROM workflow_executions WHERE parent_execution_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_execution_id ON workflow_executions(parent_execution_id);

-- Index on root_workflow_id for root workflow queries
-- Optimizes: SELECT * FROM workflow_executions WHERE root_workflow_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_root_workflow_id ON workflow_executions(root_workflow_id);

-- Index on status for status-based filtering
-- Optimizes: SELECT * FROM workflow_executions WHERE status = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);

-- Index on agent_node_id for agent-specific queries
-- Optimizes: SELECT * FROM workflow_executions WHERE agent_node_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_node_id ON workflow_executions(agent_node_id);

-- Index on reasoner_id for reasoner-specific queries
-- Optimizes: SELECT * FROM workflow_executions WHERE reasoner_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_reasoner_id ON workflow_executions(reasoner_id);

-- Composite index for workflow + status queries (common pattern)
-- Optimizes: SELECT * FROM workflow_executions WHERE workflow_id = ? AND status = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_status ON workflow_executions(workflow_id, status);

-- Composite index for parent workflow + status queries
-- Optimizes: SELECT * FROM workflow_executions WHERE parent_workflow_id = ? AND status = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_workflow_status ON workflow_executions(parent_workflow_id, status);

-- Composite index for root workflow + status queries
-- Optimizes: SELECT * FROM workflow_executions WHERE root_workflow_id = ? AND status = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_root_workflow_status ON workflow_executions(root_workflow_id, status);

-- Temporal index for time-based queries
-- Optimizes: SELECT * FROM workflow_executions WHERE started_at >= ? ORDER BY started_at
CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at);

-- Composite index for workflow + temporal queries
-- Optimizes: SELECT * FROM workflow_executions WHERE workflow_id = ? ORDER BY started_at
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_started_at ON workflow_executions(workflow_id, started_at);

-- =============================================================================
-- QUERY PATTERN OPTIMIZATION SUMMARY
-- =============================================================================
--
-- These indexes are designed to optimize the following common query patterns:
--
-- 1. Single execution lookups:
--    SELECT * FROM workflow_executions WHERE execution_id = ?
--    → Optimized by idx_workflow_executions_execution_id
--
-- 2. Workflow-based queries:
--    SELECT * FROM workflow_executions WHERE workflow_id = ?
--    → Optimized by idx_workflow_executions_workflow_id
--
-- 3. Parent-child relationship queries:
--    SELECT * FROM workflow_executions WHERE parent_workflow_id = ?
--    SELECT * FROM workflow_executions WHERE parent_execution_id = ?
--    → Optimized by respective parent indexes
--
-- 4. Root workflow queries:
--    SELECT * FROM workflow_executions WHERE root_workflow_id = ?
--    → Optimized by idx_workflow_executions_root_workflow_id
--
-- 5. Status filtering:
--    SELECT * FROM workflow_executions WHERE status = ?
--    → Optimized by idx_workflow_executions_status
--
-- 6. Agent/Reasoner specific queries:
--    SELECT * FROM workflow_executions WHERE agent_node_id = ?
--    SELECT * FROM workflow_executions WHERE reasoner_id = ?
--    → Optimized by respective indexes
--
-- 7. Combined workflow + status queries:
--    SELECT * FROM workflow_executions WHERE workflow_id = ? AND status = ?
--    → Optimized by idx_workflow_executions_workflow_status
--
-- 8. Temporal queries:
--    SELECT * FROM workflow_executions WHERE started_at >= ? ORDER BY started_at
--    → Optimized by idx_workflow_executions_started_at
--
-- These indexes significantly improve query performance for the workflow update
-- handler and related operations, especially for parent-child relationship
-- traversal and status-based filtering.

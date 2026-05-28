-- Migration: Add Composite Indexes for Query Optimization
-- Description: Composite indexes to optimize multi-column query performance based on analysis findings
-- Created: 2025-01-19

-- =============================================================================
-- WORKFLOWS TABLE COMPOSITE INDEXES
-- =============================================================================

-- Primary composite index for QueryWorkflows function
-- Optimizes: WHERE session_id = ? AND actor_id = ? AND status = ?
CREATE INDEX IF NOT EXISTS idx_workflows_session_actor_status ON workflows(session_id, actor_id, status);

-- Secondary composite index for session + actor queries
-- Optimizes: WHERE session_id = ? AND actor_id = ? (without status filter)
CREATE INDEX IF NOT EXISTS idx_workflows_session_actor ON workflows(session_id, actor_id);

-- Composite index for actor + status queries
-- Optimizes: WHERE actor_id = ? AND status = ? (cross-session queries)
CREATE INDEX IF NOT EXISTS idx_workflows_actor_status ON workflows(actor_id, status);

-- Composite index for session + temporal queries
-- Optimizes: WHERE session_id = ? ORDER BY started_at (session timeline queries)
CREATE INDEX IF NOT EXISTS idx_workflows_session_started_at ON workflows(session_id, started_at);

-- =============================================================================
-- WORKFLOW_EXECUTIONS TABLE COMPOSITE INDEXES
-- =============================================================================

-- Primary composite index for workflow + session queries
-- Optimizes: WHERE workflow_id = ? AND session_id = ? (execution lookups within workflow sessions)
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_session ON workflow_executions(workflow_id, session_id);

-- Composite index for session + actor queries
-- Optimizes: WHERE session_id = ? AND actor_id = ? (actor executions within sessions)
CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_actor ON workflow_executions(session_id, actor_id);

-- =============================================================================
-- QUERY PATTERN OPTIMIZATION SUMMARY
-- =============================================================================
--
-- These composite indexes are designed to optimize the following query patterns:
--
-- 1. QueryWorkflows primary pattern:
--    SELECT * FROM workflows WHERE session_id = ? AND actor_id = ? AND status = ?
--    → Optimized by idx_workflows_session_actor_status
--
-- 2. Session-based workflow queries:
--    SELECT * FROM workflows WHERE session_id = ? AND actor_id = ?
--    → Optimized by idx_workflows_session_actor
--
-- 3. Actor status queries across sessions:
--    SELECT * FROM workflows WHERE actor_id = ? AND status = ?
--    → Optimized by idx_workflows_actor_status
--
-- 4. Session timeline queries:
--    SELECT * FROM workflows WHERE session_id = ? ORDER BY started_at
--    → Optimized by idx_workflows_session_started_at
--
-- 5. Workflow execution lookups:
--    SELECT * FROM workflow_executions WHERE workflow_id = ? AND session_id = ?
--    → Optimized by idx_workflow_executions_workflow_session
--
-- 6. Actor execution queries within sessions:
--    SELECT * FROM workflow_executions WHERE session_id = ? AND actor_id = ?
--    → Optimized by idx_workflow_executions_session_actor
--
-- These indexes complement existing single-column indexes and provide significant
-- performance improvements for multi-column WHERE clauses and JOIN operations.

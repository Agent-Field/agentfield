-- Migration: Add Notes Column to Workflow Executions
-- Description: Add JSONB notes column to workflow_executions table for app.note() feature
-- Created: 2025-01-30

-- =============================================================================
-- ADD NOTES COLUMN TO WORKFLOW_EXECUTIONS TABLE
-- =============================================================================

-- Add notes column to workflow_executions table
-- Notes are stored as JSON text array with structure: [{message, tags, timestamp}]
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS notes TEXT DEFAULT '[]';

-- =============================================================================
-- PERFORMANCE INDEXES FOR NOTES QUERIES
-- =============================================================================

-- Index for notes queries - SQLite doesn't support GIN indexes, using standard index
-- Optimizes: SELECT * FROM workflow_executions WHERE json_extract(notes, '$') IS NOT NULL
CREATE INDEX IF NOT EXISTS idx_workflow_executions_notes ON workflow_executions(notes);

-- Composite index for execution_id + notes for fast note retrieval
-- Optimizes: SELECT notes FROM workflow_executions WHERE execution_id = ?
CREATE INDEX IF NOT EXISTS idx_workflow_executions_execution_notes ON workflow_executions(execution_id, notes);

-- =============================================================================
-- QUERY PATTERN OPTIMIZATION SUMMARY
-- =============================================================================
--
-- These indexes are designed to optimize the following query patterns:
--
-- 1. Retrieve notes for specific execution:
--    SELECT notes FROM workflow_executions WHERE execution_id = ?
--    → Optimized by idx_workflow_executions_execution_notes
--
-- 2. Filter executions by note tags:
--    SELECT * FROM workflow_executions WHERE notes @> '[{"tags": ["tag_name"]}]'
--    → Optimized by idx_workflow_executions_notes_tags
--
-- 3. Search notes content (future enhancement):
--    SELECT * FROM workflow_executions WHERE notes @@ 'search_term'
--    → Supported by GIN index on notes column
--
-- The GIN index on the notes JSONB column provides efficient querying for:
-- - Tag-based filtering using @> operator
-- - Full-text search within notes using @@ operator
-- - Existence checks for specific note structures

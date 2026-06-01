-- Migration: Add storage URI metadata to execution VCs
-- Description: Adds columns that support externalising VC documents
-- Created: 2025-02-14

ALTER TABLE execution_vcs
ADD COLUMN storage_uri TEXT DEFAULT '';

ALTER TABLE execution_vcs
ADD COLUMN document_size_bytes INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_execution_vcs_storage_uri ON execution_vcs(storage_uri);

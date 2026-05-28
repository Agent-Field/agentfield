-- Migration: Add storage URI metadata to workflow VCs
-- Description: Adds columns that support externalised workflow VC documents
-- Created: 2025-02-14

ALTER TABLE workflow_vcs
ADD COLUMN storage_uri TEXT DEFAULT '';

ALTER TABLE workflow_vcs
ADD COLUMN document_size_bytes INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_workflow_vcs_storage_uri ON workflow_vcs(storage_uri);

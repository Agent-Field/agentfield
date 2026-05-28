-- Migration: Drop unused tables
-- Description: Remove permission_approvals and protected_agents_config tables
--              that were created but never wired into any service or handler code.

DROP TABLE IF EXISTS protected_agents_config;
DROP TABLE IF EXISTS permission_approvals;

-- 033_agent_instance_id.sql
-- Adds a per-process identifier to agent_nodes so the control plane can
-- detect mid-flight redeploys.
--
-- Background: every Python/Go SDK process generates a fresh instance_id at
-- startup and sends it in the registration payload. When an agent re-registers
-- with a different instance_id than the row currently stores, every still-
-- running execution owned by that agent is failed with status_reason
-- "agent_restart_orphaned" — the previous process is dead and its in-memory
-- wait_for_execution_result polls cannot be resumed, so leaving those
-- executions in `running` would strand the parent reasoner forever.
--
-- Backward compatibility: existing rows get '' (empty), and registrations
-- from older SDKs that don't send instance_id are also '' — the orphan-reap
-- only fires when both stored and incoming are non-empty AND differ. So old
-- agents continue to work; only agents on the updated SDK get the new
-- protection.

ALTER TABLE agent_nodes
    ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT '';

import type { AgentTagSummary } from "@/services/tagApprovalApi";

export type AgentTagRowStatus = "pending_approval" | "active" | "other";

export function getAgentTagRowStatus(agent: AgentTagSummary): AgentTagRowStatus {
  if (agent.lifecycle_status === "pending_approval") return "pending_approval";
  if (
    agent.lifecycle_status === "active" ||
    agent.lifecycle_status === "online" ||
    agent.lifecycle_status === "ready" ||
    agent.lifecycle_status === "offline" ||
    agent.lifecycle_status === "degraded" ||
    agent.lifecycle_status === "starting"
  ) {
    return "active";
  }
  return "other";
}

export function countPendingAgentTags(agents: AgentTagSummary[]): number {
  return agents.filter((a) => getAgentTagRowStatus(a) === "pending_approval").length;
}

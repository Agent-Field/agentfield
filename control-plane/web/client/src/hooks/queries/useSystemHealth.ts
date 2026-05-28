import { useQuery } from "@tanstack/react-query";
import { getGlobalApiKey } from "../../services/api";
import { useSSESync } from "../useSSEQuerySync";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/ui/v1";

export type LLMCircuitState = "closed" | "open" | "half_open";

export interface LLMEndpointHealth {
  name: string;
  healthy: boolean;
  circuit_state: LLMCircuitState;
  consecutive_failures: number;
  last_error?: string;
  last_success?: string;
  last_checked?: string;
  total_checks?: number;
  total_failures?: number;
  url?: string;
  model?: string;
  latency_ms?: number;
  error?: string;
}

export interface LLMHealthResponse {
  enabled: boolean;
  healthy: boolean;
  endpoints: LLMEndpointHealth[];
  checked_at?: string;
}

export interface QueueAgentStatus {
  agent_node_id: string;
  running: number;
  max: number;
  available: number;
}

export interface QueueStatusResponse {
  enabled: boolean;
  max_per_agent: number;
  agents: QueueAgentStatus[];
  total_running: number;
}

async function fetchLLMHealth(): Promise<LLMHealthResponse> {
  const apiKey = getGlobalApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(`${API_BASE_URL}/llm/health`, { headers });
  if (!response.ok) {
    return { enabled: false, healthy: false, endpoints: [] };
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  const rawEndpoints = Array.isArray(payload.endpoints) ? payload.endpoints : [];

  const endpoints = rawEndpoints
    .map((endpoint): LLMEndpointHealth => {
      const item = endpoint as Record<string, unknown>;
      const healthy = Boolean(item.healthy);
      const circuitStateValue = String(item.circuit_state ?? item.circuitState ?? "").toLowerCase();
      const circuit_state: LLMCircuitState =
        circuitStateValue === "closed" || circuitStateValue === "open" || circuitStateValue === "half_open"
          ? circuitStateValue
          : healthy
            ? "closed"
            : "open";

      const consecutiveFailures = Number(
        item.consecutive_failures ?? item.consecutiveFailures ?? 0,
      );

      return {
        name: String(item.name ?? ""),
        healthy,
        circuit_state,
        consecutive_failures: Number.isFinite(consecutiveFailures) ? consecutiveFailures : 0,
        last_error: String(item.last_error ?? item.lastError ?? item.error ?? ""),
        last_success: typeof item.last_success === "string" ? item.last_success : undefined,
        last_checked: typeof item.last_checked === "string" ? item.last_checked : undefined,
        total_checks: Number.isFinite(Number(item.total_checks)) ? Number(item.total_checks) : undefined,
        total_failures: Number.isFinite(Number(item.total_failures)) ? Number(item.total_failures) : undefined,
        url: typeof item.url === "string" ? item.url : undefined,
        model: typeof item.model === "string" ? item.model : undefined,
        latency_ms: Number.isFinite(Number(item.latency_ms)) ? Number(item.latency_ms) : undefined,
        error: typeof item.error === "string" ? item.error : undefined,
      };
    })
    .filter((endpoint) => endpoint.name.length > 0);

  return {
    enabled: Boolean(payload.enabled),
    healthy: Boolean(payload.healthy),
    checked_at: typeof payload.checked_at === "string" ? payload.checked_at : undefined,
    endpoints,
  };
}

async function fetchQueueStatus(): Promise<QueueStatusResponse> {
  const apiKey = getGlobalApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(`${API_BASE_URL}/queue/status`, { headers });
  if (!response.ok) {
    return {
      enabled: false,
      max_per_agent: 0,
      agents: [],
      total_running: 0,
    };
  }

  const payload = await response.json();

  // Backward-compat shim for older API responses that returned agents as a map.
  if (!Array.isArray(payload?.agents) && payload?.agents && typeof payload.agents === "object") {
    const agents: QueueAgentStatus[] = Object.entries(
      payload.agents as Record<string, { running?: number; max_concurrent?: number }>,
    ).map(([agentNodeId, status]) => {
        const max = Number(status.max_concurrent ?? payload.max_per_agent ?? 0);
        const running = Number(status.running ?? 0);
        return {
          agent_node_id: agentNodeId,
          running,
          max,
          available: Math.max(0, max - running),
        };
      });

    return {
      enabled: Number(payload.max_per_agent ?? 0) > 0,
      max_per_agent: Number(payload.max_per_agent ?? 0),
      total_running: Number(
        payload.total_running ??
          agents.reduce((sum: number, item: QueueAgentStatus) => sum + item.running, 0),
      ),
      agents,
    };
  }

  const agents = (Array.isArray(payload?.agents) ? payload.agents : [])
    .map((agent: Partial<QueueAgentStatus>) => {
      const max = Number(agent.max ?? payload?.max_per_agent ?? 0);
      const running = Number(agent.running ?? 0);
      const available = Number(
        agent.available ?? Math.max(0, max - running),
      );
      return {
        agent_node_id: String(agent.agent_node_id ?? ""),
        running,
        max,
        available,
      };
    })
    .filter((agent: QueueAgentStatus) => agent.agent_node_id.length > 0);

  return {
    enabled: Boolean(payload?.enabled),
    max_per_agent: Number(payload?.max_per_agent ?? 0),
    total_running: Number(
      payload?.total_running ??
        agents.reduce((sum: number, item: QueueAgentStatus) => sum + item.running, 0),
    ),
    agents,
  };
}

export function useLLMHealth() {
  const { execConnected } = useSSESync();
  return useQuery<LLMHealthResponse>({
    queryKey: ["llm-health"],
    queryFn: fetchLLMHealth,
    refetchInterval: execConnected ? 5_000 : 3_000,
  });
}

export function useQueueStatus() {
  const { execConnected } = useSSESync();
  return useQuery<QueueStatusResponse>({
    queryKey: ["queue-status"],
    queryFn: fetchQueueStatus,
    refetchInterval: execConnected ? 5_000 : 3_000,
  });
}

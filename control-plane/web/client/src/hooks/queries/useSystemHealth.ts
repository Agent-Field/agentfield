import { useQuery } from "@tanstack/react-query";
import { getGlobalApiKey } from "../../services/api";
import { useSSESync } from "../useSSEQuerySync";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/ui/v1";

export interface LLMEndpointHealth {
  name: string;
  model?: string;
  healthy: boolean;
  latency_ms?: number;
  error?: string;
}

export interface LLMHealthResponse {
  endpoints: LLMEndpointHealth[];
  healthy: boolean;
}

export interface QueueAgentStatus {
  running: number;
  queued: number;
  max_concurrent: number;
}

export interface QueueStatusResponse {
  agents: Record<string, QueueAgentStatus>;
  total_running: number;
}

async function fetchLLMHealth(): Promise<LLMHealthResponse> {
  const apiKey = getGlobalApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(`${API_BASE_URL}/llm/health`, { headers });
  if (!response.ok) return { endpoints: [], healthy: false };
  return response.json();
}

async function fetchQueueStatus(): Promise<QueueStatusResponse> {
  const apiKey = getGlobalApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(`${API_BASE_URL}/queue/status`, { headers });
  if (!response.ok) return { agents: {}, total_running: 0 };
  return response.json();
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

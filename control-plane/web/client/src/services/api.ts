import type {
  AgentNode,
  AgentNodeSummary,
  AgentNodeDetailsForUIWithPackage,
  AppMode,
  EnvResponse,
  SetEnvRequest,
  ConfigSchemaResponse,
  AgentStatus,
  AgentStatusUpdate
} from '../types/agentfield';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';
const STORAGE_KEY = "af_api_key";

// Simple obfuscation for localStorage; not meant as real security.
const decryptKey = (value: string): string => {
  try {
    return atob(value).split("").reverse().join("");
  } catch {
    return "";
  }
};

// Initialize API key from localStorage immediately when this module loads
// This ensures the key is available before any API calls are made
let globalApiKey: string | null = (() => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const key = decryptKey(stored);
      if (key) return key;
    }
  } catch {
    // localStorage might not be available
  }
  return null;
})();

export function setGlobalApiKey(key: string | null) {
  globalApiKey = key;
}

export function getGlobalApiKey(): string | null {
  return globalApiKey;
}

// Admin token for accessing admin-only permission management routes.
// Stored separately from the API key since it provides elevated privileges.
const ADMIN_TOKEN_STORAGE_KEY = "af_admin_token";

let globalAdminToken: string | null = (() => {
  try {
    const stored = localStorage.getItem(ADMIN_TOKEN_STORAGE_KEY);
    if (stored) {
      const key = decryptKey(stored);
      if (key) return key;
    }
  } catch {
    // localStorage might not be available
  }
  return null;
})();

export function setGlobalAdminToken(token: string | null) {
  globalAdminToken = token;
}

export function getGlobalAdminToken(): string | null {
  return globalAdminToken;
}

/**
 * Enhanced fetch wrapper with retry logic and timeout support
 */
async function fetchWrapper<T>(url: string, options?: RequestInit & { timeout?: number }): Promise<T> {
  const { timeout = 10000, ...fetchOptions } = options || {};

  const headers = new Headers(fetchOptions.headers || {});
  if (globalApiKey) {
    headers.set('X-API-Key', globalApiKey);
  }

  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        message: 'Request failed with status ' + response.status
      }));

      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json() as Promise<T>;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }

    throw error;
  }
}

export async function getNodesSummary(): Promise<{ nodes: AgentNodeSummary[], count: number }> {
  return fetchWrapper<{ nodes: AgentNodeSummary[], count: number }>('/nodes/summary');
}

export async function getNodeDetails(nodeId: string): Promise<AgentNode> {
  return fetchWrapper<AgentNode>(`/nodes/${nodeId}/details`);
}

export function streamNodeEvents(): EventSource {
  const apiKey = getGlobalApiKey();
  const url = apiKey
    ? `${API_BASE_URL}/nodes/events?api_key=${encodeURIComponent(apiKey)}`
    : `${API_BASE_URL}/nodes/events`;
  return new EventSource(url);
}

// ============================================================================
// Environment Variable Management API Functions
// ============================================================================

/**
 * Get environment variables for an agent
 */
export async function getAgentEnvironmentVariables(
  agentId: string,
  packageId: string
): Promise<EnvResponse> {
  return fetchWrapper<EnvResponse>(`/agents/${agentId}/env?packageId=${packageId}`);
}

/**
 * Update environment variables for an agent
 */
export async function updateAgentEnvironmentVariables(
  agentId: string,
  packageId: string,
  variables: Record<string, string>
): Promise<{ message: string; agent_id: string; package_id: string }> {
  const request: SetEnvRequest = { variables };

  return fetchWrapper<{ message: string; agent_id: string; package_id: string }>(
    `/agents/${agentId}/env?packageId=${packageId}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    }
  );
}

/**
 * Get configuration schema for an agent
 */
export async function getAgentConfigurationSchema(
  agentId: string,
  packageId: string
): Promise<ConfigSchemaResponse> {
  return fetchWrapper<ConfigSchemaResponse>(`/agents/${agentId}/config/schema?packageId=${packageId}`);
}

/**
 * Enhanced node details with package info
 */
export async function getNodeDetailsWithPackageInfo(
  nodeId: string,
  mode: AppMode = 'user'
): Promise<AgentNodeDetailsForUIWithPackage> {
  return fetchWrapper<AgentNodeDetailsForUIWithPackage>(`/nodes/${nodeId}/details?mode=${mode}`, {
    timeout: 8000 // 8 second timeout for node details
  });
}

// ============================================================================
// Unified Status Management API Functions
// ============================================================================

/**
 * Get unified status for a specific node
 */
export async function getNodeStatus(nodeId: string): Promise<AgentStatus> {
  return fetchWrapper<AgentStatus>(`/nodes/${nodeId}/status`);
}

/**
 * Refresh status for a specific node (manual refresh)
 */
export async function refreshNodeStatus(nodeId: string): Promise<AgentStatus> {
  return fetchWrapper<AgentStatus>(`/nodes/${nodeId}/status/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Get status for multiple nodes (bulk operation)
 */
export async function bulkNodeStatus(nodeIds: string[]): Promise<Record<string, AgentStatus>> {
  return fetchWrapper<Record<string, AgentStatus>>('/nodes/status/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ node_ids: nodeIds })
  });
}

/**
 * Update status for a specific node
 */
export async function updateNodeStatus(
  nodeId: string,
  update: AgentStatusUpdate
): Promise<AgentStatus> {
  return fetchWrapper<AgentStatus>(`/nodes/${nodeId}/status`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update)
  });
}

/**
 * Start an agent with proper state transitions
 */
export async function startAgentWithStatus(nodeId: string): Promise<AgentStatus> {
  return fetchWrapper<AgentStatus>(`/nodes/${nodeId}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Stop an agent with proper state transitions
 */
export async function stopAgentWithStatus(nodeId: string): Promise<AgentStatus> {
  return fetchWrapper<AgentStatus>(`/nodes/${nodeId}/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Subscribe to unified status events via Server-Sent Events
 */
export function subscribeToUnifiedStatusEvents(): EventSource {
  const apiKey = getGlobalApiKey();
  const url = apiKey
    ? `${API_BASE_URL}/nodes/events?api_key=${encodeURIComponent(apiKey)}`
    : `${API_BASE_URL}/nodes/events`;
  return new EventSource(url);
}

// ============================================================================
// Serverless Agent Registration API Functions
// ============================================================================

/**
 * Register a serverless agent by providing its invocation URL
 * The backend will discover the agent's capabilities automatically
 */
export async function registerServerlessAgent(invocationUrl: string): Promise<{
  success: boolean;
  message: string;
  node: {
    id: string;
    version: string;
    deployment_type: string;
    invocation_url: string;
    reasoners_count: number;
    skills_count: number;
  };
}> {
  // Use /api/v1 base for this endpoint (not /api/ui/v1)
  const API_V1_BASE = '/api/v1';
  const timeout = 15000;

  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const headers = new Headers({ 'Content-Type': 'application/json' });
    if (globalApiKey) {
      headers.set('X-API-Key', globalApiKey);
    }

    const response = await fetch(`${API_V1_BASE}/nodes/register-serverless`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ invocation_url: invocationUrl }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        message: 'Request failed with status ' + response.status
      }));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }

    throw error;
  }
}

// ============================================================================
// Agent node process logs (UI proxy → NDJSON)
// ============================================================================

/** NDJSON v1 from agent process log ring (Python / Go / TypeScript SDKs). */
export type NodeLogEntry = {
  v: number;
  seq: number;
  ts: string;
  stream: string;
  line: string;
  truncated?: boolean;
  /** Optional severity when SDKs emit it (e.g. log, info, warn, error). */
  level?: string;
  /** Optional logical source (e.g. sdk id, logger name). */
  source?: string;
};

export type NodeLogProxyEffective = {
  connect_timeout: string;
  stream_idle_timeout: string;
  max_stream_duration: string;
  max_tail_lines: number;
};

export type NodeLogProxySettingsResponse = {
  effective: NodeLogProxyEffective;
  env_locks: Record<string, boolean>;
};

function nodeLogsAuthHeaders(): HeadersInit {
  const h: Record<string, string> = {};
  if (globalApiKey) {
    h["X-API-Key"] = globalApiKey;
  }
  return h;
}

/**
 * Typed error thrown by node-logs fetch/stream helpers.
 *
 * Carries the HTTP status code and (when available) the stable machine code
 * from the response body so callers can branch on structured fields instead
 * of string-matching the human message.
 */
export class NodeLogsError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "NodeLogsError";
    this.status = status;
    this.code = code;
  }
}

async function nodeLogsHttpError(response: Response): Promise<NodeLogsError> {
  let msg = `HTTP ${response.status}`;
  let code: string | undefined;
  try {
    const j = (await response.json()) as {
      message?: string;
      error?: string;
    };
    if (j.error) code = String(j.error);
    if (j.message) msg = j.message;
    else if (j.error) msg = String(j.error);
  } catch {
    try {
      const t = await response.text();
      if (t) msg = t.slice(0, 200);
    } catch {
      /* ignore */
    }
  }
  return new NodeLogsError(msg, response.status, code);
}

export function parseNodeLogsNDJSON(text: string): NodeLogEntry[] {
  const out: NodeLogEntry[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as NodeLogEntry);
    } catch {
      /* skip malformed */
    }
  }
  return out;
}

/**
 * One-shot tail or bounded fetch (not for long-lived follow streams).
 */
export async function fetchNodeLogsText(
  nodeId: string,
  params: { tail_lines?: string; since_seq?: string; follow?: string },
  init?: RequestInit
): Promise<string> {
  const sp = new URLSearchParams();
  if (params.tail_lines != null) sp.set("tail_lines", params.tail_lines);
  if (params.since_seq != null) sp.set("since_seq", params.since_seq);
  if (params.follow != null) sp.set("follow", params.follow);
  const q = sp.toString();
  const path = `/nodes/${encodeURIComponent(nodeId)}/logs${q ? `?${q}` : ""}`;
  const headers = new Headers(nodeLogsAuthHeaders());
  if (init?.headers) {
    const extra = new Headers(init.headers);
    extra.forEach((v, k) => headers.set(k, v));
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    throw await nodeLogsHttpError(response);
  }
  return response.text();
}

/**
 * Stream NDJSON log lines (use follow=1 on the agent/proxy).
 */
export async function* streamNodeLogsEntries(
  nodeId: string,
  params: { tail_lines?: string; since_seq?: string; follow?: string },
  signal: AbortSignal
): AsyncGenerator<NodeLogEntry, void, undefined> {
  const sp = new URLSearchParams();
  if (params.tail_lines != null) sp.set("tail_lines", params.tail_lines);
  if (params.since_seq != null) sp.set("since_seq", params.since_seq);
  if (params.follow != null) sp.set("follow", params.follow);
  const q = sp.toString();
  const path = `/nodes/${encodeURIComponent(nodeId)}/logs${q ? `?${q}` : ""}`;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    signal,
    headers: nodeLogsAuthHeaders(),
  });
  if (!response.ok) {
    throw await nodeLogsHttpError(response);
  }
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buf.trim()) {
        try {
          yield JSON.parse(buf) as NodeLogEntry;
        } catch {
          /* ignore trailing garbage */
        }
      }
      break;
    }
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        yield JSON.parse(line) as NodeLogEntry;
      } catch {
        /* skip */
      }
    }
  }
}

export async function getNodeLogProxySettings(): Promise<NodeLogProxySettingsResponse> {
  return fetchWrapper<NodeLogProxySettingsResponse>("/settings/node-log-proxy");
}

export async function putNodeLogProxySettings(
  body: Partial<{
    connect_timeout: string;
    stream_idle_timeout: string;
    max_stream_duration: string;
    max_tail_lines: number;
  }>
): Promise<{ effective: NodeLogProxyEffective }> {
  return fetchWrapper<{ effective: NodeLogProxyEffective }>(
    "/settings/node-log-proxy",
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

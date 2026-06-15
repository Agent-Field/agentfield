import { getGlobalApiKey } from "./api";

const API_BASE = "/api/v1";

const withAuthHeaders = (headers?: HeadersInit) => {
  const merged = new Headers(headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    merged.set("X-API-Key", apiKey);
  }
  return merged;
};

export interface StartSessionRequest {
  provider?: string;
  transport?: string;
  model?: string;
  voice?: string;
  metadata?: Record<string, unknown>;
}

export interface StartSessionResponse {
  session_id: string;
  target: string;
  provider: string;
  transport: string;
  model?: string;
  voice?: string;
  modalities?: string[];
  tags?: string[];
  tool_targets?: Record<string, string>;
  offer_url?: string;
  tool_url?: string;
  created_at?: string;
}

export interface SessionToolRequest {
  target?: string;
  input?: Record<string, unknown>;
}

export interface SessionToolResponse {
  execution_id?: string;
  run_id?: string;
  workflow_id?: string;
  status?: string;
  [key: string]: unknown;
}

async function fetchJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: withAuthHeaders({
      "Content-Type": "application/json",
      ...options.headers,
    }),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || errorData.error || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export const sessionsApi = {
  startSession(target: string, request: StartSessionRequest = {}): Promise<StartSessionResponse> {
    return fetchJson<StartSessionResponse>(`${API_BASE}/session-targets/${encodeURIComponent(target)}/start`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  invokeTool(sessionId: string, tool: string, request: SessionToolRequest): Promise<SessionToolResponse> {
    return fetchJson<SessionToolResponse>(
      `${API_BASE}/session-instances/${encodeURIComponent(sessionId)}/tools/${encodeURIComponent(tool)}`,
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    );
  },
};

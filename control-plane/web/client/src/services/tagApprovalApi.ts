/**
 * Tag Approval API
 * API client for tag approval admin endpoints
 */

import { getGlobalApiKey, getGlobalAdminToken } from './api';

const API_BASE = '/api/v1';

export interface PendingAgentResponse {
  agent_id: string;
  proposed_tags: string[];
  approved_tags?: string[];
  status: string;
  registered_at: string;
}

export interface TagApprovalRequest {
  approved_tags: string[];
  skill_tags?: Record<string, string[]>;
  reasoner_tags?: Record<string, string[]>;
  reason?: string;
}

export interface TagRejectionRequest {
  reason?: string;
}

async function fetchWithAuth(url: string, options: RequestInit = {}): Promise<Response> {
  const apiKey = getGlobalApiKey();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (apiKey) {
    (headers as Record<string, string>)['X-Api-Key'] = apiKey;
  }

  const adminToken = getGlobalAdminToken();
  if (adminToken) {
    (headers as Record<string, string>)['X-Admin-Token'] = adminToken;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || `Request failed with status ${response.status}`);
  }

  return response;
}

export async function listPendingAgents(): Promise<{ agents: PendingAgentResponse[]; total: number }> {
  const response = await fetchWithAuth(`${API_BASE}/admin/agents/pending`);
  return response.json();
}

export async function approveAgentTags(agentId: string, req: TagApprovalRequest): Promise<any> {
  const response = await fetchWithAuth(`${API_BASE}/admin/agents/${encodeURIComponent(agentId)}/approve-tags`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
  return response.json();
}

export async function rejectAgentTags(agentId: string, req: TagRejectionRequest): Promise<any> {
  const response = await fetchWithAuth(`${API_BASE}/admin/agents/${encodeURIComponent(agentId)}/reject-tags`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
  return response.json();
}

// Agent tag summary from the UI-optimized endpoint
export interface AgentTagSummary {
  agent_id: string;
  proposed_tags: string[];
  approved_tags: string[];
  lifecycle_status: string;
  registered_at: string;
}

// List ALL agents with tag data (uses UI-optimized endpoint)
export async function listAllAgentsWithTags(): Promise<{ agents: AgentTagSummary[]; total: number }> {
  const response = await fetchWithAuth('/api/ui/v1/authorization/agents');
  return response.json();
}

// Revoke agent tags
export async function revokeAgentTags(agentId: string, reason?: string): Promise<any> {
  const response = await fetchWithAuth(`${API_BASE}/admin/agents/${encodeURIComponent(agentId)}/revoke-tags`, {
    method: 'POST',
    body: JSON.stringify({ reason }),
  });
  return response.json();
}

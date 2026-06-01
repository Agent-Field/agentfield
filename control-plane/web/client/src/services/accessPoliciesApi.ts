/**
 * Access Policies API
 * API client for tag-based access policy admin endpoints
 */

import { getGlobalApiKey, getGlobalAdminToken } from './api';

const API_BASE = '/api/v1';

export interface AccessConstraint {
  operator: string; // "<=", ">=", "==", "!=", "<", ">"
  value: string | number;
}

export interface AccessPolicy {
  id: number;
  name: string;
  caller_tags: string[];
  target_tags: string[];
  allow_functions: string[];
  deny_functions: string[];
  constraints?: Record<string, AccessConstraint>;
  action: 'allow' | 'deny';
  priority: number;
  enabled: boolean;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface AccessPolicyRequest {
  name: string;
  caller_tags: string[];
  target_tags: string[];
  allow_functions?: string[];
  deny_functions?: string[];
  constraints?: Record<string, AccessConstraint>;
  action: 'allow' | 'deny';
  priority?: number;
  description?: string;
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

export async function listPolicies(): Promise<{ policies: AccessPolicy[]; total: number }> {
  const response = await fetchWithAuth(`${API_BASE}/admin/policies`);
  return response.json();
}

export async function getPolicy(id: number): Promise<AccessPolicy> {
  const response = await fetchWithAuth(`${API_BASE}/admin/policies/${id}`);
  return response.json();
}

export async function createPolicy(req: AccessPolicyRequest): Promise<AccessPolicy> {
  const response = await fetchWithAuth(`${API_BASE}/admin/policies`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
  return response.json();
}

export async function updatePolicy(id: number, req: AccessPolicyRequest): Promise<AccessPolicy> {
  const response = await fetchWithAuth(`${API_BASE}/admin/policies/${id}`, {
    method: 'PUT',
    body: JSON.stringify(req),
  });
  return response.json();
}

export async function deletePolicy(id: number): Promise<void> {
  await fetchWithAuth(`${API_BASE}/admin/policies/${id}`, {
    method: 'DELETE',
  });
}

export async function listKnownTags(): Promise<{ tags: string[]; total: number }> {
  const response = await fetchWithAuth(`${API_BASE}/admin/tags`);
  return response.json();
}

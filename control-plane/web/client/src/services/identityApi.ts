import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/ui/v1";

async function fetchWrapper<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }
  const response = await fetch(`${API_BASE_URL}${url}`, { ...options, headers });
  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({
        message: "Request failed with status " + response.status,
      }));
    throw new Error(
      errorData.message || `HTTP error! status: ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}

// DID Explorer Types
export interface DIDStatsResponse {
  total_agents: number;
  total_reasoners: number;
  total_skills: number;
  total_dids: number;
}

export interface DIDStats extends DIDStatsResponse {}

export interface DIDSearchResult {
  type: "agent" | "reasoner" | "skill";
  did: string;
  id: string; // Add missing id property
  name: string;
  parent_did?: string;
  parent_name?: string;
  derivation_path: string;
  status?: string;
  created_at: string;
}

export interface ComponentDIDInfo {
  did: string;
  name: string;
  component_name: string;
  type: "reasoner" | "skill";
  derivation_path: string;
  created_at: string;
}

export interface AgentDIDResponse {
  did: string;
  agent_node_id: string;
  status: string;
  derivation_path: string;
  created_at: string;
  reasoner_count: number;
  skill_count: number;
  reasoners?: ComponentDIDInfo[];
  skills?: ComponentDIDInfo[];
}

export interface AgentDIDsResponse extends AgentDIDResponse {}

// Alias for compatibility
export type AgentDID = AgentDIDResponse;

export interface AgentDetailsResponse {
  agent: AgentDIDResponse;
  total_reasoners: number;
  reasoners_limit: number;
  reasoners_offset: number;
  reasoners_has_more: boolean;
}

// VerifiableCredential interface for Credentials
export interface VerifiableCredential {
  vc_id: string;
  execution_id: string;
  workflow_id: string;
  session_id?: string;
  issuer_did: string;
  target_did: string;
  caller_did: string;
  reasoner_id: string;
  status: string;
  created_at: string;
  duration_ms?: number;
  verified: boolean;
  input_hash?: string;
  output_hash?: string;
  vc_json: any;
}

// Credentials Types
export interface VCSearchResult {
  vc_id: string;
  execution_id: string;
  workflow_id: string;
  workflow_name?: string;
  session_id: string;
  issuer_did: string;
  target_did: string;
  caller_did: string;
  status: string;
  created_at: string;
  duration_ms?: number;
  reasoner_id?: string;
  reasoner_name?: string;
  agent_name?: string;
  agent_node_id?: string;
  verified: boolean;
  input_hash?: string;
  output_hash?: string;
}

// DID Explorer API

export async function getDIDStats(): Promise<DIDStats> {
  return fetchWrapper<DIDStats>("/identity/dids/stats");
}

export async function searchDIDs(
  query: string,
  type: "all" | "agent" | "reasoner" | "skill" = "all",
  limit: number = 20,
  offset: number = 0
): Promise<{
  results: DIDSearchResult[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}> {
  const params = new URLSearchParams({
    q: query,
    type,
    limit: limit.toString(),
    offset: offset.toString(),
  });

  return fetchWrapper(`/identity/dids/search?${params.toString()}`);
}

export async function listAgents(
  limit: number = 10,
  offset: number = 0
): Promise<{
  agents: AgentDIDResponse[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  return fetchWrapper(`/identity/agents?${params.toString()}`);
}

export async function getAgentDetails(
  agentId: string,
  limit: number = 20,
  offset: number = 0
): Promise<AgentDetailsResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  return fetchWrapper(`/identity/agents/${agentId}/details?${params.toString()}`);
}

// Credentials API

export async function searchCredentials(filters: {
  workflow_id?: string;
  session_id?: string;
  status?: string;
  issuer_did?: string;
  agent_node_id?: string;
  execution_id?: string;
  caller_did?: string;
  target_did?: string;
  query?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}): Promise<{
  credentials: VCSearchResult[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}> {
  const params = new URLSearchParams();

  if (filters.workflow_id) params.append("workflow_id", filters.workflow_id);
  if (filters.session_id) params.append("session_id", filters.session_id);
  if (filters.status) params.append("status", filters.status);
  if (filters.issuer_did) params.append("issuer_did", filters.issuer_did);
  if (filters.agent_node_id) params.append("agent_node_id", filters.agent_node_id);
  if (filters.execution_id) params.append("execution_id", filters.execution_id);
  if (filters.caller_did) params.append("caller_did", filters.caller_did);
  if (filters.target_did) params.append("target_did", filters.target_did);
  if (filters.query) params.append("query", filters.query);
  if (filters.start_time) params.append("start_time", filters.start_time);
  if (filters.end_time) params.append("end_time", filters.end_time);
  if (filters.limit) params.append("limit", filters.limit.toString());
  if (filters.offset) params.append("offset", filters.offset.toString());

  return fetchWrapper(`/identity/credentials/search?${params.toString()}`);
}

// Missing function for getAgentDIDs
export async function getAgentDIDs(
  agentId: string,
  limit: number = 20,
  offset: number = 0
): Promise<{
  reasoners: ComponentDIDInfo[];
  skills: ComponentDIDInfo[];
  total_reasoners: number;
  total_skills: number;
}> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });

  return fetchWrapper(`/identity/agents/${agentId}/dids?${params.toString()}`);
}

// Export default object for compatibility
const identityApi = {
  searchCredentials,
  getDIDStats,
  searchDIDs,
  listAgents,
  getAgentDetails,
  getAgentDIDs,
};

export default identityApi;

import type {
  AgentDIDInfo,
  DIDRegistrationRequest,
  DIDRegistrationResponse,
  DIDFilters,
  DIDStatusSummary
} from '../types/did';
import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

async function fetchWrapper<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }
  const response = await fetch(`${API_BASE_URL}${url}`, { ...options, headers });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      message: 'Request failed with status ' + response.status
    }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// DID Management API Functions

/**
 * Register an agent with DIDs for reasoners and skills
 */
export async function registerAgentDID(
  request: DIDRegistrationRequest
): Promise<DIDRegistrationResponse> {
  return fetchWrapper<DIDRegistrationResponse>('/did/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
}

/**
 * Resolve a DID to get identity information
 */
export async function resolveDID(did: string): Promise<{
  did: string;
  public_key_jwk: any;
  component_type: string;
  function_name?: string;
  derivation_path: string;
}> {
  return fetchWrapper(`/did/resolve/${encodeURIComponent(did)}`);
}

/**
 * Get DID information for a specific agent node
 */
export async function getAgentDIDInfo(nodeId: string): Promise<AgentDIDInfo> {
  return fetchWrapper<AgentDIDInfo>(`/nodes/${nodeId}/did`);
}

/**
 * Get DID status summary for a node (for UI indicators)
 */
export async function getDIDStatusSummary(nodeId: string): Promise<DIDStatusSummary> {
  const didInfo = await getAgentDIDInfo(nodeId).catch(() => null);

  if (!didInfo) {
    return {
      has_did: false,
      did_status: 'inactive',
      reasoner_count: 0,
      skill_count: 0,
      last_updated: ''
    };
  }

  return {
    has_did: true,
    did_status: didInfo.status,
    reasoner_count: Object.keys(didInfo.reasoners).length,
    skill_count: Object.keys(didInfo.skills).length,
    last_updated: didInfo.registered_at
  };
}

/**
 * Get W3C DID Document for a DID
 */
export async function getDIDDocument(did: string): Promise<{
  '@context': string[];
  id: string;
  verificationMethod: any[];
  authentication: string[];
  assertionMethod: string[];
  service: any[];
}> {
  return fetchWrapper(`/did/document/${encodeURIComponent(did)}`);
}

/**
 * Get DID system status
 */
export async function getDIDSystemStatus(): Promise<{
  status: string;
  message: string;
  timestamp: string;
}> {
  return fetchWrapper('/did/status');
}

/**
 * List all agent DIDs with optional filtering
 */
export async function listAgentDIDs(filters?: DIDFilters): Promise<string[]> {
  const queryParams = new URLSearchParams();

  if (filters) {
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        queryParams.append(key, value.toString());
      }
    });
  }

  const queryString = queryParams.toString();
  const url = `/did/agents${queryString ? `?${queryString}` : ''}`;

  const response = await fetchWrapper<{ agent_dids: string[] }>(url);
  return response.agent_dids;
}

/**
 * Copy DID to clipboard with user feedback
 */
export async function copyDIDToClipboard(did: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(did);
    return true;
  } catch (error) {
    console.error('Failed to copy DID to clipboard:', error);
    return false;
  }
}

/**
 * Format DID for display (truncate middle for UI)
 */
export function formatDIDForDisplay(did: string, maxLength: number = 20): string {
  if (did.length <= maxLength) {
    return did;
  }

  const start = did.substring(0, Math.floor(maxLength / 2) - 2);
  const end = did.substring(did.length - Math.floor(maxLength / 2) + 2);
  return `${start}...${end}`;
}

/**
 * Validate DID format
 */
export function isValidDID(did: string): boolean {
  // Basic DID format validation: did:method:identifier
  const didRegex = /^did:[a-z0-9]+:[a-zA-Z0-9._-]+$/;
  return didRegex.test(did);
}

/**
 * Get DID method from DID string
 */
export function getDIDMethod(did: string): string | null {
  const parts = did.split(':');
  return parts.length >= 2 ? parts[1] : null;
}

/**
 * Get DID identifier from DID string
 */
export function getDIDIdentifier(did: string): string | null {
  const parts = did.split(':');
  return parts.length >= 3 ? parts.slice(2).join(':') : null;
}

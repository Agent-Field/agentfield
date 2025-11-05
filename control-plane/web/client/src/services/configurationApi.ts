import type { ConfigurationSchema, AgentConfiguration, AgentPackage, AgentLifecycleInfo } from '../types/agentfield';

const API_BASE = '/api/ui/v1';

export class ConfigurationApiError extends Error {
  public status?: number;
  
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ConfigurationApiError';
    this.status = status;
  }
}

const fetchWithTimeout = async (url: string, options: RequestInit & { timeout?: number } = {}) => {
  const { timeout = 10000, ...fetchOptions } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ConfigurationApiError(`Request timeout after ${timeout}ms`, 408);
    }
    throw error;
  }
};

const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const errorText = await response.text();
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    
    try {
      const errorData = JSON.parse(errorText);
      errorMessage = errorData.error || errorData.message || errorMessage;
    } catch {
      // If not JSON, use the text as is
      if (errorText) {
        errorMessage = errorText;
      }
    }
    
    throw new ConfigurationApiError(errorMessage, response.status);
  }
  
  return response.json();
};

/**
 * Environment file management API
 */
export interface EnvResponse {
  agent_id: string;
  package_id: string;
  variables: Record<string, string>;
  masked_keys: string[];
  file_exists: boolean;
  last_modified?: string;
}

export const getAgentEnvFile = async (
  agentId: string,
  packageId: string
): Promise<EnvResponse> => {
  const url = `${API_BASE}/agents/${agentId}/env?packageId=${encodeURIComponent(packageId)}`;
  const response = await fetch(url);
  return handleResponse(response);
};

export const setAgentEnvFile = async (
  agentId: string,
  packageId: string,
  variables: Record<string, string>
): Promise<void> => {
  const url = `${API_BASE}/agents/${agentId}/env?packageId=${encodeURIComponent(packageId)}`;
  const response = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ variables }),
  });
  await handleResponse(response);
};

export const patchAgentEnvFile = async (
  agentId: string,
  packageId: string,
  variables: Record<string, string>
): Promise<void> => {
  const url = `${API_BASE}/agents/${agentId}/env?packageId=${encodeURIComponent(packageId)}`;
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ variables }),
  });
  await handleResponse(response);
};

export const deleteAgentEnvVar = async (
  agentId: string,
  packageId: string,
  key: string
): Promise<void> => {
  const url = `${API_BASE}/agents/${agentId}/env/${encodeURIComponent(key)}?packageId=${encodeURIComponent(packageId)}`;
  const response = await fetch(url, { method: 'DELETE' });
  await handleResponse(response);
};

// Configuration Schema API
export const getConfigurationSchema = async (agentId: string): Promise<ConfigurationSchema> => {
  const response = await fetch(`${API_BASE}/agents/${agentId}/config/schema`);
  return handleResponse(response);
};

// Configuration Management API
export const getAgentConfiguration = async (agentId: string): Promise<AgentConfiguration> => {
  const response = await fetch(`${API_BASE}/agents/${agentId}/config`);
  return handleResponse(response);
};

export const setAgentConfiguration = async (
  agentId: string, 
  configuration: AgentConfiguration
): Promise<void> => {
  const response = await fetch(`${API_BASE}/agents/${agentId}/config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(configuration),
  });
  
  await handleResponse(response);
};

// Package Management API
export const getAgentPackages = async (search?: string): Promise<AgentPackage[]> => {
  const url = new URL(`${API_BASE}/agents/packages`, window.location.origin);
  if (search) {
    url.searchParams.set('search', search);
  }
  
  const response = await fetch(url.toString());
  return handleResponse(response);
};

export const getAgentPackageDetails = async (packageId: string): Promise<AgentPackage> => {
  const response = await fetch(`${API_BASE}/agents/packages/${packageId}/details`);
  return handleResponse(response);
};

// Agent Lifecycle Management API
export const startAgent = async (agentId: string): Promise<AgentLifecycleInfo> => {
  const response = await fetchWithTimeout(`${API_BASE}/agents/${agentId}/start`, {
    method: 'POST',
    timeout: 5000 // 5 second timeout for start operations
  });
  return handleResponse(response);
};

export const stopAgent = async (agentId: string): Promise<void> => {
  const response = await fetchWithTimeout(`${API_BASE}/agents/${agentId}/stop`, {
    method: 'POST',
    timeout: 5000 // 5 second timeout for stop operations
  });
  await handleResponse(response);
};

export const reconcileAgent = async (agentId: string): Promise<any> => {
  const response = await fetchWithTimeout(`${API_BASE}/agents/${agentId}/reconcile`, {
    method: 'POST',
    timeout: 3000 // 3 second timeout for reconcile operations
  });
  return handleResponse(response);
};

export const getAgentStatus = async (agentId: string): Promise<AgentLifecycleInfo> => {
  const response = await fetch(`${API_BASE}/agents/${agentId}/status`);
  return handleResponse(response);
};

export const getRunningAgents = async (): Promise<AgentLifecycleInfo[]> => {
  const response = await fetch(`${API_BASE}/agents/running`);
  return handleResponse(response);
};

// Utility functions for configuration management
export const isAgentConfigured = (pkg: AgentPackage): boolean => {
  return pkg.configuration_status === 'configured';
};

export const isAgentPartiallyConfigured = (pkg: AgentPackage): boolean => {
  return pkg.configuration_status === 'partially_configured';
};

export const getConfigurationStatusBadge = (status: AgentPackage['configuration_status']) => {
  switch (status) {
    case 'configured':
      return { variant: 'default' as const, label: 'Configured', color: 'green' };
    case 'partially_configured':
      return { variant: 'secondary' as const, label: 'Partially Configured', color: 'yellow' };
    case 'not_configured':
      return { variant: 'outline' as const, label: 'Not Configured', color: 'gray' };
    default:
      return { variant: 'outline' as const, label: 'Unknown', color: 'gray' };
  }
};

export const getAgentStatusBadge = (status: AgentLifecycleInfo['status']) => {
  switch (status) {
    case 'running':
      return { variant: 'default' as const, label: 'Running', color: 'green' };
    case 'stopped':
      return { variant: 'secondary' as const, label: 'Stopped', color: 'gray' };
    case 'starting':
      return { variant: 'secondary' as const, label: 'Starting', color: 'blue' };
    case 'stopping':
      return { variant: 'secondary' as const, label: 'Stopping', color: 'orange' };
    case 'error':
      return { variant: 'destructive' as const, label: 'Error', color: 'red' };
    default:
      return { variant: 'outline' as const, label: 'Unknown', color: 'gray' };
  }
};

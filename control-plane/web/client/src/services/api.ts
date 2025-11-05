import type {
  AgentNode,
  AgentNodeSummary,
  AgentNodeDetailsForUI,
  AgentNodeDetailsForUIWithPackage,
  MCPHealthResponse,
  MCPServerActionResponse,
  MCPToolsResponse,
  MCPOverallStatusResponse,
  MCPToolTestRequest,
  MCPToolTestResponse,
  MCPServerMetricsResponse,
  MCPHealthEventResponse,
  MCPHealthResponseModeAware,
  MCPError,
  AppMode,
  EnvResponse,
  SetEnvRequest,
  ConfigSchemaResponse,
  AgentStatus,
  AgentStatusUpdate
} from '../types/haxen';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

/**
 * Enhanced fetch wrapper with MCP-specific error handling, retry logic, and timeout support
 */
async function fetchWrapper<T>(url: string, options?: RequestInit & { timeout?: number }): Promise<T> {
  const { timeout = 10000, ...fetchOptions } = options || {};

  // Create AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      ...fetchOptions,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        message: 'Request failed with status ' + response.status
      }));

      // Create MCP-specific error if applicable
      if (url.includes('/mcp/') && errorData.code) {
        const mcpError = new Error(errorData.message || `HTTP error! status: ${response.status}`) as MCPError;
        mcpError.code = errorData.code;
        mcpError.details = errorData.details;
        mcpError.isRetryable = errorData.is_retryable || false;
        mcpError.retryAfterMs = errorData.retry_after_ms;
        throw mcpError;
      }

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

/**
 * Retry wrapper for MCP operations with exponential backoff
 */
async function retryMCPOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 1000
): Promise<T> {
  let lastError: MCPError | Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as MCPError | Error;

      // Don't retry if it's not an MCP error or not retryable
      if (!('isRetryable' in lastError) || !lastError.isRetryable) {
        throw lastError;
      }

      // Don't retry on last attempt
      if (attempt === maxRetries) {
        throw lastError;
      }

      // Calculate delay with exponential backoff
      const delay = lastError.retryAfterMs || (baseDelayMs * Math.pow(2, attempt));
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

export async function getNodesSummary(): Promise<{ nodes: AgentNodeSummary[], count: number }> {
  return fetchWrapper<{ nodes: AgentNodeSummary[], count: number }>('/nodes/summary');
}

export async function getNodeDetails(nodeId: string): Promise<AgentNode> {
  return fetchWrapper<AgentNode>(`/nodes/${nodeId}/details`);
}

export function streamNodeEvents(): EventSource {
  return new EventSource(`${API_BASE_URL}/nodes/events`);
}

// ============================================================================
// MCP (Model Context Protocol) API Functions
// ============================================================================

// MCP Health API
export async function getMCPHealth(
  nodeId: string,
  mode: AppMode = 'user'
): Promise<MCPHealthResponse> {
  return fetchWrapper<MCPHealthResponse>(`/nodes/${nodeId}/mcp/health?mode=${mode}`);
}

// MCP Server Management
/**
 * Restart a specific MCP server with retry logic
 */
export async function restartMCPServer(
  nodeId: string,
  serverId: string
): Promise<MCPServerActionResponse> {
  return retryMCPOperation(() =>
    fetchWrapper<MCPServerActionResponse>(`/nodes/${nodeId}/mcp/servers/${serverId}/restart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
  );
}

/**
 * Stop a specific MCP server
 */
export async function stopMCPServer(
  nodeId: string,
  serverId: string
): Promise<MCPServerActionResponse> {
  return retryMCPOperation(() =>
    fetchWrapper<MCPServerActionResponse>(`/nodes/${nodeId}/mcp/servers/${serverId}/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
  );
}

/**
 * Start a specific MCP server
 */
export async function startMCPServer(
  nodeId: string,
  serverId: string
): Promise<MCPServerActionResponse> {
  return retryMCPOperation(() =>
    fetchWrapper<MCPServerActionResponse>(`/nodes/${nodeId}/mcp/servers/${serverId}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
  );
}

// MCP Tools API
export async function getMCPTools(
  nodeId: string,
  alias: string
): Promise<MCPToolsResponse> {
  return fetchWrapper<MCPToolsResponse>(`/nodes/${nodeId}/mcp/servers/${alias}/tools`);
}

// Overall MCP Status
export async function getOverallMCPStatus(
  mode: AppMode = 'user'
): Promise<MCPOverallStatusResponse> {
  return fetchWrapper<MCPOverallStatusResponse>(`/mcp/status?mode=${mode}`);
}

// Enhanced Node Details with MCP
export async function getNodeDetailsWithMCP(
  nodeId: string,
  mode: AppMode = 'user'
): Promise<AgentNodeDetailsForUI> {
  return fetchWrapper<AgentNodeDetailsForUI>(`/nodes/${nodeId}/details?include_mcp=true&mode=${mode}`, {
    timeout: 8000 // 8 second timeout for node details
  });
}

// ============================================================================
// Enhanced MCP API Functions
// ============================================================================

/**
 * Test MCP tool execution with parameters
 */
export async function testMCPTool(
  nodeId: string,
  serverId: string,
  toolName: string,
  params: Record<string, any>,
  timeoutMs?: number
): Promise<MCPToolTestResponse> {
  const request: MCPToolTestRequest = {
    node_id: nodeId,
    server_alias: serverId,
    tool_name: toolName,
    parameters: params,
    timeout_ms: timeoutMs
  };

  return retryMCPOperation(() =>
    fetchWrapper<MCPToolTestResponse>(`/nodes/${nodeId}/mcp/servers/${serverId}/tools/${toolName}/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
  );
}

/**
 * Get MCP server performance metrics
 */
export async function getMCPServerMetrics(
  nodeId: string,
  serverId?: string
): Promise<MCPServerMetricsResponse> {
  const endpoint = serverId
    ? `/nodes/${nodeId}/mcp/servers/${serverId}/metrics`
    : `/nodes/${nodeId}/mcp/metrics`;

  return fetchWrapper<MCPServerMetricsResponse>(endpoint);
}

/**
 * Subscribe to MCP health events via Server-Sent Events
 */
export function subscribeMCPHealthEvents(nodeId: string): EventSource {
  return new EventSource(`${API_BASE_URL}/nodes/${nodeId}/mcp/events`);
}

/**
 * Get recent MCP health events
 */
export async function getMCPHealthEvents(
  nodeId: string,
  limit: number = 50,
  since?: string
): Promise<MCPHealthEventResponse> {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (since) {
    params.append('since', since);
  }

  return fetchWrapper<MCPHealthEventResponse>(`/nodes/${nodeId}/mcp/events/history?${params}`);
}

/**
 * Enhanced MCP health check with mode-aware responses
 */
export async function getMCPHealthModeAware(
  nodeId: string,
  mode: AppMode = 'user'
): Promise<MCPHealthResponseModeAware> {
  return fetchWrapper<MCPHealthResponseModeAware>(`/nodes/${nodeId}/mcp/health?mode=${mode}`, {
    timeout: 5000 // 5 second timeout for MCP health checks
  });
}

/**
 * Bulk MCP server actions (start/stop/restart multiple servers)
 */
export async function bulkMCPServerAction(
  nodeId: string,
  serverIds: string[],
  action: 'start' | 'stop' | 'restart'
): Promise<MCPServerActionResponse[]> {
  return retryMCPOperation(() =>
    fetchWrapper<MCPServerActionResponse[]>(`/nodes/${nodeId}/mcp/servers/bulk/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_ids: serverIds })
    })
  );
}

/**
 * Get MCP server configuration
 */
export async function getMCPServerConfig(
  nodeId: string,
  serverId: string
): Promise<{ config: Record<string, any>; schema?: Record<string, any> }> {
  return fetchWrapper<{ config: Record<string, any>; schema?: Record<string, any> }>(
    `/nodes/${nodeId}/mcp/servers/${serverId}/config`
  );
}

/**
 * Update MCP server configuration
 */
export async function updateMCPServerConfig(
  nodeId: string,
  serverId: string,
  config: Record<string, any>
): Promise<MCPServerActionResponse> {
  return retryMCPOperation(() =>
    fetchWrapper<MCPServerActionResponse>(`/nodes/${nodeId}/mcp/servers/${serverId}/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config })
    })
  );
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
  return fetchWrapper<AgentNodeDetailsForUIWithPackage>(`/nodes/${nodeId}/details?include_mcp=true&mode=${mode}`, {
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
  return new EventSource(`${API_BASE_URL}/nodes/events`);
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
    const response = await fetch(`${API_V1_BASE}/nodes/register-serverless`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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

import type {
  MCPSummaryForUI,
  MCPServerHealthForUI,
  MCPServerMetrics,
  MCPServerStatus,
  MCPHealthEvent,
  AppMode
} from '../types/haxen';

/**
 * Status color mapping for MCP servers
 */
export const MCP_STATUS_COLORS: Record<MCPServerStatus, string> = {
  running: '#10b981', // green-500
  stopped: '#6b7280', // gray-500
  error: '#ef4444', // red-500
  starting: '#f59e0b', // amber-500
  unknown: '#9ca3af' // gray-400
} as const;

/**
 * Status icons for MCP servers
 */
export const MCP_STATUS_ICONS: Record<MCPServerStatus, string> = {
  running: '●',
  stopped: '○',
  error: '✕',
  starting: '◐',
  unknown: '?'
} as const;

/**
 * Get color for MCP server status
 */
export function getMCPStatusColor(status: string): string {
  if (status in MCP_STATUS_COLORS) {
    return MCP_STATUS_COLORS[status as MCPServerStatus];
  }
  return MCP_STATUS_COLORS.unknown;
}

/**
 * Get icon for MCP server status
 */
export function getMCPStatusIcon(status: string): string {
  if (status in MCP_STATUS_ICONS) {
    return MCP_STATUS_ICONS[status as MCPServerStatus];
  }
  return MCP_STATUS_ICONS.unknown;
}

/**
 * Calculate overall health score from server health data
 */
export function calculateOverallHealth(servers: MCPServerHealthForUI[]): number {
  if (servers.length === 0) return 0;

  const weights: Record<MCPServerStatus, number> = {
    running: 1.0,
    starting: 0.7,
    stopped: 0.3,
    error: 0.0,
    unknown: 0.1
  };

  const totalWeight = servers.reduce((sum, server) => {
    const weight = weights[server.status] || 0;
    return sum + weight;
  }, 0);

  return Math.round((totalWeight / servers.length) * 100);
}

/**
 * Aggregate MCP summary from multiple servers
 */
export function aggregateMCPSummary(servers: MCPServerHealthForUI[]): MCPSummaryForUI {
  const runningServers = servers.filter(s => s.status === 'running').length;
  const totalTools = servers.reduce((sum, server) => sum + (server.tool_count || 0), 0);
  const overallHealth = calculateOverallHealth(servers);
  const hasIssues = servers.some(s => s.status === 'error' || s.error_message);

  let serviceStatus: 'ready' | 'degraded' | 'unavailable' = 'ready';
  if (runningServers === 0) {
    serviceStatus = 'unavailable';
  } else if (runningServers < servers.length || hasIssues) {
    serviceStatus = 'degraded';
  }

  return {
    total_servers: servers.length,
    running_servers: runningServers,
    total_tools: totalTools,
    overall_health: overallHealth,
    has_issues: hasIssues,
    capabilities_available: runningServers > 0,
    service_status: serviceStatus
  };
}

/**
 * Format uptime duration
 */
export function formatUptime(startedAt: string | undefined): string {
  if (!startedAt) return 'Unknown';

  const start = new Date(startedAt);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();

  if (diffMs < 0) return 'Unknown';

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  } else if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

/**
 * Format response time in human-readable format
 */
export function formatResponseTime(timeMs: number | undefined): string {
  if (timeMs === undefined || timeMs === null) return 'N/A';
  
  if (timeMs < 1000) {
    return `${Math.round(timeMs)}ms`;
  } else {
    return `${(timeMs / 1000).toFixed(1)}s`;
  }
}

/**
 * Format success rate as percentage
 */
export function formatSuccessRate(rate: number | undefined): string {
  if (rate === undefined || rate === null) return 'N/A';
  return `${Math.round(rate * 100)}%`;
}

/**
 * Format error rate as percentage
 */
export function formatErrorRate(rate: number | undefined): string {
  if (rate === undefined || rate === null) return 'N/A';
  return `${Math.round(rate)}%`;
}

/**
 * Format memory usage
 */
export function formatMemoryUsage(memoryMb: number | undefined): string {
  if (memoryMb === undefined || memoryMb === null) return 'N/A';
  
  if (memoryMb < 1024) {
    return `${Math.round(memoryMb)}MB`;
  } else {
    return `${(memoryMb / 1024).toFixed(1)}GB`;
  }
}

/**
 * Format CPU usage as percentage
 */
export function formatCpuUsage(cpuPercent: number | undefined): string {
  if (cpuPercent === undefined || cpuPercent === null) return 'N/A';
  return `${Math.round(cpuPercent)}%`;
}

/**
 * Format timestamp for display
 */
export function formatTimestamp(timestamp: string | undefined, mode: AppMode = 'user'): string {
  if (!timestamp) return 'Unknown';

  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));

  if (mode === 'user') {
    // User-friendly relative time
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  } else {
    // Developer mode - more precise
    return date.toLocaleString();
  }
}

/**
 * Get health status text based on health score
 */
export function getHealthStatusText(healthScore: number): string {
  if (healthScore >= 90) return 'Excellent';
  if (healthScore >= 75) return 'Good';
  if (healthScore >= 50) return 'Fair';
  if (healthScore >= 25) return 'Poor';
  return 'Critical';
}

/**
 * Get health status color based on health score
 */
export function getHealthStatusColor(healthScore: number): string {
  if (healthScore >= 90) return '#10b981'; // green
  if (healthScore >= 75) return '#84cc16'; // lime
  if (healthScore >= 50) return '#f59e0b'; // amber
  if (healthScore >= 25) return '#f97316'; // orange
  return '#ef4444'; // red
}

/**
 * Format error message for display based on mode
 */
export function formatErrorMessage(
  error: string | undefined,
  mode: AppMode = 'user'
): string {
  if (!error) return '';

  if (mode === 'user') {
    // Simplify error messages for users
    if (error.includes('connection refused')) return 'Connection failed';
    if (error.includes('timeout')) return 'Request timed out';
    if (error.includes('not found')) return 'Service not found';
    if (error.includes('permission denied')) return 'Access denied';
    if (error.includes('invalid')) return 'Invalid request';
    
    // Return first sentence for other errors
    const firstSentence = error.split('.')[0];
    return firstSentence.length > 100 
      ? firstSentence.substring(0, 100) + '...'
      : firstSentence;
  }

  // Developer mode - return full error
  return error;
}

/**
 * Calculate performance metrics from server metrics
 */
export function calculatePerformanceMetrics(metrics: MCPServerMetrics) {
  const successRate = metrics.total_requests > 0 
    ? metrics.successful_requests / metrics.total_requests 
    : 0;

  const errorRate = metrics.total_requests > 0
    ? (metrics.failed_requests / metrics.total_requests) * 100
    : 0;

  return {
    successRate,
    errorRate,
    avgResponseTime: metrics.avg_response_time_ms,
    peakResponseTime: metrics.peak_response_time_ms,
    requestsPerMinute: metrics.requests_per_minute,
    uptime: metrics.uptime_seconds
  };
}

/**
 * Determine if a server needs attention based on metrics
 */
export function serverNeedsAttention(server: MCPServerHealthForUI): boolean {
  if (server.status === 'error') return true;
  if (server.error_message) return true;
  if (server.success_rate !== undefined && server.success_rate < 0.9) return true;
  if (server.avg_response_time_ms !== undefined && server.avg_response_time_ms > 5000) return true;
  
  return false;
}

/**
 * Sort servers by priority (problematic servers first)
 */
export function sortServersByPriority(servers: MCPServerHealthForUI[]): MCPServerHealthForUI[] {
  return [...servers].sort((a, b) => {
    // Error status first
    if (a.status === 'error' && b.status !== 'error') return -1;
    if (b.status === 'error' && a.status !== 'error') return 1;
    
    // Then by attention needed
    const aNeeds = serverNeedsAttention(a);
    const bNeeds = serverNeedsAttention(b);
    if (aNeeds && !bNeeds) return -1;
    if (bNeeds && !aNeeds) return 1;
    
    // Then by status priority
    const statusPriority: Record<MCPServerStatus, number> = {
      error: 0,
      starting: 1,
      stopped: 2,
      running: 3,
      unknown: 4
    };
    const aPriority = statusPriority[a.status] ?? 4;
    const bPriority = statusPriority[b.status] ?? 4;
    if (aPriority !== bPriority) return aPriority - bPriority;
    
    // Finally by name
    return a.alias.localeCompare(b.alias);
  });
}

/**
 * Filter health events by type
 */
export function filterHealthEventsByType(
  events: MCPHealthEvent[],
  eventType: string
): MCPHealthEvent[] {
  return events.filter(event => event.type === eventType);
}

/**
 * Get recent health events (last N events)
 */
export function getRecentHealthEvents(
  events: MCPHealthEvent[],
  count: number = 10
): MCPHealthEvent[] {
  return events
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, count);
}

/**
 * Validate MCP tool parameters against schema
 */
export function validateToolParameters(
  parameters: Record<string, any>,
  schema: any
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  if (!schema || !schema.properties) {
    return { valid: true, errors: [] };
  }

  // Check required fields
  if (schema.required) {
    for (const field of schema.required) {
      if (!(field in parameters) || parameters[field] === undefined || parameters[field] === '') {
        errors.push(`Required field '${field}' is missing`);
      }
    }
  }

  // Basic type checking
  for (const [key, value] of Object.entries(parameters)) {
    const fieldSchema = schema.properties[key];
    if (!fieldSchema) continue;

    if (fieldSchema.type === 'number' && typeof value !== 'number') {
      if (isNaN(Number(value))) {
        errors.push(`Field '${key}' must be a number`);
      }
    }
    
    if (fieldSchema.type === 'boolean' && typeof value !== 'boolean') {
      if (value !== 'true' && value !== 'false') {
        errors.push(`Field '${key}' must be a boolean`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

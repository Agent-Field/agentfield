/**
 * Comprehensive MCP (Model Context Protocol) Module Export
 * 
 * This file provides a centralized export for all MCP-related functionality,
 * including components, hooks, utilities, and types for easy importing.
 */

// ============================================================================
// MCP Components
// ============================================================================
export {
  MCPHealthIndicator,
  MCPHealthDot,
  MCPServerCard,
  MCPServerList,
  MCPToolExplorer,
  MCPToolTester,
  MCPServerControls
} from '../components/mcp';

// ============================================================================
// MCP Custom Hooks
// ============================================================================
export {
  useMCPHealth,
  useMCPHealthSimple,
  useMCPHealthRealTime
} from '../hooks/useMCPHealth';

export {
  useMCPServers,
  useMCPServersSimple
} from '../hooks/useMCPServers';

export {
  useMCPTools,
  useMCPToolsSimple
} from '../hooks/useMCPTools';

export {
  useMCPMetrics,
  useMCPMetricsSimple,
  useMCPMetricsRealTime
} from '../hooks/useMCPMetrics';

export {
  useSSE,
  useMCPHealthSSE
} from '../hooks/useSSE';

// ============================================================================
// MCP Utilities
// ============================================================================
export {
  MCP_STATUS_COLORS,
  MCP_STATUS_ICONS,
  getMCPStatusColor,
  getMCPStatusIcon,
  calculateOverallHealth,
  aggregateMCPSummary,
  formatUptime,
  formatResponseTime,
  formatSuccessRate,
  formatErrorRate,
  formatMemoryUsage,
  formatCpuUsage,
  formatTimestamp,
  getHealthStatusText,
  getHealthStatusColor,
  formatErrorMessage,
  calculatePerformanceMetrics,
  serverNeedsAttention,
  sortServersByPriority,
  filterHealthEventsByType,
  getRecentHealthEvents,
  validateToolParameters
} from '../utils/mcpUtils';

// ============================================================================
// MCP API Services
// ============================================================================
export {
  getMCPHealth,
  getMCPHealthModeAware,
  restartMCPServer,
  stopMCPServer,
  startMCPServer,
  getMCPTools,
  getOverallMCPStatus,
  getNodeDetailsWithMCP,
  testMCPTool,
  getMCPServerMetrics,
  subscribeMCPHealthEvents,
  getMCPHealthEvents,
  bulkMCPServerAction,
  getMCPServerConfig,
  updateMCPServerConfig
} from '../services/api';

// ============================================================================
// MCP Types
// ============================================================================
export type {
  MCPServerAction,
  MCPSummaryForUI,
  MCPServerHealthForUI,
  MCPTool,
  MCPToolTestRequest,
  MCPToolTestResponse,
  MCPHealthEvent,
  MCPServerMetrics,
  MCPNodeMetrics,
  MCPErrorDetails,
  MCPError,
  AgentNodeDetailsForUI,
  MCPHealthResponse,
  MCPServerActionResponse,
  MCPToolsResponse,
  MCPOverallStatusResponse,
  MCPServerMetricsResponse,
  MCPHealthEventResponse,
  MCPHealthResponseModeAware,
  MCPHealthResponseUser,
  MCPHealthResponseDeveloper,
  AppMode
} from '../types/haxen';

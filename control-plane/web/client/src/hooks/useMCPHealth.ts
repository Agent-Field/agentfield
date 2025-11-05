import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  MCPHealthResponseModeAware,
  MCPServerHealthForUI,
  MCPSummaryForUI,
  MCPHealthEvent,
  AppMode
} from '../types/agentfield';
import { getMCPHealthModeAware } from '../services/api';
import { useMode } from '../contexts/ModeContext';
import { useMCPHealthSSE } from './useSSE';
import { aggregateMCPSummary, sortServersByPriority } from '../utils/mcpUtils';

/**
 * Configuration options for MCP health monitoring
 */
interface MCPHealthOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Whether to enable real-time updates via SSE */
  enableRealTime?: boolean;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Whether to sort servers by priority (problematic first) */
  sortByPriority?: boolean;
  /** Callback for health changes */
  onHealthChange?: (health: MCPSummaryForUI) => void;
  /** Callback for server status changes */
  onServerStatusChange?: (server: MCPServerHealthForUI) => void;
}

/**
 * MCP health state
 */
interface MCPHealthState {
  /** Current health summary */
  summary: MCPSummaryForUI | null;
  /** Individual server health data */
  servers: MCPServerHealthForUI[];
  /** Whether data is currently loading */
  loading: boolean;
  /** Last error that occurred */
  error: string | null;
  /** Timestamp of last successful fetch */
  lastFetch: Date | null;
  /** Whether data is stale (needs refresh) */
  isStale: boolean;
}

/**
 * Custom hook for managing MCP health data with real-time updates and caching
 * 
 * @param nodeId - The node ID to monitor (null to disable)
 * @param options - Configuration options
 * @returns Object containing health state and control functions
 */
export function useMCPHealth(
  nodeId: string | null,
  options: MCPHealthOptions = {}
) {
  const {
    refreshInterval = 30000, // 30 seconds default
    enableRealTime = true,
    cacheTtl = 60000, // 1 minute cache
    sortByPriority = true,
    onHealthChange,
    onServerStatusChange
  } = options;

  const { mode } = useMode();
  
  const [state, setState] = useState<MCPHealthState>({
    summary: null,
    servers: [],
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{
    data: MCPHealthResponseModeAware | null;
    timestamp: number;
    mode: AppMode;
  }>({ data: null, timestamp: 0, mode: 'user' });

  // SSE connection for real-time updates
  const {
    connected: sseConnected,
    latestEvent,
    getEventsByType
  } = useMCPHealthSSE(enableRealTime ? nodeId : null);

  /**
   * Clear refresh timeout
   */
  const clearRefreshTimeout = useCallback(() => {
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
      refreshTimeoutRef.current = null;
    }
  }, []);

  /**
   * Check if cached data is valid
   */
  const isCacheValid = useCallback(() => {
    const cache = cacheRef.current;
    if (!cache.data || cache.mode !== mode) return false;
    
    const age = Date.now() - cache.timestamp;
    return age < cacheTtl;
  }, [mode, cacheTtl]);

  /**
   * Update state from health response
   */
  const updateStateFromResponse = useCallback((response: MCPHealthResponseModeAware) => {
    if (!mountedRef.current) return;

    const servers = response.mcp_servers || [];
    const sortedServers = sortByPriority ? sortServersByPriority(servers) : servers;
    
    const newSummary = response.mcp_summary || aggregateMCPSummary(servers);

    setState(prev => {
      // Check for health changes
      if (prev.summary && onHealthChange) {
        if (prev.summary.overall_health !== newSummary.overall_health ||
            prev.summary.service_status !== newSummary.service_status) {
          onHealthChange(newSummary);
        }
      }

      // Check for server status changes
      if (onServerStatusChange && prev.servers.length > 0) {
        sortedServers.forEach(server => {
          const prevServer = prev.servers.find(s => s.alias === server.alias);
          if (prevServer && prevServer.status !== server.status) {
            onServerStatusChange(server);
          }
        });
      }

      return {
        summary: newSummary,
        servers: sortedServers,
        loading: false,
        error: null,
        lastFetch: new Date(),
        isStale: false
      };
    });

    // Update cache
    cacheRef.current = {
      data: response,
      timestamp: Date.now(),
      mode
    };
  }, [mode, sortByPriority, onHealthChange, onServerStatusChange]);

  /**
   * Fetch health data from API
   */
  const fetchHealth = useCallback(async (force: boolean = false) => {
    if (!nodeId || !mountedRef.current) return;

    // Use cache if valid and not forced
    if (!force && isCacheValid()) {
      const cachedData = cacheRef.current.data;
      if (cachedData) {
        updateStateFromResponse(cachedData);
        return;
      }
    }

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await getMCPHealthModeAware(nodeId, mode);
      updateStateFromResponse(response);
    } catch (error) {
      if (!mountedRef.current) return;
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch health data';
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
        isStale: true
      }));
    }
  }, [nodeId, mode, isCacheValid, updateStateFromResponse]);

  /**
   * Schedule next refresh
   */
  const scheduleRefresh = useCallback(() => {
    if (refreshInterval > 0) {
      clearRefreshTimeout();
      refreshTimeoutRef.current = setTimeout(() => {
        fetchHealth();
      }, refreshInterval);
    }
  }, [refreshInterval, fetchHealth, clearRefreshTimeout]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(() => {
    fetchHealth(true);
  }, [fetchHealth]);

  /**
   * Mark data as stale
   */
  const markStale = useCallback(() => {
    setState(prev => ({ ...prev, isStale: true }));
  }, []);

  /**
   * Handle SSE events
   */
  useEffect(() => {
    if (!enableRealTime || !latestEvent || !nodeId) return;

    const event = latestEvent.data as MCPHealthEvent;
    if (!event || event.node_id !== nodeId) return;

    // Update specific server status from SSE event
    if (event.type === 'server_status_change' && event.server_alias) {
      setState(prev => {
        const updatedServers = prev.servers.map(server => {
          if (server.alias === event.server_alias) {
            const updatedServer = {
              ...server,
              status: event.data.status || server.status,
              error_message: event.data.error_message || server.error_message
            } as MCPServerHealthForUI;

            // Trigger callback for status change
            if (onServerStatusChange && server.status !== updatedServer.status) {
              onServerStatusChange(updatedServer);
            }

            return updatedServer;
          }
          return server;
        });

        const newSummary = aggregateMCPSummary(updatedServers);
        
        // Trigger callback for health change
        if (onHealthChange && prev.summary) {
          if (prev.summary.overall_health !== newSummary.overall_health ||
              prev.summary.service_status !== newSummary.service_status) {
            onHealthChange(newSummary);
          }
        }

        return {
          ...prev,
          servers: sortByPriority ? sortServersByPriority(updatedServers) : updatedServers,
          summary: newSummary
        };
      });
    } else if (event.type === 'health_update') {
      // Full health update - refresh data
      markStale();
      fetchHealth();
    }
  }, [latestEvent, nodeId, enableRealTime, onServerStatusChange, onHealthChange, 
      sortByPriority, markStale, fetchHealth]);

  /**
   * Initial fetch and setup refresh cycle
   */
  useEffect(() => {
    if (nodeId) {
      fetchHealth();
      scheduleRefresh();
    } else {
      // Clear state when no nodeId
      setState({
        summary: null,
        servers: [],
        loading: false,
        error: null,
        lastFetch: null,
        isStale: false
      });
    }

    return () => {
      clearRefreshTimeout();
    };
  }, [nodeId, fetchHealth, scheduleRefresh, clearRefreshTimeout]);

  /**
   * Refresh when mode changes
   */
  useEffect(() => {
    if (nodeId) {
      fetchHealth(true);
    }
  }, [mode, nodeId, fetchHealth]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearRefreshTimeout();
    };
  }, [clearRefreshTimeout]);

  /**
   * Get servers by status
   */
  const getServersByStatus = useCallback((status: string) => {
    return state.servers.filter(server => server.status === status);
  }, [state.servers]);

  /**
   * Get servers with issues
   */
  const getProblematicServers = useCallback(() => {
    return state.servers.filter(server => 
      server.status === 'error' || 
      server.error_message ||
      (server.success_rate !== undefined && server.success_rate < 0.9)
    );
  }, [state.servers]);

  /**
   * Get health events from SSE
   */
  const getHealthEvents = useCallback((eventType?: string) => {
    if (!enableRealTime) return [];
    return eventType ? getEventsByType(eventType) : getEventsByType('health_update');
  }, [enableRealTime, getEventsByType]);

  return {
    // State
    ...state,
    
    // Real-time connection status
    realTimeConnected: sseConnected,
    
    // Control functions
    refresh,
    markStale,
    
    // Utility functions
    getServersByStatus,
    getProblematicServers,
    getHealthEvents,
    
    // Computed properties
    hasData: state.summary !== null,
    hasServers: state.servers.length > 0,
    hasIssues: state.summary?.has_issues || false,
    isHealthy: state.summary ? state.summary.overall_health >= 75 : false,
    needsAttention: getProblematicServers().length > 0
  };
}

/**
 * Simplified hook for basic health monitoring
 */
export function useMCPHealthSimple(nodeId: string | null) {
  return useMCPHealth(nodeId, {
    refreshInterval: 60000, // 1 minute
    enableRealTime: false,
    sortByPriority: false
  });
}

/**
 * Hook for real-time health monitoring with callbacks
 */
export function useMCPHealthRealTime(
  nodeId: string | null,
  onHealthChange?: (health: MCPSummaryForUI) => void,
  onServerStatusChange?: (server: MCPServerHealthForUI) => void
) {
  return useMCPHealth(nodeId, {
    refreshInterval: 15000, // 15 seconds
    enableRealTime: true,
    sortByPriority: true,
    onHealthChange,
    onServerStatusChange
  });
}
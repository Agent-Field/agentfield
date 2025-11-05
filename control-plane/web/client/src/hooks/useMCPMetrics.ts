import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  MCPServerMetrics,
  MCPNodeMetrics,
  MCPServerMetricsResponse
} from '../types/haxen';
import { getMCPServerMetrics } from '../services/api';
import { calculatePerformanceMetrics } from '../utils/mcpUtils';

/**
 * Historical metrics data point
 */
interface MetricsDataPoint {
  timestamp: Date;
  metrics: MCPServerMetrics | MCPNodeMetrics;
}

/**
 * Metrics trend data
 */
interface MetricsTrend {
  /** Current value */
  current: number;
  /** Previous value for comparison */
  previous: number;
  /** Percentage change */
  change: number;
  /** Trend direction */
  direction: 'up' | 'down' | 'stable';
}

/**
 * MCP metrics state
 */
interface MCPMetricsState {
  /** Current metrics data */
  current: MCPServerMetrics | MCPNodeMetrics | null;
  /** Historical metrics data */
  history: MetricsDataPoint[];
  /** Whether metrics are currently loading */
  loading: boolean;
  /** Last error that occurred */
  error: string | null;
  /** Timestamp of last successful fetch */
  lastFetch: Date | null;
  /** Whether data is stale */
  isStale: boolean;
}

/**
 * Options for metrics monitoring
 */
interface MCPMetricsOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Maximum number of historical data points to keep */
  maxHistorySize?: number;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Callback for metrics updates */
  onMetricsUpdate?: (metrics: MCPServerMetrics | MCPNodeMetrics) => void;
  /** Callback for performance alerts */
  onPerformanceAlert?: (alert: { type: string; message: string; severity: 'warning' | 'error' }) => void;
}

/**
 * Performance thresholds for alerts
 */
const PERFORMANCE_THRESHOLDS = {
  responseTime: {
    warning: 2000, // 2 seconds
    error: 5000    // 5 seconds
  },
  errorRate: {
    warning: 5,    // 5%
    error: 10      // 10%
  },
  successRate: {
    warning: 0.95, // 95%
    error: 0.90    // 90%
  }
};

/**
 * Custom hook for MCP performance metrics monitoring
 * 
 * @param nodeId - The node ID to monitor
 * @param serverId - Optional server ID (null for node-level metrics)
 * @param options - Configuration options
 * @returns Object containing metrics state and analysis functions
 */
export function useMCPMetrics(
  nodeId: string | null,
  serverId: string | null = null,
  options: MCPMetricsOptions = {}
) {
  const {
    refreshInterval = 30000, // 30 seconds default
    maxHistorySize = 100,
    cacheTtl = 60000, // 1 minute cache
    onMetricsUpdate,
    onPerformanceAlert
  } = options;

  const [state, setState] = useState<MCPMetricsState>({
    current: null,
    history: [],
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{
    data: MCPServerMetricsResponse | null;
    timestamp: number;
  }>({ data: null, timestamp: 0 });

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
    if (!cache.data) return false;
    
    const age = Date.now() - cache.timestamp;
    return age < cacheTtl;
  }, [cacheTtl]);

  /**
   * Check for performance issues and trigger alerts
   */
  const checkPerformanceAlerts = useCallback((metrics: MCPServerMetrics | MCPNodeMetrics) => {
    if (!onPerformanceAlert) return;

    // Check for server-level metrics
    if ('avg_response_time_ms' in metrics) {
      const serverMetrics = metrics as MCPServerMetrics;
      
      // Response time alerts
      if (serverMetrics.avg_response_time_ms >= PERFORMANCE_THRESHOLDS.responseTime.error) {
        onPerformanceAlert({
          type: 'response_time',
          message: `High response time: ${serverMetrics.avg_response_time_ms}ms`,
          severity: 'error'
        });
      } else if (serverMetrics.avg_response_time_ms >= PERFORMANCE_THRESHOLDS.responseTime.warning) {
        onPerformanceAlert({
          type: 'response_time',
          message: `Elevated response time: ${serverMetrics.avg_response_time_ms}ms`,
          severity: 'warning'
        });
      }

      // Error rate alerts
      if (serverMetrics.error_rate_percent >= PERFORMANCE_THRESHOLDS.errorRate.error) {
        onPerformanceAlert({
          type: 'error_rate',
          message: `High error rate: ${serverMetrics.error_rate_percent}%`,
          severity: 'error'
        });
      } else if (serverMetrics.error_rate_percent >= PERFORMANCE_THRESHOLDS.errorRate.warning) {
        onPerformanceAlert({
          type: 'error_rate',
          message: `Elevated error rate: ${serverMetrics.error_rate_percent}%`,
          severity: 'warning'
        });
      }

      // Success rate alerts
      const successRate = serverMetrics.total_requests > 0 
        ? serverMetrics.successful_requests / serverMetrics.total_requests 
        : 1;

      if (successRate <= PERFORMANCE_THRESHOLDS.successRate.error) {
        onPerformanceAlert({
          type: 'success_rate',
          message: `Low success rate: ${Math.round(successRate * 100)}%`,
          severity: 'error'
        });
      } else if (successRate <= PERFORMANCE_THRESHOLDS.successRate.warning) {
        onPerformanceAlert({
          type: 'success_rate',
          message: `Declining success rate: ${Math.round(successRate * 100)}%`,
          severity: 'warning'
        });
      }
    }
  }, [onPerformanceAlert]);

  /**
   * Update state from metrics response
   */
  const updateStateFromResponse = useCallback((response: MCPServerMetricsResponse) => {
    if (!mountedRef.current) return;

    const metrics = response.metrics;
    const dataPoint: MetricsDataPoint = {
      timestamp: new Date(response.timestamp),
      metrics
    };

    setState(prev => {
      // Add to history and limit size
      const newHistory = [dataPoint, ...prev.history].slice(0, maxHistorySize);

      return {
        current: metrics,
        history: newHistory,
        loading: false,
        error: null,
        lastFetch: new Date(),
        isStale: false
      };
    });

    // Update cache
    cacheRef.current = {
      data: response,
      timestamp: Date.now()
    };

    // Trigger callbacks
    onMetricsUpdate?.(metrics);
    checkPerformanceAlerts(metrics);
  }, [maxHistorySize, onMetricsUpdate, checkPerformanceAlerts]);

  /**
   * Fetch metrics data from API
   */
  const fetchMetrics = useCallback(async (force: boolean = false) => {
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
      const response = await getMCPServerMetrics(nodeId, serverId || undefined);
      updateStateFromResponse(response);
    } catch (error) {
      if (!mountedRef.current) return;
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch metrics';
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
        isStale: true
      }));
    }
  }, [nodeId, serverId, isCacheValid, updateStateFromResponse]);

  /**
   * Schedule next refresh
   */
  const scheduleRefresh = useCallback(() => {
    if (refreshInterval > 0) {
      clearRefreshTimeout();
      refreshTimeoutRef.current = setTimeout(() => {
        fetchMetrics();
      }, refreshInterval);
    }
  }, [refreshInterval, fetchMetrics, clearRefreshTimeout]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(() => {
    fetchMetrics(true);
  }, [fetchMetrics]);

  /**
   * Clear metrics history
   */
  const clearHistory = useCallback(() => {
    setState(prev => ({ ...prev, history: [] }));
  }, []);

  /**
   * Get metrics trend for a specific metric
   */
  const getMetricTrend = useCallback((metricPath: string): MetricsTrend | null => {
    if (state.history.length < 2) return null;

    const current = getNestedValue(state.current, metricPath);
    const previous = getNestedValue(state.history[1].metrics, metricPath);

    if (current === undefined || previous === undefined) return null;

    const change = previous !== 0 ? ((current - previous) / previous) * 100 : 0;
    const direction = Math.abs(change) < 1 ? 'stable' : change > 0 ? 'up' : 'down';

    return {
      current,
      previous,
      change,
      direction
    };
  }, [state.current, state.history]);

  /**
   * Get historical data for a specific metric
   */
  const getMetricHistory = useCallback((metricPath: string, limit?: number) => {
    const history = limit ? state.history.slice(0, limit) : state.history;
    
    return history
      .map(point => ({
        timestamp: point.timestamp,
        value: getNestedValue(point.metrics, metricPath)
      }))
      .filter(point => point.value !== undefined)
      .reverse(); // Chronological order
  }, [state.history]);

  /**
   * Calculate average for a metric over time
   */
  const getMetricAverage = useCallback((metricPath: string, timeWindowMs?: number) => {
    const cutoff = timeWindowMs ? new Date(Date.now() - timeWindowMs) : null;
    
    const relevantHistory = state.history.filter(point => 
      !cutoff || point.timestamp >= cutoff
    );

    if (relevantHistory.length === 0) return null;

    const values = relevantHistory
      .map(point => getNestedValue(point.metrics, metricPath))
      .filter(value => value !== undefined) as number[];

    if (values.length === 0) return null;

    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }, [state.history]);

  /**
   * Get performance summary
   */
  const getPerformanceSummary = useCallback(() => {
    if (!state.current) return null;

    if ('avg_response_time_ms' in state.current) {
      const serverMetrics = state.current as MCPServerMetrics;
      return calculatePerformanceMetrics(serverMetrics);
    }

    // Node-level metrics
    const nodeMetrics = state.current as MCPNodeMetrics;
    const serverSummaries: ReturnType<typeof calculatePerformanceMetrics>[] =
      nodeMetrics.servers.map((server) => calculatePerformanceMetrics(server));
    
    return {
      totalServers: nodeMetrics.total_servers,
      activeServers: nodeMetrics.active_servers,
      overallHealth: nodeMetrics.overall_health_score,
      avgResponseTime: serverSummaries.reduce((sum, s) => sum + s.avgResponseTime, 0) / serverSummaries.length,
      avgSuccessRate: serverSummaries.reduce((sum, s) => sum + s.successRate, 0) / serverSummaries.length,
      totalRequests: serverSummaries.reduce((sum, s) => sum + s.requestsPerMinute, 0)
    };
  }, [state.current]);

  // Initial fetch and setup refresh cycle
  useEffect(() => {
    if (nodeId) {
      fetchMetrics();
      scheduleRefresh();
    } else {
      // Clear state when no nodeId
      setState({
        current: null,
        history: [],
        loading: false,
        error: null,
        lastFetch: null,
        isStale: false
      });
    }

    return () => {
      clearRefreshTimeout();
    };
  }, [nodeId, serverId, fetchMetrics, scheduleRefresh, clearRefreshTimeout]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearRefreshTimeout();
    };
  }, [clearRefreshTimeout]);

  return {
    // State
    ...state,
    
    // Control functions
    refresh,
    clearHistory,
    
    // Analysis functions
    getMetricTrend,
    getMetricHistory,
    getMetricAverage,
    getPerformanceSummary,
    
    // Computed properties
    hasData: state.current !== null,
    hasHistory: state.history.length > 0,
    historySize: state.history.length,
    isServerMetrics: state.current ? 'alias' in state.current : false,
    isNodeMetrics: state.current ? 'servers' in state.current : false
  };
}

/**
 * Helper function to get nested object values by path
 */
function getNestedValue(obj: any, path: string): any {
  return path.split('.').reduce((current, key) => current?.[key], obj);
}

/**
 * Simplified hook for basic metrics monitoring
 */
export function useMCPMetricsSimple(nodeId: string | null, serverId?: string | null) {
  return useMCPMetrics(nodeId, serverId, {
    refreshInterval: 60000, // 1 minute
    maxHistorySize: 20
  });
}

/**
 * Hook for real-time metrics monitoring with alerts
 */
export function useMCPMetricsRealTime(
  nodeId: string | null,
  serverId: string | null = null,
  onPerformanceAlert?: (alert: { type: string; message: string; severity: 'warning' | 'error' }) => void
) {
  return useMCPMetrics(nodeId, serverId, {
    refreshInterval: 15000, // 15 seconds
    maxHistorySize: 50,
    onPerformanceAlert
  });
}

import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardSummary, DashboardError, DashboardState } from '../types/dashboard';
import { getDashboardSummary, getDashboardSummaryWithRetry } from '../services/dashboardService';

/**
 * Options for dashboard data monitoring
 */
interface DashboardOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Callback for data updates */
  onDataUpdate?: (data: DashboardSummary) => void;
  /** Callback for errors */
  onError?: (error: DashboardError) => void;
  /** Enable automatic retry on errors */
  enableRetry?: boolean;
  /** Maximum number of retries */
  maxRetries?: number;
}

/**
 * Custom hook for dashboard data fetching and management
 *
 * @param options - Configuration options
 * @returns Object containing dashboard state and control functions
 */
export function useDashboard(options: DashboardOptions = {}) {
  const {
    refreshInterval = 30000, // 30 seconds default as specified
    cacheTtl = 60000, // 1 minute cache
    onDataUpdate,
    onError,
    enableRetry = true,
    maxRetries = 3
  } = options;

  const [state, setState] = useState<DashboardState>({
    data: null,
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{
    data: DashboardSummary | null;
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
   * Update state from dashboard data
   */
  const updateStateFromData = useCallback((data: DashboardSummary) => {
    if (!mountedRef.current) return;

    setState(prev => ({
      ...prev,
      data,
      loading: false,
      error: null,
      lastFetch: new Date(),
      isStale: false
    }));

    // Update cache
    cacheRef.current = {
      data,
      timestamp: Date.now()
    };

    // Trigger callback
    onDataUpdate?.(data);
  }, [onDataUpdate]);

  /**
   * Handle errors
   */
  const handleError = useCallback((error: Error) => {
    if (!mountedRef.current) return;

    const dashboardError: DashboardError = {
      message: error.message,
      code: error.name,
      details: error
    };

    setState(prev => ({
      ...prev,
      loading: false,
      error: dashboardError,
      isStale: true
    }));

    // Trigger callback
    onError?.(dashboardError);
  }, [onError]);

  /**
   * Fetch dashboard data from API
   */
  const fetchDashboard = useCallback(async (force: boolean = false) => {
    if (!mountedRef.current) return;

    // Use cache if valid and not forced
    if (!force && isCacheValid()) {
      const cachedData = cacheRef.current.data;
      if (cachedData) {
        updateStateFromData(cachedData);
        return;
      }
    }

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = enableRetry
        ? await getDashboardSummaryWithRetry(maxRetries)
        : await getDashboardSummary();

      updateStateFromData(data);
    } catch (error) {
      handleError(error as Error);
    }
  }, [isCacheValid, updateStateFromData, handleError, enableRetry, maxRetries]);

  /**
   * Schedule next refresh
   */
  const scheduleRefresh = useCallback(() => {
    if (refreshInterval > 0) {
      clearRefreshTimeout();
      refreshTimeoutRef.current = setTimeout(() => {
        fetchDashboard();
      }, refreshInterval);
    }
  }, [refreshInterval, fetchDashboard, clearRefreshTimeout]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(() => {
    fetchDashboard(true);
  }, [fetchDashboard]);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null, isStale: false }));
  }, []);

  /**
   * Reset all state
   */
  const reset = useCallback(() => {
    clearRefreshTimeout();
    setState({
      data: null,
      loading: false,
      error: null,
      lastFetch: null,
      isStale: false
    });
    cacheRef.current = { data: null, timestamp: 0 };
  }, [clearRefreshTimeout]);

  // Initial fetch and setup refresh cycle
  useEffect(() => {
    fetchDashboard();
    scheduleRefresh();

    return () => {
      clearRefreshTimeout();
    };
  }, [fetchDashboard, scheduleRefresh, clearRefreshTimeout]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearRefreshTimeout();
    };
  }, [clearRefreshTimeout]);

  // Schedule next refresh after successful fetch
  useEffect(() => {
    if (state.data && !state.error) {
      scheduleRefresh();
    }
  }, [state.data, state.error, scheduleRefresh]);

  return {
    // State
    ...state,

    // Control functions
    refresh,
    clearError,
    reset,

    // Computed properties
    hasData: state.data !== null,
    hasError: state.error !== null,
    isRefreshing: state.loading && state.data !== null,
    isEmpty: !state.loading && !state.data && !state.error,

    // Cache info
    isCached: isCacheValid(),
    cacheAge: state.lastFetch ? Date.now() - state.lastFetch.getTime() : null
  };
}

/**
 * Simplified hook for basic dashboard monitoring
 */
export function useDashboardSimple() {
  return useDashboard({
    refreshInterval: 30000, // 30 seconds as specified
    enableRetry: true,
    maxRetries: 2
  });
}

/**
 * Hook for dashboard with custom refresh interval
 */
export function useDashboardWithInterval(intervalMs: number) {
  return useDashboard({
    refreshInterval: intervalMs,
    enableRetry: true,
    maxRetries: 3
  });
}

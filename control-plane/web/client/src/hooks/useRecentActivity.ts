import { useState, useEffect, useCallback, useRef } from 'react';
import type { RecentActivityResponse, RecentActivityError, RecentActivityState } from '../types/recentActivity';
import { getRecentActivity, getRecentActivityWithRetry } from '../services/recentActivityService';

/**
 * Options for recent activity monitoring
 */
interface RecentActivityOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Callback for data updates */
  onDataUpdate?: (data: RecentActivityResponse) => void;
  /** Callback for errors */
  onError?: (error: RecentActivityError) => void;
  /** Enable automatic retry on errors */
  enableRetry?: boolean;
  /** Maximum number of retries */
  maxRetries?: number;
}

/**
 * Custom hook for recent activity data fetching and management
 *
 * @param options - Configuration options
 * @returns Object containing recent activity state and control functions
 */
export function useRecentActivity(options: RecentActivityOptions = {}) {
  const {
    refreshInterval = 10000, // 10 seconds as specified
    cacheTtl = 30000, // 30 seconds cache
    onDataUpdate,
    onError,
    enableRetry = true,
    maxRetries = 2
  } = options;

  const [state, setState] = useState<RecentActivityState>({
    data: null,
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{
    data: RecentActivityResponse | null;
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
   * Update state from recent activity data
   */
  const updateStateFromData = useCallback((data: RecentActivityResponse) => {
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

    const recentActivityError: RecentActivityError = {
      message: error.message,
      code: error.name,
      details: error
    };

    setState(prev => ({
      ...prev,
      loading: false,
      error: recentActivityError,
      isStale: true
    }));

    // Trigger callback
    onError?.(recentActivityError);
  }, [onError]);

  /**
   * Fetch recent activity data from API
   */
  const fetchRecentActivity = useCallback(async (force: boolean = false) => {
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
        ? await getRecentActivityWithRetry(maxRetries)
        : await getRecentActivity();

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
        fetchRecentActivity();
      }, refreshInterval);
    }
  }, [refreshInterval, fetchRecentActivity, clearRefreshTimeout]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(() => {
    fetchRecentActivity(true);
  }, [fetchRecentActivity]);

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
    fetchRecentActivity();
    scheduleRefresh();

    return () => {
      clearRefreshTimeout();
    };
  }, [fetchRecentActivity, scheduleRefresh, clearRefreshTimeout]);

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
    cacheAge: state.lastFetch ? Date.now() - state.lastFetch.getTime() : null,

    // Data access helpers
    executions: state.data?.executions || [],
    executionCount: state.data?.executions.length || 0
  };
}

/**
 * Simplified hook for basic recent activity monitoring
 */
export function useRecentActivitySimple() {
  return useRecentActivity({
    refreshInterval: 10000, // 10 seconds as specified
    enableRetry: true,
    maxRetries: 2
  });
}

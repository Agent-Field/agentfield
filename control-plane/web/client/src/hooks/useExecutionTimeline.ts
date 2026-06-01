import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type {
  ExecutionTimelineResponse,
  ExecutionTimelineError,
  ExecutionTimelineState,
  ExecutionTimelineOptions,
  ExecutionTimelineHookReturn
} from '../types/executionTimeline';
import {
  getExecutionTimeline,
  getExecutionTimelineWithRetry
} from '../services/executionTimelineService';

/**
 * Custom hook for execution timeline data fetching and management
 *
 * @param options - Configuration options
 * @returns Object containing execution timeline state and control functions
 */
export function useExecutionTimeline(options: ExecutionTimelineOptions = {}): ExecutionTimelineHookReturn {
  const {
    refreshInterval = 300000, // 5 minutes (300 seconds) as specified
    cacheTtl = 300000, // 5 minutes cache to match backend cache
    onDataUpdate,
    onError,
    enableRetry = true,
    maxRetries = 2
  } = options;

  const [state, setState] = useState<ExecutionTimelineState>({
    data: null,
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{
    data: ExecutionTimelineResponse | null;
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
   * Update state from timeline data
   */
  const updateStateFromData = useCallback((data: ExecutionTimelineResponse) => {
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

    const timelineError: ExecutionTimelineError = {
      message: error.message,
      code: error.name,
      details: error
    };

    setState(prev => ({
      ...prev,
      loading: false,
      error: timelineError,
      isStale: true
    }));

    // Trigger callback
    onError?.(timelineError);
  }, [onError]);

  /**
   * Fetch execution timeline data from API with enhanced caching
   */
  const fetchExecutionTimeline = useCallback(async (force: boolean = false) => {
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
        ? await getExecutionTimelineWithRetry(maxRetries, 1000, force)
        : await getExecutionTimeline(force);

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
        fetchExecutionTimeline();
      }, refreshInterval);
    }
  }, [refreshInterval, fetchExecutionTimeline, clearRefreshTimeout]);

  /**
   * Manual refresh function with debouncing to prevent rapid calls
   */
  const refresh = useCallback(() => {
    // Debounce rapid refresh calls
    if (refreshTimeoutRef.current) {
      return; // Already refreshing, ignore additional calls
    }
    fetchExecutionTimeline(true);
  }, [fetchExecutionTimeline]);

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
    fetchExecutionTimeline();
    scheduleRefresh();

    return () => {
      clearRefreshTimeout();
    };
  }, [fetchExecutionTimeline, scheduleRefresh, clearRefreshTimeout]);

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

  // Memoize computed properties to prevent unnecessary recalculations
  const computedProperties = useMemo(() => ({
    hasData: state.data !== null,
    hasError: state.error !== null,
    isRefreshing: state.loading && state.data !== null,
    isEmpty: !state.loading && !state.data && !state.error,
    isCached: isCacheValid(),
    cacheAge: state.lastFetch ? Date.now() - state.lastFetch.getTime() : null,
    timelineData: state.data?.timeline_data || [],
    summary: state.data?.summary || null,
    dataPointCount: state.data?.timeline_data.length || 0
  }), [state, isCacheValid]);

  // Memoize the return object to prevent unnecessary re-renders
  return useMemo(() => ({
    // State
    ...state,

    // Control functions
    refresh,
    clearError,
    reset,

    // Computed properties
    ...computedProperties
  }), [state, refresh, clearError, reset, computedProperties]);
}

/**
 * Simplified hook for basic execution timeline monitoring with 5-minute polling
 */
export function useExecutionTimelineSimple() {
  return useExecutionTimeline({
    refreshInterval: 300000, // 5 minutes as specified
    enableRetry: true,
    maxRetries: 2
  });
}

/**
 * Hook for execution timeline with custom polling interval
 */
export function useExecutionTimelineWithInterval(intervalMs: number) {
  return useExecutionTimeline({
    refreshInterval: intervalMs,
    enableRetry: true,
    maxRetries: 2
  });
}

/**
 * Hook for execution timeline with enhanced error handling
 */
export function useExecutionTimelineRobust(
  onError?: (error: ExecutionTimelineError) => void,
  onDataUpdate?: (data: ExecutionTimelineResponse) => void
) {
  return useExecutionTimeline({
    refreshInterval: 300000, // 5 minutes as specified
    enableRetry: true,
    maxRetries: 3, // More retries for robust version
    onError,
    onDataUpdate
  });
}

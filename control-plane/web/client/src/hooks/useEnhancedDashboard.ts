import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type {
  EnhancedDashboardResponse,
  DashboardError,
  TimeRangePreset
} from '../types/dashboard';
import { getEnhancedDashboardSummary, type EnhancedDashboardParams } from '../services/dashboardService';

interface EnhancedDashboardOptions {
  refreshInterval?: number;
  cacheTtl?: number;
  /** Time range preset - affects cache TTL */
  preset?: TimeRangePreset;
  /** Custom start time (ISO string) - only used when preset is 'custom' */
  startTime?: string;
  /** Custom end time (ISO string) - only used when preset is 'custom' */
  endTime?: string;
  /** Whether to include comparison data */
  compare?: boolean;
  onDataUpdate?: (data: EnhancedDashboardResponse) => void;
  onError?: (error: DashboardError) => void;
}

/** Get appropriate cache TTL based on time range preset */
function getCacheTtlForPreset(preset?: TimeRangePreset): number {
  switch (preset) {
    case '1h': return 30000;   // 30 seconds
    case '24h': return 60000;  // 1 minute
    case '7d': return 120000;  // 2 minutes
    case '30d': return 300000; // 5 minutes
    default: return 60000;     // 1 minute default
  }
}

interface EnhancedDashboardState {
  data: EnhancedDashboardResponse | null;
  loading: boolean;
  error: DashboardError | null;
  lastFetch: Date | null;
  isStale: boolean;
}

export function useEnhancedDashboard(options: EnhancedDashboardOptions = {}) {
  const {
    refreshInterval = 45000,
    preset,
    startTime,
    endTime,
    compare = false,
    onDataUpdate,
    onError
  } = options;

  // Dynamic cache TTL based on preset
  const cacheTtl = options.cacheTtl ?? getCacheTtlForPreset(preset);

  // Build API params - memoized to use as dependency
  const apiParams = useMemo((): EnhancedDashboardParams => ({
    preset,
    startTime,
    endTime,
    compare,
  }), [preset, startTime, endTime, compare]);

  // Generate cache key based on params
  const cacheKey = useMemo(() => {
    return `${preset || '24h'}-${startTime || ''}-${endTime || ''}-${compare}`;
  }, [preset, startTime, endTime, compare]);

  const [state, setState] = useState<EnhancedDashboardState>({
    data: null,
    loading: false,
    error: null,
    lastFetch: null,
    isStale: false
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const cacheRef = useRef<{ data: EnhancedDashboardResponse | null; timestamp: number }>({
    data: null,
    timestamp: 0
  });

  const clearRefreshTimeout = useCallback(() => {
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
      refreshTimeoutRef.current = null;
    }
  }, []);

  const isCacheValid = useCallback(() => {
    if (!cacheRef.current.data) {
      return false;
    }
    return Date.now() - cacheRef.current.timestamp < cacheTtl;
  }, [cacheTtl]);

  const handleData = useCallback((data: EnhancedDashboardResponse) => {
    if (!mountedRef.current) {
      return;
    }

    cacheRef.current = {
      data,
      timestamp: Date.now()
    };

    setState(prev => ({
      ...prev,
      data,
      loading: false,
      error: null,
      lastFetch: new Date(),
      isStale: false
    }));

    onDataUpdate?.(data);
  }, [onDataUpdate]);

  const handleError = useCallback((error: Error) => {
    if (!mountedRef.current) {
      return;
    }

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

    onError?.(dashboardError);
  }, [onError]);

  const fetchDashboard = useCallback(async (force = false) => {
    if (!mountedRef.current) {
      return;
    }

    if (!force && isCacheValid()) {
      const cached = cacheRef.current.data;
      if (cached) {
        handleData(cached);
        return;
      }
    }

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await getEnhancedDashboardSummary(apiParams);
      handleData(data);
    } catch (error) {
      handleError(error as Error);
    }
  }, [handleData, handleError, isCacheValid, apiParams]);

  const scheduleRefresh = useCallback(() => {
    if (refreshInterval > 0) {
      clearRefreshTimeout();
      refreshTimeoutRef.current = setTimeout(() => {
        fetchDashboard();
      }, refreshInterval);
    }
  }, [clearRefreshTimeout, fetchDashboard, refreshInterval]);

  const refresh = useCallback(() => {
    fetchDashboard(true);
  }, [fetchDashboard]);

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null, isStale: false }));
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    mountedRef.current = true;
    fetchDashboard();

    return () => {
      mountedRef.current = false;
      clearRefreshTimeout();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refetch when time range changes
  useEffect(() => {
    // Clear cache when params change
    cacheRef.current = { data: null, timestamp: 0 };
    fetchDashboard(true);
  }, [cacheKey, fetchDashboard]);

  useEffect(() => {
    if (state.data && !state.loading) {
      scheduleRefresh();
    }
  }, [scheduleRefresh, state.data, state.loading]);

  return {
    ...state,
    refresh,
    clearError,
    hasData: state.data !== null,
    hasError: state.error !== null,
    isRefreshing: state.loading && state.data !== null
  };
}

export function useEnhancedDashboardSimple() {
  return useEnhancedDashboard({
    refreshInterval: 45000
  });
}

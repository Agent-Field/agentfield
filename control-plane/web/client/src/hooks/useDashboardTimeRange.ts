import { useState, useCallback, useEffect, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { TimeRangePreset } from '../types/dashboard';

export interface TimeRangeState {
  preset: TimeRangePreset;
  startTime: Date | null;
  endTime: Date | null;
  compare: boolean;
}

export interface UseDashboardTimeRangeReturn {
  timeRange: TimeRangeState;
  setPreset: (preset: TimeRangePreset) => void;
  setCustomRange: (startTime: Date, endTime: Date) => void;
  toggleCompare: () => void;
  setCompare: (compare: boolean) => void;
  /** Get params for API call */
  getApiParams: () => {
    preset: TimeRangePreset;
    startTime?: string;
    endTime?: string;
    compare: boolean;
  };
  /** Human-readable label for the current time range */
  label: string;
}

const PRESET_LABELS: Record<TimeRangePreset, string> = {
  '1h': 'Last hour',
  '24h': 'Last 24 hours',
  '7d': 'Last 7 days',
  '30d': 'Last 30 days',
  'custom': 'Custom range',
};

/**
 * Hook for managing dashboard time range state with URL persistence.
 * Syncs preset, custom dates, and comparison toggle to URL search params.
 */
export function useDashboardTimeRange(defaultPreset: TimeRangePreset = '24h'): UseDashboardTimeRangeReturn {
  const [searchParams, setSearchParams] = useSearchParams();

  // Parse initial state from URL
  const initialState = useMemo((): TimeRangeState => {
    const presetParam = searchParams.get('range') as TimeRangePreset | null;
    const startParam = searchParams.get('start');
    const endParam = searchParams.get('end');
    const compareParam = searchParams.get('compare') === 'true';

    // Validate preset
    const validPresets: TimeRangePreset[] = ['1h', '24h', '7d', '30d', 'custom'];
    const preset = presetParam && validPresets.includes(presetParam) ? presetParam : defaultPreset;

    // Parse custom dates if preset is custom
    let startTime: Date | null = null;
    let endTime: Date | null = null;

    if (preset === 'custom' && startParam && endParam) {
      try {
        startTime = new Date(startParam);
        endTime = new Date(endParam);
        // Validate dates
        if (isNaN(startTime.getTime()) || isNaN(endTime.getTime())) {
          startTime = null;
          endTime = null;
        }
      } catch {
        // Invalid dates, fall back to defaults
      }
    }

    return {
      preset,
      startTime,
      endTime,
      compare: compareParam,
    };
  }, []); // Only compute once on mount

  const [timeRange, setTimeRange] = useState<TimeRangeState>(initialState);

  // Sync state to URL
  const syncToUrl = useCallback((state: TimeRangeState) => {
    const newParams = new URLSearchParams(searchParams);

    // Always set preset if not default
    if (state.preset !== defaultPreset) {
      newParams.set('range', state.preset);
    } else {
      newParams.delete('range');
    }

    // Set custom dates
    if (state.preset === 'custom' && state.startTime && state.endTime) {
      newParams.set('start', state.startTime.toISOString());
      newParams.set('end', state.endTime.toISOString());
    } else {
      newParams.delete('start');
      newParams.delete('end');
    }

    // Set compare flag
    if (state.compare) {
      newParams.set('compare', 'true');
    } else {
      newParams.delete('compare');
    }

    setSearchParams(newParams, { replace: true });
  }, [searchParams, setSearchParams, defaultPreset]);

  // Update URL when state changes
  useEffect(() => {
    syncToUrl(timeRange);
  }, [timeRange, syncToUrl]);

  const setPreset = useCallback((preset: TimeRangePreset) => {
    setTimeRange(prev => ({
      ...prev,
      preset,
      // Clear custom dates when switching away from custom
      startTime: preset === 'custom' ? prev.startTime : null,
      endTime: preset === 'custom' ? prev.endTime : null,
    }));
  }, []);

  const setCustomRange = useCallback((startTime: Date, endTime: Date) => {
    setTimeRange(prev => ({
      ...prev,
      preset: 'custom',
      startTime,
      endTime,
    }));
  }, []);

  const toggleCompare = useCallback(() => {
    setTimeRange(prev => ({
      ...prev,
      compare: !prev.compare,
    }));
  }, []);

  const setCompare = useCallback((compare: boolean) => {
    setTimeRange(prev => ({
      ...prev,
      compare,
    }));
  }, []);

  const getApiParams = useCallback(() => {
    const params: {
      preset: TimeRangePreset;
      startTime?: string;
      endTime?: string;
      compare: boolean;
    } = {
      preset: timeRange.preset,
      compare: timeRange.compare,
    };

    if (timeRange.preset === 'custom' && timeRange.startTime && timeRange.endTime) {
      params.startTime = timeRange.startTime.toISOString();
      params.endTime = timeRange.endTime.toISOString();
    }

    return params;
  }, [timeRange]);

  const label = useMemo(() => {
    if (timeRange.preset === 'custom' && timeRange.startTime && timeRange.endTime) {
      const options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric' };
      const start = timeRange.startTime.toLocaleDateString(undefined, options);
      const end = timeRange.endTime.toLocaleDateString(undefined, options);
      return `${start} - ${end}`;
    }
    return PRESET_LABELS[timeRange.preset];
  }, [timeRange]);

  return {
    timeRange,
    setPreset,
    setCustomRange,
    toggleCompare,
    setCompare,
    getApiParams,
    label,
  };
}

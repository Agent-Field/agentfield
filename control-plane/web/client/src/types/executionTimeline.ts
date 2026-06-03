/**
 * TypeScript interfaces for execution timeline API responses
 */

export interface TimelineDataPoint {
  timestamp: string;           // ISO timestamp for the hour
  hour: string;               // "14:00" format for display
  executions: number;         // Total executions in this hour
  successful: number;         // Successful executions
  failed: number;            // Failed executions
  running: number;           // Currently running executions
  success_rate: number;      // Percentage (0-100)
  avg_duration_ms: number;   // Average execution duration
  total_duration_ms: number; // Total duration for all executions
}

export interface TimelineSummary {
  total_executions: number;
  avg_success_rate: number;
  total_errors: number;
  peak_hour: string;
  peak_executions: number;
}

export interface ExecutionTimelineResponse {
  timeline_data: TimelineDataPoint[];
  cache_timestamp: string;
  summary: TimelineSummary;
}

export interface ExecutionTimelineError {
  message: string;
  code?: string;
  details?: any;
}

export interface ExecutionTimelineState {
  data: ExecutionTimelineResponse | null;
  loading: boolean;
  error: ExecutionTimelineError | null;
  lastFetch: Date | null;
  isStale: boolean;
}

/**
 * Options for execution timeline monitoring
 */
export interface ExecutionTimelineOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Callback for data updates */
  onDataUpdate?: (data: ExecutionTimelineResponse) => void;
  /** Callback for errors */
  onError?: (error: ExecutionTimelineError) => void;
  /** Enable automatic retry on errors */
  enableRetry?: boolean;
  /** Maximum number of retries */
  maxRetries?: number;
}

/**
 * Return type for useExecutionTimeline hook
 */
export interface ExecutionTimelineHookReturn extends ExecutionTimelineState {
  // Control functions
  refresh: () => void;
  clearError: () => void;
  reset: () => void;

  // Computed properties
  hasData: boolean;
  hasError: boolean;
  isRefreshing: boolean;
  isEmpty: boolean;

  // Cache info
  isCached: boolean;
  cacheAge: number | null;

  // Data access helpers
  timelineData: TimelineDataPoint[];
  summary: TimelineSummary | null;
  dataPointCount: number;
}

/**
 * TypeScript interfaces for recent activity API responses
 */

export interface ActivityExecution {
  execution_id: string;
  agent_name: string;
  reasoner_name: string;
  status: 'running' | 'completed' | 'success' | 'failed' | 'pending';
  started_at: string;
  duration_ms?: number;
  relative_time: string;
}

export interface RecentActivityResponse {
  executions: ActivityExecution[];
  cache_timestamp: string;
}

export interface RecentActivityError {
  message: string;
  code?: string;
  details?: any;
}

export interface RecentActivityState {
  data: RecentActivityResponse | null;
  loading: boolean;
  error: RecentActivityError | null;
  lastFetch: Date | null;
  isStale: boolean;
}

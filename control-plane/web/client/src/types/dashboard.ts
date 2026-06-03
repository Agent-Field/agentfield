/**
 * TypeScript interfaces for dashboard API responses
 */

export interface DashboardSummary {
  agents: {
    running: number;
    total: number;
  };
  executions: {
    today: number;
    yesterday: number;
  };
  success_rate: number;
  packages: {
    available: number;
    installed: number;
  };
}

export interface DashboardError {
  message: string;
  code?: string;
  details?: any;
}

export interface DashboardState {
  data: DashboardSummary | null;
  loading: boolean;
  error: DashboardError | null;
  lastFetch: Date | null;
  isStale: boolean;
}

// Time range types
export type TimeRangePreset = '1h' | '24h' | '7d' | '30d' | 'custom';

export interface TimeRangeInfo {
  start_time: string;
  end_time: string;
  preset?: TimeRangePreset;
}

// Comparison data types
export interface ComparisonData {
  previous_period: TimeRangeInfo;
  overview_delta: EnhancedOverviewDelta;
}

export interface EnhancedOverviewDelta {
  executions_delta: number;
  executions_delta_pct: number;
  success_rate_delta: number;
  avg_duration_delta_ms: number;
  avg_duration_delta_pct: number;
}

// Hotspot types
export interface HotspotSummary {
  top_failing_reasoners: HotspotItem[];
}

export interface HotspotItem {
  reasoner_id: string;
  total_executions: number;
  failed_executions: number;
  error_rate: number;
  contribution_pct: number;
  top_errors: ErrorCount[];
}

export interface ErrorCount {
  message: string;
  count: number;
}

// Activity heatmap types
export interface ActivityPatterns {
  hourly_heatmap: HeatmapCell[][];
}

export interface HeatmapCell {
  total: number;
  failed: number;
  error_rate: number;
}

export interface EnhancedDashboardResponse {
  generated_at: string;
  time_range: TimeRangeInfo;
  overview: EnhancedDashboardOverview;
  execution_trends: ExecutionTrendSummary;
  agent_health: AgentHealthSummary;
  workflows: EnhancedWorkflowInsights;
  incidents: IncidentItem[];
  comparison?: ComparisonData;
  hotspots: HotspotSummary;
  activity_patterns: ActivityPatterns;
}

export interface EnhancedDashboardOverview {
  total_agents: number;
  active_agents: number;
  degraded_agents: number;
  offline_agents: number;
  total_reasoners: number;
  total_skills: number;
  executions_last_24h: number;
  executions_last_7d: number;
  success_rate_24h: number;
  average_duration_ms_24h: number;
  median_duration_ms_24h: number;
}

export interface ExecutionTrendSummary {
  last_24h: ExecutionWindowMetrics;
  last_7_days: ExecutionTrendPoint[];
}

export interface ExecutionWindowMetrics {
  total: number;
  succeeded: number;
  failed: number;
  success_rate: number;
  average_duration_ms: number;
  throughput_per_hour: number;
}

export interface ExecutionTrendPoint {
  date: string;
  total: number;
  succeeded: number;
  failed: number;
}

export interface AgentHealthSummary {
  total: number;
  active: number;
  degraded: number;
  offline: number;
  agents: AgentHealthItem[];
}

export interface AgentHealthItem {
  id: string;
  team_id: string;
  version: string;
  status: string;
  health: string;
  lifecycle: string;
  last_heartbeat: string;
  reasoners: number;
  skills: number;
  uptime?: string;
}

export interface EnhancedWorkflowInsights {
  top_workflows: WorkflowStat[];
  active_runs: ActiveWorkflowRun[];
  longest_executions: CompletedExecutionStat[];
}

export interface WorkflowStat {
  workflow_id: string;
  name?: string;
  total_executions: number;
  success_rate: number;
  failed_executions: number;
  average_duration_ms: number;
  last_activity: string;
}

export interface ActiveWorkflowRun {
  execution_id: string;
  workflow_id: string;
  name?: string;
  started_at: string;
  elapsed_ms: number;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
}

export interface CompletedExecutionStat {
  execution_id: string;
  workflow_id: string;
  name?: string;
  duration_ms: number;
  completed_at?: string;
  status: string;
}

export interface IncidentItem {
  execution_id: string;
  workflow_id: string;
  name?: string;
  status: string;
  started_at: string;
  completed_at?: string;
  agent_node_id: string;
  reasoner_id: string;
  error?: string;
}

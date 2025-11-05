import type { CanonicalStatus } from "../utils/status";

export type { CanonicalStatus };

export interface ExecutionSummary {
  id: number;
  execution_id: string;
  workflow_id: string;
  session_id?: string;
  agent_node_id: string;
  reasoner_id: string;
  status: CanonicalStatus;
  duration_ms: number;
  input_size: number;
  output_size: number;
  error_message?: string;
  created_at: string;
  // Computed fields for frontend compatibility
  started_at?: string;
  completed_at?: string;
  workflow_name?: string;
  workflow_tags?: string[];
}

export interface GroupedExecutionSummary {
  group_key: string;
  group_label: string;
  count: number;
  total_duration_ms: number;
  avg_duration_ms: number;
  status_summary: Record<string, number>;
  latest_execution: string;
  executions: ExecutionSummary[];
}

export interface ExecutionFilters {
  agent_node_id?: string;
  workflow_id?: string;
  session_id?: string;
  actor_id?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
  search?: string;
  page: number;
  page_size: number;
}

export interface ExecutionGrouping {
  group_by: "none" | "workflow" | "session" | "actor" | "agent" | "status";
  sort_by: "time" | "duration" | "status";
  sort_order: "asc" | "desc";
}

export interface PaginatedExecutions {
  executions: ExecutionSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  // Computed fields for frontend compatibility
  total_count?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

export interface GroupedExecutions {
  groups: GroupedExecutionSummary[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ExecutionStats {
  total_executions: number;
  successful_count: number;
  failed_count: number;
  running_count: number;
  average_duration_ms: number;
  executions_by_status: Record<string, number>;
  executions_by_agent: Record<string, number>;
  // Computed fields for frontend compatibility
  successful_executions?: number;
  failed_executions?: number;
  running_executions?: number;
}

export interface ExecutionEvent {
  type: "execution_started" | "execution_completed" | "execution_failed";
  execution: ExecutionSummary;
  timestamp: string;
}

export interface WorkflowExecution {
  id: number;
  workflow_id: string;
  execution_id: string;
  agentfield_request_id: string;
  session_id?: string;
  actor_id?: string;
  agent_node_id: string;
  parent_workflow_id?: string;
  root_workflow_id?: string;
  workflow_depth: number;
  reasoner_id: string;
  input_data: any;
  output_data: any;
  input_size: number;
  output_size: number;
  input_uri?: string | null;
  result_uri?: string | null;
  workflow_name?: string;
  workflow_tags: string[];
  status: CanonicalStatus;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  error_message?: string;
  retry_count: number;
  created_at: string;
  updated_at: string;
  notes?: ExecutionNote[];
  webhook_registered?: boolean;
  webhook_events?: ExecutionWebhookEvent[];
}

// Import ExecutionNote type
export interface ExecutionNote {
  message: string;
  tags: string[];
  timestamp: string;
}

export interface ExecutionWebhookEvent {
  id: number;
  execution_id: string;
  event_type: string;
  status: string;
  http_status?: number;
  payload?: any;
  response_body?: string | null;
  error_message?: string | null;
  created_at: string;
}

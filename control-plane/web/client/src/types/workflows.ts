// Enhanced types for the new workflow-centric execution page

import type { CanonicalStatus } from '../utils/status';

export interface WorkflowSummary {
  run_id: string;
  workflow_id: string;
  root_execution_id?: string;
  /**
   * Status of the root execution row, which is the unit the user actually
   * controls via Pause/Resume/Cancel. The aggregate `status` field can
   * drift from this when in-flight children are still running after the
   * user pauses or cancels the root.
   */
  root_execution_status?: CanonicalStatus;
  status: CanonicalStatus;
  root_reasoner: string;
  current_task: string;
  total_executions: number;
  max_depth: number;
  started_at: string;
  latest_activity: string;
  completed_at?: string;
  duration_ms?: number;
  display_name: string;
  agent_id?: string;
  agent_name?: string;
  session_id?: string;
  actor_id?: string;
  status_counts: Record<string, number>;
  active_executions: number;
  terminal: boolean;
}

export interface EnhancedExecution {
  execution_id: string;
  workflow_id: string;
  status: CanonicalStatus;
  task_name: string;
  workflow_name: string;
  agent_name: string;
  relative_time: string;
  duration_display: string;
  workflow_context?: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  session_id?: string;
  actor_id?: string;
}

export interface ViewMode {
  id: 'executions' | 'workflows' | 'sessions' | 'agents';
  label: string;
  description: string;
  icon: string;
}

export interface ExecutionViewFilters {
  status?: string;
  agent?: string;
  workflow?: string;
  session?: string;
  timeRange?: string;
  search?: string;
}

export interface WorkflowsResponse {
  workflows: WorkflowSummary[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_more?: boolean;
}

export interface EnhancedExecutionsResponse {
  executions: EnhancedExecution[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_more?: boolean;
}

export interface ExecutionViewState {
  viewMode: ViewMode['id'];
  filters: ExecutionViewFilters;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  page: number;
  pageSize: number;
}

export interface WorkflowTimelineNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  parent_execution_id?: string;
  workflow_depth: number;
  agent_name?: string;
  task_name?: string;
  input_data?: Record<string, unknown> | null;
  output_data?: Record<string, unknown> | null;
  webhook_registered?: boolean;
  webhook_event_count?: number;
  webhook_success_count?: number;
  webhook_failure_count?: number;
  webhook_last_status?: string;
  webhook_last_error?: string;
  webhook_last_sent_at?: string;
  webhook_last_http_status?: number;
  notes?: {
    message: string;
    tags: string[];
    timestamp: string;
  }[];
}

export interface WorkflowDAGLightweightNode {
  execution_id: string;
  parent_execution_id?: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  workflow_depth: number;
}

/** Aggregated webhook deliveries for a run (from lightweight DAG). */
export interface WebhookRunSummary {
  steps_with_webhook: number;
  total_deliveries: number;
  failed_deliveries: number;
}

/** Latest failed webhook attempt for an execution (run strip + retry). */
export interface WebhookFailurePreview {
  execution_id: string;
  agent_node_id?: string;
  reasoner_id?: string;
  event_type?: string;
  http_status?: number | null;
  created_at?: string;
}

export interface WorkflowDAGLightweightResponse {
  root_workflow_id: string;
  workflow_status: string;
  workflow_name: string;
  session_id?: string;
  actor_id?: string;
  total_nodes: number;
  max_depth: number;
  timeline: WorkflowDAGLightweightNode[];
  mode: 'lightweight';
  unique_agent_node_ids?: string[];
  /** Issuer DID from stored execution VCs for this workflow (server-issued), when present. */
  workflow_issuer_did?: string;
  webhook_summary?: WebhookRunSummary;
  /** Executions with a failed delivery (capped); for run-level retry / focus step. */
  webhook_failures?: WebhookFailurePreview[];
}

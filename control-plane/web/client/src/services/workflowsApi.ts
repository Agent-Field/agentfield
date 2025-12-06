import type {
  WorkflowsResponse,
  EnhancedExecutionsResponse,
  ExecutionViewFilters,
  WorkflowSummary,
  WorkflowDAGLightweightResponse,
} from '../types/workflows';
import { normalizeExecutionStatus } from '../utils/status';
import { getGlobalApiKey } from './api';

const API_V1_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';
const API_V2_BASE_URL = import.meta.env.VITE_API_V2_BASE_URL || '/api/ui/v2';

async function fetchWrapper<T>(url: string, options?: RequestInit, baseUrl: string = API_V1_BASE_URL): Promise<T> {
  const headers = new Headers(options?.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }
  const response = await fetch(`${baseUrl}${url}`, { ...options, headers });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      message: 'Request failed with status ' + response.status
    }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function buildQueryString(params: Record<string, any>): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      if (Array.isArray(value)) {
        value.forEach(v => searchParams.append(key, v.toString()));
      } else {
        searchParams.append(key, value.toString());
      }
    }
  });

  return searchParams.toString();
}

function normalizeFilters(filters: ExecutionViewFilters = {}) {
  const params: Record<string, unknown> = { ...filters };

  if (params.status === 'all') {
    delete params.status;
  }

  delete params.timeRange;
  delete params.agent;

  const workflowValue = params.workflow;
  if (typeof workflowValue === 'string' && workflowValue.length > 0) {
    params.workflow_id = workflowValue;
  }
  delete params.workflow;

  const sessionValue = params.session;
  if (typeof sessionValue === 'string' && sessionValue.length > 0) {
    params.session_id = sessionValue;
  }
  delete params.session;

  return params;
}

interface WorkflowRunListResponse {
  runs: ApiWorkflowRunSummary[];
  total_count: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

interface ApiWorkflowRunSummary {
  run_id: string;
  workflow_id: string;
  root_execution_id?: string | null;
  status: string;
  display_name: string;
  current_task: string;
  root_reasoner: string;
  agent_id?: string | null;
  session_id?: string | null;
  actor_id?: string | null;
  total_executions: number;
  max_depth: number;
  active_executions: number;
  status_counts?: Record<string, number>;
  started_at: string;
  updated_at: string;
  latest_activity: string;
  completed_at?: string | null;
  duration_ms?: number | null;
  terminal: boolean;
}

interface ApiWorkflowExecution {
  execution_id: string;
  workflow_id: string;
  parent_execution_id?: string | null;
  parent_workflow_id?: string | null;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string | null;
  workflow_depth: number;
  active_children: number;
  pending_children: number;
}

export interface WorkflowRunDetailResponse {
  run: {
    run_id: string;
    root_workflow_id: string;
    root_execution_id?: string | null;
    status: string;
    total_steps: number;
    completed_steps: number;
    failed_steps: number;
    returned_steps?: number;
    status_counts?: Record<string, number>;
    created_at: string;
    updated_at: string;
    completed_at?: string | null;
  };
  executions: ApiWorkflowExecution[];
}

// Get workflows summary with human-readable data
export async function getWorkflowsSummary(
  filters: ExecutionViewFilters = {},
  page: number = 1,
  pageSize: number = 20,
  sortBy: string = 'latest_activity',
  sortOrder: 'asc' | 'desc' = 'desc',
  signal?: AbortSignal
): Promise<WorkflowsResponse> {
  const normalizedFilters = normalizeFilters(filters);

  const apiSortBy = mapWorkflowSortKeyToApi(sortBy);

  const queryParams: Record<string, unknown> = {
    page,
    page_size: pageSize,
    sort_by: apiSortBy,
    sort_order: sortOrder,
  };

  if (filters.timeRange) {
    const since = resolveSinceTimestamp(filters.timeRange);
    if (since) {
      queryParams['since'] = since;
    }
  }

  if (normalizedFilters.status && normalizedFilters.status !== 'all') {
    queryParams['status'] = normalizeExecutionStatus(normalizedFilters.status as string);
  }

  if (normalizedFilters.session_id) {
    queryParams['session_id'] = normalizedFilters.session_id;
  }

  if (normalizedFilters.actor_id) {
    queryParams['actor_id'] = normalizedFilters.actor_id;
  }

  if (normalizedFilters.workflow_id) {
    queryParams['workflow_id'] = normalizedFilters.workflow_id;
  }

  if (normalizedFilters.search) {
    queryParams['search'] = normalizedFilters.search;
  }

  const queryString = buildQueryString(queryParams);
  const url = `/workflow-runs${queryString ? `?${queryString}` : ''}`;

  const response = await fetchWrapper<WorkflowRunListResponse>(url, { signal }, API_V2_BASE_URL);

  const workflows = response.runs.map(mapApiRunToWorkflowSummary);
  const totalPages = response.page_size > 0
    ? Math.ceil(response.total_count / response.page_size)
    : 0;

  return {
    workflows,
    total_count: response.total_count,
    page: response.page,
    page_size: response.page_size,
    total_pages: totalPages,
    has_more: response.has_more,
  };
}

function mapApiRunToWorkflowSummary(run: ApiWorkflowRunSummary): WorkflowSummary {
  const normalizedStatus = normalizeExecutionStatus(run.status ?? 'unknown');
  const statusCounts = run.status_counts ?? {};
  const activeExecutions = typeof run.active_executions === 'number' ? run.active_executions : 0;

  return {
    run_id: run.run_id,
    workflow_id: run.workflow_id,
    root_execution_id: run.root_execution_id ?? undefined,
    status: normalizedStatus,
    root_reasoner: run.root_reasoner || run.display_name,
    current_task: run.current_task || run.root_reasoner || run.display_name,
    total_executions: run.total_executions,
    max_depth: run.max_depth,
    started_at: run.started_at,
    latest_activity: run.latest_activity || run.updated_at,
    completed_at: run.completed_at ?? undefined,
    duration_ms: run.duration_ms ?? undefined,
    display_name: run.display_name,
    agent_id: run.agent_id ?? undefined,
    agent_name: run.agent_id ?? undefined,
    session_id: run.session_id ?? undefined,
    actor_id: run.actor_id ?? undefined,
    status_counts: statusCounts,
    active_executions: activeExecutions,
    terminal: run.terminal,
  };
}

function resolveSinceTimestamp(timeRange?: string): string | undefined {
  if (!timeRange || timeRange === 'all') {
    return undefined;
  }

  const match = /^([0-9]+)([hd])$/i.exec(timeRange.trim());
  if (!match) {
    return undefined;
  }

  const value = Number(match[1]);
  if (!Number.isFinite(value) || value <= 0) {
    return undefined;
  }

  let milliseconds = 0;
  const unit = match[2].toLowerCase();
  if (unit === 'h') {
    milliseconds = value * 60 * 60 * 1000;
  } else if (unit === 'd') {
    milliseconds = value * 24 * 60 * 60 * 1000;
  }

  if (milliseconds <= 0) {
    return undefined;
  }

  const since = new Date(Date.now() - milliseconds);
  return since.toISOString();
}

export function mapWorkflowSortKeyToApi(sortKey: string): string {
  switch (sortKey) {
    case 'status':
      return 'status';
    case 'total_executions':
    case 'nodes':
      return 'total_steps';
    case 'failed':
    case 'issues':
      return 'failed_steps';
    case 'started':
    case 'started_at':
      return 'created_at';
    case 'latest_activity':
    case 'updated_at':
    default:
      return 'updated_at';
  }
}

// Get enhanced executions with human-readable data
export async function getEnhancedExecutions(
  filters: ExecutionViewFilters = {},
  page: number = 1,
  pageSize: number = 20,
  sortBy: string = 'started_at',
  sortOrder: 'asc' | 'desc' = 'desc',
  signal?: AbortSignal
): Promise<EnhancedExecutionsResponse> {
  const normalizedFilters = normalizeFilters(filters);

  const queryParams = {
    ...normalizedFilters,
    page,
    limit: pageSize,
    sort_by: sortBy,
    sort_order: sortOrder,
  };

  const queryString = buildQueryString(queryParams);
  const url = `/executions/enhanced${queryString ? `?${queryString}` : ''}`;

  return fetchWrapper<EnhancedExecutionsResponse>(url, { signal });
}

// Get workflow details for DAG visualization
export async function getWorkflowDetails(workflowId: string): Promise<any> {
  return fetchWrapper<any>(`/workflows/${workflowId}/details`);
}

export interface WorkflowDAGRequestOptions {
  lightweight?: boolean;
  signal?: AbortSignal;
}

// Get workflow DAG data
export async function getWorkflowDAG<T = any>(
  workflowId: string,
  options: WorkflowDAGRequestOptions = {}
): Promise<T> {
  const { lightweight = false, signal } = options;
  const query = lightweight ? '?mode=lightweight' : '';
  return fetchWrapper<T>(`/workflows/${workflowId}/dag${query}`, { signal });
}

export async function getWorkflowDAGLightweight(
  workflowId: string,
  signal?: AbortSignal
): Promise<WorkflowDAGLightweightResponse> {
  return getWorkflowDAG<WorkflowDAGLightweightResponse>(workflowId, {
    lightweight: true,
    signal,
  });
}

export async function getWorkflowRunDetail(
  runId: string,
  signal?: AbortSignal
): Promise<WorkflowRunDetailResponse> {
  return fetchWrapper<WorkflowRunDetailResponse>(
    `/workflow-runs/${runId}`,
    { signal },
    API_V2_BASE_URL
  );
}

export async function getWorkflowRunSummary(
  runId: string,
  signal?: AbortSignal
): Promise<WorkflowSummary | null> {
  const query = buildQueryString({ run_id: runId, page: 1, page_size: 1 });
  const response = await fetchWrapper<WorkflowRunListResponse>(
    `/workflow-runs?${query}`,
    { signal },
    API_V2_BASE_URL
  );

  const [run] = response.runs;
  if (!run) {
    return null;
  }

  return mapApiRunToWorkflowSummary(run);
}

// Helper functions for specific view modes
export async function getExecutionsByViewMode(
  viewMode: 'executions' | 'workflows' | 'sessions' | 'agents',
  filters: ExecutionViewFilters = {},
  page: number = 1,
  pageSize: number = 20,
  sortBy?: string,
  sortOrder: 'asc' | 'desc' = 'desc'
): Promise<WorkflowsResponse | EnhancedExecutionsResponse> {
  switch (viewMode) {
    case 'workflows':
      return getWorkflowsSummary(filters, page, pageSize, sortBy || 'latest_activity', sortOrder);
    case 'executions':
    case 'sessions':
    case 'agents':
      return getEnhancedExecutions(filters, page, pageSize, sortBy || 'when', sortOrder);
    default:
      return getEnhancedExecutions(filters, page, pageSize, sortBy || 'when', sortOrder);
  }
}

// Search across all view modes
export async function searchExecutionData(
  searchTerm: string,
  viewMode: 'executions' | 'workflows' | 'sessions' | 'agents',
  filters: ExecutionViewFilters = {},
  page: number = 1,
  pageSize: number = 20
): Promise<WorkflowsResponse | EnhancedExecutionsResponse> {
  const searchFilters = {
    ...filters,
    search: searchTerm
  };

  return getExecutionsByViewMode(viewMode, searchFilters, page, pageSize);
}

// Get available filter options
export async function getFilterOptions(): Promise<{
  agents: string[];
  workflows: string[];
  sessions: string[];
  statuses: string[];
}> {
  return fetchWrapper<any>('/executions/filter-options');
}

// Get execution statistics for the current view
export async function getExecutionViewStats(
  viewMode: 'executions' | 'workflows' | 'sessions' | 'agents',
  filters: ExecutionViewFilters = {}
): Promise<{
  total_count: number;
  status_breakdown: Record<string, number>;
  recent_activity: number;
}> {
  const queryParams = {
    ...filters,
    view_mode: viewMode
  };

  const queryString = buildQueryString(queryParams);
  const url = `/executions/view-stats${queryString ? `?${queryString}` : ''}`;

  return fetchWrapper<any>(url);
}

// Workflow cleanup API
export interface WorkflowCleanupResult {
  workflow_id: string;
  dry_run: boolean;
  deleted_records: Record<string, number>;
  freed_space_bytes: number;
  duration_ms: number;
  success: boolean;
  error_message?: string;
}

// Delete a single workflow (cleanup all related data)
export async function deleteWorkflow(
  workflowId: string,
  dryRun: boolean = false
): Promise<WorkflowCleanupResult> {
  const queryParams = dryRun ? '?dry_run=true&confirm=true' : '?confirm=true';
  return fetchWrapper<WorkflowCleanupResult>(`/workflows/${workflowId}/cleanup${queryParams}`, {
    method: 'DELETE'
  });
}

// Delete multiple workflows (batch cleanup)
export async function deleteWorkflows(
  workflowIds: string[],
  dryRun: boolean = false
): Promise<WorkflowCleanupResult[]> {
  const uniqueIds = Array.from(
    new Set(
      workflowIds
        .map((id) => id?.trim())
        .filter((id): id is string => Boolean(id && id.length))
    )
  );

  if (uniqueIds.length === 0) {
    return [];
  }

  const results: WorkflowCleanupResult[] = [];

  // Process deletions sequentially to avoid overwhelming the server
  for (const workflowId of uniqueIds) {
    try {
      const result = await deleteWorkflow(workflowId, dryRun);
      results.push(result);
    } catch (error) {
      // Add failed result to maintain order
      results.push({
        workflow_id: workflowId,
        dry_run: dryRun,
        deleted_records: {},
        freed_space_bytes: 0,
        duration_ms: 0,
        success: false,
        error_message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  return results;
}

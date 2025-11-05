import type {
  PaginatedExecutions,
  GroupedExecutions,
  ExecutionStats,
  ExecutionFilters,
  ExecutionGrouping,
  ExecutionSummary,
  WorkflowExecution,
  ExecutionWebhookEvent,
} from "../types/executions";
import type { EnhancedExecutionsResponse } from "../types/workflows";
import type {
  NotesResponse,
  AddNoteRequest,
  AddNoteResponse,
  NotesFilters,
} from "../types/notes";
import { normalizeExecutionStatus } from "../utils/status";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/ui/v1";

async function fetchWrapper<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${url}`, options);
  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({
        message: "Request failed with status " + response.status,
      }));
    throw new Error(
      errorData.message || `HTTP error! status: ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}

// Transform backend ExecutionSummary to frontend format
function transformExecutionSummary(backendExecution: any): ExecutionSummary {
  return {
    ...backendExecution,
    // Map backend fields to frontend expectations
    started_at: backendExecution.created_at,
    workflow_tags: backendExecution.workflow_tags || [],
    status: normalizeExecutionStatus(backendExecution.status),
  };
}

// Transform backend PaginatedExecutions to frontend format
function transformPaginatedExecutions(
  backendResponse: any,
): PaginatedExecutions {
  return {
    executions: (backendResponse.executions || []).map(
      transformExecutionSummary,
    ),
    total: backendResponse.total || 0,
    page: backendResponse.page || 1,
    page_size: backendResponse.page_size || 20,
    total_pages: backendResponse.total_pages || 1,
    // Computed fields for frontend compatibility
    total_count: backendResponse.total || 0,
    has_next: backendResponse.page < backendResponse.total_pages,
    has_prev: backendResponse.page > 1,
  };
}

// Transform backend ExecutionStats to frontend format
function transformExecutionStats(backendStats: any): ExecutionStats {
  const executionsByStatus: Record<string, number> = {};
  if (backendStats.executions_by_status) {
    Object.entries(backendStats.executions_by_status).forEach(
      ([status, count]) => {
        const normalized = normalizeExecutionStatus(status);
        executionsByStatus[normalized] =
          (executionsByStatus[normalized] || 0) + Number(count || 0);
      },
    );
  }

  return {
    ...backendStats,
    // Map backend fields to frontend expectations
    successful_executions: backendStats.successful_count || 0,
    failed_executions: backendStats.failed_count || 0,
    running_executions: backendStats.running_count || 0,
    executions_by_status: executionsByStatus,
  };
}

function transformExecutionDetailsResponse(raw: any): WorkflowExecution {
  const workflowTags = Array.isArray(raw.workflow_tags)
    ? raw.workflow_tags
    : [];
  const notes = Array.isArray(raw.notes) ? raw.notes : [];
  const webhookEvents = Array.isArray(raw.webhook_events)
    ? raw.webhook_events
    : [];

  const normalisedWebhookEvents = webhookEvents.map((event: any) => {
    const httpStatus =
      typeof event.http_status === "number" ? event.http_status : undefined;
    return {
      id: event.id,
      execution_id: event.execution_id ?? raw.execution_id,
      event_type: event.event_type ?? "webhook",
      status: event.status ?? "unknown",
      http_status: httpStatus,
      payload: event.payload ?? null,
      response_body: event.response_body ?? null,
      error_message: event.error_message ?? null,
      created_at: event.created_at ?? raw.updated_at ?? raw.created_at,
    } as ExecutionWebhookEvent;
  });

  // Debug logging to understand what the API is returning
  console.log('Raw API response for execution:', raw.execution_id, {
    input_data: raw.input_data,
    output_data: raw.output_data,
    input_uri: raw.input_uri,
    result_uri: raw.result_uri,
    input_size: raw.input_size,
    output_size: raw.output_size,
    // Log all keys to see what's available
    keys: Object.keys(raw)
  });

  // Handle input_data more carefully - check for different possible field names
  let inputData = raw.input_data;
  if (!inputData && raw.input) {
    inputData = raw.input;
  }
  if (!inputData && raw.inputs) {
    inputData = raw.inputs;
  }
  if (!inputData && raw.request_data) {
    inputData = raw.request_data;
  }

  // Handle output_data more carefully - check for different possible field names
  let outputData = raw.output_data;
  if (!outputData && raw.output) {
    outputData = raw.output;
  }
  if (!outputData && raw.outputs) {
    outputData = raw.outputs;
  }
  if (!outputData && raw.result_data) {
    outputData = raw.result_data;
  }
  if (!outputData && raw.response_data) {
    outputData = raw.response_data;
  }

  return {
    id: raw.id,
    workflow_id: raw.workflow_id,
    execution_id: raw.execution_id,
    haxen_request_id: raw.haxen_request_id ?? "",
    session_id: raw.session_id ?? undefined,
    actor_id: raw.actor_id ?? undefined,
    agent_node_id: raw.agent_node_id,
    parent_workflow_id: raw.parent_workflow_id ?? undefined,
    root_workflow_id: raw.root_workflow_id ?? undefined,
    workflow_depth:
      typeof raw.workflow_depth === "number" ? raw.workflow_depth : 0,
    reasoner_id: raw.reasoner_id,
    input_data: inputData ?? null,
    output_data: outputData ?? null,
    input_size: typeof raw.input_size === "number" ? raw.input_size : 0,
    output_size: typeof raw.output_size === "number" ? raw.output_size : 0,
    input_uri: raw.input_uri ?? undefined,
    result_uri: raw.result_uri ?? undefined,
    workflow_name: raw.workflow_name ?? undefined,
    workflow_tags: workflowTags,
    status: normalizeExecutionStatus(raw.status),
    started_at: raw.started_at ?? raw.created_at,
    completed_at: raw.completed_at ?? undefined,
    duration_ms:
      typeof raw.duration_ms === "number" ? raw.duration_ms : undefined,
    error_message: raw.error_message ?? undefined,
    retry_count: typeof raw.retry_count === "number" ? raw.retry_count : 0,
    created_at: raw.created_at,
    updated_at: raw.updated_at ?? raw.created_at,
    notes,
    webhook_registered:
      Boolean(raw.webhook_registered) || normalisedWebhookEvents.length > 0,
    webhook_events: normalisedWebhookEvents,
  };
}

function buildQueryString(params: Record<string, any>): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      if (Array.isArray(value)) {
        value.forEach((v) => searchParams.append(key, v.toString()));
      } else {
        searchParams.append(key, value.toString());
      }
    }
  });

  return searchParams.toString();
}

// Get executions summary with filtering and pagination
export async function getExecutionsSummary(
  filters: Partial<ExecutionFilters> = {},
  grouping?: ExecutionGrouping,
): Promise<PaginatedExecutions | GroupedExecutions> {
  const queryParams = {
    ...filters,
    ...grouping,
  };

  const queryString = buildQueryString(queryParams);
  const url = `/executions/summary${queryString ? `?${queryString}` : ""}`;

  if (grouping && grouping.group_by !== "none") {
    // For now, return empty grouped executions since backend doesn't support grouping yet
    const backendResponse = await fetchWrapper<any>(url);
    const transformed = transformPaginatedExecutions(backendResponse);
    return {
      groups: [],
      total_count: transformed.total_count || 0,
      page: transformed.page,
      page_size: transformed.page_size,
      total_pages: transformed.total_pages,
      has_next: transformed.has_next || false,
      has_prev: transformed.has_prev || false,
    } as GroupedExecutions;
  } else {
    const backendResponse = await fetchWrapper<any>(url);
    return transformPaginatedExecutions(backendResponse);
  }
}

// Get detailed execution information
export async function getExecutionDetails(
  executionId: string,
): Promise<WorkflowExecution> {
  const response = await fetchWrapper<any>(
    `/executions/${executionId}/details`,
  );
  return transformExecutionDetailsResponse(response);
}

export async function retryExecutionWebhook(
  executionId: string,
): Promise<void> {
  await fetchWrapper<unknown>(`/executions/${executionId}/webhook/retry`, {
    method: "POST",
  });
}

// Get execution statistics
export async function getExecutionStats(
  filters: Partial<ExecutionFilters> = {},
): Promise<ExecutionStats> {
  const queryString = buildQueryString(filters);
  const url = `/executions/stats${queryString ? `?${queryString}` : ""}`;
  const backendResponse = await fetchWrapper<any>(url);
  return transformExecutionStats(backendResponse);
}

// Stream real-time execution events
export function streamExecutionEvents(): EventSource {
  return new EventSource(`${API_BASE_URL}/executions/events`);
}

// Helper functions for common filtering scenarios
export async function getExecutionsByWorkflow(
  workflowId: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  return getExecutionsSummary({
    workflow_id: workflowId,
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

export async function getExecutionsBySession(
  sessionId: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  return getExecutionsSummary({
    session_id: sessionId,
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

export async function getExecutionsByAgent(
  agentNodeId: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  return getExecutionsSummary({
    agent_node_id: agentNodeId,
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

export async function getExecutionsByStatus(
  status: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  return getExecutionsSummary({
    status,
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

// Get grouped executions for dashboard views
export async function getGroupedExecutionsByWorkflow(
  filters: Partial<ExecutionFilters> = {},
): Promise<GroupedExecutions> {
  return getExecutionsSummary(filters, {
    group_by: "workflow",
    sort_by: "time",
    sort_order: "desc",
  }) as Promise<GroupedExecutions>;
}

export async function getGroupedExecutionsBySession(
  filters: Partial<ExecutionFilters> = {},
): Promise<GroupedExecutions> {
  return getExecutionsSummary(filters, {
    group_by: "session",
    sort_by: "time",
    sort_order: "desc",
  }) as Promise<GroupedExecutions>;
}

export async function getGroupedExecutionsByAgent(
  filters: Partial<ExecutionFilters> = {},
): Promise<GroupedExecutions> {
  return getExecutionsSummary(filters, {
    group_by: "agent",
    sort_by: "time",
    sort_order: "desc",
  }) as Promise<GroupedExecutions>;
}

export async function getGroupedExecutionsByStatus(
  filters: Partial<ExecutionFilters> = {},
): Promise<GroupedExecutions> {
  return getExecutionsSummary(filters, {
    group_by: "status",
    sort_by: "time",
    sort_order: "desc",
  }) as Promise<GroupedExecutions>;
}

// Search executions
export async function searchExecutions(
  searchTerm: string,
  filters: Partial<ExecutionFilters> = {},
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  const result = await getExecutionsSummary({
    ...filters,
    search: searchTerm,
    page,
    page_size: pageSize,
  });

  // Ensure we return PaginatedExecutions
  if ("groups" in result) {
    // Convert GroupedExecutions to PaginatedExecutions
    const flatExecutions =
      result.groups?.flatMap((group) => group.executions) || [];
    return {
      executions: flatExecutions,
      total: result.total_count || 0,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
      total_count: result.total_count || 0,
      has_next: result.has_next || false,
      has_prev: result.has_prev || false,
    };
  }

  return result as PaginatedExecutions;
}

// Get recent executions (last 24 hours by default)
export async function getRecentExecutions(
  hours: number = 24,
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - hours * 60 * 60 * 1000);

  return getExecutionsSummary({
    start_time: startTime.toISOString(),
    end_time: endTime.toISOString(),
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

// Get executions in a time range
export async function getExecutionsInTimeRange(
  startTime: Date,
  endTime: Date,
  filters: Partial<ExecutionFilters> = {},
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedExecutions> {
  return getExecutionsSummary({
    ...filters,
    start_time: startTime.toISOString(),
    end_time: endTime.toISOString(),
    page,
    page_size: pageSize,
  }) as Promise<PaginatedExecutions>;
}

// Get enhanced executions with workflow names and better structure
export async function getEnhancedExecutions(
  filters: Partial<ExecutionFilters> = {},
  sortBy: string = "started_at",
  sortOrder: "asc" | "desc" = "desc",
  page: number = 1,
  pageSize: number = 20,
  signal?: AbortSignal,
): Promise<EnhancedExecutionsResponse> {
  const queryParams = {
    ...filters,
    sort_by: sortBy,
    sort_order: sortOrder,
    page,
    page_size: pageSize,
  };

  const queryString = buildQueryString(queryParams);
  const url = `/executions/enhanced${queryString ? `?${queryString}` : ""}`;

  return fetchWrapper<EnhancedExecutionsResponse>(url, { signal });
}

// Notes API functions

// Get notes for a specific execution
export async function getExecutionNotes(
  executionId: string,
  filters: NotesFilters = {},
): Promise<NotesResponse> {
  const queryParams: Record<string, any> = {};

  if (filters.tags && filters.tags.length > 0) {
    queryParams.tags = filters.tags.join(",");
  }

  const queryString = buildQueryString(queryParams);
  const url = `/executions/${executionId}/notes${queryString ? `?${queryString}` : ""}`;

  return fetchWrapper<NotesResponse>(url);
}

// Add a note to an execution
export async function addExecutionNote(
  executionId: string,
  noteRequest: AddNoteRequest,
): Promise<AddNoteResponse> {
  const options: RequestInit = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Execution-ID": executionId,
    },
    body: JSON.stringify(noteRequest),
  };

  return fetchWrapper<AddNoteResponse>("/executions/note", options);
}

// Get all unique tags from execution notes
export async function getExecutionNoteTags(
  executionId: string,
): Promise<string[]> {
  try {
    const notesResponse = await getExecutionNotes(executionId);
    const allTags = notesResponse.notes.flatMap((note) => note.tags || []);
    const uniqueTags = Array.from(new Set(allTags)).sort();
    return uniqueTags;
  } catch (error) {
    console.error("Failed to fetch note tags:", error);
    return [];
  }
}

// Stream real-time notes updates for an execution
export function streamExecutionNotes(executionId: string): EventSource {
  return new EventSource(
    `${API_BASE_URL}/executions/${executionId}/notes/stream`,
  );
}

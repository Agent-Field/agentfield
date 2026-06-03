import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { getWorkflowRunDetail } from '../services/workflowsApi';
import type { WorkflowRunDetailResponse } from '../services/workflowsApi';
import { normalizeExecutionStatus } from '../utils/status';

interface WorkflowDAGNode {
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
  children: WorkflowDAGNode[];
  notes: any[];
  notes_count: number;
  latest_note?: any;
}

interface WorkflowDAGResponse {
  root_workflow_id: string;
  workflow_status: string;
  workflow_name: string;
  session_id?: string;
  actor_id?: string;
  total_nodes: number;
   displayed_nodes?: number;
  max_depth: number;
  dag: WorkflowDAGNode;
  timeline: WorkflowDAGNode[];
   status_counts?: Record<string, number>;
}

interface WorkflowDAGError {
  message: string;
  code?: string;
  details?: any;
}

interface WorkflowDAGState {
  data: WorkflowDAGResponse | null;
  loading: boolean;
  error: WorkflowDAGError | null;
  lastFetch: Date | null;
}

interface UseWorkflowDAGOptions {
  /** Auto-refresh interval in milliseconds (0 to disable) */
  refreshInterval?: number;
  /** Enable smart polling that adjusts based on workflow status */
  smartPolling?: boolean;
  /** Callback for data updates */
  onDataUpdate?: (data: WorkflowDAGResponse) => void;
  /** Callback for errors */
  onError?: (error: WorkflowDAGError) => void;
  /** Enable automatic retry on errors */
  enableRetry?: boolean;
  /** Maximum number of retries */
  maxRetries?: number;
}

interface UseWorkflowDAGReturn extends WorkflowDAGState {
  refresh: () => void;
  clearError: () => void;
  reset: () => void;
  hasData: boolean;
  hasError: boolean;
  isRefreshing: boolean;
  isEmpty: boolean;
  hasRunningWorkflows: boolean;
  currentPollingInterval: number;
}

/**
 * Determines if the workflow is currently running based on the overall workflow status
 */
function hasRunningWorkflows(data: WorkflowDAGResponse | null): boolean {
  if (!data) return false;

  const normalized = normalizeExecutionStatus(data.workflow_status);
  if (normalized === 'running' || normalized === 'queued' || normalized === 'pending') {
    return true;
  }

  return data.timeline.some((node) => {
    const nodeStatus = normalizeExecutionStatus(node.status);
    return nodeStatus === 'running' || nodeStatus === 'queued' || nodeStatus === 'pending';
  });
}

/**
 * Determines the optimal polling interval based on workflow status
 */
function getSmartPollingInterval(data: WorkflowDAGResponse | null, baseInterval: number): number {
  if (!data) return baseInterval;

  const hasRunning = hasRunningWorkflows(data);

  if (hasRunning) {
    // Fast polling for active workflows (2-3 seconds)
    return 2500;
  }

  // Check if the workflow completed recently (within last 5 minutes)
  const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
  const rootNode = data.dag; // The root node represents the overall workflow

  if (rootNode.completed_at) {
    const completedTime = new Date(rootNode.completed_at).getTime();
    if (completedTime > fiveMinutesAgo) {
      // Medium polling for recently completed workflows (10-15 seconds)
      return 12000;
    }
  }

  // Slow polling for stable workflows (60 seconds)
  return 60000;
}

/**
 * Custom hook for workflow DAG data fetching with smart polling
 */
export function useWorkflowDAG(workflowId: string | null, options: UseWorkflowDAGOptions = {}): UseWorkflowDAGReturn {
  const {
    refreshInterval = 30000, // Default 30 seconds
    smartPolling = true,
    onDataUpdate,
    onError,
    enableRetry = true,
    maxRetries = 2
  } = options;

  const [state, setState] = useState<WorkflowDAGState>({
    data: null,
    loading: false,
    error: null,
    lastFetch: null,
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);
  const currentIntervalRef = useRef(refreshInterval);

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
   * Handle errors
   */
  const handleError = useCallback((error: Error) => {
    if (!mountedRef.current) return;

    const workflowError: WorkflowDAGError = {
      message: error.message,
      code: error.name,
      details: error
    };

    setState(prev => ({
      ...prev,
      loading: false,
      error: workflowError,
    }));

    onError?.(workflowError);
  }, [onError]);

  /**
   * Update state from workflow DAG data
   */
  const updateStateFromData = useCallback((data: WorkflowDAGResponse) => {
    if (!mountedRef.current) return;

    setState(prev => ({
      ...prev,
      data,
      loading: false,
      error: null,
      lastFetch: new Date(),
    }));

    onDataUpdate?.(data);
  }, [onDataUpdate]);

  /**
   * Fetch workflow DAG data from API with retry logic
   */
  const fetchWorkflowDAG = useCallback(async (retryCount = 0) => {
    if (!workflowId || !mountedRef.current) return;

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const detail = await getWorkflowRunDetail(workflowId);
      const transformed = transformRunDetailToDag(detail);
      updateStateFromData(transformed);
    } catch (error) {
      if (enableRetry && retryCount < maxRetries) {
        // Retry with exponential backoff
        const delay = 1000 * Math.pow(2, retryCount);
        setTimeout(() => {
          if (mountedRef.current) {
            fetchWorkflowDAG(retryCount + 1);
          }
        }, delay);
      } else {
        handleError(error as Error);
      }
    }
  }, [workflowId, updateStateFromData, handleError, enableRetry, maxRetries]);

  /**
   * Schedule next refresh with smart polling
   */
  const scheduleRefresh = useCallback(() => {
    if (!mountedRef.current) return;

    clearRefreshTimeout();

    const interval = smartPolling
      ? getSmartPollingInterval(state.data, refreshInterval)
      : refreshInterval;

    currentIntervalRef.current = interval;

    if (interval > 0) {
      refreshTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          fetchWorkflowDAG();
        }
      }, interval);
    }
  }, [state.data, refreshInterval, smartPolling, fetchWorkflowDAG, clearRefreshTimeout]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(() => {
    clearRefreshTimeout();
    fetchWorkflowDAG();
  }, [fetchWorkflowDAG, clearRefreshTimeout]);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
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
    });
  }, [clearRefreshTimeout]);

  // Initial fetch and setup refresh cycle
  useEffect(() => {
    if (workflowId) {
      fetchWorkflowDAG();
    }
  }, [workflowId, fetchWorkflowDAG]);

  // Schedule refresh after successful fetch
  useEffect(() => {
    if (state.data && !state.error) {
      scheduleRefresh();
    }
  }, [state.data, state.error, scheduleRefresh]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearRefreshTimeout();
    };
  }, [clearRefreshTimeout]);

  // Memoize computed properties
  const computedProperties = useMemo(() => ({
    hasData: state.data !== null,
    hasError: state.error !== null,
    isRefreshing: state.loading && state.data !== null,
    isEmpty: !state.loading && !state.data && !state.error,
    hasRunningWorkflows: hasRunningWorkflows(state.data),
    currentPollingInterval: currentIntervalRef.current,
  }), [state]);

  return useMemo(() => ({
    ...state,
    refresh,
    clearError,
    reset,
    ...computedProperties,
  }), [state, refresh, clearError, reset, computedProperties]);
}

/**
 * Hook for workflow DAG with fast polling (2.5 seconds)
 */
export function useWorkflowDAGFast(workflowId: string | null) {
  return useWorkflowDAG(workflowId, {
    refreshInterval: 2500,
    smartPolling: false,
    enableRetry: true,
    maxRetries: 2
  });
}

/**
 * Hook for workflow DAG with smart polling (adjusts based on status)
 */
export function useWorkflowDAGSmart(workflowId: string | null) {
  return useWorkflowDAG(workflowId, {
    refreshInterval: 30000, // Base interval
    smartPolling: true,
    enableRetry: true,
    maxRetries: 2
  });
}

/**
 * Hook for workflow DAG with custom interval
 */
export function useWorkflowDAGWithInterval(workflowId: string | null, intervalMs: number) {
  return useWorkflowDAG(workflowId, {
    refreshInterval: intervalMs,
    smartPolling: false,
    enableRetry: true,
    maxRetries: 2
  });
}

function transformRunDetailToDag(detail: WorkflowRunDetailResponse): WorkflowDAGResponse {
  const nodeMap = new Map<string, WorkflowDAGNode>();
  let maxDepth = 0;

  detail.executions.forEach((exec) => {
    const durationMs = computeDurationMs(exec.started_at, exec.completed_at ?? undefined);
    const node: WorkflowDAGNode = {
      workflow_id: exec.workflow_id,
      execution_id: exec.execution_id,
      agent_node_id: exec.agent_node_id,
      reasoner_id: exec.reasoner_id,
      status: exec.status,
      started_at: exec.started_at,
      completed_at: exec.completed_at ?? undefined,
      duration_ms: durationMs,
      parent_workflow_id: exec.parent_workflow_id ?? undefined,
      parent_execution_id: exec.parent_execution_id ?? undefined,
      workflow_depth: exec.workflow_depth,
      children: [],
      notes: [],
      notes_count: 0,
    };

    if (exec.workflow_depth > maxDepth) {
      maxDepth = exec.workflow_depth;
    }

    nodeMap.set(exec.execution_id, node);
  });

  const orphans: WorkflowDAGNode[] = [];

  nodeMap.forEach((node) => {
    if (node.parent_execution_id) {
      const parent = nodeMap.get(node.parent_execution_id);
      if (parent) {
        parent.children.push(node);
      } else {
        orphans.push(node);
      }
    } else {
      orphans.push(node);
    }
  });

  const rootExecutionId = detail.run.root_execution_id ?? undefined;
  const rootNode =
    (rootExecutionId && nodeMap.get(rootExecutionId)) ||
    [...nodeMap.values()].find((node) => !node.parent_execution_id) ||
    null;

  const timeline = [...nodeMap.values()].sort((a, b) => {
    return new Date(a.started_at).getTime() - new Date(b.started_at).getTime();
  });

  const normalizedStatus = normalizeExecutionStatus(detail.run.status);

  const fallbackRoot: WorkflowDAGNode = {
    workflow_id: detail.run.root_workflow_id,
    execution_id: rootExecutionId ?? detail.run.root_workflow_id,
    agent_node_id: rootNode?.agent_node_id ?? '',
    reasoner_id: rootNode?.reasoner_id ?? detail.run.root_workflow_id,
    status: normalizedStatus,
    started_at: detail.run.created_at,
    completed_at: detail.run.completed_at ?? undefined,
    duration_ms: computeDurationMs(detail.run.created_at, detail.run.completed_at ?? undefined),
    parent_workflow_id: undefined,
    parent_execution_id: undefined,
    workflow_depth: 0,
    children: orphans,
    notes: [],
    notes_count: 0,
  };

  const totalNodes = detail.run.total_steps ?? nodeMap.size;
  const displayedNodes = detail.run.returned_steps ?? nodeMap.size;

  return {
    root_workflow_id: detail.run.root_workflow_id,
    workflow_status: normalizedStatus,
    workflow_name: rootNode?.reasoner_id ?? detail.run.root_workflow_id,
    total_nodes: totalNodes,
    displayed_nodes: displayedNodes,
    max_depth: maxDepth,
    dag: rootNode ?? fallbackRoot,
    timeline,
    status_counts: detail.run.status_counts,
  };
}

function computeDurationMs(startedAt: string, completedAt?: string): number | undefined {
  if (!startedAt || !completedAt) {
    return undefined;
  }
  const start = new Date(startedAt).getTime();
  const end = new Date(completedAt).getTime();
  if (Number.isNaN(start) || Number.isNaN(end)) {
    return undefined;
  }
  const duration = end - start;
  return duration >= 0 ? duration : undefined;
}

import { useState, useEffect, useMemo } from 'react';
import { getExecutionDetails } from '../services/executionsApi';
import type { WorkflowExecution } from '../types/executions';
import {
  isSuccessStatus,
  isRunningStatus,
  isFailureStatus,
  isTimeoutStatus,
  normalizeExecutionStatus,
} from '../utils/status';

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
  session_id?: string;
  actor_id?: string;
  total_nodes: number;
  max_depth: number;
  dag: WorkflowDAGNode;
  timeline: WorkflowDAGNode[];
}

interface MainNodeExecutionState {
  execution: WorkflowExecution | null;
  loading: boolean;
  error: string | null;
}

interface UseMainNodeExecutionReturn extends MainNodeExecutionState {
  refresh: () => void;
  hasInputData: boolean;
  hasOutputData: boolean;
  isCompleted: boolean;
  isRunning: boolean;
  hasFailed: boolean;
}

/**
 * Custom hook to fetch execution data for the main workflow node (root node)
 * This hook extracts the root node from DAG data and fetches its execution details
 */
export function useMainNodeExecution(dagData: WorkflowDAGResponse | null): UseMainNodeExecutionReturn {
  const [state, setState] = useState<MainNodeExecutionState>({
    execution: null,
    loading: false,
    error: null,
  });

  // Extract main node execution ID from DAG data
  const mainNodeExecutionId = useMemo(() => {
    if (!dagData?.dag) return null;
    return dagData.dag.execution_id;
  }, [dagData]);

  // Extract main node status for quick access
  const mainNodeStatus = useMemo(() => {
    if (!dagData?.dag) return null;
    return dagData.dag.status;
  }, [dagData]);

  const fetchExecution = async (executionId: string) => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const execution = await getExecutionDetails(executionId);
      setState(prev => ({
        ...prev,
        execution,
        loading: false,
        error: null
      }));
    } catch (error) {
      console.error('Failed to fetch main node execution:', error);
      setState(prev => ({
        ...prev,
        execution: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch execution data'
      }));
    }
  };

  // Fetch execution data when main node execution ID changes
  useEffect(() => {
    if (mainNodeExecutionId) {
      fetchExecution(mainNodeExecutionId);
    } else {
      setState({
        execution: null,
        loading: false,
        error: null,
      });
    }
  }, [mainNodeExecutionId]);

  const refresh = () => {
    if (mainNodeExecutionId) {
      fetchExecution(mainNodeExecutionId);
    }
  };

  // Computed properties
  const computedProperties = useMemo(() => {
    const execution = state.execution;
    const status = normalizeExecutionStatus(execution?.status || mainNodeStatus);

    return {
      hasInputData: !!execution?.input_data,
      hasOutputData: !!execution?.output_data,
      isCompleted: isSuccessStatus(status) || isTimeoutStatus(status) || status === 'cancelled',
      isRunning: isRunningStatus(status) || status === 'queued',
      hasFailed: isFailureStatus(status) || isTimeoutStatus(status),
    };
  }, [state.execution, mainNodeStatus]);

  return {
    ...state,
    refresh,
    ...computedProperties,
  };
}

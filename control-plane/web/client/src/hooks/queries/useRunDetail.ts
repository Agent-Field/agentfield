import { useQuery } from "@tanstack/react-query";
import { getWorkflowDAGLightweight } from "../../services/workflowsApi";
import { getExecutionDetails } from "../../services/executionsApi";
import type { WorkflowDAGLightweightResponse } from "../../types/workflows";
import type { WorkflowExecution } from "../../types/executions";
import { normalizeExecutionStatus } from "../../utils/status";
import { useSSESync } from "../useSSEQuerySync";

// With the execution stream connected, events refresh a live run (coalesced
// in useSSEQuerySync); polling is only the safety net for a missed event.
// Without the stream, polling is the only source of updates.
const POLL_WITH_STREAM_MS = 10_000;
const POLL_WITHOUT_STREAM_MS = 1_500;

export function useRunDAG(runId: string | undefined) {
  const { execConnected } = useSSESync();
  return useQuery<WorkflowDAGLightweightResponse>({
    queryKey: ["run-dag", runId],
    queryFn: () => getWorkflowDAGLightweight(runId!),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.workflow_status;
      if (status === "running" || status === "pending") {
        return execConnected ? POLL_WITH_STREAM_MS : POLL_WITHOUT_STREAM_MS;
      }
      return false;
    },
  });
}

export function useStepDetail(executionId: string | undefined) {
  const { execConnected } = useSSESync();
  return useQuery<WorkflowExecution>({
    queryKey: ["step-detail", executionId],
    queryFn: () => getExecutionDetails(executionId!),
    enabled: !!executionId,
    refetchInterval: (query) => {
      const st = normalizeExecutionStatus(query.state.data?.status);
      const active =
        st === "running" ||
        st === "pending" ||
        st === "queued" ||
        st === "waiting";
      if (!active) return false;
      return execConnected ? POLL_WITH_STREAM_MS : POLL_WITHOUT_STREAM_MS;
    },
  });
}

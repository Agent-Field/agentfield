import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  cancelExecution,
  pauseExecution,
  restartExecution,
  resumeExecution,
} from "../../services/executionsApi";
import type {
  CancelExecutionResponse,
  PauseExecutionResponse,
  RestartExecutionRequest,
  RestartExecutionResponse,
  ResumeExecutionResponse,
} from "../../services/executionsApi";
import { cancelWorkflowTree } from "../../services/workflowsApi";
import { saveGoldenRun } from "../../services/workflowsApi";
import type { CancelWorkflowTreeResponse } from "../../services/workflowsApi";

export function useCancelExecution() {
  const queryClient = useQueryClient();
  return useMutation<CancelExecutionResponse, Error, string>({
    mutationFn: (executionId: string) => cancelExecution(executionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

export function usePauseExecution() {
  const queryClient = useQueryClient();
  return useMutation<PauseExecutionResponse, Error, string>({
    mutationFn: (executionId: string) => pauseExecution(executionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

export function useCancelWorkflowTree() {
  const queryClient = useQueryClient();
  return useMutation<
    CancelWorkflowTreeResponse,
    Error,
    { workflowId: string; reason?: string }
  >({
    mutationFn: ({ workflowId, reason }) =>
      cancelWorkflowTree(workflowId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

export function useResumeExecution() {
  const queryClient = useQueryClient();
  return useMutation<ResumeExecutionResponse, Error, string>({
    mutationFn: (executionId: string) => resumeExecution(executionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

export function useRestartExecution() {
  const queryClient = useQueryClient();
  return useMutation<
    RestartExecutionResponse,
    Error,
    string | { executionId: string; request?: RestartExecutionRequest }
  >({
    mutationFn: (value) => {
      if (typeof value === "string") {
        return restartExecution(value);
      }
      return restartExecution(value.executionId, value.request);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

export function useSaveGoldenRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      runId,
      name,
      tags,
    }: {
      runId: string;
      name?: string;
      tags?: string[];
    }) => saveGoldenRun(runId, { name, tags }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    },
  });
}

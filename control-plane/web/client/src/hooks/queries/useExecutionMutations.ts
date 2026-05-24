import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  cancelExecution,
  pauseExecution,
  resumeExecution,
} from "../../services/executionsApi";
import type {
  CancelExecutionResponse,
  PauseExecutionResponse,
  ResumeExecutionResponse,
} from "../../services/executionsApi";
import { cancelWorkflowTree } from "../../services/workflowsApi";
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
    mutationFn: ({ workflowId, reason }) => cancelWorkflowTree(workflowId, reason),
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

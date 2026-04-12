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

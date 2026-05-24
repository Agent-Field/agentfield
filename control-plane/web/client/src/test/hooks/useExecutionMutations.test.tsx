// @ts-nocheck
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const workflowsApiState = vi.hoisted(() => ({
  cancelWorkflowTree: vi.fn(),
}));
const executionsApiState = vi.hoisted(() => ({
  cancelExecution: vi.fn(),
  pauseExecution: vi.fn(),
  resumeExecution: vi.fn(),
}));

vi.mock("@/services/workflowsApi", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, ...workflowsApiState };
});
vi.mock("@/services/executionsApi", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, ...executionsApiState };
});

import {
  useCancelExecution,
  useCancelWorkflowTree,
  usePauseExecution,
  useResumeExecution,
} from "@/hooks/queries/useExecutionMutations";

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe("useExecutionMutations", () => {
  let client: QueryClient;

  beforeEach(() => {
    client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  afterEach(() => {
    client.clear();
    vi.clearAllMocks();
  });

  it("useCancelWorkflowTree calls cancelWorkflowTree and invalidates run queries on success", async () => {
    workflowsApiState.cancelWorkflowTree.mockResolvedValue({
      run_id: "run-1",
      total_nodes: 2,
      cancelled_count: 2,
      skipped_count: 0,
      error_count: 0,
      nodes: [],
      cancelled_at: "2026-04-08T12:00:00Z",
    });
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useCancelWorkflowTree(), { wrapper: makeWrapper(client) });

    await act(async () => {
      await result.current.mutateAsync({ workflowId: "run-1", reason: "user clicked cancel" });
    });

    expect(workflowsApiState.cancelWorkflowTree).toHaveBeenCalledWith("run-1", "user clicked cancel");
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["runs"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["run-dag"] });
  });

  it("useCancelWorkflowTree surfaces errors without invalidating", async () => {
    workflowsApiState.cancelWorkflowTree.mockRejectedValue(new Error("boom"));
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useCancelWorkflowTree(), { wrapper: makeWrapper(client) });

    await expect(
      act(async () => {
        await result.current.mutateAsync({ workflowId: "run-x" });
      })
    ).rejects.toThrow("boom");

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("useCancelExecution / usePauseExecution / useResumeExecution invalidate the same run queries", async () => {
    executionsApiState.cancelExecution.mockResolvedValue({ status: "cancelled" });
    executionsApiState.pauseExecution.mockResolvedValue({ status: "paused" });
    executionsApiState.resumeExecution.mockResolvedValue({ status: "running" });

    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const cancel = renderHook(() => useCancelExecution(), { wrapper: makeWrapper(client) });
    const pause = renderHook(() => usePauseExecution(), { wrapper: makeWrapper(client) });
    const resume = renderHook(() => useResumeExecution(), { wrapper: makeWrapper(client) });

    await act(async () => {
      await cancel.result.current.mutateAsync("exec-1");
      await pause.result.current.mutateAsync("exec-2");
      await resume.result.current.mutateAsync("exec-3");
    });

    expect(executionsApiState.cancelExecution).toHaveBeenCalledWith("exec-1");
    expect(executionsApiState.pauseExecution).toHaveBeenCalledWith("exec-2");
    expect(executionsApiState.resumeExecution).toHaveBeenCalledWith("exec-3");

    // Each of the three mutations invalidates ["runs"] and ["run-dag"] once.
    const runsCalls = invalidateSpy.mock.calls.filter(
      (c) => Array.isArray((c[0] as { queryKey: unknown[] }).queryKey) && (c[0] as { queryKey: string[] }).queryKey[0] === "runs"
    );
    const dagCalls = invalidateSpy.mock.calls.filter(
      (c) => Array.isArray((c[0] as { queryKey: unknown[] }).queryKey) && (c[0] as { queryKey: string[] }).queryKey[0] === "run-dag"
    );
    expect(runsCalls.length).toBe(3);
    expect(dagCalls.length).toBe(3);
  });
});

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  useMainNodeExecution,
} from "@/hooks/useMainNodeExecution";
import {
  useWorkflowDAG,
  useWorkflowDAGFast,
  useWorkflowDAGSmart,
  useWorkflowDAGWithInterval,
} from "@/hooks/useWorkflowDAG";

const executionsApiState = vi.hoisted(() => ({
  getExecutionDetails: vi.fn(),
}));

const workflowsApiState = vi.hoisted(() => ({
  getWorkflowRunDetail: vi.fn(),
}));

vi.mock("@/services/executionsApi", () => executionsApiState);
vi.mock("@/services/workflowsApi", () => workflowsApiState);

describe("workflow hooks", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("useMainNodeExecution fetches the root execution and derives status flags", async () => {
    const dagData = {
      dag: {
        execution_id: "exec-1",
        status: "completed",
      },
    } as any;

    executionsApiState.getExecutionDetails.mockResolvedValue({
      execution_id: "exec-1",
      status: "completed",
      input_data: { prompt: "hi" },
      output_data: { text: "done" },
    });

    const { result } = renderHook(() => useMainNodeExecution(dagData));

    await waitFor(() => expect(result.current.execution?.execution_id).toBe("exec-1"));
    expect(result.current.hasInputData).toBe(true);
    expect(result.current.hasOutputData).toBe(true);
    expect(result.current.isCompleted).toBe(true);
    expect(result.current.isRunning).toBe(false);
    expect(result.current.hasFailed).toBe(false);

    await act(async () => {
      result.current.refresh();
    });
    expect(executionsApiState.getExecutionDetails).toHaveBeenCalledTimes(2);
  });

  it("useMainNodeExecution handles missing dag data and fetch failures", async () => {
    const emptyHook = renderHook(() => useMainNodeExecution(null));
    expect(emptyHook.result.current.execution).toBeNull();
    expect(emptyHook.result.current.loading).toBe(false);

    executionsApiState.getExecutionDetails.mockRejectedValueOnce(new Error("execution failed"));

    const failedHook = renderHook(() =>
      useMainNodeExecution({
        dag: { execution_id: "exec-2", status: "failed" },
      } as any)
    );

    await waitFor(() => expect(failedHook.result.current.error).toBe("execution failed"));
    expect(failedHook.result.current.hasFailed).toBe(true);
    expect(failedHook.result.current.isCompleted).toBe(false);
  });

  it("useWorkflowDAG transforms run details and applies smart polling", async () => {
    const onDataUpdate = vi.fn();
    const detail = {
      run: {
        root_execution_id: "root-exec",
        root_workflow_id: "wf-root",
        status: "running",
        created_at: "2026-01-01T00:00:00Z",
        completed_at: null,
        total_steps: 2,
        returned_steps: 2,
        status_counts: { running: 1, completed: 1 },
      },
      executions: [
        {
          workflow_id: "wf-root",
          execution_id: "root-exec",
          agent_node_id: "node-root",
          reasoner_id: "reasoner-root",
          status: "running",
          started_at: "2026-01-01T00:00:00Z",
          completed_at: null,
          parent_workflow_id: null,
          parent_execution_id: null,
          workflow_depth: 0,
        },
        {
          workflow_id: "wf-child",
          execution_id: "child-exec",
          agent_node_id: "node-child",
          reasoner_id: "reasoner-child",
          status: "completed",
          started_at: "2026-01-01T00:00:01Z",
          completed_at: "2026-01-01T00:00:03Z",
          parent_workflow_id: "wf-root",
          parent_execution_id: "root-exec",
          workflow_depth: 1,
        },
      ],
    } as any;

    workflowsApiState.getWorkflowRunDetail.mockResolvedValue(detail);

    const { result } = renderHook(() =>
      useWorkflowDAG("wf-root", { refreshInterval: 30000, smartPolling: true, onDataUpdate })
    );

    await waitFor(() => expect(result.current.data?.root_workflow_id).toBe("wf-root"));
    expect(result.current.data?.dag.children).toHaveLength(1);
    expect(result.current.data?.timeline[1].duration_ms).toBe(2000);
    expect(result.current.hasRunningWorkflows).toBe(true);
    expect(result.current.currentPollingInterval).toBe(30000);
    expect(onDataUpdate).toHaveBeenCalled();
  });

  it("useWorkflowDAG retries on failure and then exposes errors", async () => {
    workflowsApiState.getWorkflowRunDetail.mockRejectedValue(new Error("dag failed"));

    const onError = vi.fn();
    const { result } = renderHook(() =>
      useWorkflowDAG("wf-error", {
        refreshInterval: 0,
        enableRetry: true,
        maxRetries: 1,
        onError,
      })
    );

    await waitFor(() => expect(workflowsApiState.getWorkflowRunDetail).toHaveBeenCalledTimes(2), {
      timeout: 2000,
    });
    await waitFor(() => expect(result.current.error?.message).toBe("dag failed"), {
      timeout: 2000,
    });
    expect(workflowsApiState.getWorkflowRunDetail).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "dag failed" }));

    act(() => {
      result.current.clearError();
      result.current.reset();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBeNull();
  });

  it("workflow DAG variants call through to the shared hook", async () => {
    workflowsApiState.getWorkflowRunDetail.mockResolvedValue({
      run: {
        root_execution_id: null,
        root_workflow_id: "wf-variant",
        status: "completed",
        created_at: "2026-01-01T00:00:00Z",
        completed_at: "2026-01-01T00:00:10Z",
        total_steps: 0,
        returned_steps: 0,
        status_counts: {},
      },
      executions: [],
    } as any);

    renderHook(() => useWorkflowDAGFast("wf-variant"));
    renderHook(() => useWorkflowDAGSmart("wf-variant"));
    renderHook(() => useWorkflowDAGWithInterval("wf-variant", 7777));

    await waitFor(() =>
      expect(workflowsApiState.getWorkflowRunDetail).toHaveBeenCalledTimes(3)
    );
  });
});

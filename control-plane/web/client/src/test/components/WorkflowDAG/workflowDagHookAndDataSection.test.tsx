// @ts-nocheck
import React from "react";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  getGlobalApiKey: vi.fn(),
  fetch: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  getGlobalApiKey: () => state.getGlobalApiKey(),
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  Database: ({ className }: { className?: string }) => <span className={className}>database</span>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
}));

vi.mock("@/components/ui/UnifiedDataPanel", () => ({
  UnifiedDataPanel: ({
    title,
    data,
    type,
    size,
  }: {
    title: string;
    data: unknown;
    type: string;
    size?: number;
  }) => (
    <div>
      <div>{title}</div>
      <div>{`type:${type}`}</div>
      <div>{`data:${JSON.stringify(data)}`}</div>
      <div>{`size:${size ?? "none"}`}</div>
    </div>
  ),
}));

describe("useNodeDetails and DataSection", () => {
  beforeEach(() => {
    state.getGlobalApiKey.mockReset();
    state.fetch.mockReset();
    state.getGlobalApiKey.mockReturnValue(null);
    vi.stubGlobal("fetch", state.fetch);
  });

  it("resets cleanly when executionId is missing", async () => {
    const { useNodeDetails } = await import("@/components/WorkflowDAG/hooks/useNodeDetails");
    const { result } = renderHook(() => useNodeDetails(undefined));

    expect(result.current.nodeDetails).toBeUndefined();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("fetches, transforms, caches, and refetches node details", async () => {
    const { useNodeDetails } = await import("@/components/WorkflowDAG/hooks/useNodeDetails");
    state.getGlobalApiKey.mockReturnValue("secret-key");
    state.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        input_data: { prompt: "hello" },
        output_data: { answer: "world" },
        error_message: undefined,
        cost: 1.25,
        memory_updates: [{ action: "set" }],
        performance_metrics: {
          response_time_ms: 321,
          tokens_used: 99,
        },
      }),
    });

    const { result, rerender } = renderHook(({ executionId }: { executionId?: string }) => useNodeDetails(executionId), {
      initialProps: { executionId: "exec-1" },
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.nodeDetails).toEqual({
        input: { prompt: "hello" },
        output: { answer: "world" },
        error_message: undefined,
        cost: 1.25,
        memory_updates: [{ action: "set" }],
        performance_metrics: {
          response_time_ms: 321,
          tokens_used: 99,
        },
      });
    });

    expect(state.fetch).toHaveBeenCalledWith("/api/ui/v1/executions/exec-1/details", {
      headers: { "X-API-Key": "secret-key" },
    });

    rerender({ executionId: "exec-1" });
    await waitFor(() => {
      expect(result.current.nodeDetails?.output).toEqual({ answer: "world" });
    });
    expect(state.fetch).toHaveBeenCalledTimes(1);

    state.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        input: "raw-input",
        output: "raw-output",
        duration_ms: 777,
      }),
    });

    result.current.refetch();
    await waitFor(() => {
      expect(result.current.nodeDetails).toEqual({
        input: "raw-input",
        output: "raw-output",
        error_message: undefined,
        cost: undefined,
        memory_updates: [],
        performance_metrics: undefined,
      });
    });
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("surfaces HTTP failures and thrown errors", async () => {
    const { useNodeDetails } = await import("@/components/WorkflowDAG/hooks/useNodeDetails");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    state.fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Server Error",
    });

    const first = renderHook(() => useNodeDetails("exec-fail"));
    await waitFor(() => {
      expect(first.result.current.error).toContain("Failed to fetch execution details: 500 Server Error");
    });

    state.fetch.mockRejectedValueOnce(new Error("kaboom"));
    const second = renderHook(() => useNodeDetails("exec-throw"));
    await waitFor(() => {
      expect(second.result.current.error).toBe("kaboom");
    });

    errorSpy.mockRestore();
  });

  it("renders DataSection panels and size summaries", async () => {
    const { DataSection } = await import("@/components/WorkflowDAG/sections/DataSection");

    const input = { text: "hello world" };
    const output = "x".repeat(2048);
    render(
      <DataSection
        node={{
          workflow_id: "wf-1",
          execution_id: "exec-1",
          agent_node_id: "agent-1",
          reasoner_id: "reasoner",
          status: "running",
          started_at: "2026-04-09T10:00:00Z",
          workflow_depth: 1,
        }}
        details={{ input, output }}
      />,
    );

    expect(screen.getByText("Input & Output")).toBeInTheDocument();
    expect(screen.getByText("Input Data")).toBeInTheDocument();
    expect(screen.getByText("Output Data")).toBeInTheDocument();
    expect(screen.getByText("Input Size")).toBeInTheDocument();
    expect(screen.getByText("Output Size")).toBeInTheDocument();
    expect(screen.getByText(/2.0 KB|2 KB/)).toBeInTheDocument();
  });

  it("renders DataSection without size summary when payloads are empty", async () => {
    const { DataSection } = await import("@/components/WorkflowDAG/sections/DataSection");

    render(
      <DataSection
        node={{
          workflow_id: "wf-1",
          execution_id: "exec-1",
          agent_node_id: "agent-1",
          reasoner_id: "reasoner",
          status: "running",
          started_at: "2026-04-09T10:00:00Z",
          workflow_depth: 1,
        }}
      />,
    );

    expect(screen.queryByText("Input Size")).not.toBeInTheDocument();
    expect(screen.getAllByText("size:none")).toHaveLength(2);
  });
});

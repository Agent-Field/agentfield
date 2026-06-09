// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RunTrace, buildTraceTree, formatDuration, type TraceTreeNode } from "@/components/RunTrace";
import type { WorkflowDAGLightweightNode } from "@/types/workflows";

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count, estimateSize }: { count: number; estimateSize: () => number }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, index) => ({
        index,
        key: `trace-${index}`,
        start: index * estimateSize(),
        size: estimateSize(),
      })),
    getTotalSize: () => count * estimateSize(),
  }),
}));

vi.mock("@/components/ui/status-pill", () => ({
  StatusDot: ({ status }: { status: string }) => <span>{`dot:${status}`}</span>,
}));

function createNode(overrides: Partial<WorkflowDAGLightweightNode> = {}): WorkflowDAGLightweightNode {
  return {
    execution_id: "exec-root",
    agent_node_id: "agent-root",
    reasoner_id: "root",
    status: "failed",
    started_at: "2026-04-08T10:00:00Z",
    completed_at: "2026-04-08T10:10:00Z",
    duration_ms: 600000,
    workflow_depth: 0,
    ...overrides,
  };
}

describe("RunTrace helpers", () => {
  it("builds a trace tree and attaches orphaned nodes under the root", () => {
    const timeline = [
      createNode(),
      createNode({
        execution_id: "exec-child",
        reasoner_id: "worker",
        parent_execution_id: "exec-root",
        workflow_depth: 1,
      }),
      createNode({
        execution_id: "exec-orphan",
        reasoner_id: "cleanup",
        parent_execution_id: "missing-parent",
        workflow_depth: 1,
      }),
    ];

    const tree = buildTraceTree(timeline);

    expect(tree?.execution_id).toBe("exec-root");
    expect(tree?.children.map((child) => child.execution_id)).toEqual(["exec-child", "exec-orphan"]);
  });

  it("formats durations across milliseconds, seconds, minutes, hours, and days", () => {
    expect(formatDuration(undefined)).toBe("—");
    expect(formatDuration(450)).toBe("450ms");
    expect(formatDuration(1500)).toBe("1.5s");
    expect(formatDuration(61000)).toBe("1m 1s");
    expect(formatDuration(3_780_000)).toBe("1h 3m");
    expect(formatDuration(93_600_000)).toBe("1d 2h");
  });
});

describe("RunTrace", () => {
  it("renders grouped rows, relative starts, and suppresses live child status when the root is terminal", () => {
    const onSelect = vi.fn();
    const tree: TraceTreeNode = {
      ...createNode(),
      children: [
        {
          ...createNode({
            execution_id: "exec-worker-1",
            reasoner_id: "worker",
            status: "running",
            started_at: "2026-04-08T10:00:01Z",
            duration_ms: 2000,
            parent_execution_id: "exec-root",
            workflow_depth: 1,
          }),
          children: [],
        },
        {
          ...createNode({
            execution_id: "exec-worker-2",
            reasoner_id: "worker",
            status: "running",
            started_at: "2026-04-08T10:00:02Z",
            duration_ms: 3000,
            parent_execution_id: "exec-root",
            workflow_depth: 1,
          }),
          children: [],
        },
        {
          ...createNode({
            execution_id: "exec-review",
            reasoner_id: "review",
            status: "cancelled",
            started_at: "2026-04-08T11:02:03Z",
            duration_ms: null,
            parent_execution_id: "exec-root",
            workflow_depth: 1,
          }),
          children: [],
        },
      ],
    };

    render(
      <RunTrace
        node={tree}
        maxDuration={600000}
        selectedId={null}
        onSelect={onSelect}
        runStartedAt="2026-04-08T10:00:00Z"
        rootStatus="failed"
      />
    );

    expect(screen.getByText("root")).toBeInTheDocument();
    expect(screen.getAllByText("worker")).toHaveLength(2);
    expect(screen.getByText("review")).toBeInTheDocument();
    expect(screen.getByText("×2")).toBeInTheDocument();
    expect(screen.getByText("+0:01")).toBeInTheDocument();
    expect(screen.getByText("+1:02:03")).toBeInTheDocument();
    expect(screen.getByText("10m 0s")).toBeInTheDocument();
    expect(screen.getByText("2.0s")).toBeInTheDocument();
    expect(screen.getByText("3.0s")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("dot:failed")).toBeInTheDocument();
    expect(screen.getAllByText("dot:running")).toHaveLength(2);
    expect(screen.getByText("dot:cancelled")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button")[1]);
    expect(onSelect).toHaveBeenCalledWith("exec-worker-1");
  });
});
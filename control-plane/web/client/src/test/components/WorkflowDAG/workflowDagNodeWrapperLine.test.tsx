// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  zoom: 1,
  getBezierPath: vi.fn(() => ["M 1 2 C 3 4 5 6 7 8"]),
  getEdgeParams: vi.fn(() => ({
    sx: 1,
    sy: 2,
    tx: 7,
    ty: 8,
    sourcePos: "left",
    targetPos: "right",
  })),
}));

vi.mock("@xyflow/react", () => ({
  Handle: ({ id }: { id?: string }) => <span data-testid={id ?? "handle"} />,
  Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
  useStore: (selector: (arg: { transform: [number, number, number] }) => unknown) =>
    selector({ transform: [0, 0, state.zoom] }),
  getBezierPath: (...args: unknown[]) => state.getBezierPath(...args),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <span className={className}>icon</span>;
  return {
    Calendar: Icon,
    CheckmarkFilled: Icon,
    Time: Icon,
    User: Icon,
  };
});

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/agentColorManager", () => ({
  agentColorManager: {
    getAgentColor: () => ({
      primary: "#112233",
      border: "#445566",
      text: "#ffffff",
    }),
  },
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status: string) => (status === "completed" ? "succeeded" : status),
  getStatusLabel: (status: string) => status.toUpperCase(),
  getStatusTheme: (status: string) => ({
    hexColor: status === "running" ? "#00ff00" : "#ffcc00",
    icon: ({ className }: { className?: string }) => <span className={className}>{status}-icon</span>,
    iconClass: `theme-${status}`,
    motion: status === "running" ? "live" : "still",
  }),
}));

vi.mock("@/components/WorkflowDAG/AgentBadge", () => ({
  AgentBadge: ({ agentName }: { agentName: string }) => <span>{agentName}</span>,
}));

vi.mock("@/components/WorkflowDAG/EdgeUtils", () => ({
  getEdgeParams: (...args: unknown[]) => state.getEdgeParams(...args),
}));

vi.mock("@/components/WorkflowDAG/NodeDetailSidebar", () => ({
  NodeDetailSidebar: ({
    node,
    isOpen,
    onClose,
  }: {
    node: { execution_id?: string; task_name?: string } | null;
    isOpen: boolean;
    onClose?: () => void;
  }) => (
    <div>
      <div>{`sidebar:${isOpen ? node?.execution_id ?? "open" : "closed"}`}</div>
      <button type="button" onClick={onClose}>close-sidebar</button>
    </div>
  ),
}));

function makeNodeData(overrides: Record<string, unknown> = {}) {
  return {
    workflow_id: "wf-1",
    execution_id: "exec-1",
    agent_node_id: "agent-1",
    reasoner_id: "process_payment",
    status: "running",
    started_at: "2026-04-09T10:00:00Z",
    duration_ms: 900,
    workflow_depth: 1,
    task_name: "process_payment",
    agent_name: "agent_alpha",
    ...overrides,
  };
}

describe("WorkflowNode, wrapper, and floating connection line", () => {
  beforeEach(() => {
    state.zoom = 1;
    state.getBezierPath.mockClear();
    state.getEdgeParams.mockClear();
    vi.useRealTimers();
  });

  it("renders simplified zoomed-out node content", async () => {
    state.zoom = 0.2;
    const { WorkflowNode } = await import("@/components/WorkflowDAG/WorkflowNode");

    render(<WorkflowNode data={makeNodeData()} />);

    expect(screen.getByTestId("target-left")).toBeInTheDocument();
    expect(screen.getByTitle("RUNNING - process_payment")).toBeInTheDocument();
    expect(screen.queryByText("agent_alpha")).not.toBeInTheDocument();
  });

  it("renders full standard, performance, and debug node views", async () => {
    const { WorkflowNode } = await import("@/components/WorkflowDAG/WorkflowNode");
    const { rerender } = render(
      <WorkflowNode
        data={makeNodeData({
          selected: true,
          isSearchMatch: true,
          isFocusRelated: true,
          duration_ms: 65000,
        })}
        selected
      />,
    );

    expect(screen.getAllByText("Process Payment").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Agent Alpha").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1m 5s").length).toBeGreaterThan(0);
    expect(screen.getAllByText("10:00:00").length).toBeGreaterThan(0);

    rerender(
      <WorkflowNode
        data={makeNodeData({
          viewMode: "performance",
          performanceIntensity: 0.72,
          duration_ms: 4200,
        })}
      />,
    );

    expect(screen.getByText("Load 72%")).toBeInTheDocument();
    expect(screen.getAllByText("4.2s").length).toBeGreaterThan(0);

    rerender(
      <WorkflowNode
        data={makeNodeData({
          status: "cancelled",
          viewMode: "debug",
          parent_execution_id: "parent-123456789",
        })}
      />,
    );

    expect(screen.getByText("ID: exec-1")).toBeInTheDocument();
    expect(screen.getByText("Parent: parent-1…")).toBeInTheDocument();
    expect(screen.getAllByText("CANCELLED").length).toBeGreaterThan(0);
  });

  it("renders and closes the sidebar wrapper and example usage", async () => {
    vi.useFakeTimers();
    const { WorkflowDAGWithSidebar, ExampleWorkflowDAGUsage } = await import("@/components/WorkflowDAG/WorkflowDAGWithSidebar");

    const nodes = [
      makeNodeData(),
      makeNodeData({
        execution_id: "exec-2",
        reasoner_id: "quality_check",
        task_name: "quality_check",
        status: "failed",
      }),
    ];

    const { rerender } = render(<WorkflowDAGWithSidebar nodes={nodes as never} edges={[]} />);

    fireEvent.click(screen.getByRole("button", { name: /process_payment/i }));
    expect(screen.getByText("sidebar:exec-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "close-sidebar" }));
    expect(screen.getByText("sidebar:closed")).toBeInTheDocument();

    vi.advanceTimersByTime(301);
    rerender(<WorkflowDAGWithSidebar nodes={nodes as never} edges={[]} />);
    expect(screen.getByText("Workflow DAG")).toBeInTheDocument();
    expect(screen.getByText(/Status: failed/)).toBeInTheDocument();

    render(<ExampleWorkflowDAGUsage />);
    expect(screen.getByText("Analyze Customer Sentiment")).toBeInTheDocument();
    expect(screen.getByText("Quality Check")).toBeInTheDocument();
  });

  it("renders a floating connection line and returns null without a source node", async () => {
    const { default: FloatingConnectionLine } = await import("@/components/WorkflowDAG/FloatingConnectionLine");

    const { container, rerender } = render(
      <svg>
        <FloatingConnectionLine
          toX={50}
          toY={60}
          fromPosition="left"
          toPosition="right"
          fromNode={{
            id: "source",
            measured: { width: 100, height: 50 },
            internals: { positionAbsolute: { x: 0, y: 0 } },
          }}
        />
      </svg>,
    );

    expect(state.getEdgeParams).toHaveBeenCalled();
    expect(state.getBezierPath).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceX: 1,
        sourceY: 2,
        targetX: 7,
        targetY: 8,
      }),
    );
    expect(container.querySelector("path")?.getAttribute("d")).toBe("M 1 2 C 3 4 5 6 7 8");
    expect(container.querySelector("circle")?.getAttribute("cx")).toBe("7");

    rerender(
      <svg>
        <FloatingConnectionLine
          toX={50}
          toY={60}
          fromPosition="left"
          toPosition="right"
          fromNode={null as never}
        />
      </svg>,
    );
    expect(container.querySelector("path")).toBeNull();
  });
});

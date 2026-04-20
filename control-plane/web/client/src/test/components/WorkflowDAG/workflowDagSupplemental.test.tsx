// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  fitView: vi.fn(),
  setViewport: vi.fn(),
  fitBounds: vi.fn(),
  zoomIn: vi.fn(),
  zoomOut: vi.fn(),
  getNodes: vi.fn(() => [
    { id: "exec-1", position: { x: 10, y: 20 }, width: 120, height: 60 },
    { id: "exec-2", position: { x: 220, y: 120 }, width: 120, height: 60 },
  ]),
  deckFitToContent: vi.fn(),
  deckZoomIn: vi.fn(),
  deckZoomOut: vi.fn(),
  applyLayout: vi.fn(async (nodes, edges) => ({ nodes, edges })),
  getWorkflowDAG: vi.fn(),
  lastSidebarProps: null as null | { node: { execution_id?: string } | null; isOpen: boolean },
}));

vi.mock("dagre", () => ({
  graphlib: {
    Graph: class Graph {
      setGraph() {}
      setDefaultEdgeLabel() {}
      setNode() {}
      setEdge() {}
      node() {
        return { x: 0, y: 0 };
      }
    },
  },
  layout: vi.fn(),
}));

vi.mock("elkjs/lib/elk.bundled.js", () => ({
  default: class MockELK {
    layout = vi.fn();
  },
}));

vi.mock("@deck.gl/react", () => ({
  DeckGL: ({ children }: React.PropsWithChildren) => <div data-testid="deck-gl">{children}</div>,
}));

vi.mock("@deck.gl/layers", () => ({
  ScatterplotLayer: class ScatterplotLayer {},
  LineLayer: class LineLayer {},
  TextLayer: class TextLayer {},
}));

vi.mock("@xyflow/react", () => ({
  Background: () => <div data-testid="background" />,
  BackgroundVariant: { Dots: "dots" },
  ConnectionMode: { Strict: "strict" },
  MarkerType: { Arrow: "arrow" },
  Panel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  ReactFlow: ({
    children,
    nodes,
    edges,
    onNodeClick,
    onMoveEnd,
  }: React.PropsWithChildren<{
    nodes?: Array<{ id: string; data?: unknown }>;
    edges?: Array<{ id: string }>;
    onNodeClick?: (event: React.MouseEvent, node: { id: string; data?: unknown }) => void;
    onMoveEnd?: (event: unknown, viewport: { x: number; y: number; zoom: number }) => void;
  }>) => (
    <div data-testid="react-flow">
      <div>{`nodes:${nodes?.length ?? 0}`}</div>
      <div>{`edges:${edges?.length ?? 0}`}</div>
      <button type="button" onClick={() => onNodeClick?.({} as React.MouseEvent, nodes?.[0] as never)}>
        click-node
      </button>
      <button type="button" onClick={() => onMoveEnd?.(null, { x: 7, y: 8, zoom: 0.9 })}>
        move-end
      </button>
      {children}
    </div>
  ),
  ReactFlowProvider: ({ children }: React.PropsWithChildren) => <div data-testid="provider">{children}</div>,
  useEdgesState: (initial: unknown[]) => {
    const [value, setValue] = React.useState(initial);
    return [value, setValue, vi.fn()] as const;
  },
  useNodesState: (initial: unknown[]) => {
    const [value, setValue] = React.useState(initial);
    return [value, setValue, vi.fn()] as const;
  },
  useReactFlow: () => ({
    fitView: state.fitView,
    setViewport: state.setViewport,
    getNodes: state.getNodes,
    fitBounds: state.fitBounds,
    zoomIn: state.zoomIn,
    zoomOut: state.zoomOut,
  }),
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  X: ({ className }: { className?: string }) => <span className={className}>x</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/numberFormat", () => ({
  formatNumberWithCommas: (value: number) => value.toLocaleString("en-US"),
}));

vi.mock("@/components/WorkflowDAG/AgentLegend", () => ({
  AgentLegend: ({
    selectedAgent,
    onAgentFilter,
    onExpandGraph,
    onFitView,
    onZoomIn,
    onZoomOut,
    layout,
  }: {
    selectedAgent?: string | null;
    onAgentFilter?: (agent: string | null) => void;
    onExpandGraph?: () => void;
    onFitView?: () => void;
    onZoomIn?: () => void;
    onZoomOut?: () => void;
    layout?: string;
  }) => (
    <div>
      <div>{`legend-layout:${layout}`}</div>
      <div>{`legend-selected:${selectedAgent ?? "none"}`}</div>
      <button type="button" onClick={() => onAgentFilter?.("Agent Two")}>legend-filter</button>
      <button type="button" onClick={() => onAgentFilter?.(null)}>legend-clear</button>
      <button type="button" onClick={onExpandGraph}>legend-expand</button>
      <button type="button" onClick={onFitView}>legend-fit</button>
      <button type="button" onClick={onZoomIn}>legend-zoom-in</button>
      <button type="button" onClick={onZoomOut}>legend-zoom-out</button>
    </div>
  ),
}));

vi.mock("@/components/WorkflowDAG/WorkflowGraphControls", () => ({
  WorkflowGraphControls: ({ show }: { show: boolean }) => <div>{`graph-controls:${String(show)}`}</div>,
}));

vi.mock("@/components/WorkflowDAG/FloatingConnectionLine", () => ({
  default: () => <div>floating-connection-line</div>,
}));

vi.mock("@/components/WorkflowDAG/FloatingEdge", () => ({
  default: () => <div>floating-edge</div>,
}));

vi.mock("@/components/WorkflowDAG/WorkflowNode", () => ({
  WorkflowNode: () => <div>workflow-node</div>,
}));

vi.mock("@/components/WorkflowDAG/NodeDetailSidebar", () => ({
  NodeDetailSidebar: ({
    node,
    isOpen,
    onClose,
  }: {
    node: { execution_id?: string } | null;
    isOpen: boolean;
    onClose?: () => void;
  }) => {
    state.lastSidebarProps = { node, isOpen };
    return (
      <div>
        <div>{`sidebar:${isOpen ? node?.execution_id ?? "open" : "closed"}`}</div>
        <button type="button" onClick={onClose}>sidebar-close</button>
      </div>
    );
  },
}));

vi.mock("@/components/WorkflowDAG/VirtualizedDAG", () => ({
  VirtualizedDAG: ({ nodes }: { nodes: unknown[] }) => <div>{`virtualized:${nodes.length}`}</div>,
}));

vi.mock("@/components/WorkflowDAG/DeckGLView", () => ({
  WorkflowDeckGLView: React.forwardRef(
    (
      {
        nodes,
        onNodeClick,
      }: {
        nodes: Array<{ execution_id?: string; workflow_id?: string }>;
        onNodeClick?: (node: { execution_id?: string; workflow_id?: string }) => void;
      },
      ref,
    ) => {
      React.useImperativeHandle(ref, () => ({
        fitToContent: state.deckFitToContent,
        zoomIn: state.deckZoomIn,
        zoomOut: state.deckZoomOut,
      }));

      return (
        <div>
          <div>{`deck-view:${nodes.length}`}</div>
          <button type="button" onClick={() => onNodeClick?.(nodes[0] ?? {})}>
            deck-node-click
          </button>
        </div>
      );
    },
  ),
  WorkflowDeckGraphControls: () => <div>deck-controls</div>,
}));

vi.mock("@/components/WorkflowDAG/DeckGLGraph", () => ({
  buildDeckGraph: (timeline: unknown[]) => ({ nodes: timeline, edges: [] }),
}));

vi.mock("@/components/WorkflowDAG/layouts/LayoutManager", () => ({
  LayoutManager: class LayoutManager {
    applyLayout = state.applyLayout;
    getAvailableLayouts() {
      return ["tree", "dagre"];
    }
    isSlowLayout(layout: string) {
      return layout === "dagre";
    }
    isLargeGraph(count: number) {
      return count >= 2000;
    }
    getDefaultLayout() {
      return "tree";
    }
  },
}));

vi.mock("@/components/WorkflowDAG/workflowDagUtils", () => ({
  LARGE_GRAPH_LAYOUT_THRESHOLD: 2000,
  PERFORMANCE_THRESHOLD: 300,
  isLightweightDAGResponse: (value: { lightweight?: boolean }) => Boolean(value?.lightweight),
  adaptLightweightResponse: (value: { workflow_name?: string; nodes?: unknown[]; total_nodes?: number; displayed_nodes?: number }) => ({
    workflow_name: value.workflow_name ?? "Lightweight Workflow",
    timeline: value.nodes ?? [],
    total_nodes: value.total_nodes,
    displayed_nodes: value.displayed_nodes,
  }),
  applySimpleGridLayout: (nodes: unknown[]) => nodes,
  decorateEdgesWithStatus: (edges: unknown[]) => edges,
  decorateNodesWithViewMode: (nodes: unknown[]) => nodes,
}));

vi.mock("@/services/workflowsApi", () => ({
  getWorkflowDAG: (...args: unknown[]) => state.getWorkflowDAG(...args),
}));

describe("WorkflowDAGViewer supplemental coverage", () => {
  beforeEach(() => {
    state.fitView.mockReset();
    state.setViewport.mockReset();
    state.fitBounds.mockReset();
    state.zoomIn.mockReset();
    state.zoomOut.mockReset();
    state.deckFitToContent.mockReset();
    state.deckZoomIn.mockReset();
    state.deckZoomOut.mockReset();
    state.applyLayout.mockReset();
    state.applyLayout.mockImplementation(async (nodes, edges) => ({ nodes, edges }));
    state.getWorkflowDAG.mockReset();
    state.lastSidebarProps = null;
    localStorage.clear();
    document.body.style.overflow = "";
    vi.stubGlobal(
      "ResizeObserver",
      class ResizeObserver {
        observe() {}
        disconnect() {}
      },
    );
    vi.useRealTimers();
  });

  it("renders loading and error states from props", async () => {
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");
    const { rerender } = render(<WorkflowDAGViewer workflowId="wf-loading" loading />);

    expect(screen.getByText("Loading workflow DAG...")).toBeInTheDocument();

    rerender(<WorkflowDAGViewer workflowId="wf-error" error="network down" />);
    expect(screen.getByText("Failed to load workflow DAG")).toBeInTheDocument();
    expect(screen.getByText("network down")).toBeInTheDocument();
  });

  it("renders dag data, exposes controls, persists viewport, filters nodes, and handles fullscreen", async () => {
    const onReady = vi.fn();
    const onSearchResultsChange = vi.fn();
    const onLayoutInfoChange = vi.fn();
    const onExecutionClick = vi.fn();
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");

    render(
      <WorkflowDAGViewer
        workflowId="wf-1"
        dagData={{
          workflow_name: "Alpha Workflow",
          timeline: [
            {
              workflow_id: "wf-1",
              execution_id: "exec-1",
              agent_node_id: "agent-1",
              reasoner_id: "task_alpha",
              status: "running",
              started_at: "2026-04-09T10:00:00Z",
              workflow_depth: 1,
              agent_name: "Agent One",
              task_name: "task_alpha",
            },
            {
              workflow_id: "wf-1",
              execution_id: "exec-2",
              agent_node_id: "agent-2",
              reasoner_id: "task_beta",
              status: "completed",
              started_at: "2026-04-09T10:01:00Z",
              workflow_depth: 2,
              parent_execution_id: "exec-1",
              agent_name: "Agent Two",
              task_name: "task_beta",
            },
          ],
        }}
        searchQuery="alpha"
        selectedNodeIds={["exec-1"]}
        focusMode
        focusedNodeIds={["exec-1"]}
        viewMode="debug"
        onReady={onReady}
        onSearchResultsChange={onSearchResultsChange}
        onLayoutInfoChange={onLayoutInfoChange}
        onExecutionClick={onExecutionClick}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("nodes:2")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(onSearchResultsChange).toHaveBeenCalledWith({
        totalMatches: 1,
        firstMatchId: "exec-1",
      });
    }, { timeout: 1500 });

    expect(onReady).toHaveBeenCalledTimes(1);
    expect(onLayoutInfoChange).toHaveBeenCalledWith(
      expect.objectContaining({
        currentLayout: "tree",
        availableLayouts: ["tree", "dagre"],
        isLargeGraph: false,
      }),
    );

    const controls = onReady.mock.calls[0][0];
    controls.fitToView({ padding: 0.5 });
    controls.focusOnNodes(["exec-1"], { padding: 0.3 });
    await controls.changeLayout("dagre");

    expect(state.fitView).toHaveBeenCalledWith({
      padding: 0.5,
      includeHiddenNodes: false,
    });
    expect(state.fitBounds).toHaveBeenCalledWith(
      { x: 10, y: 20, width: 120, height: 60 },
      { padding: 0.3 },
    );
    expect(state.applyLayout).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "legend-fit" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-in" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-out" }));
    expect(state.fitView).toHaveBeenCalledWith({
      padding: 0.2,
      includeHiddenNodes: false,
      duration: 220,
    });
    expect(state.zoomIn).toHaveBeenCalledWith({ duration: 200 });
    expect(state.zoomOut).toHaveBeenCalledWith({ duration: 200 });

    fireEvent.click(screen.getByRole("button", { name: "legend-filter" }));
    expect(screen.getByText("legend-selected:Agent Two")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "move-end" }));
    expect(localStorage.getItem("workflowDAGViewport:wf-1")).toBe(JSON.stringify({ x: 7, y: 8, zoom: 0.9 }));

    fireEvent.click(screen.getByRole("button", { name: "click-node" }));
    expect(onExecutionClick).toHaveBeenCalledWith(expect.objectContaining({ execution_id: "exec-1" }));

    fireEvent.click(screen.getByRole("button", { name: "legend-expand" }));
    expect(screen.getByText("Workflow graph")).toBeInTheDocument();
    expect(screen.getByText("Alpha Workflow")).toBeInTheDocument();
    expect(document.body.style.overflow).toBe("hidden");
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByText("Workflow graph")).not.toBeInTheDocument();
    });
  });

  it("uses internal fallback fetching when dagData is omitted and opens the sidebar without an execution handler", async () => {
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");
    state.getWorkflowDAG.mockResolvedValue({
      lightweight: true,
      workflow_name: "Fetched Workflow",
      nodes: [
        {
          workflow_id: "wf-fetched",
          execution_id: "exec-fetched",
          agent_node_id: "agent-fetched",
          reasoner_id: "task_fetch",
          status: "succeeded",
          started_at: "2026-04-09T10:00:00Z",
          workflow_depth: 1,
          task_name: "task_fetch",
          agent_name: "Fetch Agent",
        },
      ],
    });

    render(<WorkflowDAGViewer workflowId="wf-fetched" />);

    await waitFor(() => {
      expect(state.getWorkflowDAG).toHaveBeenCalledWith("wf-fetched", { lightweight: true });
    });
    await waitFor(() => {
      expect(screen.getByText("nodes:1")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "click-node" }));
    expect(screen.getByText("sidebar:exec-fetched")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "sidebar-close" }));
    expect(screen.getByText("sidebar:closed")).toBeInTheDocument();
  });

  it("renders the large-graph DeckGL path, exposes deck controls, and opens the sidebar from deck nodes", async () => {
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");
    const timeline = Array.from({ length: 2001 }, (_, index) => ({
      workflow_id: "wf-large",
      execution_id: `exec-${index + 1}`,
      agent_node_id: `agent-${index + 1}`,
      reasoner_id: `task_${index + 1}`,
      status: index === 0 ? "running" : "completed",
      started_at: `2026-04-09T10:${String(index % 60).padStart(2, "0")}:00Z`,
      workflow_depth: 1,
      task_name: `task_${index + 1}`,
      agent_name: `Agent ${index + 1}`,
    }));

    render(
      <WorkflowDAGViewer
        workflowId="wf-large"
        dagData={{
          workflow_name: "Large Workflow",
          timeline,
          total_nodes: 2500,
          displayed_nodes: 2001,
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("deck-view:2001")).toBeInTheDocument();
    });
    expect(screen.getByText("Large graph")).toBeInTheDocument();
    expect(screen.getByText("(2,001 shown / 2,500 total)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "legend-fit" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-in" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-out" }));
    expect(state.deckFitToContent).toHaveBeenCalled();
    expect(state.deckZoomIn).toHaveBeenCalled();
    expect(state.deckZoomOut).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "legend-expand" }));
    expect(screen.getByText("deck-controls")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "deck-node-click" }));
    expect(screen.getByText("sidebar:exec-1")).toBeInTheDocument();
  });

  it("renders the virtualized branch and warns when viewport persistence fails", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota");
    });
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");
    const virtualizedTimeline = Array.from({ length: 301 }, (_, index) => ({
      workflow_id: "wf-virtualized",
      execution_id: `exec-v-${index + 1}`,
      agent_node_id: `agent-v-${index + 1}`,
      reasoner_id: `task_v_${index + 1}`,
      status: "completed",
      started_at: "2026-04-09T10:00:00Z",
      workflow_depth: 1,
      task_name: `task_v_${index + 1}`,
      agent_name: `Agent V ${index + 1}`,
    }));

    const { rerender } = render(
      <WorkflowDAGViewer
        workflowId="wf-virtualized"
        dagData={{
          workflow_name: "Virtualized Workflow",
          timeline: virtualizedTimeline,
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("virtualized:301")).toBeInTheDocument();
    });

    rerender(
      <WorkflowDAGViewer
        workflowId="wf-small"
        dagData={{
          workflow_name: "Small Workflow",
          timeline: [
            {
              workflow_id: "wf-small",
              execution_id: "exec-small",
              agent_node_id: "agent-small",
              reasoner_id: "task_small",
              status: "running",
              started_at: "2026-04-09T10:00:00Z",
              workflow_depth: 1,
              task_name: "task_small",
              agent_name: "Agent Small",
            },
          ],
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("nodes:1")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "move-end" }));
    expect(warnSpy).toHaveBeenCalledWith(
      "Failed to persist workflow DAG viewport",
      expect.any(Error),
    );

    warnSpy.mockRestore();
    setItemSpy.mockRestore();
  });

  it("clears the pending sidebar-close timer when the sidebar is reopened and reclosed, and lets the timer fire safely", async () => {
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");
    state.getWorkflowDAG.mockResolvedValue({
      lightweight: true,
      workflow_name: "Reopen Workflow",
      nodes: [
        {
          workflow_id: "wf-reopen",
          execution_id: "exec-reopen",
          agent_node_id: "agent-reopen",
          reasoner_id: "task_reopen",
          status: "succeeded",
          started_at: "2026-04-09T10:00:00Z",
          workflow_depth: 1,
          task_name: "task_reopen",
          agent_name: "Reopen Agent",
        },
      ],
    });

    const clearSpy = vi.spyOn(globalThis, "clearTimeout");
    render(<WorkflowDAGViewer workflowId="wf-reopen" />);

    await waitFor(() => {
      expect(screen.getByText("nodes:1")).toBeInTheDocument();
    });

    // Open then close — schedules the deferred clear, no prior handle to cancel.
    fireEvent.click(screen.getByRole("button", { name: "click-node" }));
    expect(screen.getByText("sidebar:exec-reopen")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "sidebar-close" }));

    // Re-open and close again before the deferred clear fires —
    // exercises the clearTimeout branch on the prior pending handle.
    clearSpy.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "click-node" }));
    expect(screen.getByText("sidebar:exec-reopen")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "sidebar-close" }));
    expect(clearSpy).toHaveBeenCalled();

    // Let the deferred clear actually run so the setTimeout body executes.
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(screen.getByText("sidebar:closed")).toBeInTheDocument();

    clearSpy.mockRestore();
  });
});

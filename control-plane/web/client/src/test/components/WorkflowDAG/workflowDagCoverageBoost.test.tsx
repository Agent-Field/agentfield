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
  applyLayout: vi.fn(async (nodes, edges) => ({ nodes, edges })),
  getWorkflowDAG: vi.fn(),
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
  DeckGL: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
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
    onNodeClick,
  }: React.PropsWithChildren<{
    nodes?: Array<{ id: string; data?: unknown }>;
    onNodeClick?: (event: React.MouseEvent, node: { id: string; data?: unknown }) => void;
  }>) => (
    <div data-testid="react-flow">
      <div>{`nodes:${nodes?.length ?? 0}`}</div>
      <button type="button" onClick={() => onNodeClick?.({} as React.MouseEvent, nodes?.[0] as never)}>
        click-node
      </button>
      {children}
    </div>
  ),
  ReactFlowProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
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
    onExpandGraph,
    onFitView,
    onZoomIn,
    onZoomOut,
  }: {
    onExpandGraph?: () => void;
    onFitView?: () => void;
    onZoomIn?: () => void;
    onZoomOut?: () => void;
  }) => (
    <div>
      <button type="button" onClick={onExpandGraph}>legend-expand</button>
      <button type="button" onClick={onFitView}>legend-fit</button>
      <button type="button" onClick={onZoomIn}>legend-zoom-in</button>
      <button type="button" onClick={onZoomOut}>legend-zoom-out</button>
    </div>
  ),
}));

vi.mock("@/components/WorkflowDAG/WorkflowGraphControls", () => ({
  WorkflowGraphControls: () => <div>graph-controls</div>,
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
  }) => (
    <div>
      <div>{`sidebar:${isOpen ? node?.execution_id ?? "open" : "closed"}`}</div>
      <button type="button" onClick={onClose}>sidebar-close</button>
    </div>
  ),
}));

vi.mock("@/components/WorkflowDAG/VirtualizedDAG", () => ({
  VirtualizedDAG: ({ nodes }: { nodes: unknown[] }) => <div>{`virtualized:${nodes.length}`}</div>,
}));

vi.mock("@/components/WorkflowDAG/DeckGLView", () => ({
  WorkflowDeckGLView: React.forwardRef((_props, _ref) => <div>deck-view</div>),
  WorkflowDeckGraphControls: () => <div>deck-controls</div>,
}));

vi.mock("@/components/WorkflowDAG/DeckGLGraph", () => ({
  buildDeckGraph: (timeline: Array<{ execution_id: string }>) => ({
    nodes: timeline,
    edges: [],
  }),
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
  adaptLightweightResponse: (value: { workflow_name?: string; nodes?: unknown[] }) => ({
    workflow_name: value.workflow_name ?? "Lightweight Workflow",
    timeline: value.nodes ?? [],
  }),
  applySimpleGridLayout: (nodes: unknown[]) => nodes,
  decorateEdgesWithStatus: (edges: unknown[]) => edges,
  decorateNodesWithViewMode: (nodes: unknown[]) => nodes,
}));

vi.mock("@/services/workflowsApi", () => ({
  getWorkflowDAG: (...args: unknown[]) => state.getWorkflowDAG(...args),
}));

describe("WorkflowDAGViewer coverage boost", () => {
  beforeEach(() => {
    state.fitView.mockReset();
    state.setViewport.mockReset();
    state.fitBounds.mockReset();
    state.zoomIn.mockReset();
    state.zoomOut.mockReset();
    state.applyLayout.mockReset();
    state.applyLayout.mockImplementation(async (nodes, edges) => ({ nodes, edges }));
    state.getWorkflowDAG.mockReset();
    document.body.style.overflow = "";
    localStorage.clear();
    vi.stubGlobal(
      "ResizeObserver",
      class ResizeObserver {
        observe() {}
        disconnect() {}
      },
    );
  });

  it("reports zero search matches, falls back to fitView for empty focus, and closes fullscreen from the header button", async () => {
    const onReady = vi.fn();
    const onSearchResultsChange = vi.fn();
    const { WorkflowDAGViewer } = await import("@/components/WorkflowDAG");

    render(
      <WorkflowDAGViewer
        workflowId="wf-coverage"
        dagData={{
          workflow_name: "Coverage Workflow",
          timeline: [
            {
              workflow_id: "wf-coverage",
              execution_id: "exec-1",
              agent_node_id: "agent-1",
              reasoner_id: "task_alpha",
              status: "running",
              started_at: "2026-04-09T10:00:00Z",
              workflow_depth: 1,
              agent_name: "Agent One",
              task_name: "task_alpha",
            },
          ],
        }}
        searchQuery="missing-term"
        onReady={onReady}
        onSearchResultsChange={onSearchResultsChange}
      />,
    );

    await waitFor(() => {
      expect(onSearchResultsChange).toHaveBeenCalledWith({
        totalMatches: 0,
        firstMatchId: undefined,
      });
    }, { timeout: 1500 });

    const controls = onReady.mock.calls[0][0];
    state.applyLayout.mockClear();
    controls.focusOnNodes([]);
    controls.focusOnNodes(["does-not-exist"]);
    await controls.changeLayout("tree");

    expect(state.fitView).toHaveBeenCalledWith({
      padding: 0.2,
      includeHiddenNodes: false,
    });
    expect(state.fitBounds).not.toHaveBeenCalled();
    expect(state.applyLayout).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "legend-expand" }));
    expect(screen.getByText("Workflow graph")).toBeInTheDocument();
    expect(document.body.style.overflow).toBe("hidden");

    fireEvent.click(screen.getByRole("button", { name: "Exit full screen" }));
    await waitFor(() => {
      expect(screen.queryByText("Workflow graph")).not.toBeInTheDocument();
    });
  });
});

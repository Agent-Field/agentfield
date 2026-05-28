// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const layoutMock = vi.hoisted(() => vi.fn());
const flowState = vi.hoisted(() => ({
  fitView: vi.fn(),
  setViewport: vi.fn(),
  zoomIn: vi.fn(),
  zoomOut: vi.fn(),
}));

vi.mock("elkjs/lib/elk.bundled.js", () => ({
  default: class MockELK {
    layout = layoutMock;
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
  Background: ({ color }: { color?: string }) => <div>{`background:${color}`}</div>,
  BackgroundVariant: { Dots: "dots" },
  ConnectionMode: { Strict: "strict" },
  Panel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  ReactFlow: ({
    children,
    nodes,
    edges,
    onNodeClick,
    onMoveEnd,
    className,
    defaultViewport,
  }: React.PropsWithChildren<{
    nodes: Array<{ id: string }>;
    edges: Array<{ id: string }>;
    onNodeClick?: (event: React.MouseEvent, node: { id: string }) => void;
    onMoveEnd?: (event: unknown, viewport: { x: number; y: number; zoom: number }) => void;
    className?: string;
    defaultViewport?: { x: number; y: number; zoom: number };
  }>) => (
    <div data-testid="react-flow" data-class-name={className}>
      <div>{`nodes:${nodes.length}`}</div>
      <div>{`edges:${edges.length}`}</div>
      <div>{`default-zoom:${defaultViewport?.zoom}`}</div>
      <button type="button" onClick={() => onNodeClick?.({} as React.MouseEvent, nodes[0])}>
        trigger-node-click
      </button>
      <button type="button" onClick={() => onMoveEnd?.(null, { x: 11, y: 22, zoom: 0.9 })}>
        trigger-move-end
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
  useReactFlow: () => flowState,
}));

vi.mock("@/components/WorkflowDAG/AgentLegend", () => ({
  AgentLegend: ({
    compact,
    selectedAgent,
    onFitView,
    onZoomIn,
    onZoomOut,
    onAgentFilter,
    onExpandGraph,
  }: {
    compact: boolean;
    selectedAgent: string | null;
    onFitView: () => void;
    onZoomIn: () => void;
    onZoomOut: () => void;
    onAgentFilter: (agent: string | null) => void;
    onExpandGraph?: () => void;
  }) => (
    <div>
      <div>{`legend-compact:${String(compact)}`}</div>
      <div>{`legend-selected:${selectedAgent ?? "none"}`}</div>
      <button type="button" onClick={onFitView}>legend-fit</button>
      <button type="button" onClick={onZoomIn}>legend-zoom-in</button>
      <button type="button" onClick={onZoomOut}>legend-zoom-out</button>
      <button type="button" onClick={() => onAgentFilter("Agent Z")}>legend-filter</button>
      <button type="button" onClick={onExpandGraph}>legend-expand</button>
    </div>
  ),
}));

vi.mock("@/components/WorkflowDAG/WorkflowGraphControls", () => ({
  WorkflowGraphControls: ({ show }: { show: boolean }) => <div>{`graph-controls:${String(show)}`}</div>,
}));

vi.mock("@/components/WorkflowDAG/FloatingConnectionLine", () => ({
  default: () => <div>floating-connection-line</div>,
}));

describe("VirtualizedDAG", () => {
  const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
  const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;

  beforeEach(() => {
    flowState.fitView.mockReset();
    flowState.setViewport.mockReset();
    flowState.zoomIn.mockReset();
    flowState.zoomOut.mockReset();
    localStorage.clear();
    vi.restoreAllMocks();

    vi.stubGlobal("requestAnimationFrame", vi.fn((cb: FrameRequestCallback) => {
      cb(0);
      return 1;
    }));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterEach(() => {
    globalThis.requestAnimationFrame = originalRequestAnimationFrame;
    globalThis.cancelAnimationFrame = originalCancelAnimationFrame;
  });

  it("restores a saved viewport, forwards handlers, and persists move-end updates", async () => {
    localStorage.setItem("workflowDAGViewport:wf-1", JSON.stringify({ x: 5, y: 6, zoom: 0.7 }));
    const onNodeClick = vi.fn();
    const onAgentFilter = vi.fn();
    const onExpandGraph = vi.fn();
    const { VirtualizedDAG } = await import("@/components/WorkflowDAG/VirtualizedDAG");

    render(
      <VirtualizedDAG
        nodes={[{ id: "n1", position: { x: 0, y: 0 }, data: {} } as never]}
        edges={[{ id: "e1", source: "n1", target: "n1" } as never]}
        onNodeClick={onNodeClick}
        nodeTypes={{}}
        edgeTypes={{}}
        className="dag-root"
        workflowId="wf-1"
        onAgentFilter={onAgentFilter}
        selectedAgent="Agent A"
        onExpandGraph={onExpandGraph}
        graphLayout="fullscreen"
      />,
    );

    await waitFor(() => {
      expect(flowState.setViewport).toHaveBeenCalledWith({ x: 5, y: 6, zoom: 0.7 });
    });
    expect(flowState.fitView).not.toHaveBeenCalled();
    expect(screen.getByText("legend-compact:true")).toBeInTheDocument();
    expect(screen.getByText("legend-selected:Agent A")).toBeInTheDocument();
    expect(screen.getByText("graph-controls:true")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "trigger-node-click" }));
    expect(onNodeClick).toHaveBeenCalledWith(expect.any(Object), expect.objectContaining({ id: "n1" }));

    fireEvent.click(screen.getByRole("button", { name: "legend-fit" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-in" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-zoom-out" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-filter" }));
    fireEvent.click(screen.getByRole("button", { name: "legend-expand" }));
    fireEvent.click(screen.getByRole("button", { name: "trigger-move-end" }));

    expect(flowState.fitView).toHaveBeenCalledWith({
      padding: 0.2,
      includeHiddenNodes: false,
      duration: 220,
    });
    expect(flowState.zoomIn).toHaveBeenCalledWith({ duration: 200 });
    expect(flowState.zoomOut).toHaveBeenCalledWith({ duration: 200 });
    expect(onAgentFilter).toHaveBeenCalledWith("Agent Z");
    expect(onExpandGraph).toHaveBeenCalled();
    expect(localStorage.getItem("workflowDAGViewport:wf-1")).toBe(JSON.stringify({ x: 11, y: 22, zoom: 0.9 }));
  });

  it("falls back to fitView when the saved viewport is invalid and wraps provider export", async () => {
    localStorage.setItem("workflowDAGViewport:wf-2", JSON.stringify({ x: "bad" }));
    const { VirtualizedDAG, VirtualizedDAGWithProvider } = await import("@/components/WorkflowDAG/VirtualizedDAG");

    const { rerender } = render(
      <VirtualizedDAG
        nodes={[{ id: "n1", position: { x: 0, y: 0 }, data: {} } as never]}
        edges={[]}
        nodeTypes={{}}
        workflowId="wf-2"
        onAgentFilter={vi.fn()}
        selectedAgent={null}
        graphLayout="sidebar"
      />,
    );

    await waitFor(() => {
      expect(flowState.fitView).toHaveBeenCalledWith({ padding: 0.2, duration: 0 });
    });
    expect(screen.getByText("graph-controls:false")).toBeInTheDocument();

    rerender(
      <VirtualizedDAGWithProvider
        nodes={[{ id: "n1", position: { x: 0, y: 0 }, data: {} } as never]}
        edges={[]}
        nodeTypes={{}}
        workflowId="wf-3"
        onAgentFilter={vi.fn()}
        selectedAgent={null}
      />,
    );

    expect(screen.getByTestId("provider")).toBeInTheDocument();
  });
});

describe("ELKLayoutEngine", () => {
  beforeEach(() => {
    layoutMock.mockReset();
  });

  it("calculates node dimensions, applies ELK positions, and preserves edges", async () => {
    const { ELKLayoutEngine } = await import("@/components/WorkflowDAG/layouts/ELKLayoutEngine");

    layoutMock.mockResolvedValue({
      id: "root",
      children: [
        { id: "alpha", x: 40, y: 60 },
        { id: "beta", x: 180, y: 200 },
      ],
    });

    const engine = new ELKLayoutEngine();
    const nodes = [
      {
        id: "alpha",
        data: {
          task_name: "very_long_task_name_for_layout",
          agent_name: "agent_alpha",
        },
        position: { x: 0, y: 0 },
      },
      {
        id: "beta",
        width: 222,
        height: 144,
        data: {
          reasoner_id: "beta",
          agent_node_id: "node-beta",
        },
        position: { x: 10, y: 10 },
      },
    ] as any;
    const edges = [{ id: "edge-1", source: "alpha", target: "beta" }] as any;

    const result = await engine.applyLayout(nodes, edges, "box");

    expect(layoutMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "root",
        layoutOptions: expect.objectContaining({
          "elk.algorithm": "org.eclipse.elk.box",
        }),
        children: [
          expect.objectContaining({
            id: "alpha",
            width: expect.any(Number),
            height: 100,
          }),
          expect.objectContaining({
            id: "beta",
            width: 222,
            height: 144,
          }),
        ],
        edges: [{ id: "edge-1", sources: ["alpha"], targets: ["beta"] }],
      }),
    );
    expect(result).toEqual({
      nodes: [
        expect.objectContaining({ id: "alpha", position: { x: 40, y: 60 } }),
        expect.objectContaining({ id: "beta", position: { x: 180, y: 200 } }),
      ],
      edges,
    });

    expect((engine as any).calculateNodeWidth({ task_name: "short", agent_name: "tiny" })).toBeGreaterThanOrEqual(200);
    expect((engine as any).calculateNodeWidth({ task_name: "x".repeat(200), agent_name: "y" })).toBeLessThanOrEqual(360);
    expect((engine as any).calculateNodeHeight({})).toBe(100);
  });

  it("returns original graph on ELK failure and exposes static layout helpers", async () => {
    const error = new Error("layout failed");
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { ELKLayoutEngine, ELK_LAYOUTS } = await import("@/components/WorkflowDAG/layouts/ELKLayoutEngine");

    layoutMock.mockRejectedValue(error);

    const engine = new ELKLayoutEngine();
    const nodes = [{ id: "node-1", data: {}, position: { x: 1, y: 2 } }] as any;
    const edges = [{ id: "edge-1", source: "node-1", target: "node-1" }] as any;

    await expect(engine.applyLayout(nodes, edges, "layered")).resolves.toEqual({ nodes, edges });
    expect(consoleSpy).toHaveBeenCalledWith("ELK layout failed:", error);
    expect(ELKLayoutEngine.isSlowForLargeGraphs("layered")).toBe(true);
    expect(ELKLayoutEngine.isSlowForLargeGraphs("box")).toBe(false);
    expect(ELKLayoutEngine.getLayoutDescription("rectpacking")).toBe(ELK_LAYOUTS.rectpacking.description);
    expect(ELKLayoutEngine.getAvailableLayouts()).toEqual(["box", "rectpacking", "layered"]);
  });
});
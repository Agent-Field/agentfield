// @ts-nocheck
import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AgentLegend, AgentLegendMini } from "@/components/WorkflowDAG/AgentLegend";
import { GraphToolbar } from "@/components/WorkflowDAG/GraphToolbar";
import { WorkflowDAGViewer } from "@/components/WorkflowDAG";
import { WorkflowNode } from "@/components/WorkflowDAG/WorkflowNode";

let currentZoom = 1;
const fitView = vi.fn();
const setViewport = vi.fn();
const fitBounds = vi.fn();
const zoomIn = vi.fn();
const zoomOut = vi.fn();

vi.mock("@deck.gl/react", () => ({
  DeckGL: ({ children }: React.PropsWithChildren) => <div data-testid="deck-gl">{children}</div>,
}));

vi.mock("@deck.gl/layers", () => ({
  ScatterplotLayer: class ScatterplotLayer {},
  LineLayer: class LineLayer {},
  TextLayer: class TextLayer {},
}));

vi.mock("@xyflow/react", async () => {
  const ReactModule = await import("react");

  return {
    Background: () => <div data-testid="background" />,
    BackgroundVariant: { Dots: "dots" },
    ConnectionMode: { Strict: "strict" },
    Handle: ({ id }: { id?: string }) => <span data-testid={id ?? "handle"} />,
    MarkerType: { Arrow: "arrow" },
    Panel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    Position: { Left: "left", Right: "right" },
    ReactFlow: ({
      children,
      nodes,
      onMoveEnd,
    }: React.PropsWithChildren<{ nodes?: Array<{ id: string }>; onMoveEnd?: (event: unknown, viewport: unknown) => void }>) => (
      <div data-testid="react-flow">
        <div>react-flow-nodes:{nodes?.length ?? 0}</div>
        <button type="button" onClick={() => onMoveEnd?.(null, { x: 1, y: 2, zoom: 0.5 })}>
          trigger-move-end
        </button>
        {children}
      </div>
    ),
    ReactFlowProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    useEdgesState: (initial: unknown[]) => ReactModule.useState(initial).concat([(value: unknown) => value]) as unknown,
    useNodesState: (initial: unknown[]) => ReactModule.useState(initial).concat([(value: unknown) => value]) as unknown,
    useReactFlow: () => ({
      fitView,
      setViewport,
      getNodes: () => [
        { id: "exec-1", position: { x: 0, y: 0 }, width: 100, height: 80 },
        { id: "exec-2", position: { x: 40, y: 50 }, width: 100, height: 80 },
      ],
      fitBounds,
      zoomIn,
      zoomOut,
    }),
    useStore: (selector: (state: { transform: [number, number, number] }) => unknown) =>
      selector({ transform: [0, 0, currentZoom] }),
  };
});

vi.mock("@/components/ui/icon-bridge", () => {
  const icon = (name: string) => (props: Record<string, unknown>) => (
    <span data-testid={name} {...props}>
      {name}
    </span>
  );

  return {
    Calendar: icon("calendar"),
    CheckmarkFilled: icon("checkmark-filled"),
    Time: icon("time"),
    User: icon("user"),
    ChevronDown: icon("chevron-down"),
    ChevronUp: icon("chevron-up"),
    Filter: icon("filter"),
    Maximize2: icon("maximize-2"),
    Minus: icon("minus"),
    Plus: icon("plus"),
    Scan: icon("scan"),
    Search: icon("search"),
    TreeStructure: icon("tree-structure"),
    FlowArrow: icon("flow-arrow"),
    SquaresFour: icon("squares-four"),
    GridFour: icon("grid-four"),
    Layers: icon("layers"),
    Eye: icon("eye"),
    Zap: icon("zap"),
    Bug: icon("bug"),
    Focus: icon("focus"),
    EyeOff: icon("eye-off"),
    Loader2: icon("loader-2"),
    Check: icon("check"),
    X: icon("x"),
  };
});

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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardFooter: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: ({
    value,
    onChange,
    placeholder,
    className,
  }: {
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLInputElement>;
    placeholder?: string;
    className?: string;
  }) => <input value={value} onChange={onChange} placeholder={placeholder} className={className} />,
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <div data-testid="separator" />,
}));

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuLabel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuSeparator: () => <div data-testid="dropdown-separator" />,
  DropdownMenuItem: ({
    children,
    onClick,
  }: React.PropsWithChildren<{ onClick?: () => void }>) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/WorkflowDAG/AgentBadge", () => ({
  AgentBadge: ({ agentName }: { agentName: string }) => <span>{agentName}</span>,
  AgentColorDot: ({ agentName }: { agentName: string }) => <span>{agentName}-dot</span>,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/agentColorManager", () => ({
  agentColorManager: {
    cleanupUnusedAgents: vi.fn(),
    getAgentColor: (name: string) => ({
      name,
      primary: "#111111",
      border: "#222222",
    }),
  },
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status: string) => {
    if (status === "completed") return "succeeded";
    return status;
  },
  getStatusLabel: (status: string) => status.toUpperCase(),
  getStatusTheme: (status: string) => ({
    hexColor: status === "running" ? "#00aa00" : "#aaaa00",
    icon: ({ className }: { className?: string }) => <span className={className}>{status}-icon</span>,
    iconClass: `theme-${status}`,
    motion: status === "running" ? "live" : "still",
  }),
}));

vi.mock("@/components/WorkflowDAG/WorkflowGraphControls", () => ({
  WorkflowGraphControls: ({ show }: { show: boolean }) => <div>graph-controls:{String(show)}</div>,
}));

vi.mock("@/components/WorkflowDAG/FloatingConnectionLine", () => ({
  default: () => <div>connection-line</div>,
}));

vi.mock("@/components/WorkflowDAG/FloatingEdge", () => ({
  default: () => <div>floating-edge</div>,
}));

vi.mock("@/components/WorkflowDAG/NodeDetailSidebar", () => ({
  NodeDetailSidebar: ({
    node,
    isOpen,
  }: {
    node: { execution_id?: string } | null;
    isOpen: boolean;
  }) => <div>sidebar:{isOpen ? node?.execution_id ?? "open" : "closed"}</div>,
}));

vi.mock("@/components/WorkflowDAG/VirtualizedDAG", () => ({
  VirtualizedDAG: ({ nodes }: { nodes: unknown[] }) => <div>virtualized:{nodes.length}</div>,
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
    getDefaultLayout() {
      return "tree";
    }

    getAvailableLayouts() {
      return ["tree", "flow", "box"];
    }

    isSlowLayout(layout: string) {
      return layout === "flow";
    }

    isLargeGraph(count: number) {
      return count > 2000;
    }

    async applyLayout(nodes: Array<{ id: string }>, edges: Array<{ id: string }>) {
      return {
        nodes: nodes.map((node, index) => ({
          ...node,
          position: { x: index * 100, y: 0 },
        })),
        edges,
      };
    }
  },
}));

vi.mock("@/components/WorkflowDAG/AgentLegend", async () => {
  const actual = await vi.importActual<typeof import("@/components/WorkflowDAG/AgentLegend")>(
    "@/components/WorkflowDAG/AgentLegend"
  );
  return actual;
});

vi.mock("@/components/WorkflowDAG/WorkflowNode", async () => {
  const actual = await vi.importActual<typeof import("@/components/WorkflowDAG/WorkflowNode")>(
    "@/components/WorkflowDAG/WorkflowNode"
  );
  return actual;
});

vi.mock("@/services/workflowsApi", () => ({
  getWorkflowDAG: vi.fn(),
}));

vi.mock("@/utils/numberFormat", () => ({
  formatNumberWithCommas: (value: number) => value.toLocaleString("en-US"),
}));

const baseNodeData = {
  workflow_id: "wf-1",
  execution_id: "exec-1",
  agent_node_id: "agent-1",
  reasoner_id: "task_runner",
  status: "running",
  started_at: "2026-04-08T10:00:00Z",
  completed_at: "2026-04-08T10:00:10Z",
  duration_ms: 1500,
  workflow_depth: 1,
  task_name: "plan_next_step",
  agent_name: "alpha_agent",
  parent_execution_id: "parent-1",
};

describe("WorkflowDAG graph components", () => {
  beforeEach(() => {
    currentZoom = 1;
    fitView.mockReset();
    fitBounds.mockReset();
    setViewport.mockReset();
    zoomIn.mockReset();
    zoomOut.mockReset();
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        disconnect() {}
      }
    );
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      value: (cb: FrameRequestCallback) => {
        cb(0);
        return 1;
      },
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      value: vi.fn(),
    });
    localStorage.clear();
  });

  it("renders a detailed workflow node in debug mode", () => {
    render(
      <WorkflowNode
        data={{
          ...baseNodeData,
          viewMode: "debug",
          selected: true,
        } as never}
        selected
      />
    );

    expect(screen.getAllByText("Plan Next Step")).toHaveLength(2);
    expect(screen.getByText("alpha_agent")).toBeInTheDocument();
    expect(screen.getAllByText("1.5s")[0]).toBeInTheDocument();
    expect(screen.getByText(/ID: exec-1/)).toBeInTheDocument();
    expect(screen.getByText(/Parent: parent-1/)).toBeInTheDocument();
    expect(screen.getAllByText("RUNNING")).toHaveLength(2);
  });

  it("renders the simplified workflow node placeholder when zoomed out", () => {
    currentZoom = 0.2;

    render(<WorkflowNode data={baseNodeData as never} />);

    expect(screen.queryByText("Plan Next Step")).not.toBeInTheDocument();
    expect(screen.getByTitle("RUNNING - plan_next_step")).toBeInTheDocument();
  });

  it("renders agent legends across embedded, full, and mini modes", async () => {
    const user = userEvent.setup();
    const onAgentFilter = vi.fn();
    const onExpandGraph = vi.fn();
    const nodes = [
      { id: "1", data: { agent_name: "Alpha" } },
      { id: "2", data: { agent_name: "Beta" } },
      { id: "3", data: { agent_name: "Gamma" } },
      { id: "4", data: { agent_name: "Delta" } },
      { id: "5", data: { agent_name: "Epsilon" } },
      { id: "6", data: { agent_name: "Zeta" } },
      { id: "7", data: { agent_name: "Eta" } },
      { id: "8", data: { agent_name: "Theta" } },
      { id: "9", data: { agent_name: "Iota" } },
    ];

    const { rerender } = render(
      <AgentLegend
        layout="embedded"
        nodes={nodes as never}
        selectedAgent="Alpha"
        onAgentFilter={onAgentFilter}
        onFitView={vi.fn()}
        onZoomIn={vi.fn()}
        onZoomOut={vi.fn()}
        onExpandGraph={onExpandGraph}
      />
    );

    expect(screen.getByRole("button", { name: /agents: 9/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /clear filter/i }));
    expect(onAgentFilter).toHaveBeenCalledWith(null);
    await user.click(screen.getAllByRole("button", { name: /expand graph to full screen/i })[0]);
    expect(onExpandGraph).toHaveBeenCalled();

    rerender(
      <AgentLegend
        nodes={nodes as never}
        onAgentFilter={onAgentFilter}
        selectedAgent={null}
      />
    );

    await user.type(screen.getByPlaceholderText("Search agents…"), "zzz");
    expect(screen.getByText(/no agents match/i)).toBeInTheDocument();

    rerender(
      <AgentLegendMini
        nodes={nodes as never}
        onAgentFilter={onAgentFilter}
        selectedAgent="Alpha"
      />
    );

    expect(screen.getByText("+1")).toBeInTheDocument();
  });

  it("handles graph toolbar layout, mode, focus, and center actions", async () => {
    const user = userEvent.setup();
    const onLayoutChange = vi.fn();
    const onViewModeChange = vi.fn();
    const onSearchToggle = vi.fn();
    const onSmartCenter = vi.fn();
    const onFocusModeChange = vi.fn();

    render(
      <GraphToolbar
        availableLayouts={["tree", "flow", "box"]}
        currentLayout="tree"
        onLayoutChange={onLayoutChange}
        isSlowLayout={(layout) => layout === "flow"}
        isLargeGraph
        isApplyingLayout={false}
        viewMode="standard"
        onViewModeChange={onViewModeChange}
        focusMode={false}
        onFocusModeChange={onFocusModeChange}
        onSearchToggle={onSearchToggle}
        showSearch={false}
        onSmartCenter={onSmartCenter}
        hasSelection
        controlsReady
      />
    );

    await user.click(screen.getByTestId("search").closest("button") as HTMLButtonElement);
    expect(onSearchToggle).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /flow/i }));
    expect(onLayoutChange).toHaveBeenCalledWith("flow");

    await user.click(screen.getByRole("button", { name: /performance/i }));
    expect(onViewModeChange).toHaveBeenCalledWith("performance");

    await user.click(screen.getByTestId("focus").closest("button") as HTMLButtonElement);
    expect(onFocusModeChange).toHaveBeenCalledWith(true);

    await user.click(screen.getByTestId("scan").closest("button") as HTMLButtonElement);
    expect(onSmartCenter).toHaveBeenCalledTimes(1);

    expect(screen.getByText(/some layouts may be slower/i)).toBeInTheDocument();
  });

  it("renders workflow dag viewer states and exposes controls through onReady", async () => {
    const onReady = vi.fn();
    const onLayoutInfoChange = vi.fn();
    const onSearchResultsChange = vi.fn();
    const dagData = {
      root_workflow_id: "wf-1",
      session_id: "session-1",
      actor_id: "actor-1",
      total_nodes: 2,
      displayed_nodes: 2,
      max_depth: 1,
      workflow_status: "running",
      workflow_name: "Test Workflow",
      dag: baseNodeData,
      timeline: [
        baseNodeData,
        {
          ...baseNodeData,
          execution_id: "exec-2",
          agent_node_id: "agent-2",
          agent_name: "beta_agent",
          reasoner_id: "child_step",
          parent_execution_id: "exec-1",
          status: "completed",
        },
      ],
    };

    const { rerender } = render(
      <WorkflowDAGViewer workflowId="wf-loading" loading />
    );
    expect(screen.getByText("Loading workflow DAG...")).toBeInTheDocument();

    rerender(<WorkflowDAGViewer workflowId="wf-error" error="boom" />);
    expect(screen.getByText("Failed to load workflow DAG")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();

    rerender(
      <WorkflowDAGViewer
        workflowId="wf-1"
        dagData={dagData as never}
        onReady={onReady}
        onLayoutInfoChange={onLayoutInfoChange}
        onSearchResultsChange={onSearchResultsChange}
        searchQuery="child"
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("react-flow")).toBeInTheDocument();
      expect(onReady).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText("react-flow-nodes:2")).toBeInTheDocument();

    const controls = onReady.mock.calls[0][0] as {
      fitToView: (options?: { padding?: number }) => void;
      focusOnNodes: (nodeIds: string[], options?: { padding?: number }) => void;
      changeLayout: (layout: "tree" | "flow" | "box") => void;
    };

    act(() => {
      controls.fitToView({ padding: 0.4 });
      controls.focusOnNodes(["exec-1"], { padding: 0.1 });
      controls.changeLayout("flow");
    });

    expect(fitView).toHaveBeenCalled();
    expect(fitBounds).toHaveBeenCalled();

    await waitFor(() => {
      expect(onLayoutInfoChange).toHaveBeenCalled();
      expect(onSearchResultsChange).toHaveBeenCalled();
    });

    expect(onSearchResultsChange.mock.calls.at(-1)?.[0]).toMatchObject({
      totalMatches: expect.any(Number),
    });

    await userEvent.click(screen.getByRole("button", { name: /trigger-move-end/i }));
    expect(localStorage.getItem("workflowDAGViewport:wf-1")).toBe('{"x":1,"y":2,"zoom":0.5}');
  });
});
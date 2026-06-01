// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <span className={className}>icon</span>;
  return {
    ChevronDown: Icon,
    ChevronUp: Icon,
    Filter: Icon,
    Maximize2: Icon,
    Minus: Icon,
    Plus: Icon,
    Scan: Icon,
    Search: Icon,
  };
});

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <span className={className}>{children}</span>,
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
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardFooter: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
}));

vi.mock("@/components/ui/input", () => ({
  Input: ({
    value,
    onChange,
    placeholder,
  }: {
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLInputElement>;
    placeholder?: string;
  }) => <input value={value} onChange={onChange} placeholder={placeholder} />,
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <div data-testid="separator" />,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/agentColorManager", () => ({
  agentColorManager: {
    cleanupUnusedAgents: vi.fn(),
    getAgentColor: (name: string, agentId?: string) => ({
      name,
      primary: agentId ? "#123456" : "#654321",
      border: "#222222",
      text: "#ffffff",
    }),
    getAgentInitials: (name: string) =>
      name
        .split(" ")
        .map((part) => part[0])
        .join("")
        .slice(0, 2)
        .toUpperCase(),
  },
}));

function makeNodes(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `node-${index + 1}`,
    data: {
      agent_name: `Agent ${index + 1}`,
      agent_node_id: `agent-${index + 1}`,
    },
  }));
}

describe("AgentBadge and AgentLegend", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders badge sizes, initials, and dot styles", async () => {
    const { AgentBadge, AgentColorDot } = await import("@/components/WorkflowDAG/AgentBadge");

    const { rerender } = render(
      <div>
        <AgentBadge agentName="Alpha Bot" agentId="agent-1" size="sm" />
        <AgentColorDot agentName="Alpha Bot" size={12} />
      </div>,
    );

    expect(screen.getByText("AB")).toBeInTheDocument();
    expect(screen.getAllByTitle("Agent: Alpha Bot")).toHaveLength(2);

    rerender(<AgentBadge agentName="Beta Crew" size="lg" showTooltip={false} className="extra-class" />);
    expect(screen.getByText("BC").parentElement?.className).toContain("extra-class");
    expect(screen.getByText("BC").parentElement).not.toHaveAttribute("title");
  });

  it("returns null when no agents are available", async () => {
    const { AgentLegend } = await import("@/components/WorkflowDAG/AgentLegend");
    const { container } = render(<AgentLegend nodes={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders embedded layout actions and clears a selected filter", async () => {
    const onAgentFilter = vi.fn();
    const onFitView = vi.fn();
    const onZoomIn = vi.fn();
    const onZoomOut = vi.fn();
    const onExpandGraph = vi.fn();
    const { AgentLegend } = await import("@/components/WorkflowDAG/AgentLegend");

    render(
      <AgentLegend
        layout="embedded"
        nodes={makeNodes(3) as never}
        selectedAgent="Agent 2"
        onAgentFilter={onAgentFilter}
        onFitView={onFitView}
        onZoomIn={onZoomIn}
        onZoomOut={onZoomOut}
        onExpandGraph={onExpandGraph}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /agents: 3/i }));
    fireEvent.click(screen.getByRole("button", { name: /fit graph to view/i }));
    fireEvent.click(screen.getByRole("button", { name: /zoom in/i }));
    fireEvent.click(screen.getByRole("button", { name: /zoom out/i }));
    fireEvent.click(screen.getAllByRole("button", { name: /expand graph to full screen/i })[0]);
    fireEvent.click(screen.getAllByRole("button", { name: /agent 2/i })[1]);
    fireEvent.click(screen.getByRole("button", { name: /clear filter/i }));

    expect(onFitView).toHaveBeenCalled();
    expect(onZoomIn).toHaveBeenCalled();
    expect(onZoomOut).toHaveBeenCalled();
    expect(onExpandGraph).toHaveBeenCalled();
    expect(onAgentFilter).toHaveBeenNthCalledWith(1, null);
    expect(onAgentFilter).toHaveBeenNthCalledWith(2, null);
  });

  it("renders searchable fullscreen layout and empty search state", async () => {
    const onAgentFilter = vi.fn();
    const { AgentLegend } = await import("@/components/WorkflowDAG/AgentLegend");

    render(
      <AgentLegend
        nodes={makeNodes(7) as never}
        onAgentFilter={onAgentFilter}
        selectedAgent={null}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Search agents…"), {
      target: { value: "missing" },
    });
    expect(screen.getByText(/no agents match/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /collapse agent list/i }));
    expect(screen.queryByPlaceholderText("Search agents…")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /expand agent list/i }));
    fireEvent.change(screen.getByPlaceholderText("Search agents…"), {
      target: { value: "Agent 3" },
    });
    fireEvent.click(screen.getByRole("button", { name: /agent 3/i }));
    expect(onAgentFilter).toHaveBeenCalledWith("Agent 3");
  });

  it("renders compact and mini variants", async () => {
    const onAgentFilter = vi.fn();
    const { AgentLegend, AgentLegendMini } = await import("@/components/WorkflowDAG/AgentLegend");

    const { rerender } = render(
      <AgentLegend
        compact
        nodes={makeNodes(4) as never}
        selectedAgent="Agent 1"
        onAgentFilter={onAgentFilter}
      />,
    );

    expect(screen.getByText("Clear filter")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /agent 1/i }));
    expect(onAgentFilter).toHaveBeenCalledWith(null);

    rerender(
      <AgentLegendMini
        nodes={makeNodes(10) as never}
        selectedAgent="Agent 2"
        onAgentFilter={onAgentFilter}
      />,
    );

    expect(screen.getByText("+2")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("Agent 2"));
    expect(onAgentFilter).toHaveBeenCalledWith(null);
  });
});

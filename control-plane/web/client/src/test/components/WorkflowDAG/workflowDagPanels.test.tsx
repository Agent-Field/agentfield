// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionPanel } from "@/components/WorkflowDAG/ExecutionPanel";
import { HoverDetailPanel } from "@/components/WorkflowDAG/HoverDetailPanel";
import { TechnicalSection } from "@/components/WorkflowDAG/sections/TechnicalSection";

vi.mock("@/components/ui/icon-bridge", () => {
  const icon = (name: string) => (props: Record<string, unknown>) => (
    <span data-testid={name} {...props}>
      {name}
    </span>
  );

  return {
    Close: icon("close"),
    Time: icon("time"),
    Activity: icon("activity"),
    Layers: icon("layers"),
    Copy: icon("copy"),
    CheckmarkFilled: icon("checkmark-filled"),
    Calendar: icon("calendar"),
    ErrorFilled: icon("error-filled"),
    InProgress: icon("in-progress"),
    PauseFilled: icon("pause-filled"),
    ChevronDown: icon("chevron-down"),
    ChevronRight: icon("chevron-right"),
    Chip: icon("chip"),
    Code: icon("code"),
    Settings: icon("settings"),
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
    <div data-testid="card" className={className}>
      {children}
    </div>
  ),
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data, className }: { data: unknown; className?: string }) => (
    <pre className={className}>{JSON.stringify(data, null, 2)}</pre>
  ),
}));

vi.mock("@/components/WorkflowDAG/AgentBadge", () => ({
  AgentBadge: ({
    agentName,
    agentId,
  }: {
    agentName: string;
    agentId?: string;
  }) => <span>{agentName}:{agentId ?? "none"}</span>,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/agentColorManager", () => ({
  agentColorManager: {
    getAgentColor: (name: string) => ({
      name,
      primary: "#123456",
      border: "#654321",
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
    badgeVariant: status === "failed" ? "destructive" : "secondary",
    pillClass: `pill-${status}`,
  }),
}));

const baseNode = {
  workflow_id: "workflow-1234567890",
  execution_id: "execution-1234567890",
  agent_node_id: "agent-node",
  reasoner_id: "reasoner_step",
  status: "running",
  started_at: "2026-04-08T10:00:00Z",
  completed_at: "2026-04-08T10:01:00Z",
  duration_ms: 65000,
  parent_workflow_id: "parent-workflow-1234567890",
  workflow_depth: 2,
  children: [
    { execution_id: "child-1", status: "running" },
    { execution_id: "child-2", status: "failed" },
    { execution_id: "child-3", status: "queued" },
    { execution_id: "child-4", status: "succeeded" },
  ],
  agent_name: "planner_agent",
  task_name: "run_reasoner_step",
};

describe("WorkflowDAG panels and sections", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T10:05:00Z"));
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 600 });
    Object.defineProperty(window, "innerHeight", { configurable: true, value: 260 });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  it("renders hover details with adjusted position and humanized text", async () => {
    const node = {
      ...baseNode,
      status: "failed",
      task_name: "send_follow_up-email",
    };

    render(
      <HoverDetailPanel
        node={node as never}
        position={{ x: 580, y: 250 }}
        visible
      />
    );

    await act(async () => {
      vi.advanceTimersByTime(60);
    });

    const panel = screen.getByText("Send Follow Up Email").closest("div[style]");
    expect(screen.getByText("Send Follow Up Email")).toBeInTheDocument();
    expect(screen.getByText("Planner Agent:agent-node")).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByText("1m 5s")).toBeInTheDocument();
    expect(screen.getByText("Level 2")).toBeInTheDocument();
    expect(screen.getByText("execution-12...")).toBeInTheDocument();
    expect(panel).toHaveStyle({ left: "245px", top: "20px" });
  });

  it("renders the execution panel and copies identifiers", async () => {
    const onClose = vi.fn();

    render(
      <ExecutionPanel
        execution={baseNode as never}
        onClose={onClose}
        isOpen
        task_name="Human Task"
        agent_name="Human Agent"
      />
    );

    expect(screen.getByText("Execution Details")).toBeInTheDocument();
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
    expect(screen.getByText("Human Task")).toBeInTheDocument();
    expect(screen.getByText("Human Agent")).toBeInTheDocument();
    expect(screen.getByText("1.1m")).toBeInTheDocument();
    expect(screen.getByText("4 child executions")).toBeInTheDocument();
    expect(screen.getByText("+1 more")).toBeInTheDocument();

    const copyButtons = screen.getAllByRole("button");
    fireEvent.click(copyButtons[1]);

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("parent-workflow-1234567890");

    fireEvent.click(copyButtons[0]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders and expands technical details while forwarding copy actions", () => {
    const onCopy = vi.fn();

    render(
      <TechnicalSection
        node={baseNode}
        details={{
          input: { a: 1 },
          output: { ok: true },
          cost: 1.23456,
          memory_updates: [
            { action: "set", scope: "session", key: "result", value: { ok: true } },
          ],
          performance_metrics: {
            response_time_ms: 321,
            tokens_used: 12345,
          },
        }}
        onCopy={onCopy}
        copySuccess="Workflow ID"
      />
    );

    expect(screen.getByText("Technical Details")).toBeInTheDocument();
    expect(screen.getByText("321ms")).toBeInTheDocument();
    expect(screen.getByText("12,345")).toBeInTheDocument();
    expect(screen.getByText("$1.2346")).toBeInTheDocument();
    expect(screen.getAllByText("Yes")).toHaveLength(2);

    fireEvent.click(screen.getByTitle("Copy Agent Node ID"));
    expect(onCopy).toHaveBeenCalledWith("agent-node", "Agent Node ID");

    fireEvent.click(screen.getByRole("button", { name: /memory updates/i }));
    expect(screen.getByText("session/result")).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes('"ok": true'))).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /raw node data/i }));
    expect(screen.getAllByText(/workflow-1234567890/)[0]).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /raw details data/i }));
    expect(screen.getAllByText(/"response_time_ms": 321/)[0]).toBeInTheDocument();
  });

  it("returns null when the execution panel has no execution", () => {
    const { container } = render(
      <ExecutionPanel execution={null} onClose={vi.fn()} isOpen={false} />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("does not render hover details when hidden", () => {
    const { container } = render(
      <HoverDetailPanel node={baseNode as never} position={{ x: 0, y: 0 }} visible={false} />
    );

    expect(container).toBeEmptyDOMElement();
  });
});
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const sectionState = vi.hoisted(() => ({
  navigate: vi.fn(),
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => sectionState.navigate,
  };
});

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <span className={className}>icon</span>;
  return {
    Calendar: Icon,
    Time: Icon,
    Timer: Icon,
    CheckmarkFilled: Icon,
    ErrorFilled: Icon,
    InProgress: Icon,
    PauseFilled: Icon,
    WarningFilled: Icon,
    Copy: Icon,
    User: Icon,
    Launch: Icon,
  };
});

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
  CardTitle: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => <div className={className}>{children}</div>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    title,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} title={title} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <span className={className}>{children}</span>
  ),
}));

function makeNode(overrides: Record<string, unknown> = {}) {
  return {
    workflow_id: "wf-1",
    execution_id: "exec-12345678",
    agent_node_id: "agent-node-1",
    reasoner_id: "reasoner.main",
    status: "running",
    started_at: "2026-04-08T10:00:00Z",
    completed_at: "2026-04-08T10:02:30Z",
    duration_ms: 150000,
    workflow_depth: 2,
    task_name: "Main task",
    agent_name: "Agent Alpha",
    ...overrides,
  };
}

describe("Workflow DAG detail sections", () => {
  beforeEach(() => {
    sectionState.navigate.mockReset();
    vi.useRealTimers();
  });

  it("renders failed StatusSection details and truncates memory updates", async () => {
    const { StatusSection } = await import("@/components/WorkflowDAG/sections/StatusSection");

    render(
      <StatusSection
        node={makeNode({ status: "failed" }) as never}
        details={{
          error_message: "stack exploded",
          memory_updates: [
            { action: "set", scope: "session", key: "a" },
            { action: "merge", scope: "session", key: "b" },
            { action: "drop", scope: "cache", key: "c" },
            { action: "set", scope: "cache", key: "d" },
          ],
        }}
      />,
    );

    expect(screen.getByText("Execution Failed")).toBeInTheDocument();
    expect(screen.getAllByText("stack exploded")).toHaveLength(2);
    expect(screen.getByText("Error Details")).toBeInTheDocument();
    expect(screen.getByText("set session/a")).toBeInTheDocument();
    expect(screen.getByText("+1 more updates")).toBeInTheDocument();
  });

  it("renders running StatusSection progress state", async () => {
    const { StatusSection } = await import("@/components/WorkflowDAG/sections/StatusSection");

    render(<StatusSection node={makeNode({ status: "running" }) as never} />);

    expect(screen.getByText("Currently Running")).toBeInTheDocument();
    expect(screen.getByText("Execution in progress...")).toBeInTheDocument();
  });

  it("renders TimingSection timestamps, metrics, and live running duration", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T10:05:10Z"));
    const { TimingSection } = await import("@/components/WorkflowDAG/sections/TimingSection");

    render(
      <TimingSection
        node={makeNode({
          status: "running",
          started_at: "2026-04-08T10:00:00Z",
          completed_at: undefined,
          duration_ms: 900,
        }) as never}
        details={{
          performance_metrics: {
            response_time_ms: 321,
            tokens_used: 12345,
          },
        }}
      />,
    );

    expect(screen.getByText("Timing Information")).toBeInTheDocument();
    expect(screen.getByText("900ms")).toBeInTheDocument();
    expect(screen.getByText("5m 10s")).toBeInTheDocument();
    expect(screen.getByText("Response Time")).toBeInTheDocument();
    expect(screen.getByText("321ms")).toBeInTheDocument();
    expect(screen.getByText("12,345")).toBeInTheDocument();
    expect(screen.getByText("Execution in progress")).toBeInTheDocument();
    expect(screen.getByText("Started 310s ago")).toBeInTheDocument();
  });

  it("renders ExecutionHeader actions, stats, and optional cost", async () => {
    const onCopy = vi.fn();
    const { ExecutionHeader } = await import("@/components/WorkflowDAG/sections/ExecutionHeader");

    render(
      <ExecutionHeader
        node={makeNode({ status: "queued" }) as never}
        details={{
          cost: 1.23456,
          performance_metrics: {
            response_time_ms: 456,
          },
        }}
        onCopy={onCopy}
        copySuccess="Execution ID"
      />,
    );

    expect(screen.getByText("Main task")).toBeInTheDocument();
    expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
    expect(screen.getByText("Queued")).toBeInTheDocument();
    expect(screen.getByText("exec-123...")).toBeInTheDocument();
    expect(screen.getByText("Level 2")).toBeInTheDocument();
    expect(screen.getByText("456ms")).toBeInTheDocument();
    expect(screen.getByText("$1.2346")).toBeInTheDocument();

    fireEvent.click(screen.getByTitle("Copy Execution ID"));
    expect(onCopy).toHaveBeenCalledWith("exec-12345678", "Execution ID");

    fireEvent.click(screen.getByRole("button", { name: /view full execution details/i }));
    expect(sectionState.navigate).toHaveBeenCalledWith("/executions/exec-12345678");
  });
});

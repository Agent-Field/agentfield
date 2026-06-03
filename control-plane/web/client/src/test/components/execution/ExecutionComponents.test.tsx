import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CollapsibleSection } from "@/components/execution/CollapsibleSection";
import { ExecutionApprovalPanel } from "@/components/execution/ExecutionApprovalPanel";
import { ExecutionHeader } from "@/components/execution/ExecutionHeader";
import { ExecutionPerformanceStrip } from "@/components/execution/ExecutionPerformanceStrip";
import { ExecutionTimeline } from "@/components/execution/ExecutionTimeline";
import { TechnicalDetailsPanel } from "@/components/execution/TechnicalDetailsPanel";

vi.mock("@/components/ui/icon-bridge", () => ({
  ChevronDown: () => <span>chevron-down</span>,
  ChevronRight: () => <span>chevron-right</span>,
  ArrowLeft: () => <span>arrow-left</span>,
  CheckCircle: () => <span>check-circle</span>,
  XCircle: () => <span>x-circle</span>,
  Clock: () => <span>clock</span>,
  ExternalLink: () => <span>external-link</span>,
  PauseCircle: () => <span>pause-circle</span>,
  Timer: () => <span>timer</span>,
  Settings: () => <span>settings</span>,
  Play: () => <span>play</span>,
  ArrowDown: () => <span>arrow-down</span>,
  ArrowUp: () => <span>arrow-up</span>,
  RotateCcw: () => <span>rotate-ccw</span>,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/copy-button", () => ({
  CopyButton: ({ value, tooltip }: { value: string; tooltip?: string }) => (
    <button type="button" aria-label={tooltip ?? "copy"}>
      copy {value}
    </button>
  ),
}));

vi.mock("@/components/layout/ResponsiveGrid", () => ({
  ResponsiveGrid: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status: string) => {
    if (status === "completed") return "succeeded";
    return status;
  },
}));

vi.mock("@/utils/dateFormat", () => ({
  formatCompactRelativeTime: (timestamp: string) => `relative:${timestamp}`,
}));

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    execution_id: "exec-1",
    workflow_id: "wf-1",
    session_id: "session-1",
    agentfield_request_id: "request-1",
    actor_id: "actor-1",
    workflow_depth: 2,
    retry_count: 1,
    duration_ms: 65000,
    input_size: 2048,
    output_size: 1024,
    created_at: "2026-04-08T00:00:00Z",
    updated_at: "2026-04-08T00:01:00Z",
    started_at: "2026-04-08T00:00:30Z",
    completed_at: "2026-04-08T00:01:30Z",
    status: "completed",
    reasoner_id: "planner",
    approval_request_id: "approval-1",
    approval_status: "approved",
    approval_response: JSON.stringify({ feedback: "Looks good" }),
    approval_requested_at: "2026-04-08T00:00:00Z",
    approval_responded_at: "2026-04-08T00:02:00Z",
    approval_request_url: "https://example.com/review",
    ...overrides,
  };
}

describe("execution components", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("toggles a collapsible section", async () => {
    const user = userEvent.setup();

    render(
      <CollapsibleSection title="Section Title" badge={<span>badge</span>}>
        <div>Hidden content</div>
      </CollapsibleSection>
    );

    expect(screen.queryByText("Hidden content")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /section title/i }));
    expect(screen.getByText("Hidden content")).toBeInTheDocument();
  });

  it("renders header, timeline, performance strip, and technical details", () => {
    const onNavigateBack = vi.fn();
    const execution = buildExecution();

    const { rerender } = render(
      <div>
        <ExecutionHeader execution={execution as never} onNavigateBack={onNavigateBack} />
        <ExecutionTimeline execution={execution as never} />
        <ExecutionPerformanceStrip execution={execution as never} />
        <TechnicalDetailsPanel execution={execution as never} />
      </div>
    );

    fireEvent.click(screen.getByRole("button"));
    expect(onNavigateBack).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Execution: exec-1")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("planner")).toBeInTheDocument();
    expect(screen.getByText("Started relative:2026-04-08T00:00:30Z")).toBeInTheDocument();
    expect(screen.getByText("Completed relative:2026-04-08T00:01:30Z")).toBeInTheDocument();
    expect(screen.getByText("1m 5s")).toBeInTheDocument();
    expect(screen.getByText("(Retried)")).toBeInTheDocument();
    expect(screen.getByText("2.0 KB")).toBeInTheDocument();
    expect(screen.getByText("1.0 KB")).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("Technical Details")).toBeInTheDocument();
    expect(screen.getByText("request-1")).toBeInTheDocument();

    rerender(
      <ExecutionPerformanceStrip
        execution={buildExecution({
          status: "failed",
          retry_count: 0,
          input_size: 0,
          output_size: 0,
          duration_ms: 0,
        }) as never}
      />
    );
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("(Failed)")).toBeInTheDocument();
    expect(screen.getByText("No data")).toBeInTheDocument();
  });

  it("renders approval details and fallback when no approval exists", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ExecutionApprovalPanel execution={buildExecution() as never} />
    );

    expect(screen.getByText("Approval Status")).toBeInTheDocument();
    expect(screen.getByText("Approved")).toBeInTheDocument();
    expect(screen.getByText("Looks good")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /request details/i }));
    expect(screen.getByText("Requested At")).toBeInTheDocument();
    expect(screen.getByText("approval-1")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open in hub/i })).toHaveAttribute(
      "href",
      "https://example.com/review"
    );

    rerender(
      <ExecutionApprovalPanel
        execution={buildExecution({
          approval_request_id: undefined,
        }) as never}
      />
    );
    expect(screen.getByText("No approval request for this execution.")).toBeInTheDocument();
  });
});

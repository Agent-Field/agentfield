import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { StepDetail } from "@/components/StepDetail";

const state = vi.hoisted(() => ({
  useStepDetail: vi.fn(),
  invalidateQueries: vi.fn(),
  retryExecutionWebhook: vi.fn(),
  clipboardWriteText: vi.fn(),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: state.invalidateQueries,
  }),
}));

vi.mock("@/hooks/queries", () => ({
  useStepDetail: (executionId: string) => state.useStepDetail(executionId),
}));

vi.mock("@/services/executionsApi", () => ({
  retryExecutionWebhook: (executionId: string) => state.retryExecutionWebhook(executionId),
}));

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    title,
    "aria-label": ariaLabel,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      {...props}
    >
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>loading</div>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleTrigger: ({
    children,
    className,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" className={className} {...props}>
      {children}
    </button>
  ),
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuLabel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onSelect,
  }: React.PropsWithChildren<{ onSelect?: () => void }>) => (
    <button type="button" onClick={onSelect}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data }: { data: unknown }) => (
    <pre>{JSON.stringify(data, null, 2)}</pre>
  ),
}));

vi.mock("@/components/StepProvenanceCard", () => ({
  StepProvenanceCard: ({
    callerDid,
    targetDid,
  }: {
    callerDid?: string;
    targetDid?: string;
  }) => <div>Provenance {callerDid} {targetDid}</div>,
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  ChevronDown: () => <span>chevron-down</span>,
}));

vi.mock("lucide-react", () => ({
  Copy: () => <span>copy</span>,
  Check: () => <span>check</span>,
  ShieldAlert: () => <span>shield-alert</span>,
  RefreshCw: () => <span>refresh</span>,
  Terminal: () => <span>terminal</span>,
  Info: () => <span>info</span>,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/components/RunTrace", () => ({
  formatDuration: (durationMs?: number) => `${durationMs ?? 0}ms`,
}));

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    execution_id: "exec-1",
    workflow_id: "wf-1",
    reasoner_id: "planner",
    agent_node_id: "agent-1",
    duration_ms: 1250,
    workflow_depth: 2,
    caller_did: "did:caller:1",
    target_did: "did:target:1",
    input_hash: "input-hash",
    output_hash: "output-hash",
    input_data: { hello: "world" },
    output_data: { ok: true },
    notes: [
      {
        timestamp: "2026-04-08T00:00:00Z",
        message: "step note",
        tags: ["debug"],
      },
    ],
    webhook_registered: true,
    webhook_events: [
      {
        id: "event-1",
        event_type: "completed",
        status: "failed",
        http_status: 500,
        created_at: "2026-04-08T00:00:00Z",
      },
    ],
    approval_request_id: "approval-1",
    approval_status: "pending",
    approval_requested_at: "2026-04-08T00:00:00Z",
    error_message: "",
    ...overrides,
  };
}

describe("StepDetail", () => {
  beforeEach(() => {
    state.useStepDetail.mockReset();
    state.invalidateQueries.mockReset();
    state.retryExecutionWebhook.mockReset();
    state.retryExecutionWebhook.mockResolvedValue(undefined);
    state.clipboardWriteText.mockReset();
    state.clipboardWriteText.mockResolvedValue(undefined);
    vi.restoreAllMocks();

    Object.defineProperty(window, "location", {
      configurable: true,
      value: { origin: "http://localhost:3000" },
    });

    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: vi.fn(() => "api-key"),
      },
    });

    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: state.clipboardWriteText,
      },
    });
  });

  it("renders loading and empty states", () => {
    state.useStepDetail.mockReturnValueOnce({ data: undefined, isLoading: true });
    const { rerender } = render(<StepDetail executionId="exec-1" />);
    expect(screen.getAllByText("loading").length).toBeGreaterThan(0);

    state.useStepDetail.mockReturnValueOnce({ data: null, isLoading: false });
    rerender(<StepDetail executionId="exec-1" />);
    expect(screen.getByText("Step not found")).toBeInTheDocument();
  });

  it("renders execution details and supports copy and retry interactions", async () => {
    const user = userEvent.setup();
    state.useStepDetail.mockReturnValue({
      data: buildExecution(),
      isLoading: false,
    });

    render(<StepDetail executionId="exec-1" />);

    expect(screen.getByText("planner")).toBeInTheDocument();
    expect(screen.getByText(/Agent: agent-1/)).toBeInTheDocument();
    expect(screen.getByText("Provenance did:caller:1 did:target:1")).toBeInTheDocument();
    expect(screen.getByText(/step note/)).toBeInTheDocument();
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    expect(screen.getByText("Human Approval Required")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();

    await user.click(screen.getByTitle("Copy workflow ID"));
    expect(await screen.findByText("check")).toBeInTheDocument();

    await user.click(screen.getByLabelText("Copy input JSON"));
    expect(await screen.findAllByText("check")).toHaveLength(2);

  });

  it("submits approval actions and invalidates related queries", async () => {
    const user = userEvent.setup();
    state.useStepDetail.mockReturnValue({
      data: buildExecution(),
      isLoading: false,
    });

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({}),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<StepDetail executionId="exec-1" />);

    await user.click(screen.getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/webhooks/approval-response",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json",
            "X-API-Key": "api-key",
          }),
        })
      );
    });

    expect(fetchMock.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({
        body: JSON.stringify({
          requestId: "approval-1",
          decision: "approved",
        }),
      })
    );

    await waitFor(() => {
      expect(state.invalidateQueries).toHaveBeenCalledWith({
        queryKey: ["step-detail", "exec-1"],
      });
      expect(state.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["run-dag"] });
      expect(state.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["executions"] });
    });
  });

  it("shows the error panel instead of output when execution failed", () => {
    state.useStepDetail.mockReturnValue({
      data: buildExecution({
        error_message: "boom",
      }),
      isLoading: false,
    });

    render(<StepDetail executionId="exec-1" />);

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();
    expect(screen.queryByLabelText("Copy output JSON")).not.toBeInTheDocument();
  });
});

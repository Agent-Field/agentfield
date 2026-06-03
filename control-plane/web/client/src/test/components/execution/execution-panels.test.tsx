// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CompactExecutionHeader } from "@/components/execution/CompactExecutionHeader";
import { ExecutionIdentityPanel } from "@/components/execution/ExecutionIdentityPanel";
import { ExecutionRetryPanel } from "@/components/execution/ExecutionRetryPanel";
import { EnhancedNotesSection } from "@/components/execution/EnhancedNotesSection";
import {
  cancelExecution,
  pauseExecution,
  resumeExecution,
} from "@/services/executionsApi";
import { downloadExecutionVCBundle } from "@/services/vcApi";

const navigate = vi.fn();
const successNotification = vi.fn();
const errorNotification = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    ArrowLeft: Icon,
    ExternalLink: Icon,
    RotateCcw: Icon,
    PauseCircle: Icon,
    Activity: Icon,
    XCircle: Icon,
    Play: Icon,
    MoreHorizontal: Icon,
    Clock: Icon,
    Copy: Icon,
    GitBranch: Icon,
    Shield: Icon,
    AlertCircle: Icon,
    CheckCircle: Icon,
    Eye: Icon,
    Download: Icon,
    Loader2: Icon,
    ArrowCounterClockwise: Icon,
    ArrowSquareOut: Icon,
    Check: Icon,
    SpinnerGap: Icon,
    Terminal: Icon,
    Code: Icon,
    WarningCircle: Icon,
    FileText: Icon,
    RefreshCw: Icon,
    ChevronDown: Icon,
    ChevronUp: Icon,
    Tag: Icon,
    ArrowUpDown: Icon,
    ArrowUp: Icon,
    ArrowDown: Icon,
  };
});

vi.mock("@/components/ui/data-formatters", () => ({
  formatDurationHumanReadable: (value: number) => `${value}ms`,
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

vi.mock("@/components/ui/copy-identifier-chip", () => ({
  CopyIdentifierChip: ({ value }: { value: string }) => <span>{`chip:${value}`}</span>,
  truncateIdMiddle: (value: string) => `truncated:${value}`,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/alert-dialog", () => ({
  AlertDialog: ({ children, open }: React.PropsWithChildren<{ open?: boolean }>) =>
    open ? <div>{children}</div> : null,
  AlertDialogAction: ({
    children,
    onClick,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
  AlertDialogCancel: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
  AlertDialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
  }: React.PropsWithChildren<{ onClick?: () => void }>) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
  DropdownMenuSeparator: () => <div>separator</div>,
}));

vi.mock("@/components/ui/tabs", async () => {
  const ReactModule = await import("react");
  const TabsContext = ReactModule.createContext<{
    value: string;
    onValueChange?: (value: string) => void;
  }>({ value: "curl" });
  return {
    AnimatedTabs: ({
      children,
      value,
      onValueChange,
    }: React.PropsWithChildren<{ value: string; onValueChange?: (value: string) => void }>) => (
      <TabsContext.Provider value={{ value, onValueChange }}>
        <div>{children}</div>
      </TabsContext.Provider>
    ),
    AnimatedTabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    AnimatedTabsTrigger: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" onClick={() => ctx.onValueChange?.(value)}>
          {children}
        </button>
      );
    },
    Tabs: ({
      children,
      defaultValue,
    }: React.PropsWithChildren<{ defaultValue: string }>) => {
      const [value, setValue] = ReactModule.useState(defaultValue);
      return (
        <TabsContext.Provider value={{ value, onValueChange: setValue }}>
          <div>{children}</div>
        </TabsContext.Provider>
      );
    },
    TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    TabsTrigger: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" onClick={() => ctx.onValueChange?.(value)}>
          {children}
        </button>
      );
    },
    TabsContent: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return ctx.value === value ? <div>{children}</div> : null;
    },
  };
});

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status?: string) => status ?? "unknown",
  getStatusLabel: (status: string) => status.toUpperCase(),
  getStatusTheme: () => ({ indicatorClass: "indicator", textClass: "text" }),
  isPausedStatus: (status?: string) => status === "paused",
  isTerminalStatus: (status?: string) => ["completed", "failed", "cancelled"].includes(status ?? ""),
}));

vi.mock("@/services/executionsApi", () => ({
  cancelExecution: vi.fn(),
  pauseExecution: vi.fn(),
  resumeExecution: vi.fn(),
}));

vi.mock("@/components/ui/notification", () => ({
  useSuccessNotification: () => successNotification,
  useErrorNotification: () => errorNotification,
}));

vi.mock("@/components/layout/ResponsiveGrid", () => ({
  ResponsiveGrid: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/did/DIDDisplay", () => ({
  DIDDisplay: ({ nodeId }: { nodeId: string }) => <span>{`did:${nodeId}`}</span>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/vc", () => ({
  VerifiableCredentialBadge: ({ status }: { status: string }) => <span>{`badge:${status}`}</span>,
}));

vi.mock("@/components/execution/CollapsibleSection", () => ({
  CollapsibleSection: ({ children, title }: React.PropsWithChildren<{ title: string }>) => (
    <div>
      <div>{title}</div>
      {children}
    </div>
  ),
}));

vi.mock("@/services/vcApi", () => ({
  downloadExecutionVCBundle: vi.fn(),
}));

vi.mock("@/components/ui/copy-button", () => ({
  CopyButton: ({
    value,
    children,
  }: {
    value: string;
    children?: (copied: boolean) => React.ReactNode;
  }) => (
    <button type="button" aria-label={`copy:${value}`}>
      {children ? children(false) : `copy:${value}`}
    </button>
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data }: { data: unknown }) => (
    <pre>{JSON.stringify(data)}</pre>
  ),
}));

const mockedPauseExecution = vi.mocked(pauseExecution);
const mockedResumeExecution = vi.mocked(resumeExecution);
const mockedCancelExecution = vi.mocked(cancelExecution);
const mockedDownloadExecutionVCBundle = vi.mocked(downloadExecutionVCBundle);

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    workflow_id: "wf-1",
    execution_id: "exec-1234567890",
    agentfield_request_id: "req-1",
    session_id: "session-123456",
    actor_id: "actor-123456",
    agent_node_id: "node-1",
    workflow_depth: 2,
    reasoner_id: "planner",
    input_data: { task: "inspect" },
    output_data: {},
    input_size: 1,
    output_size: 1,
    workflow_tags: [],
    status: "running",
    started_at: "2026-04-08T00:00:00Z",
    duration_ms: 1500,
    retry_count: 1,
    created_at: "2026-04-08T00:00:00Z",
    updated_at: "2026-04-08T00:01:00Z",
    ...overrides,
  };
}

describe("CompactExecutionHeader", () => {
  beforeEach(() => {
    navigate.mockReset();
    successNotification.mockReset();
    errorNotification.mockReset();
    mockedPauseExecution.mockReset();
    mockedResumeExecution.mockReset();
    mockedCancelExecution.mockReset();
  });

  it("renders execution controls, switches tabs, and handles pause/cancel actions", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const onTabChange = vi.fn();
    mockedPauseExecution.mockResolvedValue(undefined as never);
    mockedCancelExecution.mockResolvedValue(undefined as never);

    render(
      <CompactExecutionHeader
        execution={buildExecution({ error_message: "boom" }) as never}
        onRefresh={onRefresh}
        activeTab="summary"
        onTabChange={onTabChange}
        navigationTabs={[
          {
            id: "summary",
            label: "Summary",
            icon: () => <span>icon</span>,
            description: "Summary",
            shortcut: "1",
          },
          {
            id: "debug",
            label: "Debug",
            icon: () => <span>icon</span>,
            description: "Debug",
            shortcut: "2",
          },
        ]}
      />
    );

    expect(screen.getAllByText("RUNNING")[0]).toBeInTheDocument();
    expect(screen.getAllByText("planner")[0]).toBeInTheDocument();
    expect(screen.getAllByText("1 retry")[0]).toBeInTheDocument();
    expect(screen.getAllByText("1 issue")[0]).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Pause execution" })[0]);
    await waitFor(() => {
      expect(mockedPauseExecution).toHaveBeenCalledWith("exec-1234567890");
    });
    expect(successNotification).toHaveBeenCalledWith(
      "Execution paused",
      expect.stringContaining("exec-123")
    );
    expect(onRefresh).toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /Debug/i }));
    expect(onTabChange).toHaveBeenCalledWith("debug");

    await user.click(screen.getAllByRole("button", { name: "Stop execution" })[0]);
    expect(screen.getByText("Stop execution?")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Stop execution" }).at(-1)!);
    await waitFor(() => {
      expect(mockedCancelExecution).toHaveBeenCalledWith("exec-1234567890");
    });
  });

  it("handles paused status resume action and back navigation", async () => {
    const user = userEvent.setup();
    mockedResumeExecution.mockResolvedValue(undefined as never);

    render(
      <CompactExecutionHeader
        execution={buildExecution({ status: "paused" }) as never}
        activeTab="summary"
        onTabChange={vi.fn()}
        navigationTabs={[
          {
            id: "summary",
            label: "Summary",
            icon: () => <span>icon</span>,
            description: "Summary",
            shortcut: "1",
          },
        ]}
      />
    );

    await user.click(screen.getAllByRole("button", { name: "Resume execution" })[0]);
    await waitFor(() => {
      expect(mockedResumeExecution).toHaveBeenCalledWith("exec-1234567890");
    });

    await user.click(screen.getAllByRole("button", { name: "Back to executions" })[0]);
    expect(navigate).toHaveBeenCalledWith("/executions");
  });
});

describe("ExecutionIdentityPanel", () => {
  beforeEach(() => {
    mockedDownloadExecutionVCBundle.mockReset();
  });

  it("renders VC details, toggles credential JSON, and downloads the bundle", async () => {
    const user = userEvent.setup();
    mockedDownloadExecutionVCBundle.mockResolvedValue(undefined as never);

    render(
      <ExecutionIdentityPanel
        execution={buildExecution() as never}
        vcStatus={{
          has_vc: true,
          status: "verified",
          vc_id: "vc-1234567890",
          created_at: "2026-04-08T00:00:00Z",
          vc_document: {
            issuer: "did:issuer:1",
            proof: { proofValue: "proof-123" },
            credentialSubject: {
              caller: { did: "did:caller:1", agentNodeDid: "caller-node" },
              target: {
                did: "did:target:1",
                agentNodeDid: "target-node",
                functionName: "plan",
              },
              execution: {
                inputHash: "input-hash",
                outputHash: "output-hash",
                durationMs: 42,
                timestamp: "2026-04-08T00:00:00Z",
              },
            },
          },
        } as never}
      />
    );

    expect(screen.getByText("did:node-1")).toBeInTheDocument();
    expect(screen.getByText("Credential Verified")).toBeInTheDocument();
    expect(screen.getByText("badge:verified")).toBeInTheDocument();
    expect(screen.getByText("input-hash")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Download JSON" }));
    expect(mockedDownloadExecutionVCBundle).toHaveBeenCalledWith("exec-1234567890");

    await user.click(screen.getByRole("button", { name: /View Details/i }));
    expect(screen.getByText("Credential Document")).toBeInTheDocument();
    expect(screen.getAllByText(/did:issuer:1/).length).toBeGreaterThan(0);
  });

  it("renders loading and missing-credential fallback states", () => {
    const { rerender } = render(
      <ExecutionIdentityPanel
        execution={buildExecution() as never}
        vcStatus={null}
        vcLoading={true}
      />
    );

    expect(screen.getByText("Loading credential status...")).toBeInTheDocument();

    rerender(
      <ExecutionIdentityPanel
        execution={buildExecution() as never}
        vcStatus={{ has_vc: false, status: "none" } as never}
      />
    );

    expect(screen.getByText("No Verifiable Credential")).toBeInTheDocument();
    expect(screen.getByText("Unverified")).toBeInTheDocument();
  });
});

describe("ExecutionRetryPanel", () => {
  beforeEach(() => {
    navigate.mockReset();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("retries an execution successfully, shows snippets, and navigates to the new run", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ execution_id: "exec-new" }),
    } as Response);

    render(<ExecutionRetryPanel execution={buildExecution() as never} />);

    expect(screen.getByText(/curl -X POST/)).toBeInTheDocument();
    expect(screen.getByText("1 input params")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Python" })[0]);
    expect(screen.getByText(/pip install requests/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry Now" }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:3000/api/v1/execute/node-1.planner",
        expect.objectContaining({ method: "POST" })
      );
    });

    expect(await screen.findByText("Success")).toBeInTheDocument();
    expect(screen.getByText("Execution ID: exec-new")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "View" }));
    expect(navigate).toHaveBeenCalledWith("/executions/exec-new");
  });

  it("shows failure and empty-input warning states", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "server error",
    } as Response);

    render(
      <ExecutionRetryPanel
        execution={buildExecution({ input_data: {}, session_id: undefined, actor_id: undefined }) as never}
      />
    );

    expect(screen.getByText("Empty input")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry Now" }));
    expect(await screen.findByText("Error")).toBeInTheDocument();
    expect(screen.getByText(/500 - server error/)).toBeInTheDocument();
  });
});

describe("EnhancedNotesSection", () => {
  it("returns null without notes and supports sorting, refresh, and expansion", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn().mockResolvedValue(undefined);
    const longMessage = "A".repeat(170);

    const firstRender = render(
      <EnhancedNotesSection execution={buildExecution({ notes: [] }) as never} />
    );
    expect(firstRender.container).toBeEmptyDOMElement();
    firstRender.unmount();

    render(
      <EnhancedNotesSection
        execution={
          buildExecution({
            notes: [
              {
                message: "Later event",
                tags: ["later"],
                timestamp: "2026-04-08T11:00:00Z",
              },
              {
                message: longMessage,
                tags: ["alpha", "beta"],
                timestamp: "2026-04-08T10:00:00Z",
              },
              {
                message: "Oldest event",
                tags: [],
                timestamp: "2026-04-08T09:00:00Z",
              },
              {
                message: "Very old",
                tags: [],
                timestamp: "2026-04-07T09:00:00Z",
              },
            ],
          }) as never
        }
        onRefresh={onRefresh}
      />
    );

    expect(screen.getByText("Execution Events")).toBeInTheDocument();
    expect(screen.getByText("4 Events")).toBeInTheDocument();
    expect(screen.getByText("Later event")).toBeInTheDocument();
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText(/Sorted by newest first/)).toBeInTheDocument();

    await user.click(screen.getByTitle("Newest first"));
    expect(screen.getByTitle("Oldest first")).toBeInTheDocument();

    await user.click(screen.getByTitle("Refresh notes"));
    expect(onRefresh).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /Show more/i }));
    expect(screen.getByText(longMessage)).toBeInTheDocument();
  });
});
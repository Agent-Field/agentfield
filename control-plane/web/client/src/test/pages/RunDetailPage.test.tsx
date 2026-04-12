import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { RunDetailPage } from "@/pages/RunDetailPage";
import type { WorkflowDAGLightweightResponse } from "@/types/workflows";

const state = vi.hoisted(() => ({
  runDag: {
    data: undefined as WorkflowDAGLightweightResponse | undefined,
    isLoading: false,
    isError: false,
    error: null as Error | null,
  },
  queryData: undefined as any,
  invalidateQueries: vi.fn<(args: unknown) => Promise<void>>(),
  cancelMutateAsync: vi.fn<(executionId: string) => Promise<void>>(),
  pauseMutateAsync: vi.fn<(executionId: string) => Promise<void>>(),
  resumeMutateAsync: vi.fn<(executionId: string) => Promise<void>>(),
  showRunNotification: vi.fn<(message: string) => void>(),
  getExecutionDetails: vi.fn<(executionId: string) => Promise<{ input_data: unknown }>>(),
  retryExecutionWebhook: vi.fn<(executionId: string) => Promise<void>>(),
  getWorkflowVCChain: vi.fn<(workflowId: string) => Promise<any>>(),
  downloadWorkflowVCAuditFile: vi.fn<(workflowId: string) => Promise<void>>(),
  navigateSpy: vi.fn(),
}));

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return {
    ...actual,
    useNavigate: () => state.navigateSpy,
  };
});

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryFn, enabled = true }: { queryFn: () => Promise<unknown>; enabled?: boolean }) => {
    if (enabled && state.queryData === undefined) {
      void queryFn();
    }
    return { data: state.queryData };
  },
  useQueryClient: () => ({
    invalidateQueries: state.invalidateQueries,
  }),
}));

vi.mock("@/hooks/queries", () => ({
  useRunDAG: () => state.runDag,
  useCancelExecution: () => ({ mutateAsync: state.cancelMutateAsync, isPending: false }),
  usePauseExecution: () => ({ mutateAsync: state.pauseMutateAsync, isPending: false }),
  useResumeExecution: () => ({ mutateAsync: state.resumeMutateAsync, isPending: false }),
}));

vi.mock("@/components/ui/notification", () => ({
  useRunNotification: () => state.showRunNotification,
}));

vi.mock("@/services/executionsApi", () => ({
  retryExecutionWebhook: (executionId: string) => state.retryExecutionWebhook(executionId),
  getExecutionDetails: (executionId: string) => state.getExecutionDetails(executionId),
}));

vi.mock("@/services/vcApi", () => ({
  getWorkflowVCChain: (workflowId: string) => state.getWorkflowVCChain(workflowId),
  downloadWorkflowVCAuditFile: (workflowId: string) => state.downloadWorkflowVCAuditFile(workflowId),
}));

vi.mock("@/components/runs/RunLifecycleMenu", () => ({
  CANCEL_RUN_COPY: {
    title: (count: number) => `Cancel ${count} run`,
    description: "Cancel description",
    confirmLabel: (count: number) => `Cancel ${count} run`,
    keepLabel: "Keep running",
    success: "Cancelled",
    error: "Cancel failed",
  },
}));

vi.mock("@/components/RunTrace", () => ({
  buildTraceTree: (timeline: Array<{ execution_id: string }>) => ({ execution_id: timeline[0]?.execution_id ?? "none" }),
  formatDuration: (duration: number) => `${duration}ms`,
  RunTrace: ({
    selectedId,
    onSelect,
  }: {
    selectedId: string | null;
    onSelect: (value: string) => void;
  }) => (
    <div>
      <div>Trace {selectedId ?? "none"}</div>
      <button type="button" onClick={() => onSelect("exec-2")}>
        Select trace exec-2
      </button>
    </div>
  ),
}));

vi.mock("@/components/StepDetail", () => ({
  StepDetail: ({ executionId }: { executionId: string }) => <div>Step {executionId}</div>,
}));

vi.mock("@/components/WorkflowDAG", () => ({
  WorkflowDAGViewer: ({
    selectedNodeIds,
    onExecutionClick,
  }: {
    selectedNodeIds?: string[];
    onExecutionClick?: (execution: { execution_id: string }) => void;
  }) => (
    <div>
      <div>Graph {selectedNodeIds?.join(",") ?? "none"}</div>
      <button type="button" onClick={() => onExecutionClick?.({ execution_id: "exec-2" })}>
        Graph select exec-2
      </button>
    </div>
  ),
}));

vi.mock("@/components/execution", () => ({
  ExecutionObservabilityPanel: ({ execution }: { execution: { execution_id: string } }) => (
    <div>Logs {execution.execution_id}</div>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/tabs", async () => {
  const ReactModule = await import("react");
  const TabsContext = ReactModule.createContext<{
    value: string;
    onValueChange?: (value: string) => void;
  }>({ value: "" });

  return {
    Tabs: ({
      children,
      value,
      onValueChange,
    }: React.PropsWithChildren<{ value: string; onValueChange?: (value: string) => void }>) => (
      <TabsContext.Provider value={{ value, onValueChange }}>
        <div>{children}</div>
      </TabsContext.Provider>
    ),
    TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    TabsTrigger: ({
      children,
      value,
      ...props
    }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" onClick={() => ctx.onValueChange?.(value)} {...props}>
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

vi.mock("@/components/ui/alert-dialog", () => ({
  AlertDialog: ({
    children,
    open,
  }: React.PropsWithChildren<{ open?: boolean }>) => (open ? <div>{children}</div> : null),
  AlertDialogAction: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => <button type="button" {...props}>{children}</button>,
  AlertDialogCancel: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => <button type="button" {...props}>{children}</button>,
  AlertDialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
  }: React.PropsWithChildren<{ onClick?: () => void }>) => <button type="button" onClick={onClick}>{children}</button>,
  DropdownMenuLabel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuSeparator: () => <div>separator</div>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>loading</div>,
}));

vi.mock("@/components/ui/copy-identifier-chip", () => ({
  CopyIdentifierChip: ({ label, value }: { label: string; value: string }) => <span>{label}:{value}</span>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ErrorBoundary", () => ({
  ErrorBoundary: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/status-pill", () => ({
  StatusPill: ({ status }: { status: string }) => <span>Status {status}</span>,
}));

vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>();
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    ...actual,
    Activity: Icon,
    BadgeCheck: Icon,
    ChevronDown: Icon,
    FileJson: Icon,
    FileCheck2: Icon,
    Info: Icon,
    Link2: Icon,
    PauseCircle: Icon,
    Play: Icon,
    RefreshCw: Icon,
    RotateCcw: Icon,
    XCircle: Icon,
  };
});

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/runs/run-1"]}>
      <Routes>
        <Route path="/runs/:runId" element={<RunDetailPage />} />
      </Routes>
    </MemoryRouter>
  );
}

function buildDag(): WorkflowDAGLightweightResponse {
  return {
    root_workflow_id: "wf-1",
    workflow_status: "running",
    workflow_name: "Run Alpha",
    session_id: "session-1",
    actor_id: "actor-1",
    total_nodes: 2,
    max_depth: 1,
    mode: "lightweight",
    unique_agent_node_ids: ["node-1", "node-2"],
    timeline: [
      {
        execution_id: "exec-1",
        agent_node_id: "node-1",
        reasoner_id: "planner",
        status: "running",
        started_at: "2026-04-08T00:00:00Z",
        duration_ms: 500,
        workflow_depth: 0,
      },
      {
        execution_id: "exec-2",
        parent_execution_id: "exec-1",
        agent_node_id: "node-2",
        reasoner_id: "worker",
        status: "failed",
        started_at: "2026-04-08T00:00:01Z",
        completed_at: "2026-04-08T00:00:02Z",
        duration_ms: 300,
        workflow_depth: 1,
      },
    ],
    webhook_summary: {
      steps_with_webhook: 1,
      total_deliveries: 1,
      failed_deliveries: 1,
    },
    webhook_failures: [
      {
        execution_id: "exec-2",
        agent_node_id: "node-2",
        reasoner_id: "worker",
        event_type: "completed",
        http_status: 500,
      },
    ],
    workflow_issuer_did: "did:example:issuer",
  };
}

describe("RunDetailPage", () => {
  beforeEach(() => {
    state.runDag = {
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
    };
    state.queryData = undefined;
    state.invalidateQueries.mockReset();
    state.cancelMutateAsync.mockReset();
    state.pauseMutateAsync.mockReset();
    state.resumeMutateAsync.mockReset();
    state.showRunNotification.mockReset();
    state.getExecutionDetails.mockReset();
    state.retryExecutionWebhook.mockReset();
    state.getWorkflowVCChain.mockReset();
    state.downloadWorkflowVCAuditFile.mockReset();
    state.navigateSpy.mockReset();
  });

  it("renders loading state then populated trace/log surfaces and replay action", async () => {
    state.runDag = {
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    };
    state.queryData = { workflow_vc: { issuer_did: "did:example:issuer" } };
    state.getWorkflowVCChain.mockResolvedValue({ workflow_vc: { issuer_did: "did:example:issuer" } });
    state.getExecutionDetails.mockResolvedValue({ input_data: { replay: true } });

    const view = renderPage();
    expect(screen.getAllByText("loading").length).toBeGreaterThan(0);

    state.runDag = {
      data: buildDag(),
      isLoading: false,
      isError: false,
      error: null,
    };
    view.rerender(
      <MemoryRouter initialEntries={["/runs/run-1"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByText("Run Alpha")).toBeInTheDocument();
    expect(screen.getByText("Trace exec-1")).toBeInTheDocument();
    expect(screen.getByText("Step exec-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Logs"));
    expect(await screen.findByText("Logs exec-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Replay"));
    await waitFor(() => {
      expect(state.getExecutionDetails).toHaveBeenCalledWith("exec-1");
    });
    expect(state.navigateSpy).toHaveBeenCalledWith("/playground/node-1.planner", {
      state: { replayInput: { replay: true } },
    });
  });

  it("supports graph selection and webhook retry strip interactions", async () => {
    state.runDag = {
      data: buildDag(),
      isLoading: false,
      isError: false,
      error: null,
    };
    state.queryData = { workflow_vc: { issuer_did: "did:example:issuer" } };
    state.retryExecutionWebhook.mockResolvedValue();

    renderPage();

    expect(await screen.findByText("Run Alpha")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Graph"));
    expect(await screen.findByText("Graph exec-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Graph select exec-2"));
    expect(await screen.findByText("Graph exec-2")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Retry"));
    await waitFor(() => {
      expect(state.retryExecutionWebhook).toHaveBeenCalledWith("exec-2");
    });
    await waitFor(() => {
      expect(state.invalidateQueries).toHaveBeenCalled();
    });
  });

  it("renders explicit error and empty states", async () => {
    state.runDag = {
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("run detail failed"),
    };

    const view = renderPage();
    expect(await screen.findByText("run detail failed")).toBeInTheDocument();

    state.runDag = {
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
    };

    view.rerender(
      <MemoryRouter initialEntries={["/runs/run-1"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByText("No data available for this run.")).toBeInTheDocument();
  });

  it("supports export actions and single-step execution view", async () => {
    state.runDag = {
      data: {
        ...buildDag(),
        total_nodes: 1,
        max_depth: 0,
        timeline: [buildDag().timeline[0]],
        webhook_failures: [],
        webhook_summary: {
          steps_with_webhook: 0,
          total_deliveries: 0,
          failed_deliveries: 0,
        },
      },
      isLoading: false,
      isError: false,
      error: null,
    };
    state.queryData = { workflow_vc: { issuer_did: "did:example:issuer" } };
    state.getWorkflowVCChain.mockResolvedValue({ workflow_vc: { issuer_did: "did:example:issuer" } });
    state.downloadWorkflowVCAuditFile.mockResolvedValue(undefined);

    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const originalCreateObjectUrl = URL.createObjectURL;
    const originalRevokeObjectUrl = URL.revokeObjectURL;
    const createObjectUrlSpy = vi.fn(() => "blob:preview");
    const revokeObjectUrlSpy = vi.fn();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: createObjectUrlSpy,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: revokeObjectUrlSpy,
    });

    renderPage();

    expect(await screen.findByText("Run Alpha")).toBeInTheDocument();
    expect(screen.getByText("Step exec-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Preview VC chain"));
    await waitFor(() => {
      expect(openSpy).toHaveBeenCalledWith("blob:preview", "_blank", "noopener,noreferrer");
    });

    fireEvent.click(screen.getByText("Download VC audit JSON"));
    await waitFor(() => {
      expect(state.downloadWorkflowVCAuditFile).toHaveBeenCalledWith("wf-1");
    });

    expect(createObjectUrlSpy).toHaveBeenCalled();
    expect(revokeObjectUrlSpy).not.toHaveBeenCalled();

    openSpy.mockRestore();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: originalCreateObjectUrl,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: originalRevokeObjectUrl,
    });
  });

  it("runs pause, resume, and cancel lifecycle actions with notifications", async () => {
    state.runDag = {
      data: buildDag(),
      isLoading: false,
      isError: false,
      error: null,
    };
    state.pauseMutateAsync.mockResolvedValue(undefined);
    state.cancelMutateAsync.mockRejectedValue(new Error("cancel exploded"));

    const view = renderPage();
    expect(await screen.findByText("Run Alpha")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Pause"));
    await waitFor(() => {
      expect(state.pauseMutateAsync).toHaveBeenCalledWith("exec-1");
    });
    expect(state.showRunNotification).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Paused", runId: "run-1" }),
    );

    fireEvent.click(screen.getByText("Cancel"));
    fireEvent.click(screen.getByRole("button", { name: "Cancel 1 run" }));
    await waitFor(() => {
      expect(state.cancelMutateAsync).toHaveBeenCalledWith("exec-1");
    });
    expect(state.showRunNotification).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Cancel failed",
        message: "cancel exploded",
      }),
    );

    state.runDag = {
      data: {
        ...buildDag(),
        workflow_status: "paused",
        timeline: [
          {
            ...buildDag().timeline[0],
            status: "paused",
          },
          {
            ...buildDag().timeline[1],
            status: "running",
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    };
    state.resumeMutateAsync.mockResolvedValue(undefined);

    view.rerender(
      <MemoryRouter initialEntries={["/runs/run-1"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByText("Pause registered")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Resume"));
    await waitFor(() => {
      expect(state.resumeMutateAsync).toHaveBeenCalledWith("exec-1");
    });
    expect(state.showRunNotification).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Resumed" }),
    );
  });

  it("shows cancellation strip when the root is cancelled but children are still running", async () => {
    state.runDag = {
      data: {
        ...buildDag(),
        workflow_status: "cancelled",
        timeline: [
          {
            ...buildDag().timeline[0],
            status: "cancelled",
          },
          {
            ...buildDag().timeline[1],
            status: "running",
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    renderPage();

    expect(await screen.findByText("Cancellation registered")).toBeInTheDocument();
    expect(
      screen.getByText(/No new nodes will start; their output will be discarded\./),
    ).toBeInTheDocument();
  });

  it("renders reasoner fallback metadata and empty webhook state when agent ids are absent", async () => {
    state.runDag = {
      data: {
        ...buildDag(),
        workflow_name: "   ",
        unique_agent_node_ids: [],
        workflow_issuer_did: "   ",
        session_id: undefined,
        actor_id: undefined,
        timeline: [
          {
            ...buildDag().timeline[0],
            agent_node_id: "   ",
            reasoner_id: "planner-fallback",
          },
          {
            ...buildDag().timeline[1],
            agent_node_id: "   ",
            reasoner_id: "worker-fallback",
          },
        ],
        webhook_summary: {
          steps_with_webhook: 0,
          total_deliveries: 0,
          failed_deliveries: 0,
        },
        webhook_failures: [],
      },
      isLoading: false,
      isError: false,
      error: null,
    };
    state.queryData = { workflow_vc: { issuer_did: "did:example:vc-chain" } };

    renderPage();

    expect(await screen.findByRole("heading", { name: "planner-fallback" })).toBeInTheDocument();
    expect(screen.getByText("Identity:did:example:vc-chain")).toBeInTheDocument();
    expect(screen.getByText("Reasoners")).toBeInTheDocument();
    expect(screen.getByText("worker-fallback")).toBeInTheDocument();
    expect(
      screen.getByText("No outbound webhooks—register a webhook URL on the reasoner to receive callbacks."),
    ).toBeInTheDocument();
  });

  it("renders empty execution detail states and pending webhook registrations", async () => {
    state.runDag = {
      data: {
        ...buildDag(),
        total_nodes: 0,
        max_depth: 0,
        unique_agent_node_ids: [],
        timeline: [],
        webhook_summary: {
          steps_with_webhook: 2,
          total_deliveries: 0,
          failed_deliveries: 0,
        },
        webhook_failures: [],
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    renderPage();

    expect(await screen.findByText("No agent or reasoner identifiers on this run.")).toBeInTheDocument();
    expect(screen.getByText("2 steps registered for callbacks — no delivery attempts recorded yet.")).toBeInTheDocument();
    expect(screen.getByText("No step selected")).toBeInTheDocument();
  });

  it("supports retry-all webhook failures and step selection from the failure list", async () => {
    state.runDag = {
      data: {
        ...buildDag(),
        webhook_summary: {
          steps_with_webhook: 2,
          total_deliveries: 2,
          failed_deliveries: 2,
        },
        webhook_failures: [
          buildDag().webhook_failures![0],
          {
            execution_id: "exec-3",
            agent_node_id: "",
            reasoner_id: "",
            event_type: "started",
            http_status: 502,
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    };
    state.retryExecutionWebhook
      .mockRejectedValueOnce(new Error("retry-all failed"))
      .mockResolvedValueOnce(undefined);

    renderPage();

    expect(await screen.findByText("Retry all")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Retry all"));
    expect(await screen.findByText("retry-all failed")).toBeInTheDocument();

    fireEvent.click(screen.getAllByText("Step")[1]);
    expect(await screen.findByText("Step exec-3")).toBeInTheDocument();
  });
});

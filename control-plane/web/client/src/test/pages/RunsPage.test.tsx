// @ts-nocheck
import * as React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  navigateMock,
  searchParamsState,
  useRunsMock,
  cancelMutationMock,
  pauseMutationMock,
  resumeMutationMock,
  useQueryMock,
  showSuccessMock,
  showErrorMock,
  showWarningMock,
  showRunNotificationMock,
  clipboardWriteTextMock,
} = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  searchParamsState: { value: new URLSearchParams() },
  useRunsMock: vi.fn(),
  cancelMutationMock: vi.fn(),
  pauseMutationMock: vi.fn(),
  resumeMutationMock: vi.fn(),
  useQueryMock: vi.fn(),
  showSuccessMock: vi.fn(),
  showErrorMock: vi.fn(),
  showWarningMock: vi.fn(),
  showRunNotificationMock: vi.fn(),
  clipboardWriteTextMock: vi.fn(),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
  useSearchParams: () => [searchParamsState.value, vi.fn()],
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
}));

vi.mock("@/services/executionsApi", () => ({
  getExecutionDetails: vi.fn(),
}));

vi.mock("@/hooks/queries", () => ({
  useRuns: useRunsMock,
  useCancelExecution: () => ({ mutateAsync: cancelMutationMock }),
  usePauseExecution: () => ({ mutateAsync: pauseMutationMock }),
  useResumeExecution: () => ({ mutateAsync: resumeMutationMock }),
}));

vi.mock("@/components/ui/notification", () => ({
  useSuccessNotification: () => showSuccessMock,
  useErrorNotification: () => showErrorMock,
  useWarningNotification: () => showWarningMock,
  useRunNotification: () => showRunNotificationMock,
}));

vi.mock("@/components/ui/sidebar", () => ({
  useSidebar: () => ({ state: "expanded", isMobile: false }),
}));

vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>();
  const ReactModule = await vi.importActual<typeof import("react")>("react");
  const makeIcon = (name: string) =>
    ReactModule.forwardRef<SVGSVGElement, { className?: string }>((props, ref) => (
      <svg ref={ref} data-testid={name} {...props} />
    ));

  return {
    ...actual,
    ArrowDown: makeIcon("arrow-down"),
    ArrowLeftRight: makeIcon("arrow-left-right"),
    ArrowUp: makeIcon("arrow-up"),
    Check: makeIcon("check"),
    Copy: makeIcon("copy"),
    Play: makeIcon("play"),
  };
});

vi.mock("@/components/ui/alert-dialog", () => ({
  AlertDialog: ({ open, children }: { open?: boolean; children: React.ReactNode }) =>
    open ? <div data-testid="alert-dialog">{children}</div> : null,
  AlertDialogAction: ({ children, onClick, disabled, className }: any) => (
    <button type="button" className={className} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  ),
  AlertDialogCancel: ({ children, onClick, disabled }: any) => (
    <button type="button" disabled={disabled} onClick={onClick}>
      {children}
    </button>
  ),
  AlertDialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/components/runs/RunLifecycleMenu", () => ({
  CANCEL_RUN_COPY: {
    title: (count: number) => (count > 1 ? `Cancel ${count} runs?` : "Cancel this run?"),
    description:
      "Nodes currently executing will finish their current step — only pending nodes will be stopped. Any in-flight work will be discarded. This cannot be undone.",
    confirmLabel: (count: number) => (count > 1 ? `Cancel ${count} runs` : "Cancel run"),
    keepLabel: "Keep running",
  },
  RunLifecycleMenu: ({
    run,
    onPause,
    onResume,
    onCancel,
  }: {
    run: (typeof baseRuns)[number];
    onPause: (run: (typeof baseRuns)[number]) => void;
    onResume: (run: (typeof baseRuns)[number]) => void;
    onCancel: (run: (typeof baseRuns)[number]) => void;
  }) => (
    <div data-testid={`run-lifecycle-menu-${run.run_id}`}>
      <button type="button" onClick={() => onPause(run)}>
        Pause {run.run_id}
      </button>
      <button type="button" onClick={() => onResume(run)}>
        Resume {run.run_id}
      </button>
      <button type="button" onClick={() => onCancel(run)}>
        Cancel {run.run_id}
      </button>
    </div>
  ),
}));

vi.mock("@/components/ui/status-pill", () => ({
  StatusDot: ({ status }: { status: string }) => <span>{status}</span>,
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: { children: React.ReactNode }) => <table>{children}</table>,
  TableBody: ({ children }: { children: React.ReactNode }) => <tbody>{children}</tbody>,
  TableCell: ({ children, ...props }: any) => <td {...props}>{children}</td>,
  TableHead: ({ children, ...props }: any) => <th {...props}>{children}</th>,
  TableHeader: ({ children }: { children: React.ReactNode }) => <thead>{children}</thead>,
  TableRow: ({ children, ...props }: any) => <tr {...props}>{children}</tr>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, disabled, type = "button", ...props }: any) => (
    <button type={type} onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  badgeVariants: () => "",
}));

vi.mock("@/components/ui/checkbox", () => ({
  Checkbox: ({ checked, onCheckedChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={Boolean(checked)}
      onChange={() => onCheckedChange?.(!checked)}
      {...props}
    />
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, variant: _variant, interactive: _interactive, ...props }: any) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/filter-combobox", () => ({
  FilterCombobox: ({ label, value, onValueChange, options }: any) => (
    <label>
      {label}
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
      >
        {options.map((option: { value: string; label: string }) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  ),
}));

vi.mock("@/components/ui/filter-multi-combobox", () => ({
  FilterMultiCombobox: ({ label, options, selected, onSelectedChange }: any) => (
    <div aria-label={label} role="group">
      {options.map((option: { value: string; label: string }) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={selected.has(option.value)}
          onClick={() =>
            onSelectedChange((prev: Set<string>) => {
              const next = new Set(prev);
              if (next.has(option.value)) {
                next.delete(option.value);
              } else {
                next.add(option.value);
              }
              return next;
            })
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  ),
}));

vi.mock("@/components/ui/SearchBar", () => ({
  SearchBar: ({ value, onChange, placeholder, "aria-label": ariaLabel }: any) => (
    <input
      value={value}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: (props: any) => <div {...props} />,
}));

vi.mock("@/components/ui/hover-card", () => ({
  HoverCard: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  HoverCardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: (props: any) => <div {...props} />,
}));

vi.mock("@/components/ui/pagination", () => ({
  Pagination: ({ children, ...props }: any) => <nav {...props}>{children}</nav>,
  PaginationContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  PaginationEllipsis: () => <span>…</span>,
  PaginationItem: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  PaginationLink: ({ children, onClick, ...props }: any) => (
    <button type="button" onClick={onClick} aria-label={props["aria-label"]} disabled={props.disabled}>
      {children}
    </button>
  ),
  PaginationNext: ({ onClick, disabled }: any) => (
    <button type="button" disabled={disabled} onClick={onClick}>
      Next
    </button>
  ),
  PaginationPrevious: ({ onClick, disabled }: any) => (
    <button type="button" disabled={disabled} onClick={onClick}>
      Previous
    </button>
  ),
}));

vi.mock("@/components/ui/select", () => ({
  Select: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectItem: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectTrigger: ({ children, className, "aria-label": ariaLabel }: any) => (
    <button type="button" className={className} aria-label={ariaLabel}>
      {children}
    </button>
  ),
  SelectValue: () => <span />,
}));

vi.mock("@/components/ui/CompactTable", () => ({
  SortableHeaderCell: ({ label, onSortChange, field }: any) => (
    <button type="button" onClick={() => onSortChange(field)}>
      {label}
    </button>
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  formatTruncatedFormattedJson: (value: unknown) =>
    value == null ? "—" : JSON.stringify(value, null, 2),
  JsonHighlightedPre: ({ text }: { text: string }) => <pre>{text}</pre>,
}));

import { RunsPage } from "@/pages/RunsPage";

const baseRuns = [
  {
    run_id: "run-001-alpha",
    workflow_id: "wf-1",
    root_execution_id: "exec-1",
    root_execution_status: "running",
    status: "running",
    root_reasoner: "alpha",
    current_task: "task-a",
    total_executions: 3,
    max_depth: 1,
    started_at: "2026-04-08T12:00:00Z",
    latest_activity: "2026-04-08T12:00:00Z",
    display_name: "Alpha",
    agent_id: "agent-one",
    agent_name: "Agent One",
    status_counts: {},
    active_executions: 1,
    terminal: false,
  },
  {
    run_id: "run-002-beta",
    workflow_id: "wf-2",
    root_execution_id: "exec-2",
    root_execution_status: "paused",
    status: "paused",
    root_reasoner: "beta",
    current_task: "task-b",
    total_executions: 5,
    max_depth: 2,
    started_at: "2026-04-08T11:00:00Z",
    latest_activity: "2026-04-08T11:05:00Z",
    display_name: "Beta",
    agent_id: "agent-two",
    agent_name: "Agent Two",
    status_counts: {},
    active_executions: 0,
    terminal: false,
  },
  {
    run_id: "run-003-gamma",
    workflow_id: "wf-3",
    root_execution_id: "exec-3",
    root_execution_status: "succeeded",
    status: "succeeded",
    root_reasoner: "gamma",
    current_task: "task-c",
    total_executions: 2,
    max_depth: 1,
    started_at: "2026-04-08T10:00:00Z",
    latest_activity: "2026-04-08T10:02:00Z",
    completed_at: "2026-04-08T10:03:00Z",
    duration_ms: 180000,
    display_name: "Gamma",
    agent_id: "agent-three",
    agent_name: "Agent Three",
    status_counts: {},
    active_executions: 0,
    terminal: true,
  },
];

function buildRunsResult(filters?: Record<string, unknown>) {
  const search = String(filters?.search ?? "").trim().toLowerCase();
  const status = String(filters?.status ?? "").trim().toLowerCase();

  let workflows = [...baseRuns];

  if (status) {
    workflows = workflows.filter((run) => (run.root_execution_status ?? run.status) === status);
  }

  if (search) {
    workflows = workflows.filter((run) =>
      [run.run_id, run.root_reasoner, run.display_name, run.agent_id, run.agent_name]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(search),
    );
  }

  return {
    workflows,
    total_count: workflows.length,
    total_pages: workflows.length > 0 ? 1 : 0,
  };
}

describe("RunsPage", () => {
  beforeEach(() => {
    vi.useRealTimers();
    navigateMock.mockReset();
    useRunsMock.mockReset();
    cancelMutationMock.mockReset();
    pauseMutationMock.mockReset();
    resumeMutationMock.mockReset();
    useQueryMock.mockReset();
    showSuccessMock.mockReset();
    showErrorMock.mockReset();
    showWarningMock.mockReset();
    showRunNotificationMock.mockReset();
    clipboardWriteTextMock.mockReset();
    searchParamsState.value = new URLSearchParams();

    useQueryMock.mockReturnValue({ data: undefined, isLoading: false });
    cancelMutationMock.mockResolvedValue(undefined);
    pauseMutationMock.mockResolvedValue(undefined);
    resumeMutationMock.mockResolvedValue(undefined);
    useRunsMock.mockImplementation((filters?: Record<string, unknown>) => ({
      data: buildRunsResult(filters),
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
    }));

    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: clipboardWriteTextMock,
      },
    });
    clipboardWriteTextMock.mockResolvedValue(undefined);
  });

  it("renders the empty state when no runs are returned", () => {
    useRunsMock.mockReturnValue({
      data: {
        workflows: [],
        total_count: 0,
        total_pages: 0,
      },
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
    });

    render(<RunsPage />);

    expect(screen.getByText("No runs found")).toBeInTheDocument();
    expect(screen.getByText("Execute a reasoner to create your first run")).toBeInTheDocument();
  });

  it("renders rows, debounces search, filters by status, navigates on row click, and copies a run id", async () => {
    const user = userEvent.setup();

    render(<RunsPage />);

    expect(screen.getByText("agent-one.")).toBeInTheDocument();
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.getByText("gamma")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Copy run ID run-001-alpha" }));
    expect(await screen.findByRole("button", { name: "Copied!" })).toBeInTheDocument();

    await user.click(screen.getByText("alpha"));
    expect(navigateMock).toHaveBeenCalledWith("/runs/run-001-alpha");

    const searchInput = screen.getByPlaceholderText("Search runs, reasoners, agents…");
    await user.clear(searchInput);
    await user.type(searchInput, "beta");

    await waitFor(() => {
      expect(useRunsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ search: "beta" }),
      );
    });
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await waitFor(() => {
      expect(screen.getByText("alpha")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Succeeded" }));
    await waitFor(() => {
      expect(useRunsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ status: "succeeded" }),
      );
    });
    expect(screen.getByText("gamma")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
  });

  it("supports selecting rows, compare navigation, and bulk cancel", async () => {
    const user = userEvent.setup();

    render(<RunsPage />);

    await user.click(screen.getByLabelText("Select run run-001-alpha").closest("td")!);
    await user.click(screen.getByLabelText("Select run run-002-beta").closest("td")!);

    expect(
      screen.getByRole("toolbar", { name: "Bulk actions for selected runs" }),
    ).toBeInTheDocument();
    const compareButton = screen.getByRole("button", { name: "Compare selected (2)" });
    expect(compareButton).toBeEnabled();

    await user.click(compareButton);
    expect(navigateMock).toHaveBeenCalledWith("/runs/compare?a=run-001-alpha&b=run-002-beta");

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByText("Cancel 2 runs?")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel 2 runs" }));

    await waitFor(() => {
      expect(cancelMutationMock).toHaveBeenCalledTimes(2);
    });
    expect(cancelMutationMock).toHaveBeenCalledWith("exec-1");
    expect(cancelMutationMock).toHaveBeenCalledWith("exec-2");
    expect(showSuccessMock).toHaveBeenCalledWith(
      "2 runs cancelled",
      "All selected runs were cancelled successfully.",
    );
  });

  it("renders API errors from the runs query", () => {
    useRunsMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: new Error("runs query failed"),
    });

    render(<RunsPage />);

    expect(screen.getByText("runs query failed")).toBeInTheDocument();
  });

  it("applies multi-status filtering client-side without sending a single API status", async () => {
    const user = userEvent.setup();

    render(<RunsPage />);

    await user.click(screen.getByRole("button", { name: "Running" }));
    await user.click(screen.getByRole("button", { name: "Succeeded" }));

    await waitFor(() => {
      expect(useRunsMock).toHaveBeenLastCalledWith(
        expect.not.objectContaining({ status: expect.anything() }),
      );
    });

    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("gamma")).toBeInTheDocument();
    expect(screen.queryByText("beta")).not.toBeInTheDocument();
  });

  it("cancels pending debounce work when filters are cleared", () => {
    vi.useFakeTimers();

    render(<RunsPage />);

    const searchInput = screen.getByPlaceholderText("Search runs, reasoners, agents…");
    fireEvent.change(searchInput, { target: { value: "beta" } });

    expect(useRunsMock).toHaveBeenLastCalledWith(
      expect.not.objectContaining({ search: "beta" }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    act(() => {
      vi.advanceTimersByTime(350);
    });

    expect(useRunsMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ search: undefined, page: 1 }),
    );
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.getByText("gamma")).toBeInTheDocument();
  });

  it("navigates from row keyboard interaction", () => {
    render(<RunsPage />);

    const row = screen.getByText("alpha").closest("tr");
    expect(row).not.toBeNull();

    act(() => {
      row?.focus();
    });
    act(() => {
      row?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });

    expect(navigateMock).toHaveBeenCalledWith("/runs/run-001-alpha");
  });

  it("shows singular bulk cancel copy and skips terminal runs", async () => {
    const user = userEvent.setup();

    render(<RunsPage />);

    await user.click(screen.getByLabelText("Select run run-001-alpha").closest("td")!);
    await user.click(screen.getByLabelText("Select run run-003-gamma").closest("td")!);
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.getByText("Cancel this run?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel run" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel run" }));

    await waitFor(() => {
      expect(cancelMutationMock).toHaveBeenCalledTimes(1);
    });
    expect(cancelMutationMock).toHaveBeenCalledWith("exec-1");
    expect(cancelMutationMock).not.toHaveBeenCalledWith("exec-3");
    expect(showSuccessMock).toHaveBeenCalledWith("1 run cancelled", undefined);
  });

  it("renders preview variants and lifecycle notifications for row actions", async () => {
    const user = userEvent.setup();

    useQueryMock.mockImplementation(({ queryKey }: { queryKey: string[] }) => {
      const execId = queryKey[1];
      if (execId === "exec-1") {
        return {
          data: { input_data: { foo: "bar" }, output_data: { ok: true } },
          isLoading: false,
        };
      }
      if (execId === "exec-2") {
        return {
          data: { input_data: { only: "input" }, output_data: null },
          isLoading: false,
        };
      }
      return {
        data: { input_data: null, output_data: { only: "output" } },
        isLoading: false,
      };
    });
    pauseMutationMock.mockRejectedValueOnce(new Error("pause denied"));
    resumeMutationMock.mockResolvedValueOnce(undefined);
    cancelMutationMock.mockResolvedValueOnce(undefined);

    render(<RunsPage />);

    expect(screen.getByRole("region", { name: "Input and output preview" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Input preview" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Output preview" })).toBeInTheDocument();
    expect(screen.getByText("Open run for full JSON and trace.")).toBeInTheDocument();
    expect(screen.getByText("Open run for output and full trace.")).toBeInTheDocument();
    expect(screen.getByText("Open run for full trace.")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Copy Input" })[0]);

    await user.click(screen.getByRole("button", { name: "Pause run-001-alpha" }));
    await waitFor(() => {
      expect(showRunNotificationMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Pause failed",
          message: "pause denied",
          runId: "run-001-alpha",
        }),
      );
    });

    await user.click(screen.getByRole("button", { name: "Resume run-002-beta" }));
    await waitFor(() => {
      expect(showRunNotificationMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Resumed",
          runId: "run-002-beta",
        }),
      );
    });

    await user.click(screen.getByRole("button", { name: "Cancel run-001-alpha" }));
    await waitFor(() => {
      expect(showRunNotificationMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Cancelled",
          runId: "run-001-alpha",
        }),
      );
    });
  });

  it("runs bulk pause and resume actions", async () => {
    const user = userEvent.setup();

    render(<RunsPage />);

    await user.click(screen.getByLabelText("Select run run-001-alpha").closest("td")!);
    await user.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => {
      expect(pauseMutationMock).toHaveBeenCalledWith("exec-1");
    });
    expect(showSuccessMock).toHaveBeenCalledWith("1 run paused", undefined);

    await user.click(screen.getByLabelText("Select run run-001-alpha").closest("td")!);
    await user.click(screen.getByLabelText("Select run run-002-beta").closest("td")!);
    await user.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => {
      expect(resumeMutationMock).toHaveBeenCalledWith("exec-2");
    });
    expect(showSuccessMock).toHaveBeenCalledWith("1 run resumed", undefined);
  });
});

// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionObservabilityPanel } from "@/components/execution/ExecutionObservabilityPanel";
import {
  getExecutionLogs,
  streamExecutionLogs,
} from "@/services/executionsApi";

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
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
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/drawer", () => ({
  Drawer: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DrawerContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DrawerDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DrawerHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DrawerTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleTrigger: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/filter-combobox", () => ({
  FilterCombobox: ({
    label,
    value,
    onValueChange,
    options,
  }: {
    label: string;
    value: string;
    onValueChange: (value: string) => void;
    options: Array<{ value: string; label: string }>;
  }) => (
    <label>
      {label}
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  ),
}));

vi.mock("@/components/ui/scroll-area", () => {
  const ReactModule = require("react") as typeof React;
  return {
    ScrollArea: ReactModule.forwardRef<
      HTMLDivElement,
      React.PropsWithChildren<{ className?: string }>
    >(({ children }, ref) => <div ref={ref}>{children}</div>),
  };
});

vi.mock("@/components/ui/SearchBar", () => ({
  SearchBar: ({
    value,
    onChange,
    placeholder,
  }: {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
  }) => (
    <input
      aria-label="observability-search"
      value={value}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({ data }: { data: unknown }) => <pre>{JSON.stringify(data)}</pre>,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    AlertCircle: Icon,
    ChevronDown: Icon,
    ChevronRight: Icon,
    Loader2: Icon,
    PauseCircle: Icon,
    Play: Icon,
    RefreshCw: Icon,
    Settings: Icon,
    Terminal: Icon,
  };
});

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/components/nodes", () => ({
  NodeProcessLogsPanel: ({ nodeId }: { nodeId?: string }) => <div>{`raw-logs:${nodeId}`}</div>,
}));

vi.mock("@/services/executionsApi", () => ({
  getExecutionLogs: vi.fn(),
  streamExecutionLogs: vi.fn(),
}));

const mockedGetExecutionLogs = vi.mocked(getExecutionLogs);
const mockedStreamExecutionLogs = vi.mocked(streamExecutionLogs);

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    workflow_id: "wf-1",
    execution_id: "exec-1",
    agentfield_request_id: "req-1",
    agent_node_id: "node-1",
    workflow_depth: 0,
    reasoner_id: "planner",
    input_data: {},
    output_data: {},
    input_size: 0,
    output_size: 0,
    workflow_tags: [],
    status: "running",
    started_at: "2026-04-08T00:00:00Z",
    retry_count: 0,
    created_at: "2026-04-08T00:00:00Z",
    updated_at: "2026-04-08T00:00:00Z",
    ...overrides,
  };
}

describe("ExecutionObservabilityPanel", () => {
  beforeEach(() => {
    mockedGetExecutionLogs.mockReset();
    mockedStreamExecutionLogs.mockReset();
  });

  it("loads logs, filters them, streams live updates, and opens raw node logs", async () => {
    const user = userEvent.setup();
    const eventSource = {
      onmessage: undefined as ((event: MessageEvent<string>) => void) | undefined,
      onerror: undefined as (() => void) | undefined,
      close: vi.fn(),
    };

    mockedGetExecutionLogs.mockResolvedValue({
      entries: [
        {
          event_id: 1,
          execution_id: "exec-1",
          workflow_id: "wf-1",
          seq: 1,
          agent_node_id: "node-1",
          reasoner_id: "planner",
          level: "info",
          source: "worker",
          event_type: "start",
          message: "Execution started",
          attributes: { step: 1 },
          system_generated: false,
          ts: "2026-04-08T00:00:00Z",
          recorded_at: "2026-04-08T00:00:00Z",
        },
        {
          event_id: 2,
          execution_id: "exec-1",
          workflow_id: "wf-1",
          seq: 2,
          agent_node_id: "node-2",
          level: "warn",
          source: "scheduler",
          message: "Delayed task",
          attributes: "{}",
          system_generated: true,
          ts: "2026-04-08T00:00:01Z",
          recorded_at: "2026-04-08T00:00:01Z",
        },
      ],
    } as never);
    mockedStreamExecutionLogs.mockReturnValue(eventSource as never);

    render(
      <ExecutionObservabilityPanel
        execution={buildExecution() as never}
        relatedNodeIds={["node-2"]}
      />
    );

    expect(await screen.findByText("Execution started")).toBeInTheDocument();
    expect(screen.getByText("Delayed task")).toBeInTheDocument();
    expect(screen.getByText("Showing 2 of 2 events")).toBeInTheDocument();
    expect(screen.getByText("1 system")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("observability-search"), {
      target: { value: "delayed" },
    });
    expect(await screen.findByText("Showing 1 of 2 events")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("observability-search"), {
      target: { value: "" },
    });
    await user.click(screen.getByRole("button", { name: /attrs/i }));
    expect(screen.getByText('{"step":1}')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Filter by node"), {
      target: { value: "node-2" },
    });
    expect(screen.getByText("Node: node-2")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Filter by source"), {
      target: { value: "scheduler" },
    });
    expect(screen.getByText("Source: scheduler")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Filter by node"), {
      target: { value: "all" },
    });
    fireEvent.change(screen.getByLabelText("Filter by source"), {
      target: { value: "all" },
    });

    await user.click(screen.getByRole("button", { name: "Go live" }));
    expect(mockedStreamExecutionLogs).toHaveBeenCalledWith("exec-1", { afterSeq: 2 });

    await act(async () => {
      eventSource.onmessage?.({
        data: JSON.stringify({
          event_id: 3,
          execution_id: "exec-1",
          workflow_id: "wf-1",
          seq: 3,
          agent_node_id: "node-1",
          level: "error",
          source: "worker",
          message: "Execution failed",
          attributes: { code: 500 },
          ts: "2026-04-08T00:00:02Z",
        }),
      } as MessageEvent<string>);
    });

    expect(await screen.findByText("Execution failed")).toBeInTheDocument();
    expect(screen.getByText("Live")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Raw node logs" }));
    expect(screen.getByText("Advanced raw node logs")).toBeInTheDocument();
    expect(screen.getByText("raw-logs:node-1")).toBeInTheDocument();
  });

  it("shows fetch and stream errors", async () => {
    const user = userEvent.setup();
    const eventSource = {
      onmessage: undefined as ((event: MessageEvent<string>) => void) | undefined,
      onerror: undefined as (() => void) | undefined,
      close: vi.fn(),
    };

    mockedGetExecutionLogs.mockRejectedValue(new Error("load failed"));
    mockedStreamExecutionLogs.mockReturnValue(eventSource as never);

    render(<ExecutionObservabilityPanel execution={buildExecution() as never} />);

    expect(await screen.findByText("Observability stream unavailable")).toBeInTheDocument();
    expect(screen.getByText("load failed")).toBeInTheDocument();

    mockedGetExecutionLogs.mockResolvedValue({ entries: [] } as never);
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText("No structured execution logs match the current filters yet.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Go live" }));
    await act(async () => {
      eventSource.onerror?.();
    });
    expect(await screen.findByText("Execution log stream interrupted")).toBeInTheDocument();
  });
});
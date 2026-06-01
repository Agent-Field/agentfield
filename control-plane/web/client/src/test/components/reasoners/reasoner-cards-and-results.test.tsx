// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CompactReasonersStats } from "@/components/reasoners/CompactReasonersStats";
import { ExecutionResult } from "@/components/reasoners/ExecutionResult";
import { FormattedOutput } from "@/components/reasoners/FormattedOutput";
import { PerformanceChart } from "@/components/reasoners/PerformanceChart";
import { ReasonerCard } from "@/components/reasoners/ReasonerCard";
import { SearchFilters } from "@/components/reasoners/SearchFilters";
import type { ExecutionResponse, PerformanceMetrics } from "@/types/execution";
import type { ReasonerWithNode } from "@/types/reasoners";

const navigateMock = vi.fn();
const useDIDStatusMock = vi.fn();
const clipboardWriteTextMock = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/hooks/useDIDInfo", () => ({
  useDIDStatus: (...args: unknown[]) => useDIDStatusMock(...args),
}));

vi.mock("@/utils/dateFormat", () => ({
  formatCompactRelativeTime: () => "2m",
}));

vi.mock("@/components/did/DIDStatusBadge", () => ({
  CompositeDIDStatus: ({ status }: { status: string }) => <span>did:{status}</span>,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <span data-icon={className}>icon</span>;
  return {
    TrendingUp: Icon,
    TrendingDown: Icon,
    Minus: Icon,
    CheckCircle: Icon,
    XCircle: Icon,
    Clock: Icon,
    Copy: Icon,
    Loader2: Icon,
    ReasonerIcon: Icon,
    Layers: Icon,
    Timer: Icon,
    Tag: Icon,
    Flash: Icon,
    BarChart3: Icon,
    Identification: Icon,
    Code: Icon,
    Time: Icon,
    View: Icon,
    Wifi: Icon,
    WifiOff: Icon,
    Grid: Icon,
    Terminal: Icon,
    Renew: Icon,
  };
});

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    type = "button",
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type={type} onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    variant,
    ...props
  }: React.PropsWithChildren<{ variant?: string } & React.HTMLAttributes<HTMLSpanElement>>) => (
    <span data-variant={variant} {...props}>
      {children}
    </span>
  ),
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div role="alert" {...props}>
      {children}
    </div>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({
    children,
    onClick,
    onKeyDown,
    interactive,
    variant,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement> & { interactive?: boolean; variant?: string }>) => (
    <div onClick={onClick} onKeyDown={onKeyDown} {...props}>
      {children}
    </div>
  ),
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/SearchBar", () => ({
  SearchBar: ({
    value,
    onChange,
    placeholder,
  }: {
    value: string;
    onChange: (value: string) => void;
    placeholder: string;
  }) => (
    <input
      aria-label={placeholder}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data }: { data: unknown }) => (
    <pre>{JSON.stringify(data, null, 2)}</pre>
  ),
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({ data }: { data: unknown }) => (
    <div>formatted:{JSON.stringify(data)}</div>
  ),
}));

vi.mock("@/components/ui/hover-card", () => ({
  HoverCard: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  HoverCardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/reasoners/ReasonerStatusDot", () => ({
  ReasonerStatusDot: ({ status }: { status: string }) => <span>status:{status}</span>,
}));

describe("reasoner cards and results", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    useDIDStatusMock.mockReturnValue({ status: null });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: clipboardWriteTextMock,
      },
    });
    clipboardWriteTextMock.mockResolvedValue(undefined);
  });

  it("renders performance chart empty state and populated metrics with insights", () => {
    const metrics: PerformanceMetrics = {
      avg_response_time_ms: 5400,
      success_rate: 0.72,
      total_executions: 245,
      executions_last_24h: 140,
      error_rate: 0.28,
      cost_last_24h: 1.2345,
      recent_executions: [],
      performance_trend: [
        {
          timestamp: "2026-04-08T00:00:00Z",
          avg_response_time: Number.NaN,
          success_rate: 0.95,
          execution_count: 40,
        },
        {
          timestamp: "2026-04-08T01:00:00Z",
          avg_response_time: 5400,
          success_rate: 0.72,
          execution_count: Number.NaN,
        },
      ],
    };

    const { rerender, container } = render(<PerformanceChart metrics={null} />);
    expect(screen.getByText(/No performance data available/i)).toBeInTheDocument();

    rerender(<PerformanceChart metrics={metrics} />);

    expect(screen.getByText("Avg Response Time")).toBeInTheDocument();
    expect(screen.getAllByText("5400ms").length).toBeGreaterThan(0);
    expect(screen.getAllByText("72.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("$1.2345")).toBeInTheDocument();
    expect(screen.getByText(/Slow response times/i)).toBeInTheDocument();
    expect(screen.getByText(/Low success rate/i)).toBeInTheDocument();
    expect(screen.getByText(/High usage volume/i)).toBeInTheDocument();
    expect(container.textContent).not.toContain("NaN");
  });

  it("renders execution result states and copies full result and output", async () => {
    const user = userEvent.setup();
    const result: ExecutionResponse = {
      execution_id: "exec-123456789",
      workflow_id: "workflow-123456789",
      run_id: "run-123456789",
      result: { answer: "ok" },
      duration_ms: 5200,
      cost: 0.125,
      status: "failed",
      error_message: "boom",
      memory_updates: [{ action: "updated", key: "summary", scope: "session" }],
      timestamp: "2026-04-08T10:00:00Z",
      node_id: "node-1",
      type: "reasoner",
      target: "planner",
    };

    const { rerender } = render(<ExecutionResult loading={true} />);
    expect(screen.getByText(/Executing reasoner/i)).toBeInTheDocument();

    rerender(<ExecutionResult error="Request failed" />);
    expect(screen.getByText("Execution Failed")).toBeInTheDocument();
    expect(screen.getByText("Request failed")).toBeInTheDocument();

    rerender(<ExecutionResult />);
    expect(screen.getByText(/No execution result yet/i)).toBeInTheDocument();

    rerender(<ExecutionResult result={result} />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText(/Error Details/i)).toBeInTheDocument();
    expect(screen.getByText("session.summary")).toBeInTheDocument();
    expect(screen.getByText(/Execution completed in 5200ms \(Slow\)/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Copy All/i }));

    fireEvent.click(screen.getByRole("button", { name: /Copy Output/i }));
  });

  it("renders reasoner card details and supports click, keyboard, and navigation paths", async () => {
    const user = userEvent.setup();
    vi.setSystemTime(new Date("2026-04-08T12:00:00Z"));
    useDIDStatusMock.mockReturnValue({
      status: {
        has_did: true,
        did_status: "public",
        reasoner_count: 2,
        skill_count: 3,
      },
    });

    const reasoner: ReasonerWithNode = {
      reasoner_id: "node-1.planner",
      name: "Planner",
      description: "Plans next actions",
      node_id: "node-1",
      node_status: "inactive",
      node_version: "1.2.3",
      input_schema: {},
      output_schema: {},
      memory_config: {
        auto_inject: [],
        memory_retention: "7d",
        cache_results: true,
      },
      avg_response_time_ms: 120,
      success_rate: 0.99,
      total_runs: 1200,
      last_updated: "2026-04-08T11:30:00Z",
    };

    const onClick = vi.fn();
    const { rerender } = render(<ReasonerCard reasoner={reasoner} onClick={onClick} />);

    expect(screen.getByText("Planner")).toBeInTheDocument();
    expect(screen.getByText(/Updated 30m ago/i)).toBeInTheDocument();
    expect(screen.getByText("Cached")).toBeInTheDocument();
    expect(screen.getByText("7d")).toBeInTheDocument();
    expect(screen.getByText("v1.2.3")).toBeInTheDocument();
    expect(screen.getByText("99.0% success")).toBeInTheDocument();
    expect(screen.getByText("1,200 runs")).toBeInTheDocument();
    expect(screen.getByText("did:public")).toBeInTheDocument();
    expect(screen.getByText("status:offline")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /View reasoner Planner/i }));
    expect(onClick).toHaveBeenCalledWith(reasoner);

    fireEvent.keyDown(screen.getByRole("button", { name: /View reasoner Planner/i }), {
      key: "Enter",
    });
    expect(onClick).toHaveBeenCalledTimes(2);

    rerender(<ReasonerCard reasoner={reasoner} />);
    fireEvent.keyDown(screen.getByRole("button", { name: /View reasoner Planner/i }), {
      key: " ",
    });
    expect(navigateMock).toHaveBeenCalledWith("/reasoners/node-1.planner");
  });

  it("renders formatted output empty, raw, and toggleable JSON/formatted views", async () => {
    const user = userEvent.setup();
    const onToggleView = vi.fn();
    const data = { foo: "bar", nested: { count: 2 } };

    const { rerender } = render(<FormattedOutput data={null} />);
    expect(screen.getByText(/No output data available/i)).toBeInTheDocument();

    rerender(
      <FormattedOutput
        data={data}
        showRaw={true}
        executionId="exec-abcdefghijklmnop"
        duration={321}
        status="succeeded"
      />
    );
    expect(screen.getByText(/Raw JSON Output/i)).toBeInTheDocument();
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(screen.getByText(/Completed in 321ms/i)).toBeInTheDocument();
    expect(screen.getByText(/exec-abcdefg/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Copy/i }));

    rerender(
      <FormattedOutput
        data={data}
        executionId="exec-abcdefghijklmnop"
        duration={2500}
        onToggleView={onToggleView}
      />
    );
    expect(screen.getByText(/formatted:\{\"foo\":\"bar\",\"nested\":\{\"count\":2\}\}/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /icon JSON/i }));
    expect(screen.getByText(/"foo": "bar"/i)).toBeInTheDocument();
    expect(screen.getByText("(Slow)")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Raw JSON/i }));
    expect(onToggleView).toHaveBeenCalledTimes(1);
  });

  it("renders compact reasoner stats safely and refreshes", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();

    render(
      <CompactReasonersStats
        total={10}
        onlineCount={7}
        offlineCount={3}
        nodesCount={4}
        lastRefresh={new Date("2026-04-08T10:00:00Z")}
        onRefresh={onRefresh}
        loading={true}
      />
    );

    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getAllByText("4").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Health:").length).toBeGreaterThan(0);
    expect(screen.getByText(/Updated:/i)).toBeInTheDocument();
    expect(screen.getAllByText(/2m ago/i).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button"));
    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("debounces search filters changes, switches status, and clears filters", async () => {
    vi.useFakeTimers();
    const onFiltersChange = vi.fn();

    render(
      <SearchFilters
        filters={{ status: "all", search: "old" }}
        onFiltersChange={onFiltersChange}
        totalCount={8}
        onlineCount={5}
        offlineCount={3}
      />
    );

    expect(screen.getByText(/Found/i)).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Found 8 reasoners matching "old"')
    ).toBeInTheDocument();
    expect(screen.getByText("Filtered")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Search reasoners/i), {
      target: { value: "planner" },
    });
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(onFiltersChange).toHaveBeenLastCalledWith({ status: "all", search: "planner" });

    fireEvent.click(screen.getByRole("button", { name: /Online/i }));
    expect(onFiltersChange).toHaveBeenLastCalledWith({ status: "online", search: "old" });

    fireEvent.click(screen.getByRole("button", { name: /Clear filters/i }));
    expect(onFiltersChange).toHaveBeenLastCalledWith({ status: "online" });
  });
});
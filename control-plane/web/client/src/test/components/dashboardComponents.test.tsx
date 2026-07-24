// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ActivityHeatmap } from "@/components/dashboard/ActivityHeatmap";
import { CompactActivityStream } from "@/components/dashboard/CompactActivityStream";
import { DashboardActiveWorkload } from "@/components/dashboard/DashboardActiveWorkload";
import { DashboardRunOutcomeStrip } from "@/components/dashboard/DashboardRunOutcomeStrip";
import { ExecutionTimeline } from "@/components/dashboard/ExecutionTimeline";
import { HotspotPanel } from "@/components/dashboard/HotspotPanel";
import { KPICard } from "@/components/dashboard/KPICard";
import { RecentActivityStream } from "@/components/dashboard/RecentActivityStream";
import { TimeRangeSelector } from "@/components/dashboard/TimeRangeSelector";
import { shortRunIdForDashboard } from "@/components/dashboard/dashboardRunUtils";

const componentState = vi.hoisted(() => ({
  navigate: vi.fn(),
  recentActivity: {
    executions: [] as Array<Record<string, unknown>>,
    loading: false,
    error: null as null | Error,
    hasError: false,
    refresh: vi.fn(),
    clearError: vi.fn(),
    isRefreshing: false,
    isEmpty: true,
  },
  executionTimeline: {
    timelineData: [] as Array<Record<string, unknown>>,
    summary: null as null | Record<string, number>,
    loading: false,
    error: null as null | Error,
    hasData: false,
    hasError: false,
    isEmpty: true,
    isRefreshing: false,
    refresh: vi.fn(),
    clearError: vi.fn(),
  },
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => componentState.navigate,
  };
});

vi.mock("@/hooks/useRecentActivity", () => ({
  useRecentActivitySimple: () => componentState.recentActivity,
}));

vi.mock("@/hooks/useExecutionTimeline", () => ({
  useExecutionTimeline: () => componentState.executionTimeline,
}));

vi.mock("recharts", () => {
  const Wrapper = ({ children }: React.PropsWithChildren) => <div>{children}</div>;
  return {
    ResponsiveContainer: Wrapper,
    CartesianGrid: () => <div />,
    XAxis: () => <div />,
    YAxis: ({ dataKey }: { dataKey?: string }) => <div>{dataKey}</div>,
    Tooltip: ({ content, ...props }: { content?: (args: unknown) => React.ReactNode }) => (
      <div>{content ? content({ active: true, payload: [{ payload: props.payload ?? {} }] }) : null}</div>
    ),
    BarChart: ({
      data,
      children,
    }: React.PropsWithChildren<{ data?: Array<{ name: string; count: number }> }>) => (
      <div>
        <div>bar-chart</div>
        {data?.map((entry) => (
          <div key={entry.name}>
            {entry.name}:{entry.count}
          </div>
        ))}
        {children}
      </div>
    ),
    Bar: Wrapper,
    Cell: () => <div />,
  };
});

vi.mock("@/components/ui/hover-card", () => ({
  HoverCard: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  HoverCardTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  HoverCardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/utils/dateFormat", () => ({
  formatCompactDate: (value: string) => `compact:${value}`,
  formatRelativeTime: (value: string) => `relative:${value}`,
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status: string) => status,
  getStatusLabel: (status: string) => status,
  getStatusTheme: (status: string) => ({
    status,
    bgClass: `bg-${status}`,
    borderClass: `border-${status}`,
    pillClass: `pill-${status}`,
    iconClass: `icon-${status}`,
  }),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <span className={className}>icon</span>;
  return {
    BarChart3: Icon,
    AlertTriangle: Icon,
    CheckmarkFilled: Icon,
    ErrorFilled: Icon,
    InProgress: Icon,
    Renew: Icon,
    Time: Icon,
    WarningAlt: Icon,
    Analytics: Icon,
    Warning: Icon,
    Restart: Icon,
  };
});

function makeHeatmap() {
  return Array.from({ length: 7 }, (_, day) =>
    Array.from({ length: 24 }, (_, hour) => ({
      total: day === 0 && hour === 0 ? 8 : 1,
      failed: day === 0 && hour === 0 ? 2 : 0,
      error_rate: day === 0 && hour === 0 ? 25 : 0,
    })),
  );
}

function makeDashboardRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-aaaaaa",
    started_at: "2026-04-08T10:00:00Z",
    latest_activity: "2026-04-08T10:05:00Z",
    status: "succeeded",
    terminal: true,
    duration_ms: 12000,
    root_reasoner: "Reasoner A",
    display_name: "Workflow A",
    current_task: "Done",
    status_counts: { succeeded: 4, failed: 0 },
    ...overrides,
  };
}

describe("dashboard components", () => {
  beforeEach(() => {
    componentState.navigate.mockReset();
    componentState.recentActivity = {
      executions: [
        {
          execution_id: "exec-1",
          status: "succeeded",
          duration_ms: 2300,
          relative_time: "2m ago",
          agent_name: "Agent Alpha",
          reasoner_name: "Reasoner A",
        },
      ],
      loading: false,
      error: null,
      hasError: false,
      refresh: vi.fn(),
      clearError: vi.fn(),
      isRefreshing: false,
      isEmpty: false,
    };
    componentState.executionTimeline = {
      timelineData: [
        { hour: "00:00", executions: 2, success_rate: 100, failed: 0 },
        { hour: "04:00", executions: 5, success_rate: 80, failed: 1 },
        { hour: "08:00", executions: 4, success_rate: 75, failed: 1 },
        { hour: "12:00", executions: 3, success_rate: 100, failed: 0 },
      ],
      summary: { total_executions: 14, avg_success_rate: 88.5, total_errors: 2 },
      loading: false,
      error: null,
      hasData: true,
      hasError: false,
      isEmpty: false,
      isRefreshing: false,
      refresh: vi.fn(),
      clearError: vi.fn(),
    };
  });

  it("renders and interacts with time range selector", () => {
    const onChange = vi.fn();
    const onCompareChange = vi.fn();

    render(
      <TimeRangeSelector
        value="24h"
        onChange={onChange}
        compare={false}
        onCompareChange={onCompareChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "7d" }));
    fireEvent.click(screen.getByLabelText("Compare"));

    expect(onChange).toHaveBeenCalledWith("7d");
    expect(onCompareChange).toHaveBeenCalledWith(true);
  });

  it("renders activity heatmap and toggles to usage view", () => {
    render(<ActivityHeatmap heatmapData={makeHeatmap()} />);

    expect(screen.getByText("Activity Patterns")).toBeInTheDocument();
    expect(screen.getByText("Low")).toBeInTheDocument();
    expect(screen.getByTitle("2 failed / 8 total (25.0%)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Usage" }));

    expect(screen.getByTitle("8 executions")).toBeInTheDocument();
  });

  it("renders hotspot panel content and empty state", () => {
    const hotspots = [
      {
        reasoner_id: "reasoner-hot",
        failed_executions: 5,
        total_executions: 10,
        contribution_pct: 62,
        error_rate: 50,
        top_errors: [{ message: "deadline exceeded", count: 2 }],
      },
    ];

    const { rerender } = render(<HotspotPanel hotspots={hotspots as never[]} />);

    expect(screen.getByText("Problem Hotspots")).toBeInTheDocument();
    expect(screen.getByText("reasoner-hot")).toBeInTheDocument();
    expect(screen.getByText("62% of errors")).toBeInTheDocument();
    expect(screen.getByText(/deadline exceeded/)).toBeInTheDocument();

    rerender(<HotspotPanel hotspots={[]} />);

    expect(screen.getByText("No failures detected in this time period.")).toBeInTheDocument();
  });

  it("renders interactive KPI card and keyboard activation", () => {
    const onClick = vi.fn();
    const TestIcon = ({ className }: { className?: string }) => <span className={className}>I</span>;

    render(
      <KPICard
        icon={TestIcon}
        label="Healthy agents"
        primaryValue="12"
        secondaryInfo="2 degraded"
        onClick={onClick}
        status="success"
      />,
    );

    fireEvent.keyDown(screen.getByRole("button", { name: "View Healthy agents details" }), {
      key: "Enter",
    });

    expect(screen.getByText("Healthy agents")).toBeInTheDocument();
    expect(screen.getByText("2 degraded")).toBeInTheDocument();
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders active workload aggregation by reasoner", () => {
    render(
      <DashboardActiveWorkload
        activeRuns={[
          { root_reasoner: "Reasoner A", display_name: "Reasoner A" },
          { root_reasoner: "Reasoner A", display_name: "Reasoner A" },
          { root_reasoner: "Reasoner B", display_name: "Reasoner B" },
        ] as never[]}
      />,
    );

    expect(screen.getByText("Active by reasoner")).toBeInTheDocument();
    expect(screen.getByText("Reasoner A:2")).toBeInTheDocument();
    expect(screen.getByText("Reasoner B:1")).toBeInTheDocument();
  });

  it("renders run outcome strip summary and handles run selection", () => {
    const onSelectRun = vi.fn();

    render(
      <DashboardRunOutcomeStrip
        runs={[
          makeDashboardRun(),
          makeDashboardRun({
            run_id: "run-bbbbbb",
            latest_activity: "2026-04-08T10:10:00Z",
            status: "failed",
            status_counts: { failed: 2 },
            root_reasoner: "Reasoner B",
          }),
        ] as never[]}
        onSelectRun={onSelectRun}
      />,
    );

    expect(screen.getByText("Run timeline")).toBeInTheDocument();
    expect(screen.getByText("2 runs · 1 ok · 1 failed")).toBeInTheDocument();
    expect(screen.getByText("Oldest")).toBeInTheDocument();
    expect(screen.getByText("Newest")).toBeInTheDocument();
    expect(screen.getByText("Reasoner A")).toBeInTheDocument();
    expect(screen.getByText("Reasoner B")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("listitem")[0]!);

    expect(onSelectRun).toHaveBeenCalled();
    expect(shortRunIdForDashboard("run-bbbbbb", 4)).toBe("…bbbb");
  });

  it("renders execution timeline metrics and refreshes", () => {
    render(<ExecutionTimeline />);

    expect(screen.getByText("Execution Timeline")).toBeInTheDocument();
    expect(screen.getByText("14")).toBeInTheDocument();
    expect(screen.getByText("88.5%")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText(/00:00: 2 executions, 100% success rate/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));

    expect(componentState.executionTimeline.refresh).toHaveBeenCalledTimes(1);
  });

  it("renders recent activity streams and navigation actions", () => {
    const { rerender } = render(<RecentActivityStream />);

    expect(screen.getByText("Recent Activity")).toBeInTheDocument();
    expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
    expect(screen.getByText("Reasoner A")).toBeInTheDocument();
    expect(screen.getByText("2s")).toBeInTheDocument();

    fireEvent.click(screen.getByText("View all executions →"));
    fireEvent.click(screen.getByText("Agent Alpha"));

    expect(componentState.navigate).toHaveBeenCalledWith("/executions");
    expect(componentState.navigate).toHaveBeenCalledWith("/executions/exec-1");

    componentState.recentActivity = {
      ...componentState.recentActivity,
      hasError: true,
      error: new Error("stale data"),
    };

    rerender(<CompactActivityStream />);

    expect(screen.getByText("Unable to refresh")).toBeInTheDocument();
    fireEvent.click(screen.getByText("×"));
    expect(componentState.recentActivity.clearError).toHaveBeenCalledTimes(1);
  });
});
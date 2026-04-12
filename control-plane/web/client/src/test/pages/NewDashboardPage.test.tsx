// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NewDashboardPage } from "@/pages/NewDashboardPage";

const pageState = vi.hoisted(() => ({
  navigate: vi.fn(),
  useQueryResult: {
    data: {
      executions: { today: 12 },
      success_rate: 0.91,
      agents: { running: 3 },
    },
    isLoading: false,
  },
  sseResult: { execConnected: true },
  runsResult: {
    isLoading: false,
    data: { workflows: [], total_count: 0 },
  },
  llmHealthResult: {
    isLoading: false,
    data: { endpoints: [] as Array<{ name: string; healthy: boolean }> },
  },
  queueResult: {
    data: { agents: {} as Record<string, { running: number; max_concurrent: number }> },
  },
  agentsResult: {
    isLoading: false,
    data: { nodes: [], count: 0 },
  },
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => pageState.navigate,
  };
});

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => pageState.useQueryResult,
}));

vi.mock("@/hooks/useSSEQuerySync", () => ({
  useSSESync: () => pageState.sseResult,
}));

vi.mock("@/hooks/queries", () => ({
  useRuns: () => pageState.runsResult,
  useLLMHealth: () => pageState.llmHealthResult,
  useQueueStatus: () => pageState.queueResult,
  useAgents: () => pageState.agentsResult,
}));

vi.mock("@/services/dashboardService", () => ({
  getDashboardSummary: vi.fn(),
}));

vi.mock("@/components/dashboard/DashboardActiveWorkload", () => ({
  DashboardActiveWorkload: ({ activeRuns }: { activeRuns: Array<{ run_id: string }> }) => (
    <section>
      <h2>Active by reasoner</h2>
      <div>{activeRuns.length} active runs</div>
    </section>
  ),
}));

vi.mock("@/components/dashboard/DashboardRunOutcomeStrip", () => ({
  DashboardRunOutcomeStrip: ({
    runs,
    onSelectRun,
  }: {
    runs: Array<{ run_id: string }>;
    onSelectRun: (runId: string) => void;
  }) => (
    <section>
      <h2>Run timeline</h2>
      <div>{runs.length} timeline runs</div>
      {runs[0] ? <button onClick={() => onSelectRun(runs[0].run_id)}>Open timeline run</button> : null}
    </section>
  ),
}));

vi.mock("@/components/ui/data-formatters", () => ({
  formatDurationHumanReadable: (value: number | null | undefined) =>
    value == null ? "—" : `${Math.round(value / 1000)}s`,
  LiveElapsedDuration: ({ startedAt }: { startedAt: string }) => <span>{startedAt}</span>,
}));

vi.mock("@/utils/dateFormat", () => ({
  formatRelativeTime: (value: string) => `relative:${value}`,
}));

vi.mock("@/utils/status", () => ({
  getStatusTheme: (status: string) => ({ status }),
  isFailureStatus: (status: string) => status === "failed",
  isTerminalStatus: (status: string) =>
    ["failed", "succeeded", "cancelled", "timeout"].includes(status),
  isTimeoutStatus: (status: string) => status === "timeout",
}));

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-1",
    display_name: "Daily sync",
    root_reasoner: "Reasoner A",
    status: "running",
    root_execution_status: "running",
    terminal: false,
    started_at: "2026-04-08T10:00:00Z",
    latest_activity: "2026-04-08T10:05:00Z",
    completed_at: null,
    duration_ms: null,
    current_task: "Fetching data",
    total_executions: 4,
    ...overrides,
  };
}

describe("NewDashboardPage", () => {
  beforeEach(() => {
    pageState.navigate.mockReset();
    pageState.useQueryResult = {
      data: {
        executions: { today: 12 },
        success_rate: 0.91,
        agents: { running: 3 },
      },
      isLoading: false,
    };
    pageState.sseResult = { execConnected: true };
    pageState.runsResult = {
      isLoading: false,
      data: {
        total_count: 3,
        workflows: [
          makeRun(),
          makeRun({
            run_id: "run-2",
            status: "failed",
            root_execution_status: "failed",
            terminal: true,
            duration_ms: 15000,
            latest_activity: "2026-04-08T09:55:00Z",
          }),
          makeRun({
            run_id: "run-3",
            status: "succeeded",
            root_execution_status: "succeeded",
            terminal: true,
            duration_ms: 8000,
            latest_activity: "2026-04-08T09:50:00Z",
            completed_at: "2026-04-08T09:50:00Z",
          }),
        ],
      },
    };
    pageState.llmHealthResult = {
      isLoading: false,
      data: { endpoints: [{ name: "primary-llm", healthy: false }] },
    };
    pageState.queueResult = {
      data: { agents: { "agent-1": { running: 2, max_concurrent: 2 } } },
    };
    pageState.agentsResult = {
      isLoading: false,
      data: {
        nodes: [
          { id: "agent-1", health_status: "ready" },
          { id: "agent-2", health_status: "active" },
          { id: "agent-3", health_status: "offline" },
        ],
        count: 3,
      },
    };
  });

  it("renders active dashboard state, issues, tables, and navigation actions", () => {
    render(<NewDashboardPage />);

    expect(screen.getByText("System issues")).toBeInTheDocument();
    expect(screen.getByText(/LLM circuit OPEN on endpoint: primary-llm/)).toBeInTheDocument();
    expect(screen.getByText(/Queue at capacity for agent: agent-1/)).toBeInTheDocument();
    expect(screen.getByText("Active runs")).toBeInTheDocument();
    expect(screen.getByText("Needs attention")).toBeInTheDocument();
    expect(screen.getByText("Run timeline")).toBeInTheDocument();
    expect(screen.getByText("Active by reasoner")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Recent runs")).toBeInTheDocument();
    expect(screen.getAllByText("Reasoner A").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByText("Open run"));
    fireEvent.click(screen.getByText("Open timeline run"));
    fireEvent.click(screen.getByText("All runs"));
    fireEvent.click(screen.getByText("View all"));
    fireEvent.click(screen.getAllByTitle("run-2")[1]!);

    expect(pageState.navigate).toHaveBeenCalledWith("/runs/run-1");
    expect(pageState.navigate).toHaveBeenCalledWith("/runs");
    expect(pageState.navigate).toHaveBeenCalledWith("/runs/run-2");
  });

  it("renders the empty-state primary focus when there are no runs", () => {
    pageState.runsResult = {
      isLoading: false,
      data: { total_count: 0, workflows: [] },
    };
    pageState.llmHealthResult = {
      isLoading: false,
      data: { endpoints: [] },
    };
    pageState.queueResult = {
      data: { agents: {} },
    };

    render(<NewDashboardPage />);

    expect(screen.getAllByText("No runs yet").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "View runs" })).toBeInTheDocument();
    expect(screen.queryByText("System issues")).not.toBeInTheDocument();
  });
});
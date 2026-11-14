import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { homeFixture } from "@/components/home/homeFixture";
import type { DeveloperHomeResponse } from "@/types/developerHome";

const mockNavigate = vi.fn();
vi.mock("react-router", async () => ({
  ...(await vi.importActual<typeof import("react-router")>("react-router")),
  useNavigate: () => mockNavigate,
}));

vi.mock("@/hooks/useSSEQuerySync", () => ({
  useSSESync: () => ({ execConnected: true }),
}));

vi.mock("@/hooks/queries", () => ({
  useRuns: vi.fn(),
  useLLMHealth: vi.fn(),
  useQueueStatus: vi.fn(),
}));

vi.mock("@/services/dashboardService", () => ({
  getDeveloperHome: vi.fn(),
}));

import { useLLMHealth, useQueueStatus, useRuns } from "@/hooks/queries";
import { getDeveloperHome } from "@/services/dashboardService";
import { NewDashboardPage } from "./NewDashboardPage";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
  return render(<NewDashboardPage />, { wrapper });
}

const run = {
  run_id: "run_9",
  status: "failed",
  root_execution_status: "failed",
  started_at: "2026-11-09T11:58:00Z",
  latest_activity: "2026-11-09T11:59:00Z",
  completed_at: "2026-11-09T11:59:00Z",
  display_name: "process_order",
  root_reasoner: "process_order",
  total_executions: 4,
  duration_ms: 1200,
  terminal: true,
};

function mockHome(data: DeveloperHomeResponse) {
  vi.mocked(getDeveloperHome).mockResolvedValue(data);
}

describe("NewDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    try {
      window.localStorage.clear();
    } catch {
      // Storage may be missing in the test environment.
    }
    vi.mocked(useRuns).mockReturnValue({
      data: { workflows: [run], total_count: 1 },
      isLoading: false,
    } as unknown as ReturnType<typeof useRuns>);
    vi.mocked(useLLMHealth).mockReturnValue({
      data: { enabled: false, healthy: false, endpoints: [] },
    } as unknown as ReturnType<typeof useLLMHealth>);
    vi.mocked(useQueueStatus).mockReturnValue({
      data: { enabled: false, max_per_agent: 0, agents: [], total_running: 0 },
    } as unknown as ReturnType<typeof useQueueStatus>);
  });

  it("shows the outage: status line, alerts, figures, map, problems, endpoints and approvals", async () => {
    mockHome(homeFixture());
    renderPage();

    expect(await screen.findByText("2 problems")).toBeInTheDocument();
    expect(screen.getByText(/2 of 4 agents healthy/)).toBeInTheDocument();
    expect(screen.getByText("orchestrator stopped sending heartbeats 20 min ago")).toBeInTheDocument();
    expect(screen.getByText(/Failed calls at 21%, usual 1.0%/)).toBeInTheDocument();
    expect(screen.getByText("Calls / hour")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Call map" })).toBeInTheDocument();
    expect(screen.getByText(/connection refused to <id> after N attempts/)).toBeInTheDocument();
    expect(screen.getByText("Endpoints")).toBeInTheDocument();
    expect(screen.getByText(/registered, waiting for tag approval/)).toBeInTheDocument();
    expect(screen.getByText("Recent runs")).toBeInTheDocument();
    // Nothing configured, nothing shown.
    expect(screen.queryByText("In and out")).not.toBeInTheDocument();
    expect(screen.queryByText("Spend / hour")).not.toBeInTheDocument();
    expect(getDeveloperHome).toHaveBeenCalledWith("1h");
  });

  it("links every row to its run or filter", async () => {
    mockHome(homeFixture());
    renderPage();
    await screen.findByText("2 problems");

    const failedRuns = screen.getAllByRole("link", { name: "Failed runs" });
    expect(failedRuns[0]).toHaveAttribute("href", "/runs?status=failed&search=orchestrator");
    expect(screen.getAllByRole("link", { name: "billing.charge" })[0]).toHaveAttribute("href", "/runs/run_9");

    const endpoints = screen.getByText("Endpoints").closest("div.rounded-lg, div[class*='rounded']")!.parentElement!;
    const row = within(endpoints).getByText("orchestrator.process_order").closest("tr")!;
    fireEvent.click(row);
    expect(mockNavigate).toHaveBeenCalledWith("/runs?search=process_order");

    fireEvent.click(screen.getByText("run_9").closest("tr")!);
    expect(mockNavigate).toHaveBeenCalledWith("/runs/run_9");
  });

  it("shows a healthy install without alerts", async () => {
    mockHome(
      homeFixture({
        stopped_agents: [],
        problems: [],
        totals: { counts: { total: 100, succeeded: 100, failed: 0, cancelled: 0, terminal: 100 }, held: 0, running_now: 1 },
        approvals: { total: 0, items: [] },
        agents: homeFixture().agents.filter((a) => a.status === "running"),
        agent_counts: { registered: 2, running: 2, offline: 0, degraded: 0, starting: 0, pending_approval: 0, unknown: 0 },
      }),
    );
    renderPage();
    expect(await screen.findByText("No problems")).toBeInTheDocument();
    expect(screen.getByText(/No failed calls in the last hour/)).toBeInTheDocument();
    expect(screen.queryByText("Waiting on people", { selector: "h3, [class*='CardTitle']" })).not.toBeInTheDocument();
  });

  it("shows only the first-agent panel on an empty install", async () => {
    vi.mocked(useRuns).mockReturnValue({
      data: { workflows: [], total_count: 0 },
      isLoading: false,
    } as unknown as ReturnType<typeof useRuns>);
    mockHome(
      homeFixture({
        agents: [],
        agent_counts: { registered: 0, running: 0, offline: 0, degraded: 0, starting: 0, pending_approval: 0, unknown: 0 },
        stopped_agents: [],
        totals: { counts: { total: 0, succeeded: 0, failed: 0, cancelled: 0, terminal: 0 }, held: 0, running_now: 0 },
        baseline: undefined,
        series: [],
        problems: [],
        edges: [],
        entries: [],
        endpoints: [],
        approvals: { total: 0, items: [] },
      }),
    );
    renderPage();
    expect(await screen.findByText("Connect your first agent")).toBeInTheDocument();
    expect(screen.getByText("pip install agentfield")).toBeInTheDocument();
    expect(screen.queryByText("Recent runs")).not.toBeInTheDocument();
    expect(screen.queryByText("Calls / hour")).not.toBeInTheDocument();
  });

  it("shows in-and-out and spend only when they are in use", async () => {
    mockHome(
      homeFixture({
        spend: { window_cost_usd: 2, window_tokens: 5000, window_rows: 3, baseline_rows: 0 },
        triggers: { configured: 1, enabled: 1, events: 12, dispatched: 12, failed: 0, dead_letters: 0 },
        webhooks: {
          registered: 3,
          delivered: 1,
          failed: 2,
          pending: 0,
          recent_failures: [{ execution_id: "exec_w", url: "https://hooks.example.test/x", attempt_count: 3, last_error: "HTTP 500" }],
        },
        forwarder: { enabled: true, dead_letters: 4 },
      }),
    );
    renderPage();
    expect(await screen.findByText("In and out")).toBeInTheDocument();
    expect(screen.getByText("Spend / hour")).toBeInTheDocument();
    expect(screen.getByText("Trigger events in")).toBeInTheDocument();
    expect(screen.getByText("2 failed")).toBeInTheDocument();
    expect(screen.getByText(/latest to hooks.example.test: HTTP 500/)).toBeInTheDocument();
    expect(screen.getByText("4 dropped")).toBeInTheDocument();
  });

  it("asks for another window and remembers it", async () => {
    mockHome(homeFixture());
    renderPage();
    await screen.findByText("2 problems");
    fireEvent.mouseDown(screen.getByRole("tab", { name: "24h" }));
    fireEvent.click(screen.getByRole("tab", { name: "24h" }));
    await waitFor(() => expect(getDeveloperHome).toHaveBeenCalledWith("24h"));
    expect(window.localStorage.getItem("af_home_window")).toBe("24h");
  });

  it("opens an agent's runs from the call map and lists approvals with their expiry", async () => {
    const base = homeFixture();
    mockHome(
      homeFixture({
        approvals: {
          total: 3,
          items: [{ ...base.approvals.items[0], expires_at: "2026-11-09T15:00:00Z" }],
        },
      }),
    );
    renderPage();
    await screen.findByText("2 problems");
    const map = screen.getByRole("img", { name: "Call map" });
    const billing = within(map).getByText("billing").closest("g")!;
    fireEvent.click(billing);
    expect(mockNavigate).toHaveBeenCalledWith("/runs?search=billing");
    expect(screen.getByText(/expires 3 h from now/)).toBeInTheDocument();
    expect(screen.getByText("and 2 more")).toBeInTheDocument();
    expect(screen.getByText(/Not called in the last hour: spare/)).toBeInTheDocument();
  });

  it("copies the first-agent snippet", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    vi.mocked(useRuns).mockReturnValue({
      data: { workflows: [], total_count: 0 },
      isLoading: false,
    } as unknown as ReturnType<typeof useRuns>);
    mockHome(
      homeFixture({
        agents: [],
        agent_counts: { registered: 0, running: 0, offline: 0, degraded: 0, starting: 0, pending_approval: 0, unknown: 0 },
        stopped_agents: [],
        totals: { counts: { total: 0, succeeded: 0, failed: 0, cancelled: 0, terminal: 0 }, held: 0, running_now: 0 },
      }),
    );
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: /Copy/ }));
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining("from agentfield import Agent"));
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("keeps the endpoints table short until asked for all of them", async () => {
    const many = Array.from({ length: 14 }, (_, i) => ({
      agent_node_id: "svc",
      reasoner_id: `r${String(i).padStart(2, "0")}`,
      calls: 100 - i,
      failed: 0,
      terminal: 100 - i,
      caller_agents: [],
      direct_calls: 100 - i,
      trigger_calls: 0,
    }));
    mockHome(homeFixture({ endpoints: many }));
    renderPage();
    await screen.findByText("2 problems");
    expect(screen.getByText("svc.r11")).toBeInTheDocument();
    expect(screen.queryByText("svc.r12")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show all 14 endpoints" }));
    expect(screen.getByText("svc.r13")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show fewer" }));
    expect(screen.queryByText("svc.r13")).not.toBeInTheDocument();
  });

  it("offers a retry when the home data cannot load", async () => {
    vi.mocked(getDeveloperHome).mockRejectedValue(new Error("boom"));
    renderPage();
    expect(await screen.findByText("Home could not load")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();
  });
});

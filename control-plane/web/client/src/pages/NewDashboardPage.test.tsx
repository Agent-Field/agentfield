// @ts-nocheck
import { render, screen, fireEvent, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { NewDashboardPage } from "./NewDashboardPage";
import { WorkflowSummary } from "@/types/workflows";

// Mock ResizeObserver for recharts
const ResizeObserverMock = vi.fn(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));
vi.stubGlobal('ResizeObserver', ResizeObserverMock);

// Mock dependencies
const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => ({
  ...(await vi.importActual("react-router-dom")),
  useNavigate: () => mockNavigate,
}));

vi.mock("@/hooks/useSSEQuerySync", () => ({
  useSSESync: () => ({ execConnected: true }),
}));

vi.mock("@/hooks/queries", () => ({
  useRuns: vi.fn(),
  useLLMHealth: vi.fn(),
  useQueueStatus: vi.fn(),
  useAgents: vi.fn(),
}));

vi.mock("@/services/dashboardService", () => ({
  getDashboardSummary: vi.fn(),
}));

// Mock child components that might have complex internal logic
vi.mock("@/components/dashboard/DashboardRunOutcomeStrip", () => ({
  DashboardRunOutcomeStrip: () => <div data-testid="mock-outcome-strip" />,
}));
vi.mock("@/components/dashboard/DashboardActiveWorkload", () => ({
  DashboardActiveWorkload: () => <div data-testid="mock-active-workload" />,
}));


// Import mocks for manipulation
import { useRuns, useLLMHealth, useQueueStatus, useAgents } from "@/hooks/queries";
import { getDashboardSummary } from "@/services/dashboardService";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      cacheTime: 0,
    },
  },
});

const wrapper = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    <MemoryRouter>{children}</MemoryRouter>
  </QueryClientProvider>
);

const mockRuns: WorkflowSummary[] = [
  {
    run_id: "run-1",
    status: "running",
    root_execution_status: "running",
    started_at: new Date().toISOString(),
    latest_activity: new Date().toISOString(),
    display_name: "Reasoner A",
    root_reasoner: "Reasoner A",
    current_task: "Doing something",
    total_executions: 5,
    duration_ms: null,
    terminal: false,
  },
  {
    run_id: "run-2",
    status: "succeeded",
    root_execution_status: "succeeded",
    started_at: new Date(Date.now() - 3600 * 1000).toISOString(),
    completed_at: new Date(Date.now() - 3500 * 1000).toISOString(),
    latest_activity: new Date(Date.now() - 3500 * 1000).toISOString(),
    display_name: "Reasoner B",
    root_reasoner: "Reasoner B",
    current_task: "Finished",
    total_executions: 10,
    duration_ms: 100000,
    terminal: true,
  },
  {
    run_id: "run-3",
    status: "failed",
    root_execution_status: "failed",
    started_at: new Date(Date.now() - 7200 * 1000).toISOString(),
    completed_at: new Date(Date.now() - 7100 * 1000).toISOString(),
    latest_activity: new Date(Date.now() - 7100 * 1000).toISOString(),
    display_name: "Reasoner C",
    root_reasoner: "Reasoner C",
    current_task: "It broke",
    total_executions: 2,
    duration_ms: 50000,
    terminal: true,
  },
];

describe("NewDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();

    // Default healthy state
    vi.mocked(useLLMHealth).mockReturnValue({
      data: { endpoints: [{ name: "test-llm", healthy: true }] },
      isLoading: false,
    });
    vi.mocked(useQueueStatus).mockReturnValue({ data: { agents: {} }, isLoading: false });
    vi.mocked(useAgents).mockReturnValue({
      data: { count: 1, nodes: [{ health_status: "ready" }] },
      isLoading: false,
    });
    vi.mocked(getDashboardSummary).mockResolvedValue({
      executions: { today: 10 },
      success_rate: 0.9,
      agents: { running: 1 },
    });
  });

  it("renders loading skeletons", () => {
    vi.mocked(useRuns).mockReturnValue({ isLoading: true, data: undefined });
    const { container } = render(<NewDashboardPage />, { wrapper });

    expect(screen.queryByText("Active runs")).toBeNull();
    const skeletons = container.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("renders empty state when no runs are available", () => {
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [], total_count: 0 } });
    render(<NewDashboardPage />, { wrapper });

    expect(screen.getAllByText("No runs yet")[0]).toBeInTheDocument();
    expect(screen.getByText(/Trigger a workflow from your agent or CLI/)).toBeInTheDocument();
  });

  it("renders active runs when there are active runs", () => {
    const activeRun = { ...mockRuns[0] };
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [activeRun], total_count: 1 } });
    render(<NewDashboardPage />, { wrapper });

    expect(screen.getByText("Active runs")).toBeInTheDocument();
    expect(screen.getAllByText("Reasoner A")[0]).toBeInTheDocument();
    expect(screen.getByText("Doing something")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open run" }));
    expect(mockNavigate).toHaveBeenCalledWith("/runs/run-1");
  });

  it("renders latest completed run when no active runs", () => {
    const completedRun = { ...mockRuns[1] };
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [completedRun], total_count: 1 } });
    render(<NewDashboardPage />, { wrapper });
    
    expect(screen.getByText("Latest run")).toBeInTheDocument();
    expect(screen.getAllByText("Reasoner B")[0]).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open run" }));
    expect(mockNavigate).toHaveBeenCalledWith("/runs/run-2");
  });

  it("renders issues banner for unhealthy LLM endpoints", () => {
    vi.mocked(useLLMHealth).mockReturnValue({
      data: { endpoints: [{ name: "bad-llm", healthy: false }] },
      isLoading: false,
    });
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [], total_count: 0 } });
    render(<NewDashboardPage />, { wrapper });

    expect(screen.getByText("System issues")).toBeInTheDocument();
    expect(screen.getByText(/LLM circuit OPEN on endpoint: bad-llm/i)).toBeInTheDocument();
  });

  it("renders issues banner for overloaded agent queues", () => {
    vi.mocked(useQueueStatus).mockReturnValue({
      data: { agents: { "agent-1": { running: 10, max_concurrent: 10 } } },
      isLoading: false,
    });
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [], total_count: 0 } });

    render(<NewDashboardPage />, { wrapper });

    expect(screen.getByText("System issues")).toBeInTheDocument();
    expect(screen.getByText(/Queue at capacity for agent: agent-1/i)).toBeInTheDocument();
  });
  
  it("renders failures attention card with failed runs", () => {
    const failedRun = { ...mockRuns[2] };
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [failedRun], total_count: 1 } });
    render(<NewDashboardPage />, { wrapper });

    expect(screen.getByText("Needs attention")).toBeInTheDocument();
    expect(screen.getAllByText("Reasoner C")[0]).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    expect(mockNavigate).toHaveBeenCalledWith("/runs/run-3");
  });

  it('navigates to all runs page on "View all" click', () => {
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: mockRuns, total_count: 3 } });
    render(<NewDashboardPage />, { wrapper });

    const viewAllButtons = screen.getAllByText("View all");
    fireEvent.click(viewAllButtons[0]);
    expect(mockNavigate).toHaveBeenCalledWith("/runs");
  });
  
  it('navigates to all runs page on "All runs" click in failures card', () => {
    const failedRun = { ...mockRuns[2] };
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: [failedRun], total_count: 1 } });
    render(<NewDashboardPage />, { wrapper });

    fireEvent.click(screen.getByText("All runs"));
    expect(mockNavigate).toHaveBeenCalledWith("/runs");
  });

  it("renders stats section correctly", async () => {
    vi.mocked(useRuns).mockReturnValue({ isLoading: false, data: { workflows: mockRuns, total_count: 3 } });
    vi.mocked(useAgents).mockReturnValue({
        data: { count: 1, nodes: [{ health_status: "ready" }] },
        isLoading: false,
    });
    vi.mocked(getDashboardSummary).mockResolvedValue({
        executions: { today: 123 },
        success_rate: 0.88,
        agents: { running: 5 },
    });
    
    render(<NewDashboardPage />, { wrapper });

    expect(await screen.findByText("123")).toBeInTheDocument();
    expect(screen.getByText("runs today")).toBeInTheDocument();
    
    expect(await screen.findByText("88%")).toBeInTheDocument();
    expect(screen.getByText("success")).toBeInTheDocument();

    const agentsOnlineText = await screen.findByText('agents online');
    const parentDiv = agentsOnlineText.parentElement;
    const agentCount = within(parentDiv).getByText('1');
    expect(agentCount).toBeInTheDocument();
  });
});

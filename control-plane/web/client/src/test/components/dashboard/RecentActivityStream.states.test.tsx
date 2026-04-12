// @ts-nocheck
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RecentActivityStream } from "@/components/dashboard/RecentActivityStream";

const state = vi.hoisted(() => ({
  navigate: vi.fn(),
  refresh: vi.fn(),
  clearError: vi.fn(),
  recentActivity: {
    executions: [] as Array<Record<string, unknown>>,
    loading: false,
    error: null as null | { message?: string },
    hasError: false,
    refresh: null as unknown,
    clearError: null as unknown,
    isRefreshing: false,
    isEmpty: true,
  },
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => state.navigate,
  };
});

vi.mock("@/hooks/useRecentActivity", () => ({
  useRecentActivitySimple: () => ({
    ...state.recentActivity,
    refresh: state.refresh,
    clearError: state.clearError,
  }),
}));

describe("RecentActivityStream", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state.recentActivity = {
      executions: [],
      loading: false,
      error: null,
      hasError: false,
      refresh: state.refresh,
      clearError: state.clearError,
      isRefreshing: false,
      isEmpty: true,
    };
  });

  it("renders loading, empty, and fatal error states with refresh controls", () => {
    state.recentActivity = {
      ...state.recentActivity,
      loading: true,
      isEmpty: false,
    };

    const { rerender } = render(<RecentActivityStream className="stream-shell" />);

    expect(screen.getByText("Recent Activity")).toBeInTheDocument();
    expect(document.querySelector(".stream-shell")).toBeTruthy();
    expect(document.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);

    state.recentActivity = {
      ...state.recentActivity,
      loading: false,
      isEmpty: true,
      isRefreshing: true,
    };

    rerender(<RecentActivityStream />);
    expect(screen.getByText("No recent activity")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /loading/i })).toBeDisabled();

    state.recentActivity = {
      ...state.recentActivity,
      hasError: true,
      error: { message: "network down" },
      isRefreshing: false,
    };

    rerender(<RecentActivityStream />);
    expect(screen.getByText("Failed to load recent activity")).toBeInTheDocument();
    expect(screen.getByText("network down")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));

    expect(state.refresh).toHaveBeenCalledTimes(1);
    expect(state.clearError).toHaveBeenCalledTimes(1);
  });

  it("renders activity rows, formats durations, navigates, and updates with refreshed data", () => {
    state.recentActivity = {
      ...state.recentActivity,
      isEmpty: false,
      executions: [
        {
          execution_id: "exec-1",
          agent_name: "Agent Alpha",
          reasoner_name: "Reasoner A",
          status: "completed",
          started_at: "2026-04-08T10:00:00Z",
          duration_ms: 2400,
          relative_time: "2 minutes ago",
        },
        {
          execution_id: "exec-2",
          agent_name: "Agent Beta",
          reasoner_name: "Reasoner B",
          status: "failed",
          started_at: "2026-04-08T10:05:00Z",
          duration_ms: 65000,
          relative_time: "1 minute ago",
        },
        {
          execution_id: "exec-3",
          agent_name: "Agent Gamma",
          reasoner_name: "Reasoner C",
          status: "running",
          started_at: "2026-04-08T10:06:00Z",
          duration_ms: 750,
          relative_time: "just now",
        },
      ],
    };

    const { rerender } = render(<RecentActivityStream />);

    expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(screen.getByText("2s")).toBeInTheDocument();
    expect(screen.getByText("1m 5s")).toBeInTheDocument();
    expect(screen.getByText("750ms")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Agent Alpha"));
    fireEvent.click(screen.getByRole("button", { name: /view all executions/i }));

    expect(state.navigate).toHaveBeenCalledWith("/executions/exec-1");
    expect(state.navigate).toHaveBeenCalledWith("/executions");

    state.recentActivity = {
      ...state.recentActivity,
      executions: [
        ...state.recentActivity.executions,
        {
          execution_id: "exec-4",
          agent_name: "Agent Delta",
          reasoner_name: "Reasoner D",
          status: "success",
          started_at: "2026-04-08T10:07:00Z",
          duration_ms: 120000,
          relative_time: "moments ago",
        },
      ],
      isRefreshing: true,
      hasError: true,
      error: { message: "stale cache" },
    };

    rerender(<RecentActivityStream />);

    expect(screen.getByText("Agent Delta")).toBeInTheDocument();
    expect(screen.getByText("2m")).toBeInTheDocument();
    expect(screen.getByText("Unable to refresh data")).toBeInTheDocument();
    expect(screen.getByText("Showing cached data. stale cache")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /refreshing/i })).toBeDisabled();

    const dismiss = screen.getByRole("button", { name: "Dismiss" });
    fireEvent.mouseEnter(dismiss);
    expect(dismiss).toHaveStyle({ opacity: "0.8" });
    fireEvent.mouseLeave(dismiss);
    expect(dismiss).toHaveStyle({ opacity: "1" });
    fireEvent.click(dismiss);

    expect(state.clearError).toHaveBeenCalledTimes(1);
  });
});

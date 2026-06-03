// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

let recentActivityState: any;
const navigate = vi.fn();

vi.mock("@/hooks/useRecentActivity", () => ({
  useRecentActivitySimple: () => recentActivityState,
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => navigate,
  };
});

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(
    ({ className }, ref) => <svg ref={ref} data-testid="icon" className={className} />,
  );
  Icon.displayName = "Icon";
  return {
    CheckmarkFilled: Icon,
    ErrorFilled: Icon,
    InProgress: Icon,
    Renew: Icon,
    Time: Icon,
    WarningAlt: Icon,
  };
});

import { CompactActivityStream } from "./CompactActivityStream";

const makeActivityState = (overrides: Record<string, unknown> = {}) => ({
  executions: [],
  loading: false,
  error: null,
  hasError: false,
  refresh: vi.fn(),
  clearError: vi.fn(),
  isRefreshing: false,
  isEmpty: false,
  ...overrides,
});

describe("CompactActivityStream", () => {
  beforeEach(() => {
    recentActivityState = makeActivityState();
    navigate.mockReset();
  });

  it("renders the loading shell with placeholder rows", () => {
    recentActivityState = makeActivityState({
      loading: true,
      executions: [],
    });

    const { container } = render(<CompactActivityStream />);

    expect(screen.getByText("Recent Activity")).toBeInTheDocument();
    expect(container.querySelectorAll(".animate-spin").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".space-y-2 > div").length).toBeGreaterThanOrEqual(18);
  });

  it("renders the empty error state and supports retry and dismiss", () => {
    const refresh = vi.fn();
    const clearError = vi.fn();
    recentActivityState = makeActivityState({
      hasError: true,
      isEmpty: true,
      error: { message: "request failed" },
      refresh,
      clearError,
    });

    render(<CompactActivityStream />);

    expect(screen.getByText("Failed to load activity")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(clearError).toHaveBeenCalledTimes(1);
  });

  it("renders the empty state and refresh button label while refreshing", () => {
    const refresh = vi.fn();
    recentActivityState = makeActivityState({
      isEmpty: true,
      isRefreshing: true,
      refresh,
    });

    render(<CompactActivityStream />);

    expect(screen.getByText("No recent activity")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load..." })).toBeDisabled();
  });

  it("renders compact execution rows, trims to 18 items, and navigates from row and footer actions", () => {
    const refresh = vi.fn();
    const clearError = vi.fn();
    const executions = Array.from({ length: 20 }, (_, index) => ({
      execution_id: `exec-${index}`,
      agent_name: `Agent ${index}`,
      reasoner_name: `Reasoner ${index}`,
      status: index === 0 ? "success" : index === 1 ? "failed" : index === 2 ? "running" : "pending",
      started_at: `2026-04-09T0${index % 10}:00:00Z`,
      duration_ms: index === 0 ? 450 : index === 1 ? 12_000 : index === 2 ? 125_000 : undefined,
      relative_time: `${index}m ago`,
    }));

    recentActivityState = makeActivityState({
      executions,
      hasError: true,
      error: { message: "stale data" },
      clearError,
      refresh,
    });

    render(<CompactActivityStream className="activity-stream" />);

    expect(document.querySelector(".activity-stream")).toBeInTheDocument();
    expect(screen.getByText("Unable to refresh")).toBeInTheDocument();
    expect(screen.queryByText("Reasoner 19")).not.toBeInTheDocument();
    expect(screen.getByText("Reasoner 17")).toBeInTheDocument();
    expect(screen.getByTitle("Agent 0 • Reasoner 0 • 0m ago • 450ms")).toBeInTheDocument();
    expect(screen.getByTitle("Agent 1 • Reasoner 1 • 1m ago • 12s")).toBeInTheDocument();
    expect(screen.getByTitle("Agent 2 • Reasoner 2 • 2m ago • 2m 5s")).toBeInTheDocument();

    fireEvent.click(screen.getByText("×"));
    expect(clearError).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText("Reasoner 1"));
    fireEvent.click(screen.getByRole("button", { name: /view all executions/i }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    expect(navigate).toHaveBeenNthCalledWith(1, "/executions/exec-1");
    expect(navigate).toHaveBeenNthCalledWith(2, "/executions");
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});

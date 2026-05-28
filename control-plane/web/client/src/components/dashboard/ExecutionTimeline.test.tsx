// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

let timelineState: any;

vi.mock("../../hooks/useExecutionTimeline", () => ({
  useExecutionTimeline: () => timelineState,
}));

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(
    ({ className }, ref) => <svg ref={ref} data-testid="icon" className={className} />,
  );
  Icon.displayName = "Icon";
  return {
    Analytics: Icon,
    Renew: Icon,
    Warning: Icon,
    Restart: Icon,
  };
});

import { ExecutionTimeline } from "./ExecutionTimeline";

const makeTimelineState = (overrides: Record<string, unknown> = {}) => ({
  timelineData: [],
  summary: null,
  loading: false,
  error: null,
  hasData: false,
  hasError: false,
  isEmpty: false,
  isRefreshing: false,
  refresh: vi.fn(),
  clearError: vi.fn(),
  ...overrides,
});

describe("ExecutionTimeline", () => {
  beforeEach(() => {
    timelineState = makeTimelineState();
  });

  it("renders loading state and keeps refresh disabled while loading", () => {
    timelineState = makeTimelineState({
      loading: true,
      hasData: false,
    });

    render(<ExecutionTimeline className="timeline-shell" />);

    expect(screen.getByText("Loading execution timeline...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
    expect(document.querySelector(".timeline-shell")).toBeInTheDocument();
  });

  it("renders error state and clears error before retrying", () => {
    const refresh = vi.fn();
    const clearError = vi.fn();
    timelineState = makeTimelineState({
      hasError: true,
      error: { message: "backend unavailable" },
      refresh,
      clearError,
    });

    render(<ExecutionTimeline />);

    expect(screen.getByText("Failed to load timeline data")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    expect(clearError).toHaveBeenCalledTimes(1);
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("renders the empty state and refreshes on demand", () => {
    const refresh = vi.fn();
    timelineState = makeTimelineState({
      isEmpty: true,
      refresh,
    });

    render(<ExecutionTimeline />);

    expect(screen.getByText("No execution data available")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("renders chart metrics from data, sanitizes invalid points, and refreshes without clearing errors", () => {
    const refresh = vi.fn();
    const clearError = vi.fn();
    timelineState = makeTimelineState({
      hasData: true,
      timelineData: [
        { hour: "00:00", executions: 0, success_rate: 20, failed: 0 },
        { hour: "01:00", executions: 5, success_rate: 0, failed: 1 },
        { hour: "02:00", executions: 15, success_rate: 75, failed: 2 },
        { hour: "03:00", executions: 8, success_rate: 50, failed: 1 },
        { hour: "04:00", executions: 12, success_rate: 100, failed: 0 },
      ],
      refresh,
      clearError,
    });

    const { container } = render(<ExecutionTimeline />);

    expect(screen.getByText("Total Executions").previousSibling).toHaveTextContent("40");
    expect(screen.getByText("Avg Success Rate").previousSibling).toHaveTextContent("49.0%");
    expect(screen.getByText("Total Errors").previousSibling).toHaveTextContent("4");
    expect(screen.getByText("04:00")).toBeInTheDocument();

    const svgPaths = Array.from(container.querySelectorAll("path"));
    expect(svgPaths.some((path) => (path.getAttribute("d") || "").includes("NaN"))).toBe(false);
    expect(container.querySelectorAll("circle")).toHaveLength(5);

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(clearError).not.toHaveBeenCalled();
  });

  it("prefers summary metrics when provided", () => {
    timelineState = makeTimelineState({
      hasData: true,
      summary: {
        total_executions: 321,
        avg_success_rate: 98.4,
        total_errors: 7,
      },
      timelineData: [
        { hour: "11:00", executions: 1, success_rate: 100, failed: 0 },
        { hour: "12:00", executions: 2, success_rate: 50, failed: 1 },
      ],
    });

    render(<ExecutionTimeline />);

    expect(screen.getByText("321")).toBeInTheDocument();
    expect(screen.getByText("98.4%")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });

  it("avoids NaN in chart path output even when timeline points are invalid", () => {
    timelineState = makeTimelineState({
      hasData: true,
      timelineData: [
        { hour: "00:00", executions: 1, success_rate: 10, failed: 0 },
        { hour: "01:00", executions: Number.NaN, success_rate: Number.NaN, failed: 0 },
        { hour: "02:00", executions: 2, success_rate: 90, failed: 0 },
      ],
    });

    const { container } = render(<ExecutionTimeline />);
    const svgPaths = Array.from(container.querySelectorAll("path"));
    expect(svgPaths.some((path) => (path.getAttribute("d") || "").includes("NaN"))).toBe(false);
  });
});

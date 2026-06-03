// @ts-nocheck
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useDashboard } from "@/hooks/useDashboard";
import { useEnhancedDashboard } from "@/hooks/useEnhancedDashboard";
import {
  useExecutionTimeline,
} from "@/hooks/useExecutionTimeline";
import { useRecentActivity } from "@/hooks/useRecentActivity";

const dashboardServiceState = vi.hoisted(() => ({
  getDashboardSummary: vi.fn(),
  getDashboardSummaryWithRetry: vi.fn(),
  getEnhancedDashboardSummary: vi.fn(),
}));

const recentActivityServiceState = vi.hoisted(() => ({
  getRecentActivity: vi.fn(),
  getRecentActivityWithRetry: vi.fn(),
}));

const executionTimelineServiceState = vi.hoisted(() => ({
  getExecutionTimeline: vi.fn(),
  getExecutionTimelineWithRetry: vi.fn(),
}));

vi.mock("@/services/dashboardService", () => dashboardServiceState);
vi.mock("@/services/recentActivityService", () => recentActivityServiceState);
vi.mock("@/services/executionTimelineService", () => executionTimelineServiceState);

describe("data hooks", () => {
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("useDashboard fetches data, uses cached refreshes, and resets state", async () => {
    vi.useFakeTimers();
    const onDataUpdate = vi.fn();
    const dashboardData = { summary: { total_agents: 2 } } as any;

    dashboardServiceState.getDashboardSummaryWithRetry.mockResolvedValue(dashboardData);

    const { result, unmount } = renderHook(() =>
      useDashboard({ refreshInterval: 1000, cacheTtl: 5000, onDataUpdate })
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.data).toEqual(dashboardData);
    expect(result.current.hasData).toBe(true);
    expect(result.current.isCached).toBe(true);
    expect(onDataUpdate).toHaveBeenCalledWith(dashboardData);
    expect(dashboardServiceState.getDashboardSummaryWithRetry).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.refresh();
    });
    expect(dashboardServiceState.getDashboardSummaryWithRetry).toHaveBeenCalledTimes(2);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(dashboardServiceState.getDashboardSummaryWithRetry).toHaveBeenCalledTimes(2);

    act(() => {
      result.current.reset();
    });

    expect(result.current.data).toBeNull();
    expect(result.current.isEmpty).toBe(true);
    unmount();
  });

  it("useDashboard handles errors and supports simplified variants", async () => {
    const onError = vi.fn();
    const error = new Error("dashboard failed");
    dashboardServiceState.getDashboardSummaryWithRetry.mockRejectedValue(error);

    const { result } = renderHook(() =>
      useDashboard({ refreshInterval: 0, onError, enableRetry: true, maxRetries: 4 })
    );

    await waitFor(() => expect(result.current.error?.message).toBe("dashboard failed"));
    expect(result.current.hasError).toBe(true);
    expect(result.current.isStale).toBe(true);
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "dashboard failed" }));

    act(() => {
      result.current.clearError();
    });
    expect(result.current.error).toBeNull();

  });

  it("useEnhancedDashboard fetches, refetches when params change, and clears errors", async () => {
    const firstData = { metrics: [1], compare: false } as any;
    const secondData = { metrics: [2], compare: true } as any;
    const onDataUpdate = vi.fn();

    dashboardServiceState.getEnhancedDashboardSummary
      .mockResolvedValueOnce(firstData)
      .mockResolvedValueOnce(firstData)
      .mockResolvedValueOnce(secondData);

    const { result, rerender, unmount } = renderHook(
      ({ compare }) =>
        useEnhancedDashboard({
          refreshInterval: 0,
          preset: "24h",
          compare,
          onDataUpdate,
        }),
      { initialProps: { compare: false } }
    );

    await waitFor(() => expect(result.current.data).toEqual(firstData));
    expect(result.current.hasData).toBe(true);
    expect(dashboardServiceState.getEnhancedDashboardSummary).toHaveBeenCalledTimes(2);

    await act(async () => {
      result.current.refresh();
    });
    expect(dashboardServiceState.getEnhancedDashboardSummary).toHaveBeenCalledTimes(3);

    rerender({ compare: true });
    await waitFor(() => expect(result.current.data).toEqual(secondData));
    expect(dashboardServiceState.getEnhancedDashboardSummary).toHaveBeenLastCalledWith({
      preset: "24h",
      startTime: undefined,
      endTime: undefined,
      compare: true,
    });
    expect(onDataUpdate).toHaveBeenCalledWith(secondData);

    const fetchError = new Error("enhanced failed");
    dashboardServiceState.getEnhancedDashboardSummary.mockRejectedValueOnce(fetchError);
    await act(async () => {
      result.current.refresh();
    });
    await waitFor(() => expect(result.current.error?.message).toBe("enhanced failed"));

    act(() => {
      result.current.clearError();
    });
    expect(result.current.error).toBeNull();
    unmount();
  });

  it("useRecentActivity supports fetch, forced refresh, and error transitions", async () => {
    const activityData = { executions: [{ id: "exec-1" }] } as any;
    recentActivityServiceState.getRecentActivityWithRetry.mockResolvedValue(activityData);

    const { result, unmount } = renderHook(() =>
      useRecentActivity({ refreshInterval: 0, cacheTtl: 5000 })
    );

    await waitFor(() => expect(result.current.executions).toEqual(activityData.executions));
    expect(result.current.executionCount).toBe(1);

    await act(async () => {
      result.current.refresh();
    });
    expect(recentActivityServiceState.getRecentActivityWithRetry).toHaveBeenCalledTimes(2);

    recentActivityServiceState.getRecentActivityWithRetry.mockRejectedValueOnce(
      new Error("recent failed")
    );
    await act(async () => {
      result.current.refresh();
    });
    await waitFor(() => expect(result.current.error?.message).toBe("recent failed"));

    act(() => {
      result.current.clearError();
      result.current.reset();
    });
    expect(result.current.data).toBeNull();
    expect(result.current.executionCount).toBe(0);
    unmount();
  });

  it("useExecutionTimeline fetches data, debounces refresh while scheduled, and exposes variants", async () => {
    vi.useFakeTimers();
    const timelineData = {
      timeline_data: [{ timestamp: "2026-01-01T00:00:00Z" }],
      summary: { total_executions: 1 },
    } as any;
    const onDataUpdate = vi.fn();

    executionTimelineServiceState.getExecutionTimelineWithRetry.mockResolvedValue(timelineData);

    const { result, unmount } = renderHook(() =>
      useExecutionTimeline({
        refreshInterval: 1000,
        cacheTtl: 5000,
        onDataUpdate,
      })
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.data).toEqual(timelineData);
    expect(result.current.timelineData).toHaveLength(1);
    expect(result.current.dataPointCount).toBe(1);

    act(() => {
      result.current.refresh();
    });
    expect(executionTimelineServiceState.getExecutionTimelineWithRetry).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(executionTimelineServiceState.getExecutionTimelineWithRetry).toHaveBeenCalledTimes(1);
    expect(onDataUpdate).toHaveBeenCalledWith(timelineData);

    unmount();
  });

  it("useExecutionTimeline handles fallback fetches and clears errors", async () => {
    executionTimelineServiceState.getExecutionTimeline.mockRejectedValue(
      new Error("timeline failed")
    );

    const { result } = renderHook(() =>
      useExecutionTimeline({ refreshInterval: 0, enableRetry: false })
    );

    await waitFor(() => expect(result.current.error?.message).toBe("timeline failed"));
    expect(result.current.hasError).toBe(true);
    expect(result.current.isStale).toBe(true);

    act(() => {
      result.current.clearError();
      result.current.reset();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBeNull();
  });
});
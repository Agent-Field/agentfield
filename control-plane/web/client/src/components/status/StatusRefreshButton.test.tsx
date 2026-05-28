import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  StatusRefreshButton,
  useOptimisticStatusRefresh,
} from "@/components/status/StatusRefreshButton";
import { bulkNodeStatus, refreshNodeStatus } from "@/services/api";
import type { AgentStatus } from "@/types/agentfield";

vi.mock("@/services/api", () => ({
  bulkNodeStatus: vi.fn(),
  refreshNodeStatus: vi.fn(),
}));

const mockRefreshNodeStatus = vi.mocked(refreshNodeStatus);
const mockBulkNodeStatus = vi.mocked(bulkNodeStatus);

describe("StatusRefreshButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("refreshes a single node and calls onRefresh", async () => {
    const user = userEvent.setup();
    const status = { status: "ok", state: "active" } as AgentStatus;
    const onRefresh = vi.fn();

    mockRefreshNodeStatus.mockResolvedValue(status);

    render(
      <StatusRefreshButton nodeId="node-1" onRefresh={onRefresh} variant="ghost" />
    );

    const button = screen.getByRole("button", { name: "Refresh" });
    expect(button).toHaveAttribute("title", "Refresh status for node node-1");

    await user.click(button);

    await waitFor(() => {
      expect(mockRefreshNodeStatus).toHaveBeenCalledWith("node-1");
      expect(onRefresh).toHaveBeenCalledWith(status);
    });
  });

  it("shows refreshing state, bulk refreshes nodes, and hides the label when requested", async () => {
    let resolveRequest: ((value: Record<string, AgentStatus>) => void) | undefined;
    mockBulkNodeStatus.mockReturnValue(
      new Promise((resolve) => {
        resolveRequest = resolve;
      }),
    );

    render(
      <StatusRefreshButton
        nodeIds={["node-1", "node-2"]}
        showLabel={false}
        size="lg"
      />,
    );

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("title", "Refresh status for 2 nodes");
    expect(button).toHaveTextContent("");

    act(() => {
      button.click();
    });

    expect(button).toBeDisabled();
    expect(mockBulkNodeStatus).toHaveBeenCalledWith(["node-1", "node-2"]);

    await act(async () => {
      resolveRequest?.({
        "node-1": { status: "ok", state: "active" } as AgentStatus,
      });
    });

    await waitFor(() => {
      expect(button).not.toBeDisabled();
    });
  });

  it("reports refresh errors", async () => {
    const user = userEvent.setup();
    const onError = vi.fn();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    mockRefreshNodeStatus.mockRejectedValue(new Error("boom"));

    render(<StatusRefreshButton nodeId="node-1" onError={onError} />);

    await user.click(screen.getByRole("button", { name: "Refresh" }));

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith("boom");
    });

    consoleError.mockRestore();
  });

  it("renders a relative last verified label", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T12:00:00Z"));

    render(
      <StatusRefreshButton
        nodeId="node-1"
        showLastVerified
        lastVerified="2026-04-08T11:58:30Z"
        size="sm"
      />,
    );

    expect(screen.getByText("Verified 1m ago")).toBeInTheDocument();
  });
});

describe("useOptimisticStatusRefresh", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("stores, reads, clears, and expires optimistic status updates", () => {
    const { result } = renderHook(() => useOptimisticStatusRefresh());
    const status = { status: "ok", state: "starting" } as AgentStatus;

    act(() => {
      result.current.setOptimisticStatus("node-1", status);
    });

    expect(result.current.hasOptimisticUpdate("node-1")).toBe(true);
    expect(result.current.getOptimisticStatus("node-1")).toEqual(status);

    act(() => {
      result.current.clearOptimisticStatus("node-1");
    });

    expect(result.current.hasOptimisticUpdate("node-1")).toBe(false);
    expect(result.current.getOptimisticStatus("node-1")).toBeNull();

    act(() => {
      result.current.setOptimisticStatus("node-2", status);
      vi.advanceTimersByTime(5000);
    });

    expect(result.current.hasOptimisticUpdate("node-2")).toBe(false);
    expect(result.current.getOptimisticStatus("node-2")).toBeNull();
  });
});

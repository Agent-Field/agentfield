// @ts-nocheck
import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const navigate = vi.fn();
const apiMocks = vi.hoisted(() => ({
  executeReasonerAsync: vi.fn(),
  getExecutionStatus: vi.fn(),
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => navigate,
  };
});

vi.mock("../../services/reasonersApi", () => ({
  reasonersApi: {
    executeReasonerAsync: apiMocks.executeReasonerAsync,
    getExecutionStatus: apiMocks.getExecutionStatus,
  },
}));

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(
    ({ className }, ref) => <svg ref={ref} data-testid="icon" className={className} />,
  );
  Icon.displayName = "Icon";
  return {
    InProgress: Icon,
    Close: Icon,
    Copy: Icon,
    Time: Icon,
    CheckmarkFilled: Icon,
    ErrorFilled: Icon,
    Analytics: Icon,
    Launch: Icon,
  };
});

import { ExecutionQueue, type ExecutionQueueRef } from "./ExecutionQueue";

describe("ExecutionQueue coverage paths", () => {
  beforeEach(() => {
    navigate.mockReset();
    apiMocks.executeReasonerAsync.mockReset();
    apiMocks.getExecutionStatus.mockReset();
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn() },
    });
  });

  it("summarizes queued inputs, cancels them into recents, copies data, and toggles inline selection", async () => {
    const onExecutionSelect = vi.fn();
    const ref = React.createRef<ExecutionQueueRef>();

    render(
      <ExecutionQueue
        ref={ref}
        reasonerId="reasoner-1"
        maxConcurrent={0}
        onExecutionSelect={onExecutionSelect}
      />,
    );

    act(() => {
      ref.current?.addExecution({});
      ref.current?.addExecution({ alpha: "one", beta: "two" });
    });

    expect(await screen.findByText("Active Executions (2)")).toBeInTheDocument();
    expect(screen.getByText("Empty object")).toBeInTheDocument();
    expect(screen.getByText("2 parameters")).toBeInTheDocument();
    expect(apiMocks.executeReasonerAsync).not.toHaveBeenCalled();

    fireEvent.click(screen.getAllByLabelText("Cancel execution")[0]);
    fireEvent.click(screen.getAllByLabelText("Cancel execution")[0]);

    expect(await screen.findByText("Recent Executions")).toBeInTheDocument();
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
    expect(screen.queryByText("navigable")).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByLabelText("Copy execution data")[0]);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining('"status": "failed"'),
    );

    const recentCards = screen.getAllByRole("button", { name: "View execution details" });
    fireEvent.click(recentCards[0]);
    fireEvent.keyDown(recentCards[0], { key: " " });

    expect(onExecutionSelect).toHaveBeenCalledTimes(2);
    expect(onExecutionSelect.mock.calls[0][0].status).toBe("failed");
    expect(onExecutionSelect.mock.calls[1][0]).toBeNull();
    expect(navigate).not.toHaveBeenCalled();
  });

  it("summarizes raw string input when async startup rejects immediately", async () => {
    const ref = React.createRef<ExecutionQueueRef>();
    apiMocks.executeReasonerAsync.mockRejectedValue(new Error("boom"));

    render(<ExecutionQueue ref={ref} reasonerId="reasoner-2" />);

    act(() => {
      ref.current?.addExecution("raw coverage input");
    });

    await waitFor(() => {
      expect(screen.getByText("Recent Executions")).toBeInTheDocument();
    });
    expect(screen.getByText("raw coverage input")).toBeInTheDocument();
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
  });
});

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

const flush = async () => {
  await act(async () => {
    await Promise.resolve();
  });
};

describe("ExecutionQueue", () => {
  beforeEach(() => {
    navigate.mockReset();
    apiMocks.executeReasonerAsync.mockReset();
    apiMocks.getExecutionStatus.mockReset();
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn() },
    });
  });

  it("returns null until an execution is added", () => {
    const { container } = render(<ExecutionQueue reasonerId="reasoner-1" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("adds an execution, completes it, copies payload data, and navigates via the detail controls", async () => {
    const ref = React.createRef<ExecutionQueueRef>();
    apiMocks.executeReasonerAsync.mockResolvedValue({
      execution_id: "backend-1",
      workflow_id: "workflow-1",
      run_id: "run-1",
      status: "accepted",
      target: "reasoner-1",
      type: "async",
      created_at: "2026-04-09T12:00:00.000Z",
    });
    apiMocks.getExecutionStatus
      .mockResolvedValueOnce({
        execution_id: "backend-1",
        workflow_id: "workflow-1",
        run_id: "run-1",
        status: "running",
        target: "reasoner-1",
        type: "async",
        progress: 50,
        started_at: "2026-04-09T12:00:00.000Z",
      })
      .mockResolvedValueOnce({
        execution_id: "backend-1",
        workflow_id: "workflow-1",
        run_id: "run-1",
        status: "completed",
        target: "reasoner-1",
        type: "async",
        progress: 100,
        started_at: "2026-04-09T12:00:00.000Z",
        duration: 321,
        result: { output: "done" },
      });

    render(<ExecutionQueue ref={ref} reasonerId="reasoner-1" />);

    act(() => {
      ref.current?.addExecution({ prompt: "hello world" });
    });
    await flush();

    expect(await screen.findByText("Recent Executions")).toBeInTheDocument();
    expect(screen.getByText("prompt: hello world")).toBeInTheDocument();
    expect(screen.getAllByText("completed").length).toBeGreaterThan(0);
    expect(screen.getByText("navigable")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Copy execution data"));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      JSON.stringify(
        {
          input: { prompt: "hello world" },
          result: { output: "done" },
          duration: 321,
          status: "completed",
        },
        null,
        2,
      ),
    );

    fireEvent.click(screen.getByLabelText("View execution details"));
    fireEvent.keyDown(screen.getByLabelText("Navigate to execution backend-1"), { key: "Enter" });

    expect(navigate).toHaveBeenNthCalledWith(1, "/executions/backend-1");
    expect(navigate).toHaveBeenNthCalledWith(2, "/executions/backend-1");
  });

  it("queues a second execution, allows cancellation, and toggles recent selection callbacks", async () => {
    const ref = React.createRef<ExecutionQueueRef>();
    const onExecutionSelect = vi.fn();
    const deferred = {} as { promise: Promise<unknown>; resolve: (value: unknown) => void };
    deferred.promise = new Promise((resolve) => {
      deferred.resolve = resolve;
    });
    apiMocks.executeReasonerAsync.mockImplementationOnce(() => deferred.promise);

    render(<ExecutionQueue ref={ref} reasonerId="reasoner-1" maxConcurrent={1} onExecutionSelect={onExecutionSelect} />);

    act(() => {
      ref.current?.addExecution({ first: "execution" });
    });
    await flush();

    apiMocks.executeReasonerAsync.mockResolvedValue({
      execution_id: "backend-queued",
      workflow_id: "workflow-queued",
      status: "accepted",
      target: "reasoner-1",
      type: "async",
      created_at: "2026-04-09T12:00:00.000Z",
    });

    act(() => {
      ref.current?.addExecution({ second: "execution" });
    });

    expect(await screen.findByText("Active Executions (2)")).toBeInTheDocument();
    fireEvent.click(screen.getAllByLabelText("Cancel execution")[1]);

    await waitFor(() => {
      expect(screen.getByText("Recent Executions")).toBeInTheDocument();
    });

    const detailsCard = screen.getByLabelText("View execution details");
    expect(detailsCard).toBeInTheDocument();

    const viewButtons = screen.getAllByLabelText("View execution details");
    const card = viewButtons[viewButtons.length - 1].closest("[role='button']");
    fireEvent.click(card as HTMLElement);
    fireEvent.click(card as HTMLElement);

    expect(onExecutionSelect).toHaveBeenCalledTimes(2);
    expect(onExecutionSelect.mock.calls[0][0].status).toBe("failed");
    expect(onExecutionSelect.mock.calls[1][0]).toBeNull();
  });

  it("marks an execution failed when async startup rejects", async () => {
    const ref = React.createRef<ExecutionQueueRef>();
    apiMocks.executeReasonerAsync.mockRejectedValue(new Error("boom"));

    render(<ExecutionQueue ref={ref} reasonerId="reasoner-1" />);

    act(() => {
      ref.current?.addExecution("raw input");
    });
    await flush();

    expect(await screen.findByText("Recent Executions")).toBeInTheDocument();
    expect(screen.getByText("raw input")).toBeInTheDocument();
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
  });
});

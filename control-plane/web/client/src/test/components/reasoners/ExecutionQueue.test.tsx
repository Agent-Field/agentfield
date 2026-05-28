import React, { createRef } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionQueue, type ExecutionQueueRef } from "@/components/reasoners/ExecutionQueue";

const navigateMock = vi.fn();
const executeReasonerAsyncMock = vi.fn();
const getExecutionStatusMock = vi.fn();
const clipboardWriteTextMock = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

vi.mock("@/services/reasonersApi", () => ({
  reasonersApi: {
    executeReasonerAsync: (...args: unknown[]) => executeReasonerAsyncMock(...args),
    getExecutionStatus: (...args: unknown[]) => getExecutionStatusMock(...args),
  },
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  InProgress: () => <span>in-progress</span>,
  Close: () => <span>close</span>,
  Copy: () => <span>copy</span>,
  Time: () => <span>time</span>,
  CheckmarkFilled: () => <span>checkmark</span>,
  ErrorFilled: () => <span>error</span>,
  Analytics: () => <span>analytics</span>,
  Launch: () => <span>launch</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

describe("ExecutionQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {},
      });
    }
    Object.defineProperty(navigator.clipboard, "writeText", {
      configurable: true,
      writable: true,
      value: clipboardWriteTextMock,
    });
    clipboardWriteTextMock.mockResolvedValue(undefined);
  });

  it("adds an execution, completes it, and navigates to the execution page", async () => {
    const ref = createRef<ExecutionQueueRef>();

    executeReasonerAsyncMock.mockResolvedValue({
      execution_id: "backend-exec-1",
      workflow_id: "wf-1",
      run_id: "run-1",
    });
    getExecutionStatusMock.mockResolvedValue({
      execution_id: "backend-exec-1",
      status: "completed",
      result: { output: "done" },
      duration: 1234,
    });

    render(
      <ExecutionQueue
        ref={ref}
        reasonerId="planner"
      />
    );

    act(() => {
      ref.current?.addExecution({ prompt: "hello" });
    });

    expect(await screen.findAllByText("completed")).toHaveLength(1);
    expect(screen.getByText(/Recent Executions/i)).toBeInTheDocument();

    const viewButtons = await screen.findAllByRole("button", {
      name: /view execution details/i,
    });
    await userEvent.setup().click(viewButtons[0]);

    expect(navigateMock).toHaveBeenCalledWith("/executions/backend-exec-1");
    expect(executeReasonerAsyncMock).toHaveBeenCalledWith(
      "planner",
      expect.objectContaining({
        input: { prompt: "hello" },
        context: expect.objectContaining({ session_id: expect.stringMatching(/^session_/) }),
      })
    );
  });

  it("shows failure state and copies execution data", async () => {
    const ref = createRef<ExecutionQueueRef>();

    executeReasonerAsyncMock.mockRejectedValue(new Error("network down"));

    render(<ExecutionQueue ref={ref} reasonerId="planner" />);

    act(() => {
      ref.current?.addExecution({ prompt: "fail please" });
    });

    expect(await screen.findByText(/Recent Executions/i)).toBeInTheDocument();
    expect(await screen.findAllByText("failed")).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: /copy execution data/i }));

    await waitFor(() =>
      expect(clipboardWriteTextMock).toHaveBeenCalledWith(
        expect.stringContaining('"status": "failed"')
      )
    );
  });
});

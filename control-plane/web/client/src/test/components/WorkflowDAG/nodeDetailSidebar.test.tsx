// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NodeDetailSidebar } from "@/components/WorkflowDAG/NodeDetailSidebar";

const mockUseNodeDetails = vi.fn();

vi.mock("@/components/ui/icon-bridge", () => ({
  Close: () => <span>close</span>,
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
  Card: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardContent: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
  CardHeader: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <div className={className}>{children}</div>
  ),
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
}));

vi.mock("@/lib/theme", () => ({
  statusTone: {
    error: {
      bg: "error-bg",
      border: "error-border",
      accent: "error-accent",
    },
  },
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/components/WorkflowDAG/hooks/useNodeDetails", () => ({
  useNodeDetails: (...args: unknown[]) => mockUseNodeDetails(...args),
}));

vi.mock("@/components/WorkflowDAG/sections/DataSection", () => ({
  DataSection: () => <div>Data section</div>,
}));

vi.mock("@/components/WorkflowDAG/sections/ExecutionHeader", () => ({
  ExecutionHeader: ({ copySuccess }: { copySuccess?: string | null }) => (
    <div>Execution header {copySuccess ?? "none"}</div>
  ),
}));

vi.mock("@/components/WorkflowDAG/sections/TimingSection", () => ({
  TimingSection: () => <div>Timing section</div>,
}));

vi.mock("@/components/WorkflowDAG/sections/TechnicalSection", () => ({
  TechnicalSection: () => <div>Technical section</div>,
}));

const node = {
  workflow_id: "wf-1",
  execution_id: "exec-1",
  agent_node_id: "agent-1",
  reasoner_id: "reasoner",
  status: "running",
  started_at: "2026-04-08T10:00:00Z",
  workflow_depth: 1,
  task_name: "My Task",
};

describe("NodeDetailSidebar", () => {
  beforeEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
    mockUseNodeDetails.mockReset();
  });

  it("renders loaded content, refetches on open, and closes from interactions", async () => {
    const refetch = vi.fn();
    const onClose = vi.fn();
    const user = userEvent.setup();

    mockUseNodeDetails.mockReturnValue({
      nodeDetails: { input: { ok: true } },
      loading: false,
      error: null,
      refetch,
    });

    render(<NodeDetailSidebar node={node} isOpen onClose={onClose} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Execution Details")).toBeInTheDocument();
    expect(screen.getByText("My Task")).toBeInTheDocument();
    expect(screen.getByText("Data section")).toBeInTheDocument();
    expect(screen.getByText("Timing section")).toBeInTheDocument();
    expect(screen.getByText("Technical section")).toBeInTheDocument();
    expect(refetch).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.click(screen.getByLabelText("Close sidebar"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("renders the loading skeleton state", () => {
    mockUseNodeDetails.mockReturnValue({
      nodeDetails: undefined,
      loading: true,
      error: null,
      refetch: vi.fn(),
    });

    render(<NodeDetailSidebar node={node} isOpen onClose={vi.fn()} />);

    expect(screen.getAllByTestId("skeleton")).toHaveLength(20);
  });

  it("renders the error state and retries", async () => {
    const user = userEvent.setup();
    const refetch = vi.fn();

    mockUseNodeDetails.mockReturnValue({
      nodeDetails: undefined,
      loading: false,
      error: "network failed",
      refetch,
    });

    render(<NodeDetailSidebar node={node} isOpen onClose={vi.fn()} />);

    expect(screen.getByText("Failed to load execution details")).toBeInTheDocument();
    expect(screen.getByText("network failed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(refetch).toHaveBeenCalled();
  });

  it("returns null when closed or missing a node", () => {
    mockUseNodeDetails.mockReturnValue({
      nodeDetails: undefined,
      loading: false,
      error: null,
      refetch: vi.fn(),
    });

    const { container, rerender } = render(
      <NodeDetailSidebar node={null} isOpen={false} onClose={vi.fn()} />
    );

    expect(container).toBeEmptyDOMElement();

    rerender(<NodeDetailSidebar node={null} isOpen onClose={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});
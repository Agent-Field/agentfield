import { fireEvent, render, screen } from "@testing-library/react";
import type { ButtonHTMLAttributes, HTMLAttributes, PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CANCEL_RUN_COPY, RunLifecycleMenu } from "@/components/runs/RunLifecycleMenu";
import type { WorkflowSummary } from "@/types/workflows";

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    ...props
  }: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: PropsWithChildren<{ open?: boolean }>) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: PropsWithChildren<{ asChild?: boolean }>) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
    ...props
  }: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
  DropdownMenuLabel: ({ children }: PropsWithChildren) => <div>{children}</div>,
  DropdownMenuSeparator: () => <div data-testid="separator" />,
}));

vi.mock("@/components/ui/alert-dialog", () => ({
  AlertDialog: ({ children, open }: PropsWithChildren<{ open?: boolean }>) => (
    <div data-state={open ? "open" : "closed"}>{children}</div>
  ),
  AlertDialogAction: ({
    children,
    onClick,
    ...props
  }: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
  AlertDialogCancel: ({
    children,
    ...props
  }: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
  AlertDialogContent: ({ children }: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: PropsWithChildren) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: PropsWithChildren) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: PropsWithChildren) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: PropsWithChildren) => <div>{children}</div>,
}));

function makeRun(overrides: Partial<WorkflowSummary> = {}): WorkflowSummary {
  return {
    run_id: "run-1",
    workflow_id: "wf-1",
    root_execution_id: "exec-1",
    root_execution_status: "running",
    status: "running",
    root_reasoner: "root",
    current_task: "Processing",
    total_executions: 3,
    max_depth: 2,
    started_at: "2026-04-08T09:00:00Z",
    latest_activity: "2026-04-08T09:05:00Z",
    display_name: "Run 1",
    status_counts: {},
    active_executions: 1,
    terminal: false,
    ...overrides,
  };
}

describe("RunLifecycleMenu", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a placeholder when no actions are available", () => {
    const { container } = render(
      <RunLifecycleMenu
        run={makeRun({ root_execution_id: undefined, status: "succeeded", root_execution_status: "succeeded", terminal: true })}
        isPending={false}
        onPause={vi.fn()}
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(container.querySelector("span[aria-hidden='true']")).not.toBeNull();
    expect(screen.queryByRole("button", { name: /run actions/i })).not.toBeInTheDocument();
  });

  it("invokes pause and shows the running menu state", () => {
    const onPause = vi.fn();
    const run = makeRun();

    render(
      <RunLifecycleMenu
        run={run}
        isPending={false}
        onPause={onPause}
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("Lifecycle")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Pause run" }));

    expect(onPause).toHaveBeenCalledWith(run);
    expect(screen.queryByText("Resume run")).not.toBeInTheDocument();
  });

  it("invokes resume for paused runs", () => {
    const onResume = vi.fn();
    const run = makeRun({ root_execution_status: "paused", status: "running" });

    render(
      <RunLifecycleMenu
        run={run}
        isPending={false}
        onPause={vi.fn()}
        onResume={onResume}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Resume run" }));
    expect(onResume).toHaveBeenCalledWith(run);
    expect(screen.queryByText("Pause run")).not.toBeInTheDocument();
  });

  it("opens the cancel confirmation and invokes cancel", () => {
    const onCancel = vi.fn();
    const run = makeRun();

    render(
      <RunLifecycleMenu
        run={run}
        isPending={false}
        onPause={vi.fn()}
        onResume={vi.fn()}
        onCancel={onCancel}
      />,
    );

    expect(CANCEL_RUN_COPY.title(2)).toBe("Cancel 2 runs?");

    fireEvent.click(screen.getAllByRole("button", { name: "Cancel run" })[0]);
    expect(screen.getByText(CANCEL_RUN_COPY.description)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: CANCEL_RUN_COPY.keepLabel })).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: CANCEL_RUN_COPY.confirmLabel(1) })[1]);
    expect(onCancel).toHaveBeenCalledWith(run);
  });

  it("disables the trigger while pending", () => {
    render(
      <RunLifecycleMenu
        run={makeRun()}
        isPending
        onPause={vi.fn()}
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Run actions for run-1" })).toBeDisabled();
  });
});

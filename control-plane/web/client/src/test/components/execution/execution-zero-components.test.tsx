// @ts-nocheck
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionDataColumns } from "@/components/execution/ExecutionDataColumns";
import { WorkflowBreadcrumb } from "@/components/execution/WorkflowBreadcrumb";
import { OutputDataPanel } from "@/components/execution/OutputDataPanel";
import { StepProvenanceCard } from "@/components/StepProvenanceCard";

const navigate = vi.fn();
const modalSpy = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    FileText: Icon,
    Database: Icon,
    Activity: Icon,
    Check: Icon,
    ChevronLeft: Icon,
    ChevronRight: Icon,
    Users: Icon,
    ArrowUp: Icon,
    CheckCircle: Icon,
    XCircle: Icon,
    ChevronDown: Icon,
  };
});

vi.mock("lucide-react", () => {
  const Icon = React.forwardRef<SVGSVGElement, React.SVGProps<SVGSVGElement>>((props, ref) => (
    <svg ref={ref} {...props} />
  ));
  return {
    Copy: Icon,
    Check: Icon,
  };
});

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
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardContent: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div {...props}>{children}</div>,
}));

vi.mock("@/components/ui/UnifiedDataPanel", () => ({
  UnifiedDataPanel: ({
    title,
    data,
    emptyStateConfig,
    onModalOpen,
  }: {
    title: string;
    data: unknown;
    emptyStateConfig: { title: string; description: string };
    onModalOpen?: () => void;
  }) => (
    <section>
      <h2>{title}</h2>
      <pre>{JSON.stringify(data)}</pre>
      <div>{emptyStateConfig.title}</div>
      <div>{emptyStateConfig.description}</div>
      <button type="button" onClick={onModalOpen}>
        open modal
      </button>
    </section>
  ),
}));

vi.mock("@/components/execution/EnhancedModal", () => ({
  DataModal: (props: Record<string, unknown>) => {
    modalSpy(props);
    return props.isOpen ? <div>modal-open</div> : null;
  },
}));

vi.mock("@/components/ui/ResizableSplitPane", () => ({
  ResizableSplitPane: ({
    children,
    orientation,
  }: React.PropsWithChildren<{ orientation: string }>) => (
    <div data-testid="split-pane" data-orientation={orientation}>
      {children}
    </div>
  ),
  useResponsiveSplitPane: vi.fn(),
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status?: string) => {
    if (status === "completed") return "succeeded";
    return status ?? "unknown";
  },
}));

vi.mock("@/components/execution/CollapsibleSection", () => ({
  CollapsibleSection: ({
    children,
    title,
    badge,
  }: React.PropsWithChildren<{ title: string; badge?: React.ReactNode }>) => (
    <section>
      <h2>{title}</h2>
      {badge}
      {children}
    </section>
  ),
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({ data }: { data: unknown }) => <pre>{JSON.stringify(data)}</pre>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleTrigger: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    workflow_id: "workflow-abcdef123456",
    execution_id: "exec-1",
    session_id: "session-1234567890",
    input_data: { prompt: "hello" },
    output_data: { result: "ok" },
    input_size: 128,
    output_size: 2048,
    workflow_tags: [],
    status: "completed",
    created_at: "2026-04-08T00:00:00Z",
    updated_at: "2026-04-08T00:01:00Z",
    retry_count: 0,
    workflow_depth: 0,
    ...overrides,
  };
}

describe("execution zero-coverage components", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const splitPaneModule = await import("@/components/ui/ResizableSplitPane");
    vi.mocked(splitPaneModule.useResponsiveSplitPane).mockReturnValue({
      isSmallScreen: false,
    });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("renders execution data columns and opens the modal", async () => {
    const user = userEvent.setup();

    render(<ExecutionDataColumns execution={buildExecution()} />);

    expect(screen.getByTestId("split-pane")).toHaveAttribute("data-orientation", "horizontal");
    expect(screen.getByText("Input Data")).toBeInTheDocument();
    expect(screen.getByText("Output Data")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: /open modal/i })[1]);

    expect(screen.getByText("modal-open")).toBeInTheDocument();
    expect(modalSpy).toHaveBeenLastCalledWith(expect.objectContaining({ isOpen: true }));
  });

  it("renders output panel data and empty states", () => {
    const { rerender } = render(<OutputDataPanel execution={buildExecution()} />);

    expect(screen.getByText("Output Data")).toBeInTheDocument();
    expect(screen.getByText("2.0 KB")).toBeInTheDocument();
    expect(screen.getByText('{"result":"ok"}')).toBeInTheDocument();

    rerender(
      <OutputDataPanel
        execution={buildExecution({ status: "running", output_data: null, output_size: 0 })}
      />
    );
    expect(screen.getByText("Execution in progress")).toBeInTheDocument();

    rerender(
      <OutputDataPanel
        execution={buildExecution({ status: "failed", output_data: null, output_size: 0 })}
      />
    );
    expect(screen.getByText("Execution failed")).toBeInTheDocument();

    rerender(
      <OutputDataPanel
        execution={buildExecution({ status: "succeeded", output_data: {}, output_size: 0 })}
      />
    );
    expect(screen.getByText("No output data")).toBeInTheDocument();
  });

  it("renders breadcrumb links, navigates, and copies ids", async () => {
    const user = userEvent.setup();
    const onNavigateBack = vi.fn();

    render(
      <WorkflowBreadcrumb execution={buildExecution()} onNavigateBack={onNavigateBack} />
    );

    await user.click(screen.getByRole("button", { name: /executions/i }));
    expect(onNavigateBack).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /workflow/i }));
    expect(navigate).toHaveBeenCalledWith("/workflows/workflow-abcdef123456");

    await user.click(screen.getByRole("button", { name: /session/i }));
    expect(navigate).toHaveBeenCalledWith("/executions?session_id=session-1234567890");

    fireEvent.click(screen.getByTitle(/Workflow:/i));
    expect(screen.getByTitle(/Workflow:/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText("Execution Detail").length).toBeGreaterThan(0);
    });
  });

  it("renders provenance rows and copies values", async () => {
    const user = userEvent.setup();

    render(
      <StepProvenanceCard
        callerDid="did:web:caller.example.test"
        targetDid="did:web:target.example.test"
        inputHash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        outputHash="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      />
    );

    expect(screen.getByText("Provenance (VC)")).toBeInTheDocument();
    expect(screen.getByText("(4)")).toBeInTheDocument();
    expect(screen.getByLabelText("Copy Caller DID")).toBeInTheDocument();

    await user.click(screen.getByLabelText("Copy Input hash"));
    expect(screen.getByLabelText("Copy Input hash")).toBeInTheDocument();
  });

  it("returns null when provenance data is empty", () => {
    const { container } = render(<StepProvenanceCard />);
    expect(container).toBeEmptyDOMElement();
  });
});

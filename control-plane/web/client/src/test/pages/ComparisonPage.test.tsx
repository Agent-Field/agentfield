import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ComparisonPage } from "@/pages/ComparisonPage";
import type { WorkflowExecution } from "@/types/executions";
import type { WorkflowDAGLightweightResponse } from "@/types/workflows";

const comparisonState = vi.hoisted(() => ({
  navigate: vi.fn<(value: string) => void>(),
  search: "",
  dagA: null as {
    data?: WorkflowDAGLightweightResponse;
    isLoading: boolean;
    isError: boolean;
    error?: Error;
  } | null,
  dagB: null as {
    data?: WorkflowDAGLightweightResponse;
    isLoading: boolean;
    isError: boolean;
    error?: Error;
  } | null,
  stepDetails: {} as Record<string, WorkflowExecution>,
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => comparisonState.navigate,
  useSearchParams: () => [new URLSearchParams(comparisonState.search)],
}));

vi.mock("@/hooks/queries", () => ({
  useRunDAG: (runId?: string) => {
    if (runId === "run-a") {
      return comparisonState.dagA ?? { isLoading: false, isError: false };
    }
    if (runId === "run-b") {
      return comparisonState.dagB ?? { isLoading: false, isError: false };
    }
    return { isLoading: false, isError: false };
  },
  useStepDetail: (executionId?: string) => ({
    data: executionId ? comparisonState.stepDetails[executionId] : undefined,
    isLoading: false,
    isError: false,
    error: null,
  }),
}));

vi.mock("@/components/RunTrace", () => ({
  formatDuration: (value?: number) => (value == null ? "-" : `${value}ms`),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h2 {...props}>{children}</h2>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children, showIcon: _showIcon, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement> & { showIcon?: boolean }>) => (
    <span {...props}>{children}</span>
  ),
  StatusBadge: ({ children, showIcon: _showIcon, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement> & { showIcon?: boolean }>) => (
    <span {...props}>{children}</span>
  ),
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />,
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLTableElement>>) => (
    <table {...props}>{children}</table>
  ),
  TableHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLTableSectionElement>>) => (
    <thead {...props}>{children}</thead>
  ),
  TableBody: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLTableSectionElement>>) => (
    <tbody {...props}>{children}</tbody>
  ),
  TableRow: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLTableRowElement>>) => (
    <tr {...props}>{children}</tr>
  ),
  TableHead: ({ children, ...props }: React.PropsWithChildren<React.ThHTMLAttributes<HTMLTableCellElement>>) => (
    <th {...props}>{children}</th>
  ),
  TableCell: ({ children, ...props }: React.PropsWithChildren<React.TdHTMLAttributes<HTMLTableCellElement>>) => (
    <td {...props}>{children}</td>
  ),
}));

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>loading</div>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>{children}</button>
  ),
  TabsContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ text, ...props }: { text: string } & React.HTMLAttributes<HTMLPreElement>) => (
    <pre {...props}>{text}</pre>
  ),
}));

vi.mock("@/utils/reasonerCompareExtract", () => ({
  extractReasonerInputLayers: () => ({ prose: ["prompt"], meta: ["context"] }),
  formatOutputUsageHint: () => "tool output",
}));

vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>();
  const Icon = ({ className }: { className?: string }) => <span className={className}>icon</span>;
  return {
    ...actual,
    AlertTriangle: Icon,
    ArrowLeft: Icon,
    ChevronDown: Icon,
    Equal: Icon,
    ExternalLink: Icon,
    Minus: Icon,
  };
});

function createDag(runId: string, workflowStatus: string): WorkflowDAGLightweightResponse {
  return {
    root_workflow_id: runId,
    workflow_status: workflowStatus,
    workflow_name: `${runId}-workflow`,
    total_nodes: 2,
    max_depth: 1,
    mode: "lightweight",
    timeline: [
      {
        execution_id: `${runId}-step-1`,
        agent_node_id: `${runId}-agent`,
        reasoner_id: `${runId}-planner`,
        status: workflowStatus,
        started_at: "2026-04-07T12:00:00Z",
        duration_ms: 1000,
        workflow_depth: 0,
      },
      {
        execution_id: `${runId}-step-2`,
        parent_execution_id: `${runId}-step-1`,
        agent_node_id: `${runId}-agent`,
        reasoner_id: `${runId}-writer`,
        status: workflowStatus === "failed" ? "failed" : "succeeded",
        started_at: "2026-04-07T12:01:00Z",
        completed_at: "2026-04-07T12:02:00Z",
        duration_ms: 2000,
        workflow_depth: 1,
      },
    ],
  };
}

function createStepDetail(executionId: string, status: WorkflowExecution["status"]): WorkflowExecution {
  return {
    id: 1,
    workflow_id: executionId.replace(/-step-.+$/, ""),
    execution_id: executionId,
    agentfield_request_id: `req-${executionId}`,
    agent_node_id: `${executionId}-agent`,
    workflow_depth: executionId.endsWith("1") ? 0 : 1,
    reasoner_id: executionId.endsWith("1") ? "planner" : "writer",
    input_data: { prompt: executionId },
    output_data: { result: `${executionId}-result` },
    input_size: 512,
    output_size: 256,
    workflow_name: `${executionId}-workflow`,
    workflow_tags: ["ops"],
    status,
    started_at: "2026-04-07T12:00:00Z",
    completed_at: "2026-04-07T12:02:00Z",
    duration_ms: 2000,
    retry_count: executionId.endsWith("2") ? 1 : 0,
    created_at: "2026-04-07T12:00:00Z",
    updated_at: "2026-04-07T12:02:00Z",
    error_message: status === "failed" ? "validation failed" : undefined,
    notes: [{ message: `note-${executionId}`, tags: ["review"], timestamp: "2026-04-07T12:01:30Z" }],
  };
}

describe("ComparisonPage", () => {
  beforeEach(() => {
    comparisonState.navigate.mockReset();
    comparisonState.search = "";
    comparisonState.dagA = { isLoading: false, isError: false, data: createDag("run-a", "running") };
    comparisonState.dagB = { isLoading: false, isError: false, data: createDag("run-b", "failed") };
    comparisonState.stepDetails = {
      "run-a-step-1": createStepDetail("run-a-step-1", "running"),
      "run-a-step-2": createStepDetail("run-a-step-2", "succeeded"),
      "run-b-step-1": createStepDetail("run-b-step-1", "failed"),
      "run-b-step-2": createStepDetail("run-b-step-2", "failed"),
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the empty-state call to action when run ids are missing", () => {
    render(<ComparisonPage />);

    expect(screen.getByText("Compare Runs")).toBeInTheDocument();
    expect(screen.getByText(/Select two runs from the Runs page/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Back to Runs/i }));
    expect(comparisonState.navigate).toHaveBeenCalledWith("/runs");
  });

  it("shows the dag loading error state", () => {
    comparisonState.search = "a=run-a&b=run-b";
    comparisonState.dagA = {
      isLoading: false,
      isError: true,
      error: new Error("dag unavailable"),
    };

    render(<ComparisonPage />);

    expect(screen.getByText(/Failed to load Run A: dag unavailable/i)).toBeInTheDocument();
  });

  it("renders run summaries and expanded step comparison details", async () => {
    comparisonState.search = "a=run-a&b=run-b";
    render(<ComparisonPage />);

    expect(screen.getAllByText("run-a-workflow").length).toBeGreaterThan(0);
    expect(screen.getAllByText("run-b-workflow").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/1000ms/).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByRole("button", { name: /Detail/i })[0]);
    expect(comparisonState.navigate).toHaveBeenCalledWith("/runs/run-a");

    fireEvent.click(screen.getAllByText("run-a-planner")[0]);

    await waitFor(() => {
      expect(
        screen.getByRole("region", { name: /Step 1 comparison details for runs A and B/i }),
      ).toBeInTheDocument();
    });

    expect(screen.getAllByText(/Step comparison/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/tool output/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/note-run-a-step-1/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/validation failed/i).length).toBeGreaterThan(0);
  });
});

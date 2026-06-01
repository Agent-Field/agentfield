// @ts-nocheck
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { PlaygroundPage } from "@/pages/PlaygroundPage";
import type { ReasonerWithNode, ReasonersResponse } from "@/types/reasoners";

const state = vi.hoisted(() => ({
  getAllReasoners: vi.fn<() => Promise<ReasonersResponse>>(),
  getReasonerDetails: vi.fn<(reasonerId: string) => Promise<ReasonerWithNode>>(),
  getExecutionHistory: vi.fn<(reasonerId: string, page: number, limit: number) => Promise<any>>(),
  executeReasoner: vi.fn<(reasonerId: string, body: unknown) => Promise<any>>(),
  writeText: vi.fn<(value: string) => Promise<void>>(),
}));

vi.mock("@/services/reasonersApi", () => ({
  reasonersApi: {
    getAllReasoners: (...args: Parameters<typeof state.getAllReasoners>) => state.getAllReasoners(...args),
    getReasonerDetails: (reasonerId: string) => state.getReasonerDetails(reasonerId),
    getExecutionHistory: (reasonerId: string, page: number, limit: number) =>
      state.getExecutionHistory(reasonerId, page, limit),
    executeReasoner: (reasonerId: string, body: unknown) => state.executeReasoner(reasonerId, body),
  },
}));

vi.mock("@/components/ui/reasoner-node-combobox", () => ({
  ReasonerNodeCombobox: ({
    reasoners,
    value,
    onValueChange,
    disabled,
  }: {
    reasoners: Array<{ reasoner_id: string; name: string }>;
    value: string | null;
    onValueChange: (value: string) => void;
    disabled?: boolean;
  }) => (
    <select
      aria-label="Reasoner / skill"
      value={value ?? ""}
      disabled={disabled}
      onChange={(event) => onValueChange(event.target.value)}
    >
      <option value="">Select</option>
      {reasoners.map((reasoner) => (
        <option key={reasoner.reasoner_id} value={reasoner.reasoner_id}>
          {reasoner.name}
        </option>
      ))}
    </select>
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data }: { data: unknown }) => (
    <pre>{JSON.stringify(data, null, 2)}</pre>
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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
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

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
  }: React.PropsWithChildren<{ onClick?: () => void }>) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <div>separator</div>,
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: React.PropsWithChildren) => <table>{children}</table>,
  TableHeader: ({ children }: React.PropsWithChildren) => <thead>{children}</thead>,
  TableBody: ({ children }: React.PropsWithChildren) => <tbody>{children}</tbody>,
  TableRow: ({ children }: React.PropsWithChildren) => <tr>{children}</tr>,
  TableHead: ({ children }: React.PropsWithChildren) => <th>{children}</th>,
  TableCell: ({ children }: React.PropsWithChildren) => <td>{children}</td>,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = () => <span>icon</span>;
  return {
    Play: Icon,
    InProgress: Icon,
    ArrowRight: Icon,
    Upload: Icon,
    Copy: Icon,
    Check: Icon,
    ChevronRight: Icon,
    ChevronDown: Icon,
  };
});

function buildReasoner(overrides: Partial<ReasonerWithNode> = {}): ReasonerWithNode {
  return {
    reasoner_id: "node-1.reasoner-1",
    name: "Planner",
    description: "Test reasoner",
    node_id: "node-1",
    node_status: "active",
    node_version: "1.0.0",
    input_schema: { type: "object", properties: { prompt: { type: "string" } } },
    output_schema: { type: "object", properties: { answer: { type: "string" } } },
    memory_config: {
      auto_inject: [],
      memory_retention: "short",
      cache_results: false,
    },
    last_updated: "2026-04-08T00:00:00Z",
    ...overrides,
  };
}

function renderPage(initialPath = "/playground/node-1.reasoner-1") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/playground/:reasonerId" element={<PlaygroundPage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("PlaygroundPage", () => {
  beforeEach(() => {
    state.getAllReasoners.mockReset();
    state.getReasonerDetails.mockReset();
    state.getExecutionHistory.mockReset();
    state.executeReasoner.mockReset();
    state.writeText.mockReset();
    vi.stubGlobal("navigator", {
      ...navigator,
      clipboard: {
        writeText: state.writeText,
      },
    });
  });

  it("renders loading, then populated reasoner details, executes, and loads prior input", async () => {
    let resolveReasoners: ((value: ReasonersResponse) => void) | null = null;
    let resolveHistory: ((value: any) => void) | null = null;

    state.getAllReasoners.mockImplementationOnce(
      () =>
        new Promise<ReasonersResponse>((resolve) => {
          resolveReasoners = resolve;
        })
    );
    state.getReasonerDetails.mockResolvedValue(buildReasoner());
    state.getExecutionHistory.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveHistory = resolve;
        })
    );
    state.getExecutionHistory.mockResolvedValue({
      executions: [
        {
          execution_id: "exec-after",
          duration_ms: 90,
          status: "succeeded",
          input_data: { prompt: "from history" },
          output_data: { answer: "done" },
          started_at: "2026-04-08T00:00:00Z",
        },
      ],
    });
    state.executeReasoner.mockResolvedValue({
      execution_id: "run-123",
      status: "succeeded",
      duration_ms: 1400,
      result: { answer: "hello" },
    });

    renderPage();

    expect(screen.getByRole("textbox")).toHaveValue("{}");

    resolveReasoners?.({
      reasoners: [buildReasoner()],
      total: 1,
      online_count: 1,
      offline_count: 0,
      nodes_count: 1,
    });

    expect(await screen.findByText("node-1.reasoner-1")).toBeInTheDocument();
    expect(screen.getByText("Loading runs…")).toBeInTheDocument();

    resolveHistory?.({
      executions: [
        {
          execution_id: "exec-history",
          duration_ms: 50,
          status: "succeeded",
          input_data: { prompt: "from run" },
          output_data: { answer: "cached" },
          started_at: "2026-04-08T00:00:00Z",
        },
      ],
    });

    expect(await screen.findByText(/"prompt": ""/)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: '{"prompt":"hello"}' },
    });
    fireEvent.click(screen.getByRole("button", { name: /execute/i }));

    await waitFor(() => {
      expect(state.executeReasoner).toHaveBeenCalledWith("node-1.reasoner-1", {
        input: { prompt: "hello" },
      });
    });

    expect(await screen.findByText(/"answer": "hello"/)).toBeInTheDocument();
    expect(screen.getByText("1.4s")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /view run/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /load input/i }));
    await waitFor(() => {
      expect(screen.getByRole("textbox")).toHaveValue('{\n  "prompt": "from history"\n}');
    });
  });
});
// @ts-nocheck
import React from "react";
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
    placeholder,
  }: {
    reasoners: Array<{ reasoner_id: string; name: string }>;
    value: string | null;
    onValueChange: (value: string) => void;
    disabled?: boolean;
    placeholder?: string;
  }) => (
    <select
      aria-label="Reasoner / skill"
      data-placeholder={placeholder}
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

function renderPage(entry: string | { pathname: string; state?: unknown } = "/playground/node-1.reasoner-1") {
  const initialEntries = [typeof entry === "string" ? entry : entry];
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route path="/playground/:reasonerId" element={<PlaygroundPage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("PlaygroundPage coverage paths", () => {
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

  it("preserves replay input, validates invalid JSON, and copies both curl commands", async () => {
    state.getAllReasoners.mockResolvedValue({
      reasoners: [buildReasoner()],
      total: 1,
      online_count: 1,
      offline_count: 0,
      nodes_count: 1,
    });
    state.getReasonerDetails.mockResolvedValue(buildReasoner());
    state.getExecutionHistory.mockResolvedValue({ executions: [] });

    renderPage({
      pathname: "/playground/node-1.reasoner-1",
      state: { replayInput: { prompt: "replayed" } },
    });

    expect(await screen.findByText("node-1.reasoner-1")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toHaveValue('{\n  "prompt": "replayed"\n}');

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "{bad json" },
    });
    fireEvent.keyDown(screen.getByRole("textbox"), {
      key: "Enter",
      ctrlKey: true,
    });

    expect(screen.getByText("Invalid JSON — please fix before executing.")).toBeInTheDocument();
    expect(state.executeReasoner).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Copy cURL \(sync\)/i }));
    fireEvent.click(screen.getByRole("button", { name: /Copy cURL \(async\)/i }));

    expect(state.writeText).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("/api/v1/execute/node-1.reasoner-1"),
    );
    expect(state.writeText).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining("/api/v1/execute/async/node-1.reasoner-1"),
    );
  });

  it("surfaces execution and history failures while keeping schema and recent-runs fallbacks usable", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    state.getAllReasoners.mockResolvedValue({
      reasoners: [buildReasoner()],
      total: 1,
      online_count: 1,
      offline_count: 0,
      nodes_count: 1,
    });
    state.getReasonerDetails.mockResolvedValue(
      buildReasoner({
        input_schema: undefined,
        output_schema: undefined,
      }),
    );
    state.getExecutionHistory.mockRejectedValue(new Error("history failed"));
    state.executeReasoner.mockRejectedValue(new Error("execution failed"));

    renderPage();

    expect(await screen.findByText("node-1.reasoner-1")).toBeInTheDocument();
    expect(await screen.findByText("No schema defined.")).toBeInTheDocument();
    expect(await screen.findByText("No runs recorded yet.")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: '{"prompt":"hello"}' },
    });
    fireEvent.click(screen.getByRole("button", { name: /execute/i }));

    expect(await screen.findByText("execution failed")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(errorSpy).toHaveBeenCalledWith("Failed to load recent runs:", expect.any(Error));

    errorSpy.mockRestore();
  });
});

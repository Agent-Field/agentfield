// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../layout/ResponsiveGrid", () => ({
  ResponsiveGrid: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-grid">{children}</div>,
}));

vi.mock("../ui/skeleton", () => ({
  Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
}));

vi.mock("../did/DIDDisplay", () => ({
  DIDIdentityBadge: ({ nodeId }: { nodeId: string }) => <span data-testid="did-badge">{nodeId}</span>,
}));

vi.mock("./EmptyReasonersState", () => ({
  EmptyReasonersState: ({ type }: { type: string }) => <div>empty:{type}</div>,
}));

vi.mock("./ReasonerCard", () => ({
  ReasonerCard: ({ reasoner, onClick }: { reasoner: any; onClick?: (reasoner: any) => void }) => (
    <button type="button" onClick={() => onClick?.(reasoner)}>
      card:{reasoner.name}
    </button>
  ),
}));

vi.mock("./ReasonerStatusDot", () => ({
  ReasonerStatusDot: ({ status }: { status: string }) => <span>status:{status}</span>,
}));

import { ReasonerGrid } from "./ReasonerGrid";

const baseReasoners = [
  {
    reasoner_id: "node-a.reasoner-a",
    name: "Reasoner A",
    description: "Handles quick requests",
    node_id: "node-a",
    node_status: "active",
    node_version: "1.0.0",
    input_schema: {},
    output_schema: {},
    memory_config: { auto_inject: [], memory_retention: "short", cache_results: true },
    avg_response_time_ms: 42,
    success_rate: 0.945,
    last_updated: "2026-04-09T11:59:30.000Z",
  },
  {
    reasoner_id: "node-b.reasoner-b",
    name: "Reasoner B",
    description: "Long running jobs",
    node_id: "node-b",
    node_status: "inactive",
    node_version: "1.0.0",
    input_schema: {},
    output_schema: {},
    memory_config: { auto_inject: [], memory_retention: "long", cache_results: false },
    last_updated: "2026-04-09T10:00:00.000Z",
  },
  {
    reasoner_id: "node-c.reasoner-c",
    name: "Reasoner C",
    description: "Fallback processor",
    node_id: "node-c",
    node_status: "unknown",
    node_version: "1.0.0",
    input_schema: {},
    output_schema: {},
    memory_config: { auto_inject: [], memory_retention: "medium", cache_results: true },
    avg_response_time_ms: 90,
    success_rate: 0.5,
    last_updated: "2026-04-07T12:00:00.000Z",
  },
];

describe("ReasonerGrid", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-09T12:00:00.000Z"));
  });

  it("renders grid cards and forwards clicks", () => {
    const onReasonerClick = vi.fn();

    render(<ReasonerGrid reasoners={baseReasoners.slice(0, 2)} onReasonerClick={onReasonerClick} />);

    expect(screen.getByTestId("responsive-grid")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "card:Reasoner A" }));
    expect(onReasonerClick).toHaveBeenCalledWith(baseReasoners[0]);
  });

  it("renders an empty state when there are no reasoners", () => {
    render(<ReasonerGrid reasoners={null} />);
    expect(screen.getByText("empty:no-reasoners")).toBeInTheDocument();
  });

  it("renders loading skeletons in both grid and table modes", () => {
    const { rerender } = render(<ReasonerGrid reasoners={baseReasoners} loading />);
    expect(screen.getByTestId("responsive-grid")).toBeInTheDocument();
    expect(screen.getAllByTestId("skeleton").length).toBeGreaterThan(8);

    rerender(<ReasonerGrid reasoners={baseReasoners} loading viewMode="table" />);
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getAllByTestId("skeleton").length).toBeGreaterThan(20);
  });

  it("renders the table view with status mapping, formatted metrics, and row interactions", () => {
    const onReasonerClick = vi.fn();

    render(<ReasonerGrid reasoners={baseReasoners} viewMode="table" onReasonerClick={onReasonerClick} />);

    expect(screen.getByText("status:online")).toBeInTheDocument();
    expect(screen.getByText("status:offline")).toBeInTheDocument();
    expect(screen.getByText("status:unknown")).toBeInTheDocument();
    expect(screen.getByText("42ms")).toBeInTheDocument();
    expect(screen.getByText("94.5%")).toBeInTheDocument();
    expect(screen.getByText("Just now")).toBeInTheDocument();
    expect(screen.getByText("2h ago")).toBeInTheDocument();
    expect(screen.getByText("2d ago")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Reasoner B"));
    fireEvent.click(screen.getAllByRole("button", { name: "Open →" })[2]);

    expect(onReasonerClick).toHaveBeenNthCalledWith(1, baseReasoners[1]);
    expect(onReasonerClick).toHaveBeenNthCalledWith(2, baseReasoners[2]);
  });
});

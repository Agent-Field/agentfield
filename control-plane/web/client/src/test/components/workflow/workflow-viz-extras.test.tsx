// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AgentHealthHeatmap } from "@/components/workflow/AgentHealthHeatmap";
import { VelocityChart } from "@/components/workflow/VelocityChart";
import type { WorkflowTimelineNode } from "@/types/workflows";

vi.mock("recharts", () => {
  const ResponsiveContainer = ({ children }: React.PropsWithChildren) => <div>{children}</div>;
  const ScatterChart = ({ children }: React.PropsWithChildren) => <div data-testid="scatter-chart">{children}</div>;
  const ComposedChart = ({
    data,
    children,
  }: React.PropsWithChildren<{ data?: Array<Record<string, unknown>> }>) => (
    <div data-testid="composed-chart" data-points={data?.length ?? 0}>
      {children}
    </div>
  );
  const Tooltip = ({
    content,
    payload,
  }: {
    content?: React.ReactElement | ((args: unknown) => React.ReactNode);
    payload?: Array<{ payload: Record<string, unknown> }>;
  }) => {
    if (!content) return null;
    const args = { active: true, payload: payload ?? [] };
    if (typeof content === "function") {
      return <div>{content(args)}</div>;
    }
    const Content = content.type as React.ElementType;
    return (
      <div>
        <Content {...content.props} {...args} />
      </div>
    );
  };
  const Scatter = ({
    data,
    children,
  }: React.PropsWithChildren<{ data?: Array<Record<string, unknown>> }>) => (
    <div>
      <div data-testid="scatter-points">{JSON.stringify(data)}</div>
      <Tooltip payload={data?.length ? [{ payload: data[0] }] : []} />
      {children}
    </div>
  );
  const Line = ({ dataKey }: { dataKey?: string }) => <div>{dataKey}</div>;
  const Area = ({ dataKey }: { dataKey?: string }) => <div>{dataKey}</div>;
  const XAxis = ({ tickFormatter }: { tickFormatter?: (value: number) => string }) => (
    <div>{tickFormatter ? tickFormatter(0) : null}</div>
  );
  const YAxis = ({ tickFormatter }: { tickFormatter?: (value: number) => string }) => (
    <div>{tickFormatter ? tickFormatter(0) : null}</div>
  );
  const ZAxis = () => <div />;
  const Cell = ({ fill, fillOpacity }: { fill?: string; fillOpacity?: number }) => (
    <div data-testid="cell" data-fill={fill} data-opacity={String(fillOpacity)} />
  );
  return {
    ResponsiveContainer,
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    ZAxis,
    Cell,
    Tooltip,
    Line,
    Area,
    ComposedChart,
  };
});

function makeNode(overrides: Partial<WorkflowTimelineNode> = {}): WorkflowTimelineNode {
  return {
    workflow_id: "wf-1",
    execution_id: "exec-12345678",
    agent_node_id: "agent-1",
    reasoner_id: "planner",
    status: "succeeded",
    started_at: "2026-04-08T10:00:00Z",
    completed_at: "2026-04-08T10:01:00Z",
    duration_ms: 60000,
    workflow_depth: 0,
    agent_name: "Agent Alpha",
    task_name: "Plan",
    ...overrides,
  };
}

describe("workflow viz extras", () => {
  it("renders empty states for both workflow visualizations", () => {
    render(
      <div>
        <AgentHealthHeatmap timedNodes={[]} />
        <VelocityChart timedNodes={[]} />
      </div>,
    );

    expect(screen.getByText("No data for heatmap")).toBeInTheDocument();
    expect(screen.getByText("Not enough temporal data to plot")).toBeInTheDocument();
  });

  it("aggregates heatmap cells and renders velocity min/max labels", () => {
    const nodes = [
      makeNode({
        execution_id: "exec-aaaa1111",
        agent_name: "Agent Alpha",
        status: "succeeded",
        started_at: "2026-04-08T10:00:00Z",
        duration_ms: 250,
      }),
      makeNode({
        execution_id: "exec-bbbb2222",
        agent_name: "Agent Alpha",
        status: "failed",
        started_at: "2026-04-08T10:04:00Z",
        duration_ms: 1200,
      }),
      makeNode({
        execution_id: "exec-cccc3333",
        agent_name: "Agent Beta",
        reasoner_id: "reviewer",
        status: "error",
        started_at: "2026-04-08T10:19:00Z",
        duration_ms: 61000,
      }),
    ];

    render(
      <div>
        <AgentHealthHeatmap timedNodes={nodes} />
        <VelocityChart timedNodes={nodes} />
      </div>,
    );

    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("Sporadic Errors")).toBeInTheDocument();
    expect(screen.getByText("High Failure")).toBeInTheDocument();

    const scatterPayload = screen.getByTestId("scatter-points").textContent ?? "";
    expect(scatterPayload).toContain("Agent Alpha");
    expect(scatterPayload).toContain('"failed":1');

    expect(screen.getAllByTestId("cell").length).toBeGreaterThan(0);
    expect(screen.getByText("Min: 250ms")).toBeInTheDocument();
    expect(screen.getByText("Max: 1m 1s")).toBeInTheDocument();
  });
});
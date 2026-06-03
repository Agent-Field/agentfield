import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";

import {
  adaptLightweightResponse,
  applySimpleGridLayout,
  decorateEdgesWithStatus,
  decorateNodesWithViewMode,
  isLightweightDAGResponse,
  LARGE_GRAPH_LAYOUT_THRESHOLD,
  PERFORMANCE_THRESHOLD,
  SIMPLE_LAYOUT_COLUMNS,
  SIMPLE_LAYOUT_X_SPACING,
  SIMPLE_LAYOUT_Y_SPACING,
  mapLightweightNode,
} from "@/components/WorkflowDAG/workflowDagUtils";

describe("workflowDagUtils", () => {
  it("detects and adapts lightweight DAG responses", () => {
    const response = {
      root_workflow_id: "wf-1",
      session_id: "session-1",
      actor_id: "actor-1",
      total_nodes: 2,
      max_depth: 1,
      workflow_status: "running",
      workflow_name: "Checkout flow",
      mode: "lightweight" as const,
      timeline: [
        {
          execution_id: "exec-1",
          agent_node_id: "agent-1",
          reasoner_id: "root",
          status: "running",
          started_at: "2026-04-08T16:00:00Z",
          workflow_depth: 0,
        },
        {
          execution_id: "exec-2",
          agent_node_id: "agent-2",
          reasoner_id: "child",
          status: "succeeded",
          started_at: "2026-04-08T16:01:00Z",
          completed_at: "2026-04-08T16:02:00Z",
          duration_ms: 1000,
          parent_execution_id: "exec-1",
          workflow_depth: 1,
        },
      ],
    };

    expect(isLightweightDAGResponse(response)).toBe(true);
    expect(isLightweightDAGResponse(null)).toBe(false);
    expect(
      isLightweightDAGResponse(
        { ...response, mode: undefined } as unknown as Parameters<typeof isLightweightDAGResponse>[0]
      )
    ).toBe(false);

    expect(mapLightweightNode(response.timeline[1], response.root_workflow_id)).toEqual({
      workflow_id: "wf-1",
      execution_id: "exec-2",
      agent_node_id: "agent-2",
      reasoner_id: "child",
      status: "succeeded",
      started_at: "2026-04-08T16:01:00Z",
      completed_at: "2026-04-08T16:02:00Z",
      duration_ms: 1000,
      parent_execution_id: "exec-1",
      workflow_depth: 1,
    });

    expect(adaptLightweightResponse(response)).toEqual({
      root_workflow_id: "wf-1",
      session_id: "session-1",
      actor_id: "actor-1",
      total_nodes: 2,
      displayed_nodes: 2,
      max_depth: 1,
      dag: {
        workflow_id: "wf-1",
        execution_id: "exec-1",
        agent_node_id: "agent-1",
        reasoner_id: "root",
        status: "running",
        started_at: "2026-04-08T16:00:00Z",
        completed_at: undefined,
        duration_ms: undefined,
        parent_execution_id: undefined,
        workflow_depth: 0,
      },
      timeline: [
        {
          workflow_id: "wf-1",
          execution_id: "exec-1",
          agent_node_id: "agent-1",
          reasoner_id: "root",
          status: "running",
          started_at: "2026-04-08T16:00:00Z",
          completed_at: undefined,
          duration_ms: undefined,
          parent_execution_id: undefined,
          workflow_depth: 0,
        },
        {
          workflow_id: "wf-1",
          execution_id: "exec-2",
          agent_node_id: "agent-2",
          reasoner_id: "child",
          status: "succeeded",
          started_at: "2026-04-08T16:01:00Z",
          completed_at: "2026-04-08T16:02:00Z",
          duration_ms: 1000,
          parent_execution_id: "exec-1",
          workflow_depth: 1,
        },
      ],
      workflow_status: "running",
      workflow_name: "Checkout flow",
      mode: "lightweight",
    });

    expect(PERFORMANCE_THRESHOLD).toBe(300);
    expect(LARGE_GRAPH_LAYOUT_THRESHOLD).toBe(2000);
    expect(SIMPLE_LAYOUT_COLUMNS).toBe(40);
    expect(SIMPLE_LAYOUT_X_SPACING).toBe(240);
    expect(SIMPLE_LAYOUT_Y_SPACING).toBe(120);
  });

  it("lays out nodes in depth and start-time order and decorates nodes and edges", () => {
    const nodes = [
      { id: "b", data: { label: "B" }, position: { x: 0, y: 0 } },
      { id: "a", data: { label: "A" }, position: { x: 0, y: 0 } },
      { id: "c", data: { label: "C" }, position: { x: 0, y: 0 } },
    ] as Node[];
    const executionMap = new Map([
      [
        "a",
        {
          workflow_id: "wf-1",
          execution_id: "a",
          agent_node_id: "agent-a",
          reasoner_id: "root",
          status: "succeeded",
          started_at: "2026-04-08T16:00:00Z",
          workflow_depth: 0,
        },
      ],
      [
        "b",
        {
          workflow_id: "wf-1",
          execution_id: "b",
          agent_node_id: "agent-b",
          reasoner_id: "child",
          status: "running",
          started_at: "2026-04-08T16:01:00Z",
          duration_ms: 250,
          workflow_depth: 1,
        },
      ],
    ]);

    const laidOut = applySimpleGridLayout(nodes, executionMap);
    expect(laidOut.map((node) => node.id)).toEqual(["c", "a", "b"]);
    expect(laidOut[0].position).toEqual({ x: 0, y: 0 });
    expect(laidOut[1].position).toEqual({ x: SIMPLE_LAYOUT_X_SPACING, y: 0 });
    expect(laidOut[2].position).toEqual({ x: SIMPLE_LAYOUT_X_SPACING * 2, y: 0 });

    const decoratedNodes = decorateNodesWithViewMode(laidOut, "performance");
    expect(decoratedNodes.every((node) => node.data?.viewMode === "performance")).toBe(true);
    expect(decoratedNodes[0].data).toMatchObject({ label: "C", viewMode: "performance" });

    const edges = [
      { id: "edge-a-b", source: "a", target: "b", data: { existing: true } },
      { id: "edge-b-c", source: "b", target: "c" },
    ] as Edge[];
    const decoratedEdges = decorateEdgesWithStatus(edges, executionMap);

    expect(decoratedEdges[0]).toMatchObject({
      id: "edge-a-b",
      animated: true,
      data: {
        existing: true,
        status: "running",
        duration: 250,
        animated: true,
      },
    });
    expect(decoratedEdges[1]).toEqual(edges[1]);
  });
});

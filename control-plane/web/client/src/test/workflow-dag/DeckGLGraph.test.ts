// @ts-nocheck
import { describe, expect, it, vi } from "vitest";

import { buildDeckGraph } from "@/components/WorkflowDAG/DeckGLGraph";

describe("buildDeckGraph", () => {
  it("returns empty graph data when timeline is empty", () => {
    expect(buildDeckGraph([])).toEqual({
      nodes: [],
      edges: [],
      agentPalette: [],
    });
  });

  it("builds positioned nodes, curved edges, and sorted agent palette entries", () => {
    const debugSpy = vi.spyOn(console, "debug").mockImplementation(() => {});

    const timeline = [
      {
        execution_id: "root",
        parent_execution_id: null,
        workflow_depth: 0,
        status: "completed",
        agent_node_id: "beta-agent",
        reasoner_id: "root_reasoner",
      },
      {
        execution_id: "child",
        parent_execution_id: "root",
        workflow_depth: 1,
        status: "running now",
        agent_node_id: "alpha-agent",
        reasoner_id: "child_reasoner",
      },
      {
        execution_id: "grandchild",
        parent_execution_id: "child",
        workflow_depth: 2,
        status: "failed badly",
        agent_node_id: "alpha-agent",
        reasoner_id: "grandchild_reasoner",
      },
    ] as any;

    const result = buildDeckGraph(timeline, 100, 75);

    expect(result.nodes).toHaveLength(3);
    expect(result.edges).toHaveLength(2);
    expect(result.agentPalette.map((entry) => entry.label)).toEqual([
      "alpha-agent",
      "beta-agent",
    ]);

    const root = result.nodes.find((node) => node.id === "root");
    const child = result.nodes.find((node) => node.id === "child");
    const grandchild = result.nodes.find((node) => node.id === "grandchild");

    expect(root?.depth).toBe(0);
    expect(child?.depth).toBe(1);
    expect(grandchild?.depth).toBe(2);
    expect(root?.position[1]).toBe(0);
    expect(child?.position[1]).toBe(75);
    expect(grandchild?.position[1]).toBe(150);
    expect(child?.fillColor[3]).toBeGreaterThan(grandchild!.fillColor[3]);
    expect(root?.borderColor[3]).toBe(255);
    expect(grandchild?.glowColor[3]).toBe(90);

    const firstEdge = result.edges.find((edge) => edge.id === "root-child");
    expect(firstEdge?.path).toHaveLength(9);
    expect(firstEdge?.path[0]).toEqual(root?.position);
    expect(firstEdge?.path.at(-1)).toEqual(child?.position);
    expect(firstEdge?.width).toBeGreaterThan(1);

    expect(debugSpy).toHaveBeenCalled();
  });

  it("falls back to the shallowest node as root when every node has a known parent", () => {
    const timeline = [
      {
        execution_id: "node-a",
        parent_execution_id: "node-b",
        workflow_depth: 1,
        status: "queued",
        agent_node_id: "agent-a",
        reasoner_id: "reasoner-a",
      },
      {
        execution_id: "node-b",
        parent_execution_id: "node-a",
        workflow_depth: 3,
        status: "timeout",
        agent_node_id: "agent-b",
        reasoner_id: "reasoner-b",
      },
    ] as any;

    const result = buildDeckGraph(timeline);

    expect(result.nodes.map((node) => node.id).sort()).toEqual(["node-a", "node-b"]);
    expect(result.nodes.find((node) => node.id === "node-a")?.depth).toBeGreaterThan(
      result.nodes.find((node) => node.id === "node-b")?.depth ?? -1,
    );
    expect(result.edges).toHaveLength(2);
  });
});
// @ts-nocheck
import { beforeEach, describe, expect, it, vi } from "vitest";

const dagreLayoutMock = vi.hoisted(() => vi.fn());
const elkApplyLayoutMock = vi.hoisted(() => vi.fn());
const elkIsSlowMock = vi.hoisted(() => vi.fn((layoutType: string) => layoutType === "layered"));
const elkDescriptionMock = vi.hoisted(() => vi.fn((layoutType: string) => `elk:${layoutType}`));

vi.mock("dagre", () => {
  class Graph {
    nodesMap = new Map<string, { width: number; height: number; x?: number; y?: number }>();
    graphConfig: Record<string, unknown> = {};

    setDefaultEdgeLabel(_fn: () => unknown) {}

    setGraph(config: Record<string, unknown>) {
      this.graphConfig = config;
    }

    setNode(id: string, value: { width: number; height: number }) {
      this.nodesMap.set(id, { ...value });
    }

    setEdge(_source: string, _target: string) {}

    node(id: string) {
      return this.nodesMap.get(id);
    }
  }

  return {
    default: {
      graphlib: { Graph },
      layout: dagreLayoutMock,
    },
  };
});

vi.mock("@/components/WorkflowDAG/layouts/ELKLayoutEngine", () => {
  return {
    ELKLayoutEngine: class MockELKLayoutEngine {
      applyLayout = elkApplyLayoutMock;

      static isSlowForLargeGraphs(layoutType: string) {
        return elkIsSlowMock(layoutType);
      }

      static getLayoutDescription(layoutType: string) {
        return elkDescriptionMock(layoutType);
      }
    },
  };
});

import { LayoutManager } from "@/components/WorkflowDAG/layouts/LayoutManager";

describe("LayoutManager", () => {
  beforeEach(() => {
    dagreLayoutMock.mockReset();
    elkApplyLayoutMock.mockReset();
    elkIsSlowMock.mockClear();
    elkDescriptionMock.mockClear();
  });

  it("reports layout metadata helpers", () => {
    const manager = new LayoutManager({ smallGraphThreshold: 10 });

    expect(manager.isLargeGraph(10)).toBe(true);
    expect(manager.isLargeGraph(9)).toBe(false);
    expect(manager.getAvailableLayouts(1)).toEqual([
      "tree",
      "flow",
      "layered",
      "box",
      "rectpacking",
    ]);
    expect(manager.getDefaultLayout(99)).toBe("tree");
    expect(manager.isSlowLayout("tree")).toBe(false);
    expect(manager.isSlowLayout("flow")).toBe(false);
    expect(manager.isSlowLayout("layered")).toBe(true);
    expect(manager.getLayoutDescription("tree")).toContain("Tree layout");
    expect(manager.getLayoutDescription("box")).toBe("elk:box");
  });

  it("applies wrapped tree layout on the main thread", async () => {
    const manager = new LayoutManager();
    const progress = vi.fn();

    const nodes = Array.from({ length: 8 }, (_, index) => ({
      id: `node-${index}`,
      data: {
        task_name: `task_name_${index}`,
        agent_name: `agent_name_${index}`,
        started_at: `2024-01-0${(index % 3) + 1}T00:00:00Z`,
      },
      position: { x: 0, y: 0 },
    })) as any;
    const edges = nodes.slice(1).map((node: any, index: number) => ({
      id: `edge-${index}`,
      source: index === 0 ? "node-0" : `node-${index}`,
      target: node.id,
    })) as any;

    const result = await manager.applyLayout(nodes, edges, "tree", progress);

    expect(progress).toHaveBeenNthCalledWith(1, 0);
    expect(progress).toHaveBeenLastCalledWith(100);
    expect(result.edges).toBe(edges);
    expect(result.nodes).toHaveLength(8);
    expect(result.nodes[0].position.y).toBeLessThan(result.nodes[1].position.y);
    expect(new Set(result.nodes.map((node: any) => `${node.position.x}:${node.position.y}`)).size).toBe(8);
    expect(result.nodes.every((node: any) => node.position.x >= 0 && node.position.y >= 0)).toBe(true);
  });

  it("applies dagre flow layout and centers nodes from dagre coordinates", async () => {
    dagreLayoutMock.mockImplementation((graph: any) => {
      let index = 0;
      const isTopBottom = graph.graphConfig.rankdir === "TB";
      for (const node of graph.nodesMap.values()) {
        node.x = isTopBottom ? 100 : 300 + index * 150;
        node.y = isTopBottom ? 200 + index * 150 : 120;
        index += 1;
      }
    });

    const manager = new LayoutManager();
    const nodes = [
      {
        id: "left",
        data: { task_name: "alpha_task", agent_name: "agent_a" },
        position: { x: 0, y: 0 },
      },
      {
        id: "right",
        data: { task_name: "beta_task", agent_name: "agent_b" },
        position: { x: 0, y: 0 },
      },
    ] as any;
    const edges = [{ id: "e1", source: "left", target: "right" }] as any;
    const progress = vi.fn();

    const result = await manager.applyLayout(nodes, edges, "flow", progress);

    expect(dagreLayoutMock).toHaveBeenCalledTimes(1);
    expect(progress).toHaveBeenNthCalledWith(1, 0);
    expect(progress).toHaveBeenLastCalledWith(100);
    expect(result.nodes[0].position.x).toBeLessThan(result.nodes[1].position.x);
    expect(result.nodes[0].position.y).toBe(70);
  });

  it("delegates non-dagre layouts to the ELK engine and preserves progress callbacks", async () => {
    elkApplyLayoutMock.mockResolvedValue({
      nodes: [{ id: "elk", position: { x: 12, y: 34 } }],
      edges: [{ id: "elk-edge" }],
    });

    const manager = new LayoutManager();
    const progress = vi.fn();
    const nodes = [{ id: "elk", data: {}, position: { x: 0, y: 0 } }] as any;
    const edges = [{ id: "elk-edge", source: "elk", target: "elk" }] as any;

    const result = await manager.applyLayout(nodes, edges, "layered", progress);

    expect(elkApplyLayoutMock).toHaveBeenCalledWith(nodes, edges, "layered");
    expect(progress.mock.calls).toEqual([[0], [25], [100]]);
    expect(result).toEqual({
      nodes: [{ id: "elk", position: { x: 12, y: 34 } }],
      edges: [{ id: "elk-edge" }],
    });
  });

  it("falls back to original graph when layout application throws", async () => {
    dagreLayoutMock.mockImplementation(() => {
      throw new Error("boom");
    });
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const manager = new LayoutManager();
    const nodes = [{ id: "node", data: {}, position: { x: 1, y: 2 } }] as any;
    const edges = [{ id: "edge", source: "node", target: "node" }] as any;

    const result = await manager.applyLayout(nodes, edges, "flow");

    expect(result).toEqual({ nodes, edges });
    expect(errorSpy).toHaveBeenCalled();
  });

  it("uses the worker when enabled and resolves progress and results from worker messages", async () => {
    const workerInstances: any[] = [];

    class WorkerMock {
      onmessage?: (event: MessageEvent<any>) => void;
      onerror?: (event: ErrorEvent) => void;
      postMessage = vi.fn();
      terminate = vi.fn();

      constructor(..._args: unknown[]) {
        workerInstances.push(this);
      }
    }

    vi.stubGlobal("window", {});
    vi.stubGlobal("Worker", WorkerMock as any);

    const manager = new LayoutManager({ enableWorker: true });
    const progress = vi.fn();
    const nodes = [{ id: "worker-node", data: {}, position: { x: 0, y: 0 } }] as any;
    const edges = [] as any;

    const promise = manager.applyLayout(nodes, edges, "tree", progress);

    expect(workerInstances).toHaveLength(1);
    expect(workerInstances[0].postMessage).toHaveBeenCalledWith({
      id: "layout-1",
      nodes,
      edges,
      layoutType: "tree",
    });
    expect(progress).toHaveBeenCalledWith(0);

    workerInstances[0].onmessage?.({
      data: { id: "layout-1", type: "progress", value: 55 },
    } as MessageEvent<any>);
    workerInstances[0].onmessage?.({
      data: { id: "layout-1", type: "result", nodes, edges },
    } as MessageEvent<any>);

    await expect(promise).resolves.toEqual({ nodes, edges });
    expect(progress.mock.calls).toContainEqual([55]);
    expect(progress.mock.calls.at(-1)).toEqual([100]);

    vi.unstubAllGlobals();
  });
});
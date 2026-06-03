import { beforeEach, describe, expect, it, vi } from "vitest";

import { ELK_ALGORITHMS } from "@/types/elk";

const layoutMock = vi.hoisted(() => vi.fn());

vi.mock("elkjs/lib/elk.bundled.js", () => {
  return {
    default: class MockELK {
      layout = layoutMock;
    },
  };
});

import {
  applyElkLayout,
  getAvailableAlgorithms,
  getRecommendedAlgorithm,
  isAlgorithmSuitableForLargeGraphs,
} from "@/utils/elkLayout";

describe("elkLayout", () => {
  beforeEach(() => {
    layoutMock.mockReset();
  });

  it("returns original graph when no nodes are provided", async () => {
    const result = await applyElkLayout([], []);

    expect(result).toEqual({ nodes: [], edges: [] });
    expect(layoutMock).not.toHaveBeenCalled();
  });

  it("applies ELK layout and maps positioned nodes back to react flow", async () => {
    layoutMock.mockResolvedValue({
      id: "root",
      children: [
        { id: "node-1", x: 12, y: 24 },
        { id: "node-2", x: 48, y: 96 },
      ],
      edges: [{ id: "edge-1", sources: ["node-1"], targets: ["node-2"] }],
    });

    const nodes = [
      {
        id: "node-1",
        data: {
          task_name: "very_long_task_name",
          agent_name: "assistant_agent",
        },
        position: { x: 0, y: 0 },
      },
      {
        id: "node-2",
        data: { reasoner_id: "brief", agent_node_id: "node-2" },
        position: { x: 5, y: 5 },
      },
    ] as any;
    const edges = [{ id: "edge-1", source: "node-1", target: "node-2" }] as any;

    const result = await applyElkLayout(nodes, edges, ELK_ALGORITHMS.LAYERED, {
      "elk.spacing.nodeNode": 123,
    });

    expect(layoutMock).toHaveBeenCalledTimes(1);
    expect(layoutMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "root",
        layoutOptions: expect.objectContaining({
          "elk.algorithm": ELK_ALGORITHMS.LAYERED,
          "elk.direction": "DOWN",
          "elk.spacing.nodeNode": 123,
        }),
        children: [
          expect.objectContaining({
            id: "node-1",
            width: expect.any(Number),
            height: 100,
          }),
          expect.objectContaining({
            id: "node-2",
            width: expect.any(Number),
            height: 100,
          }),
        ],
        edges: [{ id: "edge-1", sources: ["node-1"], targets: ["node-2"] }],
      })
    );
    expect(result).toEqual({
      nodes: [
        expect.objectContaining({ id: "node-1", position: { x: 12, y: 24 } }),
        expect.objectContaining({ id: "node-2", position: { x: 48, y: 96 } }),
      ],
      edges,
    });
  });

  it("falls back to original nodes and edges when ELK throws", async () => {
    const error = new Error("layout failed");
    const nodes = [{ id: "node-1", data: {}, position: { x: 1, y: 2 } }] as any;
    const edges = [{ id: "edge-1", source: "node-1", target: "node-1" }] as any;
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    layoutMock.mockRejectedValue(error);

    const result = await applyElkLayout(nodes, edges);

    expect(result).toEqual({ nodes, edges });
    expect(consoleSpy).toHaveBeenCalledWith("ELK Layout Error:", error);
  });

  it("reports available algorithms and recommendation helpers", () => {
    expect(getAvailableAlgorithms()).toContainEqual({
      id: ELK_ALGORITHMS.TOPDOWN_PACKING,
      name: "Topdown Packing",
      algorithm: ELK_ALGORITHMS.TOPDOWN_PACKING,
    });

    expect(isAlgorithmSuitableForLargeGraphs(ELK_ALGORITHMS.BOX)).toBe(true);
    expect(isAlgorithmSuitableForLargeGraphs(ELK_ALGORITHMS.LAYERED)).toBe(false);

    expect(getRecommendedAlgorithm(600, 100)).toBe(ELK_ALGORITHMS.BOX);
    expect(getRecommendedAlgorithm(150, 100)).toBe(ELK_ALGORITHMS.RECT_PACKING);
    expect(getRecommendedAlgorithm(10, 30)).toBe(ELK_ALGORITHMS.FORCE);
    expect(getRecommendedAlgorithm(10, 10)).toBe(ELK_ALGORITHMS.LAYERED);
  });
});

// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const edgeState = vi.hoisted(() => ({
  nodes: {} as Record<string, any>,
  getBezierPath: vi.fn(() => ["M0,0 C1,1 2,2 3,3", 120, 80] as const),
}));

vi.mock("@xyflow/react", () => ({
  BaseEdge: ({
    path,
    markerEnd,
    style,
  }: {
    path: string;
    markerEnd?: string;
    style?: React.CSSProperties;
  }) => (
    <path
      data-testid="base-edge"
      d={path}
      data-marker-end={markerEnd}
      style={style}
    />
  ),
  EdgeLabelRenderer: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Position: {
    Left: "left",
    Right: "right",
    Top: "top",
    Bottom: "bottom",
  },
  getBezierPath: (args: unknown) => edgeState.getBezierPath(args),
  useInternalNode: (id: string) => edgeState.nodes[id] ?? null,
}));

describe("Workflow DAG edge components", () => {
  beforeEach(() => {
    edgeState.nodes = {};
    edgeState.getBezierPath.mockClear();
    edgeState.getBezierPath.mockReturnValue(["M0,0 C1,1 2,2 3,3", 120, 80]);
  });

  it("renders FloatingEdge with fallback coordinates, succeeded duration label, and muted emphasis", async () => {
    const { default: FloatingEdge } = await import("@/components/WorkflowDAG/FloatingEdge");

    const { container } = render(
      <svg>
        <FloatingEdge
          id="edge-1"
          source="source-1"
          target="target-1"
          sourceX={10}
          sourceY={20}
          targetX={30}
          targetY={40}
          data={{
            status: "success",
            duration: 1500,
            emphasis: "muted",
          }}
        />
      </svg>,
    );

    expect(edgeState.getBezierPath).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceX: 10,
        sourceY: 20,
        targetX: 30,
        targetY: 40,
        sourcePosition: "bottom",
        targetPosition: "top",
      }),
    );

    const edgePath = container.querySelector("#edge-1") as SVGPathElement;
    expect(edgePath).toHaveAttribute("marker-end", "url(#arrowclosed-edge-1-succeeded)");
    expect(edgePath.style.strokeWidth).toBe("2.5");
    expect(edgePath.style.opacity).toBe("0.18");
    expect(edgePath.style.filter).toBe("grayscale(80%)");
    expect(edgePath.style.strokeDasharray).toBe("6,4");
    expect(screen.getByText("1.5s")).toBeInTheDocument();
  });

  it("renders FloatingEdge with internal node intersection math and running animation state", async () => {
    edgeState.nodes = {
      source: {
        id: "source",
        position: { x: 0, y: 0 },
        width: 120,
        height: 60,
      },
      target: {
        id: "target",
        position: { x: 300, y: 0 },
        width: 120,
        height: 60,
      },
    };

    const { default: FloatingEdge } = await import("@/components/WorkflowDAG/FloatingEdge");

    const { container } = render(
      <svg>
        <FloatingEdge
          id="edge-running"
          source="source"
          target="target"
          data={{ status: "running", animated: true, emphasis: "focus" }}
        />
      </svg>,
    );

    expect(edgeState.getBezierPath).toHaveBeenCalledWith(
      expect.objectContaining({
        sourcePosition: "right",
        targetPosition: "left",
        sourceX: expect.any(Number),
        targetX: expect.any(Number),
      }),
    );

    const edgePath = container.querySelector("#edge-running") as SVGPathElement;
    expect(edgePath.style.strokeWidth).toBe("3.6");
    expect(edgePath.style.opacity).toBe("1");
    expect(edgePath.style.animation).toContain("dash");
    expect(edgePath.style.filter).toContain("drop-shadow(0 0 6px rgba(34,197,94,0.4))");
    expect(container.querySelector("animateMotion")).toBeInTheDocument();
    expect(container.querySelector("marker")?.getAttribute("markerWidth")).toBe("16");
  });

  it("renders EnhancedEdge with failed styling and running particles when animated", async () => {
    const { EnhancedEdge, customEdgeTypes } = await import("@/components/WorkflowDAG/EnhancedEdge");

    const { container, rerender } = render(
      <svg>
        <EnhancedEdge
          id="enhanced-failed"
          source="a"
          target="b"
          sourceX={0}
          sourceY={0}
          targetX={20}
          targetY={20}
          sourcePosition={"left" as never}
          targetPosition={"right" as never}
          data={{ status: "failed", duration: 250 }}
        />
      </svg>,
    );

    const baseEdge = screen.getByTestId("base-edge");
    expect(baseEdge).toHaveStyle({
      strokeWidth: "2.5",
      strokeDasharray: "8,4",
    });
    expect(baseEdge).toHaveAttribute("data-marker-end", "url(#arrowclosed-failed)");
    expect(container.querySelector("animateMotion")).not.toBeInTheDocument();
    expect(customEdgeTypes.enhanced).toBe(EnhancedEdge);

    rerender(
      <svg>
        <EnhancedEdge
          id="enhanced-running"
          source="a"
          target="b"
          sourceX={0}
          sourceY={0}
          targetX={20}
          targetY={20}
          sourcePosition={"left" as never}
          targetPosition={"right" as never}
          data={{ status: "running", animated: true }}
        />
      </svg>,
    );

    expect(screen.getByTestId("base-edge")).toHaveStyle({
      strokeWidth: "3",
      strokeDasharray: "12,8",
    });
    expect(container.querySelector("animateMotion")).toBeInTheDocument();
    expect(container.querySelector("marker")?.getAttribute("markerWidth")).toBe("16");
  });
});
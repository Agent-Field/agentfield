// @ts-nocheck
import React from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const deckPropsHistory = vi.hoisted(() => [] as any[]);
const scatterLayerHistory = vi.hoisted(() => [] as any[]);
const pathLayerHistory = vi.hoisted(() => [] as any[]);

vi.mock("@deck.gl/react", () => ({
  default: (props: any) => {
    deckPropsHistory.push(props);
    return <div data-testid="deckgl-view" data-cursor={props.getCursor()} />;
  },
}));

vi.mock("@deck.gl/core", () => ({
  COORDINATE_SYSTEM: { CARTESIAN: "cartesian" },
  OrthographicController: class OrthographicController {},
  OrthographicView: class OrthographicView {
    constructor(public props: any) {}
  },
}));

vi.mock("@deck.gl/layers", () => ({
  ScatterplotLayer: class ScatterplotLayer {
    constructor(public props: any) {
      scatterLayerHistory.push(this);
    }
  },
  PathLayer: class PathLayer {
    constructor(public props: any) {
      pathLayerHistory.push(this);
    }
  },
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => (
    <svg data-testid="deck-icon" className={className} />
  );

  return {
    Minus: Icon,
    Plus: Icon,
    Scan: Icon,
  };
});

vi.mock("@/components/WorkflowDAG/HoverDetailPanel", () => ({
  HoverDetailPanel: ({ node, visible, position }: any) =>
    visible && node ? (
      <div data-testid="hover-panel">
        {node.execution_id}@{position.x},{position.y}
      </div>
    ) : null,
}));

import {
  WorkflowDeckGLView,
  WorkflowDeckGraphControls,
  type WorkflowDeckGLViewHandle,
} from "@/components/WorkflowDAG/DeckGLView";
import type { DeckEdge, DeckNode } from "@/components/WorkflowDAG/DeckGLGraph";

const latestDeckProps = () => deckPropsHistory.at(-1);
const latestScatterLayer = () => scatterLayerHistory.at(-1)?.props;
const latestPathLayer = () => pathLayerHistory.at(-1)?.props;

describe("WorkflowDeckGLView", () => {
  beforeEach(() => {
    deckPropsHistory.length = 0;
    scatterLayerHistory.length = 0;
    pathLayerHistory.length = 0;
    vi.useRealTimers();
  });

  const nodes: DeckNode[] = [
    {
      id: "a",
      position: [0, 0, 0],
      depth: 0,
      radius: 10,
      fillColor: [100, 110, 120, 240],
      borderColor: [1, 2, 3, 255],
      glowColor: [4, 5, 6, 90],
      original: {
        execution_id: "a",
        parent_execution_id: null,
        status: "running",
        agent_node_id: "agent-a",
        reasoner_id: "root",
      } as any,
    },
    {
      id: "b",
      position: [200, 100, 0],
      depth: 1,
      radius: 8,
      fillColor: [90, 100, 110, 230],
      borderColor: [6, 7, 8, 255],
      glowColor: [9, 10, 11, 90],
      original: {
        execution_id: "b",
        parent_execution_id: "a",
        status: "succeeded",
        agent_node_id: "agent-b",
        reasoner_id: "child",
      } as any,
    },
    {
      id: "c",
      position: [400, 100, 0],
      depth: 1,
      radius: 8,
      fillColor: [80, 90, 100, 220],
      borderColor: [10, 11, 12, 255],
      glowColor: [12, 13, 14, 90],
      original: {
        execution_id: "c",
        parent_execution_id: null,
        status: "pending",
        agent_node_id: "agent-c",
        reasoner_id: "sibling",
      } as any,
    },
  ];

  const edges: DeckEdge[] = [
    {
      id: "a-b",
      path: [
        [0, 0, 0],
        [200, 100, 0],
      ],
      color: [70, 80, 90, 120],
      width: 2,
    },
    {
      id: "b-c",
      path: [
        [200, 100, 0],
        [400, 100, 0],
      ],
      color: [50, 60, 70, 110],
      width: 1.5,
    },
  ];

  it("computes initial fit state, handles hover/click styling, and exposes imperative controls", async () => {
    vi.useFakeTimers();
    const onNodeClick = vi.fn();
    const onNodeHover = vi.fn();
    const ref = React.createRef<WorkflowDeckGLViewHandle>();

    render(
      <WorkflowDeckGLView
        ref={ref}
        nodes={nodes}
        edges={edges}
        onNodeClick={onNodeClick}
        onNodeHover={onNodeHover}
      />,
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(latestDeckProps().viewState.target).toEqual([200, 50, 0]);
    expect(latestDeckProps().viewState.zoom).toBeCloseTo(1.26, 1);
    expect(latestScatterLayer().data).toHaveLength(3);
    expect(latestPathLayer().data).toHaveLength(2);

    act(() => {
      latestScatterLayer().onHover({
        object: nodes[0],
        x: 12,
        y: 34,
      });
      vi.advanceTimersByTime(50);
    });

    expect(onNodeHover).toHaveBeenCalledWith(nodes[0].original);
    expect(screen.getByTestId("hover-panel")).toHaveTextContent("a@12,34");
    expect(latestDeckProps().getCursor()).toBe("pointer");

    act(() => {
      latestScatterLayer().onClick({ object: nodes[0] });
    });

    expect(onNodeClick).toHaveBeenCalledWith(nodes[0].original);

    const styledNodes = latestScatterLayer().data as DeckNode[];
    const styledEdges = latestPathLayer().data as DeckEdge[];
    expect(styledNodes.find((node) => node.id === "a")?.radius).toBeCloseTo(11.5);
    expect(styledNodes.find((node) => node.id === "b")?.borderColor).toEqual([34, 197, 94, 220]);
    expect(styledNodes.find((node) => node.id === "c")?.fillColor[3]).toBeLessThan(nodes[2].fillColor[3]);
    expect(styledEdges.find((edge) => edge.id === "a-b")?.color).toEqual([59, 130, 246, 255]);
    expect(styledEdges.find((edge) => edge.id === "b-c")?.color).toEqual([59, 130, 246, 180]);

    act(() => {
      ref.current?.zoomIn();
      ref.current?.zoomOut();
      ref.current?.fitToContent();
    });

    expect(ref.current).toBeTruthy();
    expect(latestDeckProps().viewState.target).toEqual([200, 50, 0]);
  });

  it("clears hovered node when hover leaves and supports external graph controls", async () => {
    const user = userEvent.setup();
    const fitToContent = vi.fn();
    const zoomIn = vi.fn();
    const zoomOut = vi.fn();

    render(
      <div>
        <WorkflowDeckGLView nodes={nodes} edges={edges} />
        <WorkflowDeckGraphControls
          deckRef={{
            current: { fitToContent, zoomIn, zoomOut },
          }}
        />
      </div>,
    );

    act(() => {
      latestScatterLayer().onHover({ object: undefined });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("hover-panel")).not.toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Fit graph to view" }));
    await user.click(screen.getByRole("button", { name: "Zoom in" }));
    await user.click(screen.getByRole("button", { name: "Zoom out" }));

    expect(fitToContent).toHaveBeenCalledTimes(1);
    expect(zoomIn).toHaveBeenCalledTimes(1);
    expect(zoomOut).toHaveBeenCalledTimes(1);
  });
});
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import {
  COORDINATE_SYSTEM,
  OrthographicController,
  OrthographicView,
  type OrthographicViewState,
} from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import { HoverDetailPanel } from "./HoverDetailPanel";
import type { DeckEdge, DeckNode, WorkflowDAGNode } from "./DeckGLGraph";

interface WorkflowDeckGLViewProps {
  nodes: DeckNode[];
  edges: DeckEdge[];
  onNodeClick?: (node: WorkflowDAGNode) => void;
  onNodeHover?: (node: WorkflowDAGNode | null) => void;
}

const initialViewState: OrthographicViewState = {
  target: [0, 0, 0],
  zoom: 0,
  maxZoom: 8,
  minZoom: -6,
};

export const WorkflowDeckGLView = ({
  nodes,
  edges,
  onNodeClick,
  onNodeHover,
}: WorkflowDeckGLViewProps) => {
  const [viewState, setViewState] =
    useState<OrthographicViewState>(initialViewState);

  // Interactive state management
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [relatedNodeIds, setRelatedNodeIds] = useState<Set<string>>(new Set());
  const [hoverPosition, setHoverPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<WorkflowDAGNode | null>(null);

  // Debounce timer ref
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Build relationship maps for efficient traversal
  const { parentMap, childMap } = useMemo(() => {
    const parentMap = new Map<string, string>();
    const childMap = new Map<string, string[]>();

    nodes.forEach(node => {
      const parentId = node.original.parent_execution_id;
      if (parentId) {
        parentMap.set(node.id, parentId);
        if (!childMap.has(parentId)) {
          childMap.set(parentId, []);
        }
        childMap.get(parentId)!.push(node.id);
      }
    });

    return { parentMap, childMap };
  }, [nodes]);

  // Get related nodes (1 level: direct parents and children)
  const getRelatedNodes = useCallback((nodeId: string): Set<string> => {
    const related = new Set<string>();

    // Add the node itself
    related.add(nodeId);

    // Add parent
    const parent = parentMap.get(nodeId);
    if (parent) {
      related.add(parent);
    }

    // Add children
    const children = childMap.get(nodeId) || [];
    children.forEach(child => related.add(child));

    return related;
  }, [parentMap, childMap]);

  // Debounced hover handler
  const handleHover = useCallback((info: PickingInfo) => {
    // Clear existing timer
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }

    // Debounce hover by 50ms
    hoverTimerRef.current = setTimeout(() => {
      const deckNode = info.object as DeckNode | undefined;

      if (deckNode?.original) {
        setHoveredNodeId(deckNode.id);
        setHoveredNode(deckNode.original);
        setHoverPosition({ x: info.x, y: info.y });
        onNodeHover?.(deckNode.original);
      } else {
        setHoveredNodeId(null);
        setHoveredNode(null);
        onNodeHover?.(null);
      }
    }, 50);
  }, [onNodeHover]);

  // Click handler with relationship traversal
  const handleClick = useCallback((info: PickingInfo) => {
    const deckNode = info.object as DeckNode | undefined;

    if (deckNode?.original) {
      const nodeId = deckNode.id;

      // Toggle selection: if clicking the same node, deselect
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(null);
        setRelatedNodeIds(new Set());
      } else {
        setSelectedNodeId(nodeId);
        const related = getRelatedNodes(nodeId);
        setRelatedNodeIds(related);
      }

      // Call parent handler for sidebar
      onNodeClick?.(deckNode.original);
    }
  }, [selectedNodeId, getRelatedNodes, onNodeClick]);

  // Cleanup hover timer on unmount
  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!nodes.length) return;

    const xs = nodes.map((node) => node.position[0]);
    const ys = nodes.map((node) => node.position[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const padding = 100;
    const width = maxX - minX || 1;
    const height = maxY - minY || 1;

    setViewState((prev) => ({
      ...prev,
      target: [minX + width / 2, minY + height / 2, 0],
      zoom: Math.log2(Math.min(1200 / (width + padding), 800 / (height + padding))),
    }));
  }, [nodes]);

  // Dynamic node styling based on selection and hover state
  const styledNodes = useMemo(() => {
    const hasSelection = selectedNodeId !== null;

    return nodes.map(node => {
      const isSelected = node.id === selectedNodeId;
      const isRelated = relatedNodeIds.has(node.id) && !isSelected;
      const isHovered = node.id === hoveredNodeId;
      const isDimmed = hasSelection && !isSelected && !isRelated;

      // Calculate dynamic properties
      const fillColor = [...node.fillColor] as [number, number, number, number];
      let borderColor = [...node.borderColor] as [number, number, number, number];
      let radius = node.radius;

      if (isDimmed) {
        // Dimmed nodes: 25% opacity, desaturated
        fillColor[3] = Math.round(fillColor[3] * 0.25);
        borderColor[3] = Math.round(borderColor[3] * 0.25);
      } else if (isSelected) {
        // Selected node: bright border, intense glow, scale up
        fillColor[3] = 255;
        borderColor = [59, 130, 246, 255]; // Bright blue border
        radius = node.radius * 1.15;
      } else if (isRelated) {
        // Related nodes: secondary highlight, 90% opacity
        fillColor[3] = Math.round(fillColor[3] * 0.9);
        borderColor = [34, 197, 94, 220]; // Green tint for related
      } else if (isHovered) {
        // Hovered node: scale up slightly
        radius = node.radius * 1.05;
      }

      return {
        ...node,
        fillColor,
        borderColor,
        radius,
      };
    });
  }, [nodes, selectedNodeId, relatedNodeIds, hoveredNodeId]);

  // Dynamic edge styling based on selection
  const styledEdges = useMemo(() => {
    const hasSelection = selectedNodeId !== null;

    return edges.map(edge => {
      // Extract source and target from edge ID (format: "source-target")
      const [sourceId, targetId] = edge.id.split('-');

      const sourceSelected = relatedNodeIds.has(sourceId);
      const targetSelected = relatedNodeIds.has(targetId);
      const isRelatedEdge = sourceSelected && targetSelected;
      const isPartiallyRelated = sourceSelected || targetSelected;
      const isDimmed = hasSelection && !isRelatedEdge && !isPartiallyRelated;

      let color = [...edge.color] as [number, number, number, number];
      let width = edge.width;

      if (isDimmed) {
        // Dimmed edges: very low opacity
        color[3] = Math.round(color[3] * 0.15);
        width = edge.width * 0.6;
      } else if (isRelatedEdge) {
        // Fully related edge: bright and thick
        color = [59, 130, 246, 255]; // Bright blue
        width = edge.width * 1.5;
      } else if (isPartiallyRelated) {
        // Partially related: semi-bright
        color = [59, 130, 246, 180];
        width = edge.width * 1.2;
      }

      return {
        ...edge,
        color,
        width,
      };
    });
  }, [edges, selectedNodeId, relatedNodeIds]);

  const layers = useMemo(() => {
    const nodeLayer = new ScatterplotLayer<DeckNode>({
      id: "workflow-nodes",
      data: styledNodes,
      pickable: true,
      radiusScale: 1,
      radiusMinPixels: 2,
      radiusMaxPixels: 24,
      getPosition: (node) => node.position,
      getRadius: (node) => node.radius,
      getFillColor: (node) => node.fillColor,
      getLineColor: (node) => node.borderColor,
      getLineWidth: () => 1.2,
      lineWidthMinPixels: 1,
      lineWidthMaxPixels: 3,
      stroked: true,
      autoHighlight: false, // Disable auto-highlight, we handle it manually
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      onClick: handleClick,
      onHover: handleHover,
      // Performance: update triggers for efficient re-rendering
      updateTriggers: {
        getFillColor: [selectedNodeId, relatedNodeIds, hoveredNodeId],
        getLineColor: [selectedNodeId, relatedNodeIds],
        getRadius: [selectedNodeId, hoveredNodeId],
      },
      // Smooth transitions
      transitions: {
        getFillColor: 200,
        getLineColor: 200,
        getRadius: 200,
      },
    });

    const edgeLayer = new PathLayer<DeckEdge>({
      id: "workflow-edges",
      data: styledEdges,
      getPath: (edge) => edge.path,
      getColor: (edge) => edge.color,
      getWidth: (edge) => edge.width,
      widthMinPixels: 1,
      widthMaxPixels: 6,
      widthUnits: "pixels",
      rounded: true,
      miterLimit: 2,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      // Performance: update triggers
      updateTriggers: {
        getColor: [selectedNodeId, relatedNodeIds],
        getWidth: [selectedNodeId, relatedNodeIds],
      },
      // Smooth transitions
      transitions: {
        getColor: 200,
        getWidth: 200,
      },
    });

    return [edgeLayer, nodeLayer];
  }, [styledNodes, styledEdges, handleClick, handleHover, selectedNodeId, relatedNodeIds, hoveredNodeId]);

  return (
    <div className="relative h-full w-full bg-muted/30">
      <DeckGL
        views={new OrthographicView({})}
        controller={{ type: OrthographicController, inertia: true }}
        viewState={viewState}
        onViewStateChange={({ viewState: next }) =>
          setViewState(next as OrthographicViewState)
        }
        layers={layers}
        style={{ width: "100%", height: "100%" }}
        getCursor={() => hoveredNodeId ? 'pointer' : 'grab'}
      />

      {/* Hover Detail Panel */}
      <HoverDetailPanel
        node={hoveredNode}
        position={hoverPosition}
        visible={!!hoveredNode && !selectedNodeId}
      />
    </div>
  );
};

import {
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
  type NodeTypes,
  type EdgeTypes,
  Panel,
  Background,
  BackgroundVariant,
  ConnectionMode,
} from "@xyflow/react";
import React, { useCallback, useMemo, useRef, type CSSProperties } from "react";

import { AgentLegend, type AgentLegendLayout } from "./AgentLegend";
import { WorkflowGraphControls } from "./WorkflowGraphControls";
import FloatingConnectionLine from "./FloatingConnectionLine";

interface VirtualizedDAGProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (event: React.MouseEvent, node: Node) => void;
  nodeTypes: Record<string, React.ComponentType<object>>;
  edgeTypes?: Record<string, React.ComponentType<object>>;
  className?: string;
  threshold?: number;
  workflowId: string;
  onAgentFilter: (agentName: string | null) => void;
  selectedAgent: string | null;
  onExpandGraph?: () => void;
  style?: CSSProperties;
  graphLayout?: AgentLegendLayout;
}

export function VirtualizedDAG({
  nodes,
  edges,
  onNodeClick,
  nodeTypes,
  edgeTypes,
  className,
  workflowId,
  onAgentFilter,
  selectedAgent,
  onExpandGraph,
  style,
  graphLayout = "fullscreen",
}: VirtualizedDAGProps) {
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(nodes);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState(edges);
  const { fitView, setViewport, zoomIn, zoomOut } = useReactFlow();

  const defaultViewport = useMemo(
    () => ({ x: 0, y: 0, zoom: 0.8 }),
    []
  );
  const viewportRef = useRef<{ x: number; y: number; zoom: number }>(defaultViewport);
  const hasInitializedViewportRef = useRef(false);
  const viewportStorageKey = useMemo(
    () => `workflowDAGViewport:${workflowId}`,
    [workflowId]
  );

  function isValidSavedViewport(v: unknown): v is { x: number; y: number; zoom: number } {
    if (!v || typeof v !== "object") return false;
    const o = v as Record<string, unknown>;
    return (
      typeof o.x === "number" &&
      Number.isFinite(o.x) &&
      typeof o.y === "number" &&
      Number.isFinite(o.y) &&
      typeof o.zoom === "number" &&
      Number.isFinite(o.zoom) &&
      o.zoom > 0
    );
  }

  React.useEffect(() => {
    hasInitializedViewportRef.current = false;
    viewportRef.current = defaultViewport;
  }, [workflowId, defaultViewport]);

  const fitViewOptions = React.useMemo(
    () => ({
      padding: 0.2,
      includeHiddenNodes: false,
      minZoom: 0,
      maxZoom: 2,
    }),
    []
  );

  const handleNodeClick = useCallback(
    (event: React.MouseEvent, node: Node) => {
      onNodeClick?.(event, node);
    },
    [onNodeClick]
  );

  React.useEffect(() => {
    setFlowNodes(nodes);
  }, [nodes, setFlowNodes]);

  React.useEffect(() => {
    setFlowEdges(edges);
  }, [edges, setFlowEdges]);

  React.useEffect(() => {
    if (flowNodes.length === 0) {
      return;
    }

    if (!hasInitializedViewportRef.current) {
      let rafOuter = 0;
      let rafInner = 0;
      const apply = () => {
        const saved = localStorage.getItem(viewportStorageKey);
        if (saved) {
          try {
            const parsed: unknown = JSON.parse(saved);
            if (isValidSavedViewport(parsed)) {
              viewportRef.current = parsed;
              setViewport(parsed);
              hasInitializedViewportRef.current = true;
              return;
            }
          } catch {
            /* fall through */
          }
        }
        fitView({ padding: 0.2, duration: 0 });
        hasInitializedViewportRef.current = true;
      };
      rafOuter = requestAnimationFrame(() => {
        rafInner = requestAnimationFrame(apply);
      });
      return () => {
        cancelAnimationFrame(rafOuter);
        cancelAnimationFrame(rafInner);
      };
    }

    const vp = viewportRef.current;
    requestAnimationFrame(() => setViewport(vp));
    return undefined;
  }, [flowNodes.length, flowEdges.length, viewportStorageKey, fitView, setViewport]);

  return (
    <ReactFlow
      style={style}
      nodes={flowNodes}
      edges={flowEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      onMoveEnd={(_, viewport) => {
        viewportRef.current = viewport;
        try {
          localStorage.setItem(
            viewportStorageKey,
            JSON.stringify(viewport)
          );
        } catch {
          /* ignore quota / private mode */
        }
      }}
      nodeTypes={nodeTypes as unknown as NodeTypes}
      edgeTypes={edgeTypes as unknown as EdgeTypes}
      connectionLineComponent={FloatingConnectionLine}
      connectionMode={ConnectionMode.Strict}
      nodesDraggable={true}
      nodesConnectable={false}
      elementsSelectable={true}
      className={className}
      fitViewOptions={fitViewOptions}
      defaultViewport={defaultViewport}
      minZoom={0}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
    >
      <Background
        variant={BackgroundVariant.Dots}
        gap={20}
        size={1}
        color="var(--border)"
      />

      <Panel position="top-left" className="z-10">
        <AgentLegend
          layout={graphLayout}
          onFitView={() =>
            void fitView({
              padding: 0.2,
              includeHiddenNodes: false,
              duration: 220,
            })
          }
          onZoomIn={() => void zoomIn({ duration: 200 })}
          onZoomOut={() => void zoomOut({ duration: 200 })}
          onAgentFilter={onAgentFilter}
          selectedAgent={selectedAgent}
          compact={nodes.length <= 20}
          nodes={flowNodes}
          onExpandGraph={onExpandGraph}
        />
      </Panel>
      <WorkflowGraphControls show={graphLayout === "fullscreen"} />
    </ReactFlow>
  );
}

export function VirtualizedDAGWithProvider(props: VirtualizedDAGProps) {
  return (
    <ReactFlowProvider>
      <VirtualizedDAG {...props} />
    </ReactFlowProvider>
  );
}

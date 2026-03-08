import {
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
  Panel,
  Background,
  BackgroundVariant,
  ConnectionMode,
} from "@xyflow/react";
import React, { useCallback, useMemo, useRef } from "react";

import { AgentLegend } from "./AgentLegend";
import FloatingConnectionLine from "./FloatingConnectionLine";

interface VirtualizedDAGProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (event: React.MouseEvent, node: Node) => void;
  nodeTypes: Record<string, React.ComponentType<any>>;
  edgeTypes?: Record<string, React.ComponentType<any>>;
  className?: string;
  threshold?: number;
  workflowId: string;
  onAgentFilter: (agentName: string | null) => void;
  selectedAgent: string | null;
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
}: VirtualizedDAGProps) {
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(nodes);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState(edges);
  const { fitView, setViewport } = useReactFlow();

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
    if (!hasInitializedViewportRef.current) {
      const saved = localStorage.getItem(viewportStorageKey);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          viewportRef.current = parsed;
          setTimeout(() => setViewport(parsed), 0);
        } catch {
          setTimeout(() => fitView({ padding: 0.2 }), 100);
        }
      } else {
        setTimeout(() => fitView({ padding: 0.2 }), 100);
      }
      hasInitializedViewportRef.current = true;
    } else {
      const vp = viewportRef.current;
      setTimeout(() => setViewport(vp), 0);
    }
  }, [flowNodes.length, flowEdges.length, viewportStorageKey, fitView, setViewport]);

  return (
    <ReactFlow
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
        } catch {}
      }}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
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
          onAgentFilter={onAgentFilter}
          selectedAgent={selectedAgent}
          compact={nodes.length <= 20}
          nodes={flowNodes}
        />
      </Panel>
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

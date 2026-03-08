"use client";

import type { Edge, Node, FitViewOptions } from "@xyflow/react";
import {
  Background,
  BackgroundVariant,
  ConnectionMode,
  MarkerType,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";

import { AgentLegend } from "./AgentLegend";
import FloatingConnectionLine from "./FloatingConnectionLine";
import FloatingEdge from "./FloatingEdge";
import { NodeDetailSidebar } from "./NodeDetailSidebar";
import { VirtualizedDAG } from "./VirtualizedDAG";
import { WorkflowNode } from "./WorkflowNode";
import { LayoutManager, type AllLayoutType } from "./layouts/LayoutManager";
import {
  WorkflowDeckGLView,
  buildDeckGraph,
  type DeckGraphData,
} from "./DeckGLView";

import { getWorkflowDAG } from "../../services/workflowsApi";
import type {
  WorkflowDAGLightweightNode,
  WorkflowDAGLightweightResponse,
} from "../../types/workflows";
import { Card, CardContent } from "../ui/card";
import { cn } from "../../lib/utils";
import { formatNumberWithCommas } from "../../utils/numberFormat";

interface WorkflowDAGNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  parent_execution_id?: string;
  workflow_depth: number;
  agent_name?: string;
  task_name?: string;
  children?: WorkflowDAGNode[];
}

export interface WorkflowDAGResponse {
  root_workflow_id: string;
  session_id?: string;
  actor_id?: string;
  total_nodes: number;
  displayed_nodes?: number;
  max_depth: number;
  dag?: WorkflowDAGNode;
  timeline: WorkflowDAGNode[];
  workflow_status?: string;
  workflow_name?: string;
  mode?: "lightweight";
  status_counts?: Record<string, number>;
}

export interface LayoutInfo {
  currentLayout: AllLayoutType;
  availableLayouts: AllLayoutType[];
  isSlowLayout: (layout: AllLayoutType) => boolean;
  isLargeGraph: boolean;
  isApplyingLayout: boolean;
}

export interface WorkflowDAGControls {
  fitToView: (options?: FitViewOptions) => void;
  focusOnNodes: (nodeIds: string[], options?: { padding?: number }) => void;
  changeLayout: (layout: AllLayoutType) => void;
}

function isLightweightDAGResponse(
  data: WorkflowDAGResponse | WorkflowDAGLightweightResponse | null
): data is WorkflowDAGLightweightResponse {
  if (!data) {
    return false;
  }
  return (data as WorkflowDAGLightweightResponse).mode === "lightweight";
}

function mapLightweightNode(
  node: WorkflowDAGLightweightNode,
  workflowId: string
): WorkflowDAGNode {
  return {
    workflow_id: workflowId,
    execution_id: node.execution_id,
    agent_node_id: node.agent_node_id,
    reasoner_id: node.reasoner_id,
    status: node.status,
    started_at: node.started_at,
    completed_at: node.completed_at,
    duration_ms: node.duration_ms,
    parent_execution_id: node.parent_execution_id,
    workflow_depth: node.workflow_depth,
  };
}

function adaptLightweightResponse(
  response: WorkflowDAGLightweightResponse
): WorkflowDAGResponse {
  const timeline = response.timeline.map((node) =>
    mapLightweightNode(node, response.root_workflow_id)
  );

  const dag = timeline.length > 0 ? { ...timeline[0] } : undefined;

  return {
    root_workflow_id: response.root_workflow_id,
    session_id: response.session_id,
    actor_id: response.actor_id,
    total_nodes: response.total_nodes,
    displayed_nodes: timeline.length,
    max_depth: response.max_depth,
    dag,
    timeline,
    workflow_status: response.workflow_status,
    workflow_name: response.workflow_name,
    mode: "lightweight",
  };
}

interface WorkflowDAGViewerProps {
  workflowId: string;
  dagData?: WorkflowDAGResponse | WorkflowDAGLightweightResponse | null;
  loading?: boolean;
  error?: string | null;
  onClose?: () => void;
  onExecutionClick?: (execution: WorkflowDAGNode) => void;
  className?: string;
  searchQuery?: string;
  focusMode?: boolean;
  focusedNodeIds?: string[];
  selectedNodeIds?: string[];
  onReady?: (controls: WorkflowDAGControls) => void;
  onSearchResultsChange?: (payload: {
    totalMatches: number;
    firstMatchId?: string;
  }) => void;
  viewMode?: "standard" | "performance" | "debug";
  onLayoutInfoChange?: (info: LayoutInfo) => void;
}

function WorkflowDAGViewerInner({
  workflowId,
  dagData,
  loading: externalLoading,
  error: externalError,
  className,
  searchQuery,
  focusMode = false,
  focusedNodeIds,
  selectedNodeIds,
  onReady,
  onSearchResultsChange,
  viewMode = "standard",
  onLayoutInfoChange,
}: WorkflowDAGViewerProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);
  const [currentLayout, setCurrentLayout] = useState<AllLayoutType>("tree");
  const [selectedNode, setSelectedNode] = useState<WorkflowDAGNode | null>(
    null
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isApplyingLayout, setIsApplyingLayout] = useState(false);
  const [_layoutProgress, setLayoutProgress] = useState(0);
  const [visualEpoch, setVisualEpoch] = useState(0);
  const hasInitialLayoutRef = useRef(false);
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<Edge[]>([]);
  const controlsRegisteredRef = useRef(false);
  const [internalDagData, setInternalDagData] =
    useState<WorkflowDAGResponse | null>(null);
  const largeGraphRef = useRef(false);
  const [deckGraphData, setDeckGraphData] = useState<DeckGraphData | null>(null);
  const handleLayoutChangeRef = useRef<(layout: AllLayoutType) => void>(() => {});

  const externalDagData = useMemo<WorkflowDAGResponse | null>(() => {
    if (dagData === undefined || dagData === null) {
      return dagData ?? null;
    }
    return isLightweightDAGResponse(dagData)
      ? adaptLightweightResponse(dagData)
      : dagData;
  }, [dagData]);

  const effectiveDagData: WorkflowDAGResponse | null =
    dagData !== undefined ? externalDagData : internalDagData;

  const graphRelationships = useMemo(() => {
    const parentMap = new Map<string, string | null>();
    const childMap = new Map<string, string[]>();

    const timeline: WorkflowDAGNode[] = effectiveDagData?.timeline ?? [];
    timeline.forEach((node) => {
      parentMap.set(node.execution_id, node.parent_execution_id ?? null);
      if (node.parent_execution_id) {
        if (!childMap.has(node.parent_execution_id)) {
          childMap.set(node.parent_execution_id, []);
        }
        childMap.get(node.parent_execution_id)!.push(node.execution_id);
      }
    });

    return { parentMap, childMap };
  }, [effectiveDagData]);

  const durationStats = useMemo(() => {
    const timeline: WorkflowDAGNode[] = effectiveDagData?.timeline ?? [];
    const durations: number[] = timeline
      .map((node) => node.duration_ms ?? 0)
      .filter((value) => value > 0);

    if (!durations.length) {
      return { max: 0, min: 0, avg: 0 };
    }

    const max = Math.max(...durations);
    const min = Math.min(...durations);
    const avg =
      durations.reduce(
        (sum: number, value: number) => sum + value,
        0
      ) / durations.length;
    return { max, min, avg };
  }, [effectiveDagData]);

  // Layout manager instance
  const layoutManager = useMemo(
    () =>
      new LayoutManager({
        enableWorker: import.meta.env?.VITE_ENABLE_LAYOUT_WORKER === "true",
      }),
    []
  );

  // Memoized objects to prevent unnecessary re-renders
  const nodeTypes = useMemo(
    () => ({
      workflow: WorkflowNode,
    }),
    []
  );

  const edgeTypes = useMemo(
    () => ({
      floating: FloatingEdge,
    }),
    []
  );

  const fitViewOptions = useMemo(
    () => ({
      padding: 0.2,
      includeHiddenNodes: false,
      minZoom: 0, // Allow unlimited zoom out for large graphs
      maxZoom: 2,
    }),
    []
  );

  const defaultViewport = useMemo(
    () => ({
      x: 0,
      y: 0,
      zoom: 0.8,
    }),
    []
  );

  // Use external loading/error states if provided, otherwise fall back to internal fetching
  const shouldUseFallback =
    dagData === undefined &&
    externalLoading === undefined &&
    externalError === undefined;
  const [internalLoading, setInternalLoading] = useState(shouldUseFallback);
  const [internalError, setInternalError] = useState<string | null>(null);

  const loading =
    externalLoading !== undefined ? externalLoading : internalLoading;
  const error = externalError !== undefined ? externalError : internalError;

  const { fitView, setViewport, getNodes, fitBounds } = useReactFlow();
  const viewportRef = useRef<{ x: number; y: number; zoom: number }>({
    x: 0,
    y: 0,
    zoom: 0.8,
  });
  const hasInitializedViewportRef = useRef(false);
  const viewportStorageKey = useMemo(
    () => `workflowDAGViewport:${workflowId}`,
    [workflowId]
  );

  // Performance threshold for switching to virtualized rendering
const PERFORMANCE_THRESHOLD = 300;
const LARGE_GRAPH_LAYOUT_THRESHOLD = 2000;
const SIMPLE_LAYOUT_COLUMNS = 40;
const SIMPLE_LAYOUT_X_SPACING = 240;
const SIMPLE_LAYOUT_Y_SPACING = 120;

function applySimpleGridLayout(
  nodes: Node[],
  executionMap: Map<string, WorkflowDAGNode>
): Node[] {
  const sortedNodes = [...nodes].sort((a, b) => {
    const depthA =
      (executionMap.get(a.id)?.workflow_depth as number | undefined) ?? 0;
    const depthB =
      (executionMap.get(b.id)?.workflow_depth as number | undefined) ?? 0;
    if (depthA !== depthB) {
      return depthA - depthB;
    }
    const startedA =
      executionMap.get(a.id)?.started_at ?? "1970-01-01T00:00:00Z";
    const startedB =
      executionMap.get(b.id)?.started_at ?? "1970-01-01T00:00:00Z";
    if (startedA !== startedB) {
      return startedA.localeCompare(startedB);
    }
    return a.id.localeCompare(b.id);
  });

  const columns = Math.max(1, SIMPLE_LAYOUT_COLUMNS);

  return sortedNodes.map((node, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      ...node,
      position: {
        x: column * SIMPLE_LAYOUT_X_SPACING,
        y: row * SIMPLE_LAYOUT_Y_SPACING,
      },
    };
  });
}

function decorateNodesWithViewMode(nodes: Node[], viewMode: string): Node[] {
  return nodes.map((node) => ({
    ...node,
    data: {
      ...(node.data as object),
      viewMode,
    },
  }));
}

function decorateEdgesWithStatus(
  edges: Edge[],
  executionMap: Map<string, WorkflowDAGNode>
): Edge[] {
  return edges.map((edge) => {
    const targetExecution = executionMap.get(edge.target);
    if (!targetExecution) {
      return edge;
    }
    const animated = targetExecution.status === "running";
    return {
      ...edge,
      animated,
      data: {
        ...(edge.data as object),
        status: targetExecution.status,
        duration: targetExecution.duration_ms,
        animated,
      },
    } as Edge;
  });
}
  const shouldUseVirtualizedDAG = useMemo(() => {
    return nodes.length > PERFORMANCE_THRESHOLD;
  }, [nodes.length]);
  const MAX_FOCUS_DEPTH = 2;

  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState(
    searchQuery ?? ""
  );

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setDebouncedSearchQuery(searchQuery ?? "");
    }, 300);

    return () => {
      window.clearTimeout(handle);
    };
  }, [searchQuery]);

  useEffect(() => {
    if (dagData === undefined) {
      setInternalDagData(null);
      hasInitialLayoutRef.current = false;
    }
  }, [workflowId, dagData]);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  useEffect(() => {
    if (!onReady || controlsRegisteredRef.current) {
      return;
    }

    const controls: WorkflowDAGControls = {
      fitToView: (options?: FitViewOptions) => {
        fitView({
          padding: options?.padding ?? 0.2,
          includeHiddenNodes: false,
        });
      },
      focusOnNodes: (nodeIds: string[], options?: { padding?: number }) => {
        if (!nodeIds || nodeIds.length === 0) {
          fitView({
            padding: options?.padding ?? 0.2,
            includeHiddenNodes: false,
          });
          return;
        }

        const nodesToFocus = getNodes().filter((node) =>
          nodeIds.includes(node.id)
        );
        if (nodesToFocus.length === 0) {
          return;
        }

        const bounds = nodesToFocus.reduce(
          (acc, node) => {
            const nodeX = node.position.x;
            const nodeY = node.position.y;
            const width = node.width ?? 240;
            const height = node.height ?? 120;

            const maxX = nodeX + width;
            const maxY = nodeY + height;

            return {
              minX: Math.min(acc.minX, nodeX),
              minY: Math.min(acc.minY, nodeY),
              maxX: Math.max(acc.maxX, maxX),
              maxY: Math.max(acc.maxY, maxY),
            };
          },
          {
            minX: Number.POSITIVE_INFINITY,
            minY: Number.POSITIVE_INFINITY,
            maxX: Number.NEGATIVE_INFINITY,
            maxY: Number.NEGATIVE_INFINITY,
          }
        );

        if (
          !Number.isFinite(bounds.minX) ||
          !Number.isFinite(bounds.minY) ||
          !Number.isFinite(bounds.maxX) ||
          !Number.isFinite(bounds.maxY)
        ) {
          return;
        }

        const rect = {
          x: bounds.minX,
          y: bounds.minY,
          width: Math.max(bounds.maxX - bounds.minX, 1),
          height: Math.max(bounds.maxY - bounds.minY, 1),
        };

        fitBounds(rect, { padding: options?.padding ?? 0.2 });
      },
      changeLayout: (layout: AllLayoutType) => {
        handleLayoutChangeRef.current(layout);
      },
    };

    controlsRegisteredRef.current = true;
    onReady(controls);
  }, [onReady, fitBounds, fitView, getNodes]);

  useEffect(() => {
    const nodesSnapshot = nodesRef.current;
    if (!nodesSnapshot.length) {
      if (onSearchResultsChange) {
        onSearchResultsChange({ totalMatches: 0 });
      }
      return;
    }

    const edgesSnapshot = edgesRef.current;
    const normalizedSearch = (debouncedSearchQuery || "")
      .trim()
      .toLowerCase();
    const focusIds = focusMode
      ? new Set(focusedNodeIds ?? [])
      : new Set<string>();
    const selectedIds = new Set(selectedNodeIds ?? []);

    const focusDistances = new Map<string, number>();
    if (focusMode && focusIds.size > 0) {
      const visited = new Set<string>();
      const queue: Array<{ id: string; distance: number }> = [];
      focusIds.forEach((id) => queue.push({ id, distance: 0 }));

      while (queue.length > 0) {
        const current = queue.shift();
        if (!current) break;
        const { id, distance } = current;
        if (visited.has(id)) continue;
        visited.add(id);
        focusDistances.set(id, distance);

        if (distance >= MAX_FOCUS_DEPTH) {
          continue;
        }

        const parentId = graphRelationships.parentMap.get(id);
        if (parentId && !visited.has(parentId)) {
          queue.push({ id: parentId, distance: distance + 1 });
        }

        const children = graphRelationships.childMap.get(id) || [];
        for (const childId of children) {
          if (!visited.has(childId)) {
            queue.push({ id: childId, distance: distance + 1 });
          }
        }
      }
    }

    const focusActive = focusMode && focusDistances.size > 0;

    const nodeInfos = nodesSnapshot.map((node) => {
      const data = node.data as unknown as WorkflowDAGNode & {
        isSearchMatch?: boolean;
        isDimmed?: boolean;
        isFocusPrimary?: boolean;
        isFocusRelated?: boolean;
        focusDistance?: number;
      };

      const agentLabel = data.agent_name || data.agent_node_id || "";
      const taskLabel = data.task_name || data.reasoner_id || "";
      const statusLabel = data.status || "";
      const searchableSource =
        `${agentLabel} ${taskLabel} ${data.execution_id} ${statusLabel}`.toLowerCase();
      const isMatch = normalizedSearch
        ? searchableSource.includes(normalizedSearch)
        : false;

      const focusDistance = focusDistances.get(node.id);
      const isFocusPrimary = focusDistance === 0;
      const isFocusRelated = focusDistance !== undefined && focusDistance > 0;

      return {
        node,
        data,
        agentLabel,
        taskLabel,
        isMatch,
        focusDistance,
        isFocusPrimary,
        isFocusRelated,
      };
    });

    const hasMatches = normalizedSearch
      ? nodeInfos.some((info) => info.isMatch)
      : false;

    const matchCandidates = nodeInfos
      .filter((info) => info.isMatch)
      .sort((a, b) => {
        const aTime = new Date(a.data.started_at || 0).getTime();
        const bTime = new Date(b.data.started_at || 0).getTime();
        return aTime - bTime;
      });

    const maxPerformance = durationStats.max || 0;

    const decoratedNodes = nodeInfos.map((info) => {
      const isForceHighlight =
        selectedIds.has(info.node.id) || info.isFocusPrimary;

      const shouldDimByAgent =
        !isForceHighlight && selectedAgent
          ? info.agentLabel !== selectedAgent
          : false;
      const shouldDimByFocus =
        !isForceHighlight && focusActive
          ? info.focusDistance === undefined
          : false;
      const shouldDimBySearch =
        !isForceHighlight && normalizedSearch
          ? hasMatches && !info.isMatch
          : false;

      const isDimmed =
        shouldDimByAgent || shouldDimByFocus || shouldDimBySearch;

      const durationValue = info.data.duration_ms ?? 0;
      const performanceIntensity =
        maxPerformance > 0
          ? Math.min(Math.max(durationValue / maxPerformance, 0), 1)
          : 0;

      return {
        ...info.node,
        selected: selectedIds.has(info.node.id),
        style: {
          ...info.node.style,
          opacity: isDimmed ? 0.35 : 1,
          filter: isDimmed ? "grayscale(65%) saturate(60%)" : undefined,
        },
        data: {
          ...info.data,
          isSearchMatch: info.isMatch,
          isDimmed,
          isFocusPrimary: info.isFocusPrimary,
          isFocusRelated: info.isFocusRelated,
          focusDistance: info.focusDistance,
          viewMode,
          performanceIntensity,
        },
      } as Node;
    });

    const infoById = new Map(nodeInfos.map((info) => [info.node.id, info]));

    const decoratedEdges = edgesSnapshot.map((edge) => {
      const sourceInfo = infoById.get(edge.source);
      const targetInfo = infoById.get(edge.target);

      const isSearchEdge = normalizedSearch
        ? !!(sourceInfo?.isMatch || targetInfo?.isMatch)
        : false;

      const connectedToFocus = focusActive
        ? Boolean(
            (sourceInfo?.focusDistance !== undefined &&
              sourceInfo.focusDistance <= 1) ||
              (targetInfo?.focusDistance !== undefined &&
                targetInfo.focusDistance <= 1)
          )
        : false;

      const focusEdgeFull = focusActive
        ? sourceInfo?.focusDistance !== undefined &&
          targetInfo?.focusDistance !== undefined
        : false;

      const shouldDimByAgent = selectedAgent
        ? Boolean(
            (sourceInfo && sourceInfo.agentLabel !== selectedAgent) ||
              (targetInfo && targetInfo.agentLabel !== selectedAgent)
          )
        : false;

      const shouldDimBySearch = normalizedSearch
        ? hasMatches && !isSearchEdge
        : false;

      const shouldDimByFocus = focusActive
        ? !(focusEdgeFull || connectedToFocus)
        : false;

      const isDimmed =
        shouldDimByAgent || shouldDimBySearch || shouldDimByFocus;

      let emphasis: "focus" | "search" | "muted" | "default" = "default";

      if (isDimmed) {
        emphasis = "muted";
      } else if (focusEdgeFull) {
        emphasis = "focus";
      } else if (isSearchEdge || connectedToFocus) {
        emphasis = "search";
      }

      const updatedStyle = {
        ...edge.style,
        opacity: isDimmed ? 0.18 : 1,
        strokeWidth:
          emphasis === "focus"
            ? Math.max((edge.style?.strokeWidth as number) || 2.5, 3.6)
            : emphasis === "search"
              ? Math.max((edge.style?.strokeWidth as number) || 2.5, 3.1)
              : edge.style?.strokeWidth,
        filter: isDimmed ? "grayscale(80%)" : edge.style?.filter,
      } as CSSProperties;

      if (!isDimmed) {
        if (emphasis === "focus") {
          updatedStyle.filter =
            `${updatedStyle.filter || ""} drop-shadow(0 0 6px color-mix(in srgb, var(--status-success) 45%, transparent))`.trim();
        } else if (emphasis === "search") {
          updatedStyle.filter =
            `${updatedStyle.filter || ""} drop-shadow(0 0 6px color-mix(in srgb, var(--status-info) 40%, transparent))`.trim();
        }
      }

      const targetDuration = targetInfo?.data?.duration_ms ?? 0;
      const targetIntensity =
        maxPerformance > 0
          ? Math.min(Math.max(targetDuration / maxPerformance, 0), 1)
          : 0;

      if (!isDimmed && viewMode === "performance") {
        updatedStyle.strokeWidth = Math.max(
          Number(updatedStyle.strokeWidth ?? 2.5),
          2.4 + targetIntensity * 2.2
        );
        const heat = Math.min(80, 35 + targetIntensity * 45);
        updatedStyle.stroke = `color-mix(in srgb, var(--status-info) ${heat}%, transparent)`;
      } else if (viewMode === "debug") {
        updatedStyle.stroke = isDimmed
          ? "color-mix(in srgb, var(--muted-foreground) 45%, transparent)"
          : "color-mix(in srgb, var(--muted-foreground) 65%, transparent)";
        updatedStyle.strokeDasharray = "4,4";
        updatedStyle.opacity = isDimmed ? 0.2 : 0.85;
      }

      return {
        ...edge,
        style: updatedStyle,
        data: {
          ...edge.data,
          emphasis,
        },
      } as Edge;
    });

    nodesRef.current = decoratedNodes;
    edgesRef.current = decoratedEdges;
    setNodes(decoratedNodes);
    setEdges(decoratedEdges);

    if (onSearchResultsChange) {
      onSearchResultsChange({
        totalMatches: matchCandidates.length,
        firstMatchId: matchCandidates[0]?.node.id,
      });
    }
  }, [
    debouncedSearchQuery,
    focusMode,
    focusedNodeIds,
    selectedAgent,
    selectedNodeIds,
    graphRelationships,
    visualEpoch,
    onSearchResultsChange,
    setNodes,
    setEdges,
    viewMode,
    durationStats,
  ]);

  // Handle node click to open sidebar - properly memoized
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const nodeData = node.data as unknown as WorkflowDAGNode;
      setSelectedNode(nodeData);
      setSidebarOpen(true);
    },
    []
  );

  // Handle sidebar close
  const handleCloseSidebar = useCallback(() => {
    setSidebarOpen(false);
    // Optionally clear selected node after animation
    setTimeout(() => setSelectedNode(null), 300);
  }, []);

  // Handle agent filter - optimized to avoid dependency on nodes array
  const handleAgentFilter = useCallback((agentName: string | null) => {
    setSelectedAgent(agentName);
  }, []);

  const buildGraphElements = useCallback(
    (timeline: WorkflowDAGNode[]) => {
      const executionMap = new Map<string, WorkflowDAGNode>();

      const nodesForLayout: Node[] = timeline.map((execution) => {
        executionMap.set(execution.execution_id, execution);
        return {
          id: execution.execution_id,
          type: "workflow",
          position: { x: 0, y: 0 },
          data: {
            ...execution,
            viewMode,
          },
        } as Node;
      });

      const edgesForLayout: Edge[] = timeline.reduce((acc, execution) => {
        if (!execution.parent_execution_id) {
          return acc;
        }

        acc.push({
          id: `${execution.parent_execution_id}-${execution.execution_id}`,
          source: execution.parent_execution_id,
          target: execution.execution_id,
          type: "floating",
          animated: execution.status === "running",
          markerEnd: { type: MarkerType.Arrow },
          data: {
            status: execution.status,
            duration: execution.duration_ms,
            animated: execution.status === "running",
          },
        } as Edge);

        return acc;
      }, [] as Edge[]);

      return { nodesForLayout, edgesForLayout, executionMap };
    },
    [viewMode]
  );

  // Handle layout change
  const handleLayoutChange = useCallback(
    async (newLayout: AllLayoutType) => {
      if (largeGraphRef.current) {
        return;
      }
      if (newLayout === currentLayout) return;

      setIsApplyingLayout(true);
      setLayoutProgress(0);
      setCurrentLayout(newLayout);

      // Re-apply layout to existing nodes and edges
      if (nodes.length > 0) {
        try {
          const { nodes: layoutedNodes, edges: layoutedEdges } =
            await layoutManager.applyLayout(
              nodes,
              edges,
              newLayout,
              (progress) => setLayoutProgress(progress)
            );

          setNodes(layoutedNodes);
          setEdges(layoutedEdges);
          nodesRef.current = layoutedNodes;
          edgesRef.current = layoutedEdges;
          setVisualEpoch((epoch) => epoch + 1);

          // Preserve current viewport after layout change
          const vp = viewportRef.current;
          setTimeout(() => setViewport(vp), 0);
        } catch (error) {
          console.error("Layout change failed:", error);
        } finally {
          setIsApplyingLayout(false);
          setLayoutProgress(0);
        }
      } else {
        setIsApplyingLayout(false);
        setLayoutProgress(0);
      }
    },
    [
      currentLayout,
      nodes,
      edges,
      setNodes,
      setEdges,
      layoutManager,
      setViewport,
    ]
  );

  // Keep the ref in sync so controls.changeLayout() always uses latest
  handleLayoutChangeRef.current = handleLayoutChange;

  // Notify parent of layout state changes
  useEffect(() => {
    if (!onLayoutInfoChange) return;
    onLayoutInfoChange({
      currentLayout,
      availableLayouts: layoutManager.getAvailableLayouts(nodes.length),
      isSlowLayout: (layout: AllLayoutType) => layoutManager.isSlowLayout(layout),
      isLargeGraph: layoutManager.isLargeGraph(nodes.length),
      isApplyingLayout,
    });
  }, [currentLayout, isApplyingLayout, nodes.length, layoutManager, onLayoutInfoChange]);

  // Utility: merge new DAG data incrementally without resetting positions
  const mergeIncrementalUpdate = useCallback(
    async (data: WorkflowDAGResponse) => {
      const timeline = data.timeline || [];
      const { nodesForLayout, edgesForLayout, executionMap } =
        buildGraphElements(timeline);

      if (largeGraphRef.current) {
        const flowNodes = applySimpleGridLayout(
          nodesForLayout,
          executionMap
        );
        const nodesWithMode = decorateNodesWithViewMode(flowNodes, viewMode);
        const edgesWithStatus = decorateEdgesWithStatus(
          edgesForLayout,
          executionMap
        );
        nodesRef.current = nodesWithMode;
        edgesRef.current = edgesWithStatus;
        setNodes(nodesWithMode);
        setEdges(edgesWithStatus);
        setVisualEpoch((epoch) => epoch + 1);
        return;
      }

      const existingIds = new Set(nodesRef.current.map((node) => node.id));
      const timelineIds = new Set(timeline.map((node) => node.execution_id));

      const hasNewNodes = nodesForLayout.some(
        (node) => !existingIds.has(node.id)
      );
      const hasRemovedNodes = nodesRef.current.some(
        (node) => !timelineIds.has(node.id)
      );

      if (hasNewNodes || hasRemovedNodes) {
        try {
          const { nodes: layoutedNodes, edges: layoutedEdges } =
            await layoutManager.applyLayout(
              nodesForLayout,
              edgesForLayout,
              currentLayout
            );

          const nodesWithMode = layoutedNodes.map((node) => ({
            ...node,
            data: {
              ...(node.data as object),
              viewMode,
            },
          }));

          const edgesWithStatus = layoutedEdges.map((edge) => {
            const targetExecution = executionMap.get(edge.target);
            if (!targetExecution) {
              return edge;
            }
            const animated = targetExecution.status === "running";
            return {
              ...edge,
              animated,
              data: {
                ...(edge.data as object),
                status: targetExecution.status,
                duration: targetExecution.duration_ms,
                animated,
              },
            } as Edge;
          });

          nodesRef.current = nodesWithMode;
          edgesRef.current = edgesWithStatus;
          setNodes(nodesWithMode);
          setEdges(edgesWithStatus);
          setVisualEpoch((epoch) => epoch + 1);
        } catch (error) {
          console.error("Incremental layout failed:", error);
        }
        return;
      }

      const updatedNodes = nodesRef.current.map((node) => {
        const execution = executionMap.get(node.id);
        if (!execution) {
          return node;
        }

        return {
          ...node,
          data: {
            ...(node.data as object),
            ...execution,
            viewMode,
          },
        } as Node;
      });

      const updatedEdges = edgesRef.current.map((edge) => {
        const targetExecution = executionMap.get(edge.target);
        if (!targetExecution) {
          return edge;
        }
        const animated = targetExecution.status === "running";
        return {
          ...edge,
          animated,
          data: {
            ...(edge.data as object),
            status: targetExecution.status,
            duration: targetExecution.duration_ms,
            animated,
          },
        } as Edge;
      });

      nodesRef.current = updatedNodes;
      edgesRef.current = updatedEdges;
      setNodes(updatedNodes);
      setEdges(updatedEdges);
      setVisualEpoch((epoch) => epoch + 1);
    },
    [
      buildGraphElements,
      currentLayout,
      layoutManager,
      setEdges,
      setNodes,
      setVisualEpoch,
      viewMode,
    ]
  );

  // Process DAG data (either from props or internal fetch)
  useEffect(() => {
    const processDAGData = async () => {
      let data: WorkflowDAGResponse | null = null;

      if (effectiveDagData) {
        data = effectiveDagData;
      } else if (shouldUseFallback) {
        try {
          setInternalLoading(true);
          setInternalError(null);
          const fetched = await getWorkflowDAG<
            WorkflowDAGResponse | WorkflowDAGLightweightResponse
          >(workflowId, { lightweight: true });

          const normalized = isLightweightDAGResponse(fetched)
            ? adaptLightweightResponse(fetched)
            : fetched;

          setInternalDagData(normalized);
          data = normalized;
        } catch (err) {
          const errorMessage =
            (err as Error)?.message || "Failed to load workflow DAG";
          setInternalError(errorMessage);
          setInternalDagData(null);
          return;
        } finally {
          setInternalLoading(false);
        }
      }

      // Process the data if we have it
      if (data) {
        console.log("🔍 DAG DEBUG: Processing data:", data);
        console.log("🔍 DAG DEBUG: Timeline executions:", data.timeline);

        const timeline = data.timeline ?? [];

        // Determine the appropriate default layout based on graph size
        const nodeCount = timeline.length;
        const defaultLayout = layoutManager.getDefaultLayout(nodeCount);
        const useSimpleLayout = nodeCount > LARGE_GRAPH_LAYOUT_THRESHOLD;
        largeGraphRef.current = useSimpleLayout;
        const { nodesForLayout, edgesForLayout, executionMap } =
          buildGraphElements(timeline);

        // For large graphs, build DeckGL data instead of React Flow layout
        if (useSimpleLayout) {
          const flowNodes = applySimpleGridLayout(
            nodesForLayout,
            executionMap
          );
          const nodesWithMode = decorateNodesWithViewMode(flowNodes, viewMode);
          const edgesWithStatus = decorateEdgesWithStatus(
            edgesForLayout,
            executionMap
          );
          setNodes(nodesWithMode);
          setEdges(edgesWithStatus);
          nodesRef.current = nodesWithMode;
          edgesRef.current = edgesWithStatus;
          setVisualEpoch((epoch) => epoch + 1);
          const deckData = buildDeckGraph(timeline);
          setDeckGraphData(deckData);
          hasInitialLayoutRef.current = true;
          return; // Skip React Flow layout
        }

        // Update current layout if it's still the initial "tree" value
        if (!useSimpleLayout && currentLayout === "tree" && defaultLayout !== "tree") {
          setCurrentLayout(defaultLayout);
        }

        // First time: compute full layout; subsequent times: merge incrementally
        if (!hasInitialLayoutRef.current) {
          const layoutToUse =
            currentLayout === "tree" ? defaultLayout : currentLayout;

          let flowNodes: Node[];
          let flowEdges: Edge[] = edgesForLayout;

          const { nodes: layoutedNodes, edges: layoutedEdges } =
            await layoutManager.applyLayout(
              nodesForLayout,
              edgesForLayout,
              layoutToUse
            );
          flowNodes = layoutedNodes;
          flowEdges = layoutedEdges;

          const nodesWithMode = decorateNodesWithViewMode(flowNodes, viewMode);
          const edgesWithStatus = decorateEdgesWithStatus(
            flowEdges,
            executionMap
          );

          setNodes(nodesWithMode);
          setEdges(edgesWithStatus);
          nodesRef.current = nodesWithMode;
          edgesRef.current = edgesWithStatus;
          setVisualEpoch((epoch) => epoch + 1);
          hasInitialLayoutRef.current = true;
        } else {
          await mergeIncrementalUpdate(data);
        }

        // Initialize viewport only once: restore saved viewport if present,
        // otherwise do a single fitView on first render.
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
          // On subsequent data refreshes, preserve current viewport
          const vp = viewportRef.current;
          setTimeout(() => setViewport(vp), 0);
        }
      }
    };

    processDAGData();
  }, [
    workflowId,
    effectiveDagData,
    currentLayout,
    shouldUseFallback,
    layoutManager,
    fitView,
    buildGraphElements,
    mergeIncrementalUpdate,
    setNodes,
    setEdges,
    setViewport,
    viewMode,
    viewportStorageKey,
  ]);

  if (loading) {
    return <WorkflowDAGSkeleton className={className} />;
  }

  if (error) {
    return (
      <Card className={cn("flex h-full flex-col", className)}>
        <CardContent className="flex flex-1 items-center justify-center px-6 py-12 text-center">
          <div>
            <div className="mb-2 text-red-600">Failed to load workflow DAG</div>
            <div className="text-body-small">{error}</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const shouldUseDeckGL = nodes.length >= LARGE_GRAPH_LAYOUT_THRESHOLD;

  // Handler for DeckGL node clicks - convert DeckGL node type to local type
  const handleDeckNodeClick = useCallback(
    (node: any) => {
      // Ensure workflow_id is set
      const localNode: WorkflowDAGNode = {
        ...node,
        workflow_id: node.workflow_id || workflowId,
      };
      setSelectedNode(localNode);
      setSidebarOpen(true);
    },
    [workflowId]
  );

  // Handler for DeckGL node hover
  const handleDeckNodeHover = useCallback(
    (_node: any) => {
      // Optional: Add hover state handling if needed
    },
    []
  );

  // Render DeckGL view for large graphs
  if (shouldUseDeckGL && deckGraphData) {
    const totalNodes =
      effectiveDagData?.total_nodes ?? deckGraphData.nodes.length;
    const displayedNodes =
      effectiveDagData?.displayed_nodes ?? deckGraphData.nodes.length;
    const hasTruncation = totalNodes > displayedNodes;

    return (
      <div className={cn("relative h-full w-full", className)}>
        <div className="flex h-full w-full flex-col">
          <div className="flex-1 overflow-hidden min-h-0">
            <div className="relative flex h-full w-full flex-1 overflow-hidden min-h-0">
              <WorkflowDeckGLView
                nodes={deckGraphData.nodes}
                edges={deckGraphData.edges}
                onNodeClick={handleDeckNodeClick}
                onNodeHover={handleDeckNodeHover}
              />

              {/* Agent Legend - positioned in top-left */}
              <div className="absolute top-4 left-4 z-30">
                <AgentLegend
                  onAgentFilter={handleAgentFilter}
                  selectedAgent={selectedAgent}
                  compact={false}
                  nodes={nodes}
                />
              </div>

              {/* Large Graph Indicator - positioned in top-right */}
              <div className="absolute top-4 right-4 z-30">
                <Card className="bg-card/95 backdrop-blur-sm border-border shadow-lg">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2 text-sm">
                      <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                      <span className="font-medium text-foreground">
                        Large Graph Mode
                      </span>
                      <span className="text-muted-foreground">
                        {hasTruncation
                          ? `(${formatNumberWithCommas(
                              displayedNodes
                            )} shown / ${formatNumberWithCommas(totalNodes)} total)`
                          : `(${formatNumberWithCommas(totalNodes)} nodes)`}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </div>

        {/* Node Detail Sidebar */}
        <NodeDetailSidebar
          node={selectedNode}
          isOpen={sidebarOpen}
          onClose={handleCloseSidebar}
        />
      </div>
    );
  }

  // Render React Flow for normal-sized graphs
  return (
    <div className={cn("relative h-full w-full", className)}>
      <div className="flex h-full w-full flex-col">
        <div className="flex-1 overflow-hidden min-h-0">
          <div className="relative flex h-full w-full flex-1 overflow-hidden bg-muted/30 min-h-0">
            {shouldUseVirtualizedDAG ? (
              <VirtualizedDAG
                nodes={nodes}
                edges={edges}
                onNodeClick={handleNodeClick}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                className="h-full w-full"
                threshold={PERFORMANCE_THRESHOLD}
                workflowId={workflowId}
                onAgentFilter={handleAgentFilter}
                selectedAgent={selectedAgent}
              />
            ) : (
              <ReactFlow
                className="h-full w-full"
                nodes={nodes}
                edges={edges}
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
                  } catch (storageError) {
                    console.warn(
                      "Failed to persist workflow DAG viewport",
                      storageError
                    );
                  }
                }}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                connectionLineComponent={FloatingConnectionLine}
                connectionMode={ConnectionMode.Strict}
                // Allow node dragging but disable edge creation
                nodesDraggable={true}
                nodesConnectable={false}
                elementsSelectable={true}
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

                {/* Agent Legend */}
                <Panel position="top-left" className="z-30">
                  <AgentLegend
                    onAgentFilter={handleAgentFilter}
                    selectedAgent={selectedAgent}
                    compact={nodes.length <= 20}
                    nodes={nodes}
                  />
                </Panel>
              </ReactFlow>
            )}
          </div>
        </div>
      </div>

      {/* Node Detail Sidebar - Rendered at root level with high z-index */}
      <NodeDetailSidebar
        node={selectedNode}
        isOpen={sidebarOpen}
        onClose={handleCloseSidebar}
      />
    </div>
  );
}

interface WorkflowDAGSkeletonProps {
  className?: string;
}

function WorkflowDAGSkeleton({ className }: WorkflowDAGSkeletonProps) {
  return (
    <Card className={cn("flex h-full w-full flex-col", className)}>
      <CardContent className="flex-1 p-0 min-h-0">
        <div className="flex h-full w-full items-center justify-center bg-muted/20">
          <div className="space-y-4 text-center">
            <div className="mx-auto h-8 w-8 animate-spin rounded-full border-b-2 border-primary"></div>
            <div className="text-body-small">
              Loading workflow DAG...
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Export the wrapped component with ReactFlowProvider
export function WorkflowDAGViewer(props: WorkflowDAGViewerProps) {
  return (
    <ReactFlowProvider>
      <WorkflowDAGViewerInner {...props} />
    </ReactFlowProvider>
  );
}

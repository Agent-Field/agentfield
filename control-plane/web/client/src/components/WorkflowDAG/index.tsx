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
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

import { AgentLegend } from "./AgentLegend";
import { WorkflowGraphControls } from "./WorkflowGraphControls";
import FloatingConnectionLine from "./FloatingConnectionLine";
import FloatingEdge from "./FloatingEdge";
import { NodeDetailSidebar } from "./NodeDetailSidebar";
import { VirtualizedDAG } from "./VirtualizedDAG";
import { WorkflowNode } from "./WorkflowNode";
import { LayoutManager, type AllLayoutType } from "./layouts/LayoutManager";
import {
  adaptLightweightResponse,
  applySimpleGridLayout,
  decorateEdgesWithStatus,
  decorateNodesWithViewMode,
  isLightweightDAGResponse,
  LARGE_GRAPH_LAYOUT_THRESHOLD,
  PERFORMANCE_THRESHOLD,
  type WorkflowDAGNode,
  type WorkflowDAGResponse,
} from "./workflowDagUtils";
import {
  WorkflowDeckGLView,
  WorkflowDeckGraphControls,
  type WorkflowDeckGLViewHandle,
} from "./DeckGLView";
import { buildDeckGraph, type DeckGraphData } from "./DeckGLGraph";

import { getWorkflowDAG } from "../../services/workflowsApi";
import type {
  WorkflowDAGLightweightResponse,
} from "../../types/workflows";
import { X } from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { cn } from "../../lib/utils";
import { formatNumberWithCommas } from "../../utils/numberFormat";

export type { WorkflowDAGNode, WorkflowDAGResponse } from "./workflowDagUtils";

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

function WorkflowGraphViewport({
  expanded,
  onCollapse,
  workflowTitle,
  children,
}: {
  expanded: boolean;
  onCollapse: () => void;
  workflowTitle?: string | null;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!expanded) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [expanded]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCollapse();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, onCollapse]);

  // Non-expanded: avoid `absolute inset-0` — it removes the pane from document flow so the
  // outer `relative h-full` box can collapse to 0 height when `%` heights don't resolve,
  // which triggers React Flow error #004 (parent needs width and height).
  return (
    <div
      className={cn(
        "flex h-full w-full min-h-0 flex-1 flex-col",
        expanded && "relative min-h-[min(380px,40vh)]",
      )}
    >
      <div
        className={cn(
          expanded
            ? "fixed inset-0 z-[100] flex flex-col bg-background"
            : "flex min-h-0 flex-1 flex-col",
        )}
      >
        {expanded ? (
          <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border bg-card px-3 shadow-sm sm:h-14 sm:px-4">
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold leading-tight text-foreground">
                Workflow graph
              </p>
              {workflowTitle ? (
                <p className="truncate text-xs text-muted-foreground">
                  {workflowTitle}
                </p>
              ) : null}
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="shrink-0"
              onClick={onCollapse}
              aria-label="Exit full screen"
            >
              <X className="size-4" />
            </Button>
          </header>
        ) : null}
        <div className="flex min-h-[280px] flex-1 flex-col">{children}</div>
      </div>
    </div>
  );
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
  onExecutionClick,
}: WorkflowDAGViewerProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);
  const [currentLayout, setCurrentLayout] = useState<AllLayoutType>("tree");
  const [selectedNode, setSelectedNode] = useState<WorkflowDAGNode | null>(
    null
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [graphExpanded, setGraphExpanded] = useState(false);
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

  const { fitView, setViewport, getNodes, fitBounds, zoomIn, zoomOut } =
    useReactFlow();
  const viewportRef = useRef<{ x: number; y: number; zoom: number }>({
    x: 0,
    y: 0,
    zoom: 0.8,
  });
  const hasInitializedViewportRef = useRef(false);
  /** Sentinel: skip reset on first mount (only reset when switching workflows). */
  const prevWorkflowIdForResetRef = useRef<string | undefined>(undefined);
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

  // New run / workflow: must reset layout and viewport; otherwise we keep the previous
  // graph's pan/zoom and the new nodes render off-screen (empty-looking graph).
  useEffect(() => {
    const prev = prevWorkflowIdForResetRef.current;
    if (prev === workflowId) {
      return;
    }
    prevWorkflowIdForResetRef.current = workflowId;
    if (prev !== undefined) {
      hasInitialLayoutRef.current = false;
      hasInitializedViewportRef.current = false;
      viewportRef.current = { x: 0, y: 0, zoom: 0.8 };
      largeGraphRef.current = false;
      setDeckGraphData(null);
      setNodes([]);
      setEdges([]);
      nodesRef.current = [];
      edgesRef.current = [];
    }
  }, [workflowId, setEdges, setNodes]);

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

  // Handle node click — delegate to parent via onExecutionClick when provided,
  // otherwise fall back to the internal NodeDetailSidebar (legacy usage).
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const nodeData = node.data as unknown as WorkflowDAGNode;
      if (onExecutionClick && nodeData) {
        onExecutionClick(nodeData);
      } else {
        setSelectedNode(nodeData);
        setSidebarOpen(true);
      }
    },
    [onExecutionClick]
  );

  // Handle sidebar close
  const closeSidebarTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const handleCloseSidebar = useCallback(() => {
    setSidebarOpen(false);
    if (closeSidebarTimeoutRef.current) {
      clearTimeout(closeSidebarTimeoutRef.current);
    }
    // Clear selected node after animation
    closeSidebarTimeoutRef.current = setTimeout(() => {
      setSelectedNode(null);
      closeSidebarTimeoutRef.current = null;
    }, 300);
  }, []);
  useEffect(() => {
    return () => {
      if (closeSidebarTimeoutRef.current) {
        clearTimeout(closeSidebarTimeoutRef.current);
      }
    };
  }, []);

  // Handle agent filter - optimized to avoid dependency on nodes array
  const handleAgentFilter = useCallback((agentName: string | null) => {
    setSelectedAgent(agentName);
  }, []);

  const handleDeckNodeClick = useCallback(
    (node: WorkflowDAGNode) => {
      const localNode: WorkflowDAGNode = {
        ...node,
        workflow_id: node.workflow_id || workflowId,
      };
      if (onExecutionClick && localNode) {
        onExecutionClick(localNode);
      } else {
        setSelectedNode(localNode);
        setSidebarOpen(true);
      }
    },
    [onExecutionClick, workflowId],
  );

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

        // After incremental updates, keep the user's pan/zoom (nodes already laid out).
        if (hasInitializedViewportRef.current) {
          const vp = viewportRef.current;
          requestAnimationFrame(() => setViewport(vp));
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
    buildGraphElements,
    mergeIncrementalUpdate,
    setNodes,
    setEdges,
    setViewport,
    viewMode,
  ]);

  const shouldUseDeckGL = nodes.length >= LARGE_GRAPH_LAYOUT_THRESHOLD;

  const graphLayout = graphExpanded ? "fullscreen" : "embedded";

  const handleFitView = useCallback(() => {
    void fitView({
      padding: 0.2,
      includeHiddenNodes: false,
      duration: 220,
    });
  }, [fitView]);

  const handleZoomIn = useCallback(() => {
    void zoomIn({ duration: 200 });
  }, [zoomIn]);

  const handleZoomOut = useCallback(() => {
    void zoomOut({ duration: 200 });
  }, [zoomOut]);

  // Apply viewport only after React Flow has committed measured nodes. Running fitView /
  // setViewport inside processDAGData right after setNodes() races the render and often
  // fits an empty graph, leaving the real nodes off-screen.
  useEffect(() => {
    if (
      loading ||
      error ||
      nodes.length >= LARGE_GRAPH_LAYOUT_THRESHOLD ||
      nodes.length === 0
    ) {
      return;
    }

    if (hasInitializedViewportRef.current) {
      return;
    }

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
          /* fall through to fitView */
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
  }, [
    loading,
    error,
    nodes.length,
    viewportStorageKey,
    fitView,
    setViewport,
  ]);

  const flowContainerRef = useRef<HTMLDivElement>(null);
  const deckGlRef = useRef<WorkflowDeckGLViewHandle>(null);
  const lastFlowHeightRef = useRef(0);

  // React Flow measures the pane on mount; if the flex parent had no height yet (e.g. after
  // switching from trace), refit when the container gains usable height.
  useEffect(() => {
    if (loading || error || shouldUseDeckGL) return;
    const el = flowContainerRef.current;
    if (!el) return;

    lastFlowHeightRef.current = el.getBoundingClientRect().height;
    const ro = new ResizeObserver(() => {
      const h = el.getBoundingClientRect().height;
      const prev = lastFlowHeightRef.current;
      lastFlowHeightRef.current = h;
      if (prev < 72 && h >= 72) {
        requestAnimationFrame(() => fitView({ padding: 0.2, duration: 0 }));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fitView, loading, error, shouldUseDeckGL, nodes.length]);

  if (loading) {
    return <WorkflowDAGSkeleton className={className} />;
  }

  if (error) {
    return (
      <Card className={cn("flex h-full flex-col", className)}>
        <CardContent className="flex flex-1 items-center justify-center px-6 py-12 text-center">
          <div>
            <div className="mb-2 text-red-600">Failed to load workflow DAG</div>
            <div className="text-sm text-muted-foreground">{error}</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render DeckGL view for large graphs
  if (shouldUseDeckGL && deckGraphData) {
    const totalNodes =
      effectiveDagData?.total_nodes ?? deckGraphData.nodes.length;
    const displayedNodes =
      effectiveDagData?.displayed_nodes ?? deckGraphData.nodes.length;
    const hasTruncation = totalNodes > displayedNodes;

    return (
      <WorkflowGraphViewport
        expanded={graphExpanded}
        onCollapse={() => setGraphExpanded(false)}
        workflowTitle={effectiveDagData?.workflow_name}
      >
        <div className={cn("relative flex h-full w-full min-h-0 flex-1 flex-col", className)}>
          <div className="flex min-h-[280px] flex-1 flex-col">
            <div className="flex min-h-[280px] flex-1 flex-col overflow-hidden">
              <div
                ref={flowContainerRef}
                className="relative flex w-full flex-1 flex-col overflow-hidden"
                style={{
                  minHeight: "max(280px, min(50vh, 24rem))",
                  width: "100%",
                  flex: "1 1 0%",
                }}
              >
                <WorkflowDeckGLView
                  ref={deckGlRef}
                  nodes={deckGraphData.nodes}
                  edges={deckGraphData.edges}
                  onNodeClick={handleDeckNodeClick as unknown as (node: import("./DeckGLGraph").WorkflowDAGNode) => void}
                />

                {graphExpanded ? (
                  <div className="absolute bottom-4 right-4 z-30">
                    <WorkflowDeckGraphControls deckRef={deckGlRef} />
                  </div>
                ) : null}

                <div className="absolute left-4 top-4 z-30">
                  <AgentLegend
                    layout={graphLayout}
                    onFitView={() => deckGlRef.current?.fitToContent()}
                    onZoomIn={() => deckGlRef.current?.zoomIn()}
                    onZoomOut={() => deckGlRef.current?.zoomOut()}
                    onAgentFilter={handleAgentFilter}
                    selectedAgent={selectedAgent}
                    compact={false}
                    nodes={nodes}
                    onExpandGraph={() => setGraphExpanded(true)}
                  />
                </div>

                <div className="absolute right-4 top-4 z-30">
                  <Card className="border-border/80 bg-card/95 shadow-md backdrop-blur-sm">
                    <CardContent className="flex items-center gap-2 p-3 text-sm">
                      <div className="size-2 shrink-0 animate-pulse rounded-full bg-primary" />
                      <span className="font-medium text-foreground">
                        Large graph
                      </span>
                      <span className="text-muted-foreground">
                        {hasTruncation
                          ? `(${formatNumberWithCommas(
                              displayedNodes,
                            )} shown / ${formatNumberWithCommas(totalNodes)} total)`
                          : `(${formatNumberWithCommas(totalNodes)} nodes)`}
                      </span>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          </div>

          {!onExecutionClick && (
            <NodeDetailSidebar
              node={selectedNode}
              isOpen={sidebarOpen}
              onClose={handleCloseSidebar}
            />
          )}
        </div>
      </WorkflowGraphViewport>
    );
  }

  // Render React Flow for normal-sized graphs
  return (
    <WorkflowGraphViewport
      expanded={graphExpanded}
      onCollapse={() => setGraphExpanded(false)}
      workflowTitle={effectiveDagData?.workflow_name}
    >
      <div className={cn("relative flex h-full w-full min-h-0 flex-1 flex-col", className)}>
        <div className="flex min-h-[280px] flex-1 flex-col">
          <div className="flex min-h-[280px] flex-1 flex-col overflow-hidden">
            <div
              ref={flowContainerRef}
              className="relative flex w-full flex-1 flex-col overflow-hidden bg-muted/30"
              style={{
                minHeight: "max(280px, min(50vh, 24rem))",
                width: "100%",
                flex: "1 1 0%",
              }}
            >
            {shouldUseVirtualizedDAG ? (
              <VirtualizedDAG
                nodes={nodes}
                edges={edges}
                onNodeClick={handleNodeClick}
                nodeTypes={nodeTypes as Record<string, React.ComponentType<object>>}
                edgeTypes={edgeTypes as Record<string, React.ComponentType<object>>}
                className="min-h-[280px] w-full flex-1"
                style={{ width: "100%", height: "100%", minHeight: 280 }}
                threshold={PERFORMANCE_THRESHOLD}
                workflowId={workflowId}
                graphLayout={graphLayout}
                onAgentFilter={handleAgentFilter}
                selectedAgent={selectedAgent}
                onExpandGraph={() => setGraphExpanded(true)}
              />
            ) : (
              <ReactFlow
                className="min-h-[280px] w-full flex-1"
                style={{ width: "100%", height: "100%", minHeight: 280 }}
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
                    layout={graphLayout}
                    onFitView={handleFitView}
                    onZoomIn={handleZoomIn}
                    onZoomOut={handleZoomOut}
                    onAgentFilter={handleAgentFilter}
                    selectedAgent={selectedAgent}
                    compact={nodes.length <= 20}
                    nodes={nodes}
                    onExpandGraph={() => setGraphExpanded(true)}
                  />
                </Panel>
                <WorkflowGraphControls show={graphExpanded} />
              </ReactFlow>
            )}
            </div>
          </div>
        </div>

        {!onExecutionClick && (
          <NodeDetailSidebar
            node={selectedNode}
            isOpen={sidebarOpen}
            onClose={handleCloseSidebar}
          />
        )}
      </div>
    </WorkflowGraphViewport>
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
            <div className="text-sm text-muted-foreground">
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

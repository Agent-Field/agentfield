import type { Edge, Node } from "@xyflow/react";

import type {
  WorkflowDAGLightweightNode,
  WorkflowDAGLightweightResponse,
} from "../../types/workflows";

export interface WorkflowDAGNode {
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

export const PERFORMANCE_THRESHOLD = 300;
export const LARGE_GRAPH_LAYOUT_THRESHOLD = 2000;
export const SIMPLE_LAYOUT_COLUMNS = 40;
export const SIMPLE_LAYOUT_X_SPACING = 240;
export const SIMPLE_LAYOUT_Y_SPACING = 120;

export function isLightweightDAGResponse(
  data: WorkflowDAGResponse | WorkflowDAGLightweightResponse | null
): data is WorkflowDAGLightweightResponse {
  if (!data) {
    return false;
  }

  return (data as WorkflowDAGLightweightResponse).mode === "lightweight";
}

export function mapLightweightNode(
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

export function adaptLightweightResponse(
  response: WorkflowDAGLightweightResponse
): WorkflowDAGResponse {
  const timeline = response.timeline.map((node) =>
    mapLightweightNode(node, response.root_workflow_id)
  );

  return {
    root_workflow_id: response.root_workflow_id,
    session_id: response.session_id,
    actor_id: response.actor_id,
    total_nodes: response.total_nodes,
    displayed_nodes: timeline.length,
    max_depth: response.max_depth,
    dag: timeline.length > 0 ? { ...timeline[0] } : undefined,
    timeline,
    workflow_status: response.workflow_status,
    workflow_name: response.workflow_name,
    mode: "lightweight",
  };
}

export function applySimpleGridLayout(
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

export function decorateNodesWithViewMode(nodes: Node[], viewMode: string): Node[] {
  return nodes.map((node) => ({
    ...node,
    data: {
      ...(node.data as object),
      viewMode,
    },
  }));
}

export function decorateEdgesWithStatus(
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

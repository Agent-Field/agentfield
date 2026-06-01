/// <reference lib="webworker" />

import type { Node, Edge } from '@xyflow/react';
import { LayoutManager, type AllLayoutType } from './LayoutManager';

interface WorkerRequest {
  id: string;
  nodes: Node[];
  edges: Edge[];
  layoutType: AllLayoutType;
}

type WorkerResponse =
  | { id: string; type: 'progress'; value: number }
  | { id: string; type: 'result'; nodes: Node[]; edges: Edge[] }
  | { id: string; type: 'error'; message: string };

const layoutManager = new LayoutManager();

const postMessageTyped = (message: WorkerResponse) => {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message);
};

(self as unknown as DedicatedWorkerGlobalScope).onmessage = async (
  event: MessageEvent<WorkerRequest>,
) => {
  const { id, nodes, edges, layoutType } = event.data;

  const reportProgress = (value: number) => {
    postMessageTyped({ id, type: 'progress', value });
  };

  try {
    const result = await layoutManager.applyLayout(nodes, edges, layoutType, reportProgress);
    postMessageTyped({
      id,
      type: 'result',
      nodes: result.nodes,
      edges: result.edges,
    });
  } catch (error) {
    postMessageTyped({
      id,
      type: 'error',
      message: (error as Error)?.message || 'Layout computation failed',
    });
  }
};

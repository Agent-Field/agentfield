import dagre from 'dagre';
import type { Node, Edge } from '@xyflow/react';
import { ELKLayoutEngine, type ELKLayoutType } from './ELKLayoutEngine';

export type DagreLayoutType = 'tree' | 'flow';
export type AllLayoutType = DagreLayoutType | ELKLayoutType;

interface LayoutWorkerRequestMessage {
  id: string;
  nodes: Node[];
  edges: Edge[];
  layoutType: AllLayoutType;
}

type LayoutWorkerResponseMessage =
  | { id: string; type: 'progress'; value: number }
  | { id: string; type: 'result'; nodes: Node[]; edges: Edge[] }
  | { id: string; type: 'error'; message: string };

export interface LayoutManagerConfig {
  smallGraphThreshold: number; // Threshold for switching to ELK layouts
  performanceThreshold: number; // Threshold for virtualized rendering
  enableWorker?: boolean;
}

const DEFAULT_CONFIG: LayoutManagerConfig = {
  smallGraphThreshold: 50,
  performanceThreshold: 300,
  enableWorker: false,
};

export class LayoutManager {
  private elkEngine: ELKLayoutEngine;
  private config: LayoutManagerConfig;
  private layoutWorker?: Worker;
  private pendingWorkerRequests = new Map<
    string,
    {
      resolve: (value: { nodes: Node[]; edges: Edge[] }) => void;
      reject: (error: Error) => void;
      onProgress?: (progress: number) => void;
    }
  >();
  private workerRequestCounter = 0;
  private workerEnabled: boolean;

  constructor(config: Partial<LayoutManagerConfig> = {}) {
    this.elkEngine = new ELKLayoutEngine();
    this.config = { ...DEFAULT_CONFIG, ...config };

    const shouldEnableWorker =
      this.config.enableWorker === true &&
      typeof window !== 'undefined' &&
      typeof Worker !== 'undefined' &&
      typeof URL !== 'undefined';

    this.workerEnabled = shouldEnableWorker;

    if (this.workerEnabled) {
      this.initializeWorker();
    }
  }

  private initializeWorker(): void {
    try {
      this.layoutWorker = new Worker(new URL('./layoutWorker.ts', import.meta.url), {
        type: 'module',
      });

      this.layoutWorker.onmessage = (event: MessageEvent<LayoutWorkerResponseMessage>) =>
        this.handleWorkerMessage(event);
      this.layoutWorker.onerror = (event) => {
        console.error('Layout worker error:', event.message);
        this.rejectPendingWorkerRequests(
          new Error(`layout worker error: ${event.message ?? 'unknown error'}`),
        );
        this.disposeWorker();
      };
    } catch (error) {
      console.warn('Failed to initialize layout worker, falling back to main thread:', error);
      this.layoutWorker = undefined;
      this.workerEnabled = false;
    }
  }

  private handleWorkerMessage(event: MessageEvent<LayoutWorkerResponseMessage>): void {
    const message = event.data;
    const pending = this.pendingWorkerRequests.get(message.id);
    if (!pending) {
      return;
    }

    if (message.type === 'progress') {
      pending.onProgress?.(message.value);
      return;
    }

    this.pendingWorkerRequests.delete(message.id);

    if (message.type === 'result') {
      pending.onProgress?.(100);
      pending.resolve({ nodes: message.nodes, edges: message.edges });
    } else if (message.type === 'error') {
      pending.reject(new Error(message.message));
      this.disposeWorker();
    }
  }

  private rejectPendingWorkerRequests(error: Error): void {
    this.pendingWorkerRequests.forEach(({ reject }) => reject(error));
    this.pendingWorkerRequests.clear();
  }

  private disposeWorker(): void {
    if (this.layoutWorker) {
      this.layoutWorker.terminate();
      this.layoutWorker = undefined;
    }
    this.pendingWorkerRequests.clear();
    this.workerEnabled = false;
  }

  private applyLayoutWithWorker(
    nodes: Node[],
    edges: Edge[],
    layoutType: AllLayoutType,
    onProgress?: (progress: number) => void,
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    if (!this.layoutWorker) {
      return this.applyLayoutMainThread(nodes, edges, layoutType, onProgress);
    }

    const requestId = `layout-${++this.workerRequestCounter}`;

    return new Promise((resolve, reject) => {
      this.pendingWorkerRequests.set(requestId, { resolve, reject, onProgress });
      try {
        onProgress?.(0);
        this.layoutWorker!.postMessage({
          id: requestId,
          nodes,
          edges,
          layoutType,
        } as LayoutWorkerRequestMessage);
      } catch (error) {
        this.pendingWorkerRequests.delete(requestId);
        console.warn('Failed to post layout job to worker, falling back to main thread:', error);
        this.applyLayoutMainThread(nodes, edges, layoutType, onProgress)
          .then(resolve)
          .catch(reject);
      }
    });
  }

  /**
   * Determine if graph should use ELK layouts based on size
   */
  isLargeGraph(nodeCount: number): boolean {
    return nodeCount >= this.config.smallGraphThreshold;
  }

  /**
   * Get available layout types — unified order for all graph sizes
   */
  getAvailableLayouts(_nodeCount: number): AllLayoutType[] {
    return ['tree', 'flow', 'layered', 'box', 'rectpacking'];
  }

  getDefaultLayout(_nodeCount: number): AllLayoutType {
    return 'tree';
  }

  /**
   * Check if a layout type is slow for large graphs
   */
  isSlowLayout(layoutType: AllLayoutType): boolean {
    if (layoutType === 'tree' || layoutType === 'flow') {
      return false; // Dagre layouts are generally fast
    }
    return ELKLayoutEngine.isSlowForLargeGraphs(layoutType as ELKLayoutType);
  }

  /**
   * Get layout description
   */
  getLayoutDescription(layoutType: AllLayoutType): string {
    switch (layoutType) {
      case 'tree':
        return 'Tree layout - Top to bottom hierarchy';
      case 'flow':
        return 'Flow layout - Left to right flow';
      default:
        return ELKLayoutEngine.getLayoutDescription(layoutType as ELKLayoutType);
    }
  }

  /**
   * Apply layout to nodes and edges
   */
  async applyLayout(
    nodes: Node[],
    edges: Edge[],
    layoutType: AllLayoutType,
    onProgress?: (progress: number) => void
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    if (this.layoutWorker) {
      try {
        return await this.applyLayoutWithWorker(nodes, edges, layoutType, onProgress);
      } catch (error) {
        console.warn('Layout worker failed, falling back to main thread:', error);
        this.disposeWorker();
      }
    }

    return this.applyLayoutMainThread(nodes, edges, layoutType, onProgress);
  }

  private async applyLayoutMainThread(
    nodes: Node[],
    edges: Edge[],
    layoutType: AllLayoutType,
    onProgress?: (progress: number) => void,
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    onProgress?.(0);

    try {
      if (layoutType === 'tree') {
        const result = this.applyWrappedTreeLayout(nodes, edges);
        onProgress?.(100);
        return result;
      } else if (layoutType === 'flow') {
        const result = this.applyDagreLayout(nodes, edges, 'flow');
        onProgress?.(100);
        return result;
      } else {
        onProgress?.(25);
        const result = await this.elkEngine.applyLayout(nodes, edges, layoutType as ELKLayoutType);
        onProgress?.(100);
        return result;
      }
    } catch (error) {
      console.error('Layout application failed:', error);
      onProgress?.(100);
      return { nodes, edges };
    }
  }

  private static readonly WRAP_THRESHOLD = 6;
  private static readonly LEVEL_GAP = 160;
  private static readonly NODE_GAP_X = 40;
  private static readonly ROW_GAP = 30;
  private static readonly MARGIN = 60;

  private applyWrappedTreeLayout(
    nodes: Node[],
    edges: Edge[],
  ): { nodes: Node[]; edges: Edge[] } {
    if (nodes.length === 0) return { nodes, edges };

    const dimMap = new Map<string, { width: number; height: number }>();
    for (const node of nodes) {
      dimMap.set(node.id, this.calculateNodeDimensions(node.data));
    }

    const parentOf = new Map<string, string>();
    for (const edge of edges) {
      parentOf.set(edge.target, edge.source);
    }

    const nodeIds = new Set(nodes.map((n) => n.id));
    const roots = nodes.filter((n) => !parentOf.has(n.id) || !nodeIds.has(parentOf.get(n.id)!));

    const depthOf = new Map<string, number>();
    const queue: Array<{ id: string; depth: number }> = roots.map((n) => ({
      id: n.id,
      depth: 0,
    }));

    const childrenOf = new Map<string, string[]>();
    for (const edge of edges) {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
      const list = childrenOf.get(edge.source) ?? [];
      list.push(edge.target);
      childrenOf.set(edge.source, list);
    }

    while (queue.length > 0) {
      const item = queue.shift()!;
      if (depthOf.has(item.id)) continue;
      depthOf.set(item.id, item.depth);
      for (const childId of childrenOf.get(item.id) ?? []) {
        if (!depthOf.has(childId)) {
          queue.push({ id: childId, depth: item.depth + 1 });
        }
      }
    }

    for (const node of nodes) {
      if (!depthOf.has(node.id)) depthOf.set(node.id, 0);
    }

    const levels = new Map<number, Node[]>();
    for (const node of nodes) {
      const d = depthOf.get(node.id) ?? 0;
      const list = levels.get(d) ?? [];
      list.push(node);
      levels.set(d, list);
    }

    for (const [, nodesAtLevel] of levels) {
      nodesAtLevel.sort((a, b) => {
        const aTime = (a.data as any)?.started_at ?? '';
        const bTime = (b.data as any)?.started_at ?? '';
        if (aTime !== bTime) return aTime < bTime ? -1 : 1;
        return a.id.localeCompare(b.id);
      });
    }

    const maxDepth = Math.max(0, ...levels.keys());
    const avgNodeWidth =
      nodes.reduce((s, n) => s + (dimMap.get(n.id)?.width ?? 240), 0) / nodes.length;
    const nodeHeight = 100;

    const maxColumnsPerRow = Math.max(
      LayoutManager.WRAP_THRESHOLD,
      Math.floor(1600 / (avgNodeWidth + LayoutManager.NODE_GAP_X)),
    );

    interface LevelMeta {
      cols: number;
      rows: number;
      totalHeight: number;
      levelWidth: number;
    }
    const levelMeta = new Map<number, LevelMeta>();
    for (let d = 0; d <= maxDepth; d++) {
      const nodesAtLevel = levels.get(d) ?? [];
      const count = nodesAtLevel.length;
      const cols = Math.min(count, maxColumnsPerRow);
      const rowCount = Math.ceil(count / cols);
      const totalHeight =
        rowCount * nodeHeight + (rowCount - 1) * LayoutManager.ROW_GAP;
      const levelWidth =
        cols * avgNodeWidth + (cols - 1) * LayoutManager.NODE_GAP_X;
      levelMeta.set(d, { cols, rows: rowCount, totalHeight, levelWidth });
    }

    const maxLevelWidth = Math.max(
      0,
      ...[...levelMeta.values()].map((m) => m.levelWidth),
    );
    const centerX = LayoutManager.MARGIN + maxLevelWidth / 2;

    const levelY = new Map<number, number>();
    let currentY = LayoutManager.MARGIN;
    for (let d = 0; d <= maxDepth; d++) {
      levelY.set(d, currentY);
      const meta = levelMeta.get(d)!;
      currentY += meta.totalHeight + LayoutManager.LEVEL_GAP;
    }

    const positionMap = new Map<string, { x: number; y: number }>();
    for (let d = 0; d <= maxDepth; d++) {
      const nodesAtLevel = levels.get(d) ?? [];
      const meta = levelMeta.get(d)!;
      const baseY = levelY.get(d)!;
      const startX = centerX - meta.levelWidth / 2;

      for (let i = 0; i < nodesAtLevel.length; i++) {
        const node = nodesAtLevel[i];
        const col = i % meta.cols;
        const row = Math.floor(i / meta.cols);
        const dim = dimMap.get(node.id) ?? { width: 240, height: 100 };

        const cellCenterX =
          startX + col * (avgNodeWidth + LayoutManager.NODE_GAP_X) + avgNodeWidth / 2;

        positionMap.set(node.id, {
          x: cellCenterX - dim.width / 2,
          y: baseY + row * (nodeHeight + LayoutManager.ROW_GAP),
        });
      }
    }

    const layoutedNodes = nodes.map((node) => ({
      ...node,
      position: positionMap.get(node.id) ?? { x: 0, y: 0 },
    }));

    return { nodes: layoutedNodes, edges };
  }

  private applyDagreLayout(nodes: Node[], edges: Edge[], layoutType: DagreLayoutType): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));

    const nodeDimensions = nodes.map(node => this.calculateNodeDimensions(node.data));
    const avgWidth = nodeDimensions.reduce((sum, dim) => sum + dim.width, 0) / nodeDimensions.length;
    const maxWidth = Math.max(...nodeDimensions.map(dim => dim.width));

    const direction = layoutType === 'tree' ? 'TB' : 'LR';
    const spacing = direction === 'TB'
      ? { rankSep: 140, nodeSep: Math.max(100, avgWidth * 0.4) }
      : { rankSep: Math.max(280, maxWidth * 1.2), nodeSep: 120 };

    g.setGraph({
      rankdir: direction,
      ranksep: spacing.rankSep,
      nodesep: spacing.nodeSep,
      marginx: 60,
      marginy: 60,
    });

    nodes.forEach((node, index) => {
      const dimensions = nodeDimensions[index];
      g.setNode(node.id, {
        width: dimensions.width,
        height: dimensions.height,
      });
    });

    edges.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    dagre.layout(g);

    const layoutedNodes = nodes.map((node, index) => {
      const nodeWithPosition = g.node(node.id);
      const dimensions = nodeDimensions[index];
      return {
        ...node,
        position: {
          x: nodeWithPosition.x - dimensions.width / 2,
          y: nodeWithPosition.y - dimensions.height / 2,
        },
      };
    });

    return { nodes: layoutedNodes, edges };
  }

  /**
   * Calculate node dimensions (same logic as in original component)
   */
  private calculateNodeDimensions(nodeData: any): { width: number; height: number } {
    const taskText = nodeData.task_name || nodeData.reasoner_id || '';
    const agentText = nodeData.agent_name || nodeData.agent_node_id || '';

    const minWidth = 200;
    const maxWidth = 360;
    const charWidth = 7.5;

    const humanizeText = (text: string): string => {
      return text
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
        .replace(/\s+/g, ' ')
        .trim();
    };

    const taskHuman = humanizeText(taskText);
    const agentHuman = humanizeText(agentText);

    const taskWordsLength = taskHuman.split(' ').reduce((max, word) => Math.max(max, word.length), 0);
    const agentWordsLength = agentHuman.split(' ').reduce((max, word) => Math.max(max, word.length), 0);

    const longestWord = Math.max(taskWordsLength, agentWordsLength);
    const estimatedWidth = Math.max(
      longestWord * charWidth * 1.8,
      (taskHuman.length / 2.2) * charWidth,
      (agentHuman.length / 2.2) * charWidth
    ) + 80;

    const width = Math.min(maxWidth, Math.max(minWidth, estimatedWidth));
    const height = 100; // Fixed height as set in WorkflowNode

    return { width, height };
  }

  /**
   * Get configuration
   */
  getConfig(): LayoutManagerConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<LayoutManagerConfig>): void {
    const merged = { ...this.config, ...newConfig };
    const workerStateChanged = merged.enableWorker !== this.config.enableWorker;
    this.config = merged;

    if (workerStateChanged) {
      if (this.config.enableWorker && !this.workerEnabled) {
        this.workerEnabled =
          typeof window !== 'undefined' &&
          typeof Worker !== 'undefined' &&
          typeof URL !== 'undefined';
        if (this.workerEnabled) {
          this.initializeWorker();
        }
      } else if (!this.config.enableWorker && this.workerEnabled) {
        this.disposeWorker();
      }
    }
  }
}

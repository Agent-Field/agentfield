import ELK, { type ElkNode, type LayoutOptions, type ElkExtendedEdge } from 'elkjs/lib/elk.bundled.js';
import type { Node, Edge } from '@xyflow/react';

export type ELKLayoutType = 'box' | 'rectpacking' | 'layered';

export interface ELKLayoutConfig {
  algorithm: string;
  options: LayoutOptions;
  isSlowForLargeGraphs: boolean;
  description: string;
}

// ELK layout configurations
export const ELK_LAYOUTS: Record<ELKLayoutType, ELKLayoutConfig> = {
  box: {
    algorithm: 'org.eclipse.elk.box',
    options: {
      'elk.algorithm': 'org.eclipse.elk.box',
      'elk.spacing.nodeNode': '80',
      'elk.spacing.edgeNode': '40',
      'elk.padding': '[top=50,left=50,bottom=50,right=50]',
      'elk.box.packingMode': 'SIMPLE',
    },
    isSlowForLargeGraphs: false,
    description: 'Box layout - Fast for large graphs'
  },
  rectpacking: {
    algorithm: 'org.eclipse.elk.rectpacking',
    options: {
      'elk.algorithm': 'org.eclipse.elk.rectpacking',
      'elk.spacing.nodeNode': '60',
      'elk.padding': '[top=50,left=50,bottom=50,right=50]',
      'elk.rectpacking.packing.strategy': 'SIMPLE',
      'elk.rectpacking.packing.compaction.rowHeightReevaluation': 'true',
    },
    isSlowForLargeGraphs: false,
    description: 'Rectangle packing - Fast for large graphs'
  },
  layered: {
    algorithm: 'org.eclipse.elk.layered',
    options: {
      'elk.algorithm': 'org.eclipse.elk.layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '80',
      'elk.spacing.edgeNode': '40',
      'elk.layered.spacing.nodeNodeBetweenLayers': '100',
      'elk.layered.spacing.edgeNodeBetweenLayers': '50',
      'elk.padding': '[top=50,left=50,bottom=50,right=50]',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.nodePlacement.strategy': 'SIMPLE',
    },
    isSlowForLargeGraphs: true,
    description: 'Layered layout - Hierarchical, slower for large graphs'
  },
};

export class ELKLayoutEngine {
  private elk: InstanceType<typeof ELK>;

  constructor() {
    this.elk = new ELK();
  }

  /**
   * Apply ELK layout to nodes and edges
   */
  async applyLayout(
    nodes: Node[],
    edges: Edge[],
    layoutType: ELKLayoutType
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    const config = ELK_LAYOUTS[layoutType];

    // Convert ReactFlow nodes to ELK format
    const elkNodes: ElkNode[] = nodes.map((node) => ({
      id: node.id,
      width: node.width || this.calculateNodeWidth(node.data),
      height: node.height || this.calculateNodeHeight(node.data),
      // Add any additional node properties if needed
    }));

    // Convert ReactFlow edges to ELK format
    const elkEdges: ElkExtendedEdge[] = edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    }));

    // Create ELK graph
    const elkGraph: ElkNode = {
      id: 'root',
      children: elkNodes,
      edges: elkEdges,
      layoutOptions: config.options,
    };

    try {
      // Apply layout
      const layoutedGraph = await this.elk.layout(elkGraph);

      // Convert back to ReactFlow format
      const layoutedNodes = nodes.map((node) => {
        const elkNode = layoutedGraph.children?.find((n: ElkNode) => n.id === node.id);
        if (elkNode) {
          return {
            ...node,
            position: {
              x: elkNode.x || 0,
              y: elkNode.y || 0,
            },
          };
        }
        return node;
      });

      return {
        nodes: layoutedNodes,
        edges, // Edges don't need position updates
      };
    } catch (error) {
      console.error('ELK layout failed:', error);
      // Return original nodes/edges on failure
      return { nodes, edges };
    }
  }

  /**
   * Calculate node width based on content
   */
  private calculateNodeWidth(nodeData: any): number {
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

    return Math.min(maxWidth, Math.max(minWidth, estimatedWidth));
  }

  /**
   * Calculate node height (fixed for workflow nodes)
   */
  private calculateNodeHeight(_nodeData: any): number {
    return 100; // Fixed height as used in WorkflowNode
  }

  /**
   * Check if a layout type is slow for large graphs
   */
  static isSlowForLargeGraphs(layoutType: ELKLayoutType): boolean {
    return ELK_LAYOUTS[layoutType].isSlowForLargeGraphs;
  }

  /**
   * Get layout description
   */
  static getLayoutDescription(layoutType: ELKLayoutType): string {
    return ELK_LAYOUTS[layoutType].description;
  }

  /**
   * Get all available layout types
   */
  static getAvailableLayouts(): ELKLayoutType[] {
    return Object.keys(ELK_LAYOUTS) as ELKLayoutType[];
  }
}

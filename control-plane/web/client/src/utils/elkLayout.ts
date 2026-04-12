import ELK from 'elkjs/lib/elk.bundled.js';
import type { Node, Edge } from '@xyflow/react';
import type {
  ElkGraph,
  ElkNode,
  ElkEdge,
  ElkLayoutOptions,
  ElkAlgorithm
} from '../types/elk';
import { ELK_ALGORITHMS } from '../types/elk';

// Initialize ELK instance
const elk = new ELK();

// Default layout options for different algorithms
const getDefaultLayoutOptions = (algorithm: ElkAlgorithm): ElkLayoutOptions => {
  const baseOptions: ElkLayoutOptions = {
    'elk.algorithm': algorithm,
    'elk.padding': '[top=50,left=50,bottom=50,right=50]',
    'elk.spacing.nodeNode': 80,
    'elk.spacing.edgeNode': 40,
    'elk.spacing.edgeEdge': 20,
  };

  switch (algorithm) {
    case ELK_ALGORITHMS.BOX:
      return {
        ...baseOptions,
        'elk.box.packingMode': 'SIMPLE',
        'elk.spacing.nodeNode': 100,
        'elk.aspectRatio': 1.6,
      };

    case ELK_ALGORITHMS.LAYERED:
      return {
        ...baseOptions,
        'elk.direction': 'DOWN',
        'elk.layered.spacing.nodeNodeBetweenLayers': 120,
        'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
        'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
        'elk.edgeRouting': 'ORTHOGONAL',
      };

    case ELK_ALGORITHMS.FORCE:
      return {
        ...baseOptions,
        'elk.force.repulsivePower': 0.5,
        'elk.force.iterations': 300,
        'elk.spacing.nodeNode': 150,
      };

    case ELK_ALGORITHMS.RADIAL:
      return {
        ...baseOptions,
        'elk.radial.radius': 200,
        'elk.spacing.nodeNode': 120,
      };

    case ELK_ALGORITHMS.MR_TREE:
      return {
        ...baseOptions,
        'elk.direction': 'DOWN',
        'elk.spacing.nodeNode': 100,
        'elk.layered.spacing.nodeNodeBetweenLayers': 100,
      };

    case ELK_ALGORITHMS.STRESS:
      return {
        ...baseOptions,
        'elk.stress.iterations': 300,
        'elk.spacing.nodeNode': 120,
      };

    case ELK_ALGORITHMS.RECT_PACKING:
      return {
        ...baseOptions,
        'elk.spacing.nodeNode': 20,
        'elk.aspectRatio': 1.6,
      };

    case ELK_ALGORITHMS.TOPDOWN_PACKING:
      return {
        ...baseOptions,
        'elk.spacing.nodeNode': 30,
        'elk.direction': 'DOWN',
      };

    default:
      return baseOptions;
  }
};

// Convert ReactFlow nodes to ELK format
const convertNodesToElk = (nodes: Node[]): ElkNode[] => {
  return nodes.map((node) => {
    // Calculate node dimensions (same logic as in the original DAG)
    const nodeData = node.data as any;
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
    const height = 100;

    return {
      id: node.id,
      width,
      height,
    };
  });
};

// Convert ReactFlow edges to ELK format
const convertEdgesToElk = (edges: Edge[]): ElkEdge[] => {
  return edges.map((edge) => ({
    id: edge.id,
    sources: [edge.source],
    targets: [edge.target],
  }));
};

// Convert ELK layout result back to ReactFlow format
const convertElkToReactFlow = (
  elkGraph: ElkGraph,
  originalNodes: Node[],
  originalEdges: Edge[]
): { nodes: Node[]; edges: Edge[] } => {
  const nodeMap = new Map(originalNodes.map(node => [node.id, node]));

  const layoutedNodes: Node[] = elkGraph.children?.map((elkNode) => {
    const originalNode = nodeMap.get(elkNode.id);
    if (!originalNode) {
      throw new Error(`Node ${elkNode.id} not found in original nodes`);
    }

    return {
      ...originalNode,
      position: {
        x: elkNode.x || 0,
        y: elkNode.y || 0,
      },
    };
  }) || [];

  // Edges don't need position updates, just return originals
  return {
    nodes: layoutedNodes,
    edges: originalEdges,
  };
};

// Main layout function
export const applyElkLayout = async (
  nodes: Node[],
  edges: Edge[],
  algorithm: ElkAlgorithm = ELK_ALGORITHMS.BOX,
  customOptions?: Partial<ElkLayoutOptions>
): Promise<{ nodes: Node[]; edges: Edge[] }> => {
  if (nodes.length === 0) {
    return { nodes, edges };
  }

  try {
    // Get default options for the algorithm and merge with custom options
    const layoutOptions = {
      ...getDefaultLayoutOptions(algorithm),
      ...customOptions,
    };

    // Convert to ELK format
    const elkNodes = convertNodesToElk(nodes);
    const elkEdges = convertEdgesToElk(edges);

    // Create ELK graph
    const elkGraph: ElkGraph = {
      id: 'root',
      layoutOptions,
      children: elkNodes,
      edges: elkEdges,
    };

    // Apply layout
    const layoutedGraph = await elk.layout(elkGraph);

    // Convert back to ReactFlow format
    const result = convertElkToReactFlow(layoutedGraph, nodes, edges);

    return result;
  } catch (error) {
    console.error('ELK Layout Error:', error);
    // Fallback to original positions if layout fails
    return { nodes, edges };
  }
};

// Utility function to get available algorithms for dropdown
export const getAvailableAlgorithms = () => {
  return Object.entries(ELK_ALGORITHMS).map(([key, value]) => ({
    id: value,
    name: key.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase()),
    algorithm: value,
  }));
};

// Utility function to check if an algorithm is suitable for large graphs
export const isAlgorithmSuitableForLargeGraphs = (algorithm: ElkAlgorithm): boolean => {
  const largeGraphAlgorithms: ElkAlgorithm[] = [
    ELK_ALGORITHMS.BOX,
    ELK_ALGORITHMS.FORCE,
    ELK_ALGORITHMS.RECT_PACKING,
    ELK_ALGORITHMS.TOPDOWN_PACKING,
  ];

  return largeGraphAlgorithms.includes(algorithm);
};

// Utility function to get recommended algorithm based on graph size and structure
export const getRecommendedAlgorithm = (nodeCount: number, edgeCount: number): ElkAlgorithm => {
  if (nodeCount > 500) {
    return ELK_ALGORITHMS.BOX; // Best for very large graphs
  } else if (nodeCount > 100) {
    return ELK_ALGORITHMS.RECT_PACKING; // Good for large graphs
  } else if (edgeCount / nodeCount > 2) {
    return ELK_ALGORITHMS.FORCE; // Good for dense graphs
  } else {
    return ELK_ALGORITHMS.LAYERED; // Good for hierarchical graphs
  }
};

// ELK Layout Types
export interface ElkNode {
  id: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  children?: ElkNode[];
  layoutOptions?: Record<string, any>;
}

export interface ElkEdge {
  id: string;
  sources: string[];
  targets: string[];
  sections?: ElkEdgeSection[];
}

export interface ElkEdgeSection {
  id: string;
  startPoint: ElkPoint;
  endPoint: ElkPoint;
  bendPoints?: ElkPoint[];
}

export interface ElkPoint {
  x: number;
  y: number;
}

export interface ElkGraph {
  id: string;
  children?: ElkNode[];
  edges?: ElkEdge[];
  layoutOptions?: Record<string, any>;
}

export interface ElkLayoutOptions {
  'elk.algorithm'?: string;
  'elk.direction'?: 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
  'elk.spacing.nodeNode'?: number;
  'elk.layered.spacing.nodeNodeBetweenLayers'?: number;
  'elk.spacing.edgeNode'?: number;
  'elk.spacing.edgeEdge'?: number;
  'elk.padding'?: string;
  'elk.aspectRatio'?: number;
  'elk.nodeLabels.placement'?: string;
  'elk.portConstraints'?: string;
  'elk.edgeRouting'?: string;
  'elk.layered.crossingMinimization.strategy'?: string;
  'elk.layered.nodePlacement.strategy'?: string;
  'elk.box.packingMode'?: string;
  'elk.force.repulsivePower'?: number;
  'elk.force.iterations'?: number;
  'elk.radial.radius'?: number;
  'elk.stress.iterations'?: number;
  [key: string]: any;
}

// ELK Algorithm identifiers
export const ELK_ALGORITHMS = {
  BOX: 'org.eclipse.elk.box',
  DISCO: 'org.eclipse.elk.disco',
  FIXED: 'org.eclipse.elk.fixed',
  FORCE: 'org.eclipse.elk.force',
  LAYERED: 'org.eclipse.elk.layered',
  MR_TREE: 'org.eclipse.elk.mrtree',
  RADIAL: 'org.eclipse.elk.radial',
  RANDOM: 'org.eclipse.elk.random',
  RECT_PACKING: 'org.eclipse.elk.rectpacking',
  SPORE_COMPACTION: 'org.eclipse.elk.sporeCompaction',
  SPORE_OVERLAP: 'org.eclipse.elk.sporeOverlap',
  STRESS: 'org.eclipse.elk.stress',
  TOPDOWN_PACKING: 'org.eclipse.elk.topdownpacking',
} as const;

export type ElkAlgorithm = typeof ELK_ALGORITHMS[keyof typeof ELK_ALGORITHMS];

// Algorithm display names and descriptions
export const ELK_ALGORITHM_INFO = {
  [ELK_ALGORITHMS.BOX]: {
    name: 'Box Layout',
    description: 'Arranges nodes in a box-like structure, good for large graphs',
    category: 'General Purpose'
  },
  [ELK_ALGORITHMS.LAYERED]: {
    name: 'Layered Layout',
    description: 'Hierarchical layout with nodes arranged in layers',
    category: 'Hierarchical'
  },
  [ELK_ALGORITHMS.FORCE]: {
    name: 'Force-Directed',
    description: 'Physics-based layout using force simulation',
    category: 'Force-Based'
  },
  [ELK_ALGORITHMS.RADIAL]: {
    name: 'Radial Layout',
    description: 'Arranges nodes in concentric circles',
    category: 'Radial'
  },
  [ELK_ALGORITHMS.MR_TREE]: {
    name: 'Tree Layout',
    description: 'Tree-like arrangement for hierarchical data',
    category: 'Tree'
  },
  [ELK_ALGORITHMS.STRESS]: {
    name: 'Stress Layout',
    description: 'Minimizes stress between connected nodes',
    category: 'Force-Based'
  },
  [ELK_ALGORITHMS.RECT_PACKING]: {
    name: 'Rectangle Packing',
    description: 'Efficiently packs rectangular nodes',
    category: 'Packing'
  },
  [ELK_ALGORITHMS.TOPDOWN_PACKING]: {
    name: 'Top-Down Packing',
    description: 'Hierarchical packing from top to bottom',
    category: 'Packing'
  },
} as const;

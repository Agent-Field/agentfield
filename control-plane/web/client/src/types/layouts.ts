import { ELK_ALGORITHMS } from './elk';
import type { ElkAlgorithm } from './elk';

// Unified layout configuration
export interface LayoutConfig {
  id: string;
  name: string;
  description: string;
  category: string;
  engine: 'dagre' | 'elk';
  dagreType?: 'tree' | 'flow';
  elkAlgorithm?: ElkAlgorithm;
  suitableForLargeGraphs: boolean;
}

// Simplified, fast layouts only
export const AVAILABLE_LAYOUTS: LayoutConfig[] = [
  // Dagre layouts (fast for small graphs)
  {
    id: 'tree',
    name: 'Tree Layout',
    description: 'Fast hierarchical top-to-bottom arrangement',
    category: 'Small Graph',
    engine: 'dagre',
    dagreType: 'tree',
    suitableForLargeGraphs: false,
  },
  {
    id: 'flow',
    name: 'Flow Layout',
    description: 'Fast left-to-right flow arrangement',
    category: 'Small Graph',
    engine: 'dagre',
    dagreType: 'flow',
    suitableForLargeGraphs: false,
  },

  // ELK Box layout only (optimized for large graphs)
  {
    id: 'box',
    name: 'Box Layout',
    description: 'Optimized for large graphs (>100 nodes)',
    category: 'Large Graph',
    engine: 'elk',
    elkAlgorithm: ELK_ALGORITHMS.BOX,
    suitableForLargeGraphs: true,
  },
];

// Helper functions
export const getLayoutById = (id: string): LayoutConfig | undefined => {
  return AVAILABLE_LAYOUTS.find(layout => layout.id === id);
};

export const getLayoutsByCategory = (category: string): LayoutConfig[] => {
  return AVAILABLE_LAYOUTS.filter(layout => layout.category === category);
};

export const getRecommendedLayout = (nodeCount: number): LayoutConfig => {
  if (nodeCount > 500) {
    return getLayoutById('box')!;
  } else if (nodeCount > 100) {
    return getLayoutById('rect-packing')!;
  } else {
    return getLayoutById('tree')!;
  }
};

export const getLayoutsForLargeGraphs = (): LayoutConfig[] => {
  return AVAILABLE_LAYOUTS.filter(layout => layout.suitableForLargeGraphs);
};

// Convert LayoutConfig to the internal LayoutType
export const layoutConfigToLayoutType = (config: LayoutConfig) => {
  return {
    engine: config.engine,
    dagreType: config.dagreType,
    elkAlgorithm: config.elkAlgorithm,
  };
};

import type { WorkflowDAGLightweightNode } from "../../types/workflows";

export type WorkflowDAGNode = WorkflowDAGLightweightNode & {
  workflow_id?: string;
  agent_name?: string;
  task_name?: string;
};

export interface DeckNode {
  id: string;
  position: [number, number, number];
  depth: number;
  radius: number;
  fillColor: [number, number, number, number];
  borderColor: [number, number, number, number];
  glowColor: [number, number, number, number];
  original: WorkflowDAGNode;
}

export interface DeckEdge {
  id: string;
  path: [number, number, number][];
  color: [number, number, number, number];
  width: number;
}

export interface AgentPaletteEntry {
  agentId: string;
  label: string;
  color: string;
  background: string;
  text: string;
}

export interface DeckGraphData {
  nodes: DeckNode[];
  edges: DeckEdge[];
  agentPalette: AgentPaletteEntry[];
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const sat = s / 100;
  const light = l / 100;

  if (sat === 0) {
    const val = Math.round(light * 255);
    return [val, val, val];
  }

  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  const q = light < 0.5 ? light * (1 + sat) : light + sat - light * sat;
  const p = 2 * light - q;
  const hk = h / 360;

  const r = Math.round(hue2rgb(p, q, hk + 1 / 3) * 255);
  const g = Math.round(hue2rgb(p, q, hk) * 255);
  const b = Math.round(hue2rgb(p, q, hk - 1 / 3) * 255);

  return [r, g, b];
}

const PROFESSIONAL_PALETTE = [
  { h: 210, s: 48, l: 62 },
  { h: 165, s: 45, l: 58 },
  { h: 280, s: 50, l: 64 },
  { h: 30, s: 52, l: 62 },
  { h: 340, s: 48, l: 64 },
  { h: 130, s: 42, l: 58 },
  { h: 45, s: 50, l: 60 },
  { h: 260, s: 46, l: 62 },
  { h: 190, s: 44, l: 60 },
  { h: 15, s: 48, l: 62 },
];

function getAgentColor(agentId: string, index: number): {
  rgb: [number, number, number];
  css: string;
} {
  const hash = hashString(agentId || `agent-${index}`);
  const paletteIndex = hash % PROFESSIONAL_PALETTE.length;
  const palette = PROFESSIONAL_PALETTE[paletteIndex];
  const hueVariation = (hash % 20) - 10;
  const hue = (palette.h + hueVariation + 360) % 360;
  const rgb = hslToRgb(hue, palette.s, palette.l);
  return { rgb, css: `rgb(${rgb.join(",")})` };
}

function mixColor(
  color: [number, number, number],
  target: [number, number, number],
  ratio: number
): [number, number, number] {
  return [
    Math.round(color[0] * ratio + target[0] * (1 - ratio)),
    Math.round(color[1] * ratio + target[1] * (1 - ratio)),
    Math.round(color[2] * ratio + target[2] * (1 - ratio)),
  ];
}

const BACKGROUND_RGB: [number, number, number] = [11, 18, 32];

const STATUS_COLORS: Record<string, [number, number, number]> = {
  succeeded: [34, 197, 94],
  failed: [239, 68, 68],
  running: [59, 130, 246],
  pending: [251, 191, 36],
  queued: [251, 191, 36],
  timeout: [148, 163, 184],
  cancelled: [148, 163, 184],
  unknown: [148, 163, 184],
};

function normalizeStatus(status: string): string {
  const normalized = status.toLowerCase();
  if (normalized.includes("success") || normalized.includes("complete")) return "succeeded";
  if (normalized.includes("fail") || normalized.includes("error")) return "failed";
  if (normalized.includes("run") || normalized.includes("progress")) return "running";
  if (normalized.includes("pend")) return "pending";
  if (normalized.includes("queue")) return "queued";
  if (normalized.includes("timeout")) return "timeout";
  if (normalized.includes("cancel")) return "cancelled";
  return normalized;
}

function getStatusColor(status: string): [number, number, number] {
  const normalized = normalizeStatus(status);
  return STATUS_COLORS[normalized] || STATUS_COLORS.unknown;
}

function createCubicBezier(
  source: [number, number, number],
  target: [number, number, number],
  curvature: number = 0.5
): [number, number, number][] {
  const dy = target[1] - source[1];
  const control1: [number, number, number] = [
    source[0],
    source[1] + dy * curvature,
    source[2],
  ];
  const control2: [number, number, number] = [
    target[0],
    target[1] - dy * curvature,
    target[2],
  ];

  const points: [number, number, number][] = [];
  const segments = 8;

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const mt = 1 - t;
    const x =
      mt * mt * mt * source[0] +
      3 * mt * mt * t * control1[0] +
      3 * mt * t * t * control2[0] +
      t * t * t * target[0];
    const y =
      mt * mt * mt * source[1] +
      3 * mt * mt * t * control1[1] +
      3 * mt * t * t * control2[1] +
      t * t * t * target[1];
    const z =
      mt * mt * mt * source[2] +
      3 * mt * mt * t * control1[2] +
      3 * mt * t * t * control2[2] +
      t * t * t * target[2];
    points.push([x, y, z]);
  }

  return points;
}

export function buildDeckGraph(
  timeline: WorkflowDAGNode[],
  horizontalSpacing: number = 120,
  verticalSpacing: number = 100
): DeckGraphData {
  if (!timeline.length) {
    return { nodes: [], edges: [], agentPalette: [] };
  }

  const nodeById = new Map<string, WorkflowDAGNode>();
  const childrenByParent = new Map<string, WorkflowDAGNode[]>();
  const parentsByChild = new Map<string, string[]>();
  const agentColors = new Map<string, { rgb: [number, number, number]; css: string }>();

  timeline.forEach((node, index) => {
    nodeById.set(node.execution_id, node);

    if (node.parent_execution_id) {
      if (!childrenByParent.has(node.parent_execution_id)) {
        childrenByParent.set(node.parent_execution_id, []);
      }
      childrenByParent.get(node.parent_execution_id)!.push(node);

      if (!parentsByChild.has(node.execution_id)) {
        parentsByChild.set(node.execution_id, []);
      }
      parentsByChild.get(node.execution_id)!.push(node.parent_execution_id);
    }

    const agentId = node.agent_node_id || `agent-${index}`;
    if (!agentColors.has(agentId)) {
      agentColors.set(agentId, getAgentColor(agentId, agentColors.size));
    }
  });

  if (process.env.NODE_ENV !== "production") {
    console.debug(
      "[DeckGL] Processing",
      timeline.length,
      "nodes with",
      childrenByParent.size,
      "parent nodes"
    );
  }

  const roots = timeline.filter(
    (node) => !node.parent_execution_id || !nodeById.has(node.parent_execution_id)
  );

  if (roots.length === 0 && timeline.length > 0) {
    const fallbackRoot = timeline.reduce((best, node) => {
      const depth = node.workflow_depth ?? Infinity;
      const bestDepth = best.workflow_depth ?? Infinity;
      return depth < bestDepth ? node : best;
    });
    roots.push(fallbackRoot);
  }

  if (process.env.NODE_ENV !== "production") {
    console.debug("[DeckGL] Found", roots.length, "root nodes");
  }

  const layers: WorkflowDAGNode[][] = [];
  const nodeToLayer = new Map<string, number>();
  const visited = new Set<string>();
  const queue: { node: WorkflowDAGNode; layer: number }[] = [];

  roots.forEach((root) => {
    queue.push({ node: root, layer: 0 });
  });

  while (queue.length > 0) {
    const next = queue.shift();
    if (!next) {
      break;
    }

    const { node, layer } = next;

    if (visited.has(node.execution_id)) {
      const currentLayer = nodeToLayer.get(node.execution_id)!;
      if (layer > currentLayer) {
        const oldLayerNodes = layers[currentLayer];
        const idx = oldLayerNodes.findIndex((entry) => entry.execution_id === node.execution_id);
        if (idx >= 0) oldLayerNodes.splice(idx, 1);

        nodeToLayer.set(node.execution_id, layer);
        if (!layers[layer]) layers[layer] = [];
        layers[layer].push(node);
      }
      continue;
    }

    visited.add(node.execution_id);
    nodeToLayer.set(node.execution_id, layer);

    if (!layers[layer]) {
      layers[layer] = [];
    }
    layers[layer].push(node);

    const children = childrenByParent.get(node.execution_id) ?? [];
    children.forEach((child) => {
      queue.push({ node: child, layer: layer + 1 });
    });
  }

  if (process.env.NODE_ENV !== "production") {
    console.debug(
      "[DeckGL] Created",
      layers.length,
      "layers, max layer size:",
      Math.max(...layers.map((layer) => layer.length))
    );
  }

  const layoutInfo = new Map<
    string,
    {
      position: [number, number, number];
      layer: number;
      color: [number, number, number];
      agentId: string;
      radius: number;
    }
  >();

  layers.forEach((layerNodes, layerIndex) => {
    const layerWidth = (layerNodes.length - 1) * horizontalSpacing;
    const startX = -layerWidth / 2;

    layerNodes.forEach((node, indexInLayer) => {
      const x = startX + indexInLayer * horizontalSpacing;
      const y = layerIndex * verticalSpacing;
      const z = 0;

      const agentId = node.agent_node_id || node.reasoner_id || "agent";
      const colorInfo = agentColors.get(agentId) ?? getAgentColor(agentId, agentColors.size + 1);
      const baseColor = colorInfo.rgb;
      const baseRadius = Math.max(6, 12 - layerIndex * 0.3);

      layoutInfo.set(node.execution_id, {
        position: [x, y, z],
        layer: layerIndex,
        color: baseColor,
        agentId,
        radius: baseRadius,
      });
    });
  });

  const deckNodes: DeckNode[] = [];
  const maxDepth = Math.max(...Array.from(layoutInfo.values()).map((info) => info.layer));

  layoutInfo.forEach((info, nodeId) => {
    const node = nodeById.get(nodeId)!;
    const statusColor = getStatusColor(node.status);
    const mixedColor = mixColor(info.color, statusColor, 0.7);
    const depthFactor = maxDepth > 0 ? info.layer / maxDepth : 0;
    const opacity = Math.round(240 - depthFactor * 40);
    const fill = [...mixedColor, opacity] as [number, number, number, number];
    const border = [...mixColor(info.color, BACKGROUND_RGB, 0.35), 255] as [
      number,
      number,
      number,
      number,
    ];
    const glow = [...mixColor(statusColor, [255, 255, 255], 0.25), 90] as [
      number,
      number,
      number,
      number,
    ];

    deckNodes.push({
      id: nodeId,
      position: info.position,
      depth: info.layer,
      radius: info.radius,
      fillColor: fill,
      borderColor: border,
      glowColor: glow,
      original: nodeById.get(nodeId)!,
    });
  });

  const deckEdges: DeckEdge[] = [];
  timeline.forEach((node) => {
    if (!node.parent_execution_id) {
      return;
    }

    const parentInfo = layoutInfo.get(node.parent_execution_id);
    const childInfo = layoutInfo.get(node.execution_id);
    if (!parentInfo || !childInfo) {
      return;
    }

    const source = parentInfo.position;
    const target = childInfo.position;
    const dy = Math.abs(target[1] - source[1]);
    const curvature = Math.min(0.6, 0.3 + dy / 1000);
    const path = createCubicBezier(source, target, curvature);
    const baseColor = parentInfo.color;
    const depthFactor = maxDepth > 0 ? childInfo.layer / maxDepth : 0;
    const edgeOpacity = Math.round(120 - depthFactor * 30);
    const edgeColor = [
      ...mixColor(baseColor, BACKGROUND_RGB, 0.55),
      edgeOpacity,
    ] as [number, number, number, number];
    const width = Math.max(1, 2.2 - childInfo.layer * 0.12);

    deckEdges.push({
      id: `${node.parent_execution_id}-${node.execution_id}`,
      path,
      color: edgeColor,
      width,
    });
  });

  const agentPalette: AgentPaletteEntry[] = [];
  agentColors.forEach((value, key) => {
    const background = mixColor(value.rgb, BACKGROUND_RGB, 0.85);
    const luminance =
      value.rgb[0] * 0.2126 + value.rgb[1] * 0.7152 + value.rgb[2] * 0.0722;
    const textColor = luminance > 140 ? "#0f172a" : "#f8fafc";
    agentPalette.push({
      agentId: key,
      label: key,
      color: value.css,
      background: `rgb(${background.join(",")})`,
      text: textColor,
    });
  });
  agentPalette.sort((a, b) => a.label.localeCompare(b.label));

  if (process.env.NODE_ENV !== "production") {
    console.debug(
      "[DeckGL] Built",
      deckNodes.length,
      "nodes and",
      deckEdges.length,
      "edges"
    );
  }

  return { nodes: deckNodes, edges: deckEdges, agentPalette };
}

import type {
  AgentNodeSummary,
  HealthStatus,
  LifecycleStatus,
} from "@/types/haxen";
import {
  getStatusTheme,
  type CanonicalStatus,
  type StatusTheme,
} from "./status";

type NodeStatusKind =
  | "ready"
  | "running"
  | "starting"
  | "degraded"
  | "offline"
  | "error"
  | "unknown";

interface NodeStatusConfig {
  canonical: CanonicalStatus;
  label: string;
  pulse?: boolean;
}

const NODE_STATUS_CONFIG: Record<NodeStatusKind, NodeStatusConfig> = {
  ready: { canonical: "running", label: "Ready" },
  running: { canonical: "running", label: "Running" },
  starting: { canonical: "pending", label: "Starting", pulse: true },
  degraded: { canonical: "pending", label: "Degraded", pulse: true },
  offline: { canonical: "failed", label: "Offline" },
  error: { canonical: "failed", label: "Error" },
  unknown: { canonical: "unknown", label: "Unknown" },
};

export interface NodeStatusPresentation {
  kind: NodeStatusKind;
  label: string;
  theme: StatusTheme;
  shouldPulse: boolean;
  canonical: CanonicalStatus;
}

const lifecycleToKind = (
  lifecycle?: LifecycleStatus | null,
  health?: HealthStatus | null
): NodeStatusKind => {
  if (health === "inactive" || lifecycle === "offline" || lifecycle === "stopped") {
    return "offline";
  }
  if (lifecycle === "error" || health === "degraded") {
    return "error";
  }
  if (lifecycle === "degraded") {
    return "degraded";
  }
  if (lifecycle === "starting" || health === "starting") {
    return "starting";
  }
  if (lifecycle === "ready" || lifecycle === "running") {
    return "ready";
  }
  return "unknown";
};

export const getNodeStatusPresentation = (
  lifecycle?: LifecycleStatus | null,
  health?: HealthStatus | null
): NodeStatusPresentation => {
  const kind = lifecycleToKind(lifecycle, health);
  const config = NODE_STATUS_CONFIG[kind] ?? NODE_STATUS_CONFIG.unknown;
  return {
    kind,
    label: config.label,
    canonical: config.canonical,
    theme: getStatusTheme(config.canonical),
    shouldPulse: Boolean(config.pulse),
  };
};

interface NodeStatusBuckets {
  total: number;
  online: number;
  offline: number;
  degraded: number;
  starting: number;
  ready: number;
}

export const summarizeNodeStatuses = (
  nodes: AgentNodeSummary[]
): NodeStatusBuckets => {
  return nodes.reduce<NodeStatusBuckets>(
    (acc, node) => {
      const { kind } = getNodeStatusPresentation(
        node.lifecycle_status,
        node.health_status
      );

      acc.total += 1;

      switch (kind) {
        case "ready":
        case "running":
          acc.online += 1;
          acc.ready += 1;
          break;
        case "starting":
          acc.online += 1;
          acc.starting += 1;
          break;
        case "degraded":
          acc.online += 1;
          acc.degraded += 1;
          break;
        case "error":
        case "offline":
          acc.offline += 1;
          break;
        default:
          acc.offline += 1;
      }

      return acc;
    },
    {
      total: 0,
      online: 0,
      offline: 0,
      degraded: 0,
      starting: 0,
      ready: 0,
    }
  );
};

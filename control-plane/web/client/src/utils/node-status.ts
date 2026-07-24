import type {
  AgentNodeSummary,
  HealthStatus,
  LifecycleStatus,
} from "@/types/agentfield";
import {
  getLifecycleTheme,
  type LifecycleTheme,
  normalizeLifecycleStatus,
} from "./lifecycle-status";

export interface NodeStatusPresentation {
  kind: LifecycleStatus;
  label: string;
  theme: LifecycleTheme;
  shouldPulse: boolean;
  online: boolean;
}

const resolveNodeLifecycleStatus = (
  lifecycle?: LifecycleStatus | null,
  health?: HealthStatus | null
): LifecycleStatus => {
  const normalizedLifecycle = normalizeLifecycleStatus(lifecycle);

  if (
    health === "inactive" ||
    normalizedLifecycle === "offline" ||
    normalizedLifecycle === "stopped"
  ) {
    return "offline";
  }

  if (normalizedLifecycle === "error" || health === "degraded") {
    return "error";
  }

  if (normalizedLifecycle === "degraded") {
    return "degraded";
  }

  if (normalizedLifecycle === "starting" || health === "starting") {
    return "starting";
  }

  if (normalizedLifecycle === "running") {
    return "running";
  }

  if (normalizedLifecycle === "ready") {
    return "ready";
  }

  return "unknown";
};

export const getNodeStatusPresentation = (
  lifecycle?: LifecycleStatus | null,
  health?: HealthStatus | null
): NodeStatusPresentation => {
  const kind = resolveNodeLifecycleStatus(lifecycle, health);
  const theme = getLifecycleTheme(kind);

  return {
    kind,
    label: theme.label,
    theme,
    shouldPulse: theme.motion !== "none",
    online: theme.online,
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
      const presentation = getNodeStatusPresentation(
        node.lifecycle_status,
        node.health_status
      );

      acc.total += 1;

      if (presentation.online) {
        acc.online += 1;
      } else {
        acc.offline += 1;
      }

      switch (presentation.kind) {
        case "ready":
        case "running":
          acc.ready += 1;
          break;
        case "starting":
          acc.starting += 1;
          break;
        case "degraded":
          acc.degraded += 1;
          break;
        default:
          break;
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

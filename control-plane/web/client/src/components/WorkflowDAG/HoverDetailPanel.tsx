import { memo, useEffect, useState } from "react";
import {
  Calendar,
  CheckmarkFilled,
  ErrorFilled,
  InProgress,
  PauseFilled,
  Time,
} from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";
import { statusTone } from "../../lib/theme";
import { agentColorManager } from "../../utils/agentColorManager";
import {
  getStatusLabel,
  normalizeExecutionStatus,
  type CanonicalStatus,
} from "../../utils/status";
import { AgentBadge } from "./AgentBadge";
import type { WorkflowDAGNode } from "./DeckGLGraph";

interface HoverDetailPanelProps {
  node: WorkflowDAGNode | null;
  position: { x: number; y: number };
  visible: boolean;
}

const STATUS_TONE_TOKEN_MAP: Record<CanonicalStatus, keyof typeof statusTone> = {
  succeeded: "success",
  failed: "error",
  running: "info",
  paused: "warning",
  waiting: "warning",
  queued: "warning",
  pending: "warning",
  timeout: "neutral",
  cancelled: "neutral",
  unknown: "neutral",
};

const getStatusIcon = (status: CanonicalStatus) => {
  const toneKey = STATUS_TONE_TOKEN_MAP[status] ?? "neutral";
  const iconClass = cn("h-3.5 w-3.5", statusTone[toneKey].accent);
  const iconProps = {
    size: 14,
    className: iconClass,
  } as const;

  switch (status) {
    case "succeeded":
      return <CheckmarkFilled {...iconProps} />;
    case "failed":
      return <ErrorFilled {...iconProps} />;
    case "running":
      return <InProgress {...iconProps} className={cn(iconClass, "animate-spin")} />;
    case "pending":
    case "queued":
      return <PauseFilled {...iconProps} />;
    case "timeout":
      return <Time {...iconProps} />;
    case "cancelled":
      return <PauseFilled {...iconProps} />;
    default:
      return (
        <span className="inline-flex h-3 w-3 rounded-full bg-muted-foreground/60" />
      );
  }
};

const formatDuration = (durationMs?: number) => {
  if (!durationMs) return "-";
  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
  const minutes = Math.floor(durationMs / 60000);
  const seconds = Math.floor((durationMs % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
};

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

// Convert Python function names to human-readable format
const humanizeText = (text: string): string => {
  return text
    .replace(/_/g, ' ')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
    .replace(/\s+/g, ' ')
    .trim();
};

export const HoverDetailPanel = memo(({ node, position, visible }: HoverDetailPanelProps) => {
  const [adjustedPosition, setAdjustedPosition] = useState(position);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (visible && node) {
      // Delay visibility for smooth fade-in
      const timer = setTimeout(() => setIsVisible(true), 50);
      return () => clearTimeout(timer);
    } else {
      setIsVisible(false);
    }
  }, [visible, node]);

  useEffect(() => {
    if (!visible || !node) return;

    // Smart positioning: keep panel within viewport
    const panelWidth = 320;
    const panelHeight = 280;
    const padding = 20;
    const offset = { x: 15, y: -10 };

    let x = position.x + offset.x;
    let y = position.y + offset.y;

    // Adjust horizontal position
    if (x + panelWidth > window.innerWidth - padding) {
      x = position.x - panelWidth - offset.x;
    }

    // Adjust vertical position
    if (y + panelHeight > window.innerHeight - padding) {
      y = window.innerHeight - panelHeight - padding;
    }
    if (y < padding) {
      y = padding;
    }

    setAdjustedPosition({ x, y });
  }, [position, visible, node]);

  if (!node || !visible) return null;

  const normalizedStatus = normalizeExecutionStatus(node.status);
  const toneKey = STATUS_TONE_TOKEN_MAP[normalizedStatus] ?? "neutral";
  const tone = statusTone[toneKey];

  // Handle optional fields that may not exist on WorkflowDAGLightweightNode
  const agentNameField = node.agent_name;
  const taskNameField = node.task_name;

  const agentColor = agentColorManager.getAgentColor(
    agentNameField || node.agent_node_id,
    node.agent_node_id
  );

  const taskName = humanizeText(taskNameField || node.reasoner_id || "Unknown Task");
  const agentName = humanizeText(agentNameField || node.agent_node_id || "Unknown Agent");

  return (
    <div
      className={cn(
        "pointer-events-none fixed z-[9999] min-w-[320px] max-w-[360px]",
        "rounded-xl border-2 border-border bg-popover/98 backdrop-blur-md shadow-2xl",
        "transition-opacity duration-150",
        isVisible ? "opacity-100" : "opacity-0"
      )}
      style={{
        left: `${adjustedPosition.x}px`,
        top: `${adjustedPosition.y}px`,
        transform: 'translate3d(0, 0, 0)', // GPU acceleration
        boxShadow: `
          0 0 0 1px ${agentColor.border},
          0 20px 25px -5px rgba(0, 0, 0, 0.3),
          0 10px 10px -5px rgba(0, 0, 0, 0.2)
        `,
      }}
    >
      <div className="p-4 space-y-3">
        {/* Header with Agent Badge */}
        <div className="flex items-start justify-between gap-3 border-b border-border/60 pb-3">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-foreground leading-tight mb-2">
              {taskName}
            </div>
            <AgentBadge
              agentName={agentName}
              agentId={node.agent_node_id}
              size="sm"
              showTooltip={false}
            />
          </div>
          <div className="flex-shrink-0">
            {getStatusIcon(normalizedStatus)}
          </div>
        </div>

        {/* Main Info Grid */}
        <div className="space-y-2.5">
          {/* Status */}
          <div className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              {getStatusIcon(normalizedStatus)}
              Status:
            </span>
            <span className={cn("text-xs font-semibold", tone.accent)}>
              {getStatusLabel(normalizedStatus)}
            </span>
          </div>

          {/* Duration */}
          <div className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Time size={12} />
              Duration:
            </span>
            <span className="font-mono text-xs font-medium text-foreground">
              {formatDuration(node.duration_ms)}
            </span>
          </div>

          {/* Started */}
          <div className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Calendar size={12} />
              Started:
            </span>
            <span className="font-mono text-xs text-foreground">
              {formatTimestamp(node.started_at)}
            </span>
          </div>

          {/* Completed (if available) */}
          {node.completed_at && (
            <div className="flex items-center justify-between gap-4">
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <CheckmarkFilled size={12} />
                Completed:
              </span>
              <span className="font-mono text-xs text-foreground">
                {formatTimestamp(node.completed_at)}
              </span>
            </div>
          )}
        </div>

        {/* Technical Details */}
        <div className="space-y-1.5 border-t border-border/60 pt-3">
          <div className="flex justify-between gap-4 text-micro">
            <span className="text-muted-foreground">Execution ID:</span>
            <span className="font-mono text-foreground">
              {node.execution_id.slice(0, 12)}...
            </span>
          </div>
          {node.workflow_id && (
            <div className="flex justify-between gap-4 text-micro">
              <span className="text-muted-foreground">Workflow ID:</span>
              <span className="font-mono text-foreground">
                {node.workflow_id.slice(0, 12)}...
              </span>
            </div>
          )}
          {node.workflow_depth !== undefined && (
            <div className="flex justify-between gap-4 text-micro">
              <span className="text-muted-foreground">Depth:</span>
              <span className="font-mono text-foreground">
                Level {node.workflow_depth}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Tooltip Arrow */}
      <div
        className="absolute -left-2 top-4 border-8 border-transparent"
        style={{
          borderRightColor: agentColor.border,
          filter: 'drop-shadow(-2px 0 2px rgba(0, 0, 0, 0.1))'
        }}
      />
    </div>
  );
});

HoverDetailPanel.displayName = "HoverDetailPanel";

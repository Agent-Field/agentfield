import { memo } from "react";
import { Handle, Position, useStore } from "@xyflow/react";
import {
  Calendar,
  CheckmarkFilled,
  Time,
  User,
} from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";
import { type StatusTone as StatusToneKey } from "../../lib/theme";
import { agentColorManager } from "../../utils/agentColorManager";
import {
  getStatusLabel,
  getStatusTheme,
  normalizeExecutionStatus,
  type CanonicalStatus,
} from "../../utils/status";
import { AgentBadge } from "./AgentBadge";

interface WorkflowNodeData {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  workflow_depth: number;
  selected?: boolean;
  task_name?: string;
  agent_name?: string;
  isSearchMatch?: boolean;
  isDimmed?: boolean;
  isFocusPrimary?: boolean;
  isFocusRelated?: boolean;
  focusDistance?: number;
  parent_execution_id?: string;
  viewMode?: 'standard' | 'performance' | 'debug';
  performanceIntensity?: number;
  minPerformance?: number;
  maxPerformance?: number;
}

interface WorkflowNodeProps {
  data: WorkflowNodeData;
  selected?: boolean;
}

// Zoom selector to determine when to show simplified view
const zoomSelector = (s: any) => s.transform[2] >= 0.4; // Show full content when zoom >= 0.4

// Simplified placeholder component for zoomed out view
const StatusPlaceholder = memo(
  ({
    status,
    agentColor,
    data,
  }: {
    status: CanonicalStatus;
    agentColor: ReturnType<typeof agentColorManager.getAgentColor>;
    data: WorkflowNodeData;
  }) => {
    const theme = getStatusTheme(status);
    const statusColorVar = theme.hexColor;
    const glowColor = `color-mix(in srgb, ${theme.hexColor} 55%, transparent)`;

    return (
      <div
        className="h-3 w-3 rounded-full border-2 transition-transform duration-200 hover:scale-110"
        style={{
          backgroundColor: statusColorVar,
          borderColor: agentColor.primary,
          boxShadow: `0 0 4px ${glowColor}`,
        }}
        title={`${getStatusLabel(status)} - ${data.task_name || data.reasoner_id}`}
      />
    );
  },
);

StatusPlaceholder.displayName = "StatusPlaceholder";

export const WorkflowNode = memo(({ data, selected }: WorkflowNodeProps) => {
  const showFullContent = useStore(zoomSelector);
  const normalizedStatus = normalizeExecutionStatus(data.status);
  const {
    isSearchMatch = false,
    isDimmed = false,
    isFocusPrimary = false,
    isFocusRelated = false,
  } = data;
  const viewMode = data.viewMode ?? 'standard';
  const performanceIntensity = Math.min(Math.max(data.performanceIntensity ?? 0, 0), 1);

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

  // Icon reflects this node's OWN canonical status only. A running child
  // under a cancelled parent must still spin/pulse — do not propagate parent
  // state into child node visuals.
  const getStatusIcon = (status: CanonicalStatus) => {
    const theme = getStatusTheme(status);
    const Icon = theme.icon;
    const iconClass = cn("h-4 w-4", theme.iconClass);
    return (
      <Icon
        size={12}
        className={cn(iconClass, theme.motion === "live" && "animate-spin")}
      />
    );
  };

  const getStatusText = (status: string) => {
    return getStatusLabel(status);
  };

  // Convert Python function names to human-readable format
  const humanizeText = (text: string): string => {
    return text
      // Replace underscores with spaces
      .replace(/_/g, ' ')
      // Replace hyphens with spaces
      .replace(/-/g, ' ')
      // Capitalize first letter of each word
      .replace(/\b\w/g, l => l.toUpperCase())
      // Clean up multiple spaces
      .replace(/\s+/g, ' ')
      .trim();
  };

  // Calculate optimal node width with generous sizing for better UX
  const calculateOptimalWidth = (taskText: string, agentText: string): number => {
    const minWidth = 200;  // Increased minimum width
    const maxWidth = 360;  // Increased maximum width for better readability
    const charWidth = 7.5; // More accurate character width for the font

    const taskHuman = humanizeText(taskText);
    const agentHuman = humanizeText(agentText);

    // Calculate width needed to fit text comfortably in two lines
    const taskWordsLength = taskHuman.split(' ').reduce((max, word) => Math.max(max, word.length), 0);
    const agentWordsLength = agentHuman.split(' ').reduce((max, word) => Math.max(max, word.length), 0);

    // Base width on longest single word plus some buffer for multi-word lines
    const longestWord = Math.max(taskWordsLength, agentWordsLength);
    const estimatedWidth = Math.max(
      longestWord * charWidth * 1.8, // 1.8x for comfortable two-line display
      (taskHuman.length / 2.2) * charWidth, // Divide by 2.2 instead of 2 for more generous spacing
      (agentHuman.length / 2.2) * charWidth
    ) + 80; // Increased padding for icons and spacing

    return Math.min(maxWidth, Math.max(minWidth, estimatedWidth));
  };

  // Smart text formatting that prefers single line when possible
  const formatTextForDisplay = (text: string, nodeWidth: number, isAgentName: boolean = false) => {
    const humanText = humanizeText(text);
    const words = humanText.split(' ');

    // Calculate available character space based on node width
    const availableWidth = nodeWidth - (isAgentName ? 100 : 80); // More space for agent names (account for icon)
    const charWidth = 7.5;
    const maxCharsForSingleLine = Math.floor(availableWidth / charWidth);
    const maxCharsPerLine = Math.floor(maxCharsForSingleLine * 0.9); // 90% for comfortable reading

    // PRIORITY 1: Try to fit in single line (especially for agent names)
    if (humanText.length <= maxCharsForSingleLine || (isAgentName && humanText.length <= maxCharsForSingleLine * 1.1)) {
      return { line1: humanText, line2: "", isSingleLine: true };
    }

    // PRIORITY 2: For agent names, be more aggressive about single line
    if (isAgentName && humanText.length <= maxCharsForSingleLine * 1.2) {
      return { line1: humanText, line2: "", isSingleLine: true };
    }

    // PRIORITY 3: Only use two lines when absolutely necessary
    if (words.length === 1) {
      // Single long word - break intelligently at natural points
      const breakPoint = Math.ceil(humanText.length / 2);
      return {
        line1: humanText.substring(0, breakPoint),
        line2: humanText.substring(breakPoint),
        isSingleLine: false
      };
    }

    // Multiple words - smart distribution
    let line1 = "";
    let line2 = "";

    // Try to fit as many complete words as possible on first line
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const testLine1 = line1 + (line1 ? " " : "") + word;

      if (testLine1.length <= maxCharsPerLine || line1 === "") {
        line1 = testLine1;
      } else {
        // Add remaining words to line2
        line2 = words.slice(i).join(" ");
        break;
      }
    }

    // Ensure line2 isn't too long
    if (line2.length > maxCharsPerLine) {
      // Rebalance by moving some words back to line1 if possible
      const allWords = words;
      const midPoint = Math.ceil(allWords.length / 2);
      line1 = allWords.slice(0, midPoint).join(" ");
      line2 = allWords.slice(midPoint).join(" ");
    }

    return { line1, line2, isSingleLine: false };
  };

  const statusTheme = getStatusTheme(normalizedStatus);
  const statusColorVar = statusTheme.hexColor;
  const statusBorderVar = `color-mix(in srgb, ${statusTheme.hexColor} 60%, transparent)`;
  const statusGlowVar = `color-mix(in srgb, ${statusTheme.hexColor} 38%, transparent)`;

  const agentColor = agentColorManager.getAgentColor(
    data.agent_name || data.agent_node_id,
    data.agent_node_id
  );

  const tokenFor = (token: StatusToneKey | "primary") => {
    if (token === "primary") {
      return {
        border: `color-mix(in srgb, var(--primary) 55%, transparent)`,
        glow: `color-mix(in srgb, var(--primary) 40%, transparent)`,
      };
    }
    return {
      border: `var(--status-${token}-border)`,
      glow: `color-mix(in srgb, var(--status-${token}) 40%, transparent)`,
    };
  };

  let borderColor = statusBorderVar;
  let glowColor = statusGlowVar;
  const hasHighlight = isFocusPrimary || isFocusRelated || isSearchMatch || selected;

  if (isFocusPrimary) {
    const highlight = tokenFor("success");
    borderColor = highlight.border;
    glowColor = highlight.glow;
  } else if (isFocusRelated || isSearchMatch) {
    const highlight = tokenFor("info");
    borderColor = highlight.border;
    glowColor = highlight.glow;
  } else if (selected) {
    const highlight = tokenFor("primary");
    borderColor = highlight.border;
    glowColor = highlight.glow;
  } else if (viewMode === "performance") {
    const heat = Math.min(70, 35 + performanceIntensity * 40);
    borderColor = `color-mix(in srgb, var(--status-warning) ${heat}%, transparent)`;
    glowColor = `color-mix(in srgb, var(--status-warning) ${Math.min(30 + performanceIntensity * 50, 85)}%, transparent)`;
  } else if (viewMode === "debug") {
    borderColor = "var(--border)";
    glowColor = "color-mix(in srgb, var(--muted-foreground) 45%, transparent)";
  }

  const baseShadow =
    "0 1px 2px color-mix(in srgb, var(--foreground) 6%, transparent), 0 1px 3px color-mix(in srgb, var(--foreground) 4%, transparent)";
  const accentShadow = `0 0 0 1px ${borderColor}`;
  const glowShadow = isDimmed ? "" : `0 0 12px -2px ${glowColor}`;
  const compositeShadow = [accentShadow, baseShadow, glowShadow].filter(Boolean).join(", ");

  const baseBackground = `linear-gradient(145deg, color-mix(in srgb, ${statusColorVar} 6%, var(--card)), var(--card))`;
  let background = baseBackground;

  if (!hasHighlight) {
    if (viewMode === "performance") {
      const heat = Math.min(65, 25 + performanceIntensity * 45);
      background = `linear-gradient(135deg, color-mix(in srgb, var(--status-warning) ${heat}%, transparent), var(--card))`;
    } else if (viewMode === "debug") {
      background = `linear-gradient(135deg, color-mix(in srgb, var(--muted) 18%, transparent), var(--card))`;
    }
  }

  // Calculate optimal node width based on content
  const taskText = data.task_name || data.reasoner_id;
  const agentText = data.agent_name || data.agent_node_id;
  const nodeWidth = calculateOptimalWidth(taskText, agentText);

  // Early return for simplified view when zoomed out
  if (!showFullContent) {
    return (
      <div className="flex items-center justify-center w-6 h-6">
        <Handle
          type="target"
          position={Position.Left}
          className="!w-1 !h-1 !bg-transparent !border-0 !opacity-0 !pointer-events-none"
          id="target-left"
        />
        <StatusPlaceholder
          status={normalizedStatus}
          agentColor={agentColor}
          data={data}
        />
        <Handle
          type="source"
          position={Position.Right}
          className="!w-1 !h-1 !bg-transparent !border-0 !opacity-0 !pointer-events-none"
          id="source-right"
        />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "group relative h-[100px] cursor-pointer overflow-hidden rounded-xl border border-border bg-card text-card-foreground shadow-sm backdrop-blur-sm transition-all duration-300 animate-fade-in",
        !isDimmed && "hover:scale-[1.01] hover:shadow-md",
      )}
      style={{
        width: `${nodeWidth}px`,
        borderColor,
        boxShadow: compositeShadow,
        opacity: isDimmed ? 0.4 : 1,
        filter: isDimmed ? "grayscale(65%) saturate(70%)" : undefined,
        background,
      }}
    >
      {/* Agent color left border accent */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl opacity-80"
        style={{
          background: `linear-gradient(to bottom, ${agentColor.primary}, ${agentColor.border})`,
        }}
      />

      {/* Agent Badge - positioned in top-left */}
      <div className="absolute top-2 left-2 z-10">
        <AgentBadge
          agentName={data.agent_name || data.agent_node_id}
          agentId={data.agent_node_id}
          size="sm"
          showTooltip={false}
        />
      </div>
      {/* Invisible connection handles - Required for ReactFlow edges but hidden from user */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-1 !h-1 !bg-transparent !border-0 !opacity-0 !pointer-events-none"
        id="target-left"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-1 !h-1 !bg-transparent !border-0 !opacity-0 !pointer-events-none"
        id="source-right"
      />

      <div className="relative flex h-full flex-col p-3">
        <div className="absolute right-2 top-2">
          {getStatusIcon(normalizedStatus)}
        </div>

        <div className="flex flex-1 flex-col justify-start pl-8 pr-6 pt-1">
          {(() => {
            const taskFormatted = formatTextForDisplay(taskText, nodeWidth, false);
            return (
              <div
                className={cn(
                  "flex flex-col justify-start text-sm font-semibold leading-[1.15] text-foreground",
                  taskFormatted.isSingleLine ? "min-h-[16px]" : "min-h-[32px]",
                  normalizedStatus === "cancelled" && "line-through opacity-60",
                )}
              >
                <div
                  className="break-words hyphens-auto"
                  title={humanizeText(taskText)}
                  style={{ wordBreak: 'break-word' }}
                >
                  {taskFormatted.line1}
                </div>
                {taskFormatted.line2 && (
                  <div
                    className="break-words hyphens-auto"
                    style={{ wordBreak: 'break-word' }}
                  >
                    {taskFormatted.line2}
                  </div>
                )}
              </div>
            );
          })()}
        </div>

        <div className="mb-2 flex min-h-[16px] items-center gap-1.5 border-t border-border/40 pt-2">
          <User size={12} className="flex-shrink-0 text-muted-foreground" />
          {(() => {
            const agentFormatted = formatTextForDisplay(agentText, nodeWidth, true);
            return (
              <div
                className={cn(
                  "flex-1 text-xs font-medium leading-[1.15] text-muted-foreground",
                  agentFormatted.isSingleLine
                    ? "flex items-center"
                    : "flex flex-col justify-start"
                )}
              >
                <div
                  className="break-words hyphens-auto"
                  title={humanizeText(agentText)}
                  style={{ wordBreak: 'break-word' }}
                >
                  {agentFormatted.line1}
                </div>
                {agentFormatted.line2 && (
                  <div
                    className="break-words hyphens-auto"
                    style={{ wordBreak: 'break-word' }}
                  >
                    {agentFormatted.line2}
                  </div>
                )}
              </div>
            );
          })()}
        </div>

        <div className="flex min-h-[16px] items-center justify-between text-xs">
          {/* Duration with Time icon */}
          <div className="flex flex-shrink-0 items-center gap-1">
            <Time size={11} className="flex-shrink-0 text-muted-foreground" />
            <span
              className="font-mono text-xs font-medium text-muted-foreground"
              title={`Duration: ${formatDuration(data.duration_ms)}`}
            >
              {formatDuration(data.duration_ms)}
            </span>
          </div>

          {/* Timestamp with Calendar icon */}
          <div className="flex flex-shrink-0 items-center gap-1">
            <Calendar size={11} className="flex-shrink-0 text-muted-foreground" />
            <span
              className="font-mono text-sm text-muted-foreground"
              title={`Started: ${formatTimestamp(data.started_at)}`}
            >
              {formatTimestamp(data.started_at)}
            </span>
          </div>
        </div>

        {viewMode === 'performance' && (
          <div className="mt-2 space-y-1">
            <div className="h-1.5 overflow-hidden rounded-full bg-muted/60">
              <div
                className="h-full rounded-full bg-status-warning"
                style={{ width: `${Math.max(6, performanceIntensity * 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-micro text-muted-foreground">
              <span>Load {(performanceIntensity * 100).toFixed(0)}%</span>
              {data.duration_ms ? <span>{formatDuration(data.duration_ms)}</span> : null}
            </div>
          </div>
        )}

        {viewMode === 'debug' && (
          <div className="mt-2 space-y-1 text-micro font-mono text-muted-foreground">
            <div>ID: {data.execution_id}</div>
            {data.parent_execution_id && <div>Parent: {data.parent_execution_id.slice(0, 8)}…</div>}
            <div>
              Status:{" "}
              <span className={cn("font-semibold", statusTheme.iconClass)}>
                {getStatusText(normalizedStatus)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Hover Tooltip */}
      <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-3 min-w-max -translate-x-1/2 rounded-xl border border-border bg-popover px-4 py-3 text-xs opacity-0 shadow-xl transition-all duration-300 group-hover:opacity-100 backdrop-blur-md">
        <div className="space-y-3">
          {/* Header */}
          <div className="border-b border-border/60 pb-2 text-sm font-semibold text-foreground">
            {humanizeText(data.task_name || data.reasoner_id)}
          </div>

          {/* Main Info */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-6">
              <span className="flex items-center gap-1 text-muted-foreground">
                <User size={12} />
                Agent:
              </span>
              <span className="font-medium text-foreground">
                {humanizeText(data.agent_name || data.agent_node_id)}
              </span>
            </div>

            <div className="flex items-center justify-between gap-6">
              <span className="flex items-center gap-1 text-muted-foreground">
                {getStatusIcon(normalizedStatus)}
                Status:
              </span>
              <span className={cn("font-medium", statusTheme.iconClass)}>
                {getStatusText(normalizedStatus)}
              </span>
            </div>

            <div className="flex items-center justify-between gap-6">
              <span className="flex items-center gap-1 text-muted-foreground">
                <Time size={12} />
                Duration:
              </span>
              <span className="font-mono font-medium text-foreground">
                {formatDuration(data.duration_ms)}
              </span>
            </div>

            <div className="flex items-center justify-between gap-6">
              <span className="flex items-center gap-1 text-muted-foreground">
                <Calendar size={12} />
                Started:
              </span>
              <span className="font-mono text-foreground">
                {formatTimestamp(data.started_at)}
              </span>
            </div>

            {data.completed_at && (
              <div className="flex items-center justify-between gap-6">
                <span className="flex items-center gap-1 text-muted-foreground">
                  <CheckmarkFilled size={12} />
                  Completed:
                </span>
                <span className="font-mono text-foreground">
                  {formatTimestamp(data.completed_at)}
                </span>
              </div>
            )}
          </div>

          {/* Technical Details */}
          <div className="space-y-1 border-t border-border/60 pt-2">
            <div className="flex justify-between gap-4 text-micro text-muted-foreground">
              <span>Execution ID:</span>
              <span className="font-mono text-foreground">
                {data.execution_id.slice(0, 8)}...
              </span>
            </div>
            <div className="flex justify-between gap-4 text-micro text-muted-foreground">
              <span>Workflow ID:</span>
              <span className="font-mono text-foreground">
                {data.workflow_id.slice(0, 8)}...
              </span>
            </div>
          </div>
        </div>

        {/* Tooltip Arrow */}
        <div
          className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent"
          style={{ borderTopColor: "var(--popover)" }}
        />
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function for React.memo
  // Only re-render if essential data has changed
  return (
    prevProps.data.execution_id === nextProps.data.execution_id &&
    prevProps.data.status === nextProps.data.status &&
    prevProps.data.duration_ms === nextProps.data.duration_ms &&
    prevProps.data.task_name === nextProps.data.task_name &&
    prevProps.data.agent_name === nextProps.data.agent_name &&
    prevProps.data.started_at === nextProps.data.started_at &&
    prevProps.data.completed_at === nextProps.data.completed_at &&
    prevProps.data.isSearchMatch === nextProps.data.isSearchMatch &&
    prevProps.data.isDimmed === nextProps.data.isDimmed &&
    prevProps.data.isFocusPrimary === nextProps.data.isFocusPrimary &&
    prevProps.data.isFocusRelated === nextProps.data.isFocusRelated &&
    prevProps.selected === nextProps.selected
  );
});

WorkflowNode.displayName = 'WorkflowNode';

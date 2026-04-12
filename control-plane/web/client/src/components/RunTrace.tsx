import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { cn } from "@/lib/utils";
import { getStatusTheme, normalizeExecutionStatus } from "@/utils/status";
import { StatusDot } from "@/components/ui/status-pill";
import type { WorkflowDAGLightweightNode } from "@/types/workflows";

// ─── Tree node type (runtime-constructed) ────────────────────────────────────

export interface TraceTreeNode extends WorkflowDAGLightweightNode {
  children: TraceTreeNode[];
}

// ─── Build tree from flat timeline ───────────────────────────────────────────

export function buildTraceTree(
  timeline: WorkflowDAGLightweightNode[],
): TraceTreeNode | null {
  if (!timeline || timeline.length === 0) return null;

  const nodeMap = new Map<string, TraceTreeNode>();
  const orphans: TraceTreeNode[] = [];

  for (const node of timeline) {
    nodeMap.set(node.execution_id, { ...node, children: [] });
  }

  for (const node of nodeMap.values()) {
    if (node.parent_execution_id) {
      const parent = nodeMap.get(node.parent_execution_id);
      if (parent) {
        parent.children.push(node);
      } else {
        orphans.push(node);
      }
    } else {
      orphans.push(node);
    }
  }

  // Return the root (depth 0 node) or first orphan
  const root =
    orphans.find((n) => n.workflow_depth === 0) ?? orphans[0] ?? null;

  if (root && orphans.length > 1) {
    // Attach remaining orphans as children of root
    for (const orphan of orphans) {
      if (orphan !== root) {
        root.children.push(orphan);
      }
    }
  }

  return root;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3_600_000) {
    const min = Math.floor(ms / 60_000);
    const sec = Math.round((ms % 60_000) / 1000);
    return `${min}m ${sec}s`;
  }
  if (ms < 86_400_000) {
    const hr = Math.floor(ms / 3_600_000);
    const min = Math.round((ms % 3_600_000) / 60_000);
    return `${hr}h ${min}m`;
  }
  const days = Math.floor(ms / 86_400_000);
  const hr = Math.round((ms % 86_400_000) / 3_600_000);
  return `${days}d ${hr}h`;
}

function formatRelativeStart(ms: number): string {
  if (ms < 0) return "+0:00";
  const secs = Math.floor(ms / 1000);
  const mins = Math.floor(secs / 60);
  const remSecs = secs % 60;
  if (mins < 60) return `+${mins}:${String(remSecs).padStart(2, "0")}`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `+${hours}:${String(remMins).padStart(2, "0")}:${String(remSecs).padStart(2, "0")}`;
}

// ─── Flat step representation ─────────────────────────────────────────────────

interface FlatStep {
  node: TraceTreeNode;
  depth: number;
  index: number;
  isFirstOfGroup: boolean;
  effectiveGroupCount: number;
  showSeparator: boolean;
}

function buildFlatSteps(root: TraceTreeNode): FlatStep[] {
  const result: FlatStep[] = [];

  // DFS traversal to produce ordered flat list
  function visit(node: TraceTreeNode, depth: number) {
    result.push({
      node,
      depth,
      index: result.length,
      isFirstOfGroup: false,
      effectiveGroupCount: 0,
      showSeparator: false,
    });
    for (const child of node.children ?? []) {
      visit(child, depth + 1);
    }
  }
  visit(root, 0);

  // Re-index
  result.forEach((step, i) => {
    step.index = i;
  });

  // Compute group separators and group count badges
  // We need sibling context: for each node, find siblings (same parent)
  const siblingMap = new Map<string | null | undefined, FlatStep[]>();
  for (const step of result) {
    const parentId = step.node.parent_execution_id ?? null;
    if (!siblingMap.has(parentId)) {
      siblingMap.set(parentId, []);
    }
    siblingMap.get(parentId)!.push(step);
  }

  for (const step of result) {
    const parentId = step.node.parent_execution_id ?? null;
    const siblings = siblingMap.get(parentId) ?? [];
    const siblingIndex = siblings.findIndex(
      (s) => s.node.execution_id === step.node.execution_id,
    );
    const prevSibling = siblingIndex > 0 ? siblings[siblingIndex - 1] : null;

    // Separator: depth > 0 and previous flat node has different reasoner_id
    const prevFlat = step.index > 0 ? result[step.index - 1] : null;
    step.showSeparator =
      step.depth > 0 &&
      prevFlat !== null &&
      prevFlat.node.reasoner_id !== step.node.reasoner_id;

    // Group badge
    const isFirstOfGroup =
      prevSibling === null || prevSibling.node.reasoner_id !== step.node.reasoner_id;
    step.isFirstOfGroup = isFirstOfGroup;

    if (isFirstOfGroup) {
      const groupCount = siblings
        .slice(siblingIndex)
        .findIndex((s) => s.node.reasoner_id !== step.node.reasoner_id);
      step.effectiveGroupCount =
        groupCount === -1 ? siblings.length - siblingIndex : groupCount;
    }
  }

  return result;
}

// StatusDot now comes from @/components/ui/status-pill so the trace dot,
// the runs-table dot, and any future consumer all share the same colour,
// motion, and label logic.

// ─── Single trace row ─────────────────────────────────────────────────────────

interface TraceRowProps {
  step: FlatStep;
  maxDuration: number;
  selectedId: string | null;
  onSelect: (executionId: string) => void;
  runStartedAt?: string | null;
  /** When the root execution is already in a terminal state, suppress
   * motion on child rows even if their own status is still running. The
   * control plane has given up on the run so any "live" motion would be
   * misleading. */
  rootTerminal: boolean;
}

function TraceRow({
  step,
  maxDuration,
  selectedId,
  onSelect,
  runStartedAt,
  rootTerminal,
}: TraceRowProps) {
  const { node, depth, index, isFirstOfGroup, effectiveGroupCount, showSeparator } = step;

  const barWidth =
    node.duration_ms != null
      ? Math.max(4, (node.duration_ms / Math.max(maxDuration, 1)) * 100)
      : 4;
  const isSelected = node.execution_id === selectedId;

  // Bar color derives from this row's own canonical status — covers
  // cancelled/paused/timeout via the theme. Motion comes from the theme
  // too, BUT the root-terminal gate wins: if the whole run has already
  // ended, no child should pretend to be making progress.
  const barTheme = getStatusTheme(node.status);
  const rowLive = barTheme.motion === "live" && !rootTerminal;
  const barColor = cn(
    barTheme.indicatorClass,
    rowLive && "motion-safe:animate-pulse",
    // When the root is terminal but this child is still "running" in the
    // DB, desaturate the bar so it reads as abandoned rather than active.
    rootTerminal && barTheme.motion === "live" && "opacity-50",
  );
  const isCancelled = normalizeExecutionStatus(node.status) === "cancelled";

  let relativeStart: string | null = null;
  if (runStartedAt && node.started_at) {
    const runStartMs = new Date(runStartedAt).getTime();
    const stepStartMs = new Date(node.started_at).getTime();
    relativeStart = formatRelativeStart(stepStartMs - runStartMs);
  }

  return (
    <div>
      {showSeparator && <div className="border-t border-border/30 my-0.5" />}

      <button
        type="button"
        onClick={() => onSelect(node.execution_id)}
        className={cn(
          "flex min-w-0 items-center gap-1 w-full rounded-md transition-colors sm:gap-1.5",
          "hover:bg-accent text-left py-1",
          isSelected && "bg-accent",
        )}
        style={{
          paddingLeft: `${depth * 14 + 8}px`,
          paddingRight: "8px",
        }}
      >
        {/* Step number */}
        <span className="text-micro text-muted-foreground/50 tabular-nums w-5 text-right shrink-0">
          {index + 1}
        </span>

        {/* Tree connector — only for children */}
        {depth > 0 && (
          <span className="text-muted-foreground/40 text-micro shrink-0 font-mono w-4">
            └─
          </span>
        )}

        {/* Status dot — forced to "cancelled" gray when the whole run is
            terminal so child rows don't advertise motion on a dead run. */}
        <StatusDot
          status={rootTerminal && rowLive ? "cancelled" : node.status}
          label={false}
        />

        {/* Reasoner name */}
        <span
          className={cn(
            "flex-1 truncate font-mono text-xs min-w-0",
            isCancelled && "line-through opacity-60",
          )}
        >
          {node.reasoner_id}
        </span>

        {/* Group count badge */}
        {isFirstOfGroup && effectiveGroupCount > 1 && (
          <span className="text-nano text-muted-foreground bg-muted rounded px-1 shrink-0">
            ×{effectiveGroupCount}
          </span>
        )}

        {/* Duration bar */}
        <div className="w-12 flex items-center shrink-0 sm:w-16">
          <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", barColor)}
              style={{ width: `${barWidth}%` }}
            />
          </div>
        </div>

        {/* Relative start */}
        {relativeStart !== null && (
          <span className="hidden text-micro text-muted-foreground/40 tabular-nums shrink-0 w-12 text-right sm:inline-block">
            {relativeStart}
          </span>
        )}

        {/* Duration text */}
        <span className="text-micro text-muted-foreground tabular-nums whitespace-nowrap shrink-0 w-10 text-right">
          {formatDuration(node.duration_ms)}
        </span>
      </button>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

interface RunTraceProps {
  node: TraceTreeNode;
  maxDuration: number;
  selectedId: string | null;
  onSelect: (executionId: string) => void;
  /** ISO string of the run's start time, used to compute relative step start offsets */
  runStartedAt?: string | null;
  /** Root execution status — when terminal (cancelled/timeout/failed/
   * succeeded) any still-running children are rendered without motion. */
  rootStatus?: string | null;
}

export function RunTrace({
  node,
  maxDuration,
  selectedId,
  onSelect,
  runStartedAt,
  rootStatus,
}: RunTraceProps) {
  const rootTerminal = rootStatus
    ? ["succeeded", "failed", "cancelled", "timeout"].includes(
        normalizeExecutionStatus(rootStatus),
      )
    : false;
  const flatSteps = useMemo(() => buildFlatSteps(node), [node]);

  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: flatSteps.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 28,
    overscan: 20,
  });

  return (
    <div ref={parentRef} className="h-full min-w-0 overflow-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: "relative",
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const step = flatSteps[virtualRow.index];
          return (
            <div
              key={virtualRow.key}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <TraceRow
                step={step}
                maxDuration={maxDuration}
                selectedId={selectedId}
                onSelect={onSelect}
                runStartedAt={runStartedAt}
                rootTerminal={rootTerminal}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

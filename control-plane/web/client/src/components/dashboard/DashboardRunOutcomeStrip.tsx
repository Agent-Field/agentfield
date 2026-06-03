import { useMemo } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { formatCompactDate, formatRelativeTime } from "@/utils/dateFormat";
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from "@/utils/status";
import type { WorkflowSummary } from "@/types/workflows";

import { shortRunIdForDashboard } from "./dashboardRunUtils";

/** ~3× prior density; packed flex like uptime bars. */
const TIMELINE_LIMIT = 36;

function formatStatusCounts(counts: Record<string, number>): string | null {
  const parts = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, n]) => `${n} ${k}`);
  return parts.length ? parts.join(" · ") : null;
}

function timelineWindowSummary(runs: WorkflowSummary[]): string {
  const n = runs.length;
  if (n === 0) return "";

  const inFlight = runs.filter((r) => !r.terminal).length;
  const terminal = runs.filter((r) => r.terminal);
  const succeeded = terminal.filter((r) => normalizeExecutionStatus(r.status) === "succeeded").length;
  const failed = terminal.filter((r) => {
    const s = normalizeExecutionStatus(r.status);
    return s === "failed" || s === "timeout";
  }).length;
  const other = terminal.length - succeeded - failed;

  const bits: string[] = [`${n} run${n === 1 ? "" : "s"}`];
  if (inFlight) bits.push(`${inFlight} in flight`);
  if (succeeded) bits.push(`${succeeded} ok`);
  if (failed) bits.push(`${failed} failed`);
  if (other > 0) bits.push(`${other} other`);
  return bits.join(" · ");
}

function allTerminalSucceeded(runs: WorkflowSummary[]): boolean {
  if (runs.length === 0) return false;
  return runs.every(
    (r) => r.terminal && normalizeExecutionStatus(r.status) === "succeeded",
  );
}

export interface DashboardRunOutcomeStripProps {
  runs: WorkflowSummary[];
  loading?: boolean;
  onSelectRun: (runId: string) => void;
  className?: string;
}

export function DashboardRunOutcomeStrip({
  runs,
  loading,
  onSelectRun,
  className,
}: DashboardRunOutcomeStripProps) {
  const ordered = useMemo(() => {
    const sorted = [...runs].sort(
      (a, b) =>
        new Date(b.latest_activity).getTime() - new Date(a.latest_activity).getTime(),
    );
    const slice = sorted.slice(0, TIMELINE_LIMIT);
    return slice.reverse();
  }, [runs]);

  const headerSummary = useMemo(() => timelineWindowSummary(ordered), [ordered]);
  const showAllOkBanner = useMemo(() => allTerminalSucceeded(ordered), [ordered]);

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader className="space-y-2 pb-2">
          <div className="flex items-start justify-between gap-3">
            <Skeleton className="h-4 w-36" />
            <Skeleton className="h-3 w-28" />
          </div>
          <Skeleton className="h-3 w-full max-w-md" />
        </CardHeader>
        <CardContent className="space-y-2 pt-0">
          <div className="flex h-9 w-full min-w-0 gap-px rounded-md border border-border bg-muted/20 p-px">
            {Array.from({ length: TIMELINE_LIMIT }).map((_, i) => (
              <Skeleton key={i} className="h-8 min-h-0 min-w-0 flex-1 rounded-[1px]" />
            ))}
          </div>
          <Skeleton className="h-2.5 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (ordered.length === 0) {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Run timeline</CardTitle>
          <CardDescription>
            Last {TIMELINE_LIMIT} runs by latest activity, oldest on the left.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <p className="text-sm text-muted-foreground">No runs in the current window.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="space-y-0.5 pb-2">
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-0.5">
          <CardTitle className="text-sm font-medium">Run timeline</CardTitle>
          <p className="text-micro-plus text-muted-foreground tabular-nums leading-tight">{headerSummary}</p>
        </div>
        <CardDescription className="text-xs leading-snug">
          Last {ordered.length} by activity ({TIMELINE_LIMIT} max). Hover a segment for details; click to open
          the run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 pt-0">
        {showAllOkBanner ? (
          <div
            className="border-l-2 border-status-success/50 pl-2.5 py-1 text-xs leading-snug text-muted-foreground"
            role="status"
          >
            <span className="text-foreground">All {ordered.length} run{ordered.length === 1 ? "" : "s"}</span> in
            this view finished successfully.
          </div>
        ) : null}

        <div className="space-y-1">
          <div
            className="flex h-9 w-full min-w-0 gap-px rounded-md border border-border bg-muted/20 p-px"
            role="list"
            aria-label="Run outcomes over time, oldest to newest"
          >
            {ordered.map((run) => {
              const theme = getStatusTheme(run.status);

              const whenLabel = formatCompactDate(run.latest_activity);
              const countsLine = formatStatusCounts(run.status_counts);

              return (
                <HoverCard key={`${run.run_id}-${run.started_at}-tl`} openDelay={120} closeDelay={80}>
                  <HoverCardTrigger asChild>
                    <button
                      type="button"
                      role="listitem"
                      onClick={() => onSelectRun(run.run_id)}
                      title={`${getStatusLabel(run.status)} — ${shortRunIdForDashboard(run.run_id)}`}
                      className={cn(
                        "min-h-0 min-w-0 flex-1 cursor-pointer rounded-[1px] border border-transparent transition-opacity hover:opacity-85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
                        theme.bgClass,
                        theme.borderClass,
                        !run.terminal && "animate-pulse",
                      )}
                      aria-label={`Run ${shortRunIdForDashboard(run.run_id)}, ${getStatusLabel(run.status)}. Open run.`}
                    />
                  </HoverCardTrigger>
                  <HoverCardContent
                    side="top"
                    align="center"
                    className="w-72 max-w-[min(100vw-2rem,18rem)] space-y-2 border-border p-3"
                  >
                    <div>
                      <p className="text-xs font-medium leading-tight text-foreground">{whenLabel}</p>
                      <p className="mt-0.5 font-mono text-micro-plus text-muted-foreground">
                        {shortRunIdForDashboard(run.run_id)}
                      </p>
                    </div>
                    <div className="rounded-md border border-border bg-muted/25 px-2 py-1.5">
                      <p className="text-xs font-medium leading-tight">{getStatusLabel(run.status)}</p>
                      {run.terminal && run.duration_ms != null ? (
                        <p className="mt-0.5 text-micro-plus text-muted-foreground">
                          Duration {formatDurationLabel(run.duration_ms)}
                        </p>
                      ) : !run.terminal ? (
                        <p className="mt-0.5 text-micro-plus leading-snug text-muted-foreground">
                          In flight · latest activity {formatRelativeTime(run.latest_activity)}
                        </p>
                      ) : null}
                    </div>
                    <Separator />
                    <div className="space-y-1 text-micro-plus leading-snug">
                      <p className="font-medium text-muted-foreground">Details</p>
                      <p>
                        <span className="text-muted-foreground">Reasoner </span>
                        <span className="text-foreground">{run.root_reasoner || run.display_name || "—"}</span>
                      </p>
                      {run.current_task?.trim() ? (
                        <p className="line-clamp-3 text-muted-foreground">{run.current_task.trim()}</p>
                      ) : null}
                      {countsLine ? (
                        <p className="text-muted-foreground">
                          <span className="text-foreground/80">Steps </span>
                          {countsLine}
                        </p>
                      ) : null}
                    </div>
                  </HoverCardContent>
                </HoverCard>
              );
            })}
          </div>
          <div className="flex justify-between px-0.5 text-micro font-normal text-muted-foreground">
            <span>Oldest</span>
            <span>Newest</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function formatDurationLabel(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h}h ${rm}m` : `${h}h`;
}

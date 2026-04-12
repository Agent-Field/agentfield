import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useSSESync } from "@/hooks/useSSEQuerySync";
import { AlertTriangle, ArrowRight, CheckCircle, Layers } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  formatDurationHumanReadable,
  LiveElapsedDuration,
} from "@/components/ui/data-formatters";

import { DashboardActiveWorkload } from "@/components/dashboard/DashboardActiveWorkload";
import { DashboardRunOutcomeStrip } from "@/components/dashboard/DashboardRunOutcomeStrip";
import { shortRunIdForDashboard as shortRunId } from "@/components/dashboard/dashboardRunUtils";

import { useRuns } from "@/hooks/queries";
import { useLLMHealth, useQueueStatus } from "@/hooks/queries";
import { useAgents } from "@/hooks/queries";
import { getDashboardSummary } from "@/services/dashboardService";
import { formatRelativeTime } from "@/utils/dateFormat";
import {
  getStatusTheme,
  isFailureStatus,
  isTerminalStatus,
  isTimeoutStatus,
} from "@/utils/status";
import type { WorkflowSummary } from "@/types/workflows";
import type { AgentNodeSummary } from "@/types/agentfield";

// ─── helpers ────────────────────────────────────────────────────────────────

function terminalActivityMs(run: WorkflowSummary): number {
  const t = run.completed_at ?? run.latest_activity ?? run.started_at;
  const ms = new Date(t).getTime();
  return Number.isNaN(ms) ? 0 : ms;
}

function partitionDashboardRuns(runs: WorkflowSummary[]) {
  // A run is "active" only when the ROOT execution is non-terminal. The
  // children-aggregated `terminal` flag lies in the presence of still-
  // dispatched child requests after the root has already been marked
  // cancelled or timed out. root_execution_status is the honest signal.
  const isActive = (r: WorkflowSummary) => {
    const effective = r.root_execution_status ?? r.status;
    return !isTerminalStatus(effective);
  };
  const active = runs.filter(isActive);
  active.sort(
    (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime(),
  );

  const terminal = runs.filter((r) => !isActive(r));
  terminal.sort((a, b) => terminalActivityMs(b) - terminalActivityMs(a));
  const latestCompleted = terminal[0];

  const failures = runs.filter((r) => {
    const effective = r.root_execution_status ?? r.status;
    return isFailureStatus(effective) || isTimeoutStatus(effective);
  });
  failures.sort(
    (a, b) =>
      new Date(b.latest_activity).getTime() - new Date(a.latest_activity).getTime(),
  );

  const reasonerFailCounts = new Map<string, number>();
  for (const r of failures) {
    const key = r.root_reasoner || r.display_name || "—";
    reasonerFailCounts.set(key, (reasonerFailCounts.get(key) ?? 0) + 1);
  }
  const topFailingReasoners = [...reasonerFailCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return { active, latestCompleted, failures, topFailingReasoners };
}

// ─── run status badge ─────────────────────────────────────────────────────────

function RunStatusBadge({ status }: { status: string }) {
  const theme = getStatusTheme(status);

  const variantMap: Record<string, "success" | "failed" | "running" | "pending" | "unknown"> = {
    succeeded: "success",
    failed: "failed",
    running: "running",
    pending: "pending",
    queued: "pending",
    waiting: "pending",
    paused: "pending",
    cancelled: "unknown",
    timeout: "failed",
    unknown: "unknown",
  };

  const badgeVariant = variantMap[theme.status] ?? "unknown";

  const labelMap: Record<string, string> = {
    succeeded: "ok",
    failed: "fail",
    running: "running",
    pending: "pending",
    queued: "queued",
    waiting: "waiting",
    paused: "paused",
    cancelled: "cancelled",
    timeout: "timeout",
    unknown: "unknown",
  };

  return (
    <Badge variant={badgeVariant} size="sm">
      {labelMap[theme.status] ?? theme.status}
    </Badge>
  );
}

// ─── issues banner ────────────────────────────────────────────────────────────

interface IssuesBannerProps {
  llmHealthLoading: boolean;
  unhealthyEndpoints: string[];
  queueOverloaded: boolean;
  overloadedAgents: string[];
}

function IssuesBanner({
  llmHealthLoading,
  unhealthyEndpoints,
  queueOverloaded,
  overloadedAgents,
}: IssuesBannerProps) {
  if (llmHealthLoading) return null;

  const issues: string[] = [];

  if (unhealthyEndpoints.length > 0) {
    const label =
      unhealthyEndpoints.length === 1
        ? `LLM circuit OPEN on endpoint: ${unhealthyEndpoints[0]}`
        : `LLM circuit OPEN on ${unhealthyEndpoints.length} endpoints: ${unhealthyEndpoints.join(", ")}`;
    issues.push(label);
  }

  if (queueOverloaded && overloadedAgents.length > 0) {
    issues.push(
      `Queue at capacity for agent${overloadedAgents.length > 1 ? "s" : ""}: ${overloadedAgents.join(", ")}`,
    );
  }

  if (issues.length === 0) return null;

  return (
    <Alert variant="destructive">
      <AlertTriangle className="size-4" />
      <AlertTitle>System issues</AlertTitle>
      <AlertDescription className="mt-1 space-y-0.5">
        {issues.map((issue, i) => (
          <div key={i}>{issue}</div>
        ))}
      </AlertDescription>
    </Alert>
  );
}

// ─── primary focus: active runs or latest completed ───────────────────────────

interface PrimaryRunFocusProps {
  loading: boolean;
  active: WorkflowSummary[];
  latestCompleted: WorkflowSummary | undefined;
  onOpenRun: (runId: string) => void;
  onViewRunsList: () => void;
}

function PrimaryRunFocus({
  loading,
  active,
  latestCompleted,
  onOpenRun,
  onViewRunsList,
}: PrimaryRunFocusProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader className="space-y-2">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-3 w-full max-w-md" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (active.length > 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="text-base font-semibold">Active runs</CardTitle>
              <CardDescription>
                Most recently started first. Open a run to see the live DAG and current step.
              </CardDescription>
            </div>
            <Badge variant="secondary" className="shrink-0">
              {active.length} active
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <ScrollArea className="max-h-[min(22rem,50vh)] pr-3">
            <ul className="space-y-3">
              {active.map((run) => (
                <li
                  key={`${run.run_id}-${run.started_at}`}
                  className="rounded-lg border border-border bg-card p-3 shadow-sm"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className="font-mono text-xs text-muted-foreground"
                          title={run.run_id}
                        >
                          {shortRunId(run.run_id)}
                        </span>
                        <RunStatusBadge status={run.root_execution_status ?? run.status} />
                      </div>
                      <p className="truncate text-sm font-medium text-foreground">
                        {run.root_reasoner || run.display_name || "—"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground/80">Current state: </span>
                        {run.current_task?.trim() ? run.current_task : "—"}
                      </p>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-2">
                      <div className="text-xs text-muted-foreground">
                        Elapsed{" "}
                        <LiveElapsedDuration
                          startedAt={run.started_at}
                          className="text-foreground"
                        />
                      </div>
                      <Button size="sm" onClick={() => onOpenRun(run.run_id)}>
                        Open run
                      </Button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </ScrollArea>
        </CardContent>
      </Card>
    );
  }

  if (latestCompleted) {
    const run = latestCompleted;
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="text-base font-semibold">Latest run</CardTitle>
              <CardDescription>
                Most recent finished run. When nothing is active, this is your fastest path back
                into the DAG.
              </CardDescription>
            </div>
            <RunStatusBadge status={run.root_execution_status ?? run.status} />
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pt-0">
          <div className="rounded-lg border border-border bg-muted/30 p-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0 space-y-2">
                <p
                  className="font-mono text-sm text-muted-foreground"
                  title={run.run_id}
                >
                  {shortRunId(run.run_id)}
                </p>
                <p className="text-sm font-medium text-foreground">
                  {run.root_reasoner || run.display_name || "—"}
                </p>
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium text-foreground/80">Last state: </span>
                  {run.current_task?.trim() ? run.current_task : "—"}
                </p>
              </div>
              <div className="grid gap-1 text-right text-xs text-muted-foreground sm:text-left">
                <div>
                  Duration{" "}
                  <span className="font-medium tabular-nums text-foreground">
                    {formatDurationHumanReadable(run.duration_ms)}
                  </span>
                </div>
                <div>
                  Started{" "}
                  <span className="font-medium text-foreground">
                    {formatRelativeTime(run.started_at)}
                  </span>
                </div>
                {run.completed_at ? (
                  <div>
                    Finished{" "}
                    <span className="font-medium text-foreground">
                      {formatRelativeTime(run.completed_at)}
                    </span>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
          <Button onClick={() => onOpenRun(run.run_id)} className="w-full sm:w-auto">
            Open run
            <ArrowRight className="ml-2 size-4" />
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="flex flex-col items-center justify-center gap-3 py-12 text-center">
        <Layers className="size-10 text-muted-foreground opacity-40" />
        <div className="space-y-1">
          <p className="text-sm font-medium text-foreground">No runs yet</p>
          <p className="max-w-sm text-xs text-muted-foreground">
            Trigger a workflow from your agent or CLI. Completed and active runs will appear here.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onViewRunsList}>
          View runs
        </Button>
      </CardContent>
    </Card>
  );
}

// ─── failures: production-style attention ─────────────────────────────────────

interface FailureReasonerBarsProps {
  rows: [string, number][];
}

function FailureReasonerBars({ rows }: FailureReasonerBarsProps) {
  const max = Math.max(...rows.map(([, c]) => c), 1);
  return (
    <div className="space-y-3" aria-label="Failure counts by reasoner">
      <p className="text-xs font-medium text-muted-foreground">By reasoner</p>
      <div className="space-y-2.5">
        {rows.map(([name, count], idx) => (
          <div key={`${name}-${idx}-bar`} className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="min-w-0 truncate text-foreground">{name}</span>
              <span className="shrink-0 tabular-nums text-muted-foreground">{count}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-destructive transition-all"
                style={{ width: `${(count / max) * 100}%` }}
                role="presentation"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


interface FailuresAttentionProps {
  failures: WorkflowSummary[];
  topFailingReasoners: [string, number][];
  onOpenRun: (runId: string) => void;
  onViewAllRuns: () => void;
}

function FailuresAttention({
  failures,
  topFailingReasoners,
  onOpenRun,
  onViewAllRuns,
}: FailuresAttentionProps) {
  if (failures.length === 0) return null;

  const preview = failures.slice(0, 5);

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="text-base font-semibold">Needs attention</CardTitle>
          <CardDescription>
            Failed or timed-out runs in the recent window. Open one to inspect errors in the DAG.
          </CardDescription>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="gap-1.5 text-muted-foreground hover:text-foreground"
          onClick={onViewAllRuns}
        >
          All runs
          <ArrowRight className="size-3.5" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        {topFailingReasoners.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {topFailingReasoners.map(([name, count], idx) => (
              <Badge key={`${name}-${count}-${idx}`} variant="outline" className="font-normal">
                <span className="max-w-[12rem] truncate">{name}</span>
                <span className="ml-1.5 tabular-nums text-muted-foreground">{count}</span>
              </Badge>
            ))}
          </div>
        ) : null}
        {topFailingReasoners.length > 0 ? (
          <FailureReasonerBars rows={topFailingReasoners} />
        ) : null}
        <ul className="space-y-2">
          {preview.map((run) => (
            <li
              key={`${run.run_id}-${run.started_at}-fail`}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border px-3 py-2"
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className="font-mono text-xs text-muted-foreground"
                    title={run.run_id}
                  >
                    {shortRunId(run.run_id)}
                  </span>
                  <RunStatusBadge status={run.root_execution_status ?? run.status} />
                </div>
                <p className="truncate text-xs text-muted-foreground">
                  {run.root_reasoner || run.display_name || "—"} ·{" "}
                  {formatRelativeTime(run.latest_activity)}
                </p>
              </div>
              <Button variant="secondary" size="sm" onClick={() => onOpenRun(run.run_id)}>
                Open
              </Button>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

// ─── recent runs table (secondary) ───────────────────────────────────────────

interface RecentRunsTableProps {
  runs: WorkflowSummary[];
  loading: boolean;
  onRowClick: (runId: string) => void;
}

function RecentRunsTable({ runs, loading, onRowClick }: RecentRunsTableProps) {
  if (loading) {
    return (
      <div className="space-y-1.5 p-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-7 w-full" />
        ))}
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
        <CheckCircle className="mb-2 size-7 opacity-40" />
        <p className="text-xs">No runs yet</p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="h-8 w-[110px] px-3 text-xs">Run</TableHead>
          <TableHead className="h-8 px-3 text-xs">Reasoner</TableHead>
          <TableHead className="h-8 w-[60px] px-3 text-right text-xs">Steps</TableHead>
          <TableHead className="h-8 w-[100px] px-3 text-xs">Status</TableHead>
          <TableHead className="h-8 w-[80px] px-3 text-right text-xs">Duration</TableHead>
          <TableHead className="h-8 w-[90px] px-3 text-right text-xs">Started</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow
            key={`${run.run_id}-${run.started_at}`}
            className="cursor-pointer"
            onClick={() => onRowClick(run.run_id)}
          >
            <TableCell className="px-3 py-1.5 font-mono text-xs text-muted-foreground">
              <span title={run.run_id}>{shortRunId(run.run_id)}</span>
            </TableCell>
            <TableCell className="max-w-[200px] truncate px-3 py-1.5 text-xs font-medium">
              {run.root_reasoner || run.display_name || "—"}
            </TableCell>
            <TableCell className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">
              {run.total_executions ?? "—"}
            </TableCell>
            <TableCell className="px-3 py-1.5">
              <RunStatusBadge status={run.root_execution_status ?? run.status} />
            </TableCell>
            <TableCell className="px-3 py-1.5 text-right font-mono text-xs text-muted-foreground tabular-nums">
              {isTerminalStatus(run.root_execution_status ?? run.status) ? (
                formatDurationHumanReadable(run.duration_ms)
              ) : (
                <LiveElapsedDuration
                  startedAt={run.started_at}
                  className="text-muted-foreground"
                />
              )}
            </TableCell>
            <TableCell className="px-3 py-1.5 text-right text-xs text-muted-foreground">
              {formatRelativeTime(run.started_at)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ─── page ─────────────────────────────────────────────────────────────────────

export function NewDashboardPage() {
  const navigate = useNavigate();
  const { execConnected } = useSSESync();

  const runsQuery = useRuns({
    timeRange: "all",
    pageSize: 100,
    sortBy: "latest_activity",
    sortOrder: "desc",
    refetchInterval: 8_000,
  });

  const llmHealthQuery = useLLMHealth();
  const queueQuery = useQueueStatus();
  const agentsQuery = useAgents({ refetchInterval: 10_000 });

  const summaryQuery = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: getDashboardSummary,
    refetchInterval: execConnected ? 30_000 : 15_000,
  });

  const unhealthyEndpoints =
    llmHealthQuery.data?.endpoints
      ?.filter((ep) => !ep.healthy)
      .map((ep) => ep.name) ?? [];

  const overloadedAgents = Object.entries(queueQuery.data?.agents ?? {})
    .filter(([, s]) => s.running >= s.max_concurrent && s.max_concurrent > 0)
    .map(([name]) => name);

  const hasIssues = unhealthyEndpoints.length > 0 || overloadedAgents.length > 0;

  const totalRuns = summaryQuery.data?.executions?.today ?? runsQuery.data?.total_count;
  const successRate = summaryQuery.data?.success_rate;
  const agentsOnline =
    agentsQuery.data?.nodes?.filter(
      (n: AgentNodeSummary) => n.health_status === "ready" || n.health_status === "active",
    ).length ??
    agentsQuery.data?.count ??
    summaryQuery.data?.agents?.running;

  const recentRuns = useMemo(
    () => runsQuery.data?.workflows ?? [],
    [runsQuery.data?.workflows],
  );

  const { active, latestCompleted, failures, topFailingReasoners } = useMemo(
    () => partitionDashboardRuns(recentRuns),
    [recentRuns],
  );

  const avgDuration = useMemo(() => {
    const completed = recentRuns.filter((r) => r.duration_ms != null && r.terminal);
    if (completed.length === 0) return null;
    const avg =
      completed.reduce((sum, r) => sum + (r.duration_ms ?? 0), 0) / completed.length;
    return formatDurationHumanReadable(avg);
  }, [recentRuns]);

  const statsLoading =
    (summaryQuery.isLoading && runsQuery.isLoading) || agentsQuery.isLoading;

  const tablePreviewRuns = useMemo(() => recentRuns.slice(0, 8), [recentRuns]);

  return (
    <div className="flex flex-col gap-6">
      {hasIssues && (
        <IssuesBanner
          llmHealthLoading={llmHealthQuery.isLoading}
          unhealthyEndpoints={unhealthyEndpoints}
          queueOverloaded={overloadedAgents.length > 0}
          overloadedAgents={overloadedAgents}
        />
      )}

      <PrimaryRunFocus
        loading={runsQuery.isLoading}
        active={active}
        latestCompleted={latestCompleted}
        onOpenRun={(runId) => navigate(`/runs/${runId}`)}
        onViewRunsList={() => navigate("/runs")}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <DashboardRunOutcomeStrip
          className={active.length > 0 ? "lg:col-span-2" : "lg:col-span-3"}
          runs={recentRuns}
          loading={runsQuery.isLoading}
          onSelectRun={(runId) => navigate(`/runs/${runId}`)}
        />
        {active.length > 0 ? (
          <DashboardActiveWorkload activeRuns={active} className="lg:col-span-1" />
        ) : null}
      </div>

      <FailuresAttention
        failures={failures}
        topFailingReasoners={topFailingReasoners}
        onOpenRun={(runId) => navigate(`/runs/${runId}`)}
        onViewAllRuns={() => navigate("/runs")}
      />

      {statsLoading ? (
        <div className="flex flex-wrap items-center gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-8 w-24" />
          ))}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <div className="flex items-center gap-1.5">
            <span className="text-2xl font-semibold tabular-nums">{totalRuns ?? "—"}</span>
            <span className="text-muted-foreground">runs today</span>
          </div>
          <Separator orientation="vertical" className="hidden h-6 sm:block" />
          <div className="flex items-center gap-1.5">
            <span className="text-2xl font-semibold tabular-nums">
              {successRate != null ? `${(successRate * 100).toFixed(0)}%` : "—"}
            </span>
            <span className="text-muted-foreground">success</span>
          </div>
          <Separator orientation="vertical" className="hidden h-6 sm:block" />
          <div className="flex items-center gap-1.5">
            <span className="text-2xl font-semibold tabular-nums">{agentsOnline ?? "—"}</span>
            <span className="text-muted-foreground">agents online</span>
          </div>
          <Separator orientation="vertical" className="hidden h-6 sm:block" />
          <div className="flex items-center gap-1.5">
            <span className="text-2xl font-semibold tabular-nums">{avgDuration ?? "—"}</span>
            <span className="text-muted-foreground">avg time</span>
          </div>
        </div>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between px-4 py-3">
          <CardTitle className="text-sm font-medium">Recent runs</CardTitle>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-muted-foreground hover:text-foreground"
            onClick={() => navigate("/runs")}
          >
            View all
            <ArrowRight className="size-3.5" />
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <RecentRunsTable
            runs={tablePreviewRuns}
            loading={runsQuery.isLoading}
            onRowClick={(runId) => navigate(`/runs/${runId}`)}
          />
        </CardContent>
      </Card>
    </div>
  );
}

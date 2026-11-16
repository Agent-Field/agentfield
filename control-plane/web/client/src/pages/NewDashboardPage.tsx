import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CheckCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/ErrorState";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  formatDurationHumanReadable,
  LiveElapsedDuration,
} from "@/components/ui/data-formatters";
import { shortRunIdForDashboard as shortRunId } from "@/components/dashboard/dashboardRunUtils";
import {
  HomeAlerts,
  HomeCallMap,
  HomeEndpoints,
  HomeFigures,
  HomeFirstAgent,
  HomeInOut,
  HomeProblems,
  HomeStatusLine,
  HomeWaiting,
} from "@/components/home/HomeSections";
import {
  HOME_WINDOWS,
  buildAlerts,
  buildFigures,
  buildStatusLine,
  isEmptyInstall,
  windowPhrase,
} from "@/components/home/homeModel";

import { useLLMHealth, useQueueStatus, useRuns } from "@/hooks/queries";
import { useSSESync } from "@/hooks/useSSEQuerySync";
import { getDeveloperHome } from "@/services/dashboardService";
import { formatRelativeTime } from "@/utils/dateFormat";
import { getStatusTheme, isTerminalStatus } from "@/utils/status";
import type { HomeWindowPreset } from "@/types/developerHome";
import type { WorkflowSummary } from "@/types/workflows";

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
          <TableHead className="h-8 w-[110px] px-3 text-right text-xs">Started</TableHead>
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
            <TableCell className="whitespace-nowrap px-3 py-1.5 text-right text-xs text-muted-foreground">
              {formatRelativeTime(run.started_at)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ─── page ─────────────────────────────────────────────────────────────────────

const WINDOW_STORAGE_KEY = "af_home_window";

function readStoredWindow(): HomeWindowPreset {
  try {
    const v = window.localStorage.getItem(WINDOW_STORAGE_KEY);
    if (v && HOME_WINDOWS.some((w) => w.value === v)) return v as HomeWindowPreset;
  } catch {
    // Storage can be unavailable; the default window is fine.
  }
  return "1h";
}

function HomeSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true">
      <Skeleton className="h-12 w-80" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

export function NewDashboardPage() {
  const navigate = useNavigate();
  const { execConnected } = useSSESync();
  const [preset, setPreset] = useState<HomeWindowPreset>(readStoredWindow);

  const homeQuery = useQuery({
    queryKey: ["dashboard-home", preset],
    queryFn: () => getDeveloperHome(preset),
    // Live execution events refresh this through a coalesced invalidation;
    // polling covers agents going silent, which emits no execution event.
    refetchInterval: execConnected ? 30_000 : 10_000,
    placeholderData: (prev) => prev,
  });

  const runsQuery = useRuns({
    timeRange: "all",
    pageSize: 10,
    sortBy: "latest_activity",
    sortOrder: "desc",
    refetchInterval: execConnected ? 30_000 : 10_000,
  });
  const llmHealthQuery = useLLMHealth();
  const queueQuery = useQueueStatus();

  const now = useMemo(
    () => (homeQuery.data ? new Date(homeQuery.data.generated_at) : new Date()),
    [homeQuery.data],
  );

  const view = useMemo(() => {
    const home = homeQuery.data;
    if (!home) return null;
    const alerts = buildAlerts(home, { llm: llmHealthQuery.data, queue: queueQuery.data }, now);
    return {
      home,
      alerts,
      status: buildStatusLine(home, alerts),
      figures: buildFigures(home, now),
      phrase: windowPhrase(home.window.preset),
    };
  }, [homeQuery.data, llmHealthQuery.data, queueQuery.data, now]);

  const changeWindow = (value: string) => {
    const next = value as HomeWindowPreset;
    setPreset(next);
    try {
      window.localStorage.setItem(WINDOW_STORAGE_KEY, next);
    } catch {
      // Remembering the window is a convenience only.
    }
  };

  if (homeQuery.isLoading && !homeQuery.data) return <HomeSkeleton />;
  if (homeQuery.isError && !homeQuery.data) {
    return (
      <ErrorState
        title="Home could not load"
        description={homeQuery.error instanceof Error ? homeQuery.error.message : undefined}
        onRetry={() => void homeQuery.refetch()}
      />
    );
  }
  if (!view) return <HomeSkeleton />;

  const runs = runsQuery.data?.workflows ?? [];
  if (isEmptyInstall(view.home, runsQuery.data?.total_count ?? runs.length)) {
    return (
      <div className="flex flex-col gap-6 py-6">
        <HomeFirstAgent origin={window.location.origin} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <HomeStatusLine status={view.status} />
        <SegmentedControl
          size="sm"
          value={preset}
          onValueChange={changeWindow}
          options={HOME_WINDOWS}
          className="shrink-0"
        />
      </div>

      <HomeAlerts alerts={view.alerts} />
      <HomeFigures figures={view.figures} />
      <HomeCallMap home={view.home} phrase={view.phrase} />
      <HomeProblems problems={view.home.problems} phrase={view.phrase} now={now} />
      <HomeEndpoints endpoints={view.home.endpoints} phrase={view.phrase} />
      <HomeWaiting home={view.home} now={now} />
      <HomeInOut home={view.home} phrase={view.phrase} />

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 px-4 py-3">
          <CardTitle className="text-sm font-medium">Recent runs</CardTitle>
          <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
            <Link to="/runs">
              View all
              <ArrowRight className="size-3.5" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <RecentRunsTable
            runs={runs}
            loading={runsQuery.isLoading}
            onRowClick={(runId) => navigate(`/runs/${runId}`)}
          />
        </CardContent>
      </Card>
    </div>
  );
}

import  { useMemo } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  cardVariants,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { TrendMetricCard } from "@/components/ui/TrendMetricCard";
import { ErrorState } from "@/components/ui/ErrorState";
import { PageHeader } from "../components/PageHeader";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { useEnhancedDashboard } from "@/hooks/useEnhancedDashboard";
import { useDashboardTimeRange } from "@/hooks/useDashboardTimeRange";
import { TimeRangeSelector } from "@/components/dashboard/TimeRangeSelector";
import { HotspotPanel } from "@/components/dashboard/HotspotPanel";
import { ActivityHeatmap } from "@/components/dashboard/ActivityHeatmap";
import type {
  EnhancedDashboardResponse,
  ExecutionTrendPoint,
  WorkflowStat,
  ActiveWorkflowRun,
  CompletedExecutionStat,
  IncidentItem,
  ComparisonData,
} from "@/types/dashboard";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Activity,
  RefreshCw,
  Gauge,
  Users,
  Zap,
  AlertTriangle,
  Timer,
  GitCommit,
  Cpu,
  Server,
} from "@/components/ui/icon-bridge";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  Tooltip,
} from "recharts";

const numberFormatter = new Intl.NumberFormat("en-US");
const decimalFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const formatPercentage = (value: number | undefined, digits = 1) => {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
};

const formatDuration = (value: number | undefined) => {
  if (!value || value <= 0) return "—";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)} s`;
  if (value < 3600000) return `${(value / 60000).toFixed(1)} m`;
  return `${(value / 3600000).toFixed(1)} h`;
};

const formatTimestamp = (value?: string) => {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
};

export function EnhancedDashboardPage() {
  // Time range state with URL persistence
  const {
    timeRange,
    setPreset,
    toggleCompare,
    getApiParams,
    label: timeRangeLabel,
  } = useDashboardTimeRange("24h");

  // Get API params from time range state
  const apiParams = getApiParams();

  // Fetch dashboard data with time range params
  const { data, loading, error, hasError, refresh, clearError, isRefreshing } =
    useEnhancedDashboard({
      preset: apiParams.preset,
      startTime: apiParams.startTime,
      endTime: apiParams.endTime,
      compare: apiParams.compare,
    });

  const reasonerStats = useMemo<ReasonerSummary[]>(() => {
    if (!data) {
      return [];
    }

    const agentMeta = new Map(
      data.agent_health.agents.map((agent) => [agent.id, agent])
    );

    const ensureEntry = (
      map: Map<string, ReasonerAccumulator>,
      reasonerId: string
    ) => {
      let entry = map.get(reasonerId);
      if (!entry) {
        entry = {
          reasonerId,
          activeRuns: 0,
          incidentCount: 0,
          agentIds: new Set<string>(),
        };
        map.set(reasonerId, entry);
      }
      return entry;
    };

    const accumulator = new Map<string, ReasonerAccumulator>();

    data.workflows.active_runs.forEach((run) => {
      if (!run.reasoner_id) {
        return;
      }
      const entry = ensureEntry(accumulator, run.reasoner_id);
      entry.activeRuns += 1;
      if (run.agent_node_id) {
        entry.agentIds.add(run.agent_node_id);
      }
    });

    data.incidents.forEach((incident) => {
      if (!incident.reasoner_id) {
        return;
      }
      const entry = ensureEntry(accumulator, incident.reasoner_id);
      entry.incidentCount += 1;
      if (incident.agent_node_id) {
        entry.agentIds.add(incident.agent_node_id);
      }
    });

    const summaries: ReasonerSummary[] = Array.from(accumulator.values()).map(
      (entry) => {
        const agentDetails = Array.from(entry.agentIds).map((agentId) => {
          const meta = agentMeta.get(agentId);
          return {
            id: agentId,
            status: meta ? meta.status : "unknown",
            lastHeartbeat: meta ? meta.last_heartbeat : undefined,
          };
        });

        const status =
          entry.activeRuns > 0
            ? "active"
            : entry.incidentCount > 0
              ? "attention"
              : "idle";

        return {
          reasonerId: entry.reasonerId,
          activeRuns: entry.activeRuns,
          incidentCount: entry.incidentCount,
          agents: agentDetails,
          status,
        } as ReasonerSummary;
      }
    );

    summaries.sort((a, b) => {
      if (a.status === b.status) {
        if (b.activeRuns === a.activeRuns) {
          return b.incidentCount - a.incidentCount;
        }
        return b.activeRuns - a.activeRuns;
      }
      const order = { active: 0, attention: 1, idle: 2 } as const;
      return order[a.status] - order[b.status];
    });

    return summaries;
  }, [data]);

  if (loading && !data) {
    return (
      <div className="space-y-8">
        <PageHeader
          title="Enhanced Dashboard"
          description="Real-time observability for distributed agent networks."
          aside={
            <div className="flex gap-4">
              <Skeleton className="h-10 w-36" />
              <Skeleton className="h-10 w-40" />
            </div>
          }
        />

        <ResponsiveGrid variant="dashboard">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton
              key={index}
              className="h-32 rounded-xl border border-border/40"
            />
          ))}
        </ResponsiveGrid>

        <ResponsiveGrid columns={{ base: 1, xl: 3 }} gap="lg">
          <ResponsiveGrid.Item span={{ xl: 2 }}>
            <Skeleton className="h-80 rounded-xl border border-border/40" />
          </ResponsiveGrid.Item>
          <ResponsiveGrid.Item className="space-y-8">
            <Skeleton className="h-56 rounded-xl border border-border/40" />
            <Skeleton className="h-64 rounded-xl border border-border/40" />
          </ResponsiveGrid.Item>
        </ResponsiveGrid>
      </div>
    );
  }

  if (hasError && !data) {
    return (
      <div className="space-y-8">
        <PageHeader
          title="Enhanced Dashboard"
          description="Real-time observability for distributed agent networks."
          aside={
            <Link to="/dashboard">
              <Button variant="ghost">Switch to classic view</Button>
            </Link>
          }
        />

        <ErrorState
          title="Failed to load dashboard data"
          description="An unexpected error occurred while fetching the enhanced dashboard."
          error={error?.message}
          onRetry={refresh}
          onDismiss={clearError}
          retrying={isRefreshing}
          variant="card"
          severity="error"
        />
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <PageHeader
        title="Dashboard"
        description="Monitor agent health, workflow performance, and system throughput across your distributed cluster."
        aside={
          <div className="flex flex-wrap items-center gap-3">
            <TimeRangeSelector
              value={timeRange.preset}
              onChange={setPreset}
              compare={timeRange.compare}
              onCompareChange={toggleCompare}
            />
            <div className="flex items-center gap-2">
              <Badge variant="pill" size="sm" className="font-mono">
                {timeRangeLabel}
              </Badge>
              <Button
                onClick={refresh}
                variant="outline"
                size="sm"
                disabled={isRefreshing}
              >
                <RefreshCw
                  className={cn("h-3 w-3", isRefreshing && "animate-spin")}
                />
                {isRefreshing ? "Refreshing" : "Refresh"}
              </Button>
            </div>
          </div>
        }
      />

      {hasError && (
        <ErrorState
          title="Unable to refresh data"
          description={`Showing cached data. ${error?.message}`}
          onDismiss={clearError}
          variant="banner"
          severity="warning"
        />
      )}

      <div className="animate-slide-in" style={{ animationDelay: "50ms" }}>
        <OverviewStrip
          overview={data.overview}
          trends={data.execution_trends.last_24h}
          trendPoints={data.execution_trends.last_7_days}
          comparison={data.comparison}
        />
      </div>

      {/* Row 1: Trend Analysis - Execution Trends + Activity Patterns */}
      <ResponsiveGrid
        columns={{ base: 1, md: 6, lg: 12 }}
        flow="dense"
        className="auto-rows-[minmax(220px,auto)] md:auto-rows-[minmax(280px,auto)]"
        gap="md"
      >
        <ResponsiveGrid.Item span={{ md: 6, lg: 7, xl: 8, "2xl": 7 }} className="animate-slide-in" style={{ animationDelay: "100ms" }}>
          <ExecutionTrendsCard
            trendPoints={data.execution_trends.last_7_days}
          />
        </ResponsiveGrid.Item>
        <ResponsiveGrid.Item span={{ md: 6, lg: 5, xl: 4, "2xl": 5 }} className="animate-slide-in" style={{ animationDelay: "150ms" }}>
          <ActivityHeatmap
            heatmapData={data.activity_patterns?.hourly_heatmap || []}
            className="h-full"
          />
        </ResponsiveGrid.Item>
      </ResponsiveGrid>

      {/* Row 2: Problem Diagnosis - Hotspots + Incidents */}
      <ResponsiveGrid
        columns={{ base: 1, lg: 2 }}
        gap="md"
        className="animate-slide-in"
        style={{ animationDelay: "200ms" }}
      >
        <ResponsiveGrid.Item>
          <HotspotPanel
            hotspots={data.hotspots?.top_failing_reasoners || []}
            className="h-[300px]"
          />
        </ResponsiveGrid.Item>
        <ResponsiveGrid.Item>
          <IncidentPanel incidents={data.incidents} className="h-[300px]" />
        </ResponsiveGrid.Item>
      </ResponsiveGrid>

      {/* Row 3: Detailed drill-downs - Workflow Insights + Reasoner Activity */}
      <ResponsiveGrid
        columns={{ base: 1, md: 6, lg: 12 }}
        flow="dense"
        className="auto-rows-[minmax(260px,auto)]"
        gap="md"
      >
        <ResponsiveGrid.Item span={{ md: 6, lg: 7, xl: 6, "2xl": 7 }} className="animate-slide-in" style={{ animationDelay: "250ms" }}>
          <WorkflowInsightsPanel insights={data.workflows} />
        </ResponsiveGrid.Item>
        <ResponsiveGrid.Item span={{ md: 6, lg: 5, xl: 6, "2xl": 5 }} className="animate-slide-in" style={{ animationDelay: "300ms" }}>
          <ReasonerActivityPanel
            reasoners={reasonerStats}
            agentSummary={data.agent_health}
          />
        </ResponsiveGrid.Item>
      </ResponsiveGrid>
    </div>
  );
}

interface OverviewStripProps {
  overview: EnhancedDashboardResponse["overview"];
  trends: EnhancedDashboardResponse["execution_trends"]["last_24h"];
  trendPoints?: ExecutionTrendPoint[];
  comparison?: ComparisonData;
}

function OverviewStrip({ overview, trends, trendPoints, comparison }: OverviewStripProps) {
  // Extract sparkline data from trend points
  const executionsSparkline = useMemo(
    () => trendPoints?.map((p) => p.total) || [],
    [trendPoints]
  );
  const successSparkline = useMemo(
    () => trendPoints?.map((p) => p.succeeded) || [],
    [trendPoints]
  );

  return (
    <ResponsiveGrid variant="dashboard">
      {/* Agents Online */}
      <TrendMetricCard
        label="Agents online"
        value={`${overview.active_agents}/${overview.total_agents}`}
        subtitle={
          overview.degraded_agents > 0
            ? `${overview.degraded_agents} degraded`
            : `${overview.offline_agents} offline`
        }
        icon={Users}
      />

      {/* Executions */}
      <TrendMetricCard
        label="Executions"
        value={numberFormatter.format(overview.executions_last_24h)}
        currentValue={comparison ? overview.executions_last_24h : undefined}
        previousValue={
          comparison
            ? overview.executions_last_24h - comparison.overview_delta.executions_delta
            : undefined
        }
        trendPolarity="up-is-good"
        sparklineData={executionsSparkline}
        subtitle={`${decimalFormatter.format(trends.throughput_per_hour)} / hr`}
        icon={Activity}
      />

      {/* Success Rate */}
      <TrendMetricCard
        label="Success rate"
        value={formatPercentage(overview.success_rate_24h)}
        currentValue={comparison ? overview.success_rate_24h : undefined}
        previousValue={
          comparison
            ? overview.success_rate_24h - comparison.overview_delta.success_rate_delta
            : undefined
        }
        trendPolarity="up-is-good"
        sparklineData={successSparkline}
        subtitle={`${numberFormatter.format(trends.succeeded)} succeeded`}
        icon={Gauge}
      />

      {/* Avg Duration */}
      <TrendMetricCard
        label="Avg duration"
        value={formatDuration(overview.average_duration_ms_24h)}
        currentValue={comparison ? overview.average_duration_ms_24h : undefined}
        previousValue={
          comparison
            ? overview.average_duration_ms_24h - comparison.overview_delta.avg_duration_delta_ms
            : undefined
        }
        trendPolarity="down-is-good"
        subtitle={`Median ${formatDuration(overview.median_duration_ms_24h)}`}
        icon={Timer}
      />
    </ResponsiveGrid>
  );
}

interface ExecutionTrendsCardProps {
  trendPoints: ExecutionTrendPoint[];
}

function ExecutionTrendsCard({
  trendPoints,
}: ExecutionTrendsCardProps) {
  const chartData = trendPoints.map((point) => ({
    ...point,
    label: new Date(point.date).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
  }));

  return (
    <Card
      variant="surface"
      interactive={false}
      className="flex h-full flex-col"
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 p-5 pb-2">
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4" /> Velocity & reliability
        </CardTitle>
        <Badge variant="pill">Last 7 days</Badge>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col p-5 pt-0">
        <div className="h-full min-h-[180px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 16, right: 16, left: 8, bottom: 8 }}
            >
              <defs>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor="var(--primary)"
                    stopOpacity={0.25}
                  />
                  <stop
                    offset="95%"
                    stopColor="var(--primary)"
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="label"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              />
              <Tooltip
                cursor={{ stroke: "var(--border)", strokeDasharray: 4 }}
                content={({ payload }) => {
                  if (!payload || !payload.length) return null;
                  const datum = payload[0].payload as ExecutionTrendPoint & {
                    label: string;
                  };
                  return (
                    <div className="rounded-md border border-border bg-background px-3 py-2 text-xs shadow-md">
                      <p className="font-medium text-foreground">
                        {datum.label}
                      </p>
                      <p className="text-text-secondary">
                        Total: {datum.total}
                      </p>
                      <p className="text-emerald-500">
                        Succeeded: {datum.succeeded}
                      </p>
                      <p className="text-destructive">Failed: {datum.failed}</p>
                    </div>
                  );
                }}
              />
              <Area
                type="monotone"
                dataKey="total"
                stroke="none"
                fill="url(#areaGradient)"
              />
              <Line
                type="monotone"
                dataKey="succeeded"
                stroke="var(--primary)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="failed"
                stroke="var(--destructive)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

interface WorkflowInsightsPanelProps {
  insights: EnhancedDashboardResponse["workflows"];
}

function ProgressBar({ value, className }: { value: number; className?: string }) {
  const colorClass =
    value >= 95
      ? "bg-emerald-500"
      : value >= 80
        ? "bg-amber-500"
        : "bg-destructive";

  return (
    <div className={cn("h-1.5 w-full rounded-full bg-muted overflow-hidden", className)}>
      <div
        className={cn("h-full rounded-full transition-all duration-500", colorClass)}
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

function WorkflowInsightsPanel({ insights }: WorkflowInsightsPanelProps) {
  return (
    <Card
      variant="surface"
      interactive={false}
      className="flex h-full flex-col"
    >
      <CardHeader className="p-5 pb-2">
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-4 w-4" /> Workflow intelligence
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col min-h-0 p-5 pt-3">
        <ResponsiveGrid
          columns={{ base: 1, lg: 2 }}
          gap="lg"
          align="start"
          className="flex-1 min-h-0"
        >
          {/* Left column: Top Workflows */}
          <div className="flex flex-col h-full gap-3">
            <div className="flex items-center gap-2 text-label flex-shrink-0">
              <GitCommit className="h-3.5 w-3.5" />
              Top workflows
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border/50">
              {insights.top_workflows.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">No executions recorded in the last 7 days.</p>
              ) : (
                insights.top_workflows.map((workflow: WorkflowStat, index: number) => (
                  <Link
                    key={workflow.workflow_id}
                    to={`/workflows/${workflow.workflow_id}/enhanced`}
                    className={cn(
                      "group relative block transition-all hover:border-border hover:bg-muted/30 min-w-0",
                      cardVariants({ variant: "muted", interactive: false }),
                      "pl-10 pr-3 py-2.5"
                    )}
                  >
                    {/* Rank Badge */}
                    <div className="absolute left-3 top-2.5 flex h-5 w-5 items-center justify-center rounded-full bg-background border border-border text-[10px] font-mono font-medium text-muted-foreground shadow-sm group-hover:border-primary/50 group-hover:text-primary transition-colors">
                      {index + 1}
                    </div>

                    <div className="space-y-1.5 min-w-0">
                      <div className="flex justify-between items-start gap-2">
                        <p className="font-medium text-sm text-foreground truncate">
                          {workflow.name || workflow.workflow_id}
                        </p>
                        <span className="text-xs font-mono text-muted-foreground whitespace-nowrap">
                          {numberFormatter.format(workflow.total_executions)} runs
                        </span>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-[10px] text-muted-foreground">
                          <span>Success Rate</span>
                          <span className={cn(
                            "font-mono",
                            workflow.success_rate >= 95 ? "text-emerald-500" : workflow.success_rate >= 80 ? "text-amber-500" : "text-destructive"
                          )}>
                            {formatPercentage(workflow.success_rate)}
                          </span>
                        </div>
                        <ProgressBar value={workflow.success_rate} />
                      </div>
                    </div>
                  </Link>
                ))
              )}
            </div>
          </div>

          {/* Right column: Active Runs + Longest Recent Runs */}
          <div className="flex flex-col h-full gap-6">
            {/* Active Runs Section - fixed height */}
            <div className="space-y-3 flex-shrink-0">
              <div className="flex items-center gap-2 text-label">
                <GitCommit className="h-3.5 w-3.5" />
                Active runs
              </div>
              {insights.active_runs.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">No workflows running right now.</p>
              ) : (
                <div className="space-y-2 max-h-[120px] overflow-y-auto pr-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border/50">
                  {insights.active_runs.map((run: ActiveWorkflowRun) => (
                    <Link
                      key={run.execution_id}
                      to={`/executions/${run.execution_id}`}
                      className={cn(
                        "group block transition-all hover:border-primary/30 hover:shadow-md min-w-0 relative overflow-hidden",
                        cardVariants({ variant: "muted", interactive: false }),
                        "px-3 py-2 text-xs bg-background/50 backdrop-blur-sm border-primary/20"
                      )}
                    >
                      <div className="absolute top-0 left-0 w-0.5 h-full bg-primary/50 group-hover:bg-primary transition-colors" />
                      <div className="flex items-center justify-between min-w-0 gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                          </span>
                          <p className="font-medium text-foreground truncate">
                            {run.name || run.workflow_id}
                          </p>
                        </div>
                        <span className="text-primary font-mono text-[10px] flex-shrink-0 bg-primary/10 px-1.5 py-0.5 rounded-full">
                          {formatDuration(run.elapsed_ms)}
                        </span>
                      </div>
                      <p className="mt-1 pl-4 text-muted-foreground truncate font-mono text-[10px]">
                        {run.execution_id}
                      </p>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* Longest Recent Runs Section - fills remaining space */}
            <div className="flex flex-col flex-1 min-h-0 gap-3">
              <div className="flex items-center gap-2 text-label flex-shrink-0">
                <GitCommit className="h-3.5 w-3.5" />
                Longest recent runs
              </div>
              {insights.longest_executions.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">Insufficient completed runs.</p>
              ) : (
                <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border/50">
                  {insights.longest_executions.map((execution: CompletedExecutionStat) => (
                    <div
                      key={execution.execution_id}
                      className={cn(
                        cardVariants({ variant: "muted", interactive: false }),
                        "px-3 py-2 text-xs min-w-0"
                      )}
                    >
                      <div className="flex justify-between items-center gap-2">
                        <p className="font-medium text-foreground truncate">
                          {execution.name || execution.workflow_id}
                        </p>
                        <span className={cn(
                          "font-mono text-[10px] px-1.5 py-0.5 rounded-full flex-shrink-0",
                          execution.duration_ms > 60000 ? "bg-amber-500/10 text-amber-600" : "bg-muted text-muted-foreground"
                        )}>
                          {formatDuration(execution.duration_ms)}
                        </span>
                      </div>
                      <p className="mt-1 text-muted-foreground truncate text-[10px]">
                        Completed {formatTimestamp(execution.completed_at)}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </ResponsiveGrid>
      </CardContent>
    </Card>
  );
}

interface IncidentPanelProps {
  incidents: IncidentItem[];
  className?: string;
}

function IncidentPanel({ incidents, className }: IncidentPanelProps) {
  return (
    <Card
      variant="surface"
      interactive={false}
      className={cn("flex h-full flex-col", className)}
    >
      <CardHeader className="p-5 pb-2">
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-destructive" /> Incident log
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 p-5 pt-0">
        <div className="flex items-center justify-between text-body">
          <span className="text-text-secondary">
            {incidents.length} issues in the last 7 days
          </span>
          {incidents.length > 0 && (
            <Badge
              variant="outline"
              className="rounded-full border-destructive/40 text-destructive bg-transparent"
            >
              Attention
            </Badge>
          )}
        </div>
        {incidents.length === 0 ? (
          <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border/40 bg-muted/10 p-4 text-center text-body-small">
            No failures or cancellations detected in the last 7 days.
          </div>
        ) : (
          <div className="h-[250px] overflow-hidden">
            <div className="max-h-[250px] space-y-4 overflow-y-auto pr-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border/70">
              {incidents.map((incident) => (
                <Link
                  key={incident.execution_id}
                  to={`/executions/${incident.execution_id}`}
                  className={cn(
                    "block transition-colors hover:border-border hover:bg-muted/20",
                    cardVariants({ variant: "muted", interactive: false }),
                    "px-3 py-3 text-xs"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-destructive"></div>
                      <p className="font-medium text-foreground">
                        {incident.name || incident.workflow_id}
                      </p>
                    </div>
                    <span className="rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-destructive">
                      {incident.status}
                    </span>
                  </div>
                  <p className="ml-4 mt-1 text-text-secondary">
                    {incident.execution_id} · {incident.reasoner_id}
                  </p>
                  {incident.error && (
                    <p className="ml-4 mt-2 line-clamp-2 text-body-small text-destructive/80">
                      {incident.error}
                    </p>
                  )}
                  <p className="ml-4 mt-2 text-[10px] text-text-tertiary">
                    Started {formatTimestamp(incident.started_at)}
                  </p>
                </Link>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface ReasonerSummary {
  reasonerId: string;
  activeRuns: number;
  incidentCount: number;
  agents: Array<{
    id: string;
    status: string;
    lastHeartbeat?: string;
  }>;
  status: "active" | "attention" | "idle";
}

interface ReasonerAccumulator {
  reasonerId: string;
  activeRuns: number;
  incidentCount: number;
  agentIds: Set<string>;
}

interface ReasonerActivityPanelProps {
  reasoners: ReasonerSummary[];
  agentSummary: EnhancedDashboardResponse["agent_health"];
}

function ReasonerActivityPanel({
  reasoners,
  agentSummary,
}: ReasonerActivityPanelProps) {
  return (
    <Card variant="surface" interactive={false} className="flex h-full flex-col">
      <CardHeader className="space-y-4 p-5 pb-2">
        <CardTitle className="flex items-center gap-2">
          <Cpu className="h-4 w-4" /> Reasoner activity
        </CardTitle>
        <div className="grid grid-cols-3 gap-2 text-center text-body-small uppercase tracking-wide text-text-tertiary">
          <StatusCounter
            label="Active agents"
            value={agentSummary.active}
            tone="success"
          />
          <StatusCounter
            label="Degraded"
            value={agentSummary.degraded}
            tone="warning"
          />
          <StatusCounter
            label="Offline"
            value={agentSummary.offline}
            tone="destructive"
          />
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 min-h-0 p-5 pt-0">
        {reasoners.length === 0 ? (
          <p className="text-body-small">
            No recent reasoner activity. Trigger a workflow or execution to
            populate this view.
          </p>
        ) : (
          <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-4">
            {reasoners.map((reasoner) => (
              <ReasonerRow key={reasoner.reasonerId} reasoner={reasoner} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ReasonerRow({ reasoner }: { reasoner: ReasonerSummary }) {
  return (
    <div
      className={cn(
        cardVariants({ variant: "muted", interactive: false }),
        "px-3 py-3 text-xs"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Badge variant="metadata" className="inline-flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {reasoner.reasonerId}
          </Badge>
          <StatusIndicator
            status={reasoner.status}
            label={
              reasoner.status === "active"
                ? "Active"
                : reasoner.status === "attention"
                  ? "Needs attention"
                  : "Idle"
            }
            size="sm"
            variant="subtle"
          />
        </div>
        <div className="flex items-center gap-4 text-[10px] text-text-tertiary">
          <span>{reasoner.activeRuns} running</span>
          <span>{reasoner.incidentCount} incidents</span>
        </div>
      </div>

      {reasoner.agents.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {reasoner.agents.map((agent) => (
            <span
              key={agent.id}
              className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-muted/30 px-2 py-1 text-[10px]"
            >
              <Server className="h-3 w-3" />
              {agent.id}
              {agent.lastHeartbeat && (
                <span className="text-[9px] text-text-tertiary">
                  · {formatTimestamp(agent.lastHeartbeat)}
                </span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface StatusCounterProps {
  label: string;
  value: number;
  tone: "success" | "warning" | "destructive";
}

function StatusCounter({ label, value, tone }: StatusCounterProps) {
  const toneClass =
    tone === "success"
      ? "text-emerald-500"
      : tone === "warning"
        ? "text-amber-500"
        : "text-destructive";

  return (
    <div className="rounded-xl border border-border/40 bg-muted/30 px-2 py-2">
      <p className="text-[10px] text-text-tertiary">{label}</p>
      <p className={cn("mt-1 text-base font-semibold", toneClass)}>{value}</p>
    </div>
  );
}

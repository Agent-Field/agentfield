import { useMemo } from "react";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Zap, GaugeCircle, Timer, Activity } from "@/components/ui/icon-bridge";
import { ExecutionScatterPlot } from "./ExecutionScatterPlot";
import { AgentHealthHeatmap } from "./AgentHealthHeatmap";
import type { WorkflowSummary, WorkflowTimelineNode } from "../../types/workflows";
import { normalizeExecutionStatus, getStatusLabel } from "../../utils/status";

interface EnhancedWorkflowPerformanceProps {
  workflow: WorkflowSummary;
  dagData?: { timeline?: WorkflowTimelineNode[] } | null;
  selectedNodeIds: string[];
  onNodeSelection: (nodeIds: string[], replace?: boolean) => void;
}

const formatDuration = (value?: number) => {
  if (!value || value <= 0) return '—';
  if (value < 1000) return `${value.toFixed(0)} ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)} s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.floor((value % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
};

const formatTimestamp = (timestamp?: string) => {
  if (!timestamp) return '—';
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
};

const percentile = (values: number[], p: number) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[index];
};

export function EnhancedWorkflowPerformance({
  dagData,
  selectedNodeIds,
  onNodeSelection,
}: EnhancedWorkflowPerformanceProps) {
  const timeline = useMemo<WorkflowTimelineNode[]>(() => {
    return dagData?.timeline ?? [];
  }, [dagData?.timeline]);

  const timedNodes = useMemo(() => {
    return timeline.filter((node) => typeof node.duration_ms === 'number' && Number(node.duration_ms) > 0);
  }, [timeline]);

  const metrics = useMemo(() => {
    if (!timedNodes.length) {
      return {
        totalDuration: 0,
        averageDuration: 0,
        medianDuration: 0,
        p95Duration: 0,
        maxDuration: 0,
        durations: [] as number[],
        spanMs: 0,
      };
    }

    const durations = timedNodes.map((node) => Number(node.duration_ms));
    const totalDuration = durations.reduce((sum, value) => sum + value, 0);
    const averageDuration = totalDuration / durations.length;
    const medianDuration = percentile(durations, 50);
    const p95Duration = percentile(durations, 95);
    const maxDuration = Math.max(...durations);

    const times = timedNodes
      .map((node) => new Date(node.started_at).getTime())
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b);

    const spanMs = times.length >= 2 ? times[times.length - 1] - times[0] : 0;

    return { totalDuration, averageDuration, medianDuration, p95Duration, maxDuration, durations, spanMs };
  }, [timedNodes]);

  const statusSummary = useMemo(() => {
    return timeline.reduce(
      (acc, node) => {
        const normalized = normalizeExecutionStatus(node.status);
        acc.total += 1;
        acc.byStatus[normalized] = (acc.byStatus[normalized] || 0) + 1;
        return acc;
      },
      { total: 0, byStatus: {} as Record<string, number> }
    );
  }, [timeline]);

  const throughputPerMinute = useMemo(() => {
    if (!metrics.spanMs || metrics.spanMs <= 0) {
      return timedNodes.length;
    }
    const minutes = metrics.spanMs / 60000;
    return minutes > 0 ? timedNodes.length / minutes : timedNodes.length;
  }, [metrics.spanMs, timedNodes.length]);

  const bottlenecks = useMemo(() => {
    return [...timedNodes]
      .sort((a, b) => Number(b.duration_ms) - Number(a.duration_ms))
      .slice(0, 5);
  }, [timedNodes]);

  const totalTimedDuration = metrics.totalDuration || 1;

  const agentPerformance = useMemo(() => {
    const map = new Map<string, { totalDuration: number; count: number; maxDuration: number }>();

    timedNodes.forEach((node) => {
      const key = node.agent_name || node.reasoner_id || 'Unknown';
      const duration = Number(node.duration_ms) || 0;
      const entry = map.get(key) || { totalDuration: 0, count: 0, maxDuration: 0 };
      entry.totalDuration += duration;
      entry.count += 1;
      entry.maxDuration = Math.max(entry.maxDuration, duration);
      map.set(key, entry);
    });

    return Array.from(map.entries())
      .map(([name, value]) => ({
        name,
        averageDuration: value.totalDuration / value.count,
        totalDuration: value.totalDuration,
        maxDuration: value.maxDuration,
        count: value.count,
      }))
      .sort((a, b) => b.totalDuration - a.totalDuration)
      .slice(0, 5);
  }, [timedNodes]);

  const depthPerformance = useMemo(() => {
    const map = new Map<number, { totalDuration: number; count: number }>();
    timedNodes.forEach((node) => {
      if (typeof node.workflow_depth !== 'number') return;
      const entry = map.get(node.workflow_depth) || { totalDuration: 0, count: 0 };
      entry.totalDuration += Number(node.duration_ms) || 0;
      entry.count += 1;
      map.set(node.workflow_depth, entry);
    });

    return Array.from(map.entries())
      .map(([depth, value]) => ({
        depth,
        averageDuration: value.totalDuration / value.count,
        count: value.count,
      }))
      .sort((a, b) => a.depth - b.depth);
  }, [timedNodes]);

  const selectedNodes = useMemo(() => {
    return timeline.filter((node) => selectedNodeIds.includes(node.execution_id));
  }, [timeline, selectedNodeIds]);

  const selectedMetrics = useMemo(() => {
    const timed = selectedNodes.filter((node) => typeof node.duration_ms === 'number');
    const total = timed.reduce((sum, node) => sum + Number(node.duration_ms), 0);
    return {
      count: selectedNodes.length,
      timedCount: timed.length,
      totalDuration: total,
      averageDuration: timed.length ? total / timed.length : 0,
      successCount: selectedNodes.filter((node) => normalizeExecutionStatus(node.status) === 'succeeded').length,
    };
  }, [selectedNodes]);


  const hasPerformanceData = timedNodes.length > 0;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between gap-4 p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-body-small">
            {timedNodes.length} timed steps
          </Badge>
          <Badge variant="outline" className="text-body-small">
            Throughput {throughputPerMinute.toFixed(1)} / min
          </Badge>
        </div>
        <div className="flex items-center gap-2 text-body-small">
          <span>Avg {formatDuration(metrics.averageDuration)}</span>
          <span>Median {formatDuration(metrics.medianDuration)}</span>
          <span>P95 {formatDuration(metrics.p95Duration)}</span>
        </div>
      </div>

      <div className="flex-1 p-6">
        <Card className="h-full">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" /> Workflow Performance Insights
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-8 p-6">
            {hasPerformanceData ? (
              <>
                <ResponsiveGrid columns={{ base: 1, md: 2, lg: 4 }} gap="md" align="start">
                  <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
                    <div className="text-caption">Total Duration</div>
                    <div className="mt-2 text-heading-2">{formatDuration(metrics.totalDuration)}</div>
                    <div className="mt-1 text-body-small">Across {timedNodes.length} measured steps</div>
                  </div>
                  <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
                    <div className="text-caption">Success Rate</div>
                    <div className="mt-2 text-heading-2">
                      {statusSummary.total
                        ? Math.round(((statusSummary.byStatus['succeeded'] || 0) / statusSummary.total) * 100)
                        : 0}%
                    </div>
                    <div className="mt-1 text-body-small">{statusSummary.total} total executions</div>
                  </div>
                  <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
                    <div className="text-caption">Fastest Step</div>
                    <div className="mt-2 text-heading-2">{formatDuration(Math.min(...metrics.durations))}</div>
                    <div className="mt-1 text-body-small">Slowest {formatDuration(metrics.maxDuration)}</div>
                  </div>
                  <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
                    <div className="text-caption">Throughput</div>
                    <div className="mt-2 text-heading-2">{throughputPerMinute.toFixed(1)} / min</div>
                    <div className="mt-1 text-body-small">Span {metrics.spanMs > 0 ? formatDuration(metrics.spanMs) : '—'}</div>
                  </div>
                </ResponsiveGrid>

                {statusSummary.total > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-body-small">
                    {Object.entries(statusSummary.byStatus).map(([status, count]) => (
                      <Badge key={status} variant="outline" className="text-body-small">
                        {getStatusLabel(status)} · {count}
                      </Badge>
                    ))}
                  </div>
                )}

                {/* Primary Visualization: Scatter Plot */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-heading-3 flex items-center gap-2">
                      <GaugeCircle className="w-4 h-4" /> Execution Velocity & Health
                    </h3>
                    <span className="text-body-small">Individual executions over time</span>
                  </div>
                  <ExecutionScatterPlot
                    timedNodes={timedNodes}
                    onNodeClick={(id: string) => onNodeSelection([id], true)}
                  />
                </div>

                {/* Secondary Visualization: Agent Heatmap */}
                <div className="space-y-4 pt-4 border-t border-border/50">
                  <div className="flex items-center justify-between">
                    <h3 className="text-heading-3 flex items-center gap-2">
                       <Activity className="w-4 h-4" /> Agent Health Map
                    </h3>
                    <span className="text-body-small">Failure patterns by agent</span>
                  </div>
                  <AgentHealthHeatmap timedNodes={timedNodes} />
                </div>

                {bottlenecks.length > 0 && (
                  <div className="space-y-3 pt-4 border-t border-border/50">
                    <div className="flex items-center gap-2 text-heading-3">
                      <Timer className="w-4 h-4" /> Top bottlenecks
                    </div>
                    <div className="space-y-3">
                      {bottlenecks.map((node, index) => {
                        const duration = Number(node.duration_ms) || 0;
                        const share = Math.round((duration / totalTimedDuration) * 100);
                        const width = metrics.maxDuration > 0 ? Math.max(8, (duration / metrics.maxDuration) * 100) : 0;
                        return (
                          <div key={node.execution_id} className="rounded-lg border border-border/60 bg-muted/15 p-3">
                            <div className="flex items-center justify-between text-sm">
                              <div className="flex items-center gap-2">
                                <Badge variant="destructive" className="text-body-small">#{index + 1}</Badge>
                                <span className="font-medium text-foreground">
                                  {node.agent_name || node.reasoner_id || 'Workflow step'}
                                </span>
                              </div>
                              <div className="text-right text-body-small">
                                <div className="font-mono text-sm text-foreground">{formatDuration(duration)}</div>
                                <div>{share}% of total runtime</div>
                              </div>
                            </div>
                            <div className="mt-3 h-2 rounded-full bg-muted overflow-hidden">
                              <div
                                className="h-2 rounded-full bg-destructive"
                                style={{ width: `${width}%` }}
                              />
                            </div>
                            <div className="mt-3 flex items-center justify-between text-body-small">
                              <span>{formatTimestamp(node.started_at)}</span>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => onNodeSelection([node.execution_id], true)}
                              >
                                Focus
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {agentPerformance.length > 0 && (
                  <div className="space-y-3 pt-4 border-t border-border/50">
                    <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                      <Activity className="w-4 h-4" /> Agent Aggregate Impact
                    </div>
                    <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="sm" align="start">
                      {agentPerformance.map((agent) => (
                        <div key={agent.name} className="rounded-lg border border-border/60 bg-muted/15 p-3 space-y-2">
                          <div className="flex items-center justify-between text-sm font-medium text-foreground">
                            <span>{agent.name}</span>
                            <span>{agent.count} steps</span>
                          </div>
                          <div className="text-body-small">Total {formatDuration(agent.totalDuration)}</div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-2 rounded-full bg-primary"
                              style={{ width: `${Math.min(100, (agent.totalDuration / totalTimedDuration) * 100)}%` }}
                            />
                          </div>
                          <div className="flex items-center justify-between text-body-small">
                            <span>Avg {formatDuration(agent.averageDuration)}</span>
                            <span>Max {formatDuration(agent.maxDuration)}</span>
                          </div>
                        </div>
                      ))}
                    </ResponsiveGrid>
                  </div>
                )}

                {depthPerformance.length > 0 && (
                  <div className="space-y-3 pt-4 border-t border-border/50">
                    <div className="text-sm font-semibold text-foreground">Depth profile</div>
                    <ResponsiveGrid columns={{ base: 1, md: 2, lg: 4 }} gap="sm" align="start">
                      {depthPerformance.map((depth) => (
                        <div key={depth.depth} className="rounded-lg border border-border/60 bg-muted/10 p-3 text-xs">
                          <div className="text-muted-foreground">Depth {depth.depth}</div>
                          <div className="mt-1 text-sm font-semibold text-foreground">{formatDuration(depth.averageDuration)}</div>
                          <div className="mt-1 text-muted-foreground">{depth.count} steps</div>
                        </div>
                      ))}
                    </ResponsiveGrid>
                  </div>
                )}

                {selectedMetrics.count > 0 && (
                  <div className="space-y-3 pt-4 border-t border-border/50 bg-muted/5 -mx-6 px-6 pb-6 mb-[-24px]">
                    <div className="text-sm font-semibold text-foreground pt-4">Selected nodes Analysis</div>
                    <ResponsiveGrid columns={{ base: 1, md: 2, lg: 4 }} gap="sm" align="start">
                      <div className="rounded-lg border border-border bg-background p-3">
                        <div className="text-body-small">Selected</div>
                        <div className="text-heading-3 text-foreground">{selectedMetrics.count}</div>
                      </div>
                      <div className="rounded-lg border border-border bg-background p-3">
                        <div className="text-body-small">Measured</div>
                        <div className="text-heading-3 text-foreground">{selectedMetrics.timedCount}</div>
                      </div>
                      <div className="rounded-lg border border-border bg-background p-3">
                        <div className="text-body-small">Avg duration</div>
                        <div className="text-heading-3 text-foreground">{formatDuration(selectedMetrics.averageDuration)}</div>
                      </div>
                      <div className="rounded-lg border border-border bg-background p-3">
                        <div className="text-body-small">Succeeded</div>
                        <div className="text-heading-3 text-foreground">{selectedMetrics.successCount}</div>
                      </div>
                    </ResponsiveGrid>
                  </div>
                )}
              </>
            ) : (
              <div className="py-16 text-center space-y-4">
                <div className="flex justify-center">
                  <Zap className="h-10 w-10 text-muted-foreground" />
                </div>
                <div>
                  <h3 className="text-heading-3 text-foreground">No performance data yet</h3>
                  <p className="text-body-small">Metrics will appear once workflow steps record execution durations.</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

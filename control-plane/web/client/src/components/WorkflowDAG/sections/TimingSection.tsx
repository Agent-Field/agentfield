import { Calendar, Time, Timer } from "@/components/ui/icon-bridge";
import { statusTone } from "../../../lib/theme";
import { cn } from "../../../lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/card";

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
  task_name?: string;
  agent_name?: string;
}

interface NodeDetails {
  input?: any;
  output?: any;
  error_message?: string;
  cost?: number;
  memory_updates?: any[];
  performance_metrics?: {
    response_time_ms: number;
    tokens_used?: number;
  };
}

interface TimingSectionProps {
  node: WorkflowNodeData;
  details?: NodeDetails;
}

export function TimingSection({ node, details }: TimingSectionProps) {
  const formatDuration = (durationMs?: number) => {
    if (!durationMs) return "-";
    if (durationMs < 1000) return `${durationMs}ms`;
    if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      }),
      time: date.toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
  };

  const getElapsedTime = () => {
    if (!node.started_at) return null;

    const startTime = new Date(node.started_at).getTime();
    const endTime = node.completed_at
      ? new Date(node.completed_at).getTime()
      : Date.now();

    return endTime - startTime;
  };

  const startedAt = formatTimestamp(node.started_at);
  const completedAt = node.completed_at
    ? formatTimestamp(node.completed_at)
    : null;
  const elapsedTime = getElapsedTime();

  return (
    <Card className="bg-muted border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
          <Time size={16} className="text-muted-foreground" />
          Timing Information
        </CardTitle>
      </CardHeader>

      <CardContent className="pt-0 space-y-4">
        {/* Start Time */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Calendar
              size={14}
              className="text-muted-foreground flex-shrink-0 mt-0.5"
            />
            <div>
              <span className="text-sm text-muted-foreground/70 block">
                Started
              </span>
              <span className="text-sm text-foreground font-medium">
                {startedAt.time}
              </span>
              <span className="text-sm text-muted-foreground block">
                {startedAt.date}
              </span>
            </div>
          </div>
        </div>

        {/* Completion Time */}
        {completedAt && (
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <Calendar
                size={14}
                className="text-muted-foreground flex-shrink-0 mt-0.5"
              />
              <div>
                <span className="text-sm text-muted-foreground/70 block">
                  Completed
                </span>
                <span className="text-sm text-foreground font-medium">
                  {completedAt.time}
                </span>
                <span className="text-sm text-muted-foreground block">
                  {completedAt.date}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Duration */}
        <div className="pt-4 border-t border-border">
          <div className="flex items-center gap-2 mb-3">
            <Timer size={14} className="text-muted-foreground" />
            <span className="text-sm text-muted-foreground/70">
              Duration
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Execution Duration */}
            <div>
              <span className="text-sm text-muted-foreground text-muted-foreground/70 block mb-1">
                Execution Time
              </span>
              <span className="text-base font-semibold font-mono text-foreground">
                {formatDuration(node.duration_ms)}
              </span>
            </div>

            {/* Total Elapsed Time */}
            <div>
              <span className="text-sm text-muted-foreground text-muted-foreground/70 block mb-1">
                {node.status === "running" ? "Elapsed Time" : "Total Time"}
              </span>
              <span className="text-base font-semibold font-mono text-foreground">
                {formatDuration(elapsedTime || undefined)}
              </span>
            </div>
          </div>

          {/* Performance Metrics */}
          {details?.performance_metrics && (
            <div className="mt-4 pt-4 border-t border-border">
              <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                <div>
                  <span className="text-muted-foreground/70 block mb-1">
                    Response Time
                  </span>
                  <span className="text-muted-foreground font-mono">
                    {details.performance_metrics.response_time_ms}ms
                  </span>
                </div>
                {details.performance_metrics.tokens_used && (
                  <div>
                    <span className="text-muted-foreground/70 block mb-1">
                      Tokens Used
                    </span>
                    <span className="text-muted-foreground font-mono">
                      {details.performance_metrics.tokens_used.toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Live Timer for Running Executions */}
          {node.status === "running" && (
            <div
              className={cn(
                "mt-4 rounded-lg p-3",
                statusTone.info.bg,
                statusTone.info.border
              )}
            >
              <div className="flex items-center gap-2">
                <div className={cn("h-2 w-2 animate-pulse rounded-full", statusTone.info.dot)} />
                <span className={cn("text-xs font-medium", statusTone.info.fg)}>
                  Execution in progress
                </span>
              </div>
              <div className="text-sm text-muted-foreground mt-1">
                Started{" "}
                {Math.floor(
                  (Date.now() - new Date(node.started_at).getTime()) / 1000
                )}
                s ago
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

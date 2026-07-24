import {
  CheckmarkFilled,
  ErrorFilled,
  InProgress,
  PauseFilled,
  WarningFilled,
} from "@/components/ui/icon-bridge";
import { cn } from "../../../lib/utils";
import { statusTone, type StatusTone as ToneKey } from "../../../lib/theme";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/card";
import {
  type CanonicalStatus,
  getStatusLabel,
  normalizeExecutionStatus,
} from "../../../utils/status";

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

interface StatusSectionProps {
  node: WorkflowNodeData;
  details?: NodeDetails;
}

const STATUS_TONE_MAP: Record<CanonicalStatus, ToneKey> = {
  succeeded: "success",
  failed: "error",
  running: "info",
  paused: "warning",
  waiting: "warning",
  pending: "warning",
  queued: "warning",
  timeout: "info",
  cancelled: "neutral",
  unknown: "neutral",
};

export function StatusSection({ node, details }: StatusSectionProps) {
  const normalizedStatus = normalizeExecutionStatus(node.status);
  const tone = statusTone[STATUS_TONE_MAP[normalizedStatus]];

  const { icon, label, description } = (() => {
    switch (normalizedStatus) {
      case "succeeded":
        return {
          icon: <CheckmarkFilled size={20} className={tone.accent} />,
          label: "Completed Successfully",
          description: "Execution finished without errors",
        };
      case "failed":
        return {
          icon: <ErrorFilled size={20} className={tone.accent} />,
          label: "Execution Failed",
          description:
            details?.error_message || "Execution encountered an error",
        };
      case "running":
        return {
          icon: (
            <InProgress
              size={20}
              className={cn(tone.accent, "animate-spin")}
            />
          ),
          label: "Currently Running",
          description: "Execution is in progress",
        };
      case "waiting":
        return {
          icon: (
            <PauseFilled
              size={20}
              className={cn(tone.accent, "animate-pulse")}
            />
          ),
          label: "Awaiting Input",
          description: "Execution is paused waiting for human input",
        };
      case "pending":
      case "queued":
        return {
          icon: <PauseFilled size={20} className={tone.accent} />,
          label: "Pending Execution",
          description: "Waiting to be executed",
        };
      case "timeout":
      case "cancelled":
      case "unknown":
      default:
        return {
          icon: <WarningFilled size={20} className={tone.accent} />,
          label: getStatusLabel(normalizedStatus),
          description: "Status information unavailable",
        };
    }
  })();

  return (
    <Card className="bg-muted border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
          {icon}
          Execution Status
        </CardTitle>
      </CardHeader>

      <CardContent className="pt-0">
        {/* Status Display */}
        <div className={cn("rounded-lg p-4", tone.bg, tone.border)}>
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">{icon}</div>
            <div className="flex-1 min-w-0">
              <h4 className={cn("mb-1 text-sm font-medium", tone.fg)}>
                {label}
              </h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {description}
              </p>
            </div>
          </div>
        </div>

        {/* Error Details */}
        {normalizedStatus === "failed" && details?.error_message && (
          <div
            className={cn(
              "mt-4 rounded-lg p-3",
              statusTone.error.bg,
              statusTone.error.border
            )}
          >
            <h5
              className={cn(
                "mb-2 text-xs font-medium",
                statusTone.error.fg
              )}
            >
              Error Details
            </h5>
            <pre className="text-sm text-muted-foreground whitespace-pre-wrap font-mono leading-relaxed">
              {details.error_message}
            </pre>
          </div>
        )}

        {/* Progress Indicator for Running Status */}
        {node.status === "running" && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-muted-foreground mb-2">
              <span>Execution in progress...</span>
              <span>
                {node.started_at &&
                  new Date(node.started_at).toLocaleTimeString()}
              </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-muted">
              <div
                className={cn(
                  "h-1.5 rounded-full animate-pulse",
                  statusTone.info.solidBg
                )}
                style={{ width: "60%" }}
              />
            </div>
          </div>
        )}

        {/* Memory Updates */}
        {details?.memory_updates && details.memory_updates.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <h5 className="text-xs font-medium text-foreground mb-2">
              Memory Updates
            </h5>
            <div className="space-y-1">
              {details.memory_updates
                .slice(0, 3)
                .map((update: any, index: number) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 text-sm text-muted-foreground"
                  >
                    <div
                      className={cn(
                        "h-1.5 w-1.5 flex-shrink-0 rounded-full",
                        statusTone.info.dot
                      )}
                    />
                    <span className="truncate">
                      {update.action} {update.scope}/{update.key}
                    </span>
                  </div>
                ))}
              {details.memory_updates.length > 3 && (
                <div className="text-sm text-muted-foreground/70 pl-3.5">
                  +{details.memory_updates.length - 3} more updates
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

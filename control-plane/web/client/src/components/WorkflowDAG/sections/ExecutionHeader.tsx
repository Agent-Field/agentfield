import { CheckmarkFilled, Copy, User, Launch } from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import { cn } from "../../../lib/utils";
import { Badge } from "../../ui/badge";
import { Button } from "../../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/card";
import { normalizeExecutionStatus, getStatusLabel } from "../../../utils/status";

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

interface ExecutionHeaderProps {
  node: WorkflowNodeData;
  details?: NodeDetails;
  onCopy: (text: string, label: string) => void;
  copySuccess: string | null;
}

export function ExecutionHeader({
  node,
  details,
  onCopy,
  copySuccess,
}: ExecutionHeaderProps) {
  const navigate = useNavigate();

  const handleViewExecution = () => {
    navigate(`/executions/${node.execution_id}`);
  };

  const normalizedStatus = normalizeExecutionStatus(node.status);

  const getStatusColors = (status: string) => {
    switch (normalizeExecutionStatus(status)) {
      case "succeeded":
        return {
          bg: "bg-status-success/10",
          border: "border-status-success/30",
          text: "text-status-success",
        };
      case "failed":
        return {
          bg: "bg-status-error/10",
          border: "border-status-error/30",
          text: "text-status-error",
        };
      case "running":
        return {
          bg: "bg-status-info/10",
          border: "border-status-info/30",
          text: "text-status-info",
        };
      case "pending":
      case "queued":
        return {
          bg: "bg-status-warning/10",
          border: "border-status-warning/30",
          text: "text-status-warning",
        };
      default:
        return {
          bg: "bg-muted",
          border: "border-border",
          text: "text-muted-foreground",
        };
    }
  };

  const statusColors = getStatusColors(normalizedStatus);

  return (
    <Card className="bg-muted border-border">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base font-semibold text-foreground mb-2">
              {node.task_name || node.reasoner_id}
            </CardTitle>

            {/* Agent Information */}
            <div className="flex items-center gap-2 mb-3">
              <User
                size={14}
                className="text-muted-foreground flex-shrink-0"
              />
              <span className="text-sm text-muted-foreground truncate">
                {node.agent_name || node.agent_node_id}
              </span>
            </div>

            {/* Status Badge */}
            <Badge
              className={cn(
                "text-xs font-medium px-3 py-1 rounded-full border",
                statusColors.bg,
                statusColors.border,
                statusColors.text
              )}
            >
              {getStatusLabel(normalizedStatus)}
            </Badge>
          </div>

          {/* Copy Execution ID Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCopy(node.execution_id, "Execution ID")}
            className="ml-2 h-8 w-8 p-0 hover:bg-muted"
            title="Copy Execution ID"
          >
            {copySuccess === "Execution ID" ? (
              <CheckmarkFilled
                size={14}
                className="text-status-success"
              />
            ) : (
              <Copy size={14} className="text-muted-foreground" />
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {/* View Full Execution Button */}
        <div className="mb-4">
          <Button
            variant="outline"
            size="sm"
            onClick={handleViewExecution}
            className="w-full h-9 text-sm font-medium bg-muted/80 hover:bg-muted border-border text-foreground"
          >
            <Launch size={16} className="mr-2" />
            View Full Execution Details
          </Button>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-muted-foreground/70 block mb-1">
              Execution ID
            </span>
            <span className="text-muted-foreground font-mono">
              {node.execution_id.slice(0, 8)}...
            </span>
          </div>
          <div>
            <span className="text-muted-foreground/70 block mb-1">
              Workflow Depth
            </span>
            <span className="text-muted-foreground">
              Level {node.workflow_depth}
            </span>
          </div>
        </div>

        {/* Performance Metrics */}
        {details?.performance_metrics && (
          <div className="mt-4 pt-4 border-t border-border">
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-muted-foreground/70 block mb-1">
                  Response Time
                </span>
                <span className="text-muted-foreground font-mono">
                  {details.performance_metrics.response_time_ms}ms
                </span>
              </div>
              {details.cost && (
                <div>
                  <span className="text-muted-foreground/70 block mb-1">
                    Cost
                  </span>
                  <span className="text-muted-foreground font-mono">
                    ${details.cost.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

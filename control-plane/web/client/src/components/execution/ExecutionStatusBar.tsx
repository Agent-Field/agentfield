import {
  ArrowLeft,
  Clock,
  RotateCcw,
  Share2,
  CheckCircle,
  XCircle,
  Loader2,
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from "../../utils/status";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { CopyButton } from "../ui/copy-button";

interface ExecutionStatusBarProps {
  execution: WorkflowExecution;
  onBack?: () => void;
}

function StatusIcon({ status }: { status: string }) {
  const normalized = normalizeExecutionStatus(status);
  const theme = getStatusTheme(normalized);
  const iconClass = `w-4 h-4 ${theme.iconClass}`;
  switch (normalized) {
    case "succeeded":
      return <CheckCircle className={iconClass} />;
    case "failed":
      return <XCircle className={iconClass} />;
    case "running":
      return <Loader2 className={`${iconClass} animate-spin`} />;
    case "pending":
    case "queued":
      return <Clock className={iconClass} />;
    case "timeout":
      return <Clock className={iconClass} />;
    default:
      return <Clock className={iconClass} />;
  }
}

function formatDuration(durationMs?: number): string {
  if (!durationMs) return "0ms";

  if (durationMs < 1000) {
    return `${durationMs}ms`;
  } else if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}

export function ExecutionStatusBar({
  execution,
  onBack,
}: ExecutionStatusBarProps) {
  const navigate = useNavigate();
  const status = normalizeExecutionStatus(execution.status);
  const theme = getStatusTheme(status);

  const handleRetry = () => {
  };

  const handleShare = () => {
    // TODO: Implement share functionality
    navigator.clipboard.writeText(window.location.href);
  };

  return (
    <div className="bg-background border-b border-border sticky top-0 z-10">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between py-3">
          {/* Left: Navigation & Status */}
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack || (() => navigate("/executions"))}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>

            <div className="flex items-center gap-3">
              <StatusIcon status={status} />
              <div>
                <h1 className="font-semibold text-foreground">
                  {execution.reasoner_id}
                </h1>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>ID:</span>
                  <code className="font-mono text-xs">
                    {execution.execution_id.slice(0, 8)}...
                    {execution.execution_id.slice(-4)}
                  </code>
                  <CopyButton
                    value={execution.execution_id}
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                    tooltip="Copy execution ID"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right: Actions & Metrics */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{formatDuration(execution.duration_ms)}</span>
              </div>
              <Badge
                variant={theme.badgeVariant}
                className={theme.pillClass}
              >
                {getStatusLabel(status)}
              </Badge>
            </div>

            <div className="flex items-center gap-2">
              {status === "failed" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRetry}
                  className="flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Retry
                </Button>
              )}

              <Button
                variant="outline"
                size="sm"
                onClick={handleShare}
                className="flex items-center gap-2"
              >
                <Share2 className="w-4 h-4" />
                Share
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

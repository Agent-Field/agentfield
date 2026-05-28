import { Clock, ArrowDown, ArrowUp, RotateCcw } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { normalizeExecutionStatus } from "../../utils/status";

interface ExecutionPerformanceStripProps {
  execution: WorkflowExecution;
}

function formatDuration(durationMs?: number): string {
  if (!durationMs) return "—";

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

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function ExecutionPerformanceStrip({ execution }: ExecutionPerformanceStripProps) {
  const status = normalizeExecutionStatus(execution.status);
  const duration = execution.duration_ms || 0;
  const retryCount = execution.retry_count || 0;
  const inputSize = execution.input_size || 0;
  const outputSize = execution.output_size || 0;

  // Simple performance assessment
  const getPerformanceIndicator = () => {
    if (status === 'failed') return { color: 'text-red-500', label: 'Failed' };
    if (retryCount > 0) return { color: 'text-yellow-500', label: 'Retried' };
    if (duration > 30000) return { color: 'text-yellow-500', label: 'Slow' }; // > 30s
    if (status === 'succeeded') return { color: 'text-green-500', label: 'Fast' };
    if (status === 'running') return { color: 'text-blue-500', label: 'Running' };
    return { color: 'text-muted-foreground', label: 'Normal' };
  };

  const performance = getPerformanceIndicator();

  return (
    <div className="border-b border-border bg-muted/20">
      <div className="px-6 py-3">
        <div className="flex flex-wrap items-center gap-6 text-sm">
          {/* Duration */}
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Duration:</span>
            <span className={`font-medium ${performance.color}`}>
              {formatDuration(duration)}
            </span>
            {performance.label !== 'Normal' && (
              <span className={`text-xs ${performance.color}`}>
                ({performance.label})
              </span>
            )}
          </div>

          {/* Input Size */}
          <div className="flex items-center gap-2">
            <ArrowDown className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Input:</span>
            <span className="font-medium text-foreground">
              {formatBytes(inputSize)}
            </span>
          </div>

          {/* Output Size */}
          <div className="flex items-center gap-2">
            <ArrowUp className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Output:</span>
            <span className="font-medium text-foreground">
              {formatBytes(outputSize)}
            </span>
          </div>

          {/* Retry Count */}
          <div className="flex items-center gap-2">
            <RotateCcw className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Retries:</span>
            <span className={`font-medium ${retryCount > 0 ? 'text-yellow-500' : 'text-foreground'}`}>
              {retryCount}
            </span>
          </div>

          {/* Data Flow Indicator */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Data Flow:</span>
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${inputSize > 0 ? 'bg-blue-500' : 'bg-muted'}`} />
              <div className="w-4 h-px bg-muted" />
              <div className={`w-2 h-2 rounded-full ${outputSize > 0 ? 'bg-green-500' : 'bg-muted'}`} />
            </div>
            <span>
              {inputSize > 0 && outputSize > 0 ? 'Complete' :
               inputSize > 0 ? 'Input only' :
               outputSize > 0 ? 'Output only' : 'No data'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

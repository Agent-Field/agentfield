import { CheckCircle, XCircle, Clock, Copy, Loader2 } from '@/components/ui/icon-bridge';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert } from '../ui/alert';
import type { ExecutionResponse } from '../../types/execution';
import { getStatusLabel, isFailureStatus, isSuccessStatus, isTimeoutStatus } from '../../utils/status';
import { JsonHighlightedPre } from '@/components/ui/json-syntax-highlight';

interface ExecutionResultProps {
  result?: ExecutionResponse | null;
  error?: string | null;
  loading?: boolean;
}

export function ExecutionResult({ result, error, loading }: ExecutionResultProps) {
  const handleCopyResult = () => {
    if (result) {
      navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    }
  };

  const handleCopyOutput = () => {
    if (result?.result) {
      navigator.clipboard.writeText(JSON.stringify(result.result, null, 2));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-center space-y-2">
          <Loader2 className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Executing reasoner...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="border-red-200 bg-red-50">
        <XCircle className="h-4 w-4 text-red-600" />
        <div>
          <h4 className="font-semibold text-red-800">Execution Failed</h4>
          <p className="text-sm text-red-700 mt-1">{error}</p>
        </div>
      </Alert>
    );
  }

  if (!result) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <p>No execution result yet. Click "Execute Reasoner" to run.</p>
      </div>
    );
  }

  const isSuccess = isSuccessStatus(result.status);
  const isFailure = isFailureStatus(result.status);
  const isTimeout = isTimeoutStatus(result.status);
  const statusLabel = getStatusLabel(result.status);

  return (
    <div className="space-y-4">
      {/* Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isSuccess ? (
            <CheckCircle className="h-5 w-5 text-green-500" />
          ) : isFailure ? (
            <XCircle className="h-5 w-5 text-red-500" />
          ) : (
            <Clock className="h-5 w-5 text-amber-500" />
          )}
          <Badge
            variant={
              isSuccess ? 'default' : isFailure ? 'destructive' : 'secondary'
            }
          >
            {statusLabel}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleCopyResult}
            className="flex items-center gap-1"
          >
            <Copy className="h-3 w-3" />
            Copy All
          </Button>
        </div>
      </div>

      {/* Execution Metadata */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-3 bg-muted/50 rounded-lg">
        <div>
          <p className="text-sm text-muted-foreground">Execution ID</p>
          <p className="text-sm font-mono">{result.execution_id.slice(0, 8)}...</p>
        </div>
        {result.workflow_id && (
          <div>
            <p className="text-sm text-muted-foreground">Workflow</p>
            <p className="text-sm font-mono">{result.workflow_id.slice(0, 8)}...</p>
          </div>
        )}
        {result.run_id && (
          <div>
            <p className="text-sm text-muted-foreground">Run</p>
            <p className="text-sm font-mono">{result.run_id.slice(0, 8)}...</p>
          </div>
        )}
        <div>
          <p className="text-sm text-muted-foreground">Duration</p>
          <p className="text-sm font-semibold">{result.duration_ms}ms</p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Timestamp</p>
          <p className="text-sm">{new Date(result.timestamp).toLocaleTimeString()}</p>
        </div>
        {result.cost && (
          <div>
            <p className="text-sm text-muted-foreground">Cost</p>
            <p className="text-sm">${result.cost.toFixed(4)}</p>
          </div>
        )}
      </div>

      {/* Error Message (if failed) */}
      {(isFailure || isTimeout) && result.error_message && (
        <Alert className="border-red-200 bg-red-50">
          <XCircle className="h-4 w-4 text-red-600" />
          <div>
            <h4 className="font-semibold text-red-800">Error Details</h4>
            <p className="text-sm text-red-700 mt-1 font-mono">{result.error_message}</p>
          </div>
        </Alert>
      )}

      {/* Result Output */}
      {result.result && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold">Output</h4>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopyOutput}
              className="flex items-center gap-1"
            >
              <Copy className="h-3 w-3" />
              Copy Output
            </Button>
          </div>

          <div className="relative">
            <JsonHighlightedPre
              data={result.result}
              className="max-h-96 overflow-auto rounded-lg border border-border bg-background p-4 text-sm"
            />
          </div>
        </div>
      )}

      {/* Memory Updates (if any) */}
      {result.memory_updates && result.memory_updates.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-semibold">Memory Updates</h4>
          <div className="space-y-1">
            {result.memory_updates.map((update, index) => (
              <div key={index} className="flex items-center gap-2 text-sm p-2 bg-muted/50 rounded">
                <Badge variant="outline" className="text-xs">
                  {update.action}
                </Badge>
                <span className="font-mono">{update.scope}.{update.key}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Indicator */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Clock className="h-4 w-4" />
        <span>
          Execution completed in {result.duration_ms}ms
          {result.duration_ms < 1000 && ' (Fast)'}
          {result.duration_ms >= 1000 && result.duration_ms < 5000 && ' (Normal)'}
          {result.duration_ms >= 5000 && ' (Slow)'}
        </span>
      </div>
    </div>
  );
}

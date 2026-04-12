import { AlertTriangle, RefreshCw, ExternalLink } from '@/components/ui/icon-bridge';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import type { WorkflowExecution } from '../../types/executions';

interface ErrorPanelProps {
  execution: WorkflowExecution;
  onRetry?: () => void;
  onViewLogs?: () => void;
}

export function ErrorPanel({ execution, onRetry, onViewLogs }: ErrorPanelProps) {
  if (!execution.error_message) {
    return null;
  }

  return (
    <Card className="border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/20">
      <CardHeader>
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-500" />
          <CardTitle className="text-base font-semibold text-red-700 dark:text-red-400">
            Execution Failed
          </CardTitle>
          {execution.retry_count > 0 && (
            <Badge variant="destructive" className="ml-auto">
              Retry #{execution.retry_count}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <div className="bg-red-100/50 dark:bg-red-950/30 rounded-lg p-4 font-mono text-sm mb-4">
          <pre className="whitespace-pre-wrap text-red-800 dark:text-red-200">
            {execution.error_message}
          </pre>
        </div>

        <div className="flex items-center gap-3">
          {onRetry && (
            <Button
              variant="outline"
              className="border-red-300 text-red-700 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-950/30"
              onClick={onRetry}
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry Execution
            </Button>
          )}
          {onViewLogs && (
            <Button variant="ghost" size="sm" onClick={onViewLogs}>
              <ExternalLink className="w-4 h-4 mr-2" />
              View Logs
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

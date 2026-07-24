import { Play, CheckCircle, XCircle, Clock } from '@/components/ui/icon-bridge';
import { formatCompactRelativeTime } from '@/utils/dateFormat';
import type { WorkflowExecution } from '../../types/executions';

interface ExecutionTimelineProps {
  execution: WorkflowExecution;
}

const formatRelativeTime = formatCompactRelativeTime;

function normalizeStatus(status: string): string {
  switch (status?.toLowerCase()) {
    case 'completed':
    case 'success':
    case 'succeeded':
      return 'completed';
    case 'failed':
    case 'error':
    case 'timeout':
      return 'failed';
    case 'running':
      return 'running';
    case 'pending':
    case 'queued':
      return 'pending';
    default:
      return status ?? 'unknown';
  }
}

export function ExecutionTimeline({ execution }: ExecutionTimelineProps) {
  const startedAt = execution.started_at || execution.created_at;
  const startedTime = formatRelativeTime(startedAt);
  const completedTime = execution.completed_at ? formatRelativeTime(execution.completed_at) : null;
  const status = normalizeStatus(execution.status);

  return (
    <div className="flex items-center gap-4 text-sm text-muted-foreground">
      <div className="flex items-center gap-2">
        <Play className="w-3 h-3" />
        <span>Started {startedTime}</span>
      </div>

      <div className="flex-1 h-px bg-border" />

      {completedTime && (
        <div className="flex items-center gap-2">
          {status === 'completed' ? (
            <CheckCircle className="w-3 h-3 text-green-500" />
          ) : status === 'failed' ? (
            <XCircle className="w-3 h-3 text-red-500" />
          ) : (
            <Clock className="w-3 h-3 text-gray-500" />
          )}
          <span>
            {status === 'completed'
              ? 'Completed'
              : status === 'failed'
              ? 'Failed'
              : 'Finished'}{' '}
            {completedTime}
          </span>
        </div>
      )}
    </div>
  );
}

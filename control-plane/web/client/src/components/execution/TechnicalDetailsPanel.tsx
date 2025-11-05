import { Settings } from '@/components/ui/icon-bridge';
import { ResponsiveGrid } from '@/components/layout/ResponsiveGrid';
import { Card, CardContent } from '../ui/card';
import type { WorkflowExecution } from '../../types/executions';

interface TechnicalDetailsPanelProps {
  execution: WorkflowExecution;
}

interface DetailItemProps {
  label: string;
  value: string | number | null | undefined;
}

function DetailItem({ label, value }: DetailItemProps) {
  const displayValue = value?.toString() || 'N/A';
  const isId = label.toLowerCase().includes('id');

  return (
    <div className="space-y-1">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        {label}
      </div>
      <div className={`text-sm break-all ${isId ? 'font-mono' : ''} text-foreground`}>
        {displayValue}
      </div>
    </div>
  );
}

function formatTimestamp(timestamp: string): string {
  if (!timestamp) return 'N/A';

  try {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }
    return date.toLocaleString();
  } catch {
    return 'Invalid date';
  }
}

export function TechnicalDetailsPanel({ execution }: TechnicalDetailsPanelProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <Settings className="w-4 h-4" />
          <span className="text-heading-3">Technical Details</span>
        </div>

        <ResponsiveGrid
          variant="metrics"
          align="start"
        >
          <DetailItem
            label="Execution ID"
            value={execution.execution_id}
          />
          <DetailItem
            label="Workflow ID"
            value={execution.workflow_id}
          />
          <DetailItem
            label="Session ID"
            value={execution.session_id}
          />
          <DetailItem
            label="Haxen Request ID"
            value={execution.haxen_request_id}
          />
          <DetailItem
            label="Actor ID"
            value={execution.actor_id}
          />
          <DetailItem
            label="Workflow Depth"
            value={execution.workflow_depth}
          />
          <DetailItem
            label="Retry Count"
            value={execution.retry_count}
          />
          <DetailItem
            label="Duration (ms)"
            value={execution.duration_ms}
          />
          <DetailItem
            label="Input Size (bytes)"
            value={execution.input_size}
          />
          <DetailItem
            label="Output Size (bytes)"
            value={execution.output_size}
          />
          <DetailItem
            label="Created At"
            value={formatTimestamp(execution.created_at)}
          />
          <DetailItem
            label="Updated At"
            value={formatTimestamp(execution.updated_at)}
          />
          <DetailItem
            label="Started At"
            value={formatTimestamp(execution.started_at)}
          />
          <DetailItem
            label="Completed At"
            value={execution.completed_at ? formatTimestamp(execution.completed_at) : 'N/A'}
          />
        </ResponsiveGrid>
      </CardContent>
    </Card>
  );
}

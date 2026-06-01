import React from 'react';
import { Close, Time, Activity, Layers, Copy, CheckmarkFilled } from '@/components/ui/icon-bridge';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { cn } from '../../lib/utils';
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from '../../utils/status';

interface WorkflowDAGNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  workflow_depth: number;
  children: WorkflowDAGNode[];
}

interface ExecutionPanelProps {
  execution: WorkflowDAGNode | null;
  onClose: () => void;
  isOpen: boolean;
  task_name?: string;
  agent_name?: string;
}

export function ExecutionPanel({ execution, onClose, isOpen, task_name, agent_name }: ExecutionPanelProps) {
  const [copiedField, setCopiedField] = React.useState<string | null>(null);

  const copyToClipboard = async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const formatDuration = (durationMs?: number) => {
    if (!durationMs) return 'N/A';
    if (durationMs < 1000) return `${durationMs}ms`;
    if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
    return `${(durationMs / 60000).toFixed(1)}m`;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString(),
      relative: getRelativeTime(date)
    };
  };

  const getRelativeTime = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();

    if (diffMs < 60000) return 'Just now';
    if (diffMs < 3600000) return `${Math.floor(diffMs / 60000)}m ago`;
    if (diffMs < 86400000) return `${Math.floor(diffMs / 3600000)}h ago`;
    return `${Math.floor(diffMs / 86400000)}d ago`;
  };

  if (!execution) return null;

  const normalizedStatus = normalizeExecutionStatus(execution.status);
  const statusTheme = getStatusTheme(normalizedStatus);
  const statusLabel = getStatusLabel(normalizedStatus);

  const startTime = formatTimestamp(execution.started_at);
  const endTime = execution.completed_at ? formatTimestamp(execution.completed_at) : null;

  return (
    <div
      className={cn(
        'fixed top-0 right-0 z-50 h-full w-96 border-l border-border bg-background shadow-xl',
        'transform transition-transform duration-300 ease-in-out',
        isOpen ? 'translate-x-0' : 'translate-x-full'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/50 p-4">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-muted-foreground" />
          <h2 className="text-base font-semibold text-foreground">Execution Details</h2>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
          <Close className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="h-full space-y-6 overflow-y-auto p-4 pb-20">
        {/* Status & Basic Info */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">Status</span>
            <Badge variant={statusTheme.badgeVariant} className={cn("text-sm", statusTheme.pillClass)}>
              {statusLabel}
            </Badge>
          </div>

          <div className="space-y-2">
            <h3 className="text-base font-semibold text-foreground">
              {task_name || execution.reasoner_id}
            </h3>
            <p className="text-sm text-muted-foreground">
              {agent_name || execution.agent_node_id}
            </p>
          </div>
        </div>

        {/* Timing Information */}
        <div className="space-y-4">
          <h4 className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Time className="h-4 w-4" />
            Timing
          </h4>

          <div className="space-y-3 pl-6">
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Started</div>
              <div className="text-sm font-mono text-foreground">{startTime.time} • {startTime.date}</div>
              <div className="text-sm text-muted-foreground">{startTime.relative}</div>
            </div>

            {endTime && (
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Completed</div>
                <div className="text-sm font-mono text-foreground">{endTime.time} • {endTime.date}</div>
                <div className="text-sm text-muted-foreground">{endTime.relative}</div>
              </div>
            )}

            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Duration</div>
              <div className="text-sm font-mono font-medium text-foreground">
                {formatDuration(execution.duration_ms)}
              </div>
            </div>
          </div>
        </div>

        {/* Workflow Information */}
        <div className="space-y-4">
          <h4 className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Layers className="h-4 w-4" />
            Workflow
          </h4>

          <div className="space-y-3 pl-6">
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Depth Level</div>
              <div className="text-sm font-medium text-foreground">{execution.workflow_depth}</div>
            </div>

            {execution.parent_workflow_id && (
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Parent Workflow</div>
                <div className="flex items-center gap-2">
                  <code className="rounded bg-muted px-2 py-1 text-xs font-mono text-foreground">
                    {execution.parent_workflow_id.slice(0, 12)}...
                  </code>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(execution.parent_workflow_id!, 'parent')}
                    className="h-6 w-6 p-0"
                  >
                    {copiedField === 'parent' ? (
                      <CheckmarkFilled className="h-3 w-3 text-status-success" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* IDs Section */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-foreground">Identifiers</h4>

          <div className="space-y-3 pl-6">
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Execution ID</div>
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded bg-muted px-2 py-1 text-xs font-mono text-foreground">
                  {execution.execution_id}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(execution.execution_id, 'execution')}
                  className="h-6 w-6 p-0"
                >
                  {copiedField === 'execution' ? (
                    <CheckmarkFilled className="h-3 w-3 text-status-success" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Workflow ID</div>
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded bg-muted px-2 py-1 text-xs font-mono text-foreground">
                  {execution.workflow_id}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(execution.workflow_id, 'workflow')}
                  className="h-6 w-6 p-0"
                >
                  {copiedField === 'workflow' ? (
                    <CheckmarkFilled className="h-3 w-3 text-status-success" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Agent Node ID</div>
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded bg-muted px-2 py-1 text-xs font-mono text-foreground">
                  {execution.agent_node_id}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(execution.agent_node_id, 'agent')}
                  className="h-6 w-6 p-0"
                >
                  {copiedField === 'agent' ? (
                    <CheckmarkFilled className="h-3 w-3 text-status-success" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Reasoner ID</div>
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded bg-muted px-2 py-1 text-xs font-mono text-foreground">
                  {execution.reasoner_id}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(execution.reasoner_id, 'reasoner')}
                  className="h-6 w-6 p-0"
                >
                  {copiedField === 'reasoner' ? (
                    <CheckmarkFilled className="h-3 w-3 text-status-success" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Children Information */}
        {execution.children && execution.children.length > 0 && (
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-foreground">Child Executions</h4>
            <div className="pl-6">
              <div className="text-sm text-muted-foreground">
                {execution.children.length} child execution{execution.children.length !== 1 ? 's' : ''}
              </div>
              <div className="mt-2 space-y-1">
                {execution.children.slice(0, 3).map((child) => (
                  <div key={child.execution_id} className="text-xs font-mono text-muted-foreground">
                    {child.execution_id.slice(0, 12)}... ({child.status})
                  </div>
                ))}
                {execution.children.length > 3 && (
                  <div className="text-sm text-muted-foreground">
                    +{execution.children.length - 3} more
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

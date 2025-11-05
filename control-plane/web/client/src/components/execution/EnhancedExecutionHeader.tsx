import {
  ArrowLeft,
  ExternalLink,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Play,
  Cog,
  XCircle
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import StatusIndicator from "../ui/status-indicator";
import { VerifiableCredentialBadge } from "../vc";
import { cn } from "../../lib/utils";
import { normalizeExecutionStatus } from "../../utils/status";
import { CopyButton } from "../ui/copy-button";

interface EnhancedExecutionHeaderProps {
  execution: WorkflowExecution;
  vcStatus?: {
    has_vc: boolean;
    vc_id?: string;
    status: string;
    created_at?: string;
    vc_document?: any;
  } | null;
  vcLoading?: boolean;
  onClose?: () => void;
}

function ExecutionTimeline({ execution }: { execution: WorkflowExecution }) {
  const navigate = useNavigate();

  const getTimelineSteps = () => {
    const steps = [
      { name: 'queued', label: 'Queued', icon: Clock },
      { name: 'started', label: 'Started', icon: Play },
      { name: 'processing', label: 'Processing', icon: Cog },
      { name: 'completed', label: 'Completed', icon: CheckCircle }
    ];
    const status = normalizeExecutionStatus(execution.status);

    let completedSteps = 0;
    switch (status) {
      case 'succeeded':
        completedSteps = 4;
        break;
      case 'failed':
        completedSteps = 3; // Failed during processing
        break;
      case 'running':
        completedSteps = 3; // Show processing as current
        break;
      case 'queued':
        completedSteps = 1;
        break;
      default:
        completedSteps = 1;
    }

    return steps.map((step, index) => ({
      ...step,
      completed: index < completedSteps,
      current: index === completedSteps - 1 && (status === 'running' || status === 'queued'),
      failed: status === 'failed' && index === completedSteps - 1
    }));
  };

  const formatTimestamp = (date: string) => {
    try {
      const timestamp = new Date(date);
      return timestamp.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
    } catch {
      return '';
    }
  };

  const formatDuration = (ms: number | null) => {
    if (!ms || ms <= 0) return "";
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const timelineSteps = getTimelineSteps();
  const duration = execution.duration_ms || 0;

  return (
    <div className="flex items-center gap-3">
      <span className="text-body-small">Timeline:</span>
      <div className="flex items-center gap-3">
        {timelineSteps.map((step, index) => {
          const Icon = step.failed ? XCircle : step.icon;
          const isActive = step.completed || step.current;

          return (
            <div key={step.name} className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                {/* Status dot */}
                <div className={cn(
                  "h-1.5 w-1.5 rounded-full flex-shrink-0",
                  step.completed ? "bg-success" :
                  step.current ? "bg-primary" :
                  step.failed ? "bg-destructive" :
                  "bg-muted-foreground/30"
                )} />

                {/* Icon */}
                <Icon className={cn(
                  "h-4 w-4 flex-shrink-0",
                  step.completed ? "text-success" :
                  step.current ? "text-primary" :
                  step.failed ? "text-destructive" :
                  "text-muted-foreground",
                  step.current && step.name === 'processing' && "animate-spin"
                )} />

                {/* Step label */}
                <span className={cn(
                  "text-xs font-medium",
                  isActive ? "text-foreground" : "text-muted-foreground"
                )}>
                  {step.label}
                </span>
              </div>

              {/* Timestamp and duration */}
              {isActive && (
                <div className="flex items-center gap-1 text-body-small font-mono">
                  {step.name === 'queued' && execution.created_at && (
                    <span title={`Queued at ${new Date(execution.created_at).toLocaleString()}`}>
                      {formatTimestamp(execution.created_at)}
                    </span>
                  )}
                  {step.name === 'started' && execution.started_at && (
                    <span title={`Started at ${new Date(execution.started_at).toLocaleString()}`}>
                      {formatTimestamp(execution.started_at)}
                    </span>
                  )}
                  {step.name === 'processing' && execution.started_at && (
                    <span title={`Processing since ${new Date(execution.started_at).toLocaleString()}`}>
                      {formatTimestamp(execution.started_at)}
                    </span>
                  )}
                  {step.name === 'completed' && execution.completed_at && (
                    <span title={`Completed at ${new Date(execution.completed_at).toLocaleString()}`}>
                      {formatTimestamp(execution.completed_at)}
                    </span>
                  )}

                  {/* Duration */}
                  {duration > 0 && step.completed && (
                    <>
                      <span className="text-muted-foreground/60">•</span>
                      <span className="text-muted-foreground">
                        {formatDuration(duration)}
                      </span>
                    </>
                  )}
                </div>
              )}

              {/* Connector line */}
              {index < timelineSteps.length - 1 && (
                <div className="h-px w-3 bg-border flex-shrink-0" />
              )}
            </div>
          );
        })}
      </div>

      <Button
        variant="ghost"
        size="sm"
        onClick={() => navigate(`/workflows/${execution.workflow_id}/enhanced`)}
        className="h-6 px-2 text-body-small hover:text-foreground ml-2"
      >
        <ExternalLink className="w-3 h-3 mr-1" />
        View Workflow
      </Button>
    </div>
  );
}

function HealthIndicator({ execution }: { execution: WorkflowExecution }) {
  const status = normalizeExecutionStatus(execution.status);
  const retryCount = execution.retry_count || 0;
  const duration = execution.duration_ms || 0;

  // Simple health logic
  const getHealthStatus = () => {
    if (status === 'failed') return { type: 'failed', label: 'Failed', color: 'text-red-500' };
    if (retryCount > 0) return { type: 'warning', label: 'Retried', color: 'text-yellow-500' };
    if (duration > 30000) return { type: 'slow', label: 'Slow', color: 'text-yellow-500' }; // > 30s
    if (status === 'succeeded') return { type: 'healthy', label: 'Healthy', color: 'text-green-500' };
    if (status === 'running') return { type: 'running', label: 'Running', color: 'text-blue-500' };
    return { type: 'unknown', label: 'Unknown', color: 'text-muted-foreground' };
  };

  const health = getHealthStatus();
  const HealthIcon = health.type === 'failed' ? AlertCircle :
                    health.type === 'healthy' ? CheckCircle :
                    health.type === 'running' ? Activity :
                    Clock;

  return (
    <div className="flex items-center gap-1">
      <HealthIcon className={cn("w-3 h-3", health.color)} />
      <span className={cn("text-xs font-medium", health.color)}>
        {health.label}
      </span>
    </div>
  );
}

function truncateId(id: string): string {
  return `${id.slice(0, 8)}...${id.slice(-4)}`;
}

export function EnhancedExecutionHeader({
  execution,
  vcStatus,
  vcLoading,
  onClose
}: EnhancedExecutionHeaderProps) {
  const navigate = useNavigate();
  const normalizedStatus = normalizeExecutionStatus(execution.status);
  const statusLabel = normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1);
  const workflowTags = execution.workflow_tags ?? [];

  const handleClose = () => {
    if (onClose) {
      onClose();
    } else {
      navigate("/executions");
    }
  };

  const handleNavigateWorkflow = () => navigate(`/workflows/${execution.workflow_id}/enhanced`);

  return (
    <div className="border-b border-border bg-card/50">
      <div className="px-6 py-4 space-y-3">
        {/* Back Navigation */}
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClose}
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground -ml-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Executions
          </Button>

          <HealthIndicator execution={execution} />
        </div>

        {/* Main Header */}
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-heading-2">{execution.reasoner_id}</h1>
            <StatusIndicator
              status={normalizedStatus}
              animated={normalizedStatus === 'running'}
              className="text-sm"
            />
            <span className="text-body">{statusLabel}</span>
          </div>

          {/* Key Information Row */}
          <div className="flex flex-wrap items-center gap-4 text-body">
            <div className="flex items-center gap-2 group">
              <span>Agent:</span>
              <code className="font-mono text-body-small text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
                {execution.agent_node_id}
              </code>
              <CopyButton
                value={execution.agent_node_id}
                variant="ghost"
                size="icon"
                className="h-4 w-4 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                tooltip="Copy agent node ID"
              />
            </div>

            <div className="flex items-center gap-2">
              <span>DID:</span>
              <DIDDisplay nodeId={execution.agent_node_id} variant="inline" className="text-body-small" />
            </div>

            <div className="flex items-center gap-2 group">
              <span>ID:</span>
              <code className="font-mono text-body-small text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
                {truncateId(execution.execution_id)}
              </code>
              <CopyButton
                value={execution.execution_id}
                variant="ghost"
                size="icon"
                className="h-4 w-4 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                tooltip="Copy execution ID"
              />
            </div>

            {vcLoading ? (
              <div className="flex items-center gap-2">
                <span>VC:</span>
                <span className="text-body-small">Loading…</span>
              </div>
            ) : vcStatus?.has_vc ? (
              <div className="flex items-center gap-2">
                <span>VC:</span>
                <VerifiableCredentialBadge
                  hasVC={vcStatus.has_vc}
                  status={vcStatus.status}
                  vcData={vcStatus as any}
                  executionId={execution.execution_id}
                  showCopyButton={false}
                  showVerifyButton={false}
                />
              </div>
            ) : null}
          </div>

          {/* Workflow Context Row */}
          <div className="flex flex-wrap items-center gap-4 text-body-small">
            <div className="flex items-center gap-2 group">
              <span>Workflow:</span>
              <button
                type="button"
                onClick={handleNavigateWorkflow}
                className="font-medium text-foreground hover:underline flex items-center gap-1"
              >
                {execution.workflow_name ?? truncateId(execution.workflow_id)}
                <ExternalLink className="w-3 h-3" />
              </button>
              <CopyButton
                value={execution.workflow_id}
                variant="ghost"
                size="icon"
                className="h-4 w-4 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                tooltip="Copy workflow ID"
              />
            </div>

            {execution.session_id && (
              <div className="flex items-center gap-2 group">
                <span>Session:</span>
                <code className="font-mono text-xs text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
                  {truncateId(execution.session_id)}
                </code>
                <CopyButton
                  value={execution.session_id}
                  variant="ghost"
                  size="icon"
                  className="h-4 w-4 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                  tooltip="Copy session ID"
                />
              </div>
            )}

            <div className="flex items-center gap-2 group">
              <span>Request:</span>
              <code className="font-mono text-xs text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
                {execution.haxen_request_id ? truncateId(execution.haxen_request_id) : 'n/a'}
              </code>
              {execution.haxen_request_id && (
                <CopyButton
                  value={execution.haxen_request_id}
                  variant="ghost"
                  size="icon"
                  className="h-4 w-4 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                  tooltip="Copy request ID"
                />
              )}
            </div>
          </div>

          {/* Timeline and Tags Row */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <ExecutionTimeline execution={execution} />

            {workflowTags.length > 0 && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-body-small">Tags:</span>
                <div className="flex flex-wrap gap-1">
                  {workflowTags.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs font-normal">
                      {tag}
                    </Badge>
                  ))}
                  {workflowTags.length > 3 && (
                    <Badge variant="outline" className="text-xs font-normal">
                      +{workflowTags.length - 3}
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

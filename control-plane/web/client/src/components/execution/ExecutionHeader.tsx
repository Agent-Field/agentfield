import {
  ArrowLeft,
  Clock,
  ArrowDown,
  ArrowUp,
  RotateCcw,
  RadioTower,
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import type { VCStatusData } from "../../types/did";
import type { CanonicalStatus } from "../../utils/status";
import { DIDDisplay } from "../did/DIDDisplay";
import StatusIndicator from "../ui/status-indicator";
import { VerifiableCredentialBadge } from "../vc";
import { cn } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { CopyButton } from "../ui/copy-button";
import { normalizeExecutionStatus } from "../../utils/status";
import { formatWebhookStatusLabel } from "../../utils/webhook";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";

interface ExecutionHeaderProps {
  execution: WorkflowExecution;
  vcStatus?: VCStatusData | null;
  vcLoading?: boolean;
  onNavigateBack?: () => void;
}

function WebhookStat({
  label,
  value,
  tone = "muted",
}: {
  label: string;
  value: string | number;
  tone?: "success" | "danger" | "muted";
}) {
  const toneClasses: Record<string, string> = {
    success: "text-emerald-500",
    danger: "text-destructive",
    muted: "text-foreground",
  };

  return (
    <div className="flex flex-col gap-1">
      <span className="text-label text-muted-foreground/80">
        {label}
      </span>
      <span className={cn("text-sm font-medium font-mono", toneClasses[tone] ?? toneClasses.muted)}>
        {value}
      </span>
    </div>
  );
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
  if (!bytes) return "—";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

function truncateId(id: string): string {
  return `${id.slice(0, 8)}...${id.slice(-4)}`;
}

function normalizeStatus(status: string): CanonicalStatus {
  return normalizeExecutionStatus(status);
}

export function ExecutionHeader({
  execution,
  vcStatus,
  vcLoading,
  onNavigateBack,
}: ExecutionHeaderProps) {
  const navigate = useNavigate();
  const normalizedStatus = normalizeStatus(execution.status);
  const workflowTags = execution.workflow_tags ?? [];
  const webhookEvents = Array.isArray(execution.webhook_events)
    ? [...execution.webhook_events].sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
    : [];
  const webhookSuccessCount = webhookEvents.filter((event) => {
    const status = event.status?.toLowerCase();
    return status === "succeeded" || status === "delivered" || status === "success";
  }).length;
  const webhookFailureCount = webhookEvents.filter((event) => {
    const status = event.status?.toLowerCase();
    return status === "failed" || Boolean(event.error_message);
  }).length;
  const webhookRegistered = Boolean(
    execution.webhook_registered || webhookEvents.length > 0,
  );
  const webhookPending =
    webhookRegistered && webhookEvents.length === 0 && webhookFailureCount === 0;
  const latestWebhookEvent = webhookEvents[0];
  const latestWebhookTimestamp = latestWebhookEvent
    ? new Date(latestWebhookEvent.created_at).toLocaleString()
    : undefined;

  const webhookBadgeLabel = webhookFailureCount > 0
    ? `${webhookFailureCount} failed`
    : webhookSuccessCount > 0
      ? `${webhookSuccessCount} delivered`
      : webhookPending
        ? "Pending"
        : "Registered";

  const handleNavigateBack = () => {
    if (onNavigateBack) {
      onNavigateBack();
    } else {
      navigate("/executions");
    }
  };

  const handleNavigateWorkflow = () =>
    navigate(`/workflows/${execution.workflow_id}`);
  const handleNavigateSession = () =>
    execution.session_id &&
    navigate(`/executions?session_id=${execution.session_id}`);

  return (
    <div className="space-y-6">
      {/* Breadcrumb Navigation */}
      <div className="flex items-center text-sm text-muted-foreground">
        <button
          onClick={handleNavigateBack}
          className="flex items-center gap-1 hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Executions
        </button>
        <span className="mx-2 text-border">/</span>
        <span className="font-mono text-foreground">{truncateId(execution.execution_id)}</span>
      </div>

      {/* Main Header */}
      <div className="space-y-6">
        {/* Top Row: Title & Status */}
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h1 className="text-heading-2 font-semibold tracking-tight">
                {execution.reasoner_id}
              </h1>
              <StatusIndicator
                status={normalizedStatus}
                animated={normalizedStatus === "running"}
                className="text-sm"
              />
              {webhookRegistered && (
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Badge
                      variant="outline"
                      className={cn(
                        "cursor-pointer gap-1.5 font-mono",
                        webhookFailureCount > 0
                          ? "border-destructive/40 text-destructive"
                          : webhookSuccessCount > 0
                            ? "border-emerald-500/40 text-emerald-500"
                            : "border-border text-muted-foreground",
                      )}
                    >
                      <RadioTower className="h-3 w-3" />
                      {webhookBadgeLabel}
                    </Badge>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80 space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium text-foreground">
                          {webhookPending
                            ? "Awaiting first delivery"
                            : latestWebhookEvent
                              ? `Last webhook ${formatWebhookStatusLabel(latestWebhookEvent.status)}`
                              : "Webhook registered"}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {webhookPending &&
                            "We will display the latest delivery details as soon as the callback is reported."}
                          {!webhookPending && latestWebhookEvent && (
                            <>
                              {formatWebhookStatusLabel(latestWebhookEvent.status)}
                              {latestWebhookEvent.http_status ? ` • HTTP ${latestWebhookEvent.http_status}` : ""}
                            </>
                          )}
                          {!webhookPending && !latestWebhookEvent && "No deliveries recorded yet."}
                        </p>
                      </div>
                      {latestWebhookTimestamp && (
                        <span className="text-xs text-muted-foreground whitespace-nowrap font-mono">
                          {latestWebhookTimestamp}
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/50">
                      <WebhookStat
                        label="Delivered"
                        value={webhookSuccessCount}
                        tone={webhookSuccessCount > 0 ? "success" : "muted"}
                      />
                      <WebhookStat
                        label="Failed"
                        value={webhookFailureCount}
                        tone={webhookFailureCount > 0 ? "danger" : "muted"}
                      />
                      <WebhookStat
                        label="Attempts"
                        value={webhookSuccessCount + webhookFailureCount}
                      />
                    </div>

                    {latestWebhookEvent?.error_message && (
                      <div className="text-xs font-mono text-destructive bg-destructive/5 border border-destructive/20 rounded px-2 py-1.5 break-all">
                        {latestWebhookEvent.error_message}
                      </div>
                    )}
                  </HoverCardContent>
                </HoverCard>
              )}
            </div>

            {/* Tags Row */}
            {workflowTags.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {workflowTags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="font-normal text-[10px] px-1.5 py-0">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Primary Actions / VC Status */}
          <div className="flex items-center gap-2">
             {vcLoading ? (
                <Badge variant="outline" className="animate-pulse">Loading VC...</Badge>
              ) : vcStatus?.has_vc ? (
                <VerifiableCredentialBadge
                  hasVC={vcStatus.has_vc}
                  status={vcStatus.status}
                  vcData={vcStatus}
                  executionId={execution.execution_id}
                  showCopyButton={false}
                  showVerifyButton={false}
                />
              ) : null}
          </div>
        </div>

        {/* Metadata Grid - The "Developer Dashboard" Look */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 rounded-lg border border-border/60 bg-muted/20">
          {/* Column 1: Identity */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="text-label">Execution ID</div>
              <div className="flex items-center gap-2 group">
                <code className="font-mono text-xs text-foreground">
                  {execution.execution_id}
                </code>
                <CopyButton
                  value={execution.execution_id}
                  variant="ghost"
                  size="icon-sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-label">Agent Node</div>
              <div className="flex items-center gap-2 group">
                <code className="font-mono text-xs text-foreground">
                  {truncateId(execution.agent_node_id)}
                </code>
                <DIDDisplay
                  nodeId={execution.agent_node_id}
                  variant="inline"
                  className="text-[10px]"
                />
                <CopyButton
                  value={execution.agent_node_id}
                  variant="ghost"
                  size="icon-sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>
            </div>
          </div>

          {/* Column 2: Context */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="text-label">Workflow</div>
              <div className="flex items-center gap-2 group">
                <button
                  type="button"
                  onClick={handleNavigateWorkflow}
                  className="font-medium text-sm text-foreground hover:underline truncate max-w-[180px] text-left"
                >
                  {execution.workflow_name ?? truncateId(execution.workflow_id)}
                </button>
                <CopyButton
                  value={execution.workflow_id}
                  variant="ghost"
                  size="icon-sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>
            </div>
            {execution.session_id && (
              <div className="space-y-1">
                <div className="text-label">Session</div>
                <div className="flex items-center gap-2 group">
                  <button
                    type="button"
                    onClick={handleNavigateSession}
                    className="font-mono text-xs text-foreground hover:underline"
                  >
                    {truncateId(execution.session_id)}
                  </button>
                  <CopyButton
                    value={execution.session_id}
                    variant="ghost"
                    size="icon-sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Column 3: Performance */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="text-label">Duration</div>
              <div className="flex items-center gap-1.5 text-sm font-medium font-mono">
                <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                {formatDuration(execution.duration_ms)}
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-label">Retries</div>
              <div className="flex items-center gap-1.5 text-sm font-medium font-mono">
                <RotateCcw className="w-3.5 h-3.5 text-muted-foreground" />
                {execution.retry_count}
              </div>
            </div>
          </div>

          {/* Column 4: I/O */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="text-label">Data Transfer</div>
              <div className="flex items-center gap-3 text-xs font-mono">
                <div className="flex items-center gap-1">
                  <ArrowDown className="w-3 h-3 text-muted-foreground" />
                  <span>{formatBytes(execution.input_size)}</span>
                </div>
                <div className="flex items-center gap-1">
                  <ArrowUp className="w-3 h-3 text-muted-foreground" />
                  <span>{formatBytes(execution.output_size)}</span>
                </div>
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-label">Request ID</div>
              <div className="flex items-center gap-2 group">
                <code className="font-mono text-xs text-muted-foreground">
                  {execution.agentfield_request_id
                    ? truncateId(execution.agentfield_request_id)
                    : "n/a"}
                </code>
                {execution.agentfield_request_id && (
                  <CopyButton
                    value={execution.agentfield_request_id}
                    variant="ghost"
                    size="icon-sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

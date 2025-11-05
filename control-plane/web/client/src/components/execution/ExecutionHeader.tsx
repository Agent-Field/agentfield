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
import type { CanonicalStatus } from "../../utils/status";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
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
  vcStatus?: {
    has_vc: boolean;
    vc_id?: string;
    status: string;
    created_at?: string;
    vc_document?: any;
  } | null;
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
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground/80">
        {label}
      </span>
      <span className={cn("text-sm font-medium", toneClasses[tone] ?? toneClasses.muted)}>
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
  const statusLabel =
    normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1);
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
        ? "Pending webhook"
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
    <div className="space-y-4">
      {/* Back Navigation */}
      <div className="flex items-center">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleNavigateBack}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground -ml-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Executions
        </Button>
      </div>

      {/* Main Header - Clean Linear Style */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3 text-body">
          <h1 className="text-heading-1">
            {execution.reasoner_id}
          </h1>
          <StatusIndicator
            status={normalizedStatus}
            animated={normalizedStatus === "running"}
            className="text-base"
          />
          <span className="text-body">{statusLabel}</span>
          {webhookRegistered && (
            <HoverCard>
              <HoverCardTrigger asChild>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-xs flex items-center gap-1 cursor-pointer",
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
                    <p className="text-body font-medium text-text-primary">
                      {webhookPending
                        ? "Awaiting first delivery"
                        : latestWebhookEvent
                          ? `Last webhook ${formatWebhookStatusLabel(latestWebhookEvent.status)}`
                          : "Webhook registered"}
                    </p>
                    <p className="text-body-small">
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
                    <span className="text-body-small text-muted-foreground whitespace-nowrap">
                      {latestWebhookTimestamp}
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-2">
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
                    label={"Attempts"}
                    value={webhookSuccessCount + webhookFailureCount}
                  />
                </div>

                {latestWebhookEvent?.error_message && (
                  <div className="text-body-small text-destructive bg-destructive/10 border border-destructive/20 rounded px-3 py-2">
                    {latestWebhookEvent.error_message}
                  </div>
                )}
              </HoverCardContent>
            </HoverCard>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-4 text-body-small">
          <div className="flex items-center gap-2 group">
            <span>Agent:</span>
            <code className="font-mono text-xs text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
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
            <DIDDisplay
              nodeId={execution.agent_node_id}
              variant="inline"
              className="text-xs"
            />
          </div>

          <div className="flex items-center gap-2 group">
            <span>ID:</span>
            <code className="font-mono text-xs text-foreground bg-muted/30 px-1.5 py-0.5 rounded">
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

        <div className="flex flex-wrap items-center gap-4 text-body-small">
          <div className="flex items-center gap-2 group">
            <span>Workflow:</span>
            <button
              type="button"
              onClick={handleNavigateWorkflow}
              className="font-medium text-foreground hover:underline"
            >
              {execution.workflow_name ?? truncateId(execution.workflow_id)}
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
              <button
                type="button"
                onClick={handleNavigateSession}
                className="font-medium text-foreground hover:underline"
              >
                {truncateId(execution.session_id)}
              </button>
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
              {execution.haxen_request_id
                ? truncateId(execution.haxen_request_id)
                : "n/a"}
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

        {workflowTags.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-body-small">
            <span>Tags:</span>
            <div className="flex flex-wrap gap-2">
              {workflowTags.map((tag) => (
                <Badge key={tag} variant="secondary" className="font-normal">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-6 pt-2 text-sm">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Duration:</span>
            <span className="font-medium text-foreground">
              {formatDuration(execution.duration_ms)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <ArrowDown className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Input:</span>
            <span className="font-medium text-foreground">
              {formatBytes(execution.input_size)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <ArrowUp className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Output:</span>
            <span className="font-medium text-foreground">
              {formatBytes(execution.output_size)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <RotateCcw className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">Retries:</span>
            <span className="font-medium text-foreground">
              {execution.retry_count}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

import {
  ArrowLeft,
  ExternalLink,
  Clock,
  RotateCcw,
  PauseCircle,
} from "@/components/ui/icon-bridge";
import { ArrowDown, ArrowUp } from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
import { CopyButton } from "../ui/copy-button";
import StatusIndicator from "../ui/status-indicator";
import { VerifiableCredentialBadge } from "../vc";
import { cn } from "../../lib/utils";
import { normalizeExecutionStatus } from "../../utils/status";

interface CompactExecutionHeaderProps {
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

function formatDuration(durationMs?: number): string {
  if (!durationMs) return "—";
  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
  const minutes = Math.floor(durationMs / 60000);
  const seconds = Math.floor((durationMs % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

function truncateId(id: string): string {
  return `${id.slice(0, 8)}...${id.slice(-4)}`;
}

export function CompactExecutionHeader({
  execution,
  vcStatus,
  vcLoading,
  onClose,
}: CompactExecutionHeaderProps) {
  const navigate = useNavigate();
  const normalizedStatus = normalizeExecutionStatus(execution.status);

  const duration = execution.duration_ms || 0;
  const retryCount = execution.retry_count || 0;
  const inputSize = execution.input_size || 0;
  const outputSize = execution.output_size || 0;

  // Performance indicator
  const getPerformanceColor = () => {
    if (normalizedStatus === "failed") return "text-red-500";
    if (retryCount > 0) return "text-yellow-500";
    if (duration > 30000) return "text-yellow-500";
    if (normalizedStatus === "succeeded") return "text-green-500";
    return "text-foreground";
  };

  const handleClose = () => {
    if (onClose) {
      onClose();
    } else {
      navigate("/executions");
    }
  };

  const handleNavigateWorkflow = () =>
    navigate(`/workflows/${execution.workflow_id}`);

  return (
    <div className="h-12 border-b border-border bg-card/50 flex-shrink-0">
      <div className="h-full flex items-center justify-between gap-4 px-6 min-w-0">
        {/* Left Section: Back + Title + Status */}
        <div className="flex items-center gap-3 min-w-0 flex-shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClose}
            className="h-8 w-8 p-0 flex-shrink-0"
            title="Back to Executions"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>

          <div className="flex items-center gap-2 min-w-0">
            <h1 className="text-sm font-semibold text-foreground">
              {execution.reasoner_id}
            </h1>
            <StatusIndicator
              status={normalizedStatus}
              animated={normalizedStatus === "running" || normalizedStatus === "waiting"}
              className="text-xs flex-shrink-0"
            />
          </div>
        </div>

        {/* Center Section: Key Information */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-body-small overflow-hidden flex-1 min-w-0">
          {/* Agent */}
          <div className="hidden lg:flex items-center gap-1.5 group flex-shrink-0 min-w-0">
            <span>Agent:</span>
            <code className="font-mono text-foreground bg-muted/30 px-1 py-0.5 rounded whitespace-nowrap max-w-[180px] truncate">
              {execution.agent_node_id}
            </code>
            <CopyButton
              value={execution.agent_node_id}
              tooltip="Copy agent node ID"
              className="opacity-0 transition-opacity duration-150 group-hover:opacity-100 h-6 w-6 rounded-md [&_svg]:!h-3 [&_svg]:!w-3"
            />
          </div>

          {/* DID */}
          <div className="hidden xl:flex items-center gap-1.5 flex-shrink-0 min-w-0">
            <span>DID:</span>
            <DIDDisplay
              nodeId={execution.agent_node_id}
              variant="inline"
              className="text-xs"
            />
          </div>

          {/* Execution ID */}
          <div className="hidden md:flex items-center gap-1.5 group flex-shrink-0 min-w-0">
            <span>ID:</span>
            <code className="font-mono text-foreground bg-muted/30 px-1 py-0.5 rounded whitespace-nowrap">
              {truncateId(execution.execution_id)}
            </code>
            <CopyButton
              value={execution.execution_id}
              tooltip="Copy execution ID"
              className="opacity-0 transition-opacity duration-150 group-hover:opacity-100 h-6 w-6 rounded-md [&_svg]:!h-3 [&_svg]:!w-3"
            />
          </div>

          {/* Workflow */}
          <div className="hidden lg:flex items-center gap-1.5 group flex-shrink-0 min-w-0">
            <span>Workflow:</span>
            <button
              type="button"
              onClick={handleNavigateWorkflow}
              className="font-medium text-foreground hover:underline flex items-center gap-1 truncate"
            >
              <span className="truncate">
                {execution.workflow_name ?? truncateId(execution.workflow_id)}
              </span>
              <ExternalLink className="w-3 h-3" />
            </button>
            <CopyButton
              value={execution.workflow_id}
              tooltip="Copy workflow ID"
              className="opacity-0 transition-opacity duration-150 group-hover:opacity-100 h-6 w-6 rounded-md [&_svg]:!h-3 [&_svg]:!w-3"
            />
          </div>

          {/* Divider */}
          <div className="hidden md:block h-4 w-px bg-border flex-shrink-0" />

          {/* Duration */}
          <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
            <Clock className="w-3 h-3" />
            <span className={cn("font-medium", getPerformanceColor())}>
              {formatDuration(duration)}
            </span>
            {duration > 30000 && normalizedStatus === "succeeded" && (
              <span className="text-yellow-500">(Slow)</span>
            )}
          </div>

          {/* Input */}
          <div className="hidden md:flex items-center gap-1.5 flex-shrink-0">
            <ArrowDown className="w-3 h-3" />
            <span className="font-medium text-foreground">
              {formatBytes(inputSize)}
            </span>
          </div>

          {/* Output */}
          <div className="hidden md:flex items-center gap-1.5 flex-shrink-0">
            <ArrowUp className="w-3 h-3" />
            <span className="font-medium text-foreground">
              {formatBytes(outputSize)}
            </span>
          </div>

          {/* Retries */}
          {retryCount > 0 && (
            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              <RotateCcw className="w-3 h-3" />
              <span className="font-medium text-yellow-500">{retryCount}</span>
            </div>
          )}

          {/* Status Reason */}
          {execution.status_reason && normalizedStatus !== "waiting" && (
            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              <span className="text-muted-foreground">{execution.status_reason.replace(/_/g, " ")}</span>
            </div>
          )}

          {/* Approval Status */}
          {normalizedStatus === "waiting" && execution.approval_request_url && (
            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              <PauseCircle className="w-3 h-3 text-amber-500" />
              <a
                href={execution.approval_request_url}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-amber-500 hover:text-amber-400 hover:underline flex items-center gap-1"
              >
                Awaiting Approval
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}

          {/* Waiting with status_reason but no approval URL */}
          {normalizedStatus === "waiting" && !execution.approval_request_url && execution.status_reason && (
            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              <PauseCircle className="w-3 h-3 text-amber-500" />
              <span className="font-medium text-amber-500">
                {execution.status_reason.replace(/_/g, " ")}
              </span>
            </div>
          )}

          {/* VC Badge */}
          {!vcLoading && vcStatus?.has_vc && (
            <div className="hidden lg:block flex-shrink-0">
              <VerifiableCredentialBadge
                hasVC={vcStatus.has_vc}
                status={vcStatus.status}
                vcData={vcStatus as any}
                executionId={execution.execution_id}
                showCopyButton={false}
                showVerifyButton={false}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

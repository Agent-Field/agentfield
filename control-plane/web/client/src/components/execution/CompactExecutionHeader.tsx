import { useState, useEffect } from "react";
import {
  ArrowLeft,
  ExternalLink,
  Clock,
  RotateCcw,
  PauseCircle,
  Activity,
  XCircle,
  Play,
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
import { CopyButton } from "../ui/copy-button";
import { Badge } from "../ui/badge";
import { VerifiableCredentialBadge } from "../vc";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../ui/alert-dialog";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";
import { cn } from "../../lib/utils";
import {
  normalizeExecutionStatus,
  getStatusLabel,
  getStatusTheme,
  isPausedStatus,
  isTerminalStatus,
} from "../../utils/status";
import {
  cancelExecution,
  pauseExecution,
  resumeExecution,
} from "../../services/executionsApi";
import {
  useErrorNotification,
  useSuccessNotification,
} from "../ui/notification";

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
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

function formatDuration(durationMs?: number | null): string {
  if (!durationMs) return "\u2014";
  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
  const minutes = Math.floor(durationMs / 60000);
  const seconds = Math.floor((durationMs % 60000) / 1000);
  if (durationMs < 3600000) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(durationMs / 3600000);
  const remainingMinutes = Math.floor((durationMs % 3600000) / 60000);
  return `${hours}h ${remainingMinutes}m`;
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

/** Live elapsed-time counter for non-terminal executions. */
function useLiveElapsed(startedAt?: string, status?: string): number | null {
  const normalized = normalizeExecutionStatus(status);
  const isActive = normalized === "running";
  const isNonTerminal =
    !isTerminalStatus(status) && normalized !== "unknown";

  const [elapsed, setElapsed] = useState<number | null>(() => {
    if (!startedAt) return null;
    return Math.max(0, Date.now() - new Date(startedAt).getTime());
  });

  useEffect(() => {
    if (!startedAt || !isNonTerminal) {
      setElapsed(null);
      return;
    }

    const compute = () =>
      Math.max(0, Date.now() - new Date(startedAt).getTime());

    if (isActive) {
      const update = () => setElapsed(compute());
      update();
      const id = setInterval(update, 1000);
      return () => clearInterval(id);
    }

    setElapsed(compute());
  }, [startedAt, isActive, isNonTerminal]);

  return elapsed;
}

export function CompactExecutionHeader({
  execution,
  vcStatus,
  vcLoading,
  onClose,
  onRefresh,
  isRefreshing,
}: CompactExecutionHeaderProps) {
  const navigate = useNavigate();
  const normalizedStatus = normalizeExecutionStatus(execution.status);
  const statusTheme = getStatusTheme(normalizedStatus);
  const isRunning = normalizedStatus === "running";
  const isPaused = isPausedStatus(normalizedStatus);
  const showControls = isRunning || isPaused;

  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isPausing, setIsPausing] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const isMutating = isCancelling || isPausing || isResuming;

  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  const liveElapsed = useLiveElapsed(execution.started_at, execution.status);
  const displayDuration = isTerminalStatus(execution.status)
    ? execution.duration_ms
    : liveElapsed;

  const retryCount = execution.retry_count || 0;
  const inputSize = execution.input_size || 0;
  const outputSize = execution.output_size || 0;

  const getPerformanceColor = () => {
    if (normalizedStatus === "failed") return "text-red-500";
    if (retryCount > 0) return "text-yellow-500";
    if (displayDuration && displayDuration > 30000) return "text-yellow-500";
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

  const handlePause = async () => {
    if (isMutating) return;
    try {
      setIsPausing(true);
      await pauseExecution(execution.execution_id);
      showSuccess(
        "Execution paused",
        `Execution ${execution.execution_id.slice(0, 8)} has been paused.`,
      );
      onRefresh?.();
    } catch (error) {
      showError(
        "Pause failed",
        error instanceof Error ? error.message : "Unable to pause execution.",
      );
    } finally {
      setIsPausing(false);
    }
  };

  const handleResume = async () => {
    if (isMutating) return;
    try {
      setIsResuming(true);
      await resumeExecution(execution.execution_id);
      showSuccess(
        "Execution resumed",
        `Execution ${execution.execution_id.slice(0, 8)} is running again.`,
      );
      onRefresh?.();
    } catch (error) {
      showError(
        "Resume failed",
        error instanceof Error
          ? error.message
          : "Unable to resume execution.",
      );
    } finally {
      setIsResuming(false);
    }
  };

  const handleCancel = async () => {
    if (isMutating) return;
    try {
      setIsCancelling(true);
      await cancelExecution(execution.execution_id);
      showSuccess(
        "Execution cancelled",
        `Execution ${execution.execution_id.slice(0, 8)} has been cancelled.`,
      );
      setCancelDialogOpen(false);
      onRefresh?.();
    } catch (error) {
      showError(
        "Cancel failed",
        error instanceof Error
          ? error.message
          : "Unable to cancel execution.",
      );
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <div className="bg-background border-b border-border px-4 h-12 flex items-center justify-between">
      {/* Left: Back + Status + Name */}
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClose}
          className="h-8 w-8 p-0 flex-shrink-0"
          title="Back to Executions"
        >
          <ArrowLeft className="w-4 h-4" />
        </Button>

        {/* Status indicator + label + badges */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div
            className={cn(
              "w-2 h-2 rounded-full",
              statusTheme.indicatorClass,
              isRunning && "animate-pulse",
            )}
          />
          <span
            className={cn(
              "text-sm font-medium whitespace-nowrap",
              statusTheme.textClass,
            )}
          >
            {getStatusLabel(normalizedStatus)}
          </span>

          {/* LIVE badge for running executions */}
          {isRunning && (
            <Badge
              variant="outline"
              className="h-5 px-1.5 text-[10px] font-semibold tracking-wider border-transparent bg-emerald-500/10 text-emerald-500"
            >
              LIVE
            </Badge>
          )}

          {/* Approval required badge for waiting executions */}
          {normalizedStatus === "waiting" &&
            execution.approval_request_url && (
              <a
                href={execution.approval_request_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center"
              >
                <Badge
                  variant="outline"
                  className="h-5 px-1.5 text-[10px] font-semibold tracking-wider border-amber-500/40 text-amber-500 hover:bg-amber-500/10 cursor-pointer"
                >
                  Approval Required
                  <ExternalLink className="w-3 h-3 ml-1" />
                </Badge>
              </a>
            )}
        </div>

        <div className="hidden sm:block w-px h-4 bg-border flex-shrink-0" />

        {/* Name + subtitle (desktop) */}
        <div className="min-w-0 flex-1 hidden sm:block">
          <h1 className="text-heading-3 text-foreground truncate">
            {execution.reasoner_id}
          </h1>
          <div className="flex items-center gap-2 text-body-small">
            <span className="truncate max-w-[140px]">
              {execution.agent_node_id}
            </span>
            <span>&bull;</span>
            <Clock className="w-3 h-3 flex-shrink-0" />
            <span className={cn("font-medium", getPerformanceColor())}>
              {formatDuration(displayDuration)}
            </span>
            {isRunning && displayDuration != null && (
              <span className="text-emerald-500 text-[10px]">&blacktriangle;</span>
            )}
            {retryCount > 0 && (
              <>
                <span>&bull;</span>
                <span className="text-yellow-500">
                  {retryCount} {retryCount === 1 ? "retry" : "retries"}
                </span>
              </>
            )}
            {execution.status_reason &&
              !(
                normalizedStatus === "waiting" &&
                execution.approval_request_url
              ) && (
                <>
                  <span>&bull;</span>
                  <span className="text-muted-foreground truncate max-w-[120px]">
                    {execution.status_reason.replace(/_/g, " ")}
                  </span>
                </>
              )}
          </div>
        </div>

        {/* Name only (mobile) */}
        <div className="min-w-0 flex-1 sm:hidden">
          <h1 className="text-sm font-semibold text-foreground truncate">
            {execution.reasoner_id}
          </h1>
        </div>

        {/* Execution ID hover card with full details */}
        <div className="hidden md:block flex-shrink-0">
          <HoverCard>
            <HoverCardTrigger asChild>
              <div className="flex items-center gap-1.5 cursor-pointer">
                <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                  {execution.execution_id.slice(0, 8)}...
                </code>
                <CopyButton
                  value={execution.execution_id}
                  tooltip="Copy execution ID"
                  className="h-6 w-6 rounded-md [&_svg]:!h-3 [&_svg]:!w-3"
                />
              </div>
            </HoverCardTrigger>
            <HoverCardContent className="w-80">
              <div className="space-y-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    Execution Details
                  </p>
                  <code className="text-xs font-mono text-muted-foreground break-all">
                    {execution.execution_id}
                  </code>
                </div>

                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <div>
                    <span className="text-muted-foreground block">Agent</span>
                    <span className="font-mono text-foreground truncate block">
                      {execution.agent_node_id}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block">
                      Workflow
                    </span>
                    <button
                      type="button"
                      onClick={handleNavigateWorkflow}
                      className="font-medium text-foreground hover:underline flex items-center gap-1"
                    >
                      <span className="truncate">
                        {execution.workflow_name ??
                          truncateId(execution.workflow_id)}
                      </span>
                      <ExternalLink className="w-3 h-3 flex-shrink-0" />
                    </button>
                  </div>
                  <div>
                    <span className="text-muted-foreground block">Input</span>
                    <span className="text-foreground">
                      {formatBytes(inputSize)}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block">Output</span>
                    <span className="text-foreground">
                      {formatBytes(outputSize)}
                    </span>
                  </div>
                  {execution.workflow_depth > 0 && (
                    <div>
                      <span className="text-muted-foreground block">
                        Depth
                      </span>
                      <span className="text-foreground">
                        {execution.workflow_depth}
                      </span>
                    </div>
                  )}
                  {retryCount > 0 && (
                    <div>
                      <span className="text-muted-foreground block">
                        Retries
                      </span>
                      <span className="text-yellow-500 font-medium">
                        {retryCount}
                      </span>
                    </div>
                  )}
                </div>

                <div className="border-t border-border pt-2">
                  <span className="text-xs text-muted-foreground block mb-1">
                    DID
                  </span>
                  <DIDDisplay
                    nodeId={execution.agent_node_id}
                    variant="inline"
                    className="text-xs"
                  />
                </div>

                {!vcLoading && vcStatus?.has_vc && (
                  <div className="border-t border-border pt-2">
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
            </HoverCardContent>
          </HoverCard>
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-1 flex-shrink-0">
        {/* Execution lifecycle controls */}
        {showControls && (
          <>
            {isRunning && (
              <Button
                variant="ghost"
                size="sm"
                disabled={isMutating}
                onClick={handlePause}
                className="h-8 w-8 p-0 hover:bg-amber-500/10 hover:text-amber-600"
                title="Pause execution"
              >
                {isPausing ? (
                  <Activity className="w-4 h-4 animate-spin" />
                ) : (
                  <PauseCircle className="w-4 h-4" />
                )}
              </Button>
            )}

            {isPaused && (
              <Button
                variant="ghost"
                size="sm"
                disabled={isMutating}
                onClick={handleResume}
                className="h-8 w-8 p-0 hover:bg-emerald-500/10 hover:text-emerald-600"
                title="Resume execution"
              >
                {isResuming ? (
                  <Activity className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
              </Button>
            )}

            <AlertDialog
              open={cancelDialogOpen}
              onOpenChange={setCancelDialogOpen}
            >
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={isMutating}
                  className="h-8 w-8 p-0 hover:bg-destructive/10 hover:text-destructive"
                  title="Cancel execution"
                >
                  {isCancelling ? (
                    <Activity className="w-4 h-4 animate-spin" />
                  ) : (
                    <XCircle className="w-4 h-4" />
                  )}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Cancel execution?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will stop the execution immediately. This action
                    cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={isCancelling}>
                    Keep running
                  </AlertDialogCancel>
                  <AlertDialogAction
                    disabled={isCancelling}
                    onClick={handleCancel}
                  >
                    {isCancelling ? "Cancelling\u2026" : "Cancel execution"}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>

            <div className="w-px h-4 bg-border mx-0.5" />
          </>
        )}

        {/* Refresh */}
        {onRefresh && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            disabled={isRefreshing}
            className="h-8 w-8 p-0"
            title="Refresh execution (Cmd/Ctrl + R)"
          >
            <RotateCcw
              className={cn("w-4 h-4", isRefreshing && "animate-spin")}
            />
          </Button>
        )}
      </div>
    </div>
  );
}

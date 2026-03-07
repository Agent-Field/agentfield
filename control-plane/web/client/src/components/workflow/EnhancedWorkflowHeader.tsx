import { useEffect, useMemo, useState } from "react";
import { useIsMobile } from "@/hooks/use-mobile";
import { formatDurationHumanReadable } from "@/components/ui/data-formatters";
import {
  ArrowLeft,
  RotateCcw,
  Maximize,
  Minimize,
  Activity,
  Copy,
  Check,
  RadioTower,
  XCircle,
  PauseCircle,
  Play,
} from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
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
  getStatusLabel,
  getStatusTheme,
  isPausedStatus,
  normalizeExecutionStatus,
} from "../../utils/status";
import { summarizeWorkflowWebhook, formatWebhookStatusLabel } from "../../utils/webhook";
import type { WorkflowSummary } from "../../types/workflows";
import {
  cancelExecution,
  pauseExecution,
  resumeExecution,
} from "../../services/executionsApi";
import {
  useErrorNotification,
  useSuccessNotification,
} from "../ui/notification";

interface EnhancedWorkflowHeaderProps {
  workflow: WorkflowSummary;
  dagData?: any;
  isLiveUpdating?: boolean;
  hasRunningWorkflows?: boolean;
  pollingInterval?: number;
  isRefreshing?: boolean;
  onRefresh?: () => void;
  onClose?: () => void;
  isFullscreen: boolean;
  onFullscreenChange: (enabled: boolean) => void;
  selectedNodeCount: number;
}

export function EnhancedWorkflowHeader({
  workflow,
  dagData,
  isLiveUpdating: _isLiveUpdating,
  hasRunningWorkflows: _hasRunningWorkflows,
  pollingInterval: _pollingInterval,
  isRefreshing,
  onRefresh,
  onClose,
  isFullscreen,
  onFullscreenChange,
  selectedNodeCount
}: EnhancedWorkflowHeaderProps) {
  const [copied, setCopied] = useState(false);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isPausing, setIsPausing] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const isMobile = useIsMobile();
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  const normalizedStatus = normalizeExecutionStatus(workflow.status);
  const isRunning = normalizedStatus === "running";
  const isPaused = isPausedStatus(normalizedStatus);

  const [liveElapsed, setLiveElapsed] = useState<number | null>(null);
  useEffect(() => {
    if (isRunning && workflow.started_at) {
      const update = () =>
        setLiveElapsed(
          Math.max(0, Date.now() - new Date(workflow.started_at).getTime()),
        );
      update();
      const id = setInterval(update, 1000);
      return () => clearInterval(id);
    }
    if (isPaused && workflow.started_at) {
      setLiveElapsed(
        Math.max(0, Date.now() - new Date(workflow.started_at).getTime()),
      );
      return;
    }
    setLiveElapsed(null);
  }, [isRunning, isPaused, workflow.started_at]);
  const displayDuration = liveElapsed ?? workflow.duration_ms;
  const executionId =
    workflow.root_execution_id ??
    (workflow as WorkflowSummary & { execution_id?: string }).execution_id ??
    dagData?.timeline?.[0]?.execution_id;
  const isMutating = isCancelling || isPausing || isResuming;
  const statusTheme = getStatusTheme(normalizedStatus);
  const statusCounts = workflow.status_counts ?? {};
  const activeExecutions = workflow.active_executions ?? 0;
  const failedExecutions = (statusCounts.failed ?? 0) + (statusCounts.timeout ?? 0);
  const webhookSummary = useMemo(
    () => summarizeWorkflowWebhook(dagData?.timeline),
    [dagData?.timeline],
  );
  const hasWebhookInsights = webhookSummary.nodesWithWebhook > 0;
  const webhookBadgeLabel = webhookSummary.failedDeliveries > 0
    ? `${webhookSummary.failedDeliveries} webhook ${webhookSummary.failedDeliveries === 1 ? "issue" : "issues"}`
    : webhookSummary.successDeliveries > 0
      ? `${webhookSummary.successDeliveries} delivered`
      : `${webhookSummary.nodesWithWebhook} webhook${webhookSummary.nodesWithWebhook === 1 ? "" : "s"}`;
  const webhookBadgeClasses = cn(
    "text-xs flex items-center gap-1 cursor-pointer",
    webhookSummary.failedDeliveries > 0
      ? "border-destructive/40 text-destructive"
      : webhookSummary.successDeliveries > 0
        ? "border-emerald-500/40 text-emerald-500"
        : "border-border text-muted-foreground",
  );
  const latestWebhookTimestamp = webhookSummary.lastSentAt
    ? new Date(webhookSummary.lastSentAt).toLocaleString()
    : undefined;

  const getStatusIcon = () => (
    <div
      className={cn(
        "w-2 h-2 rounded-full",
        statusTheme.indicatorClass,
        normalizedStatus === "running" && "animate-pulse"
      )}
    />
  );

  const formatDuration = formatDurationHumanReadable;

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(workflow.workflow_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy workflow ID:', err);
    }
  };

  const handlePause = async () => {
    if (!executionId || isMutating) {
      return;
    }
    try {
      setIsPausing(true);
      await pauseExecution(executionId);
      showSuccess("Execution paused", `Execution ${executionId.slice(0, 8)} has been paused.`);
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
    if (!executionId || isMutating) {
      return;
    }
    try {
      setIsResuming(true);
      await resumeExecution(executionId);
      showSuccess("Execution resumed", `Execution ${executionId.slice(0, 8)} is running again.`);
      onRefresh?.();
    } catch (error) {
      showError(
        "Resume failed",
        error instanceof Error ? error.message : "Unable to resume execution.",
      );
    } finally {
      setIsResuming(false);
    }
  };

  const handleCancel = async () => {
    if (!executionId || isMutating) {
      return;
    }
    try {
      setIsCancelling(true);
      await cancelExecution(executionId);
      showSuccess("Execution cancelled", `Execution ${executionId.slice(0, 8)} has been cancelled.`);
      setCancelDialogOpen(false);
      onRefresh?.();
    } catch (error) {
      showError(
        "Cancel failed",
        error instanceof Error ? error.message : "Unable to cancel execution.",
      );
    } finally {
      setIsCancelling(false);
    }
  };


  return (
    <div className={cn(
      "bg-background border-b border-border px-4",
      isMobile ? "py-2 min-h-12" : "h-12",
      "flex items-center",
      isMobile ? "flex-col gap-2" : "justify-between"
    )}>
      {/* Top Row: Main Content */}
      <div className={cn(
        "flex items-center",
        isMobile ? "w-full justify-between gap-2" : "gap-3 min-w-0 flex-1"
      )}>
        {/* Left: Navigation & Core Info */}
        <div className={cn(
          "flex items-center",
          isMobile ? "gap-2 min-w-0 flex-1" : "gap-3 min-w-0 flex-1"
        )}>
          {onClose && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0 flex-shrink-0"
              title="Back to workflows"
            >
              <ArrowLeft className="w-4 h-4" />
            </Button>
          )}

          {/* Status & Name */}
          <div className={cn(
            "flex items-center min-w-0",
            isMobile ? "gap-2 flex-1" : "gap-3"
          )}>
            <div className={cn(
              "flex items-center min-w-0",
              isMobile ? "gap-1.5 flex-wrap" : "gap-2"
            )}>
              {getStatusIcon()}
              <span className={cn("text-sm font-medium whitespace-nowrap", statusTheme.textClass)}>
                {getStatusLabel(normalizedStatus)}
              </span>

              {(activeExecutions > 0 || failedExecutions > 0) && !isMobile && (
                <div className="flex items-center gap-2">
                  {activeExecutions > 0 && (
                    <Badge variant="secondary" className="h-5 px-2 text-body-small">
                      {activeExecutions} active
                    </Badge>
                  )}
                  {failedExecutions > 0 && (
                    <Badge variant="destructive" className="h-5 px-2 text-body-small">
                      {failedExecutions} issues
                    </Badge>
                  )}
                </div>
              )}
              {hasWebhookInsights && !isMobile && (
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Badge variant="outline" className={webhookBadgeClasses}>
                      <RadioTower className="h-3 w-3" />
                      {webhookBadgeLabel}
                    </Badge>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80 space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-foreground">
                          {webhookSummary.failedDeliveries > 0
                            ? "Webhook attention required"
                            : webhookSummary.successDeliveries > 0
                              ? "Webhook activity"
                              : "Webhook registered"}
                        </p>
                        <p className="text-body-small">
                          {webhookSummary.totalDeliveries > 0
                            ? `${webhookSummary.totalDeliveries} deliveries • ${webhookSummary.successDeliveries} succeeded`
                            : webhookSummary.pendingNodes > 0
                              ? `${webhookSummary.pendingNodes} pending`
                              : "Awaiting first delivery."}
                        </p>
                      </div>
                      {latestWebhookTimestamp && (
                        <span className="text-body-small text-muted-foreground whitespace-nowrap">
                          {latestWebhookTimestamp}
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="flex flex-col gap-1">
                        <span className="uppercase tracking-wide text-[10px] text-muted-foreground/80">
                          Nodes
                        </span>
                        <span className="text-sm font-medium text-foreground">
                          {webhookSummary.nodesWithWebhook}
                        </span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="uppercase tracking-wide text-[10px] text-muted-foreground/80">
                          Delivered
                        </span>
                        <span className="text-sm font-medium text-emerald-500">
                          {webhookSummary.successDeliveries}
                        </span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="uppercase tracking-wide text-[10px] text-muted-foreground/80">
                          Failed
                        </span>
                        <span className={cn(
                          "text-sm font-medium",
                          webhookSummary.failedDeliveries > 0
                            ? "text-destructive"
                            : "text-foreground",
                        )}>
                          {webhookSummary.failedDeliveries}
                        </span>
                      </div>
                    </div>

                    {webhookSummary.lastStatus && (
                      <div className="text-body-small">
                        <span className="font-medium text-foreground">Last status:</span>{" "}
                        {formatWebhookStatusLabel(webhookSummary.lastStatus)}
                        {webhookSummary.lastHttpStatus && (
                          <span className="ml-1">• HTTP {webhookSummary.lastHttpStatus}</span>
                        )}
                      </div>
                    )}

                    {webhookSummary.lastError && (
                      <div className="text-body-small text-destructive bg-destructive/10 border border-destructive/20 rounded px-3 py-2">
                        {webhookSummary.lastError}
                      </div>
                    )}
                  </HoverCardContent>
                </HoverCard>
              )}
            </div>

            {!isMobile && <div className="w-px h-4 bg-border" />}

            <div className="min-w-0 flex-1 flex items-baseline gap-2">
              <h1 className={cn(
                "text-foreground truncate flex-shrink-0",
                isMobile ? "text-sm font-semibold" : "text-base font-semibold"
              )}>
                {workflow.display_name || "Unnamed Workflow"}
              </h1>
              {!isMobile && (
                <span className="text-xs text-muted-foreground truncate">
                  {workflow.total_executions} steps · depth {workflow.max_depth} · {formatDuration(displayDuration)}
                  {isRunning && liveElapsed != null && (
                    <span className="text-emerald-500 ml-1">{"\u25B2"}</span>
                  )}
                </span>
              )}
            </div>
          </div>

          {/* Workflow ID - Hidden on mobile */}
          {!isMobile && (
            <HoverCard>
              <HoverCardTrigger asChild>
                <div className="flex items-center gap-2 cursor-pointer flex-shrink-0">
                  <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                    {workflow.workflow_id.slice(0, 8)}...
                  </code>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCopyId}
                    className="h-6 w-6 p-0"
                    title="Copy workflow ID"
                  >
                    {copied ? (
                      <Check className="w-3 h-3 text-green-500" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
              </HoverCardTrigger>
              <HoverCardContent className="w-auto">
                <div className="space-y-2">
                  <p className="text-sm font-medium">Workflow ID</p>
                  <code className="text-xs font-mono">{workflow.workflow_id}</code>
                </div>
              </HoverCardContent>
            </HoverCard>
          )}

          {/* Selection Info */}
          {selectedNodeCount > 0 && !isMobile && (
            <Badge variant="secondary" className="text-xs flex-shrink-0">
              {selectedNodeCount} selected
            </Badge>
          )}
        </div>

        {/* Right: Controls — compact, icon-only */}
        <div className={cn(
          "flex items-center flex-shrink-0",
          isMobile ? "gap-1" : "gap-1"
        )}>
          {/* Execution Controls */}
          {(isRunning || isPaused) && executionId && (
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

              <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
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
                      This will stop the active workflow execution immediately. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel disabled={isCancelling}>Keep running</AlertDialogCancel>
                    <AlertDialogAction disabled={isCancelling} onClick={handleCancel}>
                      {isCancelling ? "Cancelling…" : "Cancel execution"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>

              <div className="w-px h-4 bg-border mx-0.5" />
            </>
          )}

          {onRefresh && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefresh}
              disabled={isRefreshing}
              className="h-8 w-8 p-0 relative"
              title={isRunning ? "Live · Refresh workflow" : "Refresh workflow"}
            >
              <RotateCcw className={cn("w-4 h-4", isRefreshing && "animate-spin")} />
              {isRunning && (
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </Button>
          )}

          {/* Fullscreen */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFullscreenChange(!isFullscreen)}
            className="h-8 w-8 p-0"
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? (
              <Minimize className="w-4 h-4" />
            ) : (
              <Maximize className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Mobile: Second Row - Steps info and badges */}
      {isMobile && (
        <div className="flex items-center gap-2 w-full text-body-small text-muted-foreground flex-wrap">
          <span>{workflow.total_executions} steps</span>
          <span>•</span>
          <span>depth {workflow.max_depth}</span>
          <span>•</span>
          <span>{formatDuration(displayDuration)}</span>
          {(activeExecutions > 0 || failedExecutions > 0) && (
            <>
              <span>•</span>
              {activeExecutions > 0 && (
                <Badge variant="secondary" className="h-5 px-2 text-body-small">
                  {activeExecutions} active
                </Badge>
              )}
              {failedExecutions > 0 && (
                <Badge variant="destructive" className="h-5 px-2 text-body-small">
                  {failedExecutions} issues
                </Badge>
              )}
            </>
          )}
          {selectedNodeCount > 0 && (
            <>
              <span>•</span>
              <Badge variant="secondary" className="text-xs">
                {selectedNodeCount} selected
              </Badge>
            </>
          )}
        </div>
      )}
    </div>
  );
}

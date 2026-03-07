import { useState, useEffect, useMemo, type ComponentType } from "react";
import {
  ArrowLeft,
  RotateCcw,
  PauseCircle,
  Activity,
  XCircle,
  Play,
  MoreHorizontal,
  Clock,
  Copy,
  GitBranch,
  Maximize,
  Minimize,
  RadioTower,
} from "@/components/ui/icon-bridge";
import { formatDurationHumanReadable } from "@/components/ui/data-formatters";
import { useNavigate } from "react-router-dom";
import type { WorkflowSummary } from "../../types/workflows";
import { Button } from "../ui/button";
import { CopyButton } from "../ui/copy-button";
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
} from "../ui/alert-dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";
import {
  AnimatedTabs,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "../ui/animated-tabs";
import { cn } from "../../lib/utils";
import {
  normalizeExecutionStatus,
  getStatusLabel,
  getStatusTheme,
  isPausedStatus,
} from "../../utils/status";
import {
  summarizeWorkflowWebhook,
  formatWebhookStatusLabel,
} from "../../utils/webhook";
import {
  cancelExecution,
  pauseExecution,
  resumeExecution,
} from "../../services/executionsApi";
import {
  useErrorNotification,
  useSuccessNotification,
} from "../ui/notification";

/* ═══════════════════════════════════════════════════════════════
   Types
   ═══════════════════════════════════════════════════════════════ */

export interface WorkflowNavigationTab {
  id: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  description: string;
  shortcut: string;
  count?: number;
}

interface EnhancedWorkflowHeaderProps {
  workflow: WorkflowSummary;
  dagData?: any;
  isRefreshing?: boolean;
  onRefresh?: () => void;
  onClose?: () => void;
  isFullscreen: boolean;
  onFullscreenChange: (enabled: boolean) => void;
  selectedNodeCount: number;
  activeTab: string;
  onTabChange: (tab: string) => void;
  navigationTabs: WorkflowNavigationTab[];
}

/* ═══════════════════════════════════════════════════════════════
   Hooks & Helpers
   ═══════════════════════════════════════════════════════════════ */

const formatDuration = formatDurationHumanReadable;

/** Live elapsed-time counter for active workflows. */
function useLiveElapsed(startedAt?: string, status?: string): number | null {
  const normalized = normalizeExecutionStatus(status);
  const isActive = normalized === "running";
  const isPaused_ = isPausedStatus(normalized);

  const [elapsed, setElapsed] = useState<number | null>(() => {
    if (!startedAt) return null;
    return Math.max(0, Date.now() - new Date(startedAt).getTime());
  });

  useEffect(() => {
    if (!startedAt) {
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

    if (isPaused_) {
      setElapsed(compute());
      return;
    }

    setElapsed(null);
  }, [startedAt, isActive, isPaused_]);

  return elapsed;
}

function truncateId(id: string, maxLen = 20): string {
  if (id.length <= maxLen) return id;
  const keep = Math.floor((maxLen - 1) / 2);
  return `${id.slice(0, keep + 2)}\u2026${id.slice(-keep)}`;
}

/* ═══════════════════════════════════════════════════════════════
   EnhancedWorkflowHeader
   ═══════════════════════════════════════════════════════════════ */

export function EnhancedWorkflowHeader({
  workflow,
  dagData,
  isRefreshing,
  onRefresh,
  onClose,
  isFullscreen,
  onFullscreenChange,
  selectedNodeCount,
  activeTab,
  onTabChange,
  navigationTabs,
}: EnhancedWorkflowHeaderProps) {
  const navigate = useNavigate();

  /* ── Status ── */
  const normalizedStatus = normalizeExecutionStatus(workflow.status);
  const statusTheme = getStatusTheme(normalizedStatus);
  const isRunning = normalizedStatus === "running";
  const isPaused = isPausedStatus(normalizedStatus);
  const showLifecycleControls = isRunning || isPaused;

  /* ── Mutation state ── */
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isPausing, setIsPausing] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const isMutating = isCancelling || isPausing || isResuming;

  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  /* ── Duration ── */
  const liveElapsed = useLiveElapsed(workflow.started_at, workflow.status);
  const displayDuration = liveElapsed ?? workflow.duration_ms;

  /* ── Identity ── */
  const executionId =
    workflow.root_execution_id ??
    (workflow as WorkflowSummary & { execution_id?: string }).execution_id ??
    dagData?.timeline?.[0]?.execution_id;
  const rootAgentNodeId = dagData?.timeline?.[0]?.agent_node_id as string | undefined;

  /* ── Workflow-level metrics ── */
  const statusCounts = workflow.status_counts ?? {};
  const activeExecutions = workflow.active_executions ?? 0;
  const failedExecutions = (statusCounts.failed ?? 0) + (statusCounts.timeout ?? 0);

  /* ── Webhook summary ── */
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

  /* ── Handlers ── */
  const handleClose = () => {
    if (onClose) onClose();
    else navigate("/workflows");
  };

  const handlePause = async () => {
    if (!executionId || isMutating) return;
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
    if (!executionId || isMutating) return;
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
    if (!executionId || isMutating) return;
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

  const handleCopyWorkflowId = async () => {
    try {
      await navigator.clipboard.writeText(workflow.workflow_id);
    } catch {
      /* non-critical */
    }
  };

  return (
    <>
      <div className="flex flex-col">
        {/* ═══════════════════════════════════════════════════════
            ROW 1 — PRIMARY WORKFLOW BAR
            ═══════════════════════════════════════════════════════ */}
        <div className="bg-background border-b border-border h-12">
          {/* ── Desktop / Tablet (md+) ───────────────────────── */}
          <div className="hidden md:flex items-center w-full h-full px-4">
            {/* Back button */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={handleClose}
                  className="flex-shrink-0 mr-3"
                  aria-label="Back to workflows"
                >
                  <ArrowLeft className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Back to workflows</TooltipContent>
            </Tooltip>

            {/* ── Status cluster ── */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div
                className={cn(
                  "w-2 h-2 rounded-full flex-shrink-0",
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

              {/* Active execution count */}
              {activeExecutions > 0 && (
                <Badge variant="secondary" className="h-5 px-2 text-body-small" showIcon={false}>
                  {activeExecutions} active
                </Badge>
              )}

              {/* Failed execution count */}
              {failedExecutions > 0 && (
                <Badge variant="destructive" className="h-5 px-2 text-body-small" showIcon={false}>
                  {failedExecutions} {failedExecutions === 1 ? "issue" : "issues"}
                </Badge>
              )}

              {/* Webhook insights badge */}
              {hasWebhookInsights && (
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Badge variant="outline" className={webhookBadgeClasses} showIcon={false}>
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
                            ? `${webhookSummary.totalDeliveries} deliveries \u00B7 ${webhookSummary.successDeliveries} succeeded`
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
                          <span className="ml-1">&bull; HTTP {webhookSummary.lastHttpStatus}</span>
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

            {/* Cluster divider */}
            <div className="w-px h-5 bg-border flex-shrink-0 mx-3" />

            {/* ── Identity cluster ── */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              {/* Workflow name — highest emphasis */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-sm font-semibold text-foreground truncate flex-shrink-0 max-w-[200px] cursor-default">
                    {workflow.display_name || "Unnamed Workflow"}
                  </span>
                </TooltipTrigger>
                {(workflow.display_name || "").length > 20 && (
                  <TooltipContent>{workflow.display_name}</TooltipContent>
                )}
              </Tooltip>

              {/* Root agent node ID — secondary mono chip */}
              {rootAgentNodeId && rootAgentNodeId !== workflow.display_name && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <code className="text-xs font-mono bg-muted text-muted-foreground px-1.5 py-0.5 rounded flex-shrink-0 max-w-[140px] truncate cursor-default">
                      {rootAgentNodeId}
                    </code>
                  </TooltipTrigger>
                  {rootAgentNodeId.length > 16 && (
                    <TooltipContent>{rootAgentNodeId}</TooltipContent>
                  )}
                </Tooltip>
              )}

              {/* Duration with clock icon */}
              {displayDuration != null && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0 cursor-default">
                      <Clock className="w-3 h-3 flex-shrink-0" />
                      <span className="font-medium tabular-nums">
                        {formatDuration(displayDuration)}
                      </span>
                      {isRunning && liveElapsed != null && (
                        <span className="text-emerald-500">{"\u25B2"}</span>
                      )}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Started{" "}
                    {new Date(workflow.started_at).toLocaleString()}
                  </TooltipContent>
                </Tooltip>
              )}

              {/* Workflow ID + Copy (lg+ only) */}
              <div className="hidden lg:flex items-center gap-1 flex-shrink-0 ml-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <code className="text-[11px] font-mono text-muted-foreground/70 px-1.5 py-0.5 rounded bg-muted/40 cursor-default">
                      {truncateId(workflow.workflow_id)}
                    </code>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <code className="text-xs font-mono">
                      {workflow.workflow_id}
                    </code>
                  </TooltipContent>
                </Tooltip>
                <CopyButton
                  value={workflow.workflow_id}
                  tooltip="Copy workflow ID"
                  copiedTooltip="Workflow ID copied"
                  className="h-6 w-6 [&_svg]:!h-3 [&_svg]:!w-3"
                />
              </div>
            </div>

            {/* Selected node count badge */}
            {selectedNodeCount > 0 && (
              <Badge variant="secondary" className="text-xs flex-shrink-0 ml-2" showIcon={false}>
                {selectedNodeCount} selected
              </Badge>
            )}

            {/* ── Controls (far right) ── */}
            <div className="flex items-center gap-1 flex-shrink-0 ml-2">
              {showLifecycleControls && executionId && (
                <>
                  {/* Pause (when running) */}
                  {isRunning && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={isMutating}
                          onClick={handlePause}
                          className="hover:bg-amber-500/10 hover:text-amber-600"
                          aria-label="Pause execution"
                        >
                          {isPausing ? (
                            <Activity className="w-4 h-4 animate-spin" />
                          ) : (
                            <PauseCircle className="w-4 h-4" />
                          )}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Pause execution</TooltipContent>
                    </Tooltip>
                  )}

                  {/* Resume (when paused) */}
                  {isPaused && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={isMutating}
                          onClick={handleResume}
                          className="hover:bg-emerald-500/10 hover:text-emerald-600"
                          aria-label="Resume execution"
                        >
                          {isResuming ? (
                            <Activity className="w-4 h-4 animate-spin" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Resume execution</TooltipContent>
                    </Tooltip>
                  )}

                  {/* Stop (destructive) */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        disabled={isMutating}
                        onClick={() => setCancelDialogOpen(true)}
                        className="hover:bg-destructive/10 hover:text-destructive"
                        aria-label="Stop execution"
                      >
                        {isCancelling ? (
                          <Activity className="w-4 h-4 animate-spin" />
                        ) : (
                          <XCircle className="w-4 h-4" />
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Stop execution</TooltipContent>
                  </Tooltip>

                  <div className="w-px h-4 bg-border mx-0.5" />
                </>
              )}

              {/* Refresh with live indicator */}
              {onRefresh && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={onRefresh}
                      disabled={isRefreshing}
                      className="relative"
                      aria-label={isRunning ? "Live \u00B7 Refresh" : "Refresh"}
                    >
                      <RotateCcw
                        className={cn(
                          "w-4 h-4",
                          isRefreshing && "animate-spin",
                        )}
                      />
                      {isRunning && (
                        <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isRunning ? "Live \u00B7 Refresh" : "Refresh"}
                  </TooltipContent>
                </Tooltip>
              )}

              <div className="w-px h-4 bg-border mx-0.5" />

              {/* Fullscreen toggle */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => onFullscreenChange(!isFullscreen)}
                    aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                  >
                    {isFullscreen ? (
                      <Minimize className="w-4 h-4" />
                    ) : (
                      <Maximize className="w-4 h-4" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>

          {/* ── Mobile (<md) ─────────────────────────────────── */}
          <div className="flex md:hidden items-center w-full h-full px-3">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={handleClose}
              aria-label="Back to workflows"
              className="flex-shrink-0"
            >
              <ArrowLeft className="w-4 h-4" />
            </Button>

            <div className="flex items-center gap-2 ml-2 flex-1 min-w-0">
              <div
                className={cn(
                  "w-2 h-2 rounded-full flex-shrink-0",
                  statusTheme.indicatorClass,
                  isRunning && "animate-pulse",
                )}
              />
              <span
                className={cn(
                  "text-sm font-medium truncate",
                  statusTheme.textClass,
                )}
              >
                {getStatusLabel(normalizedStatus)}
              </span>
            </div>

            {/* Overflow menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="relative flex-shrink-0"
                  aria-label="Workflow actions"
                >
                  <MoreHorizontal className="w-4 h-4" />
                  {isRunning && (
                    <span className="absolute top-0 right-0 w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {isRunning && executionId && (
                  <DropdownMenuItem
                    onClick={handlePause}
                    disabled={isMutating}
                  >
                    <PauseCircle className="w-4 h-4" />
                    Pause execution
                  </DropdownMenuItem>
                )}
                {isPaused && executionId && (
                  <DropdownMenuItem
                    onClick={handleResume}
                    disabled={isMutating}
                  >
                    <Play className="w-4 h-4" />
                    Resume execution
                  </DropdownMenuItem>
                )}
                {showLifecycleControls && executionId && (
                  <DropdownMenuItem
                    onClick={() => setCancelDialogOpen(true)}
                    disabled={isMutating}
                    className="text-destructive focus:text-destructive"
                  >
                    <XCircle className="w-4 h-4" />
                    Stop execution
                  </DropdownMenuItem>
                )}
                {showLifecycleControls && executionId && <DropdownMenuSeparator />}
                {onRefresh && (
                  <DropdownMenuItem
                    onClick={onRefresh}
                    disabled={isRefreshing}
                  >
                    <RotateCcw className="w-4 h-4" />
                    Refresh
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={() => onFullscreenChange(!isFullscreen)}>
                  {isFullscreen ? (
                    <Minimize className="w-4 h-4" />
                  ) : (
                    <Maximize className="w-4 h-4" />
                  )}
                  {isFullscreen ? "Exit fullscreen" : "Fullscreen"}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleCopyWorkflowId}>
                  <Copy className="w-4 h-4" />
                  Copy workflow ID
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════
            MOBILE ROW 2 — IDENTITY (<md only)
            ═══════════════════════════════════════════════════════ */}
        <div className="flex md:hidden items-center gap-2 border-b border-border px-3 py-1.5 overflow-x-auto scrollbar-none">
          <span className="text-sm font-semibold text-foreground truncate flex-shrink-0">
            {workflow.display_name || "Unnamed Workflow"}
          </span>
          {rootAgentNodeId && rootAgentNodeId !== workflow.display_name && (
            <code className="text-[11px] font-mono bg-muted text-muted-foreground px-1 py-0.5 rounded flex-shrink-0">
              {rootAgentNodeId}
            </code>
          )}
          {displayDuration != null && (
            <span className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0">
              <Clock className="w-3 h-3" />
              <span className="tabular-nums">
                {formatDuration(displayDuration)}
              </span>
              {isRunning && liveElapsed != null && (
                <span className="text-emerald-500">{"\u25B2"}</span>
              )}
            </span>
          )}
          {activeExecutions > 0 && (
            <Badge variant="secondary" className="h-5 px-2 text-body-small flex-shrink-0" showIcon={false}>
              {activeExecutions} active
            </Badge>
          )}
          {failedExecutions > 0 && (
            <Badge variant="destructive" className="h-5 px-2 text-body-small flex-shrink-0" showIcon={false}>
              {failedExecutions} {failedExecutions === 1 ? "issue" : "issues"}
            </Badge>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════
            ROW 2 — SECTION NAVIGATION + SUMMARY METRICS
            ═══════════════════════════════════════════════════════ */}
        <div className={cn(
          "h-12 border-b border-border bg-background flex items-center px-4 md:px-6 overflow-x-auto scrollbar-none",
          isFullscreen ? "" : "",
        )}>
          <div className="flex flex-1 items-center gap-4 min-w-0">
            <AnimatedTabs
              value={activeTab}
              onValueChange={onTabChange}
              className="flex h-full min-w-0 flex-1 flex-col justify-center"
            >
              <AnimatedTabsList className="h-full gap-1 flex-nowrap">
                {navigationTabs.map((tab) => {
                  const Icon = tab.icon;

                  return (
                    <AnimatedTabsTrigger
                      key={tab.id}
                      value={tab.id}
                      className="gap-2 px-3 py-2 flex-shrink-0"
                      title={`${tab.description} (Cmd/Ctrl + ${tab.shortcut})`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="whitespace-nowrap hidden sm:inline">
                        {tab.label}
                      </span>

                      {tab.count !== undefined &&
                        tab.count > 0 && (
                          <Badge
                            variant="count"
                            size="sm"
                            className="min-w-[20px]"
                            showIcon={false}
                          >
                            {tab.count > 999 ? "999+" : tab.count}
                          </Badge>
                        )}
                    </AnimatedTabsTrigger>
                  );
                })}
              </AnimatedTabsList>
            </AnimatedTabs>
          </div>

          {/* Summary metrics (far right, lg+ only) */}
          {(workflow.max_depth > 0 || workflow.total_executions > 0) && (
            <div className="hidden lg:flex items-center gap-3 flex-shrink-0 text-xs text-muted-foreground ml-4 pl-4 border-l border-border">
              {workflow.max_depth > 0 && (
                <span className="flex items-center gap-1">
                  <GitBranch className="w-3.5 h-3.5" />
                  <span>depth {workflow.max_depth}</span>
                </span>
              )}
              {workflow.total_executions > 0 && (
                <span className="flex items-center gap-1">
                  <span>{workflow.total_executions} nodes</span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════
          CANCEL CONFIRMATION DIALOG (shared by desktop + mobile)
          ═══════════════════════════════════════════════════════ */}
      <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop execution?</AlertDialogTitle>
            <AlertDialogDescription>
              This will stop the active workflow execution immediately. This action cannot be
              undone.
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
              {isCancelling ? "Stopping\u2026" : "Stop execution"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

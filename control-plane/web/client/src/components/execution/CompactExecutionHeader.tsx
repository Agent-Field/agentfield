import { useState, useEffect, type ComponentType } from "react";
import {
  ArrowLeft,
  ExternalLink,
  RotateCcw,
  PauseCircle,
  Activity,
  XCircle,
  Play,
  MoreHorizontal,
  Clock,
  Copy,
  GitBranch,
} from "@/components/ui/icon-bridge";
import { formatDurationHumanReadable } from "@/components/ui/data-formatters";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { Button } from "../ui/button";
import {
  CopyIdentifierChip,
  truncateIdMiddle,
} from "../ui/copy-identifier-chip";
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
  AnimatedTabs,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "../ui/tabs";
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

/* ═══════════════════════════════════════════════════════════════
   Types
   ═══════════════════════════════════════════════════════════════ */

export interface NavigationTab {
  id: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  description: string;
  shortcut: string;
  count?: number;
}

interface CompactExecutionHeaderProps {
  execution: WorkflowExecution;
  onClose?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  /** Currently active section tab id */
  activeTab: string;
  /** Callback when user switches section tab */
  onTabChange: (tab: string) => void;
  /** Section navigation tabs to render in Row 2 */
  navigationTabs: NavigationTab[];
}

/* ═══════════════════════════════════════════════════════════════
   Hooks & Helpers
   ═══════════════════════════════════════════════════════════════ */

const formatDuration = formatDurationHumanReadable;

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

/* ═══════════════════════════════════════════════════════════════
   CompactExecutionHeader
   ═══════════════════════════════════════════════════════════════ */

export function CompactExecutionHeader({
  execution,
  onClose,
  onRefresh,
  isRefreshing,
  activeTab,
  onTabChange,
  navigationTabs,
}: CompactExecutionHeaderProps) {
  const navigate = useNavigate();

  /* ── Status ── */
  const normalizedStatus = normalizeExecutionStatus(execution.status);
  const statusTheme = getStatusTheme(normalizedStatus);
  const isRunning = normalizedStatus === "running";
  const isPaused = isPausedStatus(normalizedStatus);
  const isTerminal = isTerminalStatus(execution.status);
  const showLifecycleControls = isRunning || isPaused;
  const hasError = !!execution.error_message;

  /* ── Mutation state ── */
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isPausing, setIsPausing] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const isMutating = isCancelling || isPausing || isResuming;

  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  /* ── Duration ── */
  const liveElapsed = useLiveElapsed(execution.started_at, execution.status);
  const displayDuration = isTerminal ? execution.duration_ms : liveElapsed;

  /* ── Identity ── */
  const showAgentNodeId =
    execution.agent_node_id &&
    execution.agent_node_id !== execution.reasoner_id;

  /* ── Handlers ── */
  const handleClose = () => {
    if (onClose) onClose();
    else navigate("/executions");
  };

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
        "Execution stopped",
        `Execution ${execution.execution_id.slice(0, 8)} has been stopped.`,
      );
      setCancelDialogOpen(false);
      onRefresh?.();
    } catch (error) {
      showError(
        "Stop failed",
        error instanceof Error
          ? error.message
          : "Unable to stop execution.",
      );
    } finally {
      setIsCancelling(false);
    }
  };

  const handleCopyRunId = async () => {
    try {
      await navigator.clipboard.writeText(execution.execution_id);
    } catch {
      /* non-critical */
    }
  };

  return (
    <>
      <div className="flex flex-col">
        {/* ═══════════════════════════════════════════════════════
            ROW 1 — PRIMARY EXECUTION BAR
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
                  aria-label="Back to executions"
                >
                  <ArrowLeft className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Back to executions</TooltipContent>
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

              {/* Approval required badge */}
              {normalizedStatus === "waiting" &&
                execution.approval_request_url && (
                  <a
                    href={execution.approval_request_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex"
                  >
                    <Badge
                      variant="outline"
                      size="sm"
                      className="border-amber-500/40 text-amber-500 hover:bg-amber-500/10 cursor-pointer"
                      showIcon={false}
                    >
                      Approval Required
                      <ExternalLink className="w-3 h-3 ml-0.5" />
                    </Badge>
                  </a>
                )}

              {/* Issue pill */}
              {hasError && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-micro-plus font-medium bg-destructive/10 text-destructive border border-destructive/20">
                  1 issue
                </span>
              )}

              {/* Retry indicator */}
              {(execution.retry_count ?? 0) > 0 && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-micro-plus font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                  {execution.retry_count}{" "}
                  {execution.retry_count === 1 ? "retry" : "retries"}
                </span>
              )}
            </div>

            {/* Cluster divider */}
            <div className="w-px h-5 bg-border flex-shrink-0 mx-3" />

            {/* ── Identity cluster ── */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              {/* Reasoner name — highest emphasis */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-sm font-semibold text-foreground truncate flex-shrink-0 max-w-[200px] cursor-default">
                    {execution.reasoner_id}
                  </span>
                </TooltipTrigger>
                {execution.reasoner_id.length > 20 && (
                  <TooltipContent>{execution.reasoner_id}</TooltipContent>
                )}
              </Tooltip>

              {/* Agent node ID — secondary mono chip */}
              {showAgentNodeId && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <code className="text-xs font-mono bg-muted text-muted-foreground px-1.5 py-0.5 rounded flex-shrink-0 max-w-[140px] truncate cursor-default">
                      {execution.agent_node_id}
                    </code>
                  </TooltipTrigger>
                  {execution.agent_node_id.length > 16 && (
                    <TooltipContent>{execution.agent_node_id}</TooltipContent>
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
                      {isRunning && (
                        <span className="text-emerald-500">{"\u25B2"}</span>
                      )}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Started{" "}
                    {new Date(execution.started_at).toLocaleString()}
                  </TooltipContent>
                </Tooltip>
              )}

              {/* Execution ID + Copy (lg+ only) */}
              <div className="hidden lg:flex flex-shrink-0 ml-1">
                <CopyIdentifierChip
                  value={execution.execution_id}
                  tooltip="Copy run ID"
                  copiedTooltip="Run ID copied"
                  formatDisplay={(v) => truncateIdMiddle(v, 20)}
                />
              </div>
            </div>

            {/* ── Execution controls (far right) ── */}
            <div className="flex items-center gap-1 flex-shrink-0 ml-2">
              {showLifecycleControls && (
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
                      aria-label={
                        isRunning ? "Live \u00B7 Refresh" : "Refresh"
                      }
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
            </div>
          </div>

          {/* ── Mobile (<md) ─────────────────────────────────── */}
          <div className="flex md:hidden items-center w-full h-full px-3">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={handleClose}
              aria-label="Back to executions"
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
                  aria-label="Execution actions"
                >
                  <MoreHorizontal className="w-4 h-4" />
                  {isRunning && (
                    <span className="absolute top-0 right-0 w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {isRunning && (
                  <DropdownMenuItem
                    onClick={handlePause}
                    disabled={isMutating}
                  >
                    <PauseCircle className="w-4 h-4" />
                    Pause execution
                  </DropdownMenuItem>
                )}
                {isPaused && (
                  <DropdownMenuItem
                    onClick={handleResume}
                    disabled={isMutating}
                  >
                    <Play className="w-4 h-4" />
                    Resume execution
                  </DropdownMenuItem>
                )}
                {showLifecycleControls && (
                  <DropdownMenuItem
                    onClick={() => setCancelDialogOpen(true)}
                    disabled={isMutating}
                    className="text-destructive focus:text-destructive"
                  >
                    <XCircle className="w-4 h-4" />
                    Stop execution
                  </DropdownMenuItem>
                )}
                {showLifecycleControls && <DropdownMenuSeparator />}
                {onRefresh && (
                  <DropdownMenuItem
                    onClick={onRefresh}
                    disabled={isRefreshing}
                  >
                    <RotateCcw className="w-4 h-4" />
                    Refresh
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={handleCopyRunId}>
                  <Copy className="w-4 h-4" />
                  Copy run ID
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
            {execution.reasoner_id}
          </span>
          {showAgentNodeId && (
            <code className="text-micro-plus font-mono bg-muted text-muted-foreground px-1 py-0.5 rounded flex-shrink-0">
              {execution.agent_node_id}
            </code>
          )}
          {displayDuration != null && (
            <span className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0">
              <Clock className="w-3 h-3" />
              <span className="tabular-nums">
                {formatDuration(displayDuration)}
              </span>
              {isRunning && (
                <span className="text-emerald-500">{"\u25B2"}</span>
              )}
            </span>
          )}
          {hasError && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-micro-plus font-medium bg-destructive/10 text-destructive flex-shrink-0">
              1 issue
            </span>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════
            ROW 2 — SECTION NAVIGATION + SUMMARY METRICS
            ═══════════════════════════════════════════════════════ */}
        <div className="h-12 border-b border-border bg-background flex items-center px-4 md:px-6 overflow-x-auto scrollbar-none">
          <div className="flex flex-1 items-center gap-4 min-w-0">
            <AnimatedTabs
              value={activeTab}
              onValueChange={onTabChange}
              className="flex h-full min-w-0 flex-1 flex-col justify-center"
            >
              <AnimatedTabsList className="h-full gap-1 flex-nowrap">
                {navigationTabs.map((tab) => {
                  const Icon = tab.icon;
                  const hasTabError =
                    tab.id === "debug" && !!execution.error_message;

                  return (
                    <AnimatedTabsTrigger
                      key={tab.id}
                      value={tab.id}
                      className="gap-2 px-3 py-2 flex-shrink-0 relative"
                      title={`${tab.description} (Cmd/Ctrl + ${tab.shortcut})`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="whitespace-nowrap hidden sm:inline">
                        {tab.label}
                      </span>

                      {hasTabError && (
                        <div className="w-2 h-2 bg-destructive rounded-full flex-shrink-0" />
                      )}

                      {tab.count !== undefined &&
                        tab.count > 0 &&
                        !hasTabError && (
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
          {execution.workflow_depth > 0 && (
            <div className="hidden lg:flex items-center gap-3 flex-shrink-0 text-xs text-muted-foreground ml-4 pl-4 border-l border-border">
              <span className="flex items-center gap-1">
                <GitBranch className="w-3.5 h-3.5" />
                <span>depth {execution.workflow_depth}</span>
              </span>
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
              This will cancel the execution immediately. This action cannot be
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

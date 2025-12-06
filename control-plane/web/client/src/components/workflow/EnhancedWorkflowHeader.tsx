import { useMemo, useState } from "react";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  ArrowLeft,
  RotateCcw,
  Maximize,
  Minimize,
  Focus,
  Eye,
  EyeOff,
  Activity,
  Zap,
  Bug,
  Copy,
  Check,
  RadioTower
} from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";
import { SegmentedControl } from "../ui/segmented-control";
import type { SegmentedControlOption } from "../ui/segmented-control";
import { cn } from "../../lib/utils";
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from "../../utils/status";
import { summarizeWorkflowWebhook, formatWebhookStatusLabel } from "../../utils/webhook";
import type { WorkflowSummary } from "../../types/workflows";

const VIEW_MODE_OPTIONS: ReadonlyArray<SegmentedControlOption> = [
  { value: "standard", label: "Standard", icon: Eye },
  { value: "performance", label: "Performance", icon: Zap },
  { value: "debug", label: "Debug", icon: Bug },
] as const;

interface EnhancedWorkflowHeaderProps {
  workflow: WorkflowSummary;
  dagData?: any;
  isLiveUpdating?: boolean;
  hasRunningWorkflows?: boolean;
  pollingInterval?: number;
  isRefreshing?: boolean;
  onRefresh?: () => void;
  onClose?: () => void;
  viewMode: 'standard' | 'performance' | 'debug';
  onViewModeChange: (mode: 'standard' | 'performance' | 'debug') => void;
  focusMode: boolean;
  onFocusModeChange: (enabled: boolean) => void;
  isFullscreen: boolean;
  onFullscreenChange: (enabled: boolean) => void;
  selectedNodeCount: number;
}

export function EnhancedWorkflowHeader({
  workflow,
  dagData,
  isLiveUpdating,
  hasRunningWorkflows,
  pollingInterval,
  isRefreshing,
  onRefresh,
  onClose,
  viewMode,
  onViewModeChange,
  focusMode,
  onFocusModeChange,
  isFullscreen,
  onFullscreenChange,
  selectedNodeCount
}: EnhancedWorkflowHeaderProps) {
  const [copied, setCopied] = useState(false);
  const isMobile = useIsMobile();

  const normalizedStatus = normalizeExecutionStatus(workflow.status);
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

  const formatDuration = (durationMs?: number) => {
    if (!durationMs) return "N/A";
    if (durationMs < 1000) return `${durationMs}ms`;
    if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
    return `${(durationMs / 60000).toFixed(1)}m`;
  };

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(workflow.workflow_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy workflow ID:', err);
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

            <div className="min-w-0 flex-1">
              <h1 className={cn(
                "text-foreground truncate",
                isMobile ? "text-sm font-semibold" : "text-heading-3"
              )}>
                {workflow.display_name || "Unnamed Workflow"}
              </h1>
              {!isMobile && (
                <div className="flex items-center gap-2 text-body-small">
                  <span>{workflow.total_executions} steps</span>
                  <span>•</span>
                  <span>depth {workflow.max_depth}</span>
                  <span>•</span>
                  <span>{formatDuration(workflow.duration_ms)}</span>
                </div>
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

        {/* Center: Live Status - Simplified on mobile */}
        {isLiveUpdating && (
          <HoverCard>
            <HoverCardTrigger asChild>
              <div className={cn(
                "flex items-center text-sm cursor-pointer flex-shrink-0",
                isMobile ? "gap-1.5" : "gap-3"
              )}>
                <div className="flex items-center gap-1.5">
                  <div className={cn(
                    "w-1.5 h-1.5 rounded-full",
                    hasRunningWorkflows ? "bg-green-500 animate-pulse" : "bg-gray-400"
                  )} />
                  {!isMobile && (
                    <span className="text-muted-foreground">
                      {hasRunningWorkflows ? "Live" : "Monitoring"}
                    </span>
                  )}
                  {isRefreshing && (
                    <Activity className="w-3 h-3 animate-spin text-muted-foreground" />
                  )}
                </div>

                {pollingInterval && !isMobile && (
                  <span className="text-body-small">
                    {Math.round(pollingInterval / 1000)}s
                  </span>
                )}
              </div>
            </HoverCardTrigger>
            <HoverCardContent className="w-auto">
              <div className="space-y-2">
                <p className="text-sm font-medium">Live Updates</p>
                <div className="text-body-small space-y-1">
                  <div>Status: {hasRunningWorkflows ? "Active" : "Monitoring"}</div>
                  <div>Interval: {pollingInterval ? Math.round(pollingInterval / 1000) : 3}s</div>
                </div>
              </div>
            </HoverCardContent>
          </HoverCard>
        )}

        {/* Right: Controls */}
        <div className={cn(
          "flex items-center flex-shrink-0",
          isMobile ? "gap-1" : "gap-2"
        )}>
          <SegmentedControl
            value={viewMode}
            onValueChange={(mode) => onViewModeChange(mode as typeof viewMode)}
            options={VIEW_MODE_OPTIONS}
            size="sm"
            hideLabel={isMobile}
            optionClassName={cn(isMobile ? "min-w-[36px] px-0" : "min-w-[104px]")}
          />

          {/* Focus Mode */}
          <Button
            variant={focusMode ? "secondary" : "ghost"}
            size="sm"
            onClick={() => onFocusModeChange(!focusMode)}
            className="h-8 w-8 p-0"
            title={focusMode ? "Exit focus mode (Cmd/Ctrl + F)" : "Enter focus mode (Cmd/Ctrl + F)"}
          >
            {focusMode ? <EyeOff className="w-4 h-4" /> : <Focus className="w-4 h-4" />}
          </Button>

          {/* Refresh */}
          {onRefresh && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefresh}
              disabled={isRefreshing}
              className="h-8 w-8 p-0"
              title="Refresh workflow (Cmd/Ctrl + R)"
            >
              <RotateCcw className={cn("w-4 h-4", isRefreshing && "animate-spin")} />
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
          <span>{formatDuration(workflow.duration_ms)}</span>
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

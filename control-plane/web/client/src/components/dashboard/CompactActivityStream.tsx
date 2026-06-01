import { useRecentActivitySimple } from "@/hooks/useRecentActivity";
import { cn } from "@/lib/utils";
import type { ActivityExecution } from "@/types/recentActivity";
import {
  CheckmarkFilled,
  ErrorFilled,
  InProgress,
  Renew,
  Time,
  WarningAlt,
} from "@/components/ui/icon-bridge";
import { Skeleton } from "@/components/ui/skeleton";
import { useNavigate } from "react-router-dom";
import {
  getStatusLabel,
  getStatusTheme,
  normalizeExecutionStatus,
} from "@/utils/status";

interface CompactActivityStreamProps {
  className?: string;
}

export function CompactActivityStream({ className }: CompactActivityStreamProps) {
  const navigate = useNavigate();
  const {
    executions,
    loading,
    error,
    hasError,
    refresh,
    clearError,
    isRefreshing,
    isEmpty,
  } = useRecentActivitySimple();

  /**
   * Get status icon based on execution status
   */
  const getStatusVisuals = (status: string) => {
    const normalized = normalizeExecutionStatus(status);
    const theme = getStatusTheme(normalized);
    const Icon =
      normalized === "succeeded"
        ? CheckmarkFilled
        : normalized === "failed"
          ? ErrorFilled
          : InProgress;

    return {
      theme,
      Icon,
      label: getStatusLabel(normalized),
    };
  };

  /**
   * Handle execution click - navigate to execution detail
   */
  const handleExecutionClick = (execution: ActivityExecution) => {
    navigate(`/executions/${execution.execution_id}`);
  };

  /**
   * Format duration for display
   */
  const formatDuration = (durationMs?: number) => {
    if (!durationMs) return null;

    if (durationMs < 1000) {
      return `${durationMs}ms`;
    }

    const seconds = Math.round(durationMs / 1000);
    if (seconds < 60) {
      return `${seconds}s`;
    }

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return remainingSeconds > 0
      ? `${minutes}m ${remainingSeconds}s`
      : `${minutes}m`;
  };


  // Get the first 18 executions to match ExecutionTimeline height
  const compactExecutions = executions.slice(0, 18);

  // Loading state
  if (loading && !executions.length) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-4 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-foreground">Recent Activity</h3>
          <div className="w-3 h-3 animate-spin rounded-full border border-current border-t-transparent text-muted-foreground" />
        </div>

        <div className="space-y-2">
          {Array.from({ length: 18 }).map((_, i) => (
            <div
              key={i}
              className="flex items-center space-x-2 py-1.5"
            >
              <Skeleton className="h-3 w-3 rounded-full flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <Skeleton className="h-3 w-3/4" />
              </div>
              <Skeleton className="h-2 w-8 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (hasError && isEmpty) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-foreground">Recent Activity</h3>
          <button
            onClick={refresh}
            disabled={isRefreshing}
            className={cn(
              "flex items-center space-x-1 px-2 py-1 rounded text-xs",
              "border border-border hover:bg-muted",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-colors duration-200"
            )}
          >
            <Renew className={cn("h-3 w-3", isRefreshing && "animate-spin")} />
            <span>{isRefreshing ? "Retry..." : "Retry"}</span>
          </button>
        </div>

        <div className="flex flex-col items-center justify-center py-6 space-y-2">
          <WarningAlt className="h-4 w-4" style={{ color: "var(--status-error)" }} />
          <p className="text-sm text-muted-foreground text-center">
            Failed to load activity
          </p>
          {error && (
            <button
              onClick={clearError}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    );
  }

  // Empty state
  if (isEmpty) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-foreground">Recent Activity</h3>
          <button
            onClick={refresh}
            disabled={isRefreshing}
            className={cn(
              "flex items-center space-x-1 px-2 py-1 rounded text-xs",
              "border border-border hover:bg-muted",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-colors duration-200"
            )}
          >
            <Renew className={cn("h-3 w-3", isRefreshing && "animate-spin")} />
            <span>{isRefreshing ? "Load..." : "Refresh"}</span>
          </button>
        </div>

        <div className="flex flex-col items-center justify-center py-6 space-y-2">
          <Time className="h-4 w-4 text-muted-foreground" />
          <p className="text-sm text-muted-foreground text-center">
            No recent activity
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-border bg-card p-4 shadow-sm", className)}>
      {/* Header with refresh button */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Recent Activity</h3>
        <button
          onClick={refresh}
          disabled={isRefreshing}
          className={cn(
            "flex items-center space-x-1 px-2 py-1 rounded text-xs",
            "border border-border hover:bg-muted",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            "transition-colors duration-200"
          )}
          title="Refresh recent activity"
        >
          <Renew className={cn("h-3 w-3", isRefreshing && "animate-spin")} />
          <span>{isRefreshing ? "Refresh..." : "Refresh"}</span>
        </button>
      </div>

      {/* Error banner (if error but we have cached data) */}
      {hasError && compactExecutions.length > 0 && (
        <div
          className="flex items-center justify-between p-2 mb-3 rounded border"
          style={{
            backgroundColor: "hsl(var(--status-warning) / 0.1)",
            borderColor: "hsl(var(--status-warning) / 0.3)"
          }}
        >
          <div className="flex items-center space-x-2">
            <WarningAlt className="h-3 w-3" style={{ color: "var(--status-warning)" }} />
            <p className="text-xs text-foreground">Unable to refresh</p>
          </div>
          <button
            onClick={clearError}
            className="text-xs transition-colors duration-200"
            style={{ color: "var(--status-warning)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = "0.8";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = "1";
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Compact activity list with responsive height to match ExecutionTimeline */}
      <div className="space-y-1 overflow-y-auto h-[280px] md:h-[280px] lg:h-[280px]">
        {compactExecutions.map((execution) => {
          const { theme, Icon: StatusIcon, label } = getStatusVisuals(execution.status);
          const duration = formatDuration(execution.duration_ms);

          return (
            <div
              key={execution.execution_id}
              onClick={() => handleExecutionClick(execution)}
              className={cn(
                "flex items-center space-x-2 py-1.5 px-2 rounded cursor-pointer",
                "hover:bg-muted/50 transition-colors duration-200",
                "group"
              )}
              title={`${execution.agent_name} • ${execution.reasoner_name} • ${execution.relative_time}${duration ? ` • ${duration}` : ''}`}
            >
              {/* Status icon */}
              <StatusIcon className={cn("h-3 w-3 flex-shrink-0", theme.iconClass)} />

              {/* Execution info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-1">
                  <span className="text-xs font-medium text-foreground truncate">
                    {execution.reasoner_name}
                  </span>
                  <span className="text-sm text-muted-foreground">·</span>
                  <span className="text-sm text-muted-foreground truncate">
                    {execution.agent_name}
                  </span>
                </div>
                <div className="text-micro text-muted-foreground">
                  {label}
                </div>
              </div>

              {/* Timestamp */}
              <div className="text-sm text-muted-foreground flex-shrink-0">
                {execution.relative_time}
              </div>
            </div>
          );
        })}
      </div>

      {/* View all link */}
      <div className="pt-2 mt-2 border-t border-border">
        <button
          onClick={() => navigate("/executions")}
          className="w-full text-center text-xs text-primary hover:text-primary/80 transition-colors duration-200"
        >
          View all executions →
        </button>
      </div>
    </div>
  );
}

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

interface RecentActivityStreamProps {
  className?: string;
}

export function RecentActivityStream({ className }: RecentActivityStreamProps) {
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
      normalized,
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

  // Loading state
  if (loading && !executions.length) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-foreground">
            Recent Activity
          </h2>
          <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent text-muted-foreground" />
        </div>

        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="flex items-center space-x-3 p-3 rounded-lg border border-border"
            >
              <Skeleton className="h-4 w-4 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-3/4 rounded" />
                <Skeleton className="h-3 w-1/2 rounded" />
              </div>
              <Skeleton className="h-3 w-16 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (hasError && isEmpty) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-foreground">
            Recent Activity
          </h2>
          <button
            onClick={refresh}
            disabled={isRefreshing}
            className={cn(
              "flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm",
              "border border-border hover:bg-muted",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-colors duration-200"
            )}
          >
            <Renew className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
            <span>{isRefreshing ? "Retrying..." : "Retry"}</span>
          </button>
        </div>

        <div className="flex flex-col items-center justify-center py-8 space-y-4">
          <div className="flex items-center space-x-3">
            <WarningAlt className="h-6 w-6" style={{ color: "var(--status-error)" }} />
            <h3 className="font-medium" style={{ color: "var(--status-error)" }}>Failed to load recent activity</h3>
          </div>
          <p className="text-muted-foreground text-center text-sm max-w-md">
            {error?.message ||
              "An unexpected error occurred while fetching recent activity."}
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
      <div className={cn("space-y-4", className)}>
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-foreground">
            Recent Activity
          </h2>
          <button
            onClick={refresh}
            disabled={isRefreshing}
            className={cn(
              "flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm",
              "border border-border hover:bg-muted",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-colors duration-200"
            )}
          >
            <Renew className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
            <span>{isRefreshing ? "Loading..." : "Refresh"}</span>
          </button>
        </div>

        <div className="flex flex-col items-center justify-center py-8 space-y-4">
          <div className="w-12 h-12 bg-muted rounded-lg flex items-center justify-center">
            <Time className="h-6 w-6 text-muted-foreground" />
          </div>
          <div className="text-center space-y-2">
            <h3 className="font-medium text-foreground">No recent activity</h3>
            <p className="text-muted-foreground text-sm">
              Recent executions will appear here once agents start running.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with refresh button */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-foreground">Recent Activity</h2>
        <button
          onClick={refresh}
          disabled={isRefreshing}
          className={cn(
            "flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm",
            "border border-border hover:bg-muted",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            "transition-colors duration-200"
          )}
          title="Refresh recent activity"
        >
          <Renew className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
          <span>{isRefreshing ? "Refreshing..." : "Refresh"}</span>
        </button>
      </div>

      {/* Error banner (if error but we have cached data) */}
      {hasError && executions.length > 0 && (
        <div
          className="flex items-center justify-between p-3 rounded-lg border"
          style={{
            backgroundColor: "hsl(var(--status-warning) / 0.1)",
            borderColor: "hsl(var(--status-warning) / 0.3)"
          }}
        >
          <div className="flex items-center space-x-3">
            <WarningAlt className="h-4 w-4" style={{ color: "var(--status-warning)" }} />
            <div>
              <p className="text-sm font-medium text-foreground">
                Unable to refresh data
              </p>
              <p className="text-sm text-muted-foreground">
                Showing cached data. {error?.message}
              </p>
            </div>
          </div>
          <button
            onClick={clearError}
            className="text-xs transition-colors duration-200"
            style={{
              color: "var(--status-warning)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = "0.8";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = "1";
            }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Activity list */}
      <div className="space-y-2">
        {executions.map((execution) => {
          const { theme, Icon: StatusIcon, label } = getStatusVisuals(execution.status);
          const duration = formatDuration(execution.duration_ms);

          return (
            <div
              key={execution.execution_id}
              onClick={() => handleExecutionClick(execution)}
              className={cn(
                "flex items-center space-x-3 p-3 rounded-lg border border-border",
                "hover:bg-muted/50 cursor-pointer transition-colors duration-200",
                "group"
              )}
            >
              {/* Status icon */}
              <StatusIcon className={cn("h-4 w-4 flex-shrink-0", theme.iconClass)} />

              {/* Execution info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-foreground truncate">
                    {execution.agent_name}
                  </span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground truncate">
                    {execution.reasoner_name}
                  </span>
                </div>
                <div className="flex items-center space-x-2 mt-1">
                  <span className="text-sm text-muted-foreground">
                    {execution.relative_time}
                  </span>
                  {duration && (
                    <>
                      <span className="text-sm text-muted-foreground">·</span>
                      <span className="text-sm text-muted-foreground">
                        {duration}
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Status badge */}
              <div className={cn("px-2 py-1 rounded-full text-xs font-medium", theme.pillClass)}>
                {label}
              </div>
            </div>
          );
        })}
      </div>

      {/* View all link */}
      <div className="pt-2 border-t border-border">
        <button
          onClick={() => navigate("/executions")}
          className="w-full text-center text-sm text-primary hover:text-primary/80 transition-colors duration-200"
        >
          View all executions →
        </button>
      </div>
    </div>
  );
}

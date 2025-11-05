import { Button } from "@/components/ui/button";
import { bulkNodeStatus, refreshNodeStatus } from "@/services/api";
import type { AgentStatus } from "@/types/haxen";
import { ArrowClockwise } from "@/components/ui/icon-bridge";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { statusTone } from "@/lib/theme";

interface StatusRefreshButtonProps {
  nodeId?: string;
  nodeIds?: string[];
  onRefresh?: (status: AgentStatus | Record<string, AgentStatus>) => void;
  onError?: (error: string) => void;
  disabled?: boolean;
  size?: "sm" | "default" | "lg";
  variant?: "default" | "outline" | "ghost";
  showLabel?: boolean;
  showLastVerified?: boolean;
  lastVerified?: string;
  className?: string;
}

export function StatusRefreshButton({
  nodeId,
  nodeIds,
  onRefresh,
  onError,
  disabled = false,
  size = "default",
  variant = "outline",
  showLabel = true,
  showLastVerified = false,
  lastVerified,
  className = "",
}: StatusRefreshButtonProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    if (isRefreshing || disabled) return;

    setIsRefreshing(true);

    try {
      let result: AgentStatus | Record<string, AgentStatus>;

      if (nodeId) {
        // Single node refresh
        result = await refreshNodeStatus(nodeId);
      } else if (nodeIds && nodeIds.length > 0) {
        // Bulk refresh
        result = await bulkNodeStatus(nodeIds);
      } else {
        throw new Error("Either nodeId or nodeIds must be provided");
      }

      onRefresh?.(result);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to refresh status";
      onError?.(errorMessage);
      console.error("Status refresh failed:", error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const getButtonSize = () => {
    switch (size) {
      case "sm":
        return "h-8 px-2";
      case "lg":
        return "h-12 px-6";
      default:
        return "h-10 px-4";
    }
  };

  const getIconSize = () => {
    switch (size) {
      case "sm":
        return 14;
      case "lg":
        return 20;
      default:
        return 16;
    }
  };

  // Helper function to format last verified timestamp
  const formatLastVerified = (timestamp?: string) => {
    if (!timestamp) return null;

    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffSeconds = Math.floor(diffMs / 1000);
      const diffMinutes = Math.floor(diffSeconds / 60);
      const diffHours = Math.floor(diffMinutes / 60);

      if (diffSeconds < 60) {
        return `${diffSeconds}s ago`;
      } else if (diffMinutes < 60) {
        return `${diffMinutes}m ago`;
      } else if (diffHours < 24) {
        return `${diffHours}h ago`;
      } else {
        return date.toLocaleDateString();
      }
    } catch {
      return null;
    }
  };

  const lastVerifiedText = formatLastVerified(lastVerified);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <Button
        variant={variant}
        size={size}
        disabled={disabled || isRefreshing}
        onClick={handleRefresh}
        className={cn(getButtonSize())}
        title={
          nodeId
            ? `Refresh status for node ${nodeId}`
            : `Refresh status for ${nodeIds?.length || 0} nodes`
        }
      >
        <ArrowClockwise
          size={getIconSize()}
          className={cn(
            isRefreshing && "animate-spin",
            showLabel && "mr-2",
            statusTone.info.accent
          )}
        />
        {showLabel && <span>{isRefreshing ? "Refreshing..." : "Refresh"}</span>}
      </Button>

      {showLastVerified && lastVerifiedText && (
        <span className="text-body-small whitespace-nowrap">
          Verified {lastVerifiedText}
        </span>
      )}
    </div>
  );
}

// Hook for optimistic UI updates
export function useOptimisticStatusRefresh() {
  const [optimisticUpdates, setOptimisticUpdates] = useState<
    Record<string, AgentStatus>
  >({});

  const setOptimisticStatus = (nodeId: string, status: AgentStatus) => {
    setOptimisticUpdates((prev) => ({
      ...prev,
      [nodeId]: status,
    }));

    // Clear optimistic update after a timeout
    setTimeout(() => {
      setOptimisticUpdates((prev) => {
        const { [nodeId]: _, ...rest } = prev;
        return rest;
      });
    }, 5000);
  };

  const clearOptimisticStatus = (nodeId: string) => {
    setOptimisticUpdates((prev) => {
      const { [nodeId]: _, ...rest } = prev;
      return rest;
    });
  };

  const getOptimisticStatus = (nodeId: string): AgentStatus | null => {
    return optimisticUpdates[nodeId] || null;
  };

  return {
    setOptimisticStatus,
    clearOptimisticStatus,
    getOptimisticStatus,
    hasOptimisticUpdate: (nodeId: string) => nodeId in optimisticUpdates,
  };
}

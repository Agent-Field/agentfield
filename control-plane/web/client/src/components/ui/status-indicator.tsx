import { cn } from "@/lib/utils";
import React from "react";
import type { HealthStatus, LifecycleStatus } from "../../types/haxen";
import type { CanonicalStatus } from "../../utils/status";
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from "../../utils/status";

interface StatusIndicatorProps {
  status:
    | LifecycleStatus
    | CanonicalStatus
    | string;
  healthStatus?: HealthStatus;
  showLabel?: boolean;
  animated?: boolean;
  className?: string;
}

interface StatusConfig {
  dotClass: string;
  textColor: string;
  label: string;
  shouldPulse?: boolean;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  healthStatus,
  showLabel = true,
  animated = true,
  className,
}) => {
  const getStatusConfig = (
    status: StatusIndicatorProps['status'],
    healthStatus?: HealthStatus
  ): StatusConfig => {
    // If health status indicates issues, override lifecycle status display
    if (healthStatus === "inactive") {
      return {
        dotClass: "status-dot bg-gray-400",
        textColor: "text-tertiary-foundation",
        label: "Offline",
      };
    }

    // Check if status is a LifecycleStatus
    const isLifecycle = ['starting', 'ready', 'degraded', 'offline'].includes(status as string);
    if (isLifecycle) {
      switch (status) {
        case "starting":
          return {
            dotClass: "status-dot bg-orange-500",
            textColor: "text-secondary-foundation",
            label: "Starting",
            shouldPulse: true,
          };
        case "ready":
          return {
            dotClass: "status-dot status-dot-success",
            textColor: "text-secondary-foundation",
            label: "Ready",
          };
        case "degraded":
          return {
            dotClass: "status-dot status-dot-pending",
            textColor: "text-secondary-foundation",
            label: "Degraded",
            shouldPulse: true,
          };
        case "offline":
          return {
            dotClass: "status-dot bg-gray-400",
            textColor: "text-tertiary-foundation",
            label: "Offline",
          };
        default:
          return {
            dotClass: "status-dot bg-gray-400",
            textColor: "text-tertiary-foundation",
            label: status as string,
          };
      }
    }

    const normalized = normalizeExecutionStatus(status as string);
    const theme = getStatusTheme(normalized);
    return {
      dotClass: theme.dotClass,
      textColor: theme.textClass,
      label: getStatusLabel(normalized),
      shouldPulse: normalized === 'running',
    };
  };

  const config = getStatusConfig(status, healthStatus);
  const shouldPulse = animated && config.shouldPulse;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2",
        className
      )}
    >
      {/* Status Dot - Foundation 6px */}
      <div className="relative flex items-center justify-center w-3 h-3">
        <div className={cn(config.dotClass)} />

        {/* Pulse Ring Animation - Extends beyond the dot */}
        {shouldPulse && (
          <div
            className={cn(
              "absolute w-3 h-3 rounded-full animate-ping opacity-40",
              normalizeExecutionStatus(status) === 'running' ? "bg-blue-500" : 
              status === 'starting' ? "bg-orange-500" : 
              "bg-yellow-500"
            )}
          />
        )}
      </div>

      {/* Status Label */}
      {showLabel && (
        <span className={cn("whitespace-nowrap", config.textColor)}>
          {config.label}
        </span>
      )}
    </div>
  );
};

// Utility function to get status priority for sorting
export const getStatusPriority = (
  status: LifecycleStatus,
  healthStatus?: HealthStatus
): number => {
  if (healthStatus === "inactive") return 0; // Highest priority for offline

  switch (status) {
    case "offline":
      return 0;
    case "degraded":
      return 1;
    case "starting":
      return 2;
    case "ready":
      return 3;
    default:
      return -1;
  }
};

// Utility function to determine if status needs attention
export const statusNeedsAttention = (
  status: LifecycleStatus,
  healthStatus?: HealthStatus
): boolean => {
  return (
    healthStatus === "inactive" || status === "offline" || status === "degraded"
  );
};

export default StatusIndicator;

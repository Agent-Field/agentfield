import React from "react";
import { Badge } from "@/components/ui/badge";
import type { AgentState, AgentStatus, HealthStatus, LifecycleStatus } from "@/types/agentfield";
import { cn } from "@/lib/utils";
import { statusTone, type StatusTone } from "@/lib/theme";
import type { ComponentProps } from "react";
import {
  CheckCircle,
  XCircle,
  ClockClockwise,
  PauseCircle,
  WarningOctagon,
} from "@/components/ui/icon-bridge";
import type { IconComponent } from "@/components/ui/icon-bridge";
import type { CanonicalStatus } from "@/utils/status";
import { getStatusLabel, getStatusTheme, normalizeExecutionStatus } from "@/utils/status";

interface UnifiedStatusIndicatorProps {
  status: AgentStatus;
  showDetails?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

// Status configuration mapping
const STATUS_CONFIG: Record<AgentState, {
  icon: IconComponent;
  tone: StatusTone;
  badgeVariant: NonNullable<ComponentProps<typeof Badge>["variant"]>;
  label: string;
  description: string;
}> = {
  active: {
    icon: CheckCircle,
    tone: "success",
    badgeVariant: "success",
    label: "Active",
    description: "Agent is running and healthy",
  },
  inactive: {
    icon: XCircle,
    tone: "neutral",
    badgeVariant: "unknown",
    label: "Inactive",
    description: "Agent is not responding or offline",
  },
  starting: {
    icon: ClockClockwise,
    tone: "info",
    badgeVariant: "running",
    label: "Starting",
    description: "Agent is initializing",
  },
  stopping: {
    icon: PauseCircle,
    tone: "warning",
    badgeVariant: "pending",
    label: "Stopping",
    description: "Agent is shutting down",
  },
  error: {
    icon: WarningOctagon,
    tone: "error",
    badgeVariant: "failed",
    label: "Error",
    description: "Agent encountered an error",
  },
};

// Health score color mapping
const getHealthScoreColor = (score?: number): string => {
  if (typeof score !== 'number') return statusTone.neutral.accent;
  if (score >= 90) return statusTone.success.accent;
  if (score >= 70) return statusTone.info.accent;
  if (score >= 50) return statusTone.warning.accent;
  return statusTone.error.accent;
};

// Size configuration
const SIZE_CONFIG = {
  sm: { icon: 12, badge: "text-xs" },
  md: { icon: 16, badge: "text-sm" },
  lg: { icon: 20, badge: "text-base" },
};

export function UnifiedStatusIndicator({
  status,
  showDetails = false,
  size = "md",
  className = "",
}: UnifiedStatusIndicatorProps) {
  const stateKey = status.state ?? 'inactive';
  const config = STATUS_CONFIG[stateKey];
  const sizeConfig = SIZE_CONFIG[size];
  const IconComponent = config.icon;

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };

  const getTransitionInfo = () => {
    if (!status.state_transition) return null;

    return (
      <div className="text-sm text-muted-foreground mt-1">
        Transitioning: {status.state_transition.from} →{" "}
        {status.state_transition.to}
        {status.state_transition.reason && (
          <div>Reason: {status.state_transition.reason}</div>
        )}
      </div>
    );
  };

  const statusContent = (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative">
        <IconComponent
          size={sizeConfig.icon}
          className={cn(
            statusTone[config.tone].accent,
            status.state_transition && "animate-pulse"
          )}
        />
        {status.state_transition && (
          <div
            className={cn(
              "absolute -top-1 -right-1 w-2 h-2 rounded-full animate-ping",
              statusTone.info.solidBg
            )}
          />
        )}
      </div>

      <div className="flex flex-col">
        <Badge variant={config.badgeVariant} className={sizeConfig.badge}>
          {config.label}
        </Badge>

        {showDetails && (
          <div className="text-sm text-muted-foreground mt-1">
            <div
              className={`font-medium ${getHealthScoreColor(
                status.health_score
              )}`}
            >
              Health: {typeof status.health_score === 'number' ? `${status.health_score}%` : 'N/A'}
            </div>
            <div>Last seen: {formatTimestamp(status.last_seen)}</div>
            {getTransitionInfo()}
          </div>
        )}
      </div>
    </div>
  );

  if (!showDetails) {
    return (
      <div
        title={`${config.description} - Health: ${
          typeof status.health_score === 'number' ? status.health_score : 'N/A'
        }% - Last seen: ${formatTimestamp(status.last_seen)}`}
      >
        {statusContent}
      </div>
    );
  }

  return statusContent;
}

// Helper function to determine if status is healthy
export function isStatusHealthy(status: AgentStatus): boolean {
  return status.state === "active" && (status.health_score ?? 0) >= 70;
}

// Helper function to get status priority for sorting (AgentState version)
export function getStatusPriority(state: AgentState): number {
  const priorities = {
    active: 1,
    starting: 2,
    stopping: 3,
    inactive: 4,
    error: 0,
  };
  return priorities[state] ?? 5;
}

/* ═══════════════════════════════════════════════════════════════
   Legacy StatusIndicator (previously in ui/status-indicator.tsx)
   Kept here so callers can import from this module.
   ═══════════════════════════════════════════════════════════════ */

interface LegacyStatusIndicatorProps {
  status: LifecycleStatus | CanonicalStatus | string;
  healthStatus?: HealthStatus;
  showLabel?: boolean;
  animated?: boolean;
  className?: string;
}

interface LegacyStatusConfig {
  dotClass: string;
  textColor: string;
  label: string;
  shouldPulse?: boolean;
}

const LegacyStatusIndicator: React.FC<LegacyStatusIndicatorProps> = ({
  status,
  healthStatus,
  showLabel = true,
  animated = true,
  className,
}) => {
  const getStatusConfig = (
    status: LegacyStatusIndicatorProps['status'],
    healthStatus?: HealthStatus
  ): LegacyStatusConfig => {
    if (healthStatus === "inactive") {
      return {
        dotClass: "status-dot bg-gray-400",
        textColor: "text-tertiary-foundation",
        label: "Offline",
      };
    }

    const isLifecycle = ['starting', 'ready', 'degraded', 'offline'].includes(status as string);
    if (isLifecycle) {
      switch (status) {
        case "starting":
          return { dotClass: "status-dot bg-orange-500", textColor: "text-secondary-foundation", label: "Starting", shouldPulse: true };
        case "ready":
          return { dotClass: "status-dot status-dot-success", textColor: "text-secondary-foundation", label: "Ready" };
        case "degraded":
          return { dotClass: "status-dot status-dot-pending", textColor: "text-secondary-foundation", label: "Degraded", shouldPulse: true };
        case "offline":
          return { dotClass: "status-dot bg-gray-400", textColor: "text-tertiary-foundation", label: "Offline" };
        default:
          return { dotClass: "status-dot bg-gray-400", textColor: "text-tertiary-foundation", label: status as string };
      }
    }

    const normalized = normalizeExecutionStatus(status as string);
    const theme = getStatusTheme(normalized);
    return {
      dotClass: theme.dotClass,
      textColor: theme.textClass,
      label: getStatusLabel(normalized),
      shouldPulse: normalized === 'running' || normalized === 'waiting',
    };
  };

  const config = getStatusConfig(status, healthStatus);
  const shouldPulse = animated && config.shouldPulse;

  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <div className="relative flex items-center justify-center w-3 h-3">
        <div className={cn(config.dotClass)} />
        {shouldPulse && (
          <div
            className={cn(
              "absolute w-3 h-3 rounded-full animate-ping opacity-40",
              normalizeExecutionStatus(status as string) === 'running' ? "bg-blue-500" :
              normalizeExecutionStatus(status as string) === 'waiting' ? "bg-amber-500" :
              status === 'starting' ? "bg-orange-500" :
              "bg-yellow-500"
            )}
          />
        )}
      </div>
      {showLabel && (
        <span className={cn("whitespace-nowrap", config.textColor)}>
          {config.label}
        </span>
      )}
    </div>
  );
};

/** Utility: get status priority for sorting nodes by lifecycle status */
export function getLifecycleStatusPriority(
  status: LifecycleStatus,
  healthStatus?: HealthStatus
): number {
  if (healthStatus === "inactive") return 0;
  switch (status) {
    case "offline": return 0;
    case "degraded": return 1;
    case "starting": return 2;
    case "ready": return 3;
    default: return -1;
  }
}

/** Utility: determine if a node needs attention */
export function statusNeedsAttention(
  status: LifecycleStatus,
  healthStatus?: HealthStatus
): boolean {
  return healthStatus === "inactive" || status === "offline" || status === "degraded";
}

export default LegacyStatusIndicator;

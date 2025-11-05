import { Badge } from "@/components/ui/badge";
import type { AgentState, AgentStatus } from "@/types/agentfield";
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
      <div className="text-body-small mt-1">
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
          <div className="text-body-small mt-1">
            <div
              className={`font-medium ${getHealthScoreColor(
                status.health_score
              )}`}
            >
              Health: {typeof status.health_score === 'number' ? `${status.health_score}%` : 'N/A'}
            </div>
            <div>Last seen: {formatTimestamp(status.last_seen)}</div>
            {status.mcp_status && (
              <div>
                MCP: {status.mcp_status.running_servers}/
                {status.mcp_status.total_servers} servers
              </div>
            )}
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

// Helper function to get status priority for sorting
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

import { Badge, type BadgeProps } from "@/components/ui/badge";
import type {
  AgentState,
  AgentStatus,
  HealthStatus,
  LifecycleStatus,
} from "@/types/haxen";
import { cn } from "@/lib/utils";
import { statusTone, type StatusTone } from "@/lib/theme";
import {
  CheckCircle,
  XCircle,
  ClockClockwise,
  PauseCircle,
  WarningDiamond,
  WarningOctagon,
  Question,
  SpinnerGap,
} from "@/components/ui/icon-bridge";
import type { IconComponent } from "@/components/ui/icon-bridge";

interface StatusBadgeProps {
  status?: AgentStatus;
  state?: AgentState;
  healthStatus?: HealthStatus;
  lifecycleStatus?: LifecycleStatus;
  showIcon?: boolean;
  showHealthScore?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

// Status configuration for different status types
type StatusConfigEntry = {
  icon: IconComponent;
  variant: NonNullable<BadgeProps["variant"]>;
  tone: StatusTone;
  label: string;
};

const STATE_CONFIG: Record<AgentState, StatusConfigEntry> = {
  active: {
    icon: CheckCircle,
    variant: "success",
    tone: "success",
    label: "Active",
  },
  inactive: {
    icon: XCircle,
    variant: "unknown",
    tone: "neutral",
    label: "Inactive",
  },
  starting: {
    icon: ClockClockwise,
    variant: "running",
    tone: "info",
    label: "Starting",
  },
  stopping: {
    icon: PauseCircle,
    variant: "pending",
    tone: "warning",
    label: "Stopping",
  },
  error: {
    icon: WarningOctagon,
    variant: "failed",
    tone: "error",
    label: "Error",
  },
};

const HEALTH_CONFIG: Record<HealthStatus, StatusConfigEntry> = {
  active: {
    icon: CheckCircle,
    variant: "success",
    tone: "success",
    label: "Healthy",
  },
  inactive: {
    icon: XCircle,
    variant: "failed",
    tone: "error",
    label: "Unhealthy",
  },
  unknown: {
    icon: Question,
    variant: "unknown",
    tone: "neutral",
    label: "Unknown",
  },
  starting: {
    icon: SpinnerGap,
    variant: "running",
    tone: "info",
    label: "Starting",
  },
  ready: {
    icon: CheckCircle,
    variant: "success",
    tone: "success",
    label: "Ready",
  },
  degraded: {
    icon: WarningDiamond,
    variant: "degraded",
    tone: "warning",
    label: "Degraded",
  },
  offline: {
    icon: XCircle,
    variant: "failed",
    tone: "error",
    label: "Offline",
  },
};

const LIFECYCLE_CONFIG: Record<LifecycleStatus, StatusConfigEntry> = {
  starting: {
    icon: SpinnerGap,
    variant: "running",
    tone: "info",
    label: "Starting",
  },
  ready: {
    icon: CheckCircle,
    variant: "success",
    tone: "success",
    label: "Ready",
  },
  degraded: {
    icon: WarningDiamond,
    variant: "degraded",
    tone: "warning",
    label: "Degraded",
  },
  offline: {
    icon: XCircle,
    variant: "failed",
    tone: "error",
    label: "Offline",
  },
  running: {
    icon: CheckCircle,
    variant: "running",
    tone: "info",
    label: "Running",
  },
  stopped: {
    icon: PauseCircle,
    variant: "unknown",
    tone: "neutral",
    label: "Stopped",
  },
  error: {
    icon: WarningOctagon,
    variant: "failed",
    tone: "error",
    label: "Error",
  },
  unknown: {
    icon: Question,
    variant: "unknown",
    tone: "neutral",
    label: "Unknown",
  },
};

// Size configuration
const SIZE_CONFIG = {
  sm: { icon: 12, text: "text-xs", padding: "px-2 py-1" },
  md: { icon: 14, text: "text-sm", padding: "px-3 py-1" },
  lg: { icon: 16, text: "text-base", padding: "px-4 py-2" },
};

export function StatusBadge({
  status,
  state,
  healthStatus,
  lifecycleStatus,
  showIcon = true,
  showHealthScore = false,
  size = "md",
  className = "",
}: StatusBadgeProps) {
  // Determine which status to display (priority: status > state > healthStatus > lifecycleStatus)
  let config: StatusConfigEntry = HEALTH_CONFIG.unknown;
  let label = "Unknown";
  let isTransitioning = false;

  if (status?.state) {
    const stateKey = status.state;
    config = STATE_CONFIG[stateKey] ?? HEALTH_CONFIG.unknown;
    label = config.label;
    isTransitioning = !!status.state_transition;

    if (showHealthScore && typeof status.health_score === 'number') {
      label += ` (${status.health_score}%)`;
    }
  } else if (state) {
    config = STATE_CONFIG[state] ?? HEALTH_CONFIG.unknown;
    label = config.label;
  } else if (healthStatus) {
    config = HEALTH_CONFIG[healthStatus] ?? HEALTH_CONFIG.unknown;
    label = config.label;
  } else if (lifecycleStatus) {
    config = LIFECYCLE_CONFIG[lifecycleStatus] ?? HEALTH_CONFIG.unknown;
    label = config.label;
  }

  const sizeConfig = SIZE_CONFIG[size];
  const IconComponent = config.icon;

  return (
    <Badge
      variant={config.variant}
      className={cn(
        sizeConfig.text,
        sizeConfig.padding,
        isTransitioning ? "animate-pulse" : undefined,
        "flex items-center gap-1",
        className
      )}
    >
      {showIcon && (
        <IconComponent
          size={sizeConfig.icon}
          className={cn(
            isTransitioning ? "animate-spin" : undefined,
            statusTone[config.tone].accent
          )}
        />
      )}
      <span>{label}</span>
      {isTransitioning && status?.state_transition && (
        <span className="text-xs opacity-75">
          → {STATE_CONFIG[status.state_transition.to]?.label}
        </span>
      )}
    </Badge>
  );
}

// Specialized badge components for specific use cases
export function AgentStateBadge({
  state,
  showIcon = true,
  size = "md",
  className = "",
}: {
  state: AgentState;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <StatusBadge
      state={state}
      showIcon={showIcon}
      size={size}
      className={className}
    />
  );
}

export function HealthStatusBadge({
  healthStatus,
  showIcon = true,
  size = "md",
  className = "",
}: {
  healthStatus: HealthStatus;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <StatusBadge
      healthStatus={healthStatus}
      showIcon={showIcon}
      size={size}
      className={className}
    />
  );
}

export function LifecycleStatusBadge({
  lifecycleStatus,
  showIcon = true,
  size = "md",
  className = "",
}: {
  lifecycleStatus: LifecycleStatus;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <StatusBadge
      lifecycleStatus={lifecycleStatus}
      showIcon={showIcon}
      size={size}
      className={className}
    />
  );
}

// Helper function to get health score color
export function getHealthScoreColor(score: number): string {
  if (score >= 90) return statusTone.success.accent;
  if (score >= 70) return statusTone.info.accent;
  if (score >= 50) return statusTone.warning.accent;
  return statusTone.error.accent;
}

// Helper function to get health score badge variant
export function getHealthScoreBadgeVariant(
  score: number
): NonNullable<BadgeProps["variant"]> {
  if (score >= 90) return "success";
  if (score >= 70) return "running";
  if (score >= 50) return "pending";
  return "failed";
}

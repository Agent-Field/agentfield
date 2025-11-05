import { Button } from "@/components/ui/button";
import {
  ArrowLeft,
  Maximize,
  Minimize,
} from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";
import { getNodeStatusPresentation } from "@/utils/node-status";
import type { HealthStatus, LifecycleStatus } from "@/types/agentfield";
import type { ReactNode } from "react";

interface EnhancedNodeDetailHeaderProps {
  nodeId: string;
  teamId?: string;
  version?: string;
  lifecycleStatus?: LifecycleStatus | null;
  healthStatus?: HealthStatus | null;
  deploymentType?: string | null;
  lastHeartbeat?: Date | string | null;
  metadata?: Array<{ label: string; value: string }>;
  onBack?: () => void;
  focusMode: boolean;
  onFocusModeChange: (enabled: boolean) => void;
  isFullscreen: boolean;
  onFullscreenChange: (enabled: boolean) => void;
  viewMode: "standard" | "performance" | "debug";
  onViewModeChange: (mode: "standard" | "performance" | "debug") => void;
  rightActions?: ReactNode;
  statusBadges?: ReactNode;
  liveStatusBadge?: ReactNode;
}

const formatHeartbeat = (heartbeat?: Date | string | null) => {
  if (!heartbeat) return null;
  const date = typeof heartbeat === "string" ? new Date(heartbeat) : heartbeat;
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toLocaleString();
};

export function EnhancedNodeDetailHeader({
  nodeId,
  lifecycleStatus,
  healthStatus,
  lastHeartbeat,
  onBack,
  isFullscreen,
  onFullscreenChange,
  rightActions,
  statusBadges,
  liveStatusBadge,
}: EnhancedNodeDetailHeaderProps) {
  const statusPresentation = getNodeStatusPresentation(
    lifecycleStatus,
    healthStatus
  );

  const heartbeatDisplay = formatHeartbeat(lastHeartbeat);

  return (
    <div className="h-14 bg-background border-b border-border px-4 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0 flex-1 overflow-hidden">
        {onBack && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="h-8 w-8 p-0 flex-shrink-0"
            title="Back to nodes"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>
        )}

        <div className="flex min-w-0 flex-col flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={cn(
                "w-2 h-2 rounded-full flex-shrink-0",
                statusPresentation.theme.indicatorClass,
                statusPresentation.shouldPulse && "animate-pulse"
              )}
            />
            <span
              className={cn(
                "text-xs font-medium uppercase tracking-wide flex-shrink-0",
                statusPresentation.theme.textClass
              )}
            >
              {statusPresentation.label}
            </span>
            <span className="text-sm font-semibold text-foreground truncate">
              {nodeId}
            </span>
          </div>
          {(heartbeatDisplay || statusBadges) && (
            <div className="hidden items-center gap-3 text-body-small md:flex">
              {heartbeatDisplay && <span>Last heartbeat {heartbeatDisplay}</span>}
              {statusBadges}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        {liveStatusBadge}

        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0"
          onClick={() => onFullscreenChange(!isFullscreen)}
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
        >
          {isFullscreen ? (
            <Minimize className="w-4 h-4" />
          ) : (
            <Maximize className="w-4 h-4" />
          )}
        </Button>

        {rightActions}
      </div>
    </div>
  );
}

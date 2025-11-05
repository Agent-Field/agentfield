import StatusIndicator from "@/components/ui/status-indicator";
import { cn } from "@/lib/utils";
import type { LifecycleStatus } from "../../types/haxen";

export type MCPHealthStatus =
  | "running"
  | "stopped"
  | "error"
  | "starting"
  | "unknown";

interface MCPHealthIndicatorProps {
  status: MCPHealthStatus;
  size?: "sm" | "md" | "lg";
  showText?: boolean;
  className?: string;
  tooltip?: string;
  uptime?: string;
}

/**
 * Reusable health status indicator component for MCP servers
 * Displays color-coded status with optional icon and text
 */
export function MCPHealthIndicator({
  status,
  showText = true,
  className,
  tooltip,
  uptime,
}: MCPHealthIndicatorProps) {
  const getStatus = (status: MCPHealthStatus): LifecycleStatus => {
    switch (status) {
      case "running":
        return "ready";
      case "error":
        return "degraded";
      case "starting":
        return "starting";
      case "stopped":
      case "unknown":
      default:
        return "offline";
    }
  };

  const tooltipText =
    tooltip || `${status}${uptime ? ` (Uptime: ${uptime})` : ""}`;

  return (
    <div
      className={cn("inline-flex items-center", className)}
      title={tooltipText}
      role="status"
      aria-label={`MCP server status: ${status}`}
    >
      <StatusIndicator
        status={getStatus(status)}
        showLabel={showText}
        animated={status === "starting"}
      />
    </div>
  );
}

/**
 * Simplified health indicator that only shows a colored dot
 */
export function MCPHealthDot({
  status,
  size = "md",
  className,
}: {
  status: MCPHealthStatus;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const getStatusConfig = (status: MCPHealthStatus) => {
    switch (status) {
      case "running":
        return "bg-status-success";
      case "starting":
        return "bg-status-info";
      case "stopped":
        return "bg-status-neutral";
      case "error":
        return "bg-status-error";
      case "unknown":
      default:
        return "bg-status-neutral";
    }
  };

  const sizeClasses = {
    sm: "w-2 h-2",
    md: "w-3 h-3",
    lg: "w-4 h-4",
  };

  return (
    <div
      className={cn(
        "rounded-full",
        sizeClasses[size],
        getStatusConfig(status),
        className
      )}
      role="status"
      aria-label={`MCP server status: ${status}`}
    />
  );
}

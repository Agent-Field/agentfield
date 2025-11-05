import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  ArrowClockwise,
  Plus,
  WifiHigh,
  WifiSlash,
} from "@/components/ui/icon-bridge";
import { getNodeStatusPresentation } from "@/utils/node-status";
import type { ReactNode } from "react";

interface EnhancedNodesHeaderProps {
  totalNodes: number;
  onlineCount: number;
  offlineCount: number;
  degradedCount: number;
  startingCount: number;
  lastUpdated?: Date;
  isConnected: boolean;
  isReconnecting?: boolean;
  onAddServerless?: () => void;
  onReconnect?: () => void;
  actions?: ReactNode;
  subtitle?: string;
}

const summaryItem = (
  count: number,
  label: string,
  canonical: Parameters<typeof getNodeStatusPresentation>[0],
  secondaryCanonical?: Parameters<typeof getNodeStatusPresentation>[1]
) => {
  const presentation = getNodeStatusPresentation(canonical, secondaryCanonical);
  return (
    <span
      key={label}
      className="flex items-center gap-1 text-body-small text-text-secondary"
    >
      <span
        className={cn(
          "w-2 h-2 rounded-full",
          presentation.theme.indicatorClass,
          presentation.shouldPulse && "animate-pulse"
        )}
      />
      {count} {label}
    </span>
  );
};

export function EnhancedNodesHeader({
  totalNodes,
  onlineCount,
  offlineCount,
  degradedCount,
  startingCount,
  lastUpdated,
  isConnected,
  isReconnecting = false,
  onAddServerless,
  onReconnect,
  actions,
  subtitle = "Monitor and manage your AI agent nodes in the Haxen orchestration platform.",
}: EnhancedNodesHeaderProps) {
  const ConnectionIcon = isConnected ? WifiHigh : WifiSlash;
  const connectionLabel = isConnected
    ? "Live updates"
    : isReconnecting
      ? "Reconnecting…"
      : "Disconnected";

  const formatLastUpdated = (date: Date) => {
    const diffMs = Date.now() - date.getTime();
    if (diffMs < 0) return "just now";
    const diffSeconds = Math.floor(diffMs / 1000);
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    const diffMinutes = Math.floor(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className="bg-background border-b border-border px-4 py-3 flex items-center justify-between min-h-[70px]">
      <div className="flex items-center gap-4 min-w-0 flex-1">
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <h1 className="text-heading-2 leading-tight">Agent Nodes</h1>
            <Badge variant="count" size="sm">
              {totalNodes} total
            </Badge>
          </div>
          <p className="hidden text-body md:block text-text-secondary">
            {subtitle}
          </p>
        </div>

        <div className="hidden items-center gap-3 md:flex">
          {summaryItem(onlineCount, "online", "ready", "active")}
          {startingCount > 0 && summaryItem(startingCount, "starting", "starting")}
          {degradedCount > 0 && summaryItem(degradedCount, "degraded", "degraded")}
          {summaryItem(offlineCount, "offline", "offline", "inactive")}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {actions}

        <Badge
          variant={isConnected ? 'success' : isReconnecting ? 'pending' : 'failed'}
          size="sm"
          showIcon={false}
          className="flex items-center gap-1 rounded-full px-3"
        >
          <ConnectionIcon className="w-3.5 h-3.5" />
          {connectionLabel}
        </Badge>

        {isConnected && lastUpdated && (
          <Badge variant="pill" className="hidden lg:flex">
            Updated {formatLastUpdated(lastUpdated)}
          </Badge>
        )}

        {!isConnected && onReconnect && (
          <Button
            variant="outline"
            size="sm"
            className="h-8 px-3"
            onClick={onReconnect}
          >
            <ArrowClockwise className="w-4 h-4" />
            <span className="ml-1">Reconnect</span>
          </Button>
        )}

        {onAddServerless && (
          <Button
            size="sm"
            className="h-8 px-3 gap-1.5"
            onClick={onAddServerless}
          >
            <Plus className="w-4 h-4" />
            Add Serverless Agent
          </Button>
        )}
      </div>
    </div>
  );
}

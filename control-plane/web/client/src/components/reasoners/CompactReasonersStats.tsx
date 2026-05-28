import {
  Wifi,
  WifiOff,
  Grid,
  Terminal,
  Renew
} from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";
import { formatCompactRelativeTime } from "@/utils/dateFormat";
import { Button } from "../ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";

interface CompactReasonersStatsProps {
  total: number;
  onlineCount: number;
  offlineCount: number;
  nodesCount: number;
  lastRefresh: Date;
  loading?: boolean;
  onRefresh?: () => void;
  className?: string;
}

export function CompactReasonersStats({
  total = 0,
  onlineCount = 0,
  offlineCount = 0,
  nodesCount = 0,
  lastRefresh,
  loading = false,
  onRefresh,
  className
}: CompactReasonersStatsProps) {
  // Ensure we have safe values
  const safeTotal = total ?? 0;
  const safeOnlineCount = onlineCount ?? 0;
  const safeOfflineCount = offlineCount ?? 0;
  const safeNodesCount = nodesCount ?? 0;
  const safeLastRefresh = lastRefresh || new Date();

  const formatRelativeTime = formatCompactRelativeTime;

  const formatLastRefresh = (timestamp: Date) => {
    try {
      return timestamp.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.warn('Error formatting last refresh time:', error);
      return "Unknown";
    }
  };

  return (
    <div className={cn(
      "bg-card border border-border rounded-lg px-4 py-3 foundation-transition",
      "hover:bg-card/80 hover:border-border/80",
      className
    )}>
      {/* Ultra-Compact Single Bar */}
      <div className="flex items-center justify-between gap-4 text-sm">
        {/* Left: Core Stats */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {/* Online Count */}
          <div className="flex items-center gap-2">
            <Wifi className="h-4 w-4 text-status-success flex-shrink-0" />
            <span className="text-status-success font-medium">{safeOnlineCount}</span>
            <span className="text-sm text-muted-foreground">online</span>
          </div>

          <span className="text-muted-foreground/60">•</span>

          {/* Offline Count */}
          <div className="flex items-center gap-2">
            <WifiOff className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span className="text-muted-foreground font-medium">{safeOfflineCount}</span>
            <span className="text-sm text-muted-foreground">offline</span>
          </div>

          <span className="text-muted-foreground/60">•</span>

          {/* Total Count */}
          <div className="flex items-center gap-2">
            <Grid className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span className="text-foreground font-medium">{safeTotal}</span>
            <span className="text-sm text-muted-foreground">total</span>
          </div>
        </div>

        {/* Center: Additional Metrics */}
        <div className="hidden md:flex items-center gap-3 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Terminal className="h-3 w-3 text-muted-foreground flex-shrink-0" />
            <span className="font-medium text-foreground">{safeNodesCount}</span>
            <span>nodes</span>
          </div>

          <span>•</span>

          <span>
            Health: <span className="font-medium text-foreground">
              {safeTotal > 0 ? Math.round((safeOnlineCount / safeTotal) * 100) : 0}%
            </span>
          </span>
        </div>

        {/* Right: Last Updated + Refresh */}
        <div className="flex items-center gap-3 text-sm text-muted-foreground flex-shrink-0">
          <HoverCard>
            <HoverCardTrigger asChild>
              <span className="cursor-pointer hover:text-foreground transition-colors">
                Updated: <span className="font-medium">{formatRelativeTime(safeLastRefresh)} ago</span>
              </span>
            </HoverCardTrigger>
            <HoverCardContent className="w-auto p-2">
              <p className="text-xs">{formatLastRefresh(safeLastRefresh)}</p>
            </HoverCardContent>
          </HoverCard>

          {onRefresh && (
            <>
              <span>•</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                disabled={loading}
                className="h-5 w-5 p-0 hover:bg-muted/80 text-muted-foreground hover:text-foreground"
              >
                <Renew className={cn("h-3 w-3", loading && "animate-spin")} />
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Mobile: Additional Metrics Row */}
      <div className="md:hidden flex items-center gap-3 text-sm text-muted-foreground mt-2 pt-2 border-t border-border/50">
        <div className="flex items-center gap-2">
          <Terminal className="h-3 w-3 text-muted-foreground flex-shrink-0" />
          <span className="font-medium text-foreground">{safeNodesCount}</span>
          <span>nodes</span>
        </div>

        <span>•</span>

        <span>
          Health: <span className="font-medium text-foreground">
            {safeTotal > 0 ? Math.round((safeOnlineCount / safeTotal) * 100) : 0}%
          </span>
        </span>

        <span>•</span>

        <span>
          Last: <span className="font-medium text-foreground">{formatRelativeTime(safeLastRefresh)} ago</span>
        </span>
      </div>
    </div>
  );
}

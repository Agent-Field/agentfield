import { useMemo } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  cardVariants,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { AlertTriangle } from "@/components/ui/icon-bridge";
import type { HotspotItem } from "@/types/dashboard";

interface HotspotPanelProps {
  /** List of hotspot items sorted by failure count */
  hotspots: HotspotItem[];
  /** Additional class name */
  className?: string;
}

/**
 * Panel showing top error contributors by reasoner.
 * Displays failure rate, contribution percentage, and top error messages.
 */
export function HotspotPanel({ hotspots, className }: HotspotPanelProps) {
  // Calculate max failed for progress bar scaling
  const maxFailed = useMemo(() => {
    if (hotspots.length === 0) return 0;
    return Math.max(...hotspots.map((h) => h.failed_executions));
  }, [hotspots]);

  return (
    <Card
      variant="surface"
      interactive={false}
      className={cn("flex h-full flex-col", className)}
    >
      <CardHeader className="p-5 pb-2">
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          Problem Hotspots
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3 p-5 pt-0 min-h-0">
        {hotspots.length === 0 ? (
          <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border/40 bg-muted/10 p-4 text-center">
            <p className="text-sm text-muted-foreground">
              No failures detected in this time period.
            </p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto space-y-3 pr-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border/70">
            {hotspots.slice(0, 5).map((hotspot, index) => (
              <HotspotRow
                key={hotspot.reasoner_id}
                hotspot={hotspot}
                rank={index + 1}
                maxFailed={maxFailed}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface HotspotRowProps {
  hotspot: HotspotItem;
  rank: number;
  maxFailed: number;
}

function HotspotRow({ hotspot, rank, maxFailed }: HotspotRowProps) {
  const progressWidth =
    maxFailed > 0 ? (hotspot.failed_executions / maxFailed) * 100 : 0;

  return (
    <div
      className={cn(
        cardVariants({ variant: "muted", interactive: false }),
        "relative overflow-hidden px-3 py-3 text-xs"
      )}
    >
      {/* Rank badge */}
      <div className="absolute left-2 top-2 flex h-5 w-5 items-center justify-center rounded-full bg-background border border-border text-[10px] font-mono font-medium text-muted-foreground">
        {rank}
      </div>

      <div className="pl-6 space-y-2">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="font-medium text-foreground truncate">
              {hotspot.reasoner_id}
            </p>
          </div>
          <Badge
            variant="outline"
            className="flex-shrink-0 rounded-full border-destructive/40 text-destructive bg-destructive/5 text-[10px]"
          >
            {hotspot.contribution_pct.toFixed(0)}% of errors
          </Badge>
        </div>

        {/* Progress bar */}
        <div className="space-y-1">
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-destructive/70 transition-all duration-500"
              style={{ width: `${progressWidth}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>
              {hotspot.failed_executions} failed / {hotspot.total_executions}{" "}
              total
            </span>
            <span className="text-destructive">
              {hotspot.error_rate.toFixed(1)}% error rate
            </span>
          </div>
        </div>

        {/* Top error */}
        {hotspot.top_errors.length > 0 && (
          <p className="text-[10px] text-muted-foreground truncate">
            <span className="text-destructive/70">Top:</span>{" "}
            {hotspot.top_errors[0].message}
            {hotspot.top_errors[0].count > 1 && (
              <span className="text-muted-foreground/70">
                {" "}
                ({hotspot.top_errors[0].count}x)
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Sparkline } from "@/components/ui/Sparkline";
import { cn } from "@/lib/utils";
import type { IconComponent } from "@/components/ui/icon-bridge";
import {
  calculateTrend,
  formatDeltaWithArrow,
  getTrendColorClass,
  type TrendPolarity,
} from "@/utils/trendUtils";

interface TrendMetricCardProps {
  /** Label shown above the value */
  label: string;
  /** Current value to display */
  value: string;
  /** Current numeric value for trend calculation */
  currentValue?: number;
  /** Previous period numeric value for trend calculation */
  previousValue?: number;
  /** How to interpret trend direction (up-is-good for success rate, down-is-good for errors) */
  trendPolarity?: TrendPolarity;
  /** Historical data for sparkline (newest last) */
  sparklineData?: number[];
  /** Optional icon */
  icon?: IconComponent;
  /** Secondary info shown below the trend */
  subtitle?: string;
  /** Loading state */
  loading?: boolean;
  /** Additional class name */
  className?: string;
}

/**
 * Metric card with trend indicator and sparkline.
 * Shows delta arrow with color coding based on polarity.
 */
export function TrendMetricCard({
  label,
  value,
  currentValue,
  previousValue,
  trendPolarity = "neutral",
  sparklineData,
  icon: Icon,
  subtitle,
  loading = false,
  className,
}: TrendMetricCardProps) {
  // Calculate trend if we have comparison values
  const trend = useMemo(() => {
    if (currentValue !== undefined && previousValue !== undefined) {
      return calculateTrend(currentValue, previousValue, trendPolarity);
    }
    return null;
  }, [currentValue, previousValue, trendPolarity]);

  // Determine sparkline color based on trend
  const sparklineColor = useMemo(() => {
    if (!trend) return "var(--primary)";
    switch (trend.color) {
      case "success":
        return "rgb(16 185 129)"; // emerald-500
      case "destructive":
        return "var(--destructive)";
      default:
        return "var(--primary)";
    }
  }, [trend]);

  return (
    <Card
      className={cn(
        "rounded-xl border-border/50 bg-gradient-to-br from-muted/30 to-muted/5",
        className
      )}
    >
      <CardContent className="flex flex-col justify-between h-28 p-4">
        {/* Header with label and icon */}
        <div className="flex items-center justify-between text-text-tertiary">
          <span className="text-caption tracking-wider">{label}</span>
          {Icon && <Icon className="h-3.5 w-3.5 text-text-tertiary" />}
        </div>

        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-6 w-24" />
            <Skeleton className="h-3 w-16" />
          </div>
        ) : (
          <div className="flex items-end justify-between gap-2">
            {/* Value and trend */}
            <div className="min-w-0 flex-1">
              <p className="text-heading-2 leading-none text-foreground">
                {value}
              </p>
              {trend && (
                <p
                  className={cn(
                    "text-xs mt-1 font-medium",
                    getTrendColorClass(trend.color)
                  )}
                >
                  {formatDeltaWithArrow(trend)} vs prev
                </p>
              )}
              {!trend && subtitle && (
                <p className="text-xs mt-1 text-muted-foreground">{subtitle}</p>
              )}
            </div>

            {/* Sparkline */}
            {sparklineData && sparklineData.length >= 2 && (
              <div className="flex-shrink-0">
                <Sparkline
                  data={sparklineData}
                  width={56}
                  height={24}
                  color={sparklineColor}
                  showArea={true}
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

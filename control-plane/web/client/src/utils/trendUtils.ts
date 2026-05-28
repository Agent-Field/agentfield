/**
 * Utilities for calculating and formatting trend data
 */

export type TrendDirection = "up" | "down" | "flat";
export type TrendPolarity = "up-is-good" | "down-is-good" | "neutral";

export interface TrendResult {
  direction: TrendDirection;
  percentage: number;
  absoluteDelta: number;
  displayText: string;
  color: "success" | "destructive" | "muted";
}

/**
 * Calculate trend between current and previous values
 */
export function calculateTrend(
  current: number,
  previous: number,
  polarity: TrendPolarity = "neutral"
): TrendResult {
  const absoluteDelta = current - previous;
  let percentage = 0;

  if (previous !== 0) {
    percentage = (absoluteDelta / previous) * 100;
  } else if (current !== 0) {
    // Previous was 0, current is not - infinite increase
    percentage = 100;
  }

  // Determine direction
  let direction: TrendDirection = "flat";
  if (Math.abs(percentage) >= 0.1) {
    direction = percentage > 0 ? "up" : "down";
  }

  // Format display text
  let displayText = "—";
  if (direction !== "flat") {
    const sign = percentage > 0 ? "+" : "";
    displayText = `${sign}${percentage.toFixed(1)}%`;
  }

  // Determine color based on polarity
  let color: "success" | "destructive" | "muted" = "muted";
  if (direction !== "flat") {
    if (polarity === "up-is-good") {
      color = direction === "up" ? "success" : "destructive";
    } else if (polarity === "down-is-good") {
      color = direction === "down" ? "success" : "destructive";
    }
  }

  return {
    direction,
    percentage,
    absoluteDelta,
    displayText,
    color,
  };
}

/**
 * Format a delta value for display with arrow
 */
export function formatDeltaWithArrow(delta: TrendResult): string {
  if (delta.direction === "flat") {
    return "—";
  }
  const arrow = delta.direction === "up" ? "↑" : "↓";
  return `${arrow} ${delta.displayText}`;
}

/**
 * Get CSS class for trend color
 */
export function getTrendColorClass(color: TrendResult["color"]): string {
  switch (color) {
    case "success":
      return "text-emerald-500";
    case "destructive":
      return "text-destructive";
    default:
      return "text-muted-foreground";
  }
}

/**
 * Normalize data points for sparkline (0-1 range)
 */
export function normalizeForSparkline(data: number[]): number[] {
  if (data.length === 0) return [];

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;

  if (range === 0) {
    // All values are the same - return flat line at 0.5
    return data.map(() => 0.5);
  }

  return data.map((value) => (value - min) / range);
}

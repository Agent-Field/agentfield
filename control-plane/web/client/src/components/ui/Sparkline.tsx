import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { normalizeForSparkline } from "@/utils/trendUtils";

interface SparklineProps {
  /** Data points to visualize */
  data: number[];
  /** Width of the sparkline */
  width?: number;
  /** Height of the sparkline */
  height?: number;
  /** Stroke color - uses CSS variable for theme support */
  color?: string;
  /** Whether to show area fill under the line */
  showArea?: boolean;
  /** Additional class name */
  className?: string;
}

/**
 * Minimal inline sparkline chart using pure SVG.
 * Automatically normalizes data to fit within the height.
 * Uses theme tokens for colors.
 */
export function Sparkline({
  data,
  width = 60,
  height = 20,
  color = "currentColor",
  showArea = true,
  className,
}: SparklineProps) {
  const { pathD, areaD } = useMemo(() => {
    if (data.length < 2) {
      return { pathD: "", areaD: "" };
    }

    const normalized = normalizeForSparkline(data);
    const padding = 2; // Padding from edges
    const effectiveWidth = width - padding * 2;
    const effectiveHeight = height - padding * 2;

    // Generate points
    const points = normalized.map((value, index) => {
      const x = padding + (index / (normalized.length - 1)) * effectiveWidth;
      // Invert Y because SVG Y-axis is top-down
      const y = padding + (1 - value) * effectiveHeight;
      return { x, y };
    });

    // Create line path
    const pathCommands = points.map((point, index) => {
      return index === 0
        ? `M ${point.x} ${point.y}`
        : `L ${point.x} ${point.y}`;
    });
    const pathD = pathCommands.join(" ");

    // Create area path (closed polygon)
    const areaCommands = [
      ...pathCommands,
      `L ${points[points.length - 1].x} ${height - padding}`,
      `L ${points[0].x} ${height - padding}`,
      "Z",
    ];
    const areaD = areaCommands.join(" ");

    return { pathD, areaD };
  }, [data, width, height]);

  if (data.length < 2) {
    return (
      <div
        className={cn("flex items-center justify-center", className)}
        style={{ width, height }}
      >
        <span className="text-[10px] text-muted-foreground">—</span>
      </div>
    );
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={cn("overflow-visible", className)}
      style={{ color }}
    >
      {showArea && (
        <path
          d={areaD}
          fill="currentColor"
          fillOpacity={0.15}
          stroke="none"
        />
      )}
      <path
        d={pathD}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

import { cn } from "@/lib/utils";
import { Analytics, Renew, Warning, Restart } from "@/components/ui/icon-bridge";
import { useMemo, useCallback, memo } from "react";
import { useExecutionTimeline } from "../../hooks/useExecutionTimeline";

interface ExecutionTimelineProps {
  className?: string;
}

export const ExecutionTimeline = memo(function ExecutionTimeline({ className }: ExecutionTimelineProps) {
  const {
    timelineData,
    summary,
    loading,
    error,
    hasData,
    hasError,
    isEmpty,
    isRefreshing,
    refresh,
    clearError
  } = useExecutionTimeline();

  // Memoize chart dimensions to prevent recalculation
  const chartDimensions = useMemo(() => ({
    chartHeight: 200,
    chartWidth: 600,
    padding: { top: 20, right: 60, bottom: 40, left: 60 }
  }), []);

  const { chartHeight, chartWidth, padding } = chartDimensions;
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Memoize data reference to prevent unnecessary recalculations
  const data = useMemo(() => timelineData, [timelineData]);

  // Memoize metrics calculations
  const metrics = useMemo(() => {
    if (!summary && (!data || data.length === 0)) {
      return {
        totalExecutions: 0,
        avgSuccessRate: 0,
        totalErrors: 0
      };
    }

    return {
      totalExecutions: summary?.total_executions ?? data.reduce((sum, d) => sum + d.executions, 0),
      avgSuccessRate: summary?.avg_success_rate ?? (data.length > 0 ? data.reduce((sum, d) => sum + d.success_rate, 0) / data.length : 0),
      totalErrors: summary?.total_errors ?? data.reduce((sum, d) => sum + d.failed, 0)
    };
  }, [summary, data]);

  // Optimized chart calculations with better memoization
  const chartData = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        maxExecutions: 0,
        minSuccessRate: 0,
        maxSuccessRate: 100,
        executionPath: "",
        successRatePath: "",
        executionArea: "",
        dataPoints: []
      };
    }

    // Calculate min/max values efficiently
    let maxExec = 0;
    let minSuccess = Infinity;
    let maxSuccess = -Infinity;

    for (const d of data) {
      maxExec = Math.max(maxExec, d.executions);
      minSuccess = Math.min(minSuccess, d.success_rate);
      maxSuccess = Math.max(maxSuccess, d.success_rate);
    }

    // Avoid division by zero
    const successRange = maxSuccess - minSuccess || 1;
    const dataLength = data.length;
    const widthStep = dataLength > 1 ? innerWidth / (dataLength - 1) : 0;

    // Pre-calculate points for better performance
    const executionPoints: string[] = [];
    const successPoints: string[] = [];
    const dataPoints: Array<{ x: number; y: number; data: typeof data[0] }> = [];

    for (let i = 0; i < dataLength; i++) {
      const d = data[i];
      const x = i * widthStep;

      // Validate and sanitize data to prevent NaN values
      const executions = typeof d.executions === 'number' && !isNaN(d.executions) ? d.executions : 0;
      const successRate = typeof d.success_rate === 'number' && !isNaN(d.success_rate) ? d.success_rate : 0;

      // Prevent division by zero
      const safeMaxExec = maxExec > 0 ? maxExec : 1;
      const safeSuccessRange = successRange > 0 ? successRange : 1;

      const execY = innerHeight - (executions / safeMaxExec) * innerHeight;
      const successY = innerHeight - ((successRate - minSuccess) / safeSuccessRange) * innerHeight;

      // Final validation to ensure no NaN values
      const safeExecY = isNaN(execY) ? innerHeight : execY;
      const safeSuccessY = isNaN(successY) ? innerHeight : successY;

      executionPoints.push(`${x},${safeExecY}`);
      successPoints.push(`${x},${safeSuccessY}`);
      dataPoints.push({ x: x + padding.left, y: execY + padding.top, data: d });
    }

    return {
      maxExecutions: maxExec,
      minSuccessRate: minSuccess,
      maxSuccessRate: maxSuccess,
      executionPath: `M ${executionPoints.join(" L ")}`,
      successRatePath: `M ${successPoints.join(" L ")}`,
      executionArea: `M 0,${innerHeight} L ${executionPoints.join(" L ")} L ${innerWidth},${innerHeight} Z`,
      dataPoints
    };
  }, [data, innerWidth, innerHeight, padding.left, padding.top]);

  // Memoized event handlers
  const handleRefresh = useCallback(() => {
    if (hasError) {
      clearError();
    }
    refresh();
  }, [hasError, clearError, refresh]);

  // Loading state
  if (loading && !hasData) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Analytics className="h-5 w-5 text-primary" />
            <div>
              <h3 className="text-sm font-medium text-foreground">Execution Timeline</h3>
              <p className="text-sm text-muted-foreground">Last 24 hours</p>
            </div>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className={cn(
              "flex items-center space-x-1 px-2 py-1 rounded text-xs",
              "border border-border hover:bg-muted",
              "transition-colors duration-200",
              loading && "opacity-50 cursor-not-allowed"
            )}
          >
            <Renew className={cn("h-3 w-3", loading && "animate-spin")} />
            <span>Refresh</span>
          </button>
        </div>

        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-2">
            <Renew className="h-8 w-8 text-muted-foreground animate-spin" />
            <p className="text-sm text-muted-foreground">Loading execution timeline...</p>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (hasError) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Analytics className="h-5 w-5 text-primary" />
            <div>
              <h3 className="text-sm font-medium text-foreground">Execution Timeline</h3>
              <p className="text-sm text-muted-foreground">Last 24 hours</p>
            </div>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className={cn(
              "flex items-center space-x-1 px-2 py-1 rounded text-xs",
              "border border-border hover:bg-muted",
              "transition-colors duration-200",
              loading && "opacity-50 cursor-not-allowed"
            )}
          >
            <Restart className={cn("h-3 w-3", loading && "animate-spin")} />
            <span>Retry</span>
          </button>
        </div>

        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4 text-center">
            <Warning className="h-12 w-12" style={{ color: "var(--status-error)" }} />
            <div>
              <p className="text-sm font-medium text-foreground mb-1">Failed to load timeline data</p>
              <p className="text-sm text-muted-foreground mb-3">{error?.message || "An unexpected error occurred"}</p>
              <button
                onClick={handleRefresh}
                disabled={loading}
                className={cn(
                  "inline-flex items-center space-x-1 px-3 py-1 rounded text-xs",
                  "bg-primary text-primary-foreground hover:bg-primary/90",
                  "transition-colors duration-200",
                  loading && "opacity-50 cursor-not-allowed"
                )}
              >
                <Restart className={cn("h-3 w-3", loading && "animate-spin")} />
                <span>Try Again</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Empty state
  if (isEmpty || !hasData || data.length === 0) {
    return (
      <div className={cn("rounded-xl border border-border bg-card p-6 shadow-sm", className)}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Analytics className="h-5 w-5 text-primary" />
            <div>
              <h3 className="text-sm font-medium text-foreground">Execution Timeline</h3>
              <p className="text-sm text-muted-foreground">Last 24 hours</p>
            </div>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className={cn(
              "flex items-center space-x-1 px-2 py-1 rounded text-xs",
              "border border-border hover:bg-muted",
              "transition-colors duration-200",
              loading && "opacity-50 cursor-not-allowed"
            )}
          >
            <Renew className={cn("h-3 w-3", loading && "animate-spin")} />
            <span>Refresh</span>
          </button>
        </div>

        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-2 text-center">
            <Analytics className="h-12 w-12 text-muted-foreground" />
            <div>
              <p className="text-sm font-medium text-foreground mb-1">No execution data available</p>
              <p className="text-sm text-muted-foreground">No executions have been recorded in the last 24 hours</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Success state with data
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-card p-6 shadow-sm",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Analytics className="h-5 w-5 text-primary" />
          <div>
            <h3 className="text-sm font-medium text-foreground">
              Execution Timeline
            </h3>
            <p className="text-sm text-muted-foreground">Last 24 hours</p>
          </div>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className={cn(
            "flex items-center space-x-1 px-2 py-1 rounded text-xs",
            "border border-border hover:bg-muted",
            "transition-colors duration-200",
            loading && "opacity-50 cursor-not-allowed"
          )}
        >
          <Renew className={cn("h-3 w-3", (loading || isRefreshing) && "animate-spin")} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-base font-semibold">
            {metrics.totalExecutions.toLocaleString()}
          </div>
          <div className="text-sm text-muted-foreground">Total Executions</div>
        </div>
        <div className="text-center">
          <div className="text-base font-semibold" style={{ color: "var(--status-success)" }}>
            {metrics.avgSuccessRate.toFixed(1)}%
          </div>
          <div className="text-sm text-muted-foreground">Avg Success Rate</div>
        </div>
        <div className="text-center">
          <div className="text-base font-semibold" style={{ color: "var(--status-error)" }}>
            {metrics.totalErrors}
          </div>
          <div className="text-sm text-muted-foreground">Total Errors</div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative">
        <svg
          width={chartWidth}
          height={chartHeight}
          className="w-full h-auto"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        >
          {/* Grid Lines */}
          <defs>
            <pattern
              id="grid"
              width="40"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 40 0 L 0 0 0 20"
                fill="none"
                stroke="var(--border)"
                strokeWidth="0.5"
                opacity="0.3"
              />
            </pattern>

            {/* Gradient for area fill */}
            <linearGradient
              id="executionGradient"
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop
                offset="0%"
                stopColor="var(--primary)"
                stopOpacity="0.3"
              />
              <stop
                offset="100%"
                stopColor="var(--primary)"
                stopOpacity="0.05"
              />
            </linearGradient>
          </defs>

          {/* Background grid */}
          <rect
            x={padding.left}
            y={padding.top}
            width={innerWidth}
            height={innerHeight}
            fill="url(#grid)"
          />

          {/* Chart area background */}
          <rect
            x={padding.left}
            y={padding.top}
            width={innerWidth}
            height={innerHeight}
            fill="var(--muted)"
            fillOpacity="0.1"
            stroke="var(--border)"
            strokeWidth="1"
          />

          {/* Execution volume area */}
          <path
            d={chartData.executionArea}
            fill="url(#executionGradient)"
            transform={`translate(${padding.left}, ${padding.top})`}
          />

          {/* Execution volume line */}
          <path
            d={chartData.executionPath}
            fill="none"
            stroke="var(--primary)"
            strokeWidth="2"
            transform={`translate(${padding.left}, ${padding.top})`}
          />

          {/* Success rate line */}
          <path
            d={chartData.successRatePath}
            fill="none"
            stroke="var(--status-success)"
            strokeWidth="2"
            strokeDasharray="4,4"
            transform={`translate(${padding.left}, ${padding.top})`}
          />

          {/* Data points */}
          {chartData.dataPoints.map((point, i) => (
            <circle
              key={i}
              cx={point.x}
              cy={point.y}
              r="2"
              fill="var(--primary)"
              className="hover:r-3 transition-all duration-200 cursor-pointer"
            >
              <title>
                {point.data.hour}: {point.data.executions} executions, {point.data.success_rate}% success rate
              </title>
            </circle>
          ))}

          {/* Y-axis labels (left) */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = innerHeight - ratio * innerHeight + padding.top;
            const value = Math.round(chartData.maxExecutions * ratio);

            return (
              <g key={ratio}>
                <line
                  x1={padding.left - 5}
                  y1={y}
                  x2={padding.left}
                  y2={y}
                  stroke="var(--muted-foreground)"
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 10}
                  y={y + 3}
                  textAnchor="end"
                  className="text-xs fill-muted-foreground"
                  style={{ fontSize: "10px" }}
                >
                  {value}
                </text>
              </g>
            );
          })}

          {/* Y-axis labels (right) - Success rate */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = innerHeight - ratio * innerHeight + padding.top;
            const value = Math.round(
              chartData.minSuccessRate + (chartData.maxSuccessRate - chartData.minSuccessRate) * ratio
            );

            return (
              <g key={`success-${ratio}`}>
                <line
                  x1={padding.left + innerWidth}
                  y1={y}
                  x2={padding.left + innerWidth + 5}
                  y2={y}
                  stroke="var(--status-success)"
                  strokeWidth="1"
                  opacity="0.7"
                />
                <text
                  x={padding.left + innerWidth + 10}
                  y={y + 3}
                  textAnchor="start"
                  className="text-xs"
                  style={{ fontSize: "10px", fill: "var(--status-success)" }}
                >
                  {value}%
                </text>
              </g>
            );
          })}

          {/* X-axis labels */}
          {data
            .filter((_, i) => i % 4 === 0)
            .map((d, i) => {
              const originalIndex = i * 4;
              const x =
                (originalIndex / (data.length - 1)) * innerWidth + padding.left;

              return (
                <g key={originalIndex}>
                  <line
                    x1={x}
                    y1={padding.top + innerHeight}
                    x2={x}
                    y2={padding.top + innerHeight + 5}
                    stroke="var(--muted-foreground)"
                    strokeWidth="1"
                  />
                  <text
                    x={x}
                    y={padding.top + innerHeight + 18}
                    textAnchor="middle"
                    className="text-xs fill-muted-foreground"
                    style={{ fontSize: "10px" }}
                  >
                    {d.hour}
                  </text>
                </g>
              );
            })}

          {/* Axis labels */}
          <text
            x={padding.left - 35}
            y={padding.top + innerHeight / 2}
            textAnchor="middle"
            className="text-xs fill-muted-foreground"
            style={{ fontSize: "10px" }}
            transform={`rotate(-90, ${padding.left - 35}, ${
              padding.top + innerHeight / 2
            })`}
          >
            Executions
          </text>

          <text
            x={padding.left + innerWidth + 35}
            y={padding.top + innerHeight / 2}
            textAnchor="middle"
            className="text-xs"
            style={{ fontSize: "10px", fill: "var(--status-success)" }}
            transform={`rotate(90, ${padding.left + innerWidth + 35}, ${
              padding.top + innerHeight / 2
            })`}
          >
            Success Rate (%)
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4 pt-4 border-t border-border">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-0.5 bg-primary"></div>
          <span className="text-sm text-muted-foreground">
            Execution Volume
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <div
            className="w-3 h-0.5"
            style={{
              backgroundColor: "var(--status-success)",
              borderTop: "2px dashed var(--status-success)",
              height: "0px"
            }}
          ></div>
          <span className="text-sm text-muted-foreground">Success Rate</span>
        </div>
      </div>
    </div>
  );
});

import { useMemo } from "react";
import {
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Area,
  ComposedChart,
} from "recharts";
import type { WorkflowTimelineNode } from "../../types/workflows";

interface VelocityChartProps {
  timedNodes: WorkflowTimelineNode[];
  className?: string;
}

interface ChartDataPoint {
  timestamp: number;
  duration: number;
  formattedTime: string;
  formattedDuration: string;
  label: string;
  executionId: string;
}

const formatDuration = (value: number): string => {
  if (value <= 0) return "0ms";
  if (value < 1000) return `${value.toFixed(0)}ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)}s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.floor((value % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
};

const formatTime = (timestamp: number): string => {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

export function VelocityChart({ timedNodes, className }: VelocityChartProps) {
  const chartData = useMemo<ChartDataPoint[]>(() => {
    if (!timedNodes.length) return [];

    return timedNodes
      .map((node) => ({
        timestamp: new Date(node.started_at).getTime(),
        duration: Number(node.duration_ms) || 0,
        formattedTime: formatTime(new Date(node.started_at).getTime()),
        formattedDuration: formatDuration(Number(node.duration_ms) || 0),
        label: node.agent_name || node.reasoner_id || "Unknown",
        executionId: node.execution_id,
      }))
      .filter((item) => Number.isFinite(item.timestamp) && item.duration > 0)
      .sort((a, b) => a.timestamp - b.timestamp);
  }, [timedNodes]);

  const { minDuration, maxDuration } = useMemo(() => {
    if (!chartData.length) return { minDuration: 0, maxDuration: 0 };

    const durations = chartData.map((d) => d.duration);
    return {
      minDuration: Math.min(...durations),
      maxDuration: Math.max(...durations),
    };
  }, [chartData]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as ChartDataPoint;
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3 text-sm">
          <div className="font-medium text-foreground">{data.label}</div>
          <div className="text-muted-foreground">
            Time: {data.formattedTime}
          </div>
          <div className="text-primary font-semibold">
            Duration: {data.formattedDuration}
          </div>
          <div className="text-sm text-muted-foreground mt-1">
            ID: {data.executionId.slice(0, 8)}...
          </div>
        </div>
      );
    }
    return null;
  };

  if (!chartData.length) {
    return (
      <div
        className={`h-32 rounded-lg border border-border/60 bg-gradient-to-b from-muted/40 to-muted/5 flex items-center justify-center ${className}`}
      >
        <div className="text-sm text-muted-foreground">
          Not enough temporal data to plot
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-32 rounded-lg border border-border/60 bg-gradient-to-b from-muted/40 to-muted/5 p-3 ${className}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={chartData}
          margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
        >
          <defs>
            <linearGradient id="velocityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="rgb(99 102 241)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="rgb(99 102 241)" stopOpacity={0} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="timestamp"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={() => ""}
            axisLine={false}
            tickLine={false}
            hide
          />

          <YAxis
            domain={[0, "dataMax"]}
            tickFormatter={() => ""}
            axisLine={false}
            tickLine={false}
            hide
          />

          <Tooltip content={<CustomTooltip />} />

          <Area
            type="monotone"
            dataKey="duration"
            stroke="none"
            fill="url(#velocityGradient)"
          />

          <Line
            type="monotone"
            dataKey="duration"
            stroke="rgb(99 102 241)"
            strokeWidth={2}
            dot={{ fill: "rgb(99 102 241)", strokeWidth: 0, r: 3 }}
            activeDot={{
              r: 5,
              fill: "rgb(20 184 166)",
              strokeWidth: 2,
              stroke: "white",
            }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Custom labels positioned outside the chart */}
      <div className="flex justify-between items-center mt-2 px-2">
        <div className="text-sm text-muted-foreground">
          Min: {formatDuration(minDuration)}
        </div>
        <div className="text-sm text-muted-foreground">
          Max: {formatDuration(maxDuration)}
        </div>
      </div>
    </div>
  );
}

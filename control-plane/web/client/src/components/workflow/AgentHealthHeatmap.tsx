import { useMemo } from "react";
import type { WorkflowTimelineNode } from "../../types/workflows";
import { format, startOfMinute, addMinutes, differenceInMinutes } from "date-fns";
import {
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Cell,
} from "recharts";

interface AgentHealthHeatmapProps {
  timedNodes: WorkflowTimelineNode[];
  className?: string;
}

interface HeatmapCell {
  x: number; // Time bucket index
  y: number; // Agent index
  z: number; // Intensity (failure rate)
  agent: string;
  timeLabel: string;
  total: number;
  failed: number;
  succeeded: number;
  failureRate: number;
}

export function AgentHealthHeatmap({ timedNodes, className }: AgentHealthHeatmapProps) {
  const { cells, uniqueAgents, timeBuckets } = useMemo(() => {
    if (!timedNodes.length) return { cells: [], uniqueAgents: [], timeBuckets: [] };

    // 1. Identify Agents
    const agents = Array.from(
      new Set(
        timedNodes.map((n) => n.agent_name || n.reasoner_id || "Unknown")
      )
    ).sort();

    // 2. Determine Time Range & Buckets (e.g., 15 minute intervals)
    const timestamps = timedNodes.map((n) => new Date(n.started_at).getTime());
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);

    // Auto-scale bucket size based on duration
    const durationMinutes = differenceInMinutes(maxTime, minTime);
    let bucketSizeMinutes = 5;
    if (durationMinutes > 60) bucketSizeMinutes = 15;
    if (durationMinutes > 24 * 60) bucketSizeMinutes = 60;

    const matchedBuckets: number[] = [];
    let currentTime = startOfMinute(minTime).getTime();
    while (currentTime <= maxTime + bucketSizeMinutes * 60000) {
      matchedBuckets.push(currentTime);
      currentTime = addMinutes(currentTime, bucketSizeMinutes).getTime();
    }

    // 3. Aggregate Data
    const cellMap = new Map<string, HeatmapCell>();

    timedNodes.forEach((node) => {
      const nodeTime = new Date(node.started_at).getTime();
      const agent = node.agent_name || node.reasoner_id || "Unknown";
      const agentIndex = agents.indexOf(agent);

      // Find time bucket
      // Simple linear scan is fine for < 100 buckets, otherwise binary search
      let bucketIndex = matchedBuckets.findIndex(t => t > nodeTime) - 1;
      if (bucketIndex < 0) bucketIndex = 0;
      // Clamp to last bucket if needed
      if (bucketIndex >= matchedBuckets.length) bucketIndex = matchedBuckets.length - 1;

      const key = `${bucketIndex}-${agentIndex}`;

      if (!cellMap.has(key)) {
        cellMap.set(key, {
          x: bucketIndex,
          y: agentIndex,
          z: 0,
          agent,
          timeLabel: format(matchedBuckets[bucketIndex], "HH:mm"),
          total: 0,
          failed: 0,
          succeeded: 0,
          failureRate: 0,
        });
      }

      const cell = cellMap.get(key)!;
      cell.total++;
      if (node.status === "failed" || node.status === "error") {
        cell.failed++;
      } else {
        cell.succeeded++;
      }
      cell.failureRate = cell.failed / cell.total;
      // Z maps to color intensity (0 = good, 1 = bad)
      cell.z = cell.failureRate;
    });

    return {
      cells: Array.from(cellMap.values()),
      uniqueAgents: agents,
      timeBuckets: matchedBuckets,
    };
  }, [timedNodes]);

  if (!cells.length) {
     return (
        <div className={`h-48 rounded-xl border border-border/40 bg-muted/5 flex items-center justify-center ${className}`}>
            <span className="text-muted-foreground text-sm">No data for heatmap</span>
        </div>
    );
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as HeatmapCell;
      return (
        <div className="bg-popover border border-border rounded-lg shadow-lg p-3 text-sm z-50">
          <div className="font-semibold text-foreground mb-1">{data.agent}</div>
          <div className="text-xs text-muted-foreground mb-2">Window: {data.timeLabel}</div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between gap-4">
              <span>Total Runs:</span>
              <span className="font-mono">{data.total}</span>
            </div>
            <div className="flex justify-between gap-4 text-emerald-500">
              <span>Succeeded:</span>
              <span className="font-mono">{data.succeeded}</span>
            </div>
             <div className="flex justify-between gap-4 text-destructive">
              <span>Failed:</span>
              <span className="font-mono">{data.failed}</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="h-64 w-full rounded-xl border border-border/40 bg-card p-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
          >
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, timeBuckets.length]}
              tickFormatter={(index) => {
                 const time = timeBuckets[index];
                 return time ? format(time, "HH:mm") : "";
              }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-0.5, uniqueAgents.length - 0.5]}
              tickFormatter={(index) => uniqueAgents[index] || ""}
              tickLine={false}
              axisLine={false}
              width={80}
              tick={{ fontSize: 10, fill: "var(--foreground)" }}
              interval={0}
            />
            <ZAxis type="number" dataKey="z" range={[1, 1]} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'var(--muted)', opacity: 0.2 }} />
            <Scatter data={cells} shape="square">
              {cells.map((entry, index) => {
                // Color scale: Green -> Yellow -> Red
                // Or simpler: Green (0 fails) -> Red (some fails) to match requested "error creeping up"

                let fill = "var(--muted)";
                let opacity = 0.3; // Default for 0 runs (shouldn't happen with this logic but good fallback)

                if (entry.total > 0) {
                    if (entry.failed === 0) {
                         // All good
                         fill = "var(--emerald-500)";
                         opacity = 0.2 + (Math.min(entry.total, 10) / 10) * 0.6; // More runs = darker green
                    } else if (entry.failureRate < 0.2) {
                        // Some sporadic errors
                        fill = "var(--amber-500)";
                        opacity = 0.8;
                    } else {
                        // High failure rate
                        fill = "var(--destructive)";
                        opacity = 0.4 + entry.failureRate * 0.6;
                    }
                }

                // Render as a rect that fills the grid cell
                // Note: Scatter specific shape rendering is tricky to make fill perfectly,
                // so we use a large square symbol (size handled by chart/css or custom shape)
                // For simplicity here, we stick to standard Scatter dots but make them large squares?
                // Actually Recharts Scatter doesn't support custom props per cell easily for size logic unless Z axis.
                // We will rely on `Cell` and just use circles for now as "heatmap dots" which is a common upscale look.

                return <Cell key={`cell-${index}`} fill={fill} fillOpacity={opacity} />;
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-end gap-4 text-[10px] text-muted-foreground px-2">
         <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 opacity-60"></span> Healthy</span>
         <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500 opacity-80"></span> Sporadic Errors</span>
         <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-destructive opacity-80"></span> High Failure</span>
      </div>
    </div>
  );
}

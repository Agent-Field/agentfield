import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
  Cell,
} from "recharts";
import type { WorkflowTimelineNode } from "../../types/workflows";
import { format } from "date-fns";

import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ExecutionScatterPlotProps {
  timedNodes: WorkflowTimelineNode[];
  className?: string;
  onNodeClick?: (executionId: string) => void;
}

interface PlotPoint {
  x: number; // Timestamp
  y: number; // Duration
  z: number; // Size (optional, currently uniform)
  id: string;
  status: string;
  agent: string;
  label: string;
  startedAt: string;
  formattedDuration: string;
}

const formatDuration = (value: number): string => {
  if (value < 1000) return `${value.toFixed(0)}ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)}s`;
  return `${(value / 60000).toFixed(1)}m`;
};

export function ExecutionScatterPlot({
  timedNodes,
  className,
  onNodeClick,
}: ExecutionScatterPlotProps) {
  const [zoomDomain, setZoomDomain] = useState<{ min: number; max: number } | null>(null);

  const data = useMemo<PlotPoint[]>(() => {
    return timedNodes
      .map((node) => ({
        x: new Date(node.started_at).getTime(),
        y: Number(node.duration_ms) || 0,
        z: 1,
        id: node.execution_id,
        status: node.status,
        agent: node.agent_name || node.reasoner_id || "Unknown",
        label: node.agent_name || node.reasoner_id || "Unknown",
        startedAt: node.started_at,
        formattedDuration: formatDuration(Number(node.duration_ms) || 0),
      }))
      .sort((a, b) => a.x - b.x);
  }, [timedNodes]);

  const { minTime, maxTime, avgDuration, maxDuration } = useMemo(() => {
    if (!data.length)
      return {
        minTime: 0,
        maxTime: 0,
        avgDuration: 0,
        maxDuration: 0,
      };

    const durations = data.map((d) => d.y);
    const times = data.map((d) => d.x);
    const total = durations.reduce((a, b) => a + b, 0);

    return {
      minTime: Math.min(...times),
      maxTime: Math.max(...times),
      avgDuration: total / durations.length,
      maxDuration: Math.max(...durations),
    };
  }, [data]);

  // Zoom Logic
  const handleZoomIn = () => {
    if (!data.length) return;
    const range = (zoomDomain?.max || maxTime) - (zoomDomain?.min || minTime);
    const center = (zoomDomain?.min || minTime) + range / 2;
    const newRange = range * 0.5;
    setZoomDomain({
      min: center - newRange / 2,
      max: center + newRange / 2,
    });
  };

  const handleZoomOut = () => {
    if (!data.length) return;
    const currentMin = zoomDomain?.min || minTime;
    const currentMax = zoomDomain?.max || maxTime;
    const range = currentMax - currentMin;
    const center = currentMin + range / 2;
    const newRange = range * 2;

    // Clamp to original bounds
    setZoomDomain({
        min: Math.max(minTime, center - newRange / 2),
        max: Math.min(maxTime, center + newRange / 2),
    });
  };

  const handleResetZoom = () => {
    setZoomDomain(null);
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload as PlotPoint;
      const isSuccess = point.status === "succeeded" || point.status === "completed";

      return (
        <div className="bg-popover border border-border rounded-lg shadow-xl p-3 text-sm z-50">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isSuccess ? "bg-emerald-500" : "bg-destructive"
              }`}
            />
            <span className="font-semibold text-foreground">{point.agent}</span>
          </div>
          <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs">
            <span className="text-muted-foreground w-16">Time:</span>
            <span className="font-mono">{format(new Date(point.x), "HH:mm:ss.S")}</span>

            <span className="text-muted-foreground w-16">Duration:</span>
            <span className="font-mono">{point.formattedDuration}</span>

            <span className="text-muted-foreground w-16">Status:</span>
            <span className={`${isSuccess ? 'text-emerald-500' : 'text-destructive'} capitalize font-medium`}>
              {point.status}
            </span>

            <span className="text-muted-foreground w-16">ID:</span>
            <span className="font-mono text-muted-foreground/70">{point.id.slice(0, 8)}...</span>
          </div>
        </div>
      );
    }
    return null;
  };

  if (!data.length) {
    return (
        <div className={`h-64 rounded-xl border border-border/40 bg-muted/5 flex items-center justify-center ${className}`}>
            <span className="text-muted-foreground text-sm">No execution data to display</span>
        </div>
    );
  }

  const domain = zoomDomain || { min: minTime, max: maxTime };
  // Add 5% padding to Y axis
  const yDomainMax = maxDuration * 1.05;

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex justify-end gap-1 mb-2">
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomOut} disabled={!zoomDomain}>
            <ZoomOut className="h-3 w-3" />
        </Button>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleResetZoom} disabled={!zoomDomain}>
            <RotateCcw className="h-3 w-3" />
        </Button>
         <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomIn}>
            <ZoomIn className="h-3 w-3" />
        </Button>
      </div>

      <div className="h-72 w-full rounded-xl border border-border/40 bg-gradient-to-b from-background to-muted/10 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.2} horizontal={true} vertical={false} />
            <XAxis
              type="number"
              dataKey="x"
              name="Time"
              domain={[domain.min, domain.max]}
              tickFormatter={(timestamp) => format(new Date(timestamp), "HH:mm")}
              stroke="var(--muted-foreground)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Duration"
              unit="ms"
              tickFormatter={(val) => formatDuration(val)}
              stroke="var(--muted-foreground)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              width={40}
              domain={[0, yDomainMax]}
            />
            <ZAxis type="number" dataKey="z" range={[50, 50]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'var(--muted-foreground)' }} />

            <ReferenceLine y={avgDuration} stroke="var(--primary)" strokeDasharray="3 3" opacity={0.5} label={{ value: "AVG", position: 'insideLeft', fill: 'var(--primary)', fontSize: 10 }} />

            <Scatter
              name="Executions"
              data={data}
              onClick={(p) => onNodeClick?.((p.payload as PlotPoint).id)}
              cursor="pointer"
            >
              {data.map((entry, index) => {
                 const isSuccess = entry.status === "succeeded" || entry.status === "completed";
                 return (
                    <Cell
                        key={`cell-${index}`}
                        fill={isSuccess ? "var(--emerald-500)" : "var(--destructive)"}
                        fillOpacity={0.6}
                        stroke={isSuccess ? "var(--emerald-600)" : "var(--destructive)"}
                        strokeWidth={1}
                    />
                 )
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between text-[10px] text-muted-foreground px-2">
        <span>Scatter plot shows distribution of {data.length} executions.</span>
        <div className="flex gap-3">
             <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 opacity-60"></span> Success</span>
             <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-destructive opacity-60"></span> Failed</span>
        </div>
      </div>
    </div>
  );
}

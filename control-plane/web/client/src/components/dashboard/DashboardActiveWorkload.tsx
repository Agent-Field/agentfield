import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Layers } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { WorkflowSummary } from "@/types/workflows";

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

export interface DashboardActiveWorkloadProps {
  activeRuns: WorkflowSummary[];
  className?: string;
}

interface Row {
  name: string;
  count: number;
  fill: string;
}

function aggregateByReasoner(runs: WorkflowSummary[]): Row[] {
  const map = new Map<string, number>();
  for (const r of runs) {
    const key = r.root_reasoner || r.display_name || "—";
    map.set(key, (map.get(key) ?? 0) + 1);
  }
  return [...map.entries()]
    .map(([name, count], i) => ({
      name,
      count,
      fill: CHART_COLORS[i % CHART_COLORS.length]!,
    }))
    .sort((a, b) => b.count - a.count);
}

export function DashboardActiveWorkload({
  activeRuns,
  className,
}: DashboardActiveWorkloadProps) {
  const data = useMemo(() => aggregateByReasoner(activeRuns), [activeRuns]);

  if (data.length === 0) return null;

  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base font-semibold">
          <Layers className="size-4 text-muted-foreground" aria-hidden />
          Active by reasoner
        </CardTitle>
        <CardDescription>Concurrent runs grouped by root reasoner.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col pt-0">
        <div className="min-h-[140px] w-full flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={data}
              margin={{ top: 4, right: 12, left: 4, bottom: 4 }}
            >
              <CartesianGrid horizontal={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="name"
                width={100}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                interval={0}
                tickFormatter={(v: string) => (v.length > 14 ? `${v.slice(0, 12)}…` : v)}
              />
              <Tooltip
                cursor={{ fill: "hsl(var(--muted) / 0.3)" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const row = payload[0]?.payload as Row;
                  return (
                    <div className="rounded-md border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md">
                      <p className="font-medium">{row.name}</p>
                      <p className="text-muted-foreground">
                        Active runs: <span className="tabular-nums">{row.count}</span>
                      </p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={28}>
                {data.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

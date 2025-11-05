import { useMemo } from "react";
import { summarizeNodeStatuses, getNodeStatusPresentation } from "@/utils/node-status";
import { Checklist } from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import type { AgentNodeSummary } from "../types/agentfield";

interface NodesStatusSummaryProps {
  nodes: AgentNodeSummary[];
  searchQuery: string;
}

export function NodesStatusSummary({
  nodes,
  searchQuery,
}: NodesStatusSummaryProps) {
  const summary = useMemo(
    () => summarizeNodeStatuses(nodes),
    [nodes]
  );

  const detailItems = [
    {
      key: "online",
      label: "online",
      count: summary.online,
      presentation: getNodeStatusPresentation("ready", "active"),
      alwaysShow: true,
    },
    {
      key: "starting",
      label: "starting",
      count: summary.starting,
      presentation: getNodeStatusPresentation("starting"),
    },
    {
      key: "degraded",
      label: "degraded",
      count: summary.degraded,
      presentation: getNodeStatusPresentation("degraded"),
    },
    {
      key: "offline",
      label: "offline",
      count: summary.offline,
      presentation: getNodeStatusPresentation("offline", "inactive"),
      alwaysShow: true,
    },
  ].filter((item) => item.alwaysShow || item.count > 0);

  if (summary.total === 0) {
    return null;
  }

  return (
    <Card variant="surface" interactive={false} className="px-4 py-3 shadow-sm backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          <Checklist className="h-3.5 w-3.5" />
          <span>Status Overview</span>
          <span className="text-muted-foreground/60">•</span>
          <span>{summary.total} node{summary.total === 1 ? "" : "s"}</span>
        </div>
        {searchQuery && (
          <span className="text-body-small">
            Showing {summary.total} result{summary.total === 1 ? "" : "s"} for&nbsp;
            <span className="font-medium text-foreground">"{searchQuery}"</span>
          </span>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
        {detailItems.map((item) => (
          <div key={item.key} className="flex items-center gap-2">
            <span
              className={cn(
                "h-2.5 w-2.5 rounded-full",
                item.presentation.theme.indicatorClass,
                item.presentation.shouldPulse && "animate-pulse"
              )}
            />
            <span className={cn("font-medium", item.presentation.theme.textClass)}>
              {item.count}
            </span>
            <span className="text-body-small">{item.label}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

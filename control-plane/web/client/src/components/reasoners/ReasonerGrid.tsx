import type {
  ReasonerGridProps,
  ReasonerWithNode,
} from "../../types/reasoners";
import { ResponsiveGrid } from "../layout/ResponsiveGrid";
import { Skeleton } from "../ui/skeleton";
import { DIDIdentityBadge } from "../did/DIDDisplay";
import { EmptyReasonersState } from "./EmptyReasonersState";
import { ReasonerCard } from "./ReasonerCard";
import { ReasonerStatusDot } from "./ReasonerStatusDot";

export function ReasonerGrid({
  reasoners,
  loading = false,
  onReasonerClick,
  viewMode = "grid",
}: ReasonerGridProps) {
  // Add null safety check for reasoners array
  const safeReasoners = reasoners || [];

  if (loading) {
    return <ReasonerGridSkeleton viewMode={viewMode} />;
  }

  if (safeReasoners.length === 0) {
    return <EmptyReasonersState type="no-reasoners" onRefresh={() => {}} />;
  }

  if (viewMode === "table") {
    return (
      <ReasonerTable
        reasoners={safeReasoners}
        onReasonerClick={onReasonerClick}
      />
    );
  }

  return (
    <ResponsiveGrid variant="dashboard" align="start">
      {safeReasoners.map((reasoner) => (
        <ReasonerCard
          key={reasoner.reasoner_id}
          reasoner={reasoner}
          onClick={onReasonerClick}
        />
      ))}
    </ResponsiveGrid>
  );
}

function ReasonerGridSkeleton({ viewMode }: { viewMode?: "grid" | "table" }) {
  if (viewMode === "table") {
    return <ReasonerTableSkeleton />;
  }
  return (
    <ResponsiveGrid variant="dashboard" align="start">
      {Array.from({ length: 8 }).map((_, index) => (
        <div
          key={index}
          className="rounded-xl border border-border bg-card p-4 space-y-4"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3 flex-1">
              <Skeleton className="h-8 w-8 rounded-lg" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            </div>
            <Skeleton className="h-6 w-16 rounded-full" />
          </div>

          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-2/3" />
          </div>

          <div className="flex gap-4">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-16" />
          </div>

          <div className="flex gap-4">
            <Skeleton className="h-3 w-12" />
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-14" />
          </div>

          <div className="flex justify-end">
            <Skeleton className="h-7 w-24 rounded-lg" />
          </div>
        </div>
      ))}
    </ResponsiveGrid>
  );
}

function ReasonerTable({
  reasoners,
  onReasonerClick,
}: {
  reasoners: ReasonerWithNode[];
  onReasonerClick?: (reasoner: ReasonerWithNode) => void;
}) {
  // Add null safety for reasoners array
  const safeReasoners = reasoners || [];

  const getStatusFromNodeStatus = (nodeStatus: string) => {
    switch (nodeStatus) {
      case "active":
        return "online";
      case "inactive":
        return "offline";
      default:
        return "unknown";
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;

    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className="bg-card border border-border rounded-lg shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-muted border-b border-border">
            <tr>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Name
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Node
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Status
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Performance
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Updated
              </th>
              <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-secondary">
            {safeReasoners.map((reasoner) => {
              const status = getStatusFromNodeStatus(reasoner.node_status);
              const isOffline = status === "offline";

              return (
                <tr
                  key={reasoner.reasoner_id}
                  className={`
                    hover:bg-muted transition-colors cursor-pointer
                    ${isOffline ? "opacity-75" : ""}
                  `}
                  onClick={() => onReasonerClick?.(reasoner)}
                >
                  <td className="py-2 px-3">
                    <div>
                      <div>
                        <span
                          className="font-medium text-foreground text-sm line-clamp-2 break-words max-w-xs"
                          title={reasoner.name}
                        >
                          {reasoner.name}
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground truncate max-w-xs mt-0.5">
                        {reasoner.description}
                      </div>
                    </div>
                  </td>
                  <td className="py-2 px-3">
                    <div className="space-y-0.5">
                      <div className="text-xs text-muted-foreground">
                        {reasoner.node_id}
                      </div>
                      <DIDIdentityBadge
                        nodeId={reasoner.node_id}
                        showDID={true}
                        className="text-xs"
                      />
                    </div>
                  </td>
                  <td className="py-2 px-3">
                    <ReasonerStatusDot
                      status={status}
                      showText={false}
                      size="sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <div className="text-xs space-y-0.5">
                      {reasoner.avg_response_time_ms && (
                        <div className="text-muted-foreground">
                          {reasoner.avg_response_time_ms}ms
                        </div>
                      )}
                      {reasoner.success_rate && (
                        <div className="text-status-success">
                          {(reasoner.success_rate * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="py-2 px-3">
                    <div className="text-xs text-muted-foreground">
                      {formatTimeAgo(reasoner.last_updated)}
                    </div>
                  </td>
                  <td className="py-2 px-3 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onReasonerClick?.(reasoner);
                      }}
                      className="text-xs text-accent-primary hover:text-accent-primary-hover transition-colors"
                    >
                      Open →
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ReasonerTableSkeleton() {
  return (
    <div className="bg-card border border-border rounded-lg shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-muted border-b border-border">
            <tr>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Name
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Node
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Status
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Performance
              </th>
              <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Updated
              </th>
              <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-secondary">
            {Array.from({ length: 5 }).map((_, index) => (
              <tr key={index}>
                <td className="py-2 px-3">
                  <div className="space-y-1">
                    <Skeleton className="h-3 w-32" />
                    <Skeleton className="h-2 w-48" />
                  </div>
                </td>
                <td className="py-2 px-3">
                  <div className="space-y-0.5">
                    <Skeleton className="h-2 w-20" />
                    <Skeleton className="h-2 w-16" />
                  </div>
                </td>
                <td className="py-2 px-3">
                  <Skeleton className="h-2 w-2 rounded-full" />
                </td>
                <td className="py-2 px-3">
                  <div className="space-y-0.5">
                    <Skeleton className="h-2 w-12" />
                    <Skeleton className="h-2 w-16" />
                  </div>
                </td>
                <td className="py-2 px-3">
                  <Skeleton className="h-2 w-16" />
                </td>
                <td className="py-2 px-3 text-right">
                  <Skeleton className="h-2 w-10 ml-auto" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
